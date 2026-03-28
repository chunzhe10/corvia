//! TantivyIndex: embedded full-text BM25 search for LiteStore.
//!
//! Stores a tantivy index in `.corvia/cache/tantivy/` as a cache (rebuildable
//! from knowledge JSON files). Uses a code-aware custom tokenizer that splits
//! snake_case, CamelCase, file paths, UUIDs, and version strings.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{KnowledgeEntry, SearchResult};
use redb::{Database, ReadableTable, TableDefinition};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use tokio::sync::Mutex;
use tracing::info;

use crate::traits::FullTextSearchable;

/// Redb META table key for the FTS write generation counter.
const FTS_WRITE_GEN_KEY: &str = "fts_write_generation";

/// Redb META table (shared with LiteStore).
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

/// Generation metadata file name (stored alongside the tantivy index).
const GEN_META_FILE: &str = "corvia_fts_gen.json";

/// Flush threshold: number of buffered entries before auto-flush.
const FLUSH_ENTRY_THRESHOLD: u64 = 100;

/// Progress logging interval during rebuild.
const REBUILD_LOG_INTERVAL: usize = 10_000;

// ---------------------------------------------------------------------------
// Code-aware tokenizer
// ---------------------------------------------------------------------------

/// A tantivy tokenizer that splits code identifiers while preserving originals.
///
/// Handles: snake_case, CamelCase, UPPER_SNAKE, file paths, version strings,
/// UUIDs, and dotted identifiers. All tokens are lowercased.
#[derive(Clone)]
pub struct CorviaCodeTokenizer;

impl Tokenizer for CorviaCodeTokenizer {
    type TokenStream<'a> = CorviaCodeTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let tokens = tokenize_code_aware(text);
        CorviaCodeTokenStream {
            tokens,
            index: 0,
            token: Token::default(),
        }
    }
}

pub struct CorviaCodeTokenStream {
    tokens: Vec<String>,
    index: usize,
    token: Token,
}

impl TokenStream for CorviaCodeTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.token.text.clear();
            self.token.text.push_str(&self.tokens[self.index]);
            self.token.position = self.index;
            self.token.offset_from = 0;
            self.token.offset_to = self.token.text.len();
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

/// Core tokenization logic: split a text string into code-aware tokens.
///
/// Rules:
/// 1. Split on whitespace first to get raw words.
/// 2. For each raw word, emit the original (lowercased).
/// 3. Split on `_`, `-`, `/`, `.` to get sub-tokens.
/// 4. Split CamelCase boundaries.
/// 5. Handle version strings (v0.4.5 -> v0.4.5, 0.4.5, v0, 4, 5).
/// 6. Deduplicate tokens per word.
pub fn tokenize_code_aware(text: &str) -> Vec<String> {
    let mut all_tokens = Vec::new();

    for word in text.split_whitespace() {
        let mut word_tokens = Vec::new();
        let lower = word.to_lowercase();

        // Always emit the full original (lowercased).
        word_tokens.push(lower.clone());

        // Check for version string pattern: v0.4.5 or 0.4.5
        if is_version_string(word) {
            tokenize_version(&lower, &mut word_tokens);
        } else {
            // Split on separators: _, -, /, .
            split_on_separators(&lower, &mut word_tokens);
        }

        // CamelCase splitting on the original (before lowercasing separators).
        split_camel_case(word, &mut word_tokens);

        // Dedup within this word's tokens.
        let mut seen = std::collections::HashSet::new();
        for tok in word_tokens {
            if !tok.is_empty() && seen.insert(tok.clone()) {
                all_tokens.push(tok);
            }
        }
    }

    all_tokens
}

fn is_version_string(word: &str) -> bool {
    let s = word.strip_prefix('v').unwrap_or(word);
    // Must start with a digit, contain at least two dots (X.Y.Z pattern),
    // and only contain digits and dots.
    s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().filter(|&c| c == '.').count() >= 2
        && s.chars().all(|c| c.is_ascii_digit() || c == '.')
}

fn tokenize_version(lower: &str, tokens: &mut Vec<String>) {
    // v0.4.5 -> also emit 0.4.5 (without v)
    if let Some(without_v) = lower.strip_prefix('v') {
        tokens.push(without_v.to_string());
        // Split on dots: v0, 4, 5
        if let Some((first, _)) = lower.split_once('.') {
            tokens.push(first.to_string()); // "v0"
        }
        for part in without_v.split('.') {
            if !part.is_empty() {
                tokens.push(part.to_string());
            }
        }
    } else {
        // Plain version like 0.4.5
        for part in lower.split('.') {
            if !part.is_empty() {
                tokens.push(part.to_string());
            }
        }
    }
}

fn split_on_separators(lower: &str, tokens: &mut Vec<String>) {
    let separators = ['_', '-', '/', '.'];
    if lower.contains(separators) {
        for part in lower.split(|c: char| separators.contains(&c)) {
            if !part.is_empty() {
                tokens.push(part.to_string());
            }
        }
    }
}

fn split_camel_case(original: &str, tokens: &mut Vec<String>) {
    let chars: Vec<char> = original.chars().collect();
    if chars.len() < 2 {
        return;
    }

    let mut parts = Vec::new();
    let mut start = 0;

    for i in 1..chars.len() {
        // Split on: lowercase -> uppercase boundary.
        let split = chars[i].is_uppercase() && chars[i - 1].is_lowercase();
        // Also split on: uppercase -> uppercase -> lowercase (e.g., "XMLParser" -> "XML", "Parser").
        let split2 = i + 1 < chars.len()
            && chars[i].is_uppercase()
            && chars[i - 1].is_uppercase()
            && chars[i + 1].is_lowercase();

        if split || split2 {
            let part: String = chars[start..i].iter().collect();
            if !part.is_empty() {
                parts.push(part.to_lowercase());
            }
            start = i;
        }
    }

    // Last part.
    let part: String = chars[start..].iter().collect();
    if !part.is_empty() {
        parts.push(part.to_lowercase());
    }

    // Only emit sub-parts if we actually split something.
    if parts.len() > 1 {
        for p in parts {
            tokens.push(p);
        }
    }
}

// ---------------------------------------------------------------------------
// TantivyIndex
// ---------------------------------------------------------------------------

/// Embedded tantivy index for BM25 full-text search.
pub struct TantivyIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    /// Store reference for looking up full entries by ID in search_text.
    store: Arc<dyn crate::traits::QueryableStore>,
    // Schema field handles.
    field_content: Field,
    field_entry_id: Field,
    field_scope_id: Field,
    field_content_role: Field,
    field_source_origin: Field,
    // Batched commit state.
    pending_count: AtomicU64,
    // Redb database (shared with LiteStore) for write generation.
    db: Arc<Database>,
    // Path to generation metadata file.
    gen_meta_path: PathBuf,
}

impl TantivyIndex {
    /// Open or create a TantivyIndex at the given cache directory.
    ///
    /// `db` is the shared Redb database (for write generation tracking).
    /// `cache_dir` is typically `.corvia/cache/tantivy/`.
    pub fn open(cache_dir: &Path, db: Arc<Database>, store: Arc<dyn crate::traits::QueryableStore>) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| CorviaError::Storage(format!("Failed to create tantivy dir: {e}")))?;

        let mut schema_builder = Schema::builder();
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("corvia_code")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let field_content = schema_builder.add_text_field("content", text_options);
        let field_entry_id = schema_builder.add_text_field("entry_id", STRING | STORED);
        let field_scope_id = schema_builder.add_text_field("scope_id", STRING);
        let field_content_role = schema_builder.add_text_field("content_role", STRING);
        let field_source_origin = schema_builder.add_text_field("source_origin", STRING);
        let schema = schema_builder.build();

        // Open or create the index.
        let mmap_dir = tantivy::directory::MmapDirectory::open(cache_dir)
            .map_err(|e| CorviaError::Storage(format!("tantivy dir open: {e}")))?;
        let index_exists = Index::exists(&mmap_dir)
            .map_err(|e| CorviaError::Storage(format!("tantivy index exists check: {e}")))?;
        let index = if index_exists {
            Index::open_in_dir(cache_dir)
                .map_err(|e| CorviaError::Storage(format!("tantivy index open: {e}")))?
        } else {
            Index::create_in_dir(cache_dir, schema.clone())
                .map_err(|e| CorviaError::Storage(format!("tantivy index create: {e}")))?
        };

        // Register the custom tokenizer.
        index
            .tokenizers()
            .register("corvia_code", CorviaCodeTokenizer);

        let writer = index
            .writer(50_000_000) // 50MB heap
            .map_err(|e| CorviaError::Storage(format!("tantivy writer: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| CorviaError::Storage(format!("tantivy reader: {e}")))?;

        let gen_meta_path = cache_dir.join(GEN_META_FILE);

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            store,
            field_content,
            field_entry_id,
            field_scope_id,
            field_content_role,
            field_source_origin,
            pending_count: AtomicU64::new(0),
            db,
            gen_meta_path,
        })
    }

    /// Check if the tantivy index is stale relative to the Redb write generation.
    pub fn is_stale(&self) -> Result<bool> {
        let redb_gen = self.read_redb_generation()?;
        let synced_gen = self.read_synced_generation();
        Ok(redb_gen != synced_gen)
    }

    /// Increment the FTS write generation in Redb.
    pub fn increment_generation(&self) -> Result<u64> {
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Storage(format!("redb write txn: {e}")))?;
        let new_gen = {
            let mut table = write_txn.open_table(META)
                .map_err(|e| CorviaError::Storage(format!("redb META open: {e}")))?;
            let current = match table.get(FTS_WRITE_GEN_KEY) {
                Ok(Some(v)) => v.value(),
                _ => 0,
            };
            let next = current + 1;
            table.insert(FTS_WRITE_GEN_KEY, next)
                .map_err(|e| CorviaError::Storage(format!("redb META insert: {e}")))?;
            next
        };
        write_txn.commit()
            .map_err(|e| CorviaError::Storage(format!("redb commit: {e}")))?;
        Ok(new_gen)
    }

    /// Persist the current Redb generation as the synced generation.
    fn persist_synced_generation(&self) -> Result<()> {
        let current_gen = self.read_redb_generation()?;
        let json = serde_json::json!({ "last_synced_generation": current_gen });
        std::fs::write(&self.gen_meta_path, serde_json::to_string_pretty(&json).unwrap())
            .map_err(|e| CorviaError::Storage(format!("write gen meta: {e}")))?;
        Ok(())
    }

    fn read_redb_generation(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Storage(format!("redb read txn: {e}")))?;
        match read_txn.open_table(META) {
            Ok(table) => match table.get(FTS_WRITE_GEN_KEY) {
                Ok(Some(v)) => Ok(v.value()),
                _ => Ok(0),
            },
            Err(_) => Ok(0),
        }
    }

    fn read_synced_generation(&self) -> u64 {
        if let Ok(data) = std::fs::read_to_string(&self.gen_meta_path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            return json.get("last_synced_generation")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }
        0
    }

    /// Flush pending writes: commit the writer and reload the reader.
    async fn do_flush(&self) -> Result<()> {
        let mut writer = self.writer.lock().await;
        writer.commit()
            .map_err(|e| CorviaError::Storage(format!("tantivy commit: {e}")))?;
        self.reader.reload()
            .map_err(|e| CorviaError::Storage(format!("tantivy reader reload: {e}")))?;
        self.pending_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Remove all entries for a scope in a single writer lock acquisition.
    pub async fn remove_scope(&self, scope_id: &str) -> Result<()> {
        let term = tantivy::Term::from_field_text(self.field_scope_id, scope_id);
        let mut writer = self.writer.lock().await;
        writer.delete_term(term);
        writer.commit()
            .map_err(|e| CorviaError::Storage(format!("tantivy commit remove_scope: {e}")))?;
        drop(writer);
        self.reader.reload()
            .map_err(|e| CorviaError::Storage(format!("tantivy reader reload: {e}")))?;
        self.pending_count.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Build a tantivy document from a knowledge entry.
    fn build_document(&self, entry: &KnowledgeEntry) -> TantivyDocument {
        let mut doc = TantivyDocument::default();
        doc.add_text(self.field_content, &entry.content);
        doc.add_text(self.field_entry_id, entry.id.to_string());
        doc.add_text(self.field_scope_id, &entry.scope_id);
        if let Some(ref role) = entry.metadata.content_role {
            doc.add_text(self.field_content_role, role);
        }
        if let Some(ref origin) = entry.metadata.source_origin {
            doc.add_text(self.field_source_origin, origin);
        }
        doc
    }
}

#[async_trait]
impl FullTextSearchable for TantivyIndex {
    async fn search_text(
        &self,
        query: &str,
        scope_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.field_content]);

        // Build a boolean query: text match AND scope filter.
        let text_query = query_parser.parse_query(query)
            .map_err(|e| CorviaError::Storage(format!("tantivy query parse: {e}")))?;

        let scope_term = tantivy::Term::from_field_text(self.field_scope_id, scope_id);
        let scope_query = Box::new(tantivy::query::TermQuery::new(
            scope_term,
            IndexRecordOption::Basic,
        ));

        let combined = tantivy::query::BooleanQuery::new(vec![
            (tantivy::query::Occur::Must, text_query),
            (tantivy::query::Occur::Must, scope_query),
        ]);

        let top_docs = searcher.search(&combined, &TopDocs::with_limit(limit))
            .map_err(|e| CorviaError::Storage(format!("tantivy search: {e}")))?;

        // Look up full entries by ID from the store.
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_addr)
                .map_err(|e| CorviaError::Storage(format!("tantivy doc fetch: {e}")))?;

            if let Some(entry_id_val) = doc.get_first(self.field_entry_id)
                && let Some(id_str) = entry_id_val.as_str()
                && let Ok(id) = uuid::Uuid::parse_str(id_str)
                && let Ok(Some(entry)) = self.store.get(&id).await {
                results.push(SearchResult {
                    entry,
                    score,
                    tier: corvia_common::types::Tier::Hot,
                    retention_score: None,
                });
            }
        }

        Ok(results)
    }

    async fn index_entry(&self, entry: &KnowledgeEntry) -> Result<()> {
        let doc = self.build_document(entry);

        // Delete any existing document with this entry_id first (upsert).
        let term = tantivy::Term::from_field_text(self.field_entry_id, &entry.id.to_string());

        let needs_flush = {
            let mut writer = self.writer.lock().await;
            writer.delete_term(term);
            writer.add_document(doc)
                .map_err(|e| CorviaError::Storage(format!("tantivy add_document: {e}")))?;
            let count = self.pending_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= FLUSH_ENTRY_THRESHOLD {
                // Flush while holding the lock to prevent double-flush races.
                writer.commit()
                    .map_err(|e| CorviaError::Storage(format!("tantivy commit: {e}")))?;
                self.pending_count.store(0, Ordering::SeqCst);
                true
            } else {
                false
            }
        };

        if needs_flush {
            self.reader.reload()
                .map_err(|e| CorviaError::Storage(format!("tantivy reader reload: {e}")))?;
        }

        Ok(())
    }

    async fn remove_entry(&self, entry_id: &uuid::Uuid) -> Result<()> {
        let term = tantivy::Term::from_field_text(self.field_entry_id, &entry_id.to_string());

        let needs_flush = {
            let mut writer = self.writer.lock().await;
            writer.delete_term(term);
            let count = self.pending_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= FLUSH_ENTRY_THRESHOLD {
                writer.commit()
                    .map_err(|e| CorviaError::Storage(format!("tantivy commit: {e}")))?;
                self.pending_count.store(0, Ordering::SeqCst);
                true
            } else {
                false
            }
        };

        if needs_flush {
            self.reader.reload()
                .map_err(|e| CorviaError::Storage(format!("tantivy reader reload: {e}")))?;
        }

        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        self.do_flush().await
    }

    async fn rebuild_from_store(&self, entries: &[KnowledgeEntry]) -> Result<usize> {
        info!(total = entries.len(), "starting tantivy index rebuild");

        // Clear existing index.
        {
            let mut writer = self.writer.lock().await;
            writer.delete_all_documents()
                .map_err(|e| CorviaError::Storage(format!("tantivy delete all: {e}")))?;
            writer.commit()
                .map_err(|e| CorviaError::Storage(format!("tantivy commit clear: {e}")))?;
        }

        // Index all entries in a single batch.
        let mut indexed = 0usize;
        {
            let mut writer = self.writer.lock().await;
            for (i, entry) in entries.iter().enumerate() {
                if entry.content.is_empty() {
                    continue;
                }
                let doc = self.build_document(entry);
                writer.add_document(doc)
                    .map_err(|e| CorviaError::Storage(format!("tantivy add_document: {e}")))?;
                indexed += 1;

                if (i + 1) % REBUILD_LOG_INTERVAL == 0 {
                    info!(progress = i + 1, total = entries.len(), "tantivy rebuild progress");
                }
            }

            writer.commit()
                .map_err(|e| CorviaError::Storage(format!("tantivy commit rebuild: {e}")))?;
        }

        self.reader.reload()
            .map_err(|e| CorviaError::Storage(format!("tantivy reader reload: {e}")))?;
        self.pending_count.store(0, Ordering::Relaxed);

        // Persist synced generation.
        self.persist_synced_generation()?;

        info!(indexed, "tantivy index rebuild complete");
        Ok(indexed)
    }

    async fn entry_count(&self) -> Result<u64> {
        let searcher = self.reader.searcher();
        Ok(searcher.num_docs())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tokenizer tests --

    #[test]
    fn test_tokenize_snake_case() {
        let tokens = tokenize_code_aware("snake_case");
        assert!(tokens.contains(&"snake_case".to_string()));
        assert!(tokens.contains(&"snake".to_string()));
        assert!(tokens.contains(&"case".to_string()));
    }

    #[test]
    fn test_tokenize_camel_case() {
        let tokens = tokenize_code_aware("CamelCase");
        assert!(tokens.contains(&"camelcase".to_string()));
        assert!(tokens.contains(&"camel".to_string()));
        assert!(tokens.contains(&"case".to_string()));
    }

    #[test]
    fn test_tokenize_upper_snake() {
        let tokens = tokenize_code_aware("EF_SEARCH");
        assert!(tokens.contains(&"ef_search".to_string()));
        assert!(tokens.contains(&"ef".to_string()));
        assert!(tokens.contains(&"search".to_string()));
    }

    #[test]
    fn test_tokenize_compound_snake() {
        let tokens = tokenize_code_aware("corvia_search");
        assert!(tokens.contains(&"corvia_search".to_string()));
        assert!(tokens.contains(&"corvia".to_string()));
        assert!(tokens.contains(&"search".to_string()));
    }

    #[test]
    fn test_tokenize_file_extension() {
        let tokens = tokenize_code_aware("retriever.rs");
        assert!(tokens.contains(&"retriever.rs".to_string()));
        assert!(tokens.contains(&"retriever".to_string()));
        assert!(tokens.contains(&"rs".to_string()));
    }

    #[test]
    fn test_tokenize_file_path() {
        let tokens = tokenize_code_aware("path/to/file.rs");
        assert!(tokens.contains(&"path/to/file.rs".to_string()));
        assert!(tokens.contains(&"path".to_string()));
        assert!(tokens.contains(&"to".to_string()));
        assert!(tokens.contains(&"file".to_string()));
        assert!(tokens.contains(&"rs".to_string()));
    }

    #[test]
    fn test_tokenize_version_with_v() {
        let tokens = tokenize_code_aware("v0.4.5");
        assert!(tokens.contains(&"v0.4.5".to_string()));
        assert!(tokens.contains(&"0.4.5".to_string()));
        assert!(tokens.contains(&"v0".to_string()));
        assert!(tokens.contains(&"4".to_string()));
        assert!(tokens.contains(&"5".to_string()));
    }

    #[test]
    fn test_tokenize_uuid() {
        let tokens = tokenize_code_aware("550e8400-e29b-41d4-a716-446655440000");
        assert!(tokens.contains(&"550e8400-e29b-41d4-a716-446655440000".to_string()));
        assert!(tokens.contains(&"550e8400".to_string()));
        assert!(tokens.contains(&"e29b".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize_code_aware("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_plain_word() {
        let tokens = tokenize_code_aware("hello");
        assert_eq!(tokens, vec!["hello"]);
    }

    // -- TantivyIndex integration tests --

    fn make_test_entry(content: &str, scope: &str) -> KnowledgeEntry {
        let mut entry = KnowledgeEntry::new(content.to_string(), scope.to_string(), "v1".to_string());
        entry.metadata.content_role = Some("code".to_string());
        entry.metadata.source_origin = Some("repo:corvia".to_string());
        // Set a minimal embedding so entries can be inserted into LiteStore.
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        entry
    }

    /// Insert entry into both tantivy and the store (for search_text lookups).
    async fn index_with_store(
        idx: &TantivyIndex,
        store: &Arc<crate::lite_store::LiteStore>,
        entry: &KnowledgeEntry,
    ) {
        use crate::traits::QueryableStore;
        store.insert(entry).await.unwrap();
        idx.index_entry(entry).await.unwrap();
    }

    fn create_test_store(dir: &Path) -> (Arc<Database>, Arc<crate::lite_store::LiteStore>) {
        let store = Arc::new(crate::lite_store::LiteStore::open(dir, 3).unwrap());
        let db = store.db().clone();
        // Ensure META table exists.
        let write_txn = db.begin_write().unwrap();
        {
            let _table = write_txn.open_table(META).unwrap();
        }
        write_txn.commit().unwrap();
        (db, store)
    }

    #[tokio::test]
    async fn test_tantivy_index_entry_and_search() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entry = make_test_entry("The VectorSearcher uses HNSW for approximate nearest neighbor search", "test-scope");
        index_with_store(&idx, &store, &entry).await;
        idx.flush().await.unwrap();

        let results = idx.search_text("HNSW nearest neighbor", "test-scope", 10).await.unwrap();
        assert!(!results.is_empty(), "should find the indexed entry");
        assert_eq!(results[0].entry.id, entry.id);
    }

    #[tokio::test]
    async fn test_tantivy_scope_isolation() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entry_a = make_test_entry("tantivy full text search", "scope-a");
        let entry_b = make_test_entry("tantivy full text search", "scope-b");
        index_with_store(&idx, &store, &entry_a).await;
        index_with_store(&idx, &store, &entry_b).await;
        idx.flush().await.unwrap();

        let results_a = idx.search_text("tantivy", "scope-a", 10).await.unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0].entry.id, entry_a.id);

        let results_b = idx.search_text("tantivy", "scope-b", 10).await.unwrap();
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].entry.id, entry_b.id);
    }

    #[tokio::test]
    async fn test_tantivy_remove_entry() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entry = make_test_entry("removable content", "test-scope");
        index_with_store(&idx, &store, &entry).await;
        idx.flush().await.unwrap();

        assert!(!idx.search_text("removable", "test-scope", 10).await.unwrap().is_empty());

        idx.remove_entry(&entry.id).await.unwrap();
        idx.flush().await.unwrap();

        assert!(idx.search_text("removable", "test-scope", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_tantivy_rebuild() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entries: Vec<KnowledgeEntry> = (0..5)
            .map(|i| make_test_entry(&format!("rebuild test entry {i}"), "rebuild-scope"))
            .collect();

        for entry in &entries {
            store.insert(entry).await.unwrap();
        }

        let count = idx.rebuild_from_store(&entries).await.unwrap();
        assert_eq!(count, 5);

        let results = idx.search_text("rebuild test", "rebuild-scope", 10).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_tantivy_entry_count() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 0);

        for i in 0..3 {
            let entry = make_test_entry(&format!("count test {i}"), "count-scope");
            idx.index_entry(&entry).await.unwrap();
        }
        idx.flush().await.unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_tantivy_empty_content_not_indexed() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entries = vec![
            make_test_entry("", "scope"), // empty
            make_test_entry("non-empty content", "scope"),
        ];

        let count = idx.rebuild_from_store(&entries).await.unwrap();
        assert_eq!(count, 1, "empty content should not be indexed");
    }

    #[tokio::test]
    async fn test_tantivy_staleness_detection() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        // Initially both are 0, so not stale.
        assert!(!idx.is_stale().unwrap());

        // Increment generation (simulating an insert without tantivy sync).
        idx.increment_generation().unwrap();
        assert!(idx.is_stale().unwrap());

        // After rebuild, persist synced generation.
        idx.persist_synced_generation().unwrap();
        assert!(!idx.is_stale().unwrap());
    }

    #[tokio::test]
    async fn test_tantivy_code_aware_search() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entry = make_test_entry(
            "The VectorSearcher in pipeline/searcher.rs uses EF_SEARCH constant",
            "test-scope",
        );
        index_with_store(&idx, &store, &entry).await;
        idx.flush().await.unwrap();

        // Should find via sub-token "searcher" from both VectorSearcher and searcher.rs.
        let results = idx.search_text("searcher", "test-scope", 10).await.unwrap();
        assert!(!results.is_empty(), "code-aware tokenizer should split CamelCase and file paths");
    }

    #[tokio::test]
    async fn test_tantivy_upsert_replaces_document() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let mut entry = make_test_entry("original content about vectors", "test-scope");
        index_with_store(&idx, &store, &entry).await;
        idx.flush().await.unwrap();

        // Update the same entry with different content.
        entry.content = "updated content about tantivy BM25 search".to_string();
        idx.index_entry(&entry).await.unwrap();
        idx.flush().await.unwrap();

        // Should NOT find old content.
        let old_results = idx.search_text("vectors", "test-scope", 10).await.unwrap();
        assert!(old_results.is_empty(), "old content should be replaced");

        // Should find new content.
        let new_results = idx.search_text("tantivy BM25", "test-scope", 10).await.unwrap();
        assert_eq!(new_results.len(), 1);
        assert_eq!(new_results[0].entry.id, entry.id);

        // Entry count should be 1, not 2.
        assert_eq!(idx.entry_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_tantivy_first_run_bootstrap() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("store");
        let cache_dir = dir.path().join("nonexistent").join("tantivy");
        let (db, store) = create_test_store(&store_dir);
        store.init_schema().await.unwrap();

        // First run: directory does not exist, should be created.
        assert!(!cache_dir.exists());
        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();
        assert!(cache_dir.exists());

        // Should work normally after bootstrap.
        let entry = make_test_entry("bootstrap test content", "bootstrap-scope");
        index_with_store(&idx, &store, &entry).await;
        idx.flush().await.unwrap();

        let results = idx.search_text("bootstrap", "bootstrap-scope", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_tantivy_remove_scope() {
        use crate::traits::QueryableStore;
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());
        store.init_schema().await.unwrap();

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        // Index entries in two scopes.
        for i in 0..3 {
            let entry = make_test_entry(&format!("scope-a content {i}"), "scope-a");
            index_with_store(&idx, &store, &entry).await;
        }
        for i in 0..2 {
            let entry = make_test_entry(&format!("scope-b content {i}"), "scope-b");
            index_with_store(&idx, &store, &entry).await;
        }
        idx.flush().await.unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 5);

        // Remove scope-a.
        idx.remove_scope("scope-a").await.unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 2);
        assert!(idx.search_text("content", "scope-a", 10).await.unwrap().is_empty());
        assert_eq!(idx.search_text("content", "scope-b", 10).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_tantivy_entry_count_after_remove() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("tantivy");
        let (db, store) = create_test_store(dir.path());

        let idx = TantivyIndex::open(&cache_dir, db, store.clone() as Arc<dyn crate::traits::QueryableStore>).unwrap();

        let entry = make_test_entry("count decrement test", "count-scope");
        idx.index_entry(&entry).await.unwrap();
        idx.flush().await.unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 1);

        idx.remove_entry(&entry.id).await.unwrap();
        idx.flush().await.unwrap();

        assert_eq!(idx.entry_count().await.unwrap(), 0);
    }
}
