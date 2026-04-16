//! Tantivy BM25 full-text index.

use std::path::Path;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, STRING, STORED, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::types::Kind;

/// Tantivy-backed BM25 full-text index for knowledge chunks.
pub struct TantivyIndex {
    index: Index,
    pub reader: IndexReader,
    #[allow(dead_code)]
    schema: Schema,
    chunk_id: Field,
    entry_id: Field,
    body: Field,
    kind: Field,
    superseded: Field,
}

impl TantivyIndex {
    /// Open (or create) a Tantivy index at the given directory path.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("creating tantivy index dir: {}", path.display()))?;

        let mut builder = Schema::builder();
        let chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
        let entry_id = builder.add_text_field("entry_id", STRING | STORED);
        let body = builder.add_text_field("body", TEXT | STORED);
        let kind = builder.add_text_field("kind", STRING);
        let superseded = builder.add_text_field("superseded", STRING);
        let schema = builder.build();

        let dir = MmapDirectory::open(path)
            .with_context(|| format!("opening MmapDirectory: {}", path.display()))?;

        let index = Index::open_or_create(dir, schema.clone())
            .context("opening or creating tantivy index")?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("creating index reader")?;

        Ok(Self {
            index,
            reader,
            schema,
            chunk_id,
            entry_id,
            body,
            kind,
            superseded,
        })
    }

    /// Create an `IndexWriter` with a 50 MB heap budget.
    pub fn writer(&self) -> Result<IndexWriter> {
        self.index
            .writer(50_000_000)
            .context("creating tantivy index writer")
    }

    /// Add a document to the index via the given writer.
    pub fn add_doc(
        &self,
        writer: &IndexWriter,
        chunk_id: &str,
        entry_id: &str,
        body: &str,
        kind: Kind,
        superseded: bool,
    ) -> Result<()> {
        let superseded_str = if superseded { "true" } else { "false" };
        let doc = tantivy::doc!(
            self.chunk_id => chunk_id,
            self.entry_id => entry_id,
            self.body => body,
            self.kind => kind.to_string(),
            self.superseded => superseded_str,
        );
        writer
            .add_document(doc)
            .context("adding document to tantivy index")?;
        Ok(())
    }

    /// Search the index, returning `(chunk_id, entry_id, score)` tuples.
    ///
    /// Always excludes superseded documents. Optionally filters by `Kind`.
    pub fn search(
        &self,
        query_text: &str,
        kind_filter: Option<Kind>,
        limit: usize,
    ) -> Result<Vec<(String, String, f32)>> {
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.body]);
        let text_query = query_parser
            .parse_query(query_text)
            .context("parsing search query")?;

        // Always exclude superseded docs.
        let not_superseded = TermQuery::new(
            Term::from_field_text(self.superseded, "false"),
            IndexRecordOption::Basic,
        );

        let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(not_superseded)),
        ];

        // Optional kind filter.
        if let Some(k) = kind_filter {
            let kind_term = TermQuery::new(
                Term::from_field_text(self.kind, &k.to_string()),
                IndexRecordOption::Basic,
            );
            clauses.push((Occur::Must, Box::new(kind_term)));
        }

        let combined = BooleanQuery::new(clauses);
        let top_docs = searcher
            .search(&combined, &TopDocs::with_limit(limit))
            .context("executing tantivy search")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_addr)
                .context("retrieving document from tantivy")?;

            let cid = doc
                .get_first(self.chunk_id)
                .and_then(|v| match v {
                    tantivy::schema::OwnedValue::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            let eid = doc
                .get_first(self.entry_id)
                .and_then(|v| match v {
                    tantivy::schema::OwnedValue::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            results.push((cid, eid, score));
        }

        Ok(results)
    }

    /// Retrieve the stored body text for a specific chunk_id.
    ///
    /// Searches the index for an exact match on the chunk_id field and returns
    /// the stored body text. Returns `None` if the chunk_id is not found.
    pub fn get_chunk_text(&self, chunk_id: &str) -> Result<Option<String>> {
        let searcher = self.reader.searcher();

        let term_query = TermQuery::new(
            Term::from_field_text(self.chunk_id, chunk_id),
            IndexRecordOption::Basic,
        );

        let top_docs = searcher
            .search(&term_query, &TopDocs::with_limit(1))
            .context("searching for chunk_id in tantivy")?;

        if let Some((_score, doc_addr)) = top_docs.first() {
            let doc: TantivyDocument = searcher
                .doc(*doc_addr)
                .context("retrieving chunk document from tantivy")?;

            let body_text = doc
                .get_first(self.body)
                .and_then(|v| match v {
                    tantivy::schema::OwnedValue::Str(s) => Some(s.clone()),
                    _ => None,
                });

            Ok(body_text)
        } else {
            Ok(None)
        }
    }

    /// Retrieve the stored kind for a specific chunk_id.
    ///
    /// Returns `None` if the chunk_id is not found or the kind field is missing.
    pub fn get_chunk_kind(&self, chunk_id: &str) -> Result<Option<Kind>> {
        let searcher = self.reader.searcher();

        let term_query = TermQuery::new(
            Term::from_field_text(self.chunk_id, chunk_id),
            IndexRecordOption::Basic,
        );

        let top_docs = searcher
            .search(&term_query, &TopDocs::with_limit(1))
            .context("searching for chunk_id kind in tantivy")?;

        if let Some((_score, doc_addr)) = top_docs.first() {
            let doc: TantivyDocument = searcher
                .doc(*doc_addr)
                .context("retrieving chunk document for kind")?;

            let kind_str = doc
                .get_first(self.kind)
                .and_then(|v| match v {
                    tantivy::schema::OwnedValue::Str(s) => Some(s.clone()),
                    _ => None,
                });

            if let Some(s) = kind_str {
                match s.parse::<Kind>() {
                    Ok(k) => Ok(Some(k)),
                    Err(_) => Ok(None),
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Delete all documents belonging to a given entry_id.
    ///
    /// This removes superseded entries from BM25 search results. Since Tantivy
    /// does not support in-place field updates, deletion is the simplest correct
    /// approach for making superseded entries invisible.
    pub fn delete_by_entry_id(&self, writer: &IndexWriter, entry_id: &str) {
        let term = Term::from_field_text(self.entry_id, entry_id);
        writer.delete_term(term);
    }

    /// Total number of documents in the index (including superseded).
    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Delete all documents from the index and commit.
    pub fn clear(&self) -> Result<()> {
        let mut writer = self.writer()?;
        writer
            .delete_all_documents()
            .context("deleting all documents")?;
        writer.commit().context("committing clear")?;
        self.reader.reload().context("reloading reader after clear")?;
        Ok(())
    }

    /// Reload the reader to reflect the latest committed state.
    pub fn reload_reader(&self) -> Result<()> {
        self.reader.reload().context("reloading tantivy reader")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a TantivyIndex in a temporary directory.
    fn temp_index() -> (TantivyIndex, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let idx = TantivyIndex::open(dir.path()).expect("failed to open index");
        (idx, dir)
    }

    #[test]
    fn add_and_search() {
        let (idx, _dir) = temp_index();
        let mut writer = idx.writer().unwrap();

        idx.add_doc(
            &writer,
            "chunk-1",
            "entry-1",
            "rust programming language systems",
            Kind::Learning,
            false,
        )
        .unwrap();

        idx.add_doc(
            &writer,
            "chunk-2",
            "entry-2",
            "chocolate cake baking recipe",
            Kind::Learning,
            false,
        )
        .unwrap();

        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        let results = idx.search("rust", None, 10).unwrap();
        assert!(!results.is_empty(), "expected at least one result");
        assert_eq!(results[0].0, "chunk-1");
        assert_eq!(results[0].1, "entry-1");
        assert!(results[0].2 > 0.0, "score should be positive");
    }

    #[test]
    fn superseded_excluded_from_search() {
        let (idx, _dir) = temp_index();
        let mut writer = idx.writer().unwrap();

        idx.add_doc(
            &writer,
            "chunk-old",
            "entry-old",
            "rust memory safety borrow checker",
            Kind::Learning,
            true, // superseded
        )
        .unwrap();

        idx.add_doc(
            &writer,
            "chunk-new",
            "entry-new",
            "rust ownership model and lifetimes",
            Kind::Learning,
            false,
        )
        .unwrap();

        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        let results = idx.search("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1, "superseded doc should be excluded");
        assert_eq!(results[0].0, "chunk-new");
    }

    #[test]
    fn kind_filter() {
        let (idx, _dir) = temp_index();
        let mut writer = idx.writer().unwrap();

        idx.add_doc(
            &writer,
            "chunk-decision",
            "entry-decision",
            "we decided to use tantivy for full-text search",
            Kind::Decision,
            false,
        )
        .unwrap();

        idx.add_doc(
            &writer,
            "chunk-instruction",
            "entry-instruction",
            "always run tantivy search before vector search",
            Kind::Instruction,
            false,
        )
        .unwrap();

        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        // Filter to decisions only.
        let results = idx.search("tantivy", Some(Kind::Decision), 10).unwrap();
        assert_eq!(results.len(), 1, "only the decision should match");
        assert_eq!(results[0].0, "chunk-decision");

        // Filter to instructions only.
        let results = idx.search("tantivy", Some(Kind::Instruction), 10).unwrap();
        assert_eq!(results.len(), 1, "only the instruction should match");
        assert_eq!(results[0].0, "chunk-instruction");
    }

    #[test]
    fn empty_index_returns_empty() {
        let (idx, _dir) = temp_index();
        let results = idx.search("anything", None, 10).unwrap();
        assert!(results.is_empty(), "empty index should return no results");
    }

    #[test]
    fn clear_removes_all() {
        let (idx, _dir) = temp_index();

        {
            let mut writer = idx.writer().unwrap();

            idx.add_doc(
                &writer,
                "chunk-1",
                "entry-1",
                "some content here",
                Kind::Learning,
                false,
            )
            .unwrap();

            writer.commit().unwrap();
        } // writer dropped here, releasing the lock

        idx.reader.reload().unwrap();
        assert_eq!(idx.doc_count(), 1, "should have 1 doc before clear");

        idx.clear().unwrap();
        assert_eq!(idx.doc_count(), 0, "should have 0 docs after clear");
    }
}
