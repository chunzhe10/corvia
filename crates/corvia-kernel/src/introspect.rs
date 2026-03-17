use serde::Deserialize;
use std::path::Path;
use crate::traits::{InferenceEngine, QueryableStore, IngestionAdapter};
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EntryMetadata, KnowledgeEntry};
use tracing::info;

/// Default batch size for embedding operations.
pub const EMBED_BATCH_SIZE: usize = 32;

/// Top-level structure of tests/introspect.toml
#[derive(Debug, Deserialize)]
pub struct IntrospectConfig {
    pub config: IntrospectMeta,
    pub query: Vec<CanonicalQuery>,
}

#[derive(Debug, Deserialize)]
pub struct IntrospectMeta {
    pub default_min_score: f64,
    pub scope_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CanonicalQuery {
    pub text: String,
    pub expect_file: String,
    pub min_score: Option<f64>,
}

impl CanonicalQuery {
    pub fn effective_min_score(&self, default: f64) -> f64 {
        self.min_score.unwrap_or(default)
    }
}

impl IntrospectConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CorviaError::Config(format!("Failed to read {}: {e}", path.display())))?;
        toml::from_str(&content)
            .map_err(|e| CorviaError::Config(format!("Failed to parse introspect config: {e}")))
    }

    pub fn default_queries() -> Self {
        Self {
            config: IntrospectMeta {
                default_min_score: 0.70,
                scope_id: "corvia-introspect".into(),
            },
            query: vec![
                CanonicalQuery {
                    text: "how does embedding work?".into(),
                    expect_file: "crates/corvia-kernel/src/embedding_pipeline.rs".into(),
                    min_score: None,
                },
                CanonicalQuery {
                    text: "how is knowledge stored in the database?".into(),
                    expect_file: "crates/corvia-kernel/src/lite_store.rs".into(),
                    min_score: None,
                },
                CanonicalQuery {
                    text: "what CLI commands are available?".into(),
                    expect_file: "crates/corvia-cli/src/main.rs".into(),
                    min_score: None,
                },
            ],
        }
    }
}

/// Result of a single canonical query evaluation.
#[derive(Debug)]
pub struct QueryResult {
    pub query_text: String,
    pub expect_file: String,
    pub actual_file: Option<String>,
    pub score: f64,
    pub min_score: f64,
}

impl QueryResult {
    pub fn file_matched(&self) -> bool {
        self.actual_file.as_ref()
            .map(|f| f.ends_with(&self.expect_file) || self.expect_file.ends_with(f.as_str()))
            .unwrap_or(false)
    }

    pub fn passed(&self) -> bool {
        self.file_matched() && self.score >= self.min_score
    }
}

/// Aggregate report from running all canonical queries.
#[derive(Debug)]
pub struct IntrospectReport {
    pub results: Vec<QueryResult>,
    pub chunks_ingested: usize,
}

impl IntrospectReport {
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed())
    }

    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed()).count()
    }

    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed()).count()
    }

    pub fn avg_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.score).sum();
        sum / self.results.len() as f64
    }

    pub fn exit_code(&self) -> i32 {
        if self.all_passed() { 0 } else { 1 }
    }
}

/// The Introspect pipeline. Orchestrates env check, self-ingest, self-query.
pub struct Introspect {
    config: IntrospectConfig,
}

impl Introspect {
    pub fn new(config: IntrospectConfig) -> Self {
        Self { config }
    }

    pub fn from_file_or_default(path: &Path) -> Self {
        let config = if path.exists() {
            IntrospectConfig::load(path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load {}: {e}, using defaults", path.display());
                IntrospectConfig::default_queries()
            })
        } else {
            info!("No introspect.toml found, using built-in queries");
            IntrospectConfig::default_queries()
        };
        Self::new(config)
    }

    /// Phase 1: Check environment for LiteStore (Ollama).
    pub async fn check_env_lite(
        &self,
        ollama_url: &str,
    ) -> Vec<(&'static str, bool)> {
        let ollama = crate::ollama_engine::OllamaEngine::check_health(ollama_url).await;
        vec![
            ("Ollama", ollama),
        ]
    }

    /// Phase 2: Ingest Corvia's own source code.
    ///
    /// Uses the D69 `ingest_sources` API: the adapter returns raw
    /// [`SourceFile`](crate::chunking_strategy::SourceFile)s which are
    /// converted to [`KnowledgeEntry`]s, embedded, and stored.
    ///
    /// NOTE: This bypasses the `ChunkingPipeline` for now (each source
    /// file becomes one entry). Task 10 will wire full pipeline chunking.
    pub async fn ingest_self(
        &self,
        source_path: &str,
        adapter: &dyn IngestionAdapter,
        engine: &dyn InferenceEngine,
        store: &dyn QueryableStore,
    ) -> Result<usize> {
        let source_files = adapter.ingest_sources(source_path).await?;
        let total = source_files.len();
        info!("{total} source files extracted from {source_path}");

        // Convert SourceFiles to KnowledgeEntries (one entry per file for now).
        let entries: Vec<KnowledgeEntry> = source_files
            .iter()
            .map(|sf| {
                let mut entry = KnowledgeEntry::new(
                    sf.content.clone(),
                    self.config.config.scope_id.clone(),
                    sf.metadata.source_version.clone(),
                );
                entry.metadata = EntryMetadata {
                    source_file: Some(sf.metadata.file_path.clone()),
                    language: sf.metadata.language.clone(),
                    chunk_type: Some("file".into()),
                    ..Default::default()
                };
                entry
            })
            .collect();

        let mut stored = 0;
        for batch in entries.chunks(EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored += 1;
            }
        }
        info!("{stored}/{total} source files embedded and stored");
        Ok(stored)
    }

    /// Phase 2 (alternative): Ingest pre-collected source files.
    ///
    /// Like `ingest_self` but takes already-collected source files instead of
    /// requiring an `IngestionAdapter`. Used when adapters are spawned as
    /// external processes via the adapter plugin system.
    pub async fn ingest_source_files(
        &self,
        source_files: &[crate::chunking_strategy::SourceFile],
        engine: &dyn InferenceEngine,
        store: &dyn QueryableStore,
    ) -> Result<usize> {
        let total = source_files.len();
        info!("{total} source files provided for introspect ingestion");

        let entries: Vec<KnowledgeEntry> = source_files
            .iter()
            .map(|sf| {
                let mut entry = KnowledgeEntry::new(
                    sf.content.clone(),
                    self.config.config.scope_id.clone(),
                    sf.metadata.source_version.clone(),
                );
                entry.metadata = EntryMetadata {
                    source_file: Some(sf.metadata.file_path.clone()),
                    language: sf.metadata.language.clone(),
                    chunk_type: Some("file".into()),
                    ..Default::default()
                };
                entry
            })
            .collect();

        let mut stored = 0;
        for batch in entries.chunks(EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored += 1;
            }
        }
        info!("{stored}/{total} source files embedded and stored");
        Ok(stored)
    }

    /// Phase 3: Run canonical queries and score results.
    pub async fn query_self(
        &self,
        engine: &dyn InferenceEngine,
        store: &dyn QueryableStore,
    ) -> Result<Vec<QueryResult>> {
        let mut results = Vec::new();
        let default_min = self.config.config.default_min_score;

        for query in &self.config.query {
            let embedding = engine.embed(&query.text).await?;
            let search_results = store.search(
                &embedding,
                &self.config.config.scope_id,
                1,
            ).await?;

            let (actual_file, score) = if let Some(top) = search_results.first() {
                let file = top.entry.metadata.source_file.clone();
                (file, top.score as f64)
            } else {
                (None, 0.0)
            };

            results.push(QueryResult {
                query_text: query.text.clone(),
                expect_file: query.expect_file.clone(),
                actual_file,
                score,
                min_score: query.effective_min_score(default_min),
            });
        }

        Ok(results)
    }

    pub fn scope_id(&self) -> &str {
        &self.config.config.scope_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_introspect_config() {
        let toml_str = r#"
[config]
default_min_score = 0.70
scope_id = "corvia-introspect"

[[query]]
text = "how does embedding work?"
expect_file = "src/embedding.rs"
min_score = 0.75

[[query]]
text = "what commands exist?"
expect_file = "src/main.rs"
"#;
        let config: IntrospectConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.config.scope_id, "corvia-introspect");
        assert_eq!(config.query.len(), 2);
        assert_eq!(config.query[0].text, "how does embedding work?");
        assert_eq!(config.query[0].expect_file, "src/embedding.rs");
        assert_eq!(config.query[0].min_score, Some(0.75));
        assert_eq!(config.query[1].min_score, None);
    }

    #[test]
    fn test_effective_min_score_fallback() {
        let q = CanonicalQuery {
            text: "test".into(),
            expect_file: "test.rs".into(),
            min_score: None,
        };
        assert_eq!(q.effective_min_score(0.70), 0.70);

        let q2 = CanonicalQuery {
            text: "test".into(),
            expect_file: "test.rs".into(),
            min_score: Some(0.80),
        };
        assert_eq!(q2.effective_min_score(0.70), 0.80);
    }

    #[test]
    fn test_query_result_pass() {
        let result = QueryResult {
            query_text: "test query".into(),
            expect_file: "src/main.rs".into(),
            actual_file: Some("src/main.rs".into()),
            score: 0.85,
            min_score: 0.70,
        };
        assert!(result.passed());
        assert!(result.file_matched());
    }

    #[test]
    fn test_query_result_fail_score() {
        let result = QueryResult {
            query_text: "test query".into(),
            expect_file: "src/main.rs".into(),
            actual_file: Some("src/main.rs".into()),
            score: 0.50,
            min_score: 0.70,
        };
        assert!(!result.passed());
        assert!(result.file_matched());
    }

    #[test]
    fn test_query_result_fail_wrong_file() {
        let result = QueryResult {
            query_text: "test query".into(),
            expect_file: "src/main.rs".into(),
            actual_file: Some("src/other.rs".into()),
            score: 0.85,
            min_score: 0.70,
        };
        assert!(!result.passed());
        assert!(!result.file_matched());
    }

    #[test]
    fn test_query_result_fail_no_results() {
        let result = QueryResult {
            query_text: "test query".into(),
            expect_file: "src/main.rs".into(),
            actual_file: None,
            score: 0.0,
            min_score: 0.70,
        };
        assert!(!result.passed());
    }

    #[test]
    fn test_introspect_report_all_passed() {
        let report = IntrospectReport {
            results: vec![
                QueryResult {
                    query_text: "q1".into(),
                    expect_file: "a.rs".into(),
                    actual_file: Some("a.rs".into()),
                    score: 0.85,
                    min_score: 0.70,
                },
                QueryResult {
                    query_text: "q2".into(),
                    expect_file: "b.rs".into(),
                    actual_file: Some("b.rs".into()),
                    score: 0.75,
                    min_score: 0.70,
                },
            ],
            chunks_ingested: 42,
        };
        assert!(report.all_passed());
        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.fail_count(), 0);
        assert!((report.avg_score() - 0.80).abs() < 0.001);
        assert_eq!(report.exit_code(), 0);
    }
}
