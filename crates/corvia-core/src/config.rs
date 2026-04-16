//! Configuration types (corvia.toml parsing).

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ── Default value functions ────────────────────────────────────────────

fn default_data_dir() -> PathBuf {
    PathBuf::from(".corvia")
}

fn default_max_tokens() -> usize {
    512
}

fn default_overlap_tokens() -> usize {
    64
}

fn default_min_tokens() -> usize {
    32
}

fn default_rrf_k() -> u32 {
    30
}

fn default_dedup_threshold() -> f32 {
    0.85
}

fn default_reranker_candidates() -> usize {
    50
}

fn default_brute_force_threshold() -> usize {
    10_000
}

fn default_default_limit() -> usize {
    5
}

fn default_embedding_model() -> String {
    "nomic-embed-text-v1.5".to_string()
}

fn default_reranker_model() -> String {
    "jina-v1-turbo".to_string()
}

// ── Config structs ─────────────────────────────────────────────────────

/// Top-level configuration loaded from `corvia.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Directory for all corvia data (entries, index, etc.).
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Chunking parameters.
    #[serde(default)]
    pub chunking: ChunkingConfig,

    /// Search parameters.
    #[serde(default)]
    pub search: SearchConfig,

    /// Embedding model configuration.
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

/// Controls how documents are split into chunks before embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum tokens per chunk.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Overlap tokens between consecutive chunks.
    #[serde(default = "default_overlap_tokens")]
    pub overlap_tokens: usize,

    /// Minimum tokens for a chunk to be kept.
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,
}

/// Controls hybrid search behavior (RRF fusion, deduplication, reranking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Reciprocal Rank Fusion constant (higher = less weight to top ranks).
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,

    /// Cosine similarity threshold above which results are considered duplicates.
    #[serde(default = "default_dedup_threshold")]
    pub dedup_threshold: f32,

    /// Number of candidates fetched before reranking.
    #[serde(default = "default_reranker_candidates")]
    pub reranker_candidates: usize,

    /// Entry count below which brute-force search is used instead of HNSW.
    #[serde(default = "default_brute_force_threshold")]
    pub brute_force_threshold: usize,

    /// Default number of results returned when the caller does not specify a limit.
    #[serde(default = "default_default_limit")]
    pub default_limit: usize,
}

/// Embedding and reranker model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model name.
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Reranker model name.
    #[serde(default = "default_reranker_model")]
    pub reranker_model: String,

    /// Optional explicit path to model weights.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
}

// ── Default implementations ────────────────────────────────────────────

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            chunking: ChunkingConfig::default(),
            search: SearchConfig::default(),
            embedding: EmbeddingConfig::default(),
        }
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            overlap_tokens: default_overlap_tokens(),
            min_tokens: default_min_tokens(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: default_rrf_k(),
            dedup_threshold: default_dedup_threshold(),
            reranker_candidates: default_reranker_candidates(),
            brute_force_threshold: default_brute_force_threshold(),
            default_limit: default_default_limit(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            reranker_model: default_reranker_model(),
            model_path: None,
        }
    }
}

// ── Config methods ─────────────────────────────────────────────────────

impl Config {
    /// Load configuration from a TOML file. Returns defaults if the file does not exist.
    ///
    /// Runs validation after loading. Returns an error if any constraint is violated.
    pub fn load(path: &Path) -> Result<Self> {
        let config = match std::fs::read_to_string(path) {
            Ok(contents) => {
                let config: Config = toml::from_str(&contents)?;
                config
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => return Err(e.into()),
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration constraints.
    ///
    /// Checks:
    /// - `overlap_tokens < max_tokens`
    /// - `min_tokens <= max_tokens`
    /// - `dedup_threshold` is between 0.0 and 1.0
    /// - `rrf_k > 0`
    pub fn validate(&self) -> Result<()> {
        if self.chunking.overlap_tokens >= self.chunking.max_tokens {
            anyhow::bail!(
                "overlap_tokens ({}) must be less than max_tokens ({})",
                self.chunking.overlap_tokens,
                self.chunking.max_tokens,
            );
        }
        if self.chunking.min_tokens > self.chunking.max_tokens {
            anyhow::bail!(
                "min_tokens ({}) must not exceed max_tokens ({})",
                self.chunking.min_tokens,
                self.chunking.max_tokens,
            );
        }
        if !(0.0..=1.0).contains(&self.search.dedup_threshold) {
            anyhow::bail!(
                "dedup_threshold ({}) must be between 0.0 and 1.0",
                self.search.dedup_threshold,
            );
        }
        if self.search.rrf_k == 0 {
            anyhow::bail!("rrf_k must be greater than 0");
        }
        Ok(())
    }

    /// Load config from a discovered project root.
    /// Looks for `.corvia/corvia.toml` relative to `base_dir`.
    pub fn load_discovered(base_dir: &Path) -> Result<Self> {
        let config_path = base_dir.join(".corvia").join("corvia.toml");
        Self::load(&config_path)
    }

    /// Path to the entries directory (`<data_dir>/entries`).
    pub fn entries_dir(&self) -> PathBuf {
        self.data_dir.join("entries")
    }

    /// Path to the index directory (`<data_dir>/index`).
    pub fn index_dir(&self) -> PathBuf {
        self.data_dir.join("index")
    }

    /// Path to the redb database file (`<data_dir>/index/store.redb`).
    pub fn redb_path(&self) -> PathBuf {
        self.index_dir().join("store.redb")
    }

    /// Path to the tantivy index directory (`<data_dir>/index/tantivy`).
    pub fn tantivy_dir(&self) -> PathBuf {
        self.index_dir().join("tantivy")
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn default_config_has_correct_values() {
        let cfg = Config::default();

        // data_dir
        assert_eq!(cfg.data_dir, PathBuf::from(".corvia"));

        // chunking
        assert_eq!(cfg.chunking.max_tokens, 512);
        assert_eq!(cfg.chunking.overlap_tokens, 64);
        assert_eq!(cfg.chunking.min_tokens, 32);

        // search
        assert_eq!(cfg.search.rrf_k, 30);
        assert!((cfg.search.dedup_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(cfg.search.reranker_candidates, 50);
        assert_eq!(cfg.search.brute_force_threshold, 10_000);
        assert_eq!(cfg.search.default_limit, 5);

        // embedding
        assert_eq!(cfg.embedding.model, "nomic-embed-text-v1.5");
        assert_eq!(cfg.embedding.reranker_model, "jina-v1-turbo");
        assert!(cfg.embedding.model_path.is_none());
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.toml");

        let cfg = Config::load(&missing).unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from(".corvia"));
        assert_eq!(cfg.chunking.max_tokens, 512);
        assert_eq!(cfg.search.rrf_k, 30);
        assert_eq!(cfg.embedding.model, "nomic-embed-text-v1.5");
    }

    #[test]
    fn load_partial_toml_fills_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corvia.toml");

        let mut file = std::fs::File::create(&path).unwrap();
        write!(
            file,
            r#"
data_dir = "/custom/data"

[chunking]
max_tokens = 1024

[search]
rrf_k = 60
"#
        )
        .unwrap();

        let cfg = Config::load(&path).unwrap();

        // Explicitly set values
        assert_eq!(cfg.data_dir, PathBuf::from("/custom/data"));
        assert_eq!(cfg.chunking.max_tokens, 1024);
        assert_eq!(cfg.search.rrf_k, 60);

        // Defaulted values
        assert_eq!(cfg.chunking.overlap_tokens, 64);
        assert_eq!(cfg.chunking.min_tokens, 32);
        assert!((cfg.search.dedup_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(cfg.search.reranker_candidates, 50);
        assert_eq!(cfg.search.brute_force_threshold, 10_000);
        assert_eq!(cfg.search.default_limit, 5);
        assert_eq!(cfg.embedding.model, "nomic-embed-text-v1.5");
        assert_eq!(cfg.embedding.reranker_model, "jina-v1-turbo");
        assert!(cfg.embedding.model_path.is_none());
    }

    #[test]
    fn load_discovered_from_corvia_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        std::fs::write(
            corvia.join("corvia.toml"),
            "[embedding]\nmodel = \"test-model\"\nreranker_model = \"test-reranker\"\n",
        )
        .unwrap();

        let config = Config::load_discovered(dir.path()).unwrap();
        assert_eq!(config.embedding.model, "test-model");
    }

    #[test]
    fn load_discovered_missing_returns_defaults() {
        let dir = tempfile::TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        // No corvia.toml — load should return defaults (NotFound -> Default)
        let config = Config::load_discovered(dir.path()).unwrap();
        assert_eq!(config.embedding.model, "nomic-embed-text-v1.5");
    }

    #[test]
    fn paths_are_derived_correctly() {
        let cfg = Config {
            data_dir: PathBuf::from("/srv/corvia"),
            ..Config::default()
        };

        assert_eq!(cfg.entries_dir(), PathBuf::from("/srv/corvia/entries"));
        assert_eq!(cfg.index_dir(), PathBuf::from("/srv/corvia/index"));
        assert_eq!(
            cfg.redb_path(),
            PathBuf::from("/srv/corvia/index/store.redb")
        );
        assert_eq!(
            cfg.tantivy_dir(),
            PathBuf::from("/srv/corvia/index/tantivy")
        );
    }
}
