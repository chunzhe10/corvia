use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::errors::{CorviaError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StoreType {
    #[default]
    Lite,
    Surrealdb,
    Postgres,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceProvider {
    #[default]
    Ollama,
    Vllm,
    Corvia,
}

fn default_repos_dir() -> String {
    "repos".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepoConfig {
    pub name: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<String>,
    pub namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceConfig {
    #[serde(default = "default_repos_dir")]
    pub repos_dir: String,
    #[serde(default)]
    pub repos: Vec<RepoConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Inference provider for reasoning (reuses the embedding provider enum).
    #[serde(default)]
    pub provider: InferenceProvider,
    /// Model for reasoning tasks (e.g., "llama3.2:3b").
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorviaConfig {
    pub project: ProjectConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub server: ServerConfig,
    #[serde(default)]
    pub agent_lifecycle: AgentLifecycleConfig,
    #[serde(default)]
    pub merge: MergeConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<WorkspaceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub rag: RagConfig,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapters: Option<AdaptersConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SourceConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub scope_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default)]
    pub store_type: StoreType,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    pub surrealdb_url: Option<String>,
    pub surrealdb_ns: Option<String>,
    pub surrealdb_db: Option<String>,
    pub surrealdb_user: Option<String>,
    pub surrealdb_pass: Option<String>,
    pub postgres_url: Option<String>,
}

fn default_data_dir() -> String {
    ".corvia".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub provider: InferenceProvider,
    pub model: String,
    pub url: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLifecycleConfig {
    #[serde(default = "default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,
    #[serde(default = "default_stale_timeout_secs")]
    pub stale_timeout_secs: u64,
    #[serde(default = "default_orphan_grace_secs")]
    pub orphan_grace_secs: u64,
    #[serde(default = "default_gc_orphan_after_secs")]
    pub gc_orphan_after_secs: u64,
    #[serde(default = "default_gc_closed_session_after_secs")]
    pub gc_closed_session_after_secs: u64,
    #[serde(default = "default_gc_inactive_agent_after_secs")]
    pub gc_inactive_agent_after_secs: u64,
}

fn default_heartbeat_interval_secs() -> u64 { 30 }
fn default_stale_timeout_secs() -> u64 { 300 }      // 5 min
fn default_orphan_grace_secs() -> u64 { 1200 }      // 20 min
fn default_gc_orphan_after_secs() -> u64 { 86400 }  // 24 hr
fn default_gc_closed_session_after_secs() -> u64 { 604800 } // 7 days
fn default_gc_inactive_agent_after_secs() -> u64 { 2592000 } // 30 days

impl Default for AgentLifecycleConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: default_heartbeat_interval_secs(),
            stale_timeout_secs: default_stale_timeout_secs(),
            orphan_grace_secs: default_orphan_grace_secs(),
            gc_orphan_after_secs: default_gc_orphan_after_secs(),
            gc_closed_session_after_secs: default_gc_closed_session_after_secs(),
            gc_inactive_agent_after_secs: default_gc_inactive_agent_after_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Inference provider for merge LLM calls.
    #[serde(default)]
    pub provider: InferenceProvider,
    /// Chat model for LLM-assisted merge (Ollama).
    #[serde(default = "default_merge_model")]
    pub model: String,
    /// Cosine similarity threshold — above this means conflict.
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Max retries for failed LLM merges.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_merge_model() -> String { "llama3.2".into() }
fn default_similarity_threshold() -> f32 { 0.85 }
fn default_max_retries() -> u32 { 3 }

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            provider: InferenceProvider::Ollama,
            model: default_merge_model(),
            similarity_threshold: default_similarity_threshold(),
            max_retries: default_max_retries(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default = "default_rag_limit")]
    pub default_limit: usize,
    #[serde(default = "default_graph_expand")]
    pub graph_expand: bool,
    #[serde(default = "default_graph_depth")]
    pub graph_depth: usize,
    #[serde(default = "default_graph_alpha")]
    pub graph_alpha: f32,
    #[serde(default = "default_reserve_for_answer")]
    pub reserve_for_answer: f32,
    #[serde(default)]
    pub max_context_tokens: usize,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default = "default_graph_oversample")]
    pub graph_oversample_factor: usize,
}

fn default_rag_limit() -> usize { 10 }
fn default_graph_expand() -> bool { true }
fn default_graph_depth() -> usize { 2 }
fn default_graph_alpha() -> f32 { 0.3 }
fn default_reserve_for_answer() -> f32 { 0.2 }
fn default_graph_oversample() -> usize { 3 }

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            default_limit: default_rag_limit(),
            graph_expand: default_graph_expand(),
            graph_depth: default_graph_depth(),
            graph_alpha: default_graph_alpha(),
            reserve_for_answer: default_reserve_for_answer(),
            max_context_tokens: 0,
            system_prompt: String::new(),
            graph_oversample_factor: default_graph_oversample(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    #[serde(default = "default_max_chunk_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_min_chunk_tokens")]
    pub min_tokens: usize,
    #[serde(default = "default_overlap_tokens")]
    pub overlap_tokens: usize,
    #[serde(default = "default_chunking_strategy")]
    pub strategy: String,
}

fn default_max_chunk_tokens() -> usize { 512 }
fn default_min_chunk_tokens() -> usize { 32 }
fn default_overlap_tokens() -> usize { 64 }
fn default_chunking_strategy() -> String { "auto".into() }

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_chunk_tokens(),
            min_tokens: default_min_chunk_tokens(),
            overlap_tokens: default_overlap_tokens(),
            strategy: default_chunking_strategy(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub search_dirs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_config: Option<toml::Value>,
}

impl Default for CorviaConfig {
    fn default() -> Self {
        Self {
            project: ProjectConfig {
                name: "default".into(),
                scope_id: "default".into(),
            },
            storage: StorageConfig {
                store_type: StoreType::Lite,
                data_dir: ".corvia".into(),
                surrealdb_url: None,
                surrealdb_ns: None,
                surrealdb_db: None,
                surrealdb_user: None,
                surrealdb_pass: None,
                postgres_url: None,
            },
            embedding: EmbeddingConfig {
                provider: InferenceProvider::Ollama,
                model: "nomic-embed-text".into(),
                url: "http://127.0.0.1:11434".into(),
                dimensions: 768,
            },
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8020,
            },
            agent_lifecycle: AgentLifecycleConfig::default(),
            merge: MergeConfig::default(),
            workspace: None,
            reasoning: None,
            rag: RagConfig::default(),
            chunking: ChunkingConfig::default(),
            adapters: None,
            sources: None,
        }
    }
}

impl CorviaConfig {
    /// Returns true if this config defines a multi-repo workspace.
    pub fn is_workspace(&self) -> bool {
        self.workspace.is_some()
    }

    /// Return a config preset for the full SurrealDB + vLLM stack.
    pub fn full_default() -> Self {
        Self {
            storage: StorageConfig {
                store_type: StoreType::Surrealdb,
                data_dir: ".corvia".into(),
                surrealdb_url: Some("127.0.0.1:8000".into()),
                surrealdb_ns: Some("corvia".into()),
                surrealdb_db: Some("main".into()),
                surrealdb_user: Some("root".into()),
                surrealdb_pass: Some("root".into()),
                postgres_url: None,
            },
            embedding: EmbeddingConfig {
                provider: InferenceProvider::Vllm,
                model: "nomic-ai/nomic-embed-text-v1.5".into(),
                url: "http://127.0.0.1:8001".into(),
                dimensions: 768,
            },
            ..Self::default()
        }
    }

    /// Return a config preset for PostgreSQL + vLLM.
    pub fn postgres_default() -> Self {
        Self {
            storage: StorageConfig {
                store_type: StoreType::Postgres,
                data_dir: ".corvia".into(),
                surrealdb_url: None,
                surrealdb_ns: None,
                surrealdb_db: None,
                surrealdb_user: None,
                surrealdb_pass: None,
                postgres_url: Some("postgres://corvia:corvia@127.0.0.1:5432/corvia".into()),
            },
            embedding: EmbeddingConfig {
                provider: InferenceProvider::Vllm,
                model: "nomic-ai/nomic-embed-text-v1.5".into(),
                url: "http://127.0.0.1:8001".into(),
                dimensions: 768,
            },
            ..Self::default()
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CorviaError::Config(format!("Failed to read {}: {}", path.display(), e)))?;
        toml::from_str(&content)
            .map_err(|e| CorviaError::Config(format!("Failed to parse config: {e}")))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| CorviaError::Config(format!("Failed to serialize config: {e}")))?;
        std::fs::write(path, content)
            .map_err(|e| CorviaError::Config(format!("Failed to write {}: {}", path.display(), e)))?;
        Ok(())
    }

    pub fn config_path() -> PathBuf {
        PathBuf::from("corvia.toml")
    }

    /// Apply environment variable overrides. Checks CORVIA_* vars
    /// and overrides the corresponding config fields if set.
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = std::env::var("CORVIA_SURREALDB_URL") {
            self.storage.surrealdb_url = Some(val);
        }
        if let Ok(val) = std::env::var("CORVIA_VLLM_URL") {
            self.embedding.url = val;
        }
        if let Ok(val) = std::env::var("CORVIA_OLLAMA_URL") {
            self.embedding.url = val;
        }
        if let Ok(val) = std::env::var("CORVIA_TEST_NAMESPACE") {
            self.storage.surrealdb_ns = Some(val);
        }
        if let Ok(val) = std::env::var("CORVIA_POSTGRES_URL") {
            self.storage.postgres_url = Some(val);
        }
        if let Ok(val) = std::env::var("CORVIA_STORE_TYPE") {
            match val.as_str() {
                "lite" => self.storage.store_type = StoreType::Lite,
                "surrealdb" => self.storage.store_type = StoreType::Surrealdb,
                "postgres" => self.storage.store_type = StoreType::Postgres,
                _ => {}
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CorviaConfig::default();
        assert_eq!(config.storage.store_type, StoreType::Lite);
        assert_eq!(config.embedding.provider, InferenceProvider::Ollama);
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.server.port, 8020);
    }

    #[test]
    fn test_lite_config_defaults() {
        let config = CorviaConfig::default();
        assert_eq!(config.storage.store_type, StoreType::Lite);
        assert_eq!(config.embedding.provider, InferenceProvider::Ollama);
        assert_eq!(config.storage.data_dir, ".corvia");
        assert_eq!(config.embedding.url, "http://127.0.0.1:11434");
        assert_eq!(config.embedding.model, "nomic-embed-text");
        assert!(config.storage.surrealdb_url.is_none());
    }

    #[test]
    fn test_full_config_defaults() {
        let config = CorviaConfig::full_default();
        assert_eq!(config.storage.store_type, StoreType::Surrealdb);
        assert_eq!(config.embedding.provider, InferenceProvider::Vllm);
        assert_eq!(config.storage.surrealdb_url, Some("127.0.0.1:8000".into()));
        assert_eq!(config.embedding.url, "http://127.0.0.1:8001");
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let config = CorviaConfig::default();
        config.save(std::path::Path::new("/tmp/corvia-test-config.toml")).unwrap();
        let loaded = CorviaConfig::load(std::path::Path::new("/tmp/corvia-test-config.toml")).unwrap();
        assert_eq!(config.project.name, loaded.project.name);
        assert_eq!(config.storage.store_type, loaded.storage.store_type);
    }

    #[test]
    fn test_full_config_roundtrip() {
        let config = CorviaConfig::full_default();
        config.save(std::path::Path::new("/tmp/corvia-test-full.toml")).unwrap();
        let loaded = CorviaConfig::load(std::path::Path::new("/tmp/corvia-test-full.toml")).unwrap();
        assert_eq!(loaded.storage.store_type, StoreType::Surrealdb);
        assert_eq!(loaded.storage.surrealdb_url, Some("127.0.0.1:8000".into()));
    }

    #[test]
    fn test_store_type_serde() {
        let json = serde_json::to_string(&StoreType::Lite).unwrap();
        assert_eq!(json, "\"lite\"");
        let parsed: StoreType = serde_json::from_str("\"surrealdb\"").unwrap();
        assert_eq!(parsed, StoreType::Surrealdb);
    }

    #[test]
    fn test_env_override_surrealdb_url() {
        unsafe { std::env::set_var("CORVIA_SURREALDB_URL", "ws://ci-host:9000"); }
        let config = CorviaConfig::default().with_env_overrides();
        assert_eq!(config.storage.surrealdb_url, Some("ws://ci-host:9000".into()));
        unsafe { std::env::remove_var("CORVIA_SURREALDB_URL"); }
    }

    #[test]
    fn test_env_override_vllm_url() {
        unsafe { std::env::set_var("CORVIA_VLLM_URL", "http://ci-host:9001"); }
        let config = CorviaConfig::default().with_env_overrides();
        assert_eq!(config.embedding.url, "http://ci-host:9001");
        unsafe { std::env::remove_var("CORVIA_VLLM_URL"); }
    }

    #[test]
    fn test_no_env_override_preserves_defaults() {
        unsafe {
            std::env::remove_var("CORVIA_SURREALDB_URL");
            std::env::remove_var("CORVIA_VLLM_URL");
            std::env::remove_var("CORVIA_OLLAMA_URL");
            std::env::remove_var("CORVIA_TEST_NAMESPACE");
            std::env::remove_var("CORVIA_STORE_TYPE");
        }
        let config = CorviaConfig::default().with_env_overrides();
        assert!(config.storage.surrealdb_url.is_none());
        assert_eq!(config.embedding.url, "http://127.0.0.1:11434");
    }

    #[test]
    fn test_workspace_config_none_by_default() {
        let config = CorviaConfig::default();
        assert!(config.workspace.is_none());
    }

    #[test]
    fn test_workspace_config_from_toml() {
        let toml_str = r#"
[project]
name = "test-workspace"
scope_id = "test"

[workspace]
repos_dir = "repos"

[[workspace.repos]]
name = "backend"
url = "https://github.com/org/backend"
namespace = "backend"

[[workspace.repos]]
name = "frontend"
url = "https://github.com/org/frontend"
namespace = "frontend"
local = "/home/dev/frontend"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let ws = config.workspace.as_ref().unwrap();
        assert_eq!(ws.repos_dir, "repos");
        assert_eq!(ws.repos.len(), 2);
        assert_eq!(ws.repos[0].name, "backend");
        assert_eq!(ws.repos[0].url, "https://github.com/org/backend");
        assert_eq!(ws.repos[0].namespace, "backend");
        assert!(ws.repos[0].local.is_none());
        assert_eq!(ws.repos[1].name, "frontend");
        assert_eq!(ws.repos[1].local.as_deref(), Some("/home/dev/frontend"));
    }

    #[test]
    fn test_workspace_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.workspace = Some(WorkspaceConfig {
            repos_dir: "repos".into(),
            repos: vec![
                RepoConfig {
                    name: "my-repo".into(),
                    url: "https://github.com/org/my-repo".into(),
                    local: None,
                    namespace: "main".into(),
                },
            ],
        });
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        let ws = loaded.workspace.unwrap();
        assert_eq!(ws.repos.len(), 1);
        assert_eq!(ws.repos[0].name, "my-repo");
    }

    #[test]
    fn test_is_workspace() {
        let config = CorviaConfig::default();
        assert!(!config.is_workspace());

        let mut ws_config = CorviaConfig::default();
        ws_config.workspace = Some(WorkspaceConfig {
            repos_dir: "repos".into(),
            repos: vec![],
        });
        assert!(ws_config.is_workspace());
    }

    #[test]
    fn test_workspace_default_repos_dir() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[workspace]

[[workspace.repos]]
name = "repo"
url = "https://github.com/org/repo"
namespace = "repo"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let ws = config.workspace.unwrap();
        assert_eq!(ws.repos_dir, "repos");
    }

    #[test]
    fn test_reasoning_config_none_by_default() {
        let config = CorviaConfig::default();
        assert!(config.reasoning.is_none());
    }

    #[test]
    fn test_reasoning_config_from_toml() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020

[reasoning]
provider = "ollama"
model = "llama3.2:3b"
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let reasoning = config.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.provider, InferenceProvider::Ollama);
        assert_eq!(reasoning.model, "llama3.2:3b");
    }

    #[test]
    fn test_reasoning_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.reasoning = Some(ReasoningConfig {
            provider: InferenceProvider::Ollama,
            model: "llama3.2:3b".into(),
        });
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        let reasoning = loaded.reasoning.unwrap();
        assert_eq!(reasoning.provider, InferenceProvider::Ollama);
        assert_eq!(reasoning.model, "llama3.2:3b");
    }

    #[test]
    fn test_reasoning_config_omitted_still_parses() {
        // Ensure existing configs without [reasoning] still parse correctly
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert!(config.reasoning.is_none(), "missing [reasoning] section should parse as None");
    }

    #[test]
    fn test_rag_config_defaults() {
        let config = RagConfig::default();
        assert_eq!(config.default_limit, 10);
        assert!(config.graph_expand);
        assert_eq!(config.graph_depth, 2);
        assert!((config.graph_alpha - 0.3).abs() < f32::EPSILON);
        assert!((config.reserve_for_answer - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.max_context_tokens, 0);
        assert!(config.system_prompt.is_empty());
    }

    #[test]
    fn test_chunking_config_defaults() {
        let config = ChunkingConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.min_tokens, 32);
        assert_eq!(config.overlap_tokens, 64);
        assert_eq!(config.strategy, "auto");
    }

    #[test]
    fn test_rag_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.rag.default_limit = 20;
        config.rag.graph_alpha = 0.5;
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.rag.default_limit, 20);
        assert!((loaded.rag.graph_alpha - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chunking_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.chunking.max_tokens = 1024;
        config.chunking.overlap_tokens = 128;
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.chunking.max_tokens, 1024);
        assert_eq!(loaded.chunking.overlap_tokens, 128);
    }

    #[test]
    fn test_inference_provider_corvia_serde() {
        let json = serde_json::to_string(&InferenceProvider::Corvia).unwrap();
        assert_eq!(json, "\"corvia\"");
        let parsed: InferenceProvider = serde_json::from_str("\"corvia\"").unwrap();
        assert_eq!(parsed, InferenceProvider::Corvia);
    }

    #[test]
    fn test_adapters_config_from_toml() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
data_dir = ".corvia"

[embedding]
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020

[adapters]
search_dirs = ["~/.config/corvia/adapters", "/opt/corvia/adapters"]
default = "git"

[[sources]]
path = "./backend"
adapter = "git"

[[sources]]
path = "https://company.atlassian.net/wiki/spaces/ENG"
adapter = "confluence"
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let adapters = config.adapters.unwrap();
        assert_eq!(adapters.search_dirs.len(), 2);
        assert_eq!(adapters.default, Some("git".into()));

        let sources = config.sources.unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].path, "./backend");
        assert_eq!(sources[0].adapter, Some("git".into()));
        assert_eq!(sources[1].adapter, Some("confluence".into()));
    }

    #[test]
    fn test_adapters_config_optional() {
        let config = CorviaConfig::default();
        assert!(config.adapters.is_none());
        assert!(config.sources.is_none());
    }

    #[test]
    fn test_adapters_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.adapters = Some(AdaptersConfig {
            search_dirs: vec!["~/adapters".into()],
            default: Some("basic".into()),
        });
        config.sources = Some(vec![SourceConfig {
            path: "./repo".into(),
            adapter: Some("git".into()),
            adapter_config: None,
        }]);

        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.adapters.unwrap().default, Some("basic".into()));
        assert_eq!(loaded.sources.unwrap().len(), 1);
    }

    #[test]
    fn test_oversample_config_default() {
        let config = RagConfig::default();
        assert_eq!(config.graph_oversample_factor, 3);
    }

    #[test]
    fn test_oversample_config_serde() {
        // Deserialize TOML without graph_oversample_factor → should get default 3
        let toml_str = r#"
default_limit = 10
graph_expand = true
graph_depth = 2
graph_alpha = 0.3
reserve_for_answer = 0.2
max_context_tokens = 0
system_prompt = ""
"#;
        let config: RagConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.graph_oversample_factor, 3);

        // With explicit value
        let toml_str2 = r#"
default_limit = 10
graph_expand = true
graph_depth = 2
graph_alpha = 0.3
reserve_for_answer = 0.2
max_context_tokens = 0
system_prompt = ""
graph_oversample_factor = 5
"#;
        let config2: RagConfig = toml::from_str(toml_str2).unwrap();
        assert_eq!(config2.graph_oversample_factor, 5);
    }

    #[test]
    fn test_existing_config_without_rag_or_chunking_still_parses() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rag.default_limit, 10);
        assert_eq!(config.chunking.max_tokens, 512);
    }
}
