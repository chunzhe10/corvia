use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::errors::{CorviaError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StoreType {
    #[default]
    Lite,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DocsConfig {
    #[serde(default)]
    pub memory_dir: Option<String>,
    #[serde(default)]
    pub workspace_docs: Option<String>,
    #[serde(default)]
    pub allowed_workspace_subdirs: Vec<String>,
    #[serde(default)]
    pub rules: Option<DocsRulesConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DocsRulesConfig {
    #[serde(default)]
    pub blocked_paths: Vec<String>,
    #[serde(default)]
    pub repo_docs_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceConfig {
    #[serde(default = "default_repos_dir")]
    pub repos_dir: String,
    #[serde(default)]
    pub repos: Vec<RepoConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docs: Option<DocsConfig>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<MergeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<WorkspaceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub rag: RagConfig,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapters: Option<AdaptersConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SourceConfig>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Vec<ScopeConfig>>,
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
    pub postgres_url: Option<String>,
}

fn default_data_dir() -> String {
    ".corvia".into()
}

fn default_device() -> String { "auto".into() }
fn default_kv_quant() -> String { "q8".into() }
fn default_flash_attention() -> bool { true }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatModelDef {
    /// HuggingFace repository (e.g. "bartowski/Qwen_Qwen3-8B-GGUF").
    pub repo: String,
    /// GGUF filename (e.g. "Qwen3-8B-Q4_K_M.gguf").
    pub filename: String,
}

/// Default chat model registry. Keep in sync with `resolve_model()` in
/// `corvia-inference/src/chat_service.rs` (the hardcoded fallback).
fn default_chat_models() -> std::collections::HashMap<String, ChatModelDef> {
    let mut m = std::collections::HashMap::new();
    m.insert("qwen3".into(), ChatModelDef {
        repo: "bartowski/Qwen_Qwen3-8B-GGUF".into(),
        filename: "Qwen3-8B-Q4_K_M.gguf".into(),
    });
    m.insert("qwen3:8b".into(), ChatModelDef {
        repo: "bartowski/Qwen_Qwen3-8B-GGUF".into(),
        filename: "Qwen3-8B-Q4_K_M.gguf".into(),
    });
    m.insert("llama3.2".into(), ChatModelDef {
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".into(),
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
    });
    m.insert("llama3.2:3b".into(), ChatModelDef {
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".into(),
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
    });
    m.insert("llama3.2:1b".into(), ChatModelDef {
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF".into(),
        filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf".into(),
    });
    m
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Device preference: "auto" (default), "gpu", or "cpu".
    #[serde(default = "default_device")]
    pub device: String,
    /// Backend override for chat models: "cuda", "openvino", or "" (auto-select).
    /// For embedding-specific override, use `embedding_backend`.
    #[serde(default)]
    pub backend: String,
    /// Backend override specifically for embedding models: "openvino", "cuda", or "" (use `backend`).
    /// Allows ONNX embedding on Intel iGPU (OpenVINO) while chat runs on NVIDIA (CUDA).
    /// Falls back to `backend` when empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub embedding_backend: String,
    /// KV cache quantization: "q8" (default), "q4", "none".
    #[serde(default = "default_kv_quant")]
    pub kv_quant: String,
    /// Enable flash attention (default: true).
    #[serde(default = "default_flash_attention")]
    pub flash_attention: bool,
    /// Chat model registry: short name → HF repo + GGUF filename.
    #[serde(default = "default_chat_models")]
    pub chat_models: std::collections::HashMap<String, ChatModelDef>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            backend: String::new(),
            embedding_backend: String::new(),
            kv_quant: default_kv_quant(),
            flash_attention: default_flash_attention(),
            chat_models: default_chat_models(),
        }
    }
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
    /// Retriever strategy name. Built-in: "vector", "graph_expand".
    /// Default: "graph_expand" (falls back to "vector" if no graph store).
    #[serde(default = "default_retriever")]
    pub retriever: String,
    /// Augmenter strategy name. Built-in: "structured".
    #[serde(default = "default_augmenter")]
    pub augmenter: String,
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
    /// Master toggle for the dynamic skill system. When false, no skills are
    /// loaded or matched — identical to pre-skill behavior. Default: false.
    #[serde(default)]
    pub skills_enabled: bool,
    /// Directories to load skill files from. Globs `*.md` in each directory.
    /// Later directories override same-named skills. Default: `["skills"]`.
    #[serde(default = "default_skills_dirs")]
    pub skills_dirs: Vec<String>,
    /// Maximum number of skills injected per query. Default: 3.
    #[serde(default = "default_max_skills")]
    pub max_skills: usize,
    /// Minimum cosine similarity between query and skill description to select. Default: 0.3.
    #[serde(default = "default_skill_threshold")]
    pub skill_threshold: f32,
    /// Fraction of context window reserved for skill content. Default: 0.15.
    #[serde(default = "default_reserve_for_skills")]
    pub reserve_for_skills: f32,
}

fn default_retriever() -> String { "graph_expand".into() }
fn default_augmenter() -> String { "structured".into() }
fn default_rag_limit() -> usize { 10 }
fn default_graph_expand() -> bool { true }
fn default_graph_depth() -> usize { 2 }
fn default_graph_alpha() -> f32 { 0.3 }
fn default_reserve_for_answer() -> f32 { 0.2 }
fn default_graph_oversample() -> usize { 3 }
fn default_skills_dirs() -> Vec<String> { vec!["skills".into()] }
fn default_max_skills() -> usize { 3 }
fn default_skill_threshold() -> f32 { 0.3 }
fn default_reserve_for_skills() -> f32 { 0.15 }

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            retriever: default_retriever(),
            augmenter: default_augmenter(),
            default_limit: default_rag_limit(),
            graph_expand: default_graph_expand(),
            graph_depth: default_graph_depth(),
            graph_alpha: default_graph_alpha(),
            reserve_for_answer: default_reserve_for_answer(),
            max_context_tokens: 0,
            system_prompt: String::new(),
            graph_oversample_factor: default_graph_oversample(),
            skills_enabled: false,
            skills_dirs: default_skills_dirs(),
            max_skills: default_max_skills(),
            skill_threshold: default_skill_threshold(),
            reserve_for_skills: default_reserve_for_skills(),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_telemetry_exporter")]
    pub exporter: String,
    #[serde(default)]
    pub otlp_endpoint: String,
    #[serde(default = "default_telemetry_otlp_protocol")]
    pub otlp_protocol: String,
    #[serde(default = "default_telemetry_service_name")]
    pub service_name: String,
    #[serde(default = "default_telemetry_log_format")]
    pub log_format: String,
    #[serde(default = "default_telemetry_metrics_enabled")]
    pub metrics_enabled: bool,
}

fn default_telemetry_exporter() -> String { "stdout".into() }
fn default_telemetry_otlp_protocol() -> String { "grpc".into() }
fn default_telemetry_service_name() -> String { "corvia".into() }
fn default_telemetry_log_format() -> String { "text".into() }
fn default_telemetry_metrics_enabled() -> bool { true }

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            exporter: default_telemetry_exporter(),
            otlp_endpoint: String::new(),
            otlp_protocol: default_telemetry_otlp_protocol(),
            service_name: default_telemetry_service_name(),
            log_format: default_telemetry_log_format(),
            metrics_enabled: default_telemetry_metrics_enabled(),
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

/// Per-scope configuration (e.g. `[[scope]]` in corvia.toml).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeConfig {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_days: Option<u32>,
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
            merge: None,
            workspace: None,
            reasoning: None,
            rag: RagConfig::default(),
            chunking: ChunkingConfig::default(),
            telemetry: TelemetryConfig::default(),
            inference: InferenceConfig::default(),
            adapters: None,
            sources: None,
            scope: None,
        }
    }
}

impl CorviaConfig {
    /// Returns true if this config defines a multi-repo workspace.
    pub fn is_workspace(&self) -> bool {
        self.workspace.is_some()
    }

    /// Return a config preset for PostgreSQL + vLLM.
    pub fn postgres_default() -> Self {
        Self {
            storage: StorageConfig {
                store_type: StoreType::Postgres,
                data_dir: ".corvia".into(),
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
        if let Ok(val) = std::env::var("CORVIA_VLLM_URL") {
            self.embedding.url = val;
        }
        if let Ok(val) = std::env::var("CORVIA_OLLAMA_URL") {
            self.embedding.url = val;
        }
        if let Ok(val) = std::env::var("CORVIA_POSTGRES_URL") {
            self.storage.postgres_url = Some(val);
        }
        if let Ok(val) = std::env::var("CORVIA_STORE_TYPE") {
            match val.as_str() {
                "lite" => self.storage.store_type = StoreType::Lite,
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
    fn test_store_type_serde() {
        let json = serde_json::to_string(&StoreType::Lite).unwrap();
        assert_eq!(json, "\"lite\"");
        let parsed: StoreType = serde_json::from_str("\"postgres\"").unwrap();
        assert_eq!(parsed, StoreType::Postgres);
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
            std::env::remove_var("CORVIA_VLLM_URL");
            std::env::remove_var("CORVIA_OLLAMA_URL");
            std::env::remove_var("CORVIA_STORE_TYPE");
        }
        let config = CorviaConfig::default().with_env_overrides();
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
            docs: None,
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
            docs: None,
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
    fn test_telemetry_config_new_fields() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "corvia");
        assert_eq!(config.otlp_protocol, "grpc");
    }

    #[test]
    fn test_telemetry_config_deserialize_new_fields() {
        let toml_str = r#"
            exporter = "otlp"
            otlp_endpoint = "http://localhost:4317"
            otlp_protocol = "grpc"
            service_name = "corvia-inference"
            log_format = "json"
            metrics_enabled = true
        "#;
        let config: TelemetryConfig = toml::de::from_str(toml_str).unwrap();
        assert_eq!(config.service_name, "corvia-inference");
        assert_eq!(config.otlp_protocol, "grpc");
        assert_eq!(config.otlp_endpoint, "http://localhost:4317");
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

    #[test]
    fn test_inference_config_defaults() {
        let config = InferenceConfig::default();
        assert_eq!(config.device, "auto");
        assert!(config.backend.is_empty());
        assert_eq!(config.kv_quant, "q8");
        assert!(config.flash_attention);
    }

    #[test]
    fn test_corvia_config_has_inference_defaults() {
        let config = CorviaConfig::default();
        assert_eq!(config.inference.device, "auto");
        assert_eq!(config.inference.kv_quant, "q8");
        assert!(config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_from_toml() {
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

[inference]
device = "gpu"
backend = "cuda"
kv_quant = "q4"
flash_attention = false
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.inference.device, "gpu");
        assert_eq!(config.inference.backend, "cuda");
        assert_eq!(config.inference.kv_quant, "q4");
        assert!(!config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_omitted_still_parses() {
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
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.inference.device, "auto");
        assert_eq!(config.inference.kv_quant, "q8");
        assert!(config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.inference.kv_quant = "q4".into();
        config.inference.flash_attention = false;
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.inference.kv_quant, "q4");
        assert!(!loaded.inference.flash_attention);
    }

    #[test]
    fn test_chat_models_config_defaults_and_roundtrip() {
        let config = InferenceConfig::default();
        assert!(config.chat_models.contains_key("qwen3"));
        assert!(config.chat_models.contains_key("llama3.2"));
        assert!(config.chat_models.contains_key("llama3.2:1b"));
        assert_eq!(config.chat_models["qwen3"].repo, "bartowski/Qwen_Qwen3-8B-GGUF");

        // Roundtrip through TOML
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: InferenceConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.chat_models.len(), config.chat_models.len());
        assert_eq!(loaded.chat_models["qwen3"], config.chat_models["qwen3"]);
    }

    #[test]
    fn test_chat_models_config_from_toml() {
        let toml_str = r#"
[chat_models.custom-model]
repo = "user/Custom-Model-GGUF"
filename = "Custom-Model-Q4_K_M.gguf"
"#;
        let loaded: InferenceConfig = toml::from_str(toml_str).unwrap();
        assert!(loaded.chat_models.contains_key("custom-model"));
        assert_eq!(loaded.chat_models["custom-model"].repo, "user/Custom-Model-GGUF");
        // Explicit config replaces defaults — only the custom model should be present
        assert!(!loaded.chat_models.contains_key("qwen3"));
    }

    #[test]
    fn test_docs_config_from_toml() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[workspace]
repos_dir = "repos"

[workspace.docs]
memory_dir = ".memory"
workspace_docs = "docs"
allowed_workspace_subdirs = ["decisions", "learnings"]

[workspace.docs.rules]
blocked_paths = ["docs/superpowers/*"]
repo_docs_pattern = "docs/"

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
        let docs = ws.docs.as_ref().unwrap();
        assert_eq!(docs.memory_dir, Some(".memory".into()));
        assert_eq!(docs.workspace_docs, Some("docs".into()));
        assert_eq!(docs.allowed_workspace_subdirs, vec!["decisions", "learnings"]);
        let rules = docs.rules.as_ref().unwrap();
        assert_eq!(rules.blocked_paths, vec!["docs/superpowers/*"]);
        assert_eq!(rules.repo_docs_pattern, Some("docs/".into()));
    }

    #[test]
    fn test_docs_config_optional() {
        // Existing workspace configs without [workspace.docs] still parse
        let config = CorviaConfig::default();
        assert!(config.workspace.is_none());

        let mut ws_config = CorviaConfig::default();
        ws_config.workspace = Some(WorkspaceConfig {
            repos_dir: "repos".into(),
            repos: vec![],
            docs: None,
        });
        let toml_str = toml::to_string_pretty(&ws_config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert!(loaded.workspace.unwrap().docs.is_none());
    }
}
