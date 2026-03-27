use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::errors::{CorviaError, Result};
use crate::types::MemoryType;

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

fn default_true() -> bool { true }

/// Configuration for Claude Code lifecycle hooks.
/// All hooks default to enabled; set to `false` to disable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HooksConfig {
    /// Master switch — set to false to disable all hooks at once.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub session_recording: bool,
    #[serde(default = "default_true")]
    pub doc_placement: bool,
    #[serde(default = "default_true")]
    pub agent_check: bool,
    #[serde(default = "default_true")]
    pub write_reminder: bool,
    #[serde(default = "default_true")]
    pub orphan_cleanup: bool,
    #[serde(default = "default_true")]
    pub corvia_first_reminder: bool,
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            session_recording: true,
            doc_placement: true,
            agent_check: true,
            write_reminder: true,
            orphan_cleanup: true,
            corvia_first_reminder: true,
        }
    }
}

impl HooksConfig {
    /// Check if a specific hook is active (enabled globally AND individually).
    pub fn is_active(&self, hook: &str) -> bool {
        if !self.enabled { return false; }
        match hook {
            "session_recording" => self.session_recording,
            "doc_placement" => self.doc_placement,
            "agent_check" => self.agent_check,
            "write_reminder" => self.write_reminder,
            "orphan_cleanup" => self.orphan_cleanup,
            "corvia_first_reminder" => self.corvia_first_reminder,
            _ => true,
        }
    }
}

fn default_stale_threshold() -> f64 { 0.9 }
fn default_coverage_ttl() -> u64 { 60 }
const MIN_COVERAGE_TTL: u64 = 5;

/// Dashboard configuration for index coverage monitoring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardSection {
    /// Coverage below this ratio triggers `index_stale: true`. Range: 0.0..=1.0.
    #[serde(default = "default_stale_threshold")]
    pub stale_threshold: f64,
    /// TTL in seconds for cached coverage computation. Minimum: 5.
    #[serde(default = "default_coverage_ttl")]
    pub coverage_ttl_secs: u64,
}

impl Default for DashboardSection {
    fn default() -> Self {
        Self {
            stale_threshold: 0.9,
            coverage_ttl_secs: 60,
        }
    }
}

impl DashboardSection {
    /// Validate and clamp values. Call after deserialization.
    pub fn validate(&mut self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.stale_threshold) {
            return Err(CorviaError::Config(format!(
                "dashboard.stale_threshold must be 0.0..=1.0, got {}",
                self.stale_threshold
            )));
        }
        if self.coverage_ttl_secs < MIN_COVERAGE_TTL {
            eprintln!(
                "warning: dashboard.coverage_ttl_secs={} below minimum {}, clamping",
                self.coverage_ttl_secs, MIN_COVERAGE_TTL
            );
            self.coverage_ttl_secs = MIN_COVERAGE_TTL;
        }
        Ok(())
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forgetting: Option<ForgettingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hooks: Option<HooksConfig>,
    #[serde(default)]
    pub dashboard: DashboardSection,
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
fn default_health_probe_interval_secs() -> u64 { 60 }
fn default_health_probe_drift_threshold_pct() -> f64 { 100.0 }

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
    /// Interval in seconds between inference health probes.
    #[serde(default = "default_health_probe_interval_secs")]
    pub health_probe_interval_secs: u64,
    /// Drift threshold percentage before marking inference as degraded.
    #[serde(default = "default_health_probe_drift_threshold_pct")]
    pub health_probe_drift_threshold_pct: f64,
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
            health_probe_interval_secs: default_health_probe_interval_secs(),
            health_probe_drift_threshold_pct: default_health_probe_drift_threshold_pct(),
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

// ── Forgetting policy configuration ─────────────────────────────────────────

fn default_interval_minutes() -> u32 { 60 }
fn default_max_inactive_days() -> u32 { 90 }
fn default_budget_top_n() -> u32 { 10_000 }

/// Global defaults for forgetting policies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingPolicyConfig {
    /// Force entry to Cold if inactive longer than this AND retention_score < 0.60.
    /// Must be > 0.
    #[serde(default = "default_max_inactive_days")]
    pub max_inactive_days: u32,
    /// Per-scope cap on active entries (Hot + Warm). 0 = no limit.
    /// Pinned entries excluded from count.
    #[serde(default = "default_budget_top_n")]
    pub budget_top_n: u32,
}

impl Default for ForgettingPolicyConfig {
    fn default() -> Self {
        Self {
            max_inactive_days: default_max_inactive_days(),
            budget_top_n: default_budget_top_n(),
        }
    }
}

/// Override layer for forgetting policy. All fields optional — `None` inherits
/// from the next-lower layer in the hierarchy. Used for both per-memory-type
/// and per-scope overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ForgettingOverride {
    /// Override enabled/disabled. `None` = inherit from parent layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Override max inactive days. `None` = inherit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_inactive_days: Option<u32>,
    /// Override budget cap. `None` = inherit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_top_n: Option<u32>,
}

/// Per-memory-type override alias (same fields, distinct name for clarity).
pub type PerTypeForgettingConfig = ForgettingOverride;

/// Per-scope forgetting override alias.
pub type ScopeForgettingOverride = ForgettingOverride;

/// Top-level `[forgetting]` section in corvia.toml.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingConfig {
    /// Master switch. When false, all forgetting is disabled.
    #[serde(default)]
    pub enabled: bool,
    /// How often the GC worker runs (minutes). Must be > 0.
    #[serde(default = "default_interval_minutes")]
    pub interval_minutes: u32,
    /// Global default policy values.
    #[serde(default)]
    pub defaults: ForgettingPolicyConfig,
    /// Per-memory-type overrides. Keys are MemoryType variants (snake_case).
    #[serde(default)]
    pub by_type: HashMap<MemoryType, PerTypeForgettingConfig>,
}

impl Default for ForgettingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_minutes: default_interval_minutes(),
            defaults: ForgettingPolicyConfig::default(),
            by_type: HashMap::new(),
        }
    }
}

/// Fully resolved forgetting policy for a specific (scope, memory_type) pair.
/// All values are concrete — no `Option`s.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPolicy {
    pub enabled: bool,
    pub max_inactive_days: u32,
    pub budget_top_n: u32,
}

impl ForgettingConfig {
    /// Merge the 3-level hierarchy: global defaults → per-type → per-scope.
    ///
    /// Resolution order (most specific wins per field):
    /// 1. Start with global `defaults`
    /// 2. If global `enabled == false`, short-circuit to disabled
    /// 3. Layer per-type override (non-None fields replace)
    /// 4. If per-type `enabled == Some(false)`, short-circuit to disabled.
    ///    **Note:** This is a hard disable — scope overrides cannot re-enable a
    ///    type disabled at the per-type level. This allows admins to guarantee
    ///    certain memory types are never subject to forgetting.
    /// 5. Layer per-scope override (non-None fields replace)
    /// 6. Return merged `ResolvedPolicy`
    pub fn resolve_policy(
        &self,
        memory_type: MemoryType,
        scope_override: Option<&ScopeForgettingOverride>,
    ) -> ResolvedPolicy {
        // Global disabled → everything off
        if !self.enabled {
            return ResolvedPolicy {
                enabled: false,
                max_inactive_days: self.defaults.max_inactive_days,
                budget_top_n: self.defaults.budget_top_n,
            };
        }

        let mut enabled = true;
        let mut max_inactive_days = self.defaults.max_inactive_days;
        let mut budget_top_n = self.defaults.budget_top_n;

        // Layer 2: per-type override
        if let Some(type_cfg) = self.by_type.get(&memory_type) {
            if let Some(e) = type_cfg.enabled {
                enabled = e;
            }
            if !enabled {
                return ResolvedPolicy { enabled, max_inactive_days, budget_top_n };
            }
            if let Some(d) = type_cfg.max_inactive_days {
                max_inactive_days = d;
            }
            if let Some(b) = type_cfg.budget_top_n {
                budget_top_n = b;
            }
        }

        // Layer 3: per-scope override (most specific)
        if let Some(scope_cfg) = scope_override {
            if let Some(e) = scope_cfg.enabled {
                enabled = e;
            }
            if let Some(d) = scope_cfg.max_inactive_days {
                max_inactive_days = d;
            }
            if let Some(b) = scope_cfg.budget_top_n {
                budget_top_n = b;
            }
        }

        ResolvedPolicy { enabled, max_inactive_days, budget_top_n }
    }

    /// Validate configuration values. Call after deserialization.
    pub fn validate(&self) -> Result<()> {
        if self.interval_minutes == 0 {
            return Err(CorviaError::Config(
                "forgetting.interval_minutes must be > 0".into(),
            ));
        }
        if self.defaults.max_inactive_days == 0 {
            return Err(CorviaError::Config(
                "forgetting.defaults.max_inactive_days must be > 0".into(),
            ));
        }
        for (memory_type, cfg) in &self.by_type {
            if let Some(d) = cfg.max_inactive_days {
                if d == 0 {
                    return Err(CorviaError::Config(format!(
                        "forgetting.by_type.{memory_type}.max_inactive_days must be > 0"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Per-scope configuration (e.g. `[[scope]]` in corvia.toml).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeConfig {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_days: Option<u32>,
    /// Per-scope forgetting policy overrides. Most specific layer in the hierarchy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forgetting: Option<ScopeForgettingOverride>,
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
            forgetting: None,
            hooks: None,
            dashboard: DashboardSection::default(),
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
        let config: Self = toml::from_str(&content)
            .map_err(|e| CorviaError::Config(format!("Failed to parse config: {e}")))?;
        if let Some(ref fg) = config.forgetting {
            fg.validate()?;
        }
        // Validate scope-level forgetting overrides
        if let Some(ref scopes) = config.scope {
            for scope in scopes {
                if let Some(ref override_cfg) = scope.forgetting {
                    if let Some(d) = override_cfg.max_inactive_days {
                        if d == 0 {
                            return Err(CorviaError::Config(format!(
                                "scope.{}.forgetting.max_inactive_days must be > 0",
                                scope.id
                            )));
                        }
                    }
                }
            }
        }
        Ok(config)
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

    #[test]
    fn test_dashboard_defaults() {
        let config = CorviaConfig::default();
        assert_eq!(config.dashboard.stale_threshold, 0.9);
        assert_eq!(config.dashboard.coverage_ttl_secs, 60);
    }

    #[test]
    fn test_dashboard_partial_override() {
        // Serialize default, modify dashboard section, deserialize
        let config = CorviaConfig::default();
        let mut toml_str = toml::to_string_pretty(&config).unwrap();
        // Replace the default threshold in the serialized TOML
        toml_str = toml_str.replace("stale_threshold = 0.9", "stale_threshold = 0.5");
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.dashboard.stale_threshold, 0.5);
        assert_eq!(loaded.dashboard.coverage_ttl_secs, 60);
    }

    #[test]
    fn test_dashboard_threshold_out_of_range_high() {
        let mut section = DashboardSection { stale_threshold: 1.5, coverage_ttl_secs: 60 };
        assert!(section.validate().is_err());
    }

    #[test]
    fn test_dashboard_threshold_out_of_range_negative() {
        let mut section = DashboardSection { stale_threshold: -0.1, coverage_ttl_secs: 60 };
        assert!(section.validate().is_err());
    }

    #[test]
    fn test_dashboard_ttl_below_minimum() {
        let mut section = DashboardSection { stale_threshold: 0.9, coverage_ttl_secs: 1 };
        section.validate().unwrap();
        assert_eq!(section.coverage_ttl_secs, 5);
    }

    // ── Forgetting policy config tests ──────────────────────────────────

    #[test]
    fn test_forgetting_defaults() {
        let cfg = ForgettingConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.interval_minutes, 60);
        assert_eq!(cfg.defaults.max_inactive_days, 90);
        assert_eq!(cfg.defaults.budget_top_n, 10_000);
        assert!(cfg.by_type.is_empty());
    }

    #[test]
    fn test_forgetting_global_defaults_apply() {
        let cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        let resolved = cfg.resolve_policy(MemoryType::Episodic, None);
        assert!(resolved.enabled);
        assert_eq!(resolved.max_inactive_days, 90);
        assert_eq!(resolved.budget_top_n, 10_000);
    }

    #[test]
    fn test_forgetting_per_type_override_wins_over_global() {
        let mut cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        cfg.by_type.insert(MemoryType::Episodic, PerTypeForgettingConfig {
            enabled: None,
            max_inactive_days: Some(14),
            budget_top_n: None,
        });
        let resolved = cfg.resolve_policy(MemoryType::Episodic, None);
        assert!(resolved.enabled);
        assert_eq!(resolved.max_inactive_days, 14);
        assert_eq!(resolved.budget_top_n, 10_000); // inherited from global
    }

    #[test]
    fn test_forgetting_per_scope_override_wins_over_per_type() {
        let mut cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        cfg.by_type.insert(MemoryType::Episodic, PerTypeForgettingConfig {
            enabled: None,
            max_inactive_days: Some(14),
            budget_top_n: None,
        });
        let scope_override = ScopeForgettingOverride {
            enabled: None,
            max_inactive_days: Some(3650),
            budget_top_n: Some(0),
        };
        let resolved = cfg.resolve_policy(MemoryType::Episodic, Some(&scope_override));
        assert!(resolved.enabled);
        assert_eq!(resolved.max_inactive_days, 3650);
        assert_eq!(resolved.budget_top_n, 0);
    }

    #[test]
    fn test_forgetting_global_disabled_overrides_all() {
        let mut cfg = ForgettingConfig::default(); // enabled: false
        cfg.by_type.insert(MemoryType::Episodic, PerTypeForgettingConfig {
            enabled: Some(true),
            max_inactive_days: Some(14),
            budget_top_n: None,
        });
        let scope_override = ScopeForgettingOverride {
            enabled: Some(true),
            max_inactive_days: Some(7),
            budget_top_n: None,
        };
        let resolved = cfg.resolve_policy(MemoryType::Episodic, Some(&scope_override));
        assert!(!resolved.enabled);
    }

    #[test]
    fn test_forgetting_per_type_disabled() {
        let mut cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        cfg.by_type.insert(MemoryType::Structural, PerTypeForgettingConfig {
            enabled: Some(false),
            max_inactive_days: None,
            budget_top_n: None,
        });
        let resolved = cfg.resolve_policy(MemoryType::Structural, None);
        assert!(!resolved.enabled);
    }

    #[test]
    fn test_forgetting_per_type_disabled_scope_cannot_reenable() {
        // Per-type enabled=false is a hard disable — scope overrides cannot
        // re-enable it. This allows admins to guarantee certain memory types
        // are never subject to forgetting.
        let mut cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        cfg.by_type.insert(MemoryType::Structural, PerTypeForgettingConfig {
            enabled: Some(false),
            max_inactive_days: None,
            budget_top_n: None,
        });
        let scope_override = ScopeForgettingOverride {
            enabled: Some(true), // attempt to re-enable
            max_inactive_days: Some(365),
            budget_top_n: None,
        };
        let resolved = cfg.resolve_policy(MemoryType::Structural, Some(&scope_override));
        assert!(!resolved.enabled, "per-type disabled is a hard disable");
    }

    #[test]
    fn test_forgetting_scope_max_inactive_days_zero_rejected() {
        // Scope-level max_inactive_days = 0 should be caught by validation in load()
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

[[scope]]
id = "bad-scope"
[scope.forgetting]
max_inactive_days = 0
"#;
        let path = std::path::Path::new("/tmp/corvia-test-scope-forgetting-validate.toml");
        std::fs::write(path, toml_str).unwrap();
        let result = CorviaConfig::load(path);
        assert!(result.is_err(), "scope forgetting max_inactive_days=0 should fail validation");
    }

    #[test]
    fn test_forgetting_missing_config_backward_compat() {
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
        assert!(config.forgetting.is_none());
    }

    #[test]
    fn test_forgetting_full_toml_parse() {
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

[forgetting]
enabled = true
interval_minutes = 30

[forgetting.defaults]
max_inactive_days = 90
budget_top_n = 10000

[forgetting.by_type.episodic]
max_inactive_days = 14

[forgetting.by_type.structural]
enabled = false

[forgetting.by_type.decisional]
max_inactive_days = 365

[forgetting.by_type.procedural]
max_inactive_days = 180

[[scope]]
id = "compliance"
description = "Compliance data"
[scope.forgetting]
max_inactive_days = 3650
budget_top_n = 0
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let fg = config.forgetting.as_ref().unwrap();
        assert!(fg.enabled);
        assert_eq!(fg.interval_minutes, 30);
        assert_eq!(fg.defaults.max_inactive_days, 90);
        assert_eq!(fg.defaults.budget_top_n, 10_000);

        // Per-type checks
        assert_eq!(fg.by_type.len(), 4);
        let episodic = fg.by_type.get(&MemoryType::Episodic).unwrap();
        assert_eq!(episodic.max_inactive_days, Some(14));
        let structural = fg.by_type.get(&MemoryType::Structural).unwrap();
        assert_eq!(structural.enabled, Some(false));
        let decisional = fg.by_type.get(&MemoryType::Decisional).unwrap();
        assert_eq!(decisional.max_inactive_days, Some(365));

        // Per-scope check
        let scopes = config.scope.as_ref().unwrap();
        assert_eq!(scopes.len(), 1);
        let scope_fg = scopes[0].forgetting.as_ref().unwrap();
        assert_eq!(scope_fg.max_inactive_days, Some(3650));
        assert_eq!(scope_fg.budget_top_n, Some(0));
    }

    #[test]
    fn test_forgetting_budget_zero_means_no_limit() {
        let cfg = ForgettingConfig {
            enabled: true,
            defaults: ForgettingPolicyConfig {
                max_inactive_days: 90,
                budget_top_n: 0,
            },
            ..ForgettingConfig::default()
        };
        let resolved = cfg.resolve_policy(MemoryType::Episodic, None);
        assert_eq!(resolved.budget_top_n, 0);
    }

    #[test]
    fn test_forgetting_validation_ok() {
        let cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_forgetting_validation_interval_zero() {
        let cfg = ForgettingConfig {
            enabled: true,
            interval_minutes: 0,
            ..ForgettingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_forgetting_validation_max_inactive_days_zero() {
        let cfg = ForgettingConfig {
            enabled: true,
            defaults: ForgettingPolicyConfig {
                max_inactive_days: 0,
                budget_top_n: 10_000,
            },
            ..ForgettingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_forgetting_validation_per_type_max_inactive_zero() {
        let mut cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        cfg.by_type.insert(MemoryType::Episodic, PerTypeForgettingConfig {
            enabled: None,
            max_inactive_days: Some(0),
            budget_top_n: None,
        });
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_forgetting_resolve_no_type_override_uses_defaults() {
        let cfg = ForgettingConfig {
            enabled: true,
            ..ForgettingConfig::default()
        };
        // Analytical has no by_type entry → should get global defaults
        let resolved = cfg.resolve_policy(MemoryType::Analytical, None);
        assert!(resolved.enabled);
        assert_eq!(resolved.max_inactive_days, 90);
        assert_eq!(resolved.budget_top_n, 10_000);
    }

    #[test]
    fn test_forgetting_scope_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.forgetting = Some(ForgettingConfig {
            enabled: true,
            interval_minutes: 30,
            defaults: ForgettingPolicyConfig {
                max_inactive_days: 90,
                budget_top_n: 5000,
            },
            by_type: HashMap::new(),
        });
        config.scope = Some(vec![ScopeConfig {
            id: "test-scope".into(),
            description: Some("Test".into()),
            ttl_days: None,
            forgetting: Some(ScopeForgettingOverride {
                enabled: None,
                max_inactive_days: Some(365),
                budget_top_n: None,
            }),
        }]);
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        let fg = loaded.forgetting.unwrap();
        assert!(fg.enabled);
        assert_eq!(fg.interval_minutes, 30);
        assert_eq!(fg.defaults.budget_top_n, 5000);
        let scopes = loaded.scope.unwrap();
        let scope_fg = scopes[0].forgetting.as_ref().unwrap();
        assert_eq!(scope_fg.max_inactive_days, Some(365));
    }
}
