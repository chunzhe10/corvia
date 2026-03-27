use corvia_common::agent_types::{AgentPermission, VisibilityMode};
use corvia_common::types::SearchResult;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Options controlling the retrieval stage.
#[derive(Debug, Clone)]
pub struct RetrievalOpts {
    /// Maximum number of results to return.
    pub limit: usize,
    /// Whether to expand results by following graph edges.
    pub expand_graph: bool,
    /// Maximum graph traversal depth (hops).
    pub graph_depth: usize,
    /// Visibility mode for entry filtering.
    pub visibility: VisibilityMode,
    /// Agent ID for visibility filtering.
    pub agent_id: Option<String>,
    /// Agent permissions for RBAC scope filtering.
    pub permissions: Option<AgentPermission>,
    /// Oversample factor for vector search (more seeds = more graph expansion).
    pub oversample_factor: usize,
    /// Filter results by content_role (post-filter, Option A from spec).
    pub content_role: Option<String>,
    /// Filter results by source_origin (post-filter, Option A from spec).
    pub source_origin: Option<String>,
    /// Filter results by workstream (e.g. git branch name).
    pub workstream: Option<String>,
    /// Include Cold-tier entries via brute-force cosine scan (default false).
    pub include_cold: bool,
}

impl Default for RetrievalOpts {
    fn default() -> Self {
        Self {
            limit: 10,
            expand_graph: true,
            graph_depth: 2,
            visibility: VisibilityMode::All,
            agent_id: None,
            permissions: None,
            oversample_factor: 2,
            content_role: None,
            source_origin: None,
            workstream: None,
            include_cold: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_opts_filter_defaults() {
        let opts = RetrievalOpts::default();
        assert!(opts.content_role.is_none());
        assert!(opts.source_origin.is_none());
        assert!(opts.workstream.is_none());
        assert!(!opts.include_cold);
    }
}

/// Token budget for context assembly.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Override for max context tokens. None = model-aware auto-sizing.
    pub max_context_tokens: Option<usize>,
    /// Fraction of context window reserved for answer generation.
    pub reserve_for_answer: f32,
    /// Fraction of context window reserved for skill content. 0.0 = skills disabled.
    pub reserve_for_skills: f32,
    /// Query embedding for skill matching. None = skip skill matching.
    pub query_embedding: Option<Vec<f32>>,
    /// Maximum number of skills to inject.
    pub max_skills: usize,
    /// Minimum cosine similarity to select a skill.
    pub skill_threshold: f32,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_context_tokens: None,
            reserve_for_answer: 0.2,
            reserve_for_skills: 0.0,
            query_embedding: None,
            max_skills: 0,
            skill_threshold: 0.3,
        }
    }
}

/// Output of the retrieval stage.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub results: Vec<SearchResult>,
    pub metrics: RetrievalMetrics,
    /// Query embedding produced during retrieval, reused for skill matching.
    pub query_embedding: Option<Vec<f32>>,
}

/// Metrics from the retrieval stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub latency_ms: u64,
    /// Time spent on embedding inference (gRPC → ONNX Runtime).
    #[serde(default)]
    pub embed_latency_ms: u64,
    /// Time spent on HNSW vector search + graph expansion + post-filtering.
    #[serde(default)]
    pub search_latency_ms: u64,
    /// Time spent on HNSW vector search only (subset of search_latency_ms).
    #[serde(default)]
    pub hnsw_latency_ms: u64,
    /// Time spent on graph expansion only (subset of search_latency_ms).
    #[serde(default)]
    pub graph_latency_ms: u64,
    /// Time spent on sort + filter (subset of search_latency_ms).
    #[serde(default)]
    pub filter_latency_ms: u64,
    /// Time spent on brute-force cosine scan over Cold-tier entries.
    #[serde(default)]
    pub cold_scan_latency_ms: u64,
    pub vector_results: usize,
    /// Number of Cold-tier entries returned from brute-force scan.
    #[serde(default)]
    pub cold_results: usize,
    pub graph_expanded: usize,
    #[serde(default)]
    pub graph_reinforced: usize,
    pub post_filter_count: usize,
    pub retriever_name: String,
}

/// Output of the augmentation stage.
#[derive(Debug, Clone)]
pub struct AugmentedContext {
    pub system_prompt: String,
    pub context: String,
    pub sources: Vec<SearchResult>,
    pub metrics: AugmentationMetrics,
}

/// Metrics from the augmentation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationMetrics {
    pub latency_ms: u64,
    pub token_estimate: usize,
    pub token_budget: usize,
    pub sources_included: usize,
    pub sources_truncated: usize,
    pub augmenter_name: String,
    /// Names of dynamic skills injected into the system prompt.
    #[serde(default)]
    pub skills_used: Vec<String>,
}

// Re-export from traits — single definition used by both GenerationEngine and RAG pipeline.
pub use crate::traits::GenerationResult;

/// Metrics from the generation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub latency_ms: u64,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Full pipeline trace for observability and eval (D62 Layer B).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTrace {
    pub trace_id: Uuid,
    pub retrieval: RetrievalMetrics,
    pub augmentation: AugmentationMetrics,
    pub generation: Option<GenerationMetrics>,
    pub total_latency_ms: u64,
}

/// Complete RAG response returned to callers.
#[derive(Debug, Clone)]
pub struct RagResponse {
    /// Generated answer. None in context-only mode.
    pub answer: Option<String>,
    /// Assembled context with sources.
    pub context: AugmentedContext,
    /// Full pipeline trace for eval/observability.
    pub trace: PipelineTrace,
}

// Re-export from config — single definition for TOML deserialization and runtime use.
pub use corvia_common::config::RagConfig;
