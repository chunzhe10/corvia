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
        }
    }
}

/// Token budget for context assembly.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Override for max context tokens. None = model-aware auto-sizing.
    pub max_context_tokens: Option<usize>,
    /// Fraction of context window reserved for answer generation.
    pub reserve_for_answer: f32,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_context_tokens: None,
            reserve_for_answer: 0.2,
        }
    }
}

/// Output of the retrieval stage.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub results: Vec<SearchResult>,
    pub metrics: RetrievalMetrics,
}

/// Metrics from the retrieval stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub latency_ms: u64,
    pub vector_results: usize,
    pub graph_expanded: usize,
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
}

/// Output of the generation stage.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

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

/// RAG pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub default_limit: usize,
    pub graph_expand: bool,
    pub graph_depth: usize,
    pub graph_alpha: f32,
    pub reserve_for_answer: f32,
    pub max_context_tokens: usize,
    pub system_prompt: Option<String>,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            graph_expand: true,
            graph_depth: 2,
            graph_alpha: 0.3,
            reserve_for_answer: 0.2,
            max_context_tokens: 0,
            system_prompt: None,
        }
    }
}
