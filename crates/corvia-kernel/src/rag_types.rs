use corvia_common::agent_types::{AgentPermission, VisibilityMode};
use corvia_common::types::SearchResult;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// --- Agentic Retrieval Protocol (Issue #47) ---

/// Confidence level for search quality signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

/// Quality signal appended to search responses.
/// Enables agents to assess result relevance and retry with better queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignal {
    pub confidence: ConfidenceLevel,
    pub top_score: f32,
    pub result_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    pub gap_detected: bool,
    /// Number of results filtered out by min_score parameter.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub below_threshold_count: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// Thresholds for confidence level computation.
const HIGH_SCORE_THRESHOLD: f32 = 0.65;
const LOW_SCORE_THRESHOLD: f32 = 0.45;
const HIGH_COUNT_THRESHOLD: usize = 3;

impl QualitySignal {
    /// Compute quality signal from search results.
    pub fn from_results(results: &[SearchResult], query: &str, limit: usize) -> Self {
        let result_count = results.len();
        let top_score = results.first().map(|r| r.score).unwrap_or(0.0);

        let confidence = if top_score >= HIGH_SCORE_THRESHOLD && result_count >= HIGH_COUNT_THRESHOLD {
            ConfidenceLevel::High
        } else if top_score >= LOW_SCORE_THRESHOLD || result_count >= HIGH_COUNT_THRESHOLD {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        };

        let gap_detected = top_score < LOW_SCORE_THRESHOLD && result_count < limit / 2;

        let suggestion = match confidence {
            ConfidenceLevel::High => None,
            ConfidenceLevel::Medium => {
                Some("Consider adding context: scope, component name, or time period.".to_string())
            }
            ConfidenceLevel::Low => {
                if result_count < HIGH_COUNT_THRESHOLD {
                    let key_terms = extract_key_terms(query);
                    Some(format!("Try broader terms: {key_terms}"))
                } else {
                    Some("Results may not match intent. Try more specific terms.".to_string())
                }
            }
        };

        QualitySignal {
            confidence,
            top_score,
            result_count,
            suggestion,
            gap_detected,
            below_threshold_count: 0,
        }
    }
}

/// Extract key nouns/terms from a query for suggestion generation.
/// Simple heuristic: split on whitespace, filter short words and stop words.
fn extract_key_terms(query: &str) -> String {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "how", "what", "when", "where",
        "which", "who", "whom", "why", "this", "that", "these", "those", "it",
        "its", "my", "your", "his", "her", "our", "their", "i", "me", "we",
        "you", "he", "she", "they", "them",
    ];
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .filter(|w| !STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .take(5)
        .collect();
    if terms.is_empty() {
        query.to_string()
    } else {
        terms.join(", ")
    }
}

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
    use corvia_common::types::{KnowledgeEntry, Tier, EntryMetadata};
    use corvia_common::agent_types::EntryStatus;

    #[test]
    fn test_retrieval_opts_filter_defaults() {
        let opts = RetrievalOpts::default();
        assert!(opts.content_role.is_none());
        assert!(opts.source_origin.is_none());
        assert!(opts.workstream.is_none());
        assert!(!opts.include_cold);
    }

    fn mock_result(score: f32) -> SearchResult {
        let mut entry = KnowledgeEntry::new("test".into(), "s".into(), "v1".into());
        entry.entry_status = EntryStatus::Merged;
        SearchResult { entry, score, tier: Tier::Hot, retention_score: None }
    }

    #[test]
    fn test_quality_signal_high_confidence() {
        let results: Vec<SearchResult> = (0..5).map(|i| mock_result(0.80 - i as f32 * 0.03)).collect();
        let qs = QualitySignal::from_results(&results, "test query", 10);
        assert_eq!(qs.confidence, ConfidenceLevel::High);
        assert!(qs.suggestion.is_none());
        assert!(!qs.gap_detected);
    }

    #[test]
    fn test_quality_signal_medium_confidence() {
        let results = vec![mock_result(0.50), mock_result(0.40)];
        let qs = QualitySignal::from_results(&results, "test query", 10);
        assert_eq!(qs.confidence, ConfidenceLevel::Medium);
        assert!(qs.suggestion.is_some());
    }

    #[test]
    fn test_quality_signal_low_confidence() {
        let results = vec![mock_result(0.30)];
        let qs = QualitySignal::from_results(&results, "HNSW configuration", 10);
        assert_eq!(qs.confidence, ConfidenceLevel::Low);
        assert!(qs.gap_detected);
        let suggestion = qs.suggestion.unwrap();
        assert!(suggestion.starts_with("Try broader terms:"));
        assert!(suggestion.contains("HNSW"));
    }

    #[test]
    fn test_quality_signal_empty_results() {
        let qs = QualitySignal::from_results(&[], "anything", 10);
        assert_eq!(qs.confidence, ConfidenceLevel::Low);
        assert!(qs.gap_detected);
        assert_eq!(qs.top_score, 0.0);
    }

    #[test]
    fn test_extract_key_terms_filters_stop_words() {
        let terms = extract_key_terms("how does the HNSW index work with embeddings");
        assert!(terms.contains("HNSW"));
        assert!(terms.contains("index"));
        assert!(terms.contains("embeddings"));
        assert!(!terms.contains("how"));
        assert!(!terms.contains("the"));
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
    /// BM25 search latency (pipeline mode only, Phase 2a+).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_latency_ms: Option<u64>,
    /// Number of BM25 results before fusion (pipeline mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_results: Option<usize>,
    /// Fusion method used (pipeline mode only, e.g. "passthrough", "rrf").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_method: Option<String>,
    /// Fusion stage latency (pipeline mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fusion_latency_ms: Option<u64>,
    /// Per-stage metrics (pipeline mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stages: Option<Vec<crate::pipeline::StageMetrics>>,
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
