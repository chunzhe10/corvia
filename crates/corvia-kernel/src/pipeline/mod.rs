//! Composable retrieval pipeline for the RAG system.
//!
//! Decomposes retrieval into four pluggable stages:
//! 1. **Searcher** — produce ranked candidates from a query (vector, BM25, etc.)
//! 2. **Fusion** — merge results from multiple searchers into a single ranked set
//! 3. **Expander** — enrich results via graph traversal or other expansion
//! 4. **Reranker** — re-score/re-order the final candidate set
//!
//! Each stage is behind an async trait. A [`RetrievalPipeline`] wires them together
//! and implements the existing [`Retriever`](crate::retriever::Retriever) trait so
//! it can be used as a drop-in replacement.

pub mod expander;
pub mod fusion;
pub mod core;
pub mod registry;
pub mod searcher;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::KnowledgeEntry;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::rag_types::RetrievalOpts;

// Re-exports for convenience.
pub use expander::Expander;
pub use fusion::{Fusion, RRFusion};
pub use core::RetrievalPipeline;
pub use registry::{ComponentDeps, PipelineRegistry};
pub use searcher::{BM25Searcher, Searcher};

// ---------------------------------------------------------------------------
// NormalizedScore
// ---------------------------------------------------------------------------

/// A score guaranteed to be in `[0.0, 1.0]`.
///
/// Constructed via [`NormalizedScore::new`] which clamps out-of-range values
/// and warns on NaN. This prevents score corruption from propagating through
/// the pipeline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalizedScore(f32);

impl NormalizedScore {
    /// Create a new normalized score, clamping to `[0.0, 1.0]`.
    ///
    /// Logs a warning if the input is out of range or NaN.
    pub fn new(value: f32) -> Self {
        if value.is_nan() {
            warn!(value, "NormalizedScore received NaN, clamping to 0.0");
            return Self(0.0);
        }
        if !(0.0..=1.0).contains(&value) {
            warn!(value, "NormalizedScore out of range, clamping to [0,1]");
        }
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the inner f32 value.
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }
}

impl PartialEq for NormalizedScore {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for NormalizedScore {}

impl PartialOrd for NormalizedScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NormalizedScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Safe: NaN is excluded by construction in new().
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl From<NormalizedScore> for f32 {
    fn from(s: NormalizedScore) -> f32 {
        s.0
    }
}

// ---------------------------------------------------------------------------
// CandidateScores
// ---------------------------------------------------------------------------

/// Per-component score breakdown for a single candidate.
///
/// `components` maps searcher/stage names to their raw contribution.
/// `final_score` is the fused, normalized result used for ranking.
#[derive(Debug, Clone)]
pub struct CandidateScores {
    /// Raw score contributions keyed by component name (e.g. "vector", "bm25").
    pub components: HashMap<String, f32>,
    /// The final fused score used for ranking.
    pub final_score: NormalizedScore,
}

// ---------------------------------------------------------------------------
// RankedCandidate
// ---------------------------------------------------------------------------

/// A knowledge entry with its composite score breakdown.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    /// The underlying knowledge entry (shared ownership for dedup efficiency).
    pub entry: Arc<KnowledgeEntry>,
    /// Per-component and final scores.
    pub scores: CandidateScores,
}

// ---------------------------------------------------------------------------
// StageMetrics / RankedSet
// ---------------------------------------------------------------------------

/// Timing and diagnostic metrics from a single pipeline stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageMetrics {
    /// Human-readable stage name (e.g. "vector_searcher", "passthrough_fusion").
    pub stage_name: String,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u64,
    /// Number of candidates entering this stage.
    pub input_count: usize,
    /// Number of candidates leaving this stage.
    pub output_count: usize,
    /// Non-fatal warnings emitted during this stage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// A set of ranked candidates produced by a pipeline stage.
#[derive(Debug, Clone)]
pub struct RankedSet {
    /// Candidates sorted by `final_score` descending.
    pub candidates: Vec<RankedCandidate>,
    /// Metrics from the stage that produced this set.
    pub metrics: StageMetrics,
}

// ---------------------------------------------------------------------------
// SearchContext
// ---------------------------------------------------------------------------

/// Shared context threaded through all pipeline stages.
#[derive(Debug, Clone)]
pub struct SearchContext {
    /// The query embedding (computed once, shared via Arc).
    pub query_embedding: Arc<Vec<f32>>,
    /// The scope being searched.
    pub scope_id: String,
    /// The original query string.
    pub query: String,
    /// Retrieval options (limit, filters, graph depth, etc.).
    pub opts: RetrievalOpts,
}

// ---------------------------------------------------------------------------
// Reranker trait
// ---------------------------------------------------------------------------

/// Final re-scoring/re-ordering stage.
///
/// The default implementation is an identity pass-through.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Human-readable name for metrics.
    fn name(&self) -> &str;

    /// Re-rank the candidate set.
    async fn rerank(&self, ctx: &SearchContext, set: RankedSet) -> Result<RankedSet>;
}

/// Identity reranker that passes candidates through unchanged.
pub struct IdentityReranker;

#[async_trait]
impl Reranker for IdentityReranker {
    fn name(&self) -> &str {
        "identity"
    }

    async fn rerank(&self, _ctx: &SearchContext, set: RankedSet) -> Result<RankedSet> {
        Ok(set)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalized_score_normal() {
        let s = NormalizedScore::new(0.5);
        assert!((s.value() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_clamps_above() {
        let s = NormalizedScore::new(1.5);
        assert!((s.value() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_clamps_below() {
        let s = NormalizedScore::new(-0.3);
        assert!((s.value() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_nan() {
        let s = NormalizedScore::new(f32::NAN);
        assert!((s.value() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_edge_zero() {
        let s = NormalizedScore::new(0.0);
        assert!((s.value() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_edge_one() {
        let s = NormalizedScore::new(1.0);
        assert!((s.value() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalized_score_into_f32() {
        let s = NormalizedScore::new(0.75);
        let v: f32 = s.into();
        assert!((v - 0.75).abs() < f32::EPSILON);
    }
}
