//! Searcher trait and implementations for the composable pipeline.
//!
//! A `Searcher` produces a [`RankedSet`] from a [`SearchContext`]. Multiple
//! searchers can run in parallel; their results are merged by a [`Fusion`] stage.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::{SearchResult, Tier};
use tracing::{info, warn};

use super::{
    CandidateScores, NormalizedScore, RankedCandidate, RankedSet, SearchContext, StageMetrics,
};
use crate::lite_store::LiteStore;
use crate::retriever::tier_weight;
use crate::traits::{FullTextSearchable, InferenceEngine, QueryableStore};

/// First stage of the pipeline: query -> ranked candidates.
///
/// Each searcher declares whether it needs an embedding via [`needs_embedding`].
/// The pipeline computes the embedding once and shares it via `SearchContext`.
///
/// Searchers MUST normalize their scores to `[0, 1]` (using [`NormalizedScore`]).
#[async_trait]
pub trait Searcher: Send + Sync {
    /// Human-readable name for metrics attribution.
    fn name(&self) -> &str;

    /// Whether this searcher requires a query embedding.
    /// If false, the pipeline may skip embedding computation when this is the only searcher.
    fn needs_embedding(&self) -> bool;

    /// Produce ranked candidates for the given search context.
    async fn search(&self, ctx: &SearchContext) -> Result<RankedSet>;
}

// ---------------------------------------------------------------------------
// VectorSearcher — extracted from VectorRetriever
// ---------------------------------------------------------------------------

/// Vector similarity searcher using the store's HNSW index.
///
/// Extracted from the monolithic `VectorRetriever`. Produces candidates with
/// cosine similarity scores normalized by tier weight to `[0, 1]`.
pub struct VectorSearcher {
    store: Arc<dyn QueryableStore>,
}

impl VectorSearcher {
    pub fn new(store: Arc<dyn QueryableStore>, _engine: Arc<dyn InferenceEngine>) -> Self {
        // engine is accepted for API compatibility but not stored;
        // embedding is provided via SearchContext.query_embedding.
        Self { store }
    }

    /// Convert raw `SearchResult`s into `RankedCandidate`s with tier weighting.
    fn to_candidates(results: Vec<SearchResult>, component_name: &str) -> Vec<RankedCandidate> {
        results
            .into_iter()
            .filter(|sr| sr.entry.tier != Tier::Forgotten)
            .map(|sr| {
                let weighted = sr.score * tier_weight(sr.entry.tier);
                let components = HashMap::from([(component_name.to_string(), weighted)]);
                RankedCandidate {
                    entry: Arc::new(sr.entry),
                    scores: CandidateScores {
                        components,
                        final_score: NormalizedScore::new(weighted),
                    },
                }
            })
            .collect()
    }
}

#[async_trait]
impl Searcher for VectorSearcher {
    fn name(&self) -> &str {
        "vector"
    }

    fn needs_embedding(&self) -> bool {
        true
    }

    async fn search(&self, ctx: &SearchContext) -> Result<RankedSet> {
        let start = Instant::now();

        // Oversample to allow for post-filter elimination.
        let search_limit = if ctx.opts.content_role.is_some()
            || ctx.opts.source_origin.is_some()
            || ctx.opts.workstream.is_some()
        {
            ctx.opts.limit * 3
        } else {
            ctx.opts.limit
        };
        let fetch_limit = (search_limit * ctx.opts.oversample_factor).max(10);

        let raw_results = self
            .store
            .search(&ctx.query_embedding, &ctx.scope_id, fetch_limit)
            .await?;
        let vector_count = raw_results.len();

        // Merge cold-tier results if requested.
        let mut all_results = raw_results;
        let mut cold_count = 0usize;
        if ctx.opts.include_cold
            && let Some(ls) = self.store.as_any().downcast_ref::<LiteStore>() {
                match ls.scan_cold_entries(&ctx.query_embedding, &ctx.scope_id, ctx.opts.limit) {
                    Ok(cold) => {
                        let existing: std::collections::HashSet<uuid::Uuid> =
                            all_results.iter().map(|sr| sr.entry.id).collect();
                        for sr in cold {
                            if !existing.contains(&sr.entry.id) {
                                cold_count += 1;
                                all_results.push(sr);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "Cold scan failed in VectorSearcher");
                    }
                }
            }

        let mut candidates = Self::to_candidates(all_results, "vector");

        // Sort by final_score descending.
        candidates.sort_unstable_by_key(|c| std::cmp::Reverse(c.scores.final_score));

        let output_count = candidates.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        info!(
            searcher = self.name(),
            vector_count,
            cold_count,
            output_count,
            latency_ms,
            "vector search complete"
        );

        Ok(RankedSet {
            candidates,
            metrics: StageMetrics {
                stage_name: self.name().to_string(),
                latency_ms,
                input_count: 0, // searchers have no input
                output_count,
                warnings: Vec::new(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Bm25Searcher — full-text BM25 search via FullTextSearchable
// ---------------------------------------------------------------------------

/// BM25 full-text searcher using a [`FullTextSearchable`] backend (e.g., TantivyIndex).
///
/// Does NOT require an embedding (uses the raw query string). Looks up full
/// entries from the store to apply tier weighting and build RankedCandidates.
pub struct Bm25Searcher {
    store: Arc<dyn QueryableStore>,
    fts: Arc<dyn FullTextSearchable>,
}

impl Bm25Searcher {
    pub fn new(store: Arc<dyn QueryableStore>, fts: Arc<dyn FullTextSearchable>) -> Self {
        Self { store, fts }
    }
}

#[async_trait]
impl Searcher for Bm25Searcher {
    fn name(&self) -> &str {
        "bm25"
    }

    fn needs_embedding(&self) -> bool {
        false
    }

    async fn search(&self, ctx: &SearchContext) -> Result<RankedSet> {
        let start = Instant::now();

        let fetch_limit = (ctx.opts.limit * ctx.opts.oversample_factor).max(10);
        let text_results = self
            .fts
            .search_text(&ctx.query, &ctx.scope_id, fetch_limit)
            .await?;

        let bm25_count = text_results.len();

        // Find max BM25 score for normalization.
        let max_score = text_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);

        // Look up full entries and build candidates.
        let mut candidates = Vec::with_capacity(text_results.len());
        for tr in text_results {
            if let Ok(Some(entry)) = self.store.get(&tr.entry_id).await {
                if entry.tier == corvia_common::types::Tier::Forgotten {
                    continue;
                }

                // Normalize BM25 score to [0, 1] via max-normalization.
                let normalized = if max_score > 0.0 {
                    tr.score / max_score
                } else {
                    0.0
                };
                let weighted = normalized * tier_weight(entry.tier);
                let components = HashMap::from([("bm25".to_string(), weighted)]);

                candidates.push(RankedCandidate {
                    entry: Arc::new(entry),
                    scores: CandidateScores {
                        components,
                        final_score: NormalizedScore::new(weighted),
                    },
                });
            }
        }

        candidates.sort_unstable_by_key(|c| std::cmp::Reverse(c.scores.final_score));

        let output_count = candidates.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        info!(
            searcher = self.name(),
            bm25_count,
            output_count,
            latency_ms,
            "BM25 search complete"
        );

        Ok(RankedSet {
            candidates,
            metrics: StageMetrics {
                stage_name: self.name().to_string(),
                latency_ms,
                input_count: 0,
                output_count,
                warnings: Vec::new(),
            },
        })
    }
}
