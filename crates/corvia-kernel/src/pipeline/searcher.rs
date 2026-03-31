//! Searcher trait and implementations for the composable pipeline.
//!
//! A `Searcher` produces a [`RankedSet`] from a [`SearchContext`]. Multiple
//! searchers can run in parallel; their results are merged by a [`Fusion`] stage.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::{KnowledgeEntry, SearchResult, Tier};
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

/// Compute the fetch limit for a searcher, accounting for metadata filters
/// and oversample factor. Shared across all searcher implementations.
fn fetch_limit(ctx: &SearchContext) -> usize {
    let base = if ctx.opts.content_role.is_some()
        || ctx.opts.source_origin.is_some()
        || ctx.opts.workstream.is_some()
    {
        ctx.opts.limit * 3
    } else {
        ctx.opts.limit
    };
    (base * ctx.opts.oversample_factor).max(10)
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

        let limit = fetch_limit(ctx);
        let raw_results = self
            .store
            .search(&ctx.query_embedding, &ctx.scope_id, limit)
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
// BM25Searcher — full-text search via FullTextSearchable
// ---------------------------------------------------------------------------

/// BM25/full-text searcher using a [`FullTextSearchable`] backend.
///
/// Store-agnostic: works with tantivy (LiteStore) or tsvector (PostgresStore).
/// Applies min-max normalization to raw text-relevance scores before returning.
pub struct BM25Searcher {
    fts: Arc<dyn FullTextSearchable>,
}

/// Intermediate representation for BM25 candidates before normalization.
#[derive(Debug)]
struct RawBM25Candidate {
    entry: Arc<KnowledgeEntry>,
    weighted: f32,
}

impl BM25Searcher {
    pub fn new(fts: Arc<dyn FullTextSearchable>) -> Self {
        Self { fts }
    }
}

#[async_trait]
impl Searcher for BM25Searcher {
    fn name(&self) -> &str {
        "bm25"
    }

    fn needs_embedding(&self) -> bool {
        false
    }

    async fn search(&self, ctx: &SearchContext) -> Result<RankedSet> {
        let start = Instant::now();

        let limit = fetch_limit(ctx);
        let raw_results = self
            .fts
            .search_text(&ctx.query, &ctx.scope_id, limit)
            .await?;

        // Convert to intermediate form with tier-weighted scores.
        // Filter must precede tier_weight multiplication (Forgotten weight = 0.0).
        let raw_candidates: Vec<RawBM25Candidate> = raw_results
            .into_iter()
            .filter(|sr| sr.entry.tier != Tier::Forgotten)
            .map(|sr| {
                let weighted = sr.score * tier_weight(sr.entry.tier);
                RawBM25Candidate {
                    entry: Arc::new(sr.entry),
                    weighted,
                }
            })
            .collect();

        // Min-max normalization on weighted scores before creating NormalizedScore.
        // Raw BM25/text-relevance scores are unbounded positive; this maps them to [0,1].
        // Note: lowest result maps to 0.0 even if highly relevant. This is acceptable
        // because RRF uses ranks (not scores) and PassThrough re-normalizes.
        let mut candidates: Vec<RankedCandidate> = if raw_candidates.is_empty() {
            Vec::new()
        } else {
            let (min_s, max_s) = raw_candidates.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(mn, mx), c| (mn.min(c.weighted), mx.max(c.weighted)),
            );
            let range = max_s - min_s;

            raw_candidates
                .into_iter()
                .map(|rc| {
                    let normalized = if range > f32::EPSILON {
                        (rc.weighted - min_s) / range
                    } else {
                        // Degenerate case: all scores identical (or single result) -> 1.0.
                        1.0
                    };
                    RankedCandidate {
                        entry: rc.entry,
                        scores: CandidateScores {
                            // Store tier-weighted score (consistent with VectorSearcher).
                            components: HashMap::from([("bm25".to_string(), rc.weighted)]),
                            final_score: NormalizedScore::new(normalized),
                        },
                    }
                })
                .collect()
        };

        // Sort descending by final_score.
        candidates.sort_unstable_by_key(|c| std::cmp::Reverse(c.scores.final_score));

        let output_count = candidates.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        info!(
            searcher = self.name(),
            output_count,
            latency_ms,
            "bm25 search complete"
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

// ---------------------------------------------------------------------------
// MultiChannelSearcher — per-memory-type retrieval with RRF fusion
// ---------------------------------------------------------------------------

/// Multi-channel searcher that runs per-memory-type retrieval strategies in
/// parallel and fuses results via Reciprocal Rank Fusion (RRF).
///
/// Each of the 5 memory types is assigned a strategy ("vector" or "bm25") via
/// [`ChannelsConfig`]. Channels run concurrently with a per-channel timeout.
/// If a channel times out or errors, it is skipped and a warning is logged.
///
/// Graph expansion is handled by the pipeline's Expander stage (post-fusion),
/// not per-channel. This preserves the pipeline's single-responsibility design.
///
/// **Pipeline integration:** When using MultiChannelSearcher as the sole searcher,
/// set `fusion = "passthrough"` in pipeline config since MultiChannelSearcher
/// performs its own internal RRF fusion.
pub struct MultiChannelSearcher {
    store: Arc<dyn QueryableStore>,
    fts: Option<Arc<dyn FullTextSearchable>>,
    channels: corvia_common::config::ChannelsConfig,
    timeout_ms: u64,
    rrf_k: usize,
}

impl MultiChannelSearcher {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        fts: Option<Arc<dyn FullTextSearchable>>,
        channels: corvia_common::config::ChannelsConfig,
        timeout_ms: u64,
        rrf_k: usize,
    ) -> Self {
        Self { store, fts, channels, timeout_ms, rrf_k }
    }
}

#[async_trait]
impl Searcher for MultiChannelSearcher {
    fn name(&self) -> &str {
        "multichannel"
    }

    fn needs_embedding(&self) -> bool {
        true
    }

    async fn search(&self, ctx: &SearchContext) -> Result<RankedSet> {
        use corvia_common::types::MemoryType;
        use super::fusion::RRFusion;
        use super::Fusion;

        let start = Instant::now();
        let limit = fetch_limit(ctx);
        let timeout = std::time::Duration::from_millis(self.timeout_ms);

        // Launch per-type channels concurrently.
        let mut handles = Vec::new();
        let mut warnings = Vec::new();

        for mt in MemoryType::ALL {
            let strategy = self.channels.strategy_for(mt).to_string();
            let embedding = Arc::clone(&ctx.query_embedding);
            let scope_id = ctx.scope_id.clone();
            let query = ctx.query.clone();

            match strategy.as_str() {
                "vector" => {
                    let store = Arc::clone(&self.store);
                    handles.push((mt, tokio::spawn(async move {
                        store.search_by_memory_type(&embedding, &scope_id, limit, mt).await
                    })));
                }
                "bm25" => {
                    if let Some(fts) = self.fts.clone() {
                        // Over-fetch by 3x to compensate for post-filter by memory type.
                        // BM25 search returns all types; post-filter keeps only the target type.
                        let bm25_fetch = limit * 3;
                        handles.push((mt, tokio::spawn(async move {
                            let results = fts.search_text(&query, &scope_id, bm25_fetch).await?;
                            Ok(results.into_iter()
                                .filter(|r| r.entry.memory_type == mt)
                                .take(limit)
                                .collect::<Vec<_>>())
                        })));
                    } else {
                        warnings.push(format!("{mt} channel: BM25 requested but no FTS available, skipping"));
                    }
                }
                other => {
                    warnings.push(format!("{mt} channel: unknown strategy '{other}', skipping"));
                }
            }
        }

        // Collect results with timeout. Channels that timeout or error are skipped.
        let mut channel_sets: Vec<RankedSet> = Vec::new();

        for (mt, handle) in handles {
            match tokio::time::timeout(timeout, handle).await {
                Ok(Ok(Ok(results))) => {
                    let candidates = VectorSearcher::to_candidates(results, &mt.to_string());
                    channel_sets.push(RankedSet {
                        candidates,
                        metrics: StageMetrics {
                            stage_name: format!("channel:{mt}"),
                            latency_ms: 0,
                            input_count: 0,
                            output_count: 0,
                            warnings: Vec::new(),
                        },
                    });
                }
                Ok(Ok(Err(e))) => {
                    warnings.push(format!("{mt} channel error: {e}"));
                }
                Ok(Err(e)) => {
                    warnings.push(format!("{mt} channel join error: {e}"));
                }
                Err(_) => {
                    warnings.push(format!("{mt} channel timed out after {}ms", self.timeout_ms));
                }
            }
        }

        // Fuse via RRF
        let fusion = RRFusion::new(self.rrf_k);
        let mut fused = fusion.fuse(channel_sets).await?;

        let output_count = fused.candidates.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        fused.metrics = StageMetrics {
            stage_name: self.name().to_string(),
            latency_ms,
            input_count: 0,
            output_count,
            warnings,
        };

        info!(
            searcher = self.name(),
            output_count,
            latency_ms,
            "multichannel search complete"
        );

        Ok(fused)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::types::KnowledgeEntry;

    /// Mock FullTextSearchable for testing BM25Searcher.
    struct MockFts {
        results: Vec<SearchResult>,
    }

    #[async_trait]
    impl FullTextSearchable for MockFts {
        async fn search_text(
            &self,
            _query: &str,
            _scope_id: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>> {
            Ok(self.results.clone())
        }
        async fn index_entry(&self, _entry: &KnowledgeEntry) -> Result<()> { Ok(()) }
        async fn remove_entry(&self, _entry_id: &uuid::Uuid) -> Result<()> { Ok(()) }
        async fn rebuild_from_store(&self, _entries: &[KnowledgeEntry]) -> Result<usize> { Ok(0) }
        async fn entry_count(&self) -> Result<u64> { Ok(0) }
    }

    fn make_search_result(id_suffix: u8, score: f32) -> SearchResult {
        let mut entry = KnowledgeEntry::new(
            format!("content-{id_suffix}"),
            "scope".into(),
            format!("v{id_suffix}"),
        );
        entry.id = uuid::Uuid::from_bytes([
            id_suffix, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        SearchResult {
            entry,
            score,
            tier: corvia_common::types::Tier::Hot,
            retention_score: None,
        }
    }

    fn make_ctx() -> SearchContext {
        use crate::rag_types::RetrievalOpts;
        SearchContext {
            query_embedding: Arc::new(vec![]),
            scope_id: "test".into(),
            query: "test query".into(),
            opts: RetrievalOpts {
                limit: 10,
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn test_bm25_searcher_normalization() {
        let fts = MockFts {
            results: vec![
                make_search_result(1, 8.0),
                make_search_result(2, 4.0),
                make_search_result(3, 2.0),
            ],
        };
        let searcher = BM25Searcher::new(Arc::new(fts));
        let result = searcher.search(&make_ctx()).await.unwrap();

        assert_eq!(result.candidates.len(), 3);
        // Highest raw score -> normalized 1.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
        // Lowest raw score -> normalized 0.0.
        assert!((result.candidates[2].scores.final_score.value() - 0.0).abs() < f32::EPSILON);
        // Tier-weighted scores preserved in components (Hot tier weight = 1.0).
        assert!(result.candidates[0].scores.components.contains_key("bm25"));
        assert!((result.candidates[0].scores.components["bm25"] - 8.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_bm25_searcher_single_result() {
        let fts = MockFts {
            results: vec![make_search_result(1, 5.0)],
        };
        let searcher = BM25Searcher::new(Arc::new(fts));
        let result = searcher.search(&make_ctx()).await.unwrap();

        assert_eq!(result.candidates.len(), 1);
        // Single result degenerate case -> 1.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_bm25_searcher_empty_results() {
        let fts = MockFts { results: vec![] };
        let searcher = BM25Searcher::new(Arc::new(fts));
        let result = searcher.search(&make_ctx()).await.unwrap();

        assert!(result.candidates.is_empty());
    }
}
