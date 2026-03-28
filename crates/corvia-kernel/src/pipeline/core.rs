//! The composable [`RetrievalPipeline`] that implements the [`Retriever`] trait.
//!
//! Wires searchers -> fusion -> expander -> reranker into a single pipeline
//! that is a drop-in replacement for the legacy monolithic retrievers.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::SearchResult;
use tracing::{info, warn};

use super::expander::Expander;
use super::fusion::Fusion;
use super::searcher::Searcher;
use super::{Reranker, SearchContext};
use crate::rag_types::{RetrievalMetrics, RetrievalOpts, RetrievalResult};
use crate::retriever::{self, Retriever};
use crate::traits::InferenceEngine;

/// Composable retrieval pipeline.
///
/// Runs the four-stage pipeline:
/// 1. RBAC check
/// 2. Embed query (shared across searchers)
/// 3. Run searchers in parallel (`tokio::spawn` per searcher)
/// 4. Fuse results (dedup by UUID)
/// 5. Expand (graph traversal)
/// 6. Rerank
/// 7. Apply visibility + metadata filters
/// 8. Truncate to limit
/// 9. Record access
pub struct RetrievalPipeline {
    searchers: Vec<Arc<dyn Searcher>>,
    fusion: Arc<dyn Fusion>,
    expander: Arc<dyn Expander>,
    reranker: Arc<dyn Reranker>,
    engine: Arc<dyn InferenceEngine>,
    store: Arc<dyn crate::traits::QueryableStore>,
    /// Timeout for individual searchers in milliseconds.
    searcher_timeout_ms: u64,
}

impl RetrievalPipeline {
    pub fn new(
        searchers: Vec<Arc<dyn Searcher>>,
        fusion: Arc<dyn Fusion>,
        expander: Arc<dyn Expander>,
        reranker: Arc<dyn Reranker>,
        engine: Arc<dyn InferenceEngine>,
        store: Arc<dyn crate::traits::QueryableStore>,
        searcher_timeout_ms: u64,
    ) -> Self {
        Self {
            searchers,
            fusion,
            expander,
            reranker,
            engine,
            store,
            searcher_timeout_ms,
        }
    }
}

#[async_trait]
impl Retriever for RetrievalPipeline {
    fn name(&self) -> &str {
        "pipeline"
    }

    async fn retrieve(
        &self,
        query: &str,
        scope_id: &str,
        opts: &RetrievalOpts,
    ) -> Result<RetrievalResult> {
        let start = Instant::now();

        // Step 0: RBAC check.
        if let Some(denied) = retriever::check_rbac_scope(opts, scope_id, self.name(), &start) {
            return Ok(denied);
        }

        // Step 1: Embed query.
        let embed_start = Instant::now();
        let embedding = self.engine.embed(query).await?;
        let embed_latency_ms = embed_start.elapsed().as_millis() as u64;
        let query_embedding = Arc::new(embedding.clone());

        let ctx = SearchContext {
            query_embedding: query_embedding.clone(),
            scope_id: scope_id.to_string(),
            query: query.to_string(),
            opts: opts.clone(),
        };

        // Step 2: Run searchers in parallel with timeout + panic isolation.
        let search_start = Instant::now();
        let mut handles = Vec::new();
        let timeout = std::time::Duration::from_millis(self.searcher_timeout_ms);

        for searcher in &self.searchers {
            let s = Arc::clone(searcher);
            let c = ctx.clone();
            let t = timeout;
            handles.push(tokio::spawn(async move {
                match tokio::time::timeout(t, s.search(&c)).await {
                    Ok(Ok(set)) => Ok(set),
                    Ok(Err(e)) => {
                        warn!(searcher = s.name(), error = %e, "searcher failed");
                        Err(e)
                    }
                    Err(_) => {
                        warn!(searcher = s.name(), "searcher timed out");
                        Err(corvia_common::errors::CorviaError::Infra(
                            format!("searcher '{}' timed out", s.name()),
                        ))
                    }
                }
            }));
        }

        let mut searcher_results = Vec::new();
        let mut search_warnings = Vec::new();
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(set)) => searcher_results.push(set),
                Ok(Err(e)) => {
                    let name = self.searchers.get(i).map(|s| s.name()).unwrap_or("unknown");
                    search_warnings.push(format!("searcher '{name}' failed: {e}"));
                }
                Err(e) => {
                    let name = self.searchers.get(i).map(|s| s.name()).unwrap_or("unknown");
                    search_warnings.push(format!("searcher '{name}' panicked: {e}"));
                }
            }
        }

        let hnsw_latency_ms = search_start.elapsed().as_millis() as u64;
        let vector_results: usize = searcher_results.iter().map(|s| s.candidates.len()).sum();

        // Step 3: Fuse.
        let fused = self.fusion.fuse(searcher_results).await?;
        let fusion_latency_ms = fused.metrics.latency_ms;

        // Step 4: Expand.
        let expanded = self.expander.expand(&ctx, fused).await?;
        let graph_latency_ms = expanded.metrics.latency_ms;
        let graph_expanded = expanded.metrics.output_count.saturating_sub(expanded.metrics.input_count);

        // Step 5: Rerank.
        let reranked = self.reranker.rerank(&ctx, expanded).await?;

        // Step 6: Convert to SearchResult, apply visibility + metadata filters.
        let filter_start = Instant::now();
        let candidates_as_results: Vec<SearchResult> = reranked
            .candidates
            .into_iter()
            .map(|c| {
                let entry_ref = c.entry.as_ref();
                SearchResult {
                    score: c.scores.final_score.value(),
                    tier: entry_ref.tier,
                    retention_score: entry_ref.retention_score,
                    entry: entry_ref.clone(),
                }
            })
            .collect();

        // Visibility filter.
        let vis_filtered: Vec<SearchResult> = candidates_as_results
            .into_iter()
            .filter(|sr| {
                retriever::visibility_filter(
                    &sr.entry,
                    &opts.visibility,
                    opts.agent_id.as_deref(),
                )
            })
            .collect();

        // Metadata post-filter.
        let meta_filtered = retriever::post_filter_metadata(
            vis_filtered,
            opts.content_role.as_deref(),
            opts.source_origin.as_deref(),
            opts.workstream.as_deref(),
        );

        // Truncate to limit.
        let mut filtered = meta_filtered;
        filtered.truncate(opts.limit);
        let filter_latency_ms = filter_start.elapsed().as_millis() as u64;

        let post_filter_count = filtered.len();
        let search_latency_ms = search_start.elapsed().as_millis() as u64;

        // Step 7: Record access (fire-and-forget).
        retriever::spawn_access_recording(Arc::clone(&self.store), &filtered);

        let latency_ms = start.elapsed().as_millis() as u64;

        info!(
            retriever = self.name(),
            scope_id,
            vector_results,
            graph_expanded,
            post_filter_count,
            embed_latency_ms,
            search_latency_ms,
            latency_ms,
            "pipeline retrieval complete"
        );

        Ok(RetrievalResult {
            results: filtered,
            metrics: RetrievalMetrics {
                latency_ms,
                embed_latency_ms,
                search_latency_ms,
                hnsw_latency_ms,
                graph_latency_ms,
                filter_latency_ms,
                cold_scan_latency_ms: 0,
                vector_results,
                cold_results: 0,
                graph_expanded,
                graph_reinforced: 0,
                post_filter_count,
                retriever_name: self.name().to_string(),
                bm25_latency_ms: None,
                bm25_results: None,
                fusion_method: Some(self.fusion.name().to_string()),
                fusion_latency_ms: Some(fusion_latency_ms),
                stages: None,
            },
            query_embedding: Some(embedding),
        })
    }
}
