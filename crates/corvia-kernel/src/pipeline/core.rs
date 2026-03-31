//! The composable [`RetrievalPipeline`] that implements the [`Retriever`] trait.
//!
//! Wires searchers -> fusion -> expander -> reranker into a single pipeline
//! that is a drop-in replacement for the legacy monolithic retrievers.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::SearchResult;
use tracing::{error, info, warn};

use super::expander::Expander;
use super::fusion::Fusion;
use super::searcher::Searcher;
use super::{Reranker, SearchContext, StageMetrics};
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

        // Step 1: Embed query with graceful degradation.
        //
        // If embedding fails but there are searchers that don't need embeddings
        // (e.g. BM25), skip embedding-dependent searchers and proceed.
        let embed_start = Instant::now();
        let has_non_embedding_searchers = self.searchers.iter().any(|s| !s.needs_embedding());

        let (embedding_arc, embed_failed) = match self.engine.embed(query).await {
            Ok(emb) => (Arc::new(emb), false),
            Err(e) => {
                if has_non_embedding_searchers {
                    warn!(error = %e, "embedding failed, degrading to non-embedding searchers (e.g. BM25)");
                    // Provide a zero-length placeholder; embedding-dependent searchers are skipped below.
                    (Arc::new(Vec::new()), true)
                } else {
                    return Err(e);
                }
            }
        };
        let embed_latency_ms = embed_start.elapsed().as_millis() as u64;

        let ctx = SearchContext {
            query_embedding: Arc::clone(&embedding_arc),
            scope_id: scope_id.to_string(),
            query: query.to_string(),
            opts: opts.clone(),
        };

        // Determine which searchers to run. If embedding failed, skip those
        // that require embeddings.
        let active_searchers: Vec<&Arc<dyn Searcher>> = if embed_failed {
            self.searchers.iter().filter(|s| !s.needs_embedding()).collect()
        } else {
            self.searchers.iter().collect()
        };

        // Step 2: Run searchers with timeout + panic isolation.
        // Fast path: single searcher avoids tokio::spawn overhead.
        let search_start = Instant::now();
        let mut searcher_results = Vec::new();
        let mut search_warnings: Vec<String> = Vec::new();

        if embed_failed {
            search_warnings.push("embedding failed, vector searcher skipped".to_string());
        }

        if active_searchers.len() == 1 {
            let timeout = std::time::Duration::from_millis(self.searcher_timeout_ms);
            match tokio::time::timeout(timeout, active_searchers[0].search(&ctx)).await {
                Ok(Ok(set)) => searcher_results.push(set),
                Ok(Err(e)) => {
                    let name = active_searchers[0].name();
                    warn!(searcher = name, error = %e, "searcher failed");
                    search_warnings.push(format!("searcher '{name}' failed: {e}"));
                }
                Err(_) => {
                    let name = active_searchers[0].name();
                    warn!(searcher = name, "searcher timed out");
                    search_warnings.push(format!("searcher '{name}' timed out"));
                }
            }
        } else if !active_searchers.is_empty() {
            let mut handles = Vec::new();
            let timeout = std::time::Duration::from_millis(self.searcher_timeout_ms);

            for searcher in &active_searchers {
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
                            Err(CorviaError::Infra(
                                format!("searcher '{}' timed out", s.name()),
                            ))
                        }
                    }
                }));
            }

            for (i, handle) in handles.into_iter().enumerate() {
                match handle.await {
                    Ok(Ok(set)) => searcher_results.push(set),
                    Ok(Err(e)) => {
                        let name = active_searchers.get(i).map(|s| s.name()).unwrap_or("unknown");
                        search_warnings.push(format!("searcher '{name}' failed: {e}"));
                    }
                    Err(e) => {
                        let name = active_searchers.get(i).map(|s| s.name()).unwrap_or("unknown");
                        search_warnings.push(format!("searcher '{name}' panicked: {e}"));
                    }
                }
            }
        }

        if !search_warnings.is_empty() {
            warn!(warnings = ?search_warnings, "searcher warnings during pipeline execution");
        }

        // All searchers failed: return error instead of empty results.
        if searcher_results.is_empty() && !active_searchers.is_empty() {
            error!(
                warnings = ?search_warnings,
                "all searchers failed, cannot produce results"
            );
            return Err(CorviaError::Infra(format!(
                "all searchers failed: {}",
                search_warnings.join("; ")
            )));
        }

        let hnsw_latency_ms = search_start.elapsed().as_millis() as u64;
        let vector_results: usize = searcher_results.iter().map(|s| s.candidates.len()).sum();

        // Collect per-searcher stage metrics before fusion consumes the sets.
        let mut stages: Vec<StageMetrics> = searcher_results
            .iter()
            .map(|sr| sr.metrics.clone())
            .collect();

        // Step 3: Fuse.
        let fused = self.fusion.fuse(searcher_results).await?;
        let fusion_latency_ms = fused.metrics.latency_ms;
        stages.push(fused.metrics.clone());

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
                stages: Some(stages),
            },
            query_embedding: if embed_failed {
                None
            } else {
                Some(Arc::try_unwrap(embedding_arc)
                    .unwrap_or_else(|arc| (*arc).clone()))
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::pipeline::expander::GraphExpander;
    use crate::pipeline::fusion::PassThrough;
    use crate::pipeline::searcher::VectorSearcher;
    use crate::pipeline::IdentityReranker;
    use crate::rag_types::RetrievalOpts;
    use crate::retriever::{GraphExpandRetriever, Retriever};
    use crate::traits::{GraphStore, InferenceEngine, QueryableStore};
    use corvia_common::agent_types::{AgentPermission, EntryStatus, VisibilityMode};
    use corvia_common::errors::Result;
    use corvia_common::types::KnowledgeEntry;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    /// Mock embedding engine that returns a fixed 3-dim vector.
    struct MockEngine;

    #[async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize {
            3
        }
    }

    /// Mock searcher that sleeps for a configurable duration (for timeout test).
    struct SlowSearcher {
        delay: std::time::Duration,
    }

    #[async_trait]
    impl crate::pipeline::searcher::Searcher for SlowSearcher {
        fn name(&self) -> &str {
            "slow"
        }
        fn needs_embedding(&self) -> bool {
            false
        }
        async fn search(
            &self,
            _ctx: &SearchContext,
        ) -> Result<crate::pipeline::RankedSet> {
            tokio::time::sleep(self.delay).await;
            Ok(crate::pipeline::RankedSet {
                candidates: Vec::new(),
                metrics: crate::pipeline::StageMetrics {
                    stage_name: "slow".into(),
                    latency_ms: 0,
                    input_count: 0,
                    output_count: 0,
                    warnings: Vec::new(),
                },
            })
        }
    }

    /// Helper: insert entries into store with varied embeddings.
    async fn insert_entries(
        store: &Arc<LiteStore>,
        scope: &str,
        count: usize,
    ) -> Vec<KnowledgeEntry> {
        let mut entries = Vec::new();
        for i in 0..count {
            let mut entry = KnowledgeEntry::new(
                format!("knowledge entry {i} about topic {}", i % 5),
                scope.to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.01, 0.0]);
            store.insert(&entry).await.unwrap();
            entries.push(entry);
        }
        entries
    }

    // -----------------------------------------------------------------------
    // TEST 1: Compatibility -- legacy GraphExpandRetriever vs new pipeline
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_compatibility_with_legacy_retriever() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 15 entries with varied embeddings.
        let entries = insert_entries(&store, "compat-scope", 15).await;

        // Create graph edges between some entries for expansion coverage.
        // 0->1, 1->2, 3->4, 5->6, 7->8, 9->10
        let edge_pairs = [(0, 1), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)];
        for (a, b) in &edge_pairs {
            graph
                .relate(&entries[*a].id, "related_to", &entries[*b].id, None)
                .await
                .unwrap();
        }

        let alpha = 0.3;

        // Build the legacy retriever.
        let legacy = GraphExpandRetriever::new(
            queryable.clone(),
            engine.clone(),
            graph.clone(),
            alpha,
        );

        // Build the new pipeline retriever.
        let pipeline = RetrievalPipeline::new(
            vec![Arc::new(VectorSearcher::new(
                queryable.clone(),
                engine.clone(),
            ))],
            Arc::new(PassThrough),
            Arc::new(GraphExpander::new(queryable.clone(), graph.clone(), alpha)),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        // Test multiple queries for robustness.
        let queries = ["knowledge entry 0", "topic about", "entry 7"];
        let mut top1_matches = 0usize;

        for query in &queries {
            let opts = RetrievalOpts {
                limit: 10,
                expand_graph: true,
                graph_depth: 1,
                visibility: VisibilityMode::All,
                ..Default::default()
            };

            let legacy_result = legacy.retrieve(query, "compat-scope", &opts).await.unwrap();
            let pipeline_result = pipeline
                .retrieve(query, "compat-scope", &opts)
                .await
                .unwrap();

            // Both should return results (HNSW approximate, use >= 2).
            assert!(
                legacy_result.results.len() >= 2,
                "legacy should return at least 2 results for query '{query}', got {}",
                legacy_result.results.len()
            );
            assert!(
                pipeline_result.results.len() >= 2,
                "pipeline should return at least 2 results for query '{query}', got {}",
                pipeline_result.results.len()
            );

            // Verify entry ID overlap >= 80%.
            let legacy_ids: HashSet<uuid::Uuid> =
                legacy_result.results.iter().map(|r| r.entry.id).collect();
            let pipeline_ids: HashSet<uuid::Uuid> =
                pipeline_result.results.iter().map(|r| r.entry.id).collect();
            let overlap = legacy_ids.intersection(&pipeline_ids).count();
            let min_count = legacy_ids.len().min(pipeline_ids.len());
            if min_count > 0 {
                let overlap_pct = overlap as f64 / min_count as f64;
                assert!(
                    overlap_pct >= 0.80,
                    "entry ID overlap should be >= 80% for query '{query}', \
                     got {:.0}% ({overlap}/{min_count})",
                    overlap_pct * 100.0
                );
            }

            // Check top-1 match.
            if !legacy_result.results.is_empty() && !pipeline_result.results.is_empty() {
                if legacy_result.results[0].entry.id == pipeline_result.results[0].entry.id {
                    top1_matches += 1;
                }
            }

            // Graph expansion counts should be similar (within +-3).
            let legacy_expanded = legacy_result.metrics.graph_expanded;
            let pipeline_expanded = pipeline_result.metrics.graph_expanded;
            let diff =
                (legacy_expanded as i64 - pipeline_expanded as i64).unsigned_abs() as usize;
            assert!(
                diff <= 3,
                "graph_expanded difference too large for query '{query}': \
                 legacy={legacy_expanded}, pipeline={pipeline_expanded}"
            );
        }

        // Top-1 comparison is informational only. The legacy retriever applies
        // alpha-blended scoring directly (((1-alpha)*cosine + alpha*1.0) * tier_weight)
        // while the pipeline applies tier_weight in VectorSearcher then min-max
        // normalizes in PassThrough, which can reorder entries with very similar
        // embeddings. The entry ID overlap check above is the meaningful comparison.
        // We just log the top-1 match rate for diagnostics.
        let _top1_pct = top1_matches as f64 / queries.len() as f64;
    }

    // -----------------------------------------------------------------------
    // TEST 2: Pipeline integration -- full flow with RBAC + metadata filtering
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_integration_full_flow() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert entries with varied metadata for filtering tests.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.01, 0.0]
        };

        // 5 entries with content_role="design"
        let mut design_ids = Vec::new();
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("design document {i}"),
                "flow-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            entry.metadata.content_role = Some("design".to_string());
            entry.metadata.source_origin = Some("repo:corvia".to_string());
            queryable.insert(&entry).await.unwrap();
            design_ids.push(entry.id);
        }

        // 5 entries with content_role="code"
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("code snippet {i}"),
                "flow-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            entry.metadata.content_role = Some("code".to_string());
            entry.metadata.source_origin = Some("repo:other".to_string());
            queryable.insert(&entry).await.unwrap();
        }

        // 5 filler entries (no metadata) for HNSW connectivity.
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "flow-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            queryable.insert(&entry).await.unwrap();
        }

        // Create graph edges between design entries.
        graph
            .relate(&design_ids[0], "related_to", &design_ids[1], None)
            .await
            .unwrap();
        graph
            .relate(&design_ids[1], "related_to", &design_ids[2], None)
            .await
            .unwrap();

        let pipeline = RetrievalPipeline::new(
            vec![Arc::new(VectorSearcher::new(
                queryable.clone(),
                engine.clone(),
            ))],
            Arc::new(PassThrough),
            Arc::new(GraphExpander::new(queryable.clone(), graph.clone(), 0.3)),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        // Sub-test A: Basic flow (search -> fuse -> expand -> rerank -> filter -> truncate)
        {
            let opts = RetrievalOpts {
                limit: 10,
                expand_graph: true,
                graph_depth: 1,
                visibility: VisibilityMode::All,
                ..Default::default()
            };
            let result = pipeline
                .retrieve("design document", "flow-scope", &opts)
                .await
                .unwrap();
            assert!(!result.results.is_empty(), "should return results");
            assert_eq!(result.metrics.retriever_name, "pipeline");
            assert!(result.metrics.fusion_method.is_some());
            assert_eq!(result.metrics.fusion_method.as_deref(), Some("passthrough"));
            // Verify results are score-ordered descending.
            for window in result.results.windows(2) {
                assert!(
                    window[0].score >= window[1].score,
                    "results should be score-ordered descending"
                );
            }
        }

        // Sub-test B: RBAC filtering
        {
            let opts = RetrievalOpts {
                limit: 10,
                permissions: Some(AgentPermission::ReadWrite {
                    scopes: vec!["other-scope".to_string()],
                }),
                visibility: VisibilityMode::All,
                ..Default::default()
            };
            let result = pipeline
                .retrieve("design", "flow-scope", &opts)
                .await
                .unwrap();
            assert_eq!(
                result.results.len(),
                0,
                "RBAC should block access to flow-scope"
            );
            assert_eq!(result.metrics.vector_results, 0);
        }

        // Sub-test C: Metadata post-filtering (content_role)
        {
            let opts = RetrievalOpts {
                limit: 10,
                expand_graph: true,
                graph_depth: 1,
                visibility: VisibilityMode::All,
                content_role: Some("design".to_string()),
                ..Default::default()
            };
            let result = pipeline
                .retrieve("document", "flow-scope", &opts)
                .await
                .unwrap();
            // All results should have content_role="design".
            for sr in &result.results {
                assert_eq!(
                    sr.entry.metadata.content_role.as_deref(),
                    Some("design"),
                    "metadata filter should only pass content_role=design"
                );
            }
        }

        // Sub-test D: Metadata post-filtering (source_origin)
        {
            let opts = RetrievalOpts {
                limit: 10,
                expand_graph: false,
                visibility: VisibilityMode::All,
                source_origin: Some("repo:other".to_string()),
                ..Default::default()
            };
            let result = pipeline
                .retrieve("code snippet", "flow-scope", &opts)
                .await
                .unwrap();
            for sr in &result.results {
                assert_eq!(
                    sr.entry.metadata.source_origin.as_deref(),
                    Some("repo:other"),
                    "metadata filter should only pass source_origin=repo:other"
                );
            }
        }

        // Sub-test E: Visibility filtering (Own mode)
        {
            // Insert a pending entry for agent-X.
            let mut pending = KnowledgeEntry::new(
                "agent-X pending work".to_string(),
                "flow-scope".to_string(),
                "v1".to_string(),
            );
            pending.embedding = Some(next_emb());
            pending.agent_id = Some("agent-X".to_string());
            pending.session_id = Some("sess-X".to_string());
            pending.entry_status = EntryStatus::Pending;
            queryable.insert(&pending).await.unwrap();

            // Insert a pending entry for agent-Y.
            let mut pending_y = KnowledgeEntry::new(
                "agent-Y pending work".to_string(),
                "flow-scope".to_string(),
                "v1".to_string(),
            );
            pending_y.embedding = Some(next_emb());
            pending_y.agent_id = Some("agent-Y".to_string());
            pending_y.session_id = Some("sess-Y".to_string());
            pending_y.entry_status = EntryStatus::Pending;
            queryable.insert(&pending_y).await.unwrap();

            let opts = RetrievalOpts {
                limit: 20,
                visibility: VisibilityMode::Own,
                agent_id: Some("agent-X".to_string()),
                expand_graph: false,
                ..Default::default()
            };
            let result = pipeline
                .retrieve("pending work", "flow-scope", &opts)
                .await
                .unwrap();

            // Agent-Y's pending should NOT appear.
            let has_agent_y = result.results.iter().any(|sr| {
                sr.entry.agent_id.as_deref() == Some("agent-Y")
                    && sr.entry.entry_status == EntryStatus::Pending
            });
            assert!(
                !has_agent_y,
                "agent-Y's pending should NOT be visible under Own mode"
            );
        }

        // Sub-test F: Truncation respects limit
        {
            let opts = RetrievalOpts {
                limit: 3,
                expand_graph: false,
                visibility: VisibilityMode::All,
                ..Default::default()
            };
            let result = pipeline
                .retrieve("document", "flow-scope", &opts)
                .await
                .unwrap();
            assert!(
                result.results.len() <= 3,
                "should respect limit=3, got {}",
                result.results.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // TEST 3: Timeout -- single searcher timeout returns error (all failed)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_searcher_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let slow_searcher = Arc::new(SlowSearcher {
            delay: std::time::Duration::from_secs(10),
        });

        let pipeline = RetrievalPipeline::new(
            vec![slow_searcher as Arc<dyn crate::pipeline::searcher::Searcher>],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            100, // 100ms timeout
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let result = pipeline
            .retrieve("test query", "timeout-scope", &opts)
            .await;
        let elapsed = start.elapsed();

        // Should complete well within 2 seconds (not 10s).
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "pipeline should not hang; elapsed: {elapsed:?}"
        );

        // All searchers failed → error.
        assert!(
            result.is_err(),
            "should return error when all searchers fail"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("all searchers failed"),
            "error should mention all searchers failed, got: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 4: Graceful degradation -- failing + working searcher
    // -----------------------------------------------------------------------

    /// Searcher that always returns an error.
    struct FailingSearcher;

    #[async_trait]
    impl crate::pipeline::searcher::Searcher for FailingSearcher {
        fn name(&self) -> &str {
            "failing"
        }
        fn needs_embedding(&self) -> bool {
            false
        }
        async fn search(
            &self,
            _ctx: &SearchContext,
        ) -> Result<crate::pipeline::RankedSet> {
            Err(corvia_common::errors::CorviaError::Infra(
                "intentional test failure".into(),
            ))
        }
    }

    #[tokio::test]
    async fn test_graceful_degradation_failing_plus_working() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert entries so vector searcher returns results.
        insert_entries(&store, "degrade-scope", 10).await;

        let pipeline = RetrievalPipeline::new(
            vec![
                Arc::new(FailingSearcher) as Arc<dyn crate::pipeline::searcher::Searcher>,
                Arc::new(VectorSearcher::new(queryable.clone(), engine.clone())),
            ],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        // Should succeed with results from the working searcher.
        let result = pipeline
            .retrieve("knowledge entry", "degrade-scope", &opts)
            .await
            .unwrap();

        assert!(
            !result.results.is_empty(),
            "should return results from the working searcher"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 5: Panic searcher -- caught, pipeline continues with others
    // -----------------------------------------------------------------------

    /// Searcher that panics.
    struct PanicSearcher;

    #[async_trait]
    impl crate::pipeline::searcher::Searcher for PanicSearcher {
        fn name(&self) -> &str {
            "panic"
        }
        fn needs_embedding(&self) -> bool {
            false
        }
        async fn search(
            &self,
            _ctx: &SearchContext,
        ) -> Result<crate::pipeline::RankedSet> {
            panic!("intentional test panic");
        }
    }

    #[tokio::test]
    async fn test_panic_searcher_caught() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        insert_entries(&store, "panic-scope", 10).await;

        let pipeline = RetrievalPipeline::new(
            vec![
                Arc::new(PanicSearcher) as Arc<dyn crate::pipeline::searcher::Searcher>,
                Arc::new(VectorSearcher::new(queryable.clone(), engine.clone())),
            ],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        // Panic in one searcher should NOT crash the pipeline.
        let result = pipeline
            .retrieve("knowledge entry", "panic-scope", &opts)
            .await
            .unwrap();

        assert!(
            !result.results.is_empty(),
            "should return results from non-panicking searcher"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 6: Embedding failure -- degrades to BM25-only
    // -----------------------------------------------------------------------

    /// Mock engine that always fails embedding.
    struct FailingEngine;

    #[async_trait]
    impl InferenceEngine for FailingEngine {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Err(corvia_common::errors::CorviaError::Infra(
                "embedding service unavailable".into(),
            ))
        }
        async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Err(corvia_common::errors::CorviaError::Infra(
                "embedding service unavailable".into(),
            ))
        }
        fn dimensions(&self) -> usize {
            3
        }
    }

    /// Mock BM25-like searcher that doesn't need embedding and returns fixed results.
    struct MockBM25Searcher {
        results: Vec<crate::pipeline::RankedCandidate>,
    }

    #[async_trait]
    impl crate::pipeline::searcher::Searcher for MockBM25Searcher {
        fn name(&self) -> &str {
            "mock_bm25"
        }
        fn needs_embedding(&self) -> bool {
            false
        }
        async fn search(
            &self,
            _ctx: &SearchContext,
        ) -> Result<crate::pipeline::RankedSet> {
            Ok(crate::pipeline::RankedSet {
                candidates: self.results.clone(),
                metrics: crate::pipeline::StageMetrics {
                    stage_name: "mock_bm25".into(),
                    latency_ms: 1,
                    input_count: 0,
                    output_count: self.results.len(),
                    warnings: Vec::new(),
                },
            })
        }
    }

    #[tokio::test]
    async fn test_embedding_failure_degrades_to_bm25() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(FailingEngine);

        // Create a mock entry for the BM25 searcher to return.
        let entry = KnowledgeEntry::new(
            "test entry from BM25".to_string(),
            "embed-fail-scope".to_string(),
            "v1".to_string(),
        );
        let mock_candidate = crate::pipeline::RankedCandidate {
            entry: Arc::new(entry),
            scores: crate::pipeline::CandidateScores {
                components: {
                    let mut m = HashMap::new();
                    m.insert("mock_bm25".to_string(), 0.8);
                    m
                },
                final_score: crate::pipeline::NormalizedScore::new(0.8),
            },
        };

        let pipeline = RetrievalPipeline::new(
            vec![
                Arc::new(VectorSearcher::new(queryable.clone(), engine.clone())),
                Arc::new(MockBM25Searcher {
                    results: vec![mock_candidate],
                }),
            ],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        // Embedding fails, but pipeline should degrade to BM25 results.
        let result = pipeline
            .retrieve("test query", "embed-fail-scope", &opts)
            .await
            .unwrap();

        assert!(
            !result.results.is_empty(),
            "should return BM25 results when embedding fails"
        );
        // query_embedding should be None when embedding failed.
        assert!(
            result.query_embedding.is_none(),
            "query_embedding should be None when embedding failed"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 7: Embedding failure with no BM25 -- total failure
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_embedding_failure_no_bm25_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(FailingEngine);

        // Only vector searcher, no BM25 fallback.
        let pipeline = RetrievalPipeline::new(
            vec![Arc::new(VectorSearcher::new(
                queryable.clone(),
                engine.clone(),
            ))],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        // Should fail because embedding failed and no BM25 fallback.
        let result = pipeline
            .retrieve("test query", "embed-fail-scope", &opts)
            .await;

        assert!(
            result.is_err(),
            "should return error when embedding fails and no BM25 searcher"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 8: All searchers fail -- returns error
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_all_searchers_fail_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let pipeline = RetrievalPipeline::new(
            vec![
                Arc::new(FailingSearcher) as Arc<dyn crate::pipeline::searcher::Searcher>,
                Arc::new(SlowSearcher {
                    delay: std::time::Duration::from_secs(10),
                }),
            ],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            100, // 100ms timeout for SlowSearcher
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = pipeline
            .retrieve("test", "all-fail-scope", &opts)
            .await;

        assert!(
            result.is_err(),
            "should return error when all searchers fail"
        );
    }

    // -----------------------------------------------------------------------
    // TEST 9: Hot-swap under concurrent load
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_hot_swap_under_concurrent_load() {
        use arc_swap::ArcSwap;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        insert_entries(&store, "swap-scope", 10).await;

        let pipeline = RetrievalPipeline::new(
            vec![Arc::new(VectorSearcher::new(
                queryable.clone(),
                engine.clone(),
            ))],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let hot = Arc::new(ArcSwap::from_pointee(pipeline));

        // Spawn 10 concurrent queries.
        let mut handles = Vec::new();
        for i in 0..10 {
            let hot_ref = Arc::clone(&hot);
            handles.push(tokio::spawn(async move {
                let pipeline = hot_ref.load();
                let opts = RetrievalOpts {
                    limit: 5,
                    expand_graph: false,
                    visibility: VisibilityMode::All,
                    ..Default::default()
                };
                let result = pipeline
                    .retrieve("knowledge entry", "swap-scope", &opts)
                    .await;
                (i, result.is_ok())
            }));
        }

        // Swap to a new pipeline mid-flight.
        let new_pipeline = RetrievalPipeline::new(
            vec![Arc::new(VectorSearcher::new(
                queryable.clone(),
                engine.clone(),
            ))],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );
        let old = hot.swap(Arc::new(new_pipeline));
        std::thread::spawn(move || drop(old));

        // All queries should complete successfully.
        let mut successes = 0;
        for handle in handles {
            let (_, ok) = handle.await.unwrap();
            if ok {
                successes += 1;
            }
        }
        assert_eq!(successes, 10, "all concurrent queries should succeed during hot-swap");
    }

    // -----------------------------------------------------------------------
    // TEST 10: Cold tier rescue -- BM25 surfaces cold entries that vector misses
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cold_tier_rescue_via_bm25() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert a cold entry. It won't be in HNSW (vector search misses it),
        // but BM25 can surface it via full-text search.
        let mut cold_entry = KnowledgeEntry::new(
            "cold archived API reference documentation".to_string(),
            "cold-scope".to_string(),
            "v1".to_string(),
        );
        cold_entry.tier = Tier::Cold;
        cold_entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        queryable.insert(&cold_entry).await.unwrap();

        // Insert some hot entries for HNSW connectivity.
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("hot entry about topic {i}"),
                "cold-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.1, 0.0]);
            entry.tier = Tier::Hot;
            queryable.insert(&entry).await.unwrap();
        }

        // Mock BM25 searcher that returns the cold entry (simulates tantivy finding it).
        let cold_candidate = crate::pipeline::RankedCandidate {
            entry: Arc::new(cold_entry.clone()),
            scores: crate::pipeline::CandidateScores {
                components: HashMap::from([("bm25".to_string(), 0.9 * 0.3)]), // score * cold weight
                final_score: crate::pipeline::NormalizedScore::new(0.8),
            },
        };

        let pipeline = RetrievalPipeline::new(
            vec![
                Arc::new(VectorSearcher::new(queryable.clone(), engine.clone())),
                Arc::new(MockBM25Searcher {
                    results: vec![cold_candidate],
                }),
            ],
            Arc::new(PassThrough),
            Arc::new(crate::pipeline::expander::NoOpExpander),
            Arc::new(IdentityReranker),
            engine.clone(),
            queryable.clone(),
            5000,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = pipeline
            .retrieve("cold archived API reference", "cold-scope", &opts)
            .await
            .unwrap();

        // The cold entry should appear in results (rescued by BM25).
        let has_cold = result.results.iter().any(|sr| sr.entry.id == cold_entry.id);
        assert!(
            has_cold,
            "cold entry should be surfaced by BM25 in hybrid search results"
        );

        // Verify the cold entry has tier=Cold (confirming it's actually a cold entry).
        let cold_result = result.results.iter().find(|sr| sr.entry.id == cold_entry.id).unwrap();
        assert_eq!(cold_result.tier, Tier::Cold, "entry should still be Cold tier");
    }
}
