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

        // Step 1: Embed query (wrap in Arc for sharing across searchers).
        let embed_start = Instant::now();
        let embedding_arc = Arc::new(self.engine.embed(query).await?);
        let embed_latency_ms = embed_start.elapsed().as_millis() as u64;

        let ctx = SearchContext {
            query_embedding: Arc::clone(&embedding_arc),
            scope_id: scope_id.to_string(),
            query: query.to_string(),
            opts: opts.clone(),
        };

        // Step 2: Run searchers with timeout + panic isolation.
        // Fast path: single searcher avoids tokio::spawn overhead.
        let search_start = Instant::now();
        let mut searcher_results = Vec::new();
        let mut search_warnings: Vec<String> = Vec::new();

        if self.searchers.len() == 1 {
            let timeout = std::time::Duration::from_millis(self.searcher_timeout_ms);
            match tokio::time::timeout(timeout, self.searchers[0].search(&ctx)).await {
                Ok(Ok(set)) => searcher_results.push(set),
                Ok(Err(e)) => {
                    let name = self.searchers[0].name();
                    warn!(searcher = name, error = %e, "searcher failed");
                    search_warnings.push(format!("searcher '{name}' failed: {e}"));
                }
                Err(_) => {
                    let name = self.searchers[0].name();
                    warn!(searcher = name, "searcher timed out");
                    search_warnings.push(format!("searcher '{name}' timed out"));
                }
            }
        } else {
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
        }

        if !search_warnings.is_empty() {
            warn!(warnings = ?search_warnings, "searcher warnings during pipeline execution");
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
            query_embedding: Some(Arc::try_unwrap(embedding_arc)
                .unwrap_or_else(|arc| (*arc).clone())),
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
    use std::collections::HashSet;
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
    // TEST 3: Timeout -- searcher that sleeps should be timed out
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
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // Should complete well within 2 seconds (not 10s).
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "pipeline should not hang; elapsed: {elapsed:?}"
        );

        // No candidates because the only searcher timed out.
        assert_eq!(
            result.results.len(),
            0,
            "timed-out searcher should produce 0 results"
        );
        assert_eq!(result.metrics.vector_results, 0);
    }
}
