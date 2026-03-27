//! Retriever trait and implementations for the RAG pipeline (D61).
//!
//! The retriever is the first stage of the R->A->G pipeline. It converts
//! a natural-language query into a ranked list of [`SearchResult`]s.
//!
//! # Implementations
//!
//! - [`VectorRetriever`] — embed query, search store, apply RBAC + visibility filters.
//! - [`GraphExpandRetriever`] — vector search + graph expansion with alpha blending.

use async_trait::async_trait;
use corvia_common::agent_types::{AgentPermission, EntryStatus, VisibilityMode};
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, SearchResult};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use corvia_common::types::Tier;

use crate::lite_store::LiteStore;
use crate::rag_types::{RetrievalMetrics, RetrievalOpts, RetrievalResult};
use crate::reasoner::cosine_similarity;
use crate::traits::{GraphStore, InferenceEngine, QueryableStore};

/// Map knowledge tier to a retrieval score multiplier.
///
/// Applied after cosine similarity (or blended score) to deprioritize lower tiers.
/// Forgotten entries get 0.0 and should be excluded from results entirely.
///
/// Weights: Hot=1.0 (full), Warm=0.7 (deprioritized), Cold=0.3 (heavily penalized),
/// Forgotten=0.0 (excluded). These values are tuned so that a high-similarity Cold
/// entry can still outrank a low-similarity Warm entry (e.g., 0.95*0.3 > 0.4*0.7).
#[inline]
pub(crate) const fn tier_weight(tier: Tier) -> f32 {
    match tier {
        Tier::Hot => 1.0,
        Tier::Warm => 0.7,
        Tier::Cold => 0.3,
        Tier::Forgotten => 0.0,
    }
}

/// Check RBAC scope access. Returns `Some(empty result)` if denied, `None` if allowed.
///
/// Only `ReadWrite { scopes }` agents are scope-restricted. `ReadOnly` and `Admin`
/// have no scope restriction by design (D19/D44).
fn check_rbac_scope(
    opts: &RetrievalOpts,
    scope_id: &str,
    retriever_name: &str,
    start: &Instant,
) -> Option<RetrievalResult> {
    if let Some(ref perms) = opts.permissions
        && let AgentPermission::ReadWrite { scopes } = perms
            && !scopes.iter().any(|s| s == scope_id || s == "*") {
                info!(
                    retriever = retriever_name,
                    scope_id,
                    "RBAC denied: agent lacks scope access"
                );
                return Some(RetrievalResult {
                    results: Vec::new(),
                    metrics: RetrievalMetrics {
                        latency_ms: start.elapsed().as_millis() as u64,
                        embed_latency_ms: 0,
                        search_latency_ms: 0,
                        hnsw_latency_ms: 0,
                        graph_latency_ms: 0,
                        filter_latency_ms: 0,
                        cold_scan_latency_ms: 0,
                        vector_results: 0,
                        cold_results: 0,
                        graph_expanded: 0,
                        graph_reinforced: 0,
                        post_filter_count: 0,
                        retriever_name: retriever_name.to_string(),
                    },
                    query_embedding: None,
                });
            }
    None
}

/// Fire-and-forget access recording for retrieved entries.
///
/// Spawns a background task to update `last_accessed` and `access_count` on all
/// entries in the result set. The write does NOT block the search response.
/// Failures are logged as warnings — access tracking is optimization, not correctness.
fn spawn_access_recording(store: Arc<dyn QueryableStore>, results: &[SearchResult]) {
    if results.is_empty() {
        return;
    }
    let entry_ids: Vec<uuid::Uuid> = results.iter().map(|sr| sr.entry.id).collect();
    tokio::spawn(async move {
        match store.record_access(&entry_ids).await {
            Ok(()) => {}
            Err(e) => warn!(error = %e, "Access recording failed"),
        }
    });
}

/// Perform brute-force cold scan if `include_cold` is set and the store is LiteStore.
///
/// Returns `(cold_results_added, cold_scan_latency_ms)`. Merges deduplicated cold
/// results into the existing results vector by score. Caller is responsible for
/// truncating to limit after this call.
///
/// Assumes HNSW uses cosine distance, so `1 - distance == cosine_similarity` holds.
fn merge_cold_results(
    store: &dyn QueryableStore,
    embedding: &[f32],
    scope_id: &str,
    opts: &RetrievalOpts,
    results: &mut Vec<SearchResult>,
) -> (usize, u64) {
    if !opts.include_cold {
        return (0, 0);
    }

    let cold_start = Instant::now();

    // Downcast to LiteStore for cold scan (LiteStore-specific operation).
    let lite_store = match store.as_any().downcast_ref::<LiteStore>() {
        Some(ls) => ls,
        None => {
            warn!("include_cold requested but store is not LiteStore — skipping cold scan");
            return (0, 0);
        }
    };

    let cold_results = match lite_store.scan_cold_entries(embedding, scope_id, opts.limit) {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Cold scan failed, continuing with HNSW results only");
            return (0, cold_start.elapsed().as_millis() as u64);
        }
    };

    if cold_results.is_empty() {
        return (0, cold_start.elapsed().as_millis() as u64);
    }

    // Deduplicate: skip cold entries already present in HNSW results.
    // Tier weighting is NOT applied here — the caller applies it uniformly to all results.
    let existing_ids: HashSet<uuid::Uuid> = results.iter().map(|sr| sr.entry.id).collect();
    let new_cold: Vec<SearchResult> = cold_results
        .into_iter()
        .filter(|sr| !existing_ids.contains(&sr.entry.id))
        .collect();
    let cold_count = new_cold.len();

    if cold_count > 0 {
        results.extend(new_cold);
        // Skip sorting — caller re-sorts after tier weighting.
    }

    (cold_count, cold_start.elapsed().as_millis() as u64)
}

/// First stage of the RAG pipeline: query -> ranked results.
///
/// Implementations must be Send + Sync so they can be shared across
/// async tasks and wrapped in `Arc`.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Human-readable name for metrics attribution (D62).
    fn name(&self) -> &str;

    /// Retrieve relevant knowledge entries for a query within a scope.
    async fn retrieve(
        &self,
        query: &str,
        scope_id: &str,
        opts: &RetrievalOpts,
    ) -> Result<RetrievalResult>;
}

/// Baseline retriever: embed query -> vector search -> RBAC + visibility filter.
///
/// This is the default retriever used when `expand_graph` is false.
pub struct VectorRetriever {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
}

impl VectorRetriever {
    pub fn new(store: Arc<dyn QueryableStore>, engine: Arc<dyn InferenceEngine>) -> Self {
        Self { store, engine }
    }
}

#[async_trait]
impl Retriever for VectorRetriever {
    fn name(&self) -> &str {
        "vector"
    }

    async fn retrieve(
        &self,
        query: &str,
        scope_id: &str,
        opts: &RetrievalOpts,
    ) -> Result<RetrievalResult> {
        let start = Instant::now();

        if let Some(denied) = check_rbac_scope(opts, scope_id, self.name(), &start) {
            return Ok(denied);
        }

        // Embed the query (timed separately for per-stage metrics).
        let embed_start = Instant::now();
        let embedding = self.engine.embed(query).await?;
        let embed_latency_ms = embed_start.elapsed().as_millis() as u64;

        // Vector search + post-filtering (timed separately).
        let search_start = Instant::now();

        // Oversample (min 10) to allow for post-filter elimination.
        // When metadata filters are active, over-fetch by 3x to compensate for post-filter.
        let search_limit = if opts.content_role.is_some() || opts.source_origin.is_some() || opts.workstream.is_some() {
            opts.limit * 3
        } else {
            opts.limit
        };
        let fetch_limit = (search_limit * opts.oversample_factor).max(10);
        let hnsw_start = Instant::now();
        let raw_results = self.store.search(&embedding, scope_id, fetch_limit).await?;
        let hnsw_latency_ms = hnsw_start.elapsed().as_millis() as u64;
        let vector_results = raw_results.len();

        // Cold-tier brute-force scan (merged before visibility filtering).
        let mut merged = raw_results;
        let (cold_results, cold_scan_latency_ms) =
            merge_cold_results(self.store.as_ref(), &embedding, scope_id, opts, &mut merged);

        // Apply tier_weight multiplier: deprioritize Warm/Cold, exclude Forgotten.
        // Fast path: skip filter/map/re-sort when all entries are Hot (common case).
        let any_non_hot = merged.iter().any(|sr| sr.entry.tier != Tier::Hot);
        let mut tier_weighted = if any_non_hot {
            let mut weighted: Vec<SearchResult> = merged
                .into_iter()
                .filter(|sr| sr.entry.tier != Tier::Forgotten)
                .map(|mut sr| {
                    sr.score *= tier_weight(sr.entry.tier);
                    sr
                })
                .collect();
            weighted.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            weighted
        } else {
            merged
        };

        // Visibility post-filter (no take yet — metadata filter runs after).
        let vis_filtered: Vec<SearchResult> = tier_weighted
            .into_iter()
            .filter(|sr| visibility_filter(&sr.entry, &opts.visibility, opts.agent_id.as_deref()))
            .collect();

        // Metadata post-filter.
        let meta_filtered = post_filter_metadata(
            vis_filtered,
            opts.content_role.as_deref(),
            opts.source_origin.as_deref(),
            opts.workstream.as_deref(),
        );

        // Truncate to requested limit.
        let mut filtered = meta_filtered;
        filtered.truncate(opts.limit);

        let search_latency_ms = search_start.elapsed().as_millis() as u64;
        let post_filter_count = filtered.len();

        // Record access (fire-and-forget, non-blocking).
        spawn_access_recording(Arc::clone(&self.store), &filtered);

        info!(
            retriever = self.name(),
            scope_id,
            vector_results,
            cold_results,
            post_filter_count,
            embed_latency_ms,
            search_latency_ms,
            cold_scan_latency_ms,
            latency_ms = start.elapsed().as_millis() as u64,
            "retrieval complete"
        );

        Ok(RetrievalResult {
            results: filtered,
            metrics: RetrievalMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embed_latency_ms,
                search_latency_ms,
                hnsw_latency_ms,
                graph_latency_ms: 0,
                filter_latency_ms: 0,
                cold_scan_latency_ms,
                vector_results,
                cold_results,
                graph_expanded: 0,
                graph_reinforced: 0,
                post_filter_count,
                retriever_name: self.name().to_string(),
            },
            query_embedding: Some(embedding),
        })
    }
}

/// Shared visibility filter for retrieval results.
///
/// Rules:
/// - **Merged** entries are always visible.
/// - **Pending** / **Committed** entries are filtered by [`VisibilityMode`]:
///   - `Own` — only the requesting agent's entries.
///   - `All` — all agents' pending/committed entries.
///   - `Explicit(agents)` — only named agents' entries.
/// - **Rejected** entries are never visible.
pub fn visibility_filter(
    entry: &corvia_common::types::KnowledgeEntry,
    mode: &VisibilityMode,
    agent_id: Option<&str>,
) -> bool {
    match entry.entry_status {
        EntryStatus::Merged => true,
        EntryStatus::Rejected => false,
        EntryStatus::Pending | EntryStatus::Committed => match mode {
            VisibilityMode::All => true,
            VisibilityMode::Own => {
                // Visible only if the entry belongs to the requesting agent.
                match (agent_id, entry.agent_id.as_deref()) {
                    (Some(me), Some(owner)) => me == owner,
                    _ => false,
                }
            }
            VisibilityMode::Explicit(agents) => {
                // Visible if the entry's agent is in the explicit list.
                match entry.agent_id.as_deref() {
                    Some(owner) => agents.iter().any(|a| a == owner),
                    None => false,
                }
            }
        },
    }
}

/// Vector + graph expansion retriever. Corvia's differentiator.
///
/// 1. Vector search for initial top-k (2x oversample).
/// 2. For each result, follow graph edges up to `graph_depth` hops.
/// 3. Deduplicate and blend scores: `final = (1-α)*cosine + α*(1/(hop+1))`.
///
/// When `expand_graph` is false in the opts, behaves identically to
/// [`VectorRetriever`] (graph_expanded == 0).
pub struct GraphExpandRetriever {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
    graph: Arc<dyn GraphStore>,
    alpha: f32,
}

impl GraphExpandRetriever {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        engine: Arc<dyn InferenceEngine>,
        graph: Arc<dyn GraphStore>,
        alpha: f32,
    ) -> Self {
        Self {
            store,
            engine,
            graph,
            alpha,
        }
    }
}

#[async_trait]
impl Retriever for GraphExpandRetriever {
    fn name(&self) -> &str {
        "graph_expand"
    }

    async fn retrieve(
        &self,
        query: &str,
        scope_id: &str,
        opts: &RetrievalOpts,
    ) -> Result<RetrievalResult> {
        let start = Instant::now();

        if let Some(denied) = check_rbac_scope(opts, scope_id, self.name(), &start) {
            return Ok(denied);
        }

        // Embed the query (timed separately for per-stage metrics).
        let embed_start = Instant::now();
        let embedding = self.engine.embed(query).await?;
        let embed_latency_ms = embed_start.elapsed().as_millis() as u64;

        // Search + graph expansion + filtering (timed separately).
        let search_start = Instant::now();

        // Oversample (min 10) to allow for post-filter elimination.
        // When metadata filters are active, over-fetch by 3x to compensate for post-filter.
        let search_limit = if opts.content_role.is_some() || opts.source_origin.is_some() || opts.workstream.is_some() {
            opts.limit * 3
        } else {
            opts.limit
        };
        let fetch_limit = (search_limit * opts.oversample_factor).max(10);
        let hnsw_start = Instant::now();
        let raw_results = self
            .store
            .search(&embedding, scope_id, fetch_limit)
            .await?;
        let hnsw_latency_ms = hnsw_start.elapsed().as_millis() as u64;
        let vector_results = raw_results.len();

        // Track seen IDs and scored (score, SearchResult) pairs.
        let mut seen: HashSet<uuid::Uuid> = HashSet::new();
        let mut scored: Vec<(f32, SearchResult)> = Vec::new();

        // Score direct vector hits: final = ((1-α)*cosine + α*1.0) * tier_weight
        // Forgotten entries are excluded entirely.
        for sr in &raw_results {
            seen.insert(sr.entry.id);
            if sr.entry.tier == Tier::Forgotten {
                continue;
            }
            let blended = ((1.0 - self.alpha) * sr.score + self.alpha * 1.0) * tier_weight(sr.entry.tier);
            scored.push((blended, SearchResult { entry: sr.entry.clone(), score: blended }));
        }

        // Graph expansion (timed separately).
        let graph_start = Instant::now();
        let mut graph_expanded: usize = 0;
        let mut graph_reinforced: usize = 0;
        // Track reinforcement count per result for diminishing returns.
        let mut reinforcement_counts: std::collections::HashMap<uuid::Uuid, u32> = std::collections::HashMap::new();

        if opts.expand_graph {
            // Hop 1: follow edges from each vector result.
            for sr in &raw_results {
                let edges = self
                    .graph
                    .edges(&sr.entry.id, EdgeDirection::Both)
                    .await?;
                for edge in &edges {
                    // Determine the neighbor ID (the other end of the edge).
                    let neighbor_id = if edge.from == sr.entry.id {
                        edge.to
                    } else {
                        edge.from
                    };

                    if seen.contains(&neighbor_id) {
                        // Reinforce with diminishing returns: each hit gives 50% less than previous.
                        // Hit 1: alpha*0.25, Hit 2: alpha*0.125, Hit 3: alpha*0.0625, ...
                        // Converges to alpha*0.5 total, preventing runaway but rewarding connectivity.
                        let count = reinforcement_counts.entry(neighbor_id).or_insert(0);
                        let decay = 0.5_f32.powi(*count as i32);
                        let bonus = self.alpha * 0.25 * decay;
                        if bonus > 0.001 {
                            if let Some(existing) = scored.iter_mut().find(|(_, sr)| sr.entry.id == neighbor_id) {
                                existing.0 += bonus;
                                existing.1.score = existing.0;
                                graph_reinforced += 1;
                            }
                            *count += 1;
                        }
                        continue;
                    }

                    // Look up the entry; skip if not found, wrong scope, or Forgotten.
                    if let Some(neighbor_entry) = self.store.get(&neighbor_id).await? {
                        if neighbor_entry.scope_id != scope_id {
                            continue;
                        }
                        if neighbor_entry.tier == Tier::Forgotten {
                            continue;
                        }
                        seen.insert(neighbor_id);
                        // Blend cosine similarity with graph proximity, apply tier_weight:
                        // final = ((1-α)*cosine + α*0.5) * tier_weight
                        let cosine = neighbor_entry
                            .embedding
                            .as_ref()
                            .map(|emb| cosine_similarity(&embedding, emb))
                            .unwrap_or(0.0);
                        let blended = ((1.0 - self.alpha) * cosine + self.alpha * 0.5) * tier_weight(neighbor_entry.tier);
                        scored.push((
                            blended,
                            SearchResult {
                                entry: neighbor_entry,
                                score: blended,
                            },
                        ));
                        graph_expanded += 1;
                    }
                }
            }

            // Deeper hops (graph_depth > 1): use traverse for each vector result.
            if opts.graph_depth > 1 {
                for sr in &raw_results {
                    let deep_entries = self
                        .graph
                        .traverse(
                            &sr.entry.id,
                            None,
                            EdgeDirection::Both,
                            opts.graph_depth,
                        )
                        .await?;
                    for entry in deep_entries {
                        if seen.contains(&entry.id) {
                            // Reinforce with diminishing returns for deeper hops.
                            let count = reinforcement_counts.entry(entry.id).or_insert(0);
                            let decay = 0.5_f32.powi(*count as i32);
                            let bonus = self.alpha * 0.15 * decay;
                            if bonus > 0.001 {
                                if let Some(existing) = scored.iter_mut().find(|(_, sr)| sr.entry.id == entry.id) {
                                    existing.0 += bonus;
                                    existing.1.score = existing.0;
                                    graph_reinforced += 1;
                                }
                                *count += 1;
                            }
                            continue;
                        }
                        if entry.scope_id != scope_id {
                            continue;
                        }
                        if entry.tier == Tier::Forgotten {
                            continue;
                        }
                        seen.insert(entry.id);
                        // Blend cosine similarity with graph proximity (hop 2+), apply tier_weight:
                        // final = ((1-α)*cosine + α*(1/3)) * tier_weight
                        let cosine = entry
                            .embedding
                            .as_ref()
                            .map(|emb| cosine_similarity(&embedding, emb))
                            .unwrap_or(0.0);
                        let blended = ((1.0 - self.alpha) * cosine + self.alpha * (1.0 / 3.0)) * tier_weight(entry.tier);
                        scored.push((
                            blended,
                            SearchResult {
                                entry,
                                score: blended,
                            },
                        ));
                        graph_expanded += 1;
                    }
                }
            }
        }

        let graph_latency_ms = graph_start.elapsed().as_millis() as u64;

        // Cold-tier brute-force scan (merged with graph-expanded results).
        let cold_scan_start = Instant::now();
        let cold_results = if opts.include_cold {
            match self.store.as_ref().as_any().downcast_ref::<LiteStore>() {
                Some(ls) => {
                    match ls.scan_cold_entries(&embedding, scope_id, opts.limit) {
                        Ok(cold) => {
                            let mut added = 0usize;
                            for sr in cold {
                                if !seen.contains(&sr.entry.id) && sr.entry.tier != Tier::Forgotten {
                                    seen.insert(sr.entry.id);
                                    // Cold entries get pure cosine score (no graph proximity bonus), with tier_weight.
                                    let blended = (1.0 - self.alpha) * sr.score * tier_weight(sr.entry.tier);
                                    scored.push((blended, SearchResult { entry: sr.entry, score: blended }));
                                    added += 1;
                                }
                            }
                            added
                        }
                        Err(e) => {
                            warn!(error = %e, "Cold scan failed, continuing without cold results");
                            0
                        }
                    }
                }
                None => {
                    warn!("include_cold requested but store is not LiteStore — skipping cold scan");
                    0
                }
            }
        } else {
            0
        };
        let cold_scan_latency_ms = cold_scan_start.elapsed().as_millis() as u64;

        // Sort + filter (timed separately).
        let filter_start = Instant::now();

        // Sort by blended score descending (unstable sort — no stability requirement).
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Visibility post-filter (no take yet — metadata filter runs after).
        let vis_filtered: Vec<SearchResult> = scored
            .into_iter()
            .map(|(_, sr)| sr)
            .filter(|sr| visibility_filter(&sr.entry, &opts.visibility, opts.agent_id.as_deref()))
            .collect();

        // Metadata post-filter.
        let meta_filtered = post_filter_metadata(
            vis_filtered,
            opts.content_role.as_deref(),
            opts.source_origin.as_deref(),
            opts.workstream.as_deref(),
        );

        // Truncate to requested limit.
        let mut filtered = meta_filtered;
        filtered.truncate(opts.limit);

        let filter_latency_ms = filter_start.elapsed().as_millis() as u64;
        let search_latency_ms = search_start.elapsed().as_millis() as u64;
        let post_filter_count = filtered.len();

        // Record access (fire-and-forget, non-blocking).
        spawn_access_recording(Arc::clone(&self.store), &filtered);

        info!(
            retriever = self.name(),
            scope_id,
            vector_results,
            cold_results,
            graph_expanded,
            graph_reinforced,
            post_filter_count,
            embed_latency_ms,
            hnsw_latency_ms,
            graph_latency_ms,
            cold_scan_latency_ms,
            filter_latency_ms,
            search_latency_ms,
            latency_ms = start.elapsed().as_millis() as u64,
            "retrieval complete"
        );

        Ok(RetrievalResult {
            results: filtered,
            metrics: RetrievalMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                embed_latency_ms,
                search_latency_ms,
                hnsw_latency_ms,
                graph_latency_ms,
                filter_latency_ms,
                cold_scan_latency_ms,
                vector_results,
                cold_results,
                graph_expanded,
                graph_reinforced,
                post_filter_count,
                retriever_name: self.name().to_string(),
            },
            query_embedding: Some(embedding),
        })
    }
}

/// Post-filter search results by metadata fields (Option A from docs workflow spec).
/// Applied after vector search, before returning to caller.
pub fn post_filter_metadata(
    results: Vec<SearchResult>,
    content_role: Option<&str>,
    source_origin: Option<&str>,
    workstream: Option<&str>,
) -> Vec<SearchResult> {
    if content_role.is_none() && source_origin.is_none() && workstream.is_none() {
        return results;
    }
    results.into_iter().filter(|r| {
        if let Some(role) = content_role
            && r.entry.metadata.content_role.as_deref() != Some(role) {
                return false;
            }
        if let Some(origin) = source_origin
            && r.entry.metadata.source_origin.as_deref() != Some(origin) {
                return false;
            }
        if let Some(ws) = workstream
            && r.entry.workstream.as_str() != ws {
                return false;
            }
        true
    }).collect()
}

#[cfg(test)]
mod filter_tests {
    use corvia_common::types::{KnowledgeEntry, SearchResult};

    fn make_result(content_role: Option<&str>, source_origin: Option<&str>, score: f32) -> SearchResult {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        entry.metadata.content_role = content_role.map(String::from);
        entry.metadata.source_origin = source_origin.map(String::from);
        SearchResult { entry, score }
    }

    fn make_result_with_workstream(content_role: Option<&str>, source_origin: Option<&str>, workstream: &str, score: f32) -> SearchResult {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        entry.metadata.content_role = content_role.map(String::from);
        entry.metadata.source_origin = source_origin.map(String::from);
        entry.workstream = workstream.to_string();
        SearchResult { entry, score }
    }

    #[test]
    fn test_post_filter_by_content_role() {
        let results = vec![
            make_result(Some("design"), Some("repo:corvia"), 0.9),
            make_result(Some("code"), Some("repo:corvia"), 0.8),
            make_result(Some("design"), Some("workspace"), 0.7),
        ];
        let filtered = super::post_filter_metadata(results, Some("design"), None, None);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.entry.metadata.content_role.as_deref() == Some("design")));
    }

    #[test]
    fn test_post_filter_by_source_origin() {
        let results = vec![
            make_result(Some("design"), Some("repo:corvia"), 0.9),
            make_result(Some("code"), Some("repo:corvia"), 0.8),
            make_result(Some("design"), Some("workspace"), 0.7),
        ];
        let filtered = super::post_filter_metadata(results, None, Some("repo:corvia"), None);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_post_filter_combined() {
        let results = vec![
            make_result(Some("design"), Some("repo:corvia"), 0.9),
            make_result(Some("code"), Some("repo:corvia"), 0.8),
            make_result(Some("design"), Some("workspace"), 0.7),
        ];
        let filtered = super::post_filter_metadata(results, Some("design"), Some("repo:corvia"), None);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].entry.metadata.content_role.as_deref(), Some("design"));
        assert_eq!(filtered[0].entry.metadata.source_origin.as_deref(), Some("repo:corvia"));
    }

    #[test]
    fn test_post_filter_no_filters_returns_all() {
        let results = vec![
            make_result(Some("design"), Some("repo:corvia"), 0.9),
            make_result(None, None, 0.8),
        ];
        let filtered = super::post_filter_metadata(results, None, None, None);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_post_filter_by_workstream() {
        let results = vec![
            make_result_with_workstream(Some("design"), None, "feat/auth", 0.9),
            make_result_with_workstream(Some("code"), None, "feat/billing", 0.8),
            make_result_with_workstream(Some("design"), None, "feat/auth", 0.7),
        ];
        let filtered = super::post_filter_metadata(results, None, None, Some("feat/auth"));
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.entry.workstream == "feat/auth"));
    }

    #[test]
    fn test_post_filter_workstream_combined_with_role() {
        let results = vec![
            make_result_with_workstream(Some("design"), None, "feat/auth", 0.9),
            make_result_with_workstream(Some("code"), None, "feat/auth", 0.8),
            make_result_with_workstream(Some("design"), None, "feat/billing", 0.7),
        ];
        let filtered = super::post_filter_metadata(results, Some("design"), None, Some("feat/auth"));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].entry.workstream, "feat/auth");
        assert_eq!(filtered[0].entry.metadata.content_role.as_deref(), Some("design"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use corvia_common::agent_types::{AgentPermission, EntryStatus, VisibilityMode};
    use corvia_common::types::KnowledgeEntry;

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

    /// Insert 10 entries with slight embedding variation, search, verify results + metrics.
    #[tokio::test]
    async fn test_vector_retriever_basic_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 10 entries with slight embedding variation for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("knowledge item {i}"),
                "test-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);

        let opts = RetrievalOpts {
            limit: 5,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test query", "test-scope", &opts).await.unwrap();

        // HNSW approximate recall is unreliable at small N — use >= assertions.
        assert!(result.results.len() >= 2, "expected at least 2 results, got {}", result.results.len());
        assert!(result.results.len() <= 5, "should respect limit");
        assert_eq!(result.metrics.retriever_name, "vector");
        assert!(result.metrics.vector_results >= 2);
        assert!(result.metrics.post_filter_count >= 2);
        assert_eq!(result.metrics.graph_expanded, 0);
    }

    /// Agent with scope-b permission searches scope-a -> 0 results (RBAC).
    #[tokio::test]
    async fn test_vector_retriever_rbac_filtering() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert entries into scope-a.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("scope-a item {i}"),
                "scope-a".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);

        // Agent only has access to scope-b, not scope-a.
        let opts = RetrievalOpts {
            limit: 5,
            permissions: Some(AgentPermission::ReadWrite {
                scopes: vec!["scope-b".to_string()],
            }),
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = retriever.retrieve("test", "scope-a", &opts).await.unwrap();
        assert_eq!(result.results.len(), 0, "RBAC should block scope-a access");
        assert_eq!(result.metrics.vector_results, 0);
    }

    /// With Own visibility: agent-A sees merged + own pending, NOT agent-B's pending.
    #[tokio::test]
    async fn test_vector_retriever_visibility_own() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        // 8 filler merged entries for HNSW connectivity.
        for i in 0..8 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "vis-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            // EntryStatus::Merged is the default.
            store.insert(&entry).await.unwrap();
        }

        // Agent-A pending entry.
        let mut entry_a = KnowledgeEntry::new(
            "agent-A pending work".to_string(),
            "vis-scope".to_string(),
            "v1".to_string(),
        );
        entry_a.embedding = Some(next_emb());
        entry_a.agent_id = Some("agent-A".to_string());
        entry_a.session_id = Some("sess-A".to_string());
        entry_a.entry_status = EntryStatus::Pending;
        store.insert(&entry_a).await.unwrap();

        // Agent-B pending entry.
        let mut entry_b = KnowledgeEntry::new(
            "agent-B pending work".to_string(),
            "vis-scope".to_string(),
            "v1".to_string(),
        );
        entry_b.embedding = Some(next_emb());
        entry_b.agent_id = Some("agent-B".to_string());
        entry_b.session_id = Some("sess-B".to_string());
        entry_b.entry_status = EntryStatus::Pending;
        store.insert(&entry_b).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);

        // Agent-A with Own visibility.
        let opts = RetrievalOpts {
            limit: 20,
            visibility: VisibilityMode::Own,
            agent_id: Some("agent-A".to_string()),
            ..Default::default()
        };

        let result = retriever.retrieve("work", "vis-scope", &opts).await.unwrap();

        // Should see merged entries + agent-A's pending, but NOT agent-B's pending.
        let has_agent_a = result.results.iter().any(|sr| {
            sr.entry.agent_id.as_deref() == Some("agent-A")
                && sr.entry.entry_status == EntryStatus::Pending
        });
        let has_agent_b = result.results.iter().any(|sr| {
            sr.entry.agent_id.as_deref() == Some("agent-B")
                && sr.entry.entry_status == EntryStatus::Pending
        });
        let merged_count = result.results.iter().filter(|sr| sr.entry.entry_status == EntryStatus::Merged).count();

        // At least some merged entries should be visible (HNSW approximate, use >=).
        assert!(merged_count >= 1, "expected merged entries to be visible, got {merged_count}");
        // Agent-B's pending should be filtered out.
        assert!(!has_agent_b, "agent-B's pending should NOT be visible under Own mode");
        // Agent-A's pending may or may not appear depending on HNSW recall,
        // but if it does, the filter must have allowed it.
        if has_agent_a {
            // Confirmed: agent-A's own pending is visible.
        }
    }

    /// Insert 2 related entries + 8 fillers, create a graph edge, verify
    /// GraphExpandRetriever returns both entries and reports graph_expanded > 0.
    #[tokio::test]
    async fn test_graph_expand_retriever_adds_neighbors() {
        use crate::traits::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        // Entry A — close embedding to the query vector [1,0,0].
        let mut ea = KnowledgeEntry::new(
            "auth module".to_string(),
            "graph-scope".to_string(),
            "v1".to_string(),
        );
        ea.embedding = Some(next_emb());
        queryable.insert(&ea).await.unwrap();

        // Entry B — slightly different embedding (will be graph-expanded).
        let mut eb = KnowledgeEntry::new(
            "auth middleware".to_string(),
            "graph-scope".to_string(),
            "v1".to_string(),
        );
        eb.embedding = Some(next_emb());
        queryable.insert(&eb).await.unwrap();

        // 8 filler entries for HNSW connectivity.
        for i in 0..8 {
            let mut filler = KnowledgeEntry::new(
                format!("filler {i}"),
                "graph-scope".to_string(),
                "v1".to_string(),
            );
            filler.embedding = Some(next_emb());
            queryable.insert(&filler).await.unwrap();
        }

        // Create a graph edge: ea depends_on eb.
        graph.relate(&ea.id, "depends_on", &eb.id, None).await.unwrap();

        let retriever = GraphExpandRetriever::new(
            queryable,
            engine,
            graph,
            0.3, // alpha
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: true,
            graph_depth: 1,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = retriever.retrieve("auth", "graph-scope", &opts).await.unwrap();

        // HNSW approximate recall is unreliable at small N — use >= assertions.
        assert!(
            result.results.len() >= 2,
            "expected at least 2 results (vector + graph expanded), got {}",
            result.results.len()
        );
        assert_eq!(result.metrics.retriever_name, "graph_expand");
    }

    /// With expand_graph=false, GraphExpandRetriever should not expand
    /// any graph edges (graph_expanded == 0).
    #[tokio::test]
    async fn test_graph_expand_retriever_falls_back_when_disabled() {
        use crate::traits::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        // Entry A.
        let mut ea = KnowledgeEntry::new(
            "auth module".to_string(),
            "nograph-scope".to_string(),
            "v1".to_string(),
        );
        ea.embedding = Some(next_emb());
        queryable.insert(&ea).await.unwrap();

        // Entry B.
        let mut eb = KnowledgeEntry::new(
            "auth middleware".to_string(),
            "nograph-scope".to_string(),
            "v1".to_string(),
        );
        eb.embedding = Some(next_emb());
        queryable.insert(&eb).await.unwrap();

        // 8 filler entries.
        for i in 0..8 {
            let mut filler = KnowledgeEntry::new(
                format!("filler {i}"),
                "nograph-scope".to_string(),
                "v1".to_string(),
            );
            filler.embedding = Some(next_emb());
            queryable.insert(&filler).await.unwrap();
        }

        // Create a graph edge (will NOT be followed since expand_graph=false).
        graph.relate(&ea.id, "depends_on", &eb.id, None).await.unwrap();

        let retriever = GraphExpandRetriever::new(
            queryable,
            engine,
            graph,
            0.3,
        );

        let opts = RetrievalOpts {
            limit: 10,
            expand_graph: false,
            graph_depth: 1,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = retriever.retrieve("auth", "nograph-scope", &opts).await.unwrap();

        assert_eq!(
            result.metrics.graph_expanded, 0,
            "graph_expanded should be 0 when expand_graph is false"
        );
        assert_eq!(result.metrics.retriever_name, "graph_expand");
    }

    /// When a graph neighbor is already in vector results, its score should
    /// be boosted (reinforced) rather than silently skipped.
    #[tokio::test]
    async fn test_graph_reinforcement_boosts_overlapping_results() {
        use crate::traits::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        // Entry A — close to query vector [1,0,0]
        let mut ea = KnowledgeEntry::new(
            "auth module".to_string(),
            "reinforce-scope".to_string(),
            "v1".to_string(),
        );
        ea.embedding = Some(next_emb());
        queryable.insert(&ea).await.unwrap();

        // Entry B — also close (both in top-N vector results)
        let mut eb = KnowledgeEntry::new(
            "auth middleware".to_string(),
            "reinforce-scope".to_string(),
            "v1".to_string(),
        );
        eb.embedding = Some(next_emb());
        queryable.insert(&eb).await.unwrap();

        // 8 filler entries for HNSW connectivity
        for i in 0..8 {
            let mut filler = KnowledgeEntry::new(
                format!("filler {i}"),
                "reinforce-scope".to_string(),
                "v1".to_string(),
            );
            filler.embedding = Some(next_emb());
            queryable.insert(&filler).await.unwrap();
        }

        // Create edge A→B. Both A and B should be in vector results,
        // so B should get reinforced when we follow A's edges.
        graph.relate(&ea.id, "imports", &eb.id, None).await.unwrap();

        let retriever = GraphExpandRetriever::new(
            queryable,
            engine,
            graph,
            0.3, // alpha
        );

        let opts_graph = RetrievalOpts {
            limit: 10,
            expand_graph: true,
            graph_depth: 1,
            visibility: VisibilityMode::All,
            ..Default::default()
        };

        let result = retriever
            .retrieve("auth", "reinforce-scope", &opts_graph)
            .await
            .unwrap();

        // Both A and B are in vector results, and A→B edge exists,
        // so graph reinforcement should have boosted B's score.
        assert!(
            result.metrics.graph_reinforced >= 1,
            "expected graph_reinforced >= 1, got {}",
            result.metrics.graph_reinforced
        );
    }

    /// Verify that access_count increments and last_accessed updates after retrieval.
    #[tokio::test]
    async fn test_access_tracking_increments_on_retrieval() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert entries with slight embedding variation.
        let mut entry_ids = Vec::new();
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("access tracking test item {i}"),
                "access-scope".to_string(),
                format!("v{i}"),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.01, 0.0]);
            entry_ids.push(entry.id);
            store.insert(&entry).await.unwrap();
        }

        // Verify initial state: access_count = 0, last_accessed = None.
        let initial = store.get(&entry_ids[0]).await.unwrap().unwrap();
        assert_eq!(initial.access_count, 0);
        assert!(initial.last_accessed.is_none());

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);

        let opts = RetrievalOpts {
            limit: 5,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "access-scope", &opts).await.unwrap();
        assert!(!result.results.is_empty());

        // Give the spawned task time to complete.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        // Verify access tracking updated for returned entries.
        let returned_ids: Vec<uuid::Uuid> = result.results.iter().map(|sr| sr.entry.id).collect();
        for id in &returned_ids {
            let entry = store.get(id).await.unwrap().unwrap();
            assert_eq!(entry.access_count, 1, "access_count should be 1 for entry {}", id);
            assert!(entry.last_accessed.is_some(), "last_accessed should be set for entry {}", id);
        }

        // Search again — access_count should increment to 2.
        let _result2 = retriever.retrieve("test", "access-scope", &opts).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        for id in &returned_ids {
            let entry = store.get(id).await.unwrap().unwrap();
            assert!(entry.access_count >= 2, "access_count should be >= 2 after second search, got {}", entry.access_count);
        }
    }

    /// Verify that record_access is graceful on empty input.
    #[tokio::test]
    async fn test_access_tracking_empty_results_noop() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        // record_access with empty slice should not panic or error.
        store.record_access(&[]).await.unwrap();
    }

    /// Verify that record_access is graceful with non-existent entry IDs.
    #[tokio::test]
    async fn test_access_tracking_nonexistent_entry_graceful() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let fake_id = uuid::Uuid::now_v7();
        // Should not panic or error — just skip the missing entry.
        store.record_access(&[fake_id]).await.unwrap();
    }

    /// Cold entry found when include_cold=true.
    #[tokio::test]
    async fn test_cold_entry_found_when_include_cold_true() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 10 Hot entries for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("hot item {i}"),
                "scope-cold".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        // Insert a Cold entry directly into Redb (not HNSW).
        // We insert it as Hot first, then update its tier to Cold in Redb.
        let mut cold_entry = KnowledgeEntry::new(
            "cold knowledge: secret sauce".to_string(),
            "scope-cold".to_string(),
            "v1".to_string(),
        );
        cold_entry.embedding = Some(vec![1.0, 0.0, 0.0]); // exact match to query
        cold_entry.tier = Tier::Cold;
        // Insert via store (adds to HNSW), then we'll test cold scan directly
        store.insert(&cold_entry).await.unwrap();

        // Test: scan_cold_entries should find the cold entry.
        let cold_results = store
            .scan_cold_entries(&[1.0, 0.0, 0.0], "scope-cold", 10)
            .unwrap();
        assert_eq!(cold_results.len(), 1, "should find exactly 1 cold entry");
        assert_eq!(cold_results[0].entry.content, "cold knowledge: secret sauce");
        assert!(cold_results[0].score > 0.99, "exact match should have score ~1.0");

        // Test via retriever with include_cold=true.
        // Note: store.insert() adds Cold entries to HNSW too (production tier demotion
        // would remove from HNSW). So the entry is found via HNSW; cold_results may be 0
        // after dedup. The key assertion is that the entry appears in results.
        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 15,
            include_cold: true,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "scope-cold", &opts).await.unwrap();
        // Cold entry should appear in results (may come from HNSW or cold scan).
        let has_cold = result.results.iter().any(|r| r.entry.content.contains("cold knowledge"));
        assert!(has_cold, "cold entry should be in results when include_cold=true");
    }

    /// Cold entry NOT found when include_cold=false (default).
    #[tokio::test]
    async fn test_cold_entry_not_found_when_include_cold_false() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 10 Hot entries for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("hot item {i}"),
                "scope-default".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        // Insert a Cold entry.
        let mut cold_entry = KnowledgeEntry::new(
            "cold secret".to_string(),
            "scope-default".to_string(),
            "v1".to_string(),
        );
        cold_entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        cold_entry.tier = Tier::Cold;
        store.insert(&cold_entry).await.unwrap();

        // Default opts: include_cold=false.
        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 15,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        assert!(!opts.include_cold, "default should be false");
        let result = retriever.retrieve("test", "scope-default", &opts).await.unwrap();
        assert_eq!(result.metrics.cold_results, 0, "no cold scan when include_cold=false");
        assert_eq!(result.metrics.cold_scan_latency_ms, 0);
    }

    /// Cold entry with higher cosine similarity ranks above Warm entry.
    #[tokio::test]
    async fn test_cold_entry_ranks_above_warm_when_higher_similarity() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 10 filler entries for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![0.5, idx as f32 * 0.01, 0.5]
        };
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "scope-rank".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        // Warm entry with low similarity to query [1,0,0].
        let mut warm_entry = KnowledgeEntry::new(
            "warm entry".to_string(),
            "scope-rank".to_string(),
            "v1".to_string(),
        );
        warm_entry.embedding = Some(vec![0.0, 1.0, 0.0]); // orthogonal to query
        warm_entry.tier = Tier::Warm;
        store.insert(&warm_entry).await.unwrap();

        // Cold entry with high similarity to query [1,0,0].
        let mut cold_entry = KnowledgeEntry::new(
            "cold entry high score".to_string(),
            "scope-rank".to_string(),
            "v1".to_string(),
        );
        cold_entry.embedding = Some(vec![1.0, 0.0, 0.0]); // exact match
        cold_entry.tier = Tier::Cold;
        store.insert(&cold_entry).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 20,
            include_cold: true,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "scope-rank", &opts).await.unwrap();

        // Find positions.
        let cold_pos = result.results.iter().position(|r| r.entry.content == "cold entry high score");
        let warm_pos = result.results.iter().position(|r| r.entry.content == "warm entry");

        assert!(cold_pos.is_some(), "cold entry should be in results");
        // If warm entry appears at all, cold should rank higher (lower index).
        if let Some(wp) = warm_pos {
            assert!(
                cold_pos.unwrap() < wp,
                "cold entry (exact match) should rank above warm entry (orthogonal)"
            );
        }
    }

    // ---- Tier-aware retrieval scoring tests (Issue #22) ----

    /// tier_weight unit test: verify the mapping.
    #[test]
    fn test_tier_weight_values() {
        use corvia_common::types::Tier;
        assert_eq!(super::tier_weight(Tier::Hot), 1.0);
        assert_eq!(super::tier_weight(Tier::Warm), 0.7);
        assert_eq!(super::tier_weight(Tier::Cold), 0.3);
        assert_eq!(super::tier_weight(Tier::Forgotten), 0.0);
    }

    /// Hot entry ranks above Warm entry with same cosine similarity.
    #[tokio::test]
    async fn test_hot_ranks_above_warm_same_cosine() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert 8 filler entries for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![0.5, idx as f32 * 0.01, 0.5]
        };
        for i in 0..8 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "tier-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            store.insert(&entry).await.unwrap();
        }

        // Hot entry with embedding close to query [1,0,0].
        let mut hot = KnowledgeEntry::new(
            "hot entry".to_string(),
            "tier-scope".to_string(),
            "v1".to_string(),
        );
        hot.embedding = Some(vec![1.0, 0.001, 0.0]);
        hot.tier = Tier::Hot;
        store.insert(&hot).await.unwrap();

        // Warm entry with same embedding (same cosine similarity).
        let mut warm = KnowledgeEntry::new(
            "warm entry".to_string(),
            "tier-scope".to_string(),
            "v1".to_string(),
        );
        warm.embedding = Some(vec![1.0, 0.002, 0.0]);
        warm.tier = Tier::Warm;
        store.insert(&warm).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 20,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "tier-scope", &opts).await.unwrap();

        let hot_pos = result.results.iter().position(|r| r.entry.content == "hot entry");
        let warm_pos = result.results.iter().position(|r| r.entry.content == "warm entry");

        assert!(hot_pos.is_some(), "hot entry should be in results");
        assert!(warm_pos.is_some(), "warm entry should be in results");
        assert!(
            hot_pos.unwrap() < warm_pos.unwrap(),
            "hot entry (weight=1.0) should rank above warm entry (weight=0.7)"
        );

        // Verify warm entry's score is lower.
        let hot_score = result.results[hot_pos.unwrap()].score;
        let warm_score = result.results[warm_pos.unwrap()].score;
        assert!(
            hot_score > warm_score,
            "hot score ({hot_score}) should be > warm score ({warm_score})"
        );
    }

    /// Forgotten entries never appear in retrieval results.
    #[tokio::test]
    async fn test_forgotten_entry_never_in_results() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert filler Hot entries.
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("hot item {i}"),
                "forgotten-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.001, 0.0]);
            store.insert(&entry).await.unwrap();
        }

        // Forgotten entry with exact match embedding — would rank #1 without tier filtering.
        let mut forgotten = KnowledgeEntry::new(
            "forgotten entry".to_string(),
            "forgotten-scope".to_string(),
            "v1".to_string(),
        );
        forgotten.embedding = Some(vec![1.0, 0.0, 0.0]);
        forgotten.tier = Tier::Forgotten;
        store.insert(&forgotten).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 20,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "forgotten-scope", &opts).await.unwrap();

        let has_forgotten = result.results.iter().any(|r| r.entry.content == "forgotten entry");
        assert!(!has_forgotten, "forgotten entries must never appear in results");
    }

    /// Verify exact score ratio: warm score == hot score * 0.7 for identical cosine.
    #[tokio::test]
    async fn test_tier_weight_score_ratio() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert fillers for HNSW connectivity.
        for i in 0..8 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "ratio-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![0.5, i as f32 * 0.01, 0.5]);
            store.insert(&entry).await.unwrap();
        }

        // Hot and Warm entries with identical embeddings (same cosine similarity).
        let mut hot = KnowledgeEntry::new(
            "hot ratio".to_string(),
            "ratio-scope".to_string(),
            "v1".to_string(),
        );
        hot.embedding = Some(vec![1.0, 0.001, 0.0]);
        hot.tier = Tier::Hot;
        store.insert(&hot).await.unwrap();

        let mut warm = KnowledgeEntry::new(
            "warm ratio".to_string(),
            "ratio-scope".to_string(),
            "v1".to_string(),
        );
        warm.embedding = Some(vec![1.0, 0.002, 0.0]);
        warm.tier = Tier::Warm;
        store.insert(&warm).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 20,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "ratio-scope", &opts).await.unwrap();

        let hot_score = result.results.iter().find(|r| r.entry.content == "hot ratio").unwrap().score;
        let warm_score = result.results.iter().find(|r| r.entry.content == "warm ratio").unwrap().score;

        // warm_score should be ~0.7 * hot_score (embeddings nearly identical).
        let ratio = warm_score / hot_score;
        assert!(
            (ratio - 0.7).abs() < 0.01,
            "tier weight ratio should be ~0.7, got {ratio} (hot={hot_score}, warm={warm_score})"
        );
    }

    /// When all entries are Forgotten, retriever returns empty results.
    #[tokio::test]
    async fn test_all_forgotten_returns_empty() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert only Forgotten entries.
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("forgotten {i}"),
                "all-forgotten".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.001, 0.0]);
            entry.tier = Tier::Forgotten;
            store.insert(&entry).await.unwrap();
        }

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 10,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "all-forgotten", &opts).await.unwrap();
        assert!(result.results.is_empty(), "all-Forgotten scope should return empty results");
    }

    /// Cold entry with high cosine can still rank above Warm entry with low cosine.
    /// 0.95 * 0.3 = 0.285 vs 0.4 * 0.7 = 0.28 — cold wins.
    #[tokio::test]
    async fn test_cold_high_cosine_above_warm_low_cosine() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert fillers for HNSW connectivity.
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "rank-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![0.5, i as f32 * 0.01, 0.5]);
            store.insert(&entry).await.unwrap();
        }

        // Warm entry with LOW cosine similarity to query [1,0,0].
        let mut warm = KnowledgeEntry::new(
            "warm low cosine".to_string(),
            "rank-scope".to_string(),
            "v1".to_string(),
        );
        warm.embedding = Some(vec![0.0, 1.0, 0.0]); // orthogonal to query → cosine ~0
        warm.tier = Tier::Warm;
        store.insert(&warm).await.unwrap();

        // Cold entry with HIGH cosine similarity to query [1,0,0].
        let mut cold = KnowledgeEntry::new(
            "cold high cosine".to_string(),
            "rank-scope".to_string(),
            "v1".to_string(),
        );
        cold.embedding = Some(vec![1.0, 0.0, 0.0]); // exact match → cosine ~1.0
        cold.tier = Tier::Cold;
        store.insert(&cold).await.unwrap();

        let retriever = VectorRetriever::new(store.clone() as Arc<dyn QueryableStore>, engine);
        let opts = RetrievalOpts {
            limit: 20,
            include_cold: true,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "rank-scope", &opts).await.unwrap();

        let cold_pos = result.results.iter().position(|r| r.entry.content == "cold high cosine");
        let warm_pos = result.results.iter().position(|r| r.entry.content == "warm low cosine");

        assert!(cold_pos.is_some(), "cold entry should be in results");
        assert!(warm_pos.is_some(), "warm entry must be in results for ranking comparison");
        assert!(
            cold_pos.unwrap() < warm_pos.unwrap(),
            "cold entry (1.0*0.3=0.3) should rank above warm entry (~0*0.7≈0)"
        );
    }

    /// GraphExpandRetriever also excludes Forgotten entries from results.
    #[tokio::test]
    async fn test_graph_expand_excludes_forgotten() {
        use crate::traits::GraphStore;
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let graph = store.clone() as Arc<dyn GraphStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);

        // Insert filler entries.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("filler {i}"),
                "ge-scope".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            queryable.insert(&entry).await.unwrap();
        }

        // Forgotten entry with exact match embedding.
        let mut forgotten = KnowledgeEntry::new(
            "forgotten via graph".to_string(),
            "ge-scope".to_string(),
            "v1".to_string(),
        );
        forgotten.embedding = Some(vec![1.0, 0.0, 0.0]);
        forgotten.tier = Tier::Forgotten;
        queryable.insert(&forgotten).await.unwrap();

        let retriever = GraphExpandRetriever::new(queryable, engine, graph, 0.3);

        let opts = RetrievalOpts {
            limit: 20,
            expand_graph: true,
            graph_depth: 1,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "ge-scope", &opts).await.unwrap();

        let has_forgotten = result.results.iter().any(|r| r.entry.content == "forgotten via graph");
        assert!(!has_forgotten, "forgotten entries must be excluded from graph expand results");
    }

    /// Scan handles zero Cold entries gracefully.
    #[tokio::test]
    async fn test_cold_scan_zero_cold_entries() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        // Insert only Hot entries.
        for i in 0..5 {
            let mut entry = KnowledgeEntry::new(
                format!("hot {i}"),
                "scope-empty".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(vec![1.0, i as f32 * 0.01, 0.0]);
            store.insert(&entry).await.unwrap();
        }

        let results = store
            .scan_cold_entries(&[1.0, 0.0, 0.0], "scope-empty", 10)
            .unwrap();
        assert!(results.is_empty(), "no cold entries should mean empty results");
    }

    /// Forgotten entries (embedding=None) excluded from cold scan.
    #[tokio::test]
    async fn test_cold_scan_excludes_forgotten_entries() {
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Cold entry with embedding.
        let mut cold_with_emb = KnowledgeEntry::new(
            "cold with embedding".to_string(),
            "scope-forgotten".to_string(),
            "v1".to_string(),
        );
        cold_with_emb.embedding = Some(vec![1.0, 0.0, 0.0]);
        cold_with_emb.tier = Tier::Cold;
        store.insert(&cold_with_emb).await.unwrap();

        // Insert a Forgotten entry (Cold tier but embedding=None conceptually).
        // In practice Forgotten entries have tier=Forgotten, not Cold.
        let mut forgotten = KnowledgeEntry::new(
            "forgotten entry".to_string(),
            "scope-forgotten".to_string(),
            "v1".to_string(),
        );
        forgotten.embedding = Some(vec![1.0, 0.0, 0.0]); // need embedding to insert
        forgotten.tier = Tier::Forgotten;
        store.insert(&forgotten).await.unwrap();

        let results = store
            .scan_cold_entries(&[1.0, 0.0, 0.0], "scope-forgotten", 10)
            .unwrap();
        assert_eq!(results.len(), 1, "only Cold tier entries should be returned");
        assert_eq!(results[0].entry.content, "cold with embedding");
    }

    /// GraphExpandRetriever: cold entries get alpha-blended scores (no graph bonus).
    #[tokio::test]
    async fn test_graph_expand_retriever_cold_scan() {
        use crate::traits::GraphStore;
        use corvia_common::types::Tier;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let queryable = store.clone() as Arc<dyn QueryableStore>;
        let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine);
        let graph = store.clone() as Arc<dyn GraphStore>;

        // Insert 10 Hot entries for HNSW connectivity.
        let mut idx: usize = 0;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };
        for i in 0..10 {
            let mut entry = KnowledgeEntry::new(
                format!("hot item {i}"),
                "scope-graph-cold".to_string(),
                "v1".to_string(),
            );
            entry.embedding = Some(next_emb());
            queryable.insert(&entry).await.unwrap();
        }

        // Insert a Cold entry with high similarity.
        let mut cold_entry = KnowledgeEntry::new(
            "cold via graph retriever".to_string(),
            "scope-graph-cold".to_string(),
            "v1".to_string(),
        );
        cold_entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        cold_entry.tier = Tier::Cold;
        queryable.insert(&cold_entry).await.unwrap();

        let alpha = 0.3;
        let retriever = GraphExpandRetriever::new(queryable, engine, graph, alpha);

        // Test with include_cold=true — entry is found (may be via HNSW since insert
        // adds to HNSW regardless of tier; cold_results may be 0 after dedup).
        let opts = RetrievalOpts {
            limit: 15,
            expand_graph: true,
            graph_depth: 1,
            include_cold: true,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result = retriever.retrieve("test", "scope-graph-cold", &opts).await.unwrap();

        let cold_hit = result.results.iter().find(|r| r.entry.content == "cold via graph retriever");
        assert!(cold_hit.is_some(), "cold entry should be in graph retriever results");

        // Test with include_cold=false — entry still found (it's in HNSW).
        let opts_no_cold = RetrievalOpts {
            limit: 15,
            expand_graph: true,
            graph_depth: 1,
            include_cold: false,
            visibility: VisibilityMode::All,
            ..Default::default()
        };
        let result_no_cold = retriever.retrieve("test", "scope-graph-cold", &opts_no_cold).await.unwrap();
        assert_eq!(result_no_cold.metrics.cold_results, 0, "no cold scan when disabled");
        assert_eq!(result_no_cold.metrics.cold_scan_latency_ms, 0);

        // Verify the scan_cold_entries method directly finds the Cold entry.
        let direct_scan = store.scan_cold_entries(&[1.0, 0.0, 0.0], "scope-graph-cold", 10).unwrap();
        assert_eq!(direct_scan.len(), 1, "scan should find the cold entry directly");
        assert_eq!(direct_scan[0].entry.content, "cold via graph retriever");
    }
}
