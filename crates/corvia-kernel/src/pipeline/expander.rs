//! Expander trait and implementations for enriching results via graph traversal.
//!
//! The expander runs after fusion, adding graph-connected entries to the result set.
//! [`GraphExpander`] is extracted from the monolithic `GraphExpandRetriever`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, Tier};
use tracing::info;

use super::{CandidateScores, NormalizedScore, RankedCandidate, RankedSet, SearchContext, StageMetrics};
use crate::reasoner::cosine_similarity;
use crate::retriever::tier_weight;
use crate::traits::{GraphStore, QueryableStore};

/// Expand a ranked set by discovering related entries (e.g., via graph edges).
///
/// Reads expansion depth from `ctx.opts.graph_depth`. When `ctx.opts.expand_graph`
/// is false, implementations should pass through unchanged.
#[async_trait]
pub trait Expander: Send + Sync {
    /// Human-readable name for metrics.
    fn name(&self) -> &str;

    /// Expand the candidate set.
    async fn expand(&self, ctx: &SearchContext, set: RankedSet) -> Result<RankedSet>;
}

/// No-op expander that passes candidates through unchanged.
pub struct NoOpExpander;

#[async_trait]
impl Expander for NoOpExpander {
    fn name(&self) -> &str {
        "noop"
    }

    async fn expand(&self, _ctx: &SearchContext, set: RankedSet) -> Result<RankedSet> {
        Ok(set)
    }
}

// ---------------------------------------------------------------------------
// GraphExpander — extracted from GraphExpandRetriever
// ---------------------------------------------------------------------------

/// Graph-based expander that follows edges from seed candidates.
///
/// Extracted from the monolithic `GraphExpandRetriever`. Seeds use fused
/// `final_score`; neighbors compute fresh cosine similarity against the
/// query embedding.
///
/// Scoring: `final = ((1-alpha) * cosine + alpha * proximity_bonus) * tier_weight`
/// where proximity_bonus is 0.5 for hop-1 and 1/3 for hop-2+.
/// Reinforcement: already-seen entries get diminishing bonuses (50% decay per hit).
pub struct GraphExpander {
    store: Arc<dyn QueryableStore>,
    graph: Arc<dyn GraphStore>,
    alpha: f32,
}

impl GraphExpander {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        graph: Arc<dyn GraphStore>,
        alpha: f32,
    ) -> Self {
        Self { store, graph, alpha }
    }
}

#[async_trait]
impl Expander for GraphExpander {
    fn name(&self) -> &str {
        "graph"
    }

    async fn expand(&self, ctx: &SearchContext, set: RankedSet) -> Result<RankedSet> {
        if !ctx.opts.expand_graph {
            return Ok(set);
        }

        let start = Instant::now();
        let input_count = set.candidates.len();

        // Track seen IDs and build working set.
        let mut seen: HashSet<uuid::Uuid> = HashSet::new();
        let mut scored: Vec<RankedCandidate> = Vec::new();
        let mut reinforcement_counts: HashMap<uuid::Uuid, u32> = HashMap::new();
        let mut graph_expanded: usize = 0;
        let mut graph_reinforced: usize = 0;
        let mut warnings: Vec<String> = Vec::new();

        // Copy seed candidates.
        for candidate in &set.candidates {
            seen.insert(candidate.entry.id);
            scored.push(candidate.clone());
        }

        let embedding = &*ctx.query_embedding;

        // Hop 1: follow edges from each seed.
        for seed in &set.candidates {
            let edges = self
                .graph
                .edges(&seed.entry.id, EdgeDirection::Both)
                .await?;

            for edge in &edges {
                let neighbor_id = if edge.from == seed.entry.id {
                    edge.to
                } else {
                    edge.from
                };

                if seen.contains(&neighbor_id) {
                    // Reinforce with diminishing returns.
                    let count = reinforcement_counts.entry(neighbor_id).or_insert(0);
                    let decay = 0.5_f32.powi(*count as i32);
                    let bonus = self.alpha * 0.25 * decay;
                    if bonus > 0.001 {
                        if let Some(existing) = scored.iter_mut().find(|c| c.entry.id == neighbor_id) {
                            let new_score = existing.scores.final_score.value() + bonus;
                            existing.scores.final_score = NormalizedScore::new(new_score.min(1.0));
                            existing.scores.components.insert("graph_reinforce".to_string(),
                                existing.scores.components.get("graph_reinforce").unwrap_or(&0.0) + bonus);
                            graph_reinforced += 1;
                        }
                        *count += 1;
                    }
                    continue;
                }

                if let Some(neighbor_entry) = self.store.get(&neighbor_id).await? {
                    if neighbor_entry.scope_id != ctx.scope_id {
                        continue;
                    }
                    if neighbor_entry.tier == Tier::Forgotten {
                        continue;
                    }
                    seen.insert(neighbor_id);

                    let cosine = neighbor_entry
                        .embedding
                        .as_ref()
                        .map(|emb| cosine_similarity(embedding, emb))
                        .unwrap_or(0.0);
                    let blended =
                        ((1.0 - self.alpha) * cosine + self.alpha * 0.5)
                            * tier_weight(neighbor_entry.tier);

                    let mut components = HashMap::new();
                    components.insert("cosine".to_string(), cosine);
                    components.insert("graph_hop1".to_string(), self.alpha * 0.5);

                    scored.push(RankedCandidate {
                        entry: Arc::new(neighbor_entry),
                        scores: CandidateScores {
                            components,
                            final_score: NormalizedScore::new(blended),
                        },
                    });
                    graph_expanded += 1;
                }
            }
        }

        // Deeper hops (graph_depth > 1).
        if ctx.opts.graph_depth > 1 {
            for seed in &set.candidates {
                let deep_entries = self
                    .graph
                    .traverse(
                        &seed.entry.id,
                        None,
                        EdgeDirection::Both,
                        ctx.opts.graph_depth,
                    )
                    .await?;

                for entry in deep_entries {
                    if seen.contains(&entry.id) {
                        // Reinforce with diminishing returns for deeper hops.
                        let count = reinforcement_counts.entry(entry.id).or_insert(0);
                        let decay = 0.5_f32.powi(*count as i32);
                        let bonus = self.alpha * 0.15 * decay;
                        if bonus > 0.001 {
                            if let Some(existing) = scored.iter_mut().find(|c| c.entry.id == entry.id) {
                                let new_score = existing.scores.final_score.value() + bonus;
                                existing.scores.final_score = NormalizedScore::new(new_score.min(1.0));
                                graph_reinforced += 1;
                            }
                            *count += 1;
                        }
                        continue;
                    }
                    if entry.scope_id != ctx.scope_id {
                        continue;
                    }
                    if entry.tier == Tier::Forgotten {
                        continue;
                    }
                    seen.insert(entry.id);

                    let cosine = entry
                        .embedding
                        .as_ref()
                        .map(|emb| cosine_similarity(embedding, emb))
                        .unwrap_or(0.0);
                    let blended = ((1.0 - self.alpha) * cosine + self.alpha * (1.0 / 3.0))
                        * tier_weight(entry.tier);

                    let mut components = HashMap::new();
                    components.insert("cosine".to_string(), cosine);
                    components.insert("graph_deep".to_string(), self.alpha * (1.0 / 3.0));

                    scored.push(RankedCandidate {
                        entry: Arc::new(entry),
                        scores: CandidateScores {
                            components,
                            final_score: NormalizedScore::new(blended),
                        },
                    });
                    graph_expanded += 1;
                }
            }
        }

        // Re-sort by final_score descending.
        scored.sort_unstable_by(|a, b| {
            b.scores
                .final_score
                .value()
                .partial_cmp(&a.scores.final_score.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let output_count = scored.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        if graph_expanded == 0 && input_count > 0 {
            warnings.push("graph expansion produced no new candidates".to_string());
        }

        info!(
            expander = self.name(),
            input_count,
            graph_expanded,
            graph_reinforced,
            output_count,
            latency_ms,
            "graph expansion complete"
        );

        Ok(RankedSet {
            candidates: scored,
            metrics: StageMetrics {
                stage_name: self.name().to_string(),
                latency_ms,
                input_count,
                output_count,
                warnings,
            },
        })
    }
}
