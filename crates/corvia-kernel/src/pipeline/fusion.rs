//! Fusion trait and implementations for merging multiple searcher results.
//!
//! When the pipeline has a single searcher, [`PassThrough`] applies min-max
//! normalization. When multiple searchers produce results, a fusion strategy
//! (e.g., RRF in Phase 2c) merges and re-scores them.

use std::collections::HashMap;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;

use std::sync::Arc;

use super::{CandidateScores, NormalizedScore, RankedCandidate, RankedSet, StageMetrics};

/// Merge results from one or more searchers into a single ranked set.
///
/// The output MUST have `NormalizedScore` values in `[0, 1]`.
/// Implementations must handle deduplication (same entry from multiple searchers).
#[async_trait]
pub trait Fusion: Send + Sync {
    /// Human-readable name for metrics.
    fn name(&self) -> &str;

    /// Fuse multiple ranked sets into one.
    async fn fuse(&self, sets: Vec<RankedSet>) -> Result<RankedSet>;
}

// ---------------------------------------------------------------------------
// PassThrough — single-searcher min-max normalization
// ---------------------------------------------------------------------------

/// Passthrough fusion for single-searcher pipelines.
///
/// Applies min-max normalization to scores. When all scores are identical
/// (degenerate case), all candidates receive a score of 1.0.
/// When multiple sets are provided, deduplicates by entry UUID (keeps highest score).
pub struct PassThrough;

#[async_trait]
impl Fusion for PassThrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn fuse(&self, sets: Vec<RankedSet>) -> Result<RankedSet> {
        let start = Instant::now();
        let input_count: usize = sets.iter().map(|s| s.candidates.len()).sum();

        // Collect all candidates, deduplicating by entry ID (keep highest score).
        // Uses into_iter to move candidates rather than cloning.
        let mut by_id: HashMap<uuid::Uuid, RankedCandidate> = HashMap::new();
        for set in sets {
            for candidate in set.candidates {
                let id = candidate.entry.id;
                use std::collections::hash_map::Entry;
                match by_id.entry(id) {
                    Entry::Occupied(mut e) => {
                        if candidate.scores.final_score > e.get().scores.final_score {
                            e.insert(candidate);
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(candidate);
                    }
                }
            }
        }

        let mut candidates: Vec<RankedCandidate> = by_id.into_values().collect();

        // Min-max normalize.
        if !candidates.is_empty() {
            let (min_score, max_score) = candidates.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), c| {
                let v = c.scores.final_score.value();
                (mn.min(v), mx.max(v))
            });

            let range = max_score - min_score;
            if range > f32::EPSILON {
                for c in &mut candidates {
                    let normalized = (c.scores.final_score.value() - min_score) / range;
                    c.scores.final_score = NormalizedScore::new(normalized);
                }
            } else {
                // Degenerate case: all scores identical -> 1.0.
                for c in &mut candidates {
                    c.scores.final_score = NormalizedScore::new(1.0);
                }
            }
        }

        // Sort descending by final_score.
        candidates.sort_unstable_by_key(|c| std::cmp::Reverse(c.scores.final_score));

        let output_count = candidates.len();
        Ok(RankedSet {
            candidates,
            metrics: StageMetrics {
                stage_name: self.name().to_string(),
                latency_ms: start.elapsed().as_millis() as u64,
                input_count,
                output_count,
                warnings: Vec::new(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// RRFusion — Reciprocal Rank Fusion for multi-searcher pipelines
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion (RRF) for combining results from multiple searchers.
///
/// RRF formula: `score(d) = sum_i 1 / (k + rank_i(d))` for each searcher set
/// where `rank_i(d)` is the 1-indexed rank of document d in set i.
///
/// Deduplicates entries by UUID, merges component scores from overlapping entries,
/// and normalizes the final RRF score by dividing by the maximum.
pub struct RRFusion {
    k: usize,
}

impl RRFusion {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

#[async_trait]
impl Fusion for RRFusion {
    fn name(&self) -> &str {
        "rrf"
    }

    async fn fuse(&self, sets: Vec<RankedSet>) -> Result<RankedSet> {
        let start = Instant::now();
        let input_count: usize = sets.iter().map(|s| s.candidates.len()).sum();

        // Degenerate cases.
        if sets.is_empty() {
            return Ok(RankedSet {
                candidates: Vec::new(),
                metrics: StageMetrics {
                    stage_name: self.name().to_string(),
                    latency_ms: 0,
                    input_count: 0,
                    output_count: 0,
                    warnings: Vec::new(),
                },
            });
        }

        // Single set: delegate to PassThrough behavior (min-max normalization).
        if sets.len() == 1 {
            return PassThrough.fuse(sets).await;
        }

        // Accumulate RRF scores and merge component scores by entry UUID.
        // Each candidate's rank within its searcher set determines its RRF contribution.
        let mut rrf_scores: HashMap<uuid::Uuid, f64> = HashMap::new();
        let mut merged_components: HashMap<uuid::Uuid, HashMap<String, f32>> = HashMap::new();
        let mut entries: HashMap<uuid::Uuid, Arc<corvia_common::types::KnowledgeEntry>> =
            HashMap::new();

        for set in &sets {
            for (rank_0, candidate) in set.candidates.iter().enumerate() {
                let id = candidate.entry.id;
                let rank = (rank_0 + 1) as f64; // 1-indexed
                let rrf_contribution = 1.0 / (self.k as f64 + rank);

                *rrf_scores.entry(id).or_insert(0.0) += rrf_contribution;

                // Merge component scores (keeps all components from all searchers).
                let comps = merged_components.entry(id).or_default();
                for (key, &val) in &candidate.scores.components {
                    comps.entry(key.clone()).or_insert(val);
                }

                entries.entry(id).or_insert_with(|| Arc::clone(&candidate.entry));
            }
        }

        // Find max RRF score for normalization.
        let max_rrf = rrf_scores.values().copied().fold(0.0f64, f64::max);

        // Build final candidates with normalized scores.
        let mut candidates: Vec<RankedCandidate> = rrf_scores
            .into_iter()
            .map(|(id, rrf_score)| {
                let normalized = if max_rrf > f64::EPSILON {
                    (rrf_score / max_rrf) as f32
                } else {
                    1.0
                };
                RankedCandidate {
                    entry: entries.remove(&id).unwrap(),
                    scores: CandidateScores {
                        components: merged_components.remove(&id).unwrap_or_default(),
                        final_score: NormalizedScore::new(normalized),
                    },
                }
            })
            .collect();

        // Sort descending by final_score.
        candidates.sort_unstable_by_key(|c| std::cmp::Reverse(c.scores.final_score));

        let output_count = candidates.len();
        Ok(RankedSet {
            candidates,
            metrics: StageMetrics {
                stage_name: self.name().to_string(),
                latency_ms: start.elapsed().as_millis() as u64,
                input_count,
                output_count,
                warnings: Vec::new(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::CandidateScores;
    use corvia_common::types::KnowledgeEntry;
    use std::sync::Arc;

    fn make_candidate(id_suffix: u8, score: f32) -> RankedCandidate {
        let mut entry = KnowledgeEntry::new(
            format!("entry-{id_suffix}"),
            "scope".into(),
            "v1".into(),
        );
        // Set a deterministic UUID for testing.
        entry.id = uuid::Uuid::from_bytes([
            id_suffix, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let mut components = HashMap::new();
        components.insert("test".to_string(), score);
        RankedCandidate {
            entry: Arc::new(entry),
            scores: CandidateScores {
                components,
                final_score: NormalizedScore::new(score),
            },
        }
    }

    fn make_set(candidates: Vec<RankedCandidate>) -> RankedSet {
        let count = candidates.len();
        RankedSet {
            candidates,
            metrics: StageMetrics {
                stage_name: "test".into(),
                latency_ms: 0,
                input_count: 0,
                output_count: count,
                warnings: Vec::new(),
            },
        }
    }

    #[tokio::test]
    async fn test_passthrough_single_result() {
        let set = make_set(vec![make_candidate(1, 0.8)]);
        let result = PassThrough.fuse(vec![set]).await.unwrap();
        // Single result with degenerate range -> 1.0.
        assert_eq!(result.candidates.len(), 1);
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_passthrough_all_same_scores() {
        let set = make_set(vec![
            make_candidate(1, 0.5),
            make_candidate(2, 0.5),
            make_candidate(3, 0.5),
        ]);
        let result = PassThrough.fuse(vec![set]).await.unwrap();
        assert_eq!(result.candidates.len(), 3);
        // All same -> degenerate case -> all 1.0.
        for c in &result.candidates {
            assert!((c.scores.final_score.value() - 1.0).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_passthrough_normal_normalization() {
        let set = make_set(vec![
            make_candidate(1, 0.9),
            make_candidate(2, 0.5),
            make_candidate(3, 0.1),
        ]);
        let result = PassThrough.fuse(vec![set]).await.unwrap();
        assert_eq!(result.candidates.len(), 3);
        // Highest should be 1.0, lowest should be 0.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
        assert!(
            (result.candidates[result.candidates.len() - 1]
                .scores
                .final_score
                .value()
                - 0.0)
                .abs()
                < f32::EPSILON
        );
    }

    #[tokio::test]
    async fn test_passthrough_deduplicates_by_uuid() {
        // Same candidate in two sets with different scores.
        let set1 = make_set(vec![make_candidate(1, 0.9), make_candidate(2, 0.3)]);
        let set2 = make_set(vec![make_candidate(1, 0.5), make_candidate(3, 0.7)]);
        let result = PassThrough.fuse(vec![set1, set2]).await.unwrap();
        // Should have 3 unique entries (candidate 1 deduped, keeps higher score).
        assert_eq!(result.candidates.len(), 3);
    }

    #[tokio::test]
    async fn test_passthrough_empty() {
        let result = PassThrough.fuse(vec![]).await.unwrap();
        assert!(result.candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // RRFusion tests
    // -----------------------------------------------------------------------

    fn make_named_candidate(id_suffix: u8, score: f32, component: &str) -> RankedCandidate {
        let mut entry = KnowledgeEntry::new(
            format!("entry-{id_suffix}"),
            "scope".into(),
            "v1".into(),
        );
        entry.id = uuid::Uuid::from_bytes([
            id_suffix, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let mut components = HashMap::new();
        components.insert(component.to_string(), score);
        RankedCandidate {
            entry: Arc::new(entry),
            scores: CandidateScores {
                components,
                final_score: NormalizedScore::new(score),
            },
        }
    }

    #[tokio::test]
    async fn test_rrf_two_sets_with_overlap() {
        // Entry 1 appears in both sets (should get higher RRF score).
        let set1 = make_set(vec![
            make_named_candidate(1, 0.9, "vector"),
            make_named_candidate(2, 0.7, "vector"),
        ]);
        let set2 = make_set(vec![
            make_named_candidate(1, 0.8, "bm25"),
            make_named_candidate(3, 0.6, "bm25"),
        ]);
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![set1, set2]).await.unwrap();

        // 3 unique entries.
        assert_eq!(result.candidates.len(), 3);
        // Entry 1 (overlapping) should be ranked first.
        assert_eq!(
            result.candidates[0].entry.id.as_bytes()[0],
            1,
            "overlapping entry should rank first"
        );
        // Entry 1 should have both component scores merged.
        assert!(result.candidates[0].scores.components.contains_key("vector"));
        assert!(result.candidates[0].scores.components.contains_key("bm25"));
        // Top candidate should have normalized score 1.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_rrf_disjoint_sets() {
        // No overlap: entries only appear in one set.
        let set1 = make_set(vec![
            make_named_candidate(1, 0.9, "vector"),
            make_named_candidate(2, 0.7, "vector"),
        ]);
        let set2 = make_set(vec![
            make_named_candidate(3, 0.8, "bm25"),
            make_named_candidate(4, 0.6, "bm25"),
        ]);
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![set1, set2]).await.unwrap();

        assert_eq!(result.candidates.len(), 4);
        // All rank-1 entries should tie (both get 1/(60+1) = same RRF score).
        // Top should be 1.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_rrf_single_set_delegates_to_passthrough() {
        let set = make_set(vec![
            make_named_candidate(1, 0.9, "vector"),
            make_named_candidate(2, 0.5, "vector"),
            make_named_candidate(3, 0.1, "vector"),
        ]);
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![set]).await.unwrap();

        assert_eq!(result.candidates.len(), 3);
        // Passthrough: highest -> 1.0, lowest -> 0.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
        assert!((result.candidates[2].scores.final_score.value() - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_rrf_empty_input() {
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![]).await.unwrap();
        assert!(result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_rrf_scores_are_normalized() {
        let set1 = make_set(vec![
            make_named_candidate(1, 0.9, "vector"),
            make_named_candidate(2, 0.5, "vector"),
        ]);
        let set2 = make_set(vec![
            make_named_candidate(3, 0.8, "bm25"),
            make_named_candidate(4, 0.4, "bm25"),
        ]);
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![set1, set2]).await.unwrap();

        for c in &result.candidates {
            let score = c.scores.final_score.value();
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} not in [0,1]"
            );
        }
    }

    #[tokio::test]
    async fn test_rrf_top_candidate_always_1_0() {
        let set1 = make_set(vec![
            make_named_candidate(1, 0.9, "vector"),
            make_named_candidate(2, 0.7, "vector"),
            make_named_candidate(3, 0.5, "vector"),
        ]);
        let set2 = make_set(vec![
            make_named_candidate(4, 0.8, "bm25"),
            make_named_candidate(5, 0.6, "bm25"),
            make_named_candidate(6, 0.4, "bm25"),
        ]);
        let rrf = RRFusion::new(60);
        let result = rrf.fuse(vec![set1, set2]).await.unwrap();

        // Top candidate (highest RRF) should always be normalized to 1.0.
        assert!((result.candidates[0].scores.final_score.value() - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_bm25_only_preset_passthrough_ordering() {
        // Simulates [searchers = ["bm25"]] + passthrough fusion.
        // BM25 results should maintain relative ordering (not all 1.0).
        let set = make_set(vec![
            make_named_candidate(1, 0.9, "bm25"),
            make_named_candidate(2, 0.5, "bm25"),
            make_named_candidate(3, 0.1, "bm25"),
        ]);
        // PassThrough is used for single-searcher.
        let result = PassThrough.fuse(vec![set]).await.unwrap();

        assert_eq!(result.candidates.len(), 3);
        // Should be ordered descending, not all 1.0.
        assert!(result.candidates[0].scores.final_score.value() > result.candidates[1].scores.final_score.value());
        assert!(result.candidates[1].scores.final_score.value() > result.candidates[2].scores.final_score.value());
    }
}
