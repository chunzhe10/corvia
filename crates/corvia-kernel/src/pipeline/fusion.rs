//! Fusion trait and implementations for merging multiple searcher results.
//!
//! When the pipeline has a single searcher, [`PassThrough`] applies min-max
//! normalization. When multiple searchers produce results, a fusion strategy
//! (e.g., RRF in Phase 2c) merges and re-scores them.

use std::collections::HashMap;
use std::time::Instant;

use async_trait::async_trait;
use corvia_common::errors::Result;

use super::{NormalizedScore, RankedCandidate, RankedSet, StageMetrics};

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
        let mut by_id: HashMap<uuid::Uuid, RankedCandidate> = HashMap::new();
        for set in &sets {
            for candidate in &set.candidates {
                let id = candidate.entry.id;
                match by_id.get(&id) {
                    Some(existing)
                        if existing.scores.final_score.value()
                            >= candidate.scores.final_score.value() =>
                    {
                        // Keep existing (higher score).
                    }
                    _ => {
                        by_id.insert(id, candidate.clone());
                    }
                }
            }
        }

        let mut candidates: Vec<RankedCandidate> = by_id.into_values().collect();

        // Min-max normalize.
        if !candidates.is_empty() {
            let (min_score, max_score) = candidates.iter().fold((f32::MAX, f32::MIN), |(mn, mx), c| {
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
        candidates.sort_unstable_by(|a, b| {
            b.scores
                .final_score
                .value()
                .partial_cmp(&a.scores.final_score.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
}
