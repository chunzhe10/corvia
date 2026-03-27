//! Retention scoring for tiered knowledge lifecycle.
//!
//! Pure computation — no I/O, no async, no store access. Computes the composite
//! retention score that drives tier classification decisions.
//!
//! # Formula
//!
//! ```text
//! retention_score = 0.35 * D(t, alpha) + 0.30 * A(count, recency) + 0.20 * G(edges) + 0.15 * C(conf)
//! if superseded: retention_score *= 0.5
//! ```

use corvia_common::types::{MemoryType, Tier};

// ── Weight constants ────────────────────────────────────────────────────────

/// Weight for time-decay component.
const W_DECAY: f64 = 0.35;
/// Weight for access-signal component.
const W_ACCESS: f64 = 0.30;
/// Weight for graph-connectivity component.
const W_GRAPH: f64 = 0.20;
/// Weight for confidence component.
const W_CONFIDENCE: f64 = 0.15;

/// Normalizer for access frequency: `log(101.0)`.
const FREQ_NORMALIZER: f64 = 4.615_120_516_934_844; // ln(101) precomputed

/// Supersession penalty multiplier.
const SUPERSESSION_PENALTY: f64 = 0.5;

/// Default confidence when none is provided.
const DEFAULT_CONFIDENCE: f64 = 0.7;

/// Access recency decay exponent.
const ACCESS_RECENCY_ALPHA: f64 = 0.3;

/// Graph connectivity cap (inbound edges).
const GRAPH_EDGE_CAP: f64 = 10.0;

// ── Alpha values per MemoryType ─────────────────────────────────────────────

const ALPHA_STRUCTURAL: f64 = 0.0;
const ALPHA_PROCEDURAL: f64 = 0.10;
const ALPHA_DECISIONAL: f64 = 0.15;
const ALPHA_ANALYTICAL: f64 = 0.30;
const ALPHA_EPISODIC: f64 = 0.60;

// ── Tier transition thresholds (with hysteresis) ────────────────────────────

const THRESHOLD_HOT_TO_WARM: f32 = 0.50;
const THRESHOLD_WARM_TO_COLD: f32 = 0.25;
const THRESHOLD_COLD_TO_FORGOTTEN: f32 = 0.05;
const THRESHOLD_COLD_TO_WARM: f32 = 0.35;
const THRESHOLD_WARM_TO_HOT: f32 = 0.60;

// ── Public API ──────────────────────────────────────────────────────────────

/// Inputs for retention score computation.
///
/// Decoupled from `KnowledgeEntry` to keep scoring pure and independently testable.
#[derive(Debug, Clone)]
pub struct RetentionParams {
    pub memory_type: MemoryType,
    pub days_since_creation: f64,
    pub access_count: u32,
    /// `None` means the entry has never been accessed.
    pub days_since_access: Option<f64>,
    pub inbound_edges: u32,
    pub confidence: Option<f32>,
    pub is_superseded: bool,
}

/// Returns the decay exponent (alpha) for a memory type.
///
/// Special case: Structural entries that are superseded use `alpha = 0.60` (Episodic rate)
/// instead of `alpha = 0` so they actually decay.
pub fn alpha_for_type(memory_type: MemoryType, is_superseded: bool) -> f64 {
    match memory_type {
        MemoryType::Structural if is_superseded => ALPHA_EPISODIC,
        MemoryType::Structural => ALPHA_STRUCTURAL,
        MemoryType::Procedural => ALPHA_PROCEDURAL,
        MemoryType::Decisional => ALPHA_DECISIONAL,
        MemoryType::Analytical => ALPHA_ANALYTICAL,
        MemoryType::Episodic => ALPHA_EPISODIC,
    }
}

/// Compute the composite retention score.
///
/// Returns a value in `[0.0, 1.0]` (clamped). Higher means the entry is more
/// likely to remain in its current tier.
pub fn compute_retention_score(params: &RetentionParams) -> f32 {
    let alpha = alpha_for_type(params.memory_type, params.is_superseded);

    let d = decay_component(params.days_since_creation, alpha);
    let a = access_component(params.access_count, params.days_since_access);
    let g = graph_component(params.inbound_edges);
    let c = confidence_component(params.confidence);

    let mut score = W_DECAY * d + W_ACCESS * a + W_GRAPH * g + W_CONFIDENCE * c;

    if params.is_superseded {
        score *= SUPERSESSION_PENALTY;
    }

    (score as f32).clamp(0.0, 1.0)
}

/// Determine whether the entry should transition to a different tier.
///
/// Returns `Some(new_tier)` if a transition is warranted, `None` if the entry
/// stays in its current tier. Hysteresis gaps prevent oscillation at boundaries.
pub fn determine_tier_transition(current_tier: Tier, score: f32) -> Option<Tier> {
    match current_tier {
        Tier::Hot => {
            if score < THRESHOLD_HOT_TO_WARM {
                Some(Tier::Warm)
            } else {
                None
            }
        }
        Tier::Warm => {
            if score >= THRESHOLD_WARM_TO_HOT {
                Some(Tier::Hot)
            } else if score < THRESHOLD_WARM_TO_COLD {
                Some(Tier::Cold)
            } else {
                None
            }
        }
        Tier::Cold => {
            if score >= THRESHOLD_COLD_TO_WARM {
                Some(Tier::Warm)
            } else if score < THRESHOLD_COLD_TO_FORGOTTEN {
                Some(Tier::Forgotten)
            } else {
                None
            }
        }
        Tier::Forgotten => {
            // Forgotten is the terminal state — no promotion back.
            None
        }
    }
}

// ── Component functions (private) ───────────────────────────────────────────

/// Time decay: `D(t, alpha) = (1 + t_days)^(-alpha)`.
///
/// `t_days` is clamped to >= 0.
fn decay_component(days_since_creation: f64, alpha: f64) -> f64 {
    let t = days_since_creation.max(0.0);
    (1.0 + t).powf(-alpha)
}

/// Access signal: frequency weighted by recency.
///
/// `A = (log(1 + count) / FREQ_NORMALIZER) * (1 + days_since_access)^(-0.3)`
///
/// Returns 0 for never-accessed entries.
fn access_component(access_count: u32, days_since_access: Option<f64>) -> f64 {
    match days_since_access {
        None => 0.0,
        Some(days) => {
            let freq = (1.0 + access_count as f64).ln() / FREQ_NORMALIZER;
            let recency = (1.0 + days.max(0.0)).powf(-ACCESS_RECENCY_ALPHA);
            freq * recency
        }
    }
}

/// Graph connectivity: `min(inbound_edges / 10, 1.0)`.
fn graph_component(inbound_edges: u32) -> f64 {
    (inbound_edges as f64 / GRAPH_EDGE_CAP).min(1.0)
}

/// Confidence: `confidence.unwrap_or(0.7)`.
fn confidence_component(confidence: Option<f32>) -> f64 {
    confidence.map(|c| c as f64).unwrap_or(DEFAULT_CONFIDENCE)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approximately equal within tolerance
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── Alpha values ────────────────────────────────────────────────────────

    #[test]
    fn test_alpha_structural() {
        assert_eq!(alpha_for_type(MemoryType::Structural, false), 0.0);
    }

    #[test]
    fn test_alpha_procedural() {
        assert_eq!(alpha_for_type(MemoryType::Procedural, false), 0.10);
    }

    #[test]
    fn test_alpha_decisional() {
        assert_eq!(alpha_for_type(MemoryType::Decisional, false), 0.15);
    }

    #[test]
    fn test_alpha_analytical() {
        assert_eq!(alpha_for_type(MemoryType::Analytical, false), 0.30);
    }

    #[test]
    fn test_alpha_episodic() {
        assert_eq!(alpha_for_type(MemoryType::Episodic, false), 0.60);
    }

    #[test]
    fn test_alpha_structural_superseded_uses_episodic_rate() {
        assert_eq!(alpha_for_type(MemoryType::Structural, true), ALPHA_EPISODIC);
    }

    // ── Worked examples from RFC ────────────────────────────────────────────

    #[test]
    fn test_rfc_example_1_episodic_14d() {
        // Episodic 14d, 2 accesses, 0 edges, conf 0.6 → ~0.217 (Cold)
        // RFC assumes last access ~1 day ago
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 14.0,
            access_count: 2,
            days_since_access: Some(1.0),
            inbound_edges: 0,
            confidence: Some(0.6),
            is_superseded: false,
        };
        let score = compute_retention_score(&params);
        assert!(
            approx_eq(score, 0.217, 0.03),
            "Expected ~0.217, got {score}"
        );
        // Should be Cold tier (below 0.25 warm threshold)
        assert_eq!(
            determine_tier_transition(Tier::Warm, score),
            Some(Tier::Cold)
        );
    }

    #[test]
    fn test_rfc_example_2_decisional_180d() {
        // Decisional 180d, 30 accesses, 6 edges, conf 0.9 → ~0.576 (Hot stays)
        // RFC assumes last access ~2 days ago
        let params = RetentionParams {
            memory_type: MemoryType::Decisional,
            days_since_creation: 180.0,
            access_count: 30,
            days_since_access: Some(2.0),
            inbound_edges: 6,
            confidence: Some(0.9),
            is_superseded: false,
        };
        let score = compute_retention_score(&params);
        assert!(
            approx_eq(score, 0.576, 0.03),
            "Expected ~0.576, got {score}"
        );
        // Hot stays Hot (score >= 0.50)
        assert_eq!(determine_tier_transition(Tier::Hot, score), None);
    }

    #[test]
    fn test_rfc_example_3_episodic_60d() {
        // Episodic 60d, 0 accesses, 0 edges, conf 0.5 → ~0.105 (Cold)
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 60.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: Some(0.5),
            is_superseded: false,
        };
        let score = compute_retention_score(&params);
        assert!(
            approx_eq(score, 0.105, 0.02),
            "Expected ~0.105, got {score}"
        );
    }

    // ── Supersession penalty ────────────────────────────────────────────────

    #[test]
    fn test_superseded_entry_gets_half_penalty() {
        let base = RetentionParams {
            memory_type: MemoryType::Decisional,
            days_since_creation: 30.0,
            access_count: 5,
            days_since_access: Some(2.0),
            inbound_edges: 3,
            confidence: Some(0.8),
            is_superseded: false,
        };
        let mut superseded = base.clone();
        superseded.is_superseded = true;

        let base_score = compute_retention_score(&base);
        let super_score = compute_retention_score(&superseded);

        // Superseded score should be roughly half of base (not exact because
        // alpha changes for Structural, but for Decisional alpha is unchanged)
        assert!(
            approx_eq(super_score, base_score * 0.5, 0.01),
            "Expected {}, got {super_score}",
            base_score * 0.5
        );
    }

    #[test]
    fn test_structural_superseded_uses_episodic_alpha() {
        // Structural + superseded should decay (alpha=0.60 instead of alpha=0)
        let fresh = RetentionParams {
            memory_type: MemoryType::Structural,
            days_since_creation: 0.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: None,
            is_superseded: false,
        };
        let old_superseded = RetentionParams {
            memory_type: MemoryType::Structural,
            days_since_creation: 365.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: None,
            is_superseded: true,
        };

        let fresh_score = compute_retention_score(&fresh);
        let old_score = compute_retention_score(&old_superseded);

        // Fresh structural should have high decay component (alpha=0 means no decay)
        // Old superseded structural should have much lower score
        assert!(
            old_score < fresh_score * 0.3,
            "Old superseded structural ({old_score}) should be much lower than fresh ({fresh_score})"
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_negative_days_clamps_to_zero() {
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: -5.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: Some(0.7),
            is_superseded: false,
        };
        let score = compute_retention_score(&params);
        // With t clamped to 0: D = (1+0)^(-0.6) = 1.0
        // A = 0, G = 0, C = 0.7
        // score = 0.35*1.0 + 0 + 0 + 0.15*0.7 = 0.455
        assert!(
            approx_eq(score, 0.455, 0.01),
            "Expected ~0.455, got {score}"
        );
    }

    #[test]
    fn test_never_accessed_entry_access_is_zero() {
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 10.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: Some(0.7),
            is_superseded: false,
        };
        let score = compute_retention_score(&params);
        // Access component should be exactly 0
        let params_with_access = RetentionParams {
            access_count: 10,
            days_since_access: Some(1.0),
            ..params.clone()
        };
        let score_with_access = compute_retention_score(&params_with_access);
        assert!(score_with_access > score, "Access should increase score");
    }

    // ── Hysteresis tests ────────────────────────────────────────────────────

    #[test]
    fn test_hysteresis_hot_stays_hot_at_051() {
        // score=0.51 — above hot_to_warm threshold (0.50), Hot stays Hot
        assert_eq!(determine_tier_transition(Tier::Hot, 0.51), None);
    }

    #[test]
    fn test_hysteresis_warm_stays_warm_at_059() {
        // score=0.59 — below warm_to_hot threshold (0.60), Warm stays Warm
        assert_eq!(determine_tier_transition(Tier::Warm, 0.59), None);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        // An entry at score=0.55 should stay in whatever tier it's in:
        // - If Hot: 0.55 >= 0.50, stays Hot
        // - If Warm: 0.55 < 0.60, stays Warm
        assert_eq!(determine_tier_transition(Tier::Hot, 0.55), None);
        assert_eq!(determine_tier_transition(Tier::Warm, 0.55), None);
    }

    #[test]
    fn test_hot_demotes_to_warm_below_050() {
        assert_eq!(
            determine_tier_transition(Tier::Hot, 0.49),
            Some(Tier::Warm)
        );
    }

    #[test]
    fn test_warm_promotes_to_hot_at_060() {
        assert_eq!(
            determine_tier_transition(Tier::Warm, 0.60),
            Some(Tier::Hot)
        );
    }

    #[test]
    fn test_warm_demotes_to_cold_below_025() {
        assert_eq!(
            determine_tier_transition(Tier::Warm, 0.24),
            Some(Tier::Cold)
        );
    }

    #[test]
    fn test_cold_promotes_to_warm_at_035() {
        assert_eq!(
            determine_tier_transition(Tier::Cold, 0.35),
            Some(Tier::Warm)
        );
    }

    #[test]
    fn test_cold_demotes_to_forgotten_below_005() {
        assert_eq!(
            determine_tier_transition(Tier::Cold, 0.04),
            Some(Tier::Forgotten)
        );
    }

    #[test]
    fn test_cold_stays_cold_at_010() {
        // Between 0.05 and 0.35 — stays Cold
        assert_eq!(determine_tier_transition(Tier::Cold, 0.10), None);
    }

    #[test]
    fn test_forgotten_is_terminal() {
        // Even high scores can't promote out of Forgotten
        assert_eq!(determine_tier_transition(Tier::Forgotten, 0.99), None);
        assert_eq!(determine_tier_transition(Tier::Forgotten, 0.0), None);
    }

    // ── Component unit tests ────────────────────────────────────────────────

    #[test]
    fn test_decay_structural_never_decays() {
        // alpha=0 means (1+t)^0 = 1.0 always
        assert_eq!(decay_component(0.0, 0.0), 1.0);
        assert_eq!(decay_component(365.0, 0.0), 1.0);
        assert_eq!(decay_component(10000.0, 0.0), 1.0);
    }

    #[test]
    fn test_decay_episodic_drops_fast() {
        let d0 = decay_component(0.0, ALPHA_EPISODIC);
        let d14 = decay_component(14.0, ALPHA_EPISODIC);
        let d60 = decay_component(60.0, ALPHA_EPISODIC);
        assert_eq!(d0, 1.0);
        assert!(d14 < 0.3, "14-day episodic decay should be < 0.3, got {d14}");
        assert!(d60 < 0.1, "60-day episodic decay should be < 0.1, got {d60}");
    }

    #[test]
    fn test_access_component_frequency_scales_logarithmically() {
        let a1 = access_component(1, Some(0.0));
        let a10 = access_component(10, Some(0.0));
        let a100 = access_component(100, Some(0.0));
        assert!(a10 > a1);
        assert!(a100 > a10);
        // Logarithmic: a100/a10 should be less than a10/a1
        assert!((a100 - a10) < (a10 - a1) * 2.0);
    }

    #[test]
    fn test_graph_component_caps_at_10_edges() {
        assert_eq!(graph_component(0), 0.0);
        assert_eq!(graph_component(5), 0.5);
        assert_eq!(graph_component(10), 1.0);
        assert_eq!(graph_component(100), 1.0);
    }

    #[test]
    fn test_confidence_defaults_to_07() {
        assert_eq!(confidence_component(None), 0.7);
        assert!((confidence_component(Some(0.9)) - 0.9).abs() < 1e-6);
    }

    // ── Score range ─────────────────────────────────────────────────────────

    #[test]
    fn test_score_always_in_0_1_range() {
        // Maximum possible score
        let max_params = RetentionParams {
            memory_type: MemoryType::Structural,
            days_since_creation: 0.0,
            access_count: 1000,
            days_since_access: Some(0.0),
            inbound_edges: 100,
            confidence: Some(1.0),
            is_superseded: false,
        };
        let max_score = compute_retention_score(&max_params);
        assert!(max_score <= 1.0, "Max score {max_score} exceeds 1.0");
        assert!(max_score > 0.9, "Max score {max_score} should be near 1.0");

        // Minimum possible score
        let min_params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 100000.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: Some(0.0),
            is_superseded: true,
        };
        let min_score = compute_retention_score(&min_params);
        assert!(min_score >= 0.0, "Min score {min_score} is negative");
        assert!(min_score < 0.01, "Min score {min_score} should be near 0.0");
    }
}
