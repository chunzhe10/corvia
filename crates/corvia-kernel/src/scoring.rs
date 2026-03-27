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
//!
//! Each component returns a value in `[0.0, 1.0]`. Weights sum to 1.0, so the
//! un-penalized score is also in `[0.0, 1.0]`. The supersession penalty halves it.

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

/// Normalizer for access frequency: `ln(101.0)`.
/// Caps the frequency component at ~1.0 when `access_count >= 100`.
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
/// All fields are `Copy` — the struct is cheap to pass by value or reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetentionParams {
    /// Classification of the knowledge entry (determines decay alpha).
    pub memory_type: MemoryType,
    /// Days elapsed since the entry was created. Negative values are clamped to 0.
    pub days_since_creation: f64,
    /// Number of times the entry has been accessed.
    pub access_count: u32,
    /// Days since last access. `None` means the entry has never been accessed.
    pub days_since_access: Option<f64>,
    /// Number of inbound graph edges pointing to this entry.
    pub inbound_edges: u32,
    /// Confidence score in `[0.0, 1.0]`. `None` defaults to 0.7.
    pub confidence: Option<f32>,
    /// Whether this entry has been superseded by a newer entry.
    pub is_superseded: bool,
}

impl Default for RetentionParams {
    fn default() -> Self {
        Self {
            memory_type: MemoryType::default(),
            days_since_creation: 0.0,
            access_count: 0,
            days_since_access: None,
            inbound_edges: 0,
            confidence: None,
            is_superseded: false,
        }
    }
}

/// Returns the decay exponent (alpha) for a memory type.
///
/// Special case: Structural entries that are superseded use `alpha = 0.60` (Episodic rate)
/// instead of `alpha = 0` so they actually decay.
#[inline]
#[must_use]
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

/// Individual component scores for observability/logging.
#[derive(Debug, Clone, Copy)]
pub struct ScoreBreakdown {
    /// Time-decay component (weighted).
    pub d_score: f32,
    /// Access-signal component (weighted).
    pub a_score: f32,
    /// Graph-connectivity component (weighted).
    pub g_score: f32,
    /// Confidence component (weighted).
    pub c_score: f32,
    /// Final composite score (after supersession penalty, clamped to 0.0-1.0).
    pub total: f32,
}

/// Compute the retention score with individual component breakdown.
///
/// Returns the composite score and each weighted component for structured logging.
#[inline]
#[must_use]
pub fn compute_retention_score_breakdown(params: &RetentionParams) -> ScoreBreakdown {
    let alpha = alpha_for_type(params.memory_type, params.is_superseded);

    let d = (W_DECAY * decay_component(params.days_since_creation, alpha)) as f32;
    let a = (W_ACCESS * access_component(params.access_count, params.days_since_access)) as f32;
    let g = (W_GRAPH * graph_component(params.inbound_edges)) as f32;
    let c = (W_CONFIDENCE * confidence_component(params.confidence)) as f32;

    let mut total = d + a + g + c;
    if params.is_superseded {
        total *= SUPERSESSION_PENALTY as f32;
    }
    if total.is_nan() {
        total = 0.0;
    }
    total = total.clamp(0.0, 1.0);

    ScoreBreakdown { d_score: d, a_score: a, g_score: g, c_score: c, total }
}

/// Compute the composite retention score.
///
/// Returns a value in `[0.0, 1.0]` (clamped). Higher means the entry is more
/// likely to remain in its current tier. NaN inputs are treated as 0.
#[inline]
#[must_use]
pub fn compute_retention_score(params: &RetentionParams) -> f32 {
    compute_retention_score_breakdown(params).total
}

/// Determine whether the entry should transition to a different tier.
///
/// Returns `Some(new_tier)` if a transition is warranted, `None` if the entry
/// stays in its current tier. Hysteresis gaps prevent oscillation at boundaries.
#[inline]
#[must_use]
pub fn determine_tier_transition(current_tier: Tier, score: f32) -> Option<Tier> {
    match current_tier {
        Tier::Hot if score < THRESHOLD_HOT_TO_WARM => Some(Tier::Warm),
        Tier::Warm if score >= THRESHOLD_WARM_TO_HOT => Some(Tier::Hot),
        Tier::Warm if score < THRESHOLD_WARM_TO_COLD => Some(Tier::Cold),
        Tier::Cold if score >= THRESHOLD_COLD_TO_WARM => Some(Tier::Warm),
        Tier::Cold if score < THRESHOLD_COLD_TO_FORGOTTEN => Some(Tier::Forgotten),
        _ => None, // includes Forgotten (terminal) and all in-range cases
    }
}

// ── Component functions (private) ───────────────────────────────────────────

/// Time decay: `D(t, alpha) = (1 + t_days)^(-alpha)`.
///
/// `t_days` is clamped to >= 0. Short-circuits to 1.0 when alpha is 0 (Structural).
#[inline]
fn decay_component(days_since_creation: f64, alpha: f64) -> f64 {
    if alpha == 0.0 {
        return 1.0;
    }
    let t = sanitize_f64(days_since_creation);
    (1.0 + t).powf(-alpha)
}

/// Access signal: frequency weighted by recency.
///
/// `A = (ln(1 + count) / FREQ_NORMALIZER) * (1 + days_since_access)^(-0.3)`
///
/// Returns 0 for never-accessed entries. Frequency is capped at 1.0.
#[inline]
fn access_component(access_count: u32, days_since_access: Option<f64>) -> f64 {
    match days_since_access {
        None => 0.0,
        Some(days) => {
            let freq = ((1.0 + access_count as f64).ln() / FREQ_NORMALIZER).min(1.0);
            let recency = (1.0 + sanitize_f64(days)).powf(-ACCESS_RECENCY_ALPHA);
            freq * recency
        }
    }
}

/// Graph connectivity: `min(inbound_edges / 10, 1.0)`.
#[inline]
fn graph_component(inbound_edges: u32) -> f64 {
    (inbound_edges as f64 / GRAPH_EDGE_CAP).min(1.0)
}

/// Confidence: `confidence.unwrap_or(0.7)`, clamped to `[0.0, 1.0]`.
#[inline]
fn confidence_component(confidence: Option<f32>) -> f64 {
    match confidence {
        Some(c) if c.is_nan() => DEFAULT_CONFIDENCE,
        Some(c) => (c as f64).clamp(0.0, 1.0),
        None => DEFAULT_CONFIDENCE,
    }
}

/// Sanitize an f64 input: clamp to >= 0, replace NaN/infinity with 0.
#[inline]
fn sanitize_f64(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() {
        0.0
    } else {
        v.max(0.0)
    }
}

// ── Auto-protection ─────────────────────────────────────────────────────────

/// Parameters for auto-protection evaluation.
///
/// Decoupled from `KnowledgeEntry` to keep scoring pure and independently testable.
/// Uses borrowed `content_role` to avoid per-entry heap allocation in hot loops.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoProtectParams<'a> {
    pub memory_type: MemoryType,
    /// Whether this entry is the HEAD of a supersession chain (not superseded by anything).
    pub is_chain_head: bool,
    pub inbound_edges: u32,
    pub content_role: Option<&'a str>,
    pub confidence: Option<f32>,
}

/// Check whether an entry is auto-protected from Cold → Forgotten transitions.
///
/// Auto-protected entries can still be demoted to Warm or Cold by scoring/budget,
/// but they cannot reach Forgotten tier.
///
/// Rules (any one is sufficient):
/// 1. `memory_type == Structural` (refreshed by ingestion)
/// 2. HEAD of supersession chain (not superseded by anything)
/// 3. `inbound_edges >= 5`
/// 4. `content_role == "decision"` AND `confidence >= 0.9`
#[must_use]
pub fn is_auto_protected(params: &AutoProtectParams<'_>) -> bool {
    // Rule 1: Structural entries
    if params.memory_type == MemoryType::Structural {
        return true;
    }

    // Rule 2: HEAD of supersession chain
    if params.is_chain_head {
        return true;
    }

    // Rule 3: Highly connected entries
    if params.inbound_edges >= 5 {
        return true;
    }

    // Rule 4: High-confidence decisions
    if let Some(role) = params.content_role
        && role == "decision"
        && let Some(conf) = params.confidence
        && conf >= 0.9
    {
        return true;
    }

    false
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approximately equal within tolerance
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── Invariant tests ─────────────────────────────────────────────────────

    #[test]
    fn test_weights_sum_to_one() {
        let sum = W_DECAY + W_ACCESS + W_GRAPH + W_CONFIDENCE;
        assert!((sum - 1.0).abs() < 1e-10, "Weights sum to {sum}, expected 1.0");
    }

    #[test]
    fn test_freq_normalizer_matches_ln_101() {
        assert!((FREQ_NORMALIZER - 101.0_f64.ln()).abs() < 1e-10);
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

    #[test]
    fn test_alpha_non_structural_superseded_unchanged() {
        // Superseded flag only changes alpha for Structural
        assert_eq!(alpha_for_type(MemoryType::Episodic, true), ALPHA_EPISODIC);
        assert_eq!(alpha_for_type(MemoryType::Decisional, true), ALPHA_DECISIONAL);
        assert_eq!(alpha_for_type(MemoryType::Procedural, true), ALPHA_PROCEDURAL);
        assert_eq!(alpha_for_type(MemoryType::Analytical, true), ALPHA_ANALYTICAL);
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
            confidence: Some(0.6),
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(
            approx_eq(score, 0.217, 0.03),
            "Expected ~0.217, got {score}"
        );
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
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(
            approx_eq(score, 0.576, 0.03),
            "Expected ~0.576, got {score}"
        );
        assert_eq!(determine_tier_transition(Tier::Hot, score), None);
    }

    #[test]
    fn test_rfc_example_3_episodic_60d() {
        // Episodic 60d, 0 accesses, 0 edges, conf 0.5 → ~0.105 (Cold)
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 60.0,
            confidence: Some(0.5),
            ..Default::default()
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
        let superseded = RetentionParams { is_superseded: true, ..base };

        let base_score = compute_retention_score(&base);
        let super_score = compute_retention_score(&superseded);

        assert!(
            approx_eq(super_score, base_score * 0.5, 0.01),
            "Expected {}, got {super_score}",
            base_score * 0.5
        );
    }

    #[test]
    fn test_structural_superseded_uses_episodic_alpha() {
        let fresh = RetentionParams {
            memory_type: MemoryType::Structural,
            ..Default::default()
        };
        let old_superseded = RetentionParams {
            memory_type: MemoryType::Structural,
            days_since_creation: 365.0,
            is_superseded: true,
            ..Default::default()
        };

        let fresh_score = compute_retention_score(&fresh);
        let old_score = compute_retention_score(&old_superseded);

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
            confidence: Some(0.7),
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        // D = (1+0)^(-0.6) = 1.0, A = 0, G = 0, C = 0.7
        // score = 0.35*1.0 + 0 + 0 + 0.15*0.7 = 0.455
        assert!(approx_eq(score, 0.455, 0.01), "Expected ~0.455, got {score}");
    }

    #[test]
    fn test_never_accessed_entry_access_is_zero() {
        assert_eq!(access_component(0, None), 0.0);
        assert_eq!(access_component(100, None), 0.0);
    }

    #[test]
    fn test_access_increases_score() {
        let base = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 10.0,
            confidence: Some(0.7),
            ..Default::default()
        };
        let with_access = RetentionParams {
            access_count: 10,
            days_since_access: Some(1.0),
            ..base
        };
        assert!(compute_retention_score(&with_access) > compute_retention_score(&base));
    }

    #[test]
    fn test_nan_days_since_creation_returns_zero() {
        let params = RetentionParams {
            days_since_creation: f64::NAN,
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(!score.is_nan(), "NaN should not propagate");
        assert!(score >= 0.0);
    }

    #[test]
    fn test_nan_days_since_access_returns_valid() {
        let params = RetentionParams {
            access_count: 5,
            days_since_access: Some(f64::NAN),
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(!score.is_nan(), "NaN should not propagate");
    }

    #[test]
    fn test_infinity_days_since_creation() {
        let params = RetentionParams {
            days_since_creation: f64::INFINITY,
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(!score.is_nan());
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_u32_max_access_count_stays_clamped() {
        let params = RetentionParams {
            access_count: u32::MAX,
            days_since_access: Some(0.0),
            ..Default::default()
        };
        let score = compute_retention_score(&params);
        assert!(score <= 1.0, "Score {score} exceeds 1.0 with u32::MAX access_count");
        assert!(score >= 0.0);
    }

    #[test]
    fn test_negative_days_since_access_clamps() {
        let a_neg = access_component(5, Some(-3.0));
        let a_zero = access_component(5, Some(0.0));
        assert_eq!(a_neg, a_zero, "Negative days_since_access should clamp to 0");
    }

    #[test]
    fn test_confidence_zero() {
        assert_eq!(confidence_component(Some(0.0)), 0.0);
    }

    #[test]
    fn test_confidence_over_one_clamped() {
        assert_eq!(confidence_component(Some(1.5)), 1.0);
    }

    #[test]
    fn test_confidence_nan_uses_default() {
        assert_eq!(confidence_component(Some(f32::NAN)), DEFAULT_CONFIDENCE);
    }

    // ── Hysteresis tests ────────────────────────────────────────────────────

    #[test]
    fn test_hysteresis_hot_stays_hot_at_051() {
        assert_eq!(determine_tier_transition(Tier::Hot, 0.51), None);
    }

    #[test]
    fn test_hysteresis_warm_stays_warm_at_059() {
        assert_eq!(determine_tier_transition(Tier::Warm, 0.59), None);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        // score=0.55 stays in whatever tier it's in
        assert_eq!(determine_tier_transition(Tier::Hot, 0.55), None);
        assert_eq!(determine_tier_transition(Tier::Warm, 0.55), None);
    }

    #[test]
    fn test_hot_demotes_to_warm_below_050() {
        assert_eq!(determine_tier_transition(Tier::Hot, 0.49), Some(Tier::Warm));
    }

    #[test]
    fn test_warm_promotes_to_hot_at_060() {
        assert_eq!(determine_tier_transition(Tier::Warm, 0.60), Some(Tier::Hot));
    }

    #[test]
    fn test_warm_demotes_to_cold_below_025() {
        assert_eq!(determine_tier_transition(Tier::Warm, 0.24), Some(Tier::Cold));
    }

    #[test]
    fn test_cold_promotes_to_warm_at_035() {
        assert_eq!(determine_tier_transition(Tier::Cold, 0.35), Some(Tier::Warm));
    }

    #[test]
    fn test_cold_demotes_to_forgotten_below_005() {
        assert_eq!(determine_tier_transition(Tier::Cold, 0.04), Some(Tier::Forgotten));
    }

    #[test]
    fn test_cold_stays_cold_at_010() {
        assert_eq!(determine_tier_transition(Tier::Cold, 0.10), None);
    }

    #[test]
    fn test_forgotten_is_terminal() {
        assert_eq!(determine_tier_transition(Tier::Forgotten, 0.99), None);
        assert_eq!(determine_tier_transition(Tier::Forgotten, 0.0), None);
    }

    // Exact boundary tests
    #[test]
    fn test_exact_boundary_hot_at_050_stays() {
        assert_eq!(determine_tier_transition(Tier::Hot, 0.50), None);
    }

    #[test]
    fn test_exact_boundary_warm_at_025_stays() {
        assert_eq!(determine_tier_transition(Tier::Warm, 0.25), None);
    }

    #[test]
    fn test_exact_boundary_cold_at_005_stays() {
        assert_eq!(determine_tier_transition(Tier::Cold, 0.05), None);
    }

    // ── Component unit tests ────────────────────────────────────────────────

    #[test]
    fn test_decay_structural_never_decays() {
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
        assert!((a100 - a10) < (a10 - a1) * 2.0);
    }

    #[test]
    fn test_access_component_max_at_100_accesses() {
        // With 100 accesses and 0 days since access, frequency should be ~1.0
        let a = access_component(100, Some(0.0));
        assert!((a - 1.0).abs() < 0.01, "Expected ~1.0, got {a}");
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
            access_count: 1000,
            days_since_access: Some(0.0),
            inbound_edges: 100,
            confidence: Some(1.0),
            ..Default::default()
        };
        let max_score = compute_retention_score(&max_params);
        assert!(max_score <= 1.0, "Max score {max_score} exceeds 1.0");
        assert!(max_score > 0.9, "Max score {max_score} should be near 1.0");

        // Minimum possible score
        let min_params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 100000.0,
            confidence: Some(0.0),
            is_superseded: true,
            ..Default::default()
        };
        let min_score = compute_retention_score(&min_params);
        assert!(min_score >= 0.0, "Min score {min_score} is negative");
        assert!(min_score < 0.01, "Min score {min_score} should be near 0.0");
    }

    // ── Default ─────────────────────────────────────────────────────────────

    #[test]
    fn test_retention_params_default() {
        let params = RetentionParams::default();
        assert_eq!(params.memory_type, MemoryType::Episodic);
        assert_eq!(params.days_since_creation, 0.0);
        assert_eq!(params.access_count, 0);
        assert!(params.days_since_access.is_none());
        assert_eq!(params.inbound_edges, 0);
        assert!(params.confidence.is_none());
        assert!(!params.is_superseded);
    }

    // ── Auto-protection tests ──────────────────────────────────────────────

    fn base_auto_protect_params() -> AutoProtectParams<'static> {
        AutoProtectParams {
            memory_type: MemoryType::Episodic,
            is_chain_head: false,
            inbound_edges: 0,
            content_role: None,
            confidence: None,
        }
    }

    #[test]
    fn test_auto_protect_structural() {
        let params = AutoProtectParams {
            memory_type: MemoryType::Structural,
            ..base_auto_protect_params()
        };
        assert!(is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_chain_head() {
        let params = AutoProtectParams {
            is_chain_head: true,
            ..base_auto_protect_params()
        };
        assert!(is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_high_inbound_edges() {
        let params = AutoProtectParams {
            inbound_edges: 5,
            ..base_auto_protect_params()
        };
        assert!(is_auto_protected(&params));

        // 4 edges: not protected
        let params4 = AutoProtectParams {
            inbound_edges: 4,
            ..base_auto_protect_params()
        };
        assert!(!is_auto_protected(&params4));
    }

    #[test]
    fn test_auto_protect_decision_high_confidence() {
        let params = AutoProtectParams {
            content_role: Some("decision"),
            confidence: Some(0.9),
            ..base_auto_protect_params()
        };
        assert!(is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_decision_low_confidence() {
        let params = AutoProtectParams {
            content_role: Some("decision"),
            confidence: Some(0.89),
            ..base_auto_protect_params()
        };
        assert!(!is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_non_decision_high_confidence() {
        let params = AutoProtectParams {
            content_role: Some("design"),
            confidence: Some(0.95),
            ..base_auto_protect_params()
        };
        assert!(!is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_none_matches_nothing() {
        let params = base_auto_protect_params();
        assert!(!is_auto_protected(&params));
    }

    #[test]
    fn test_auto_protect_decision_no_confidence() {
        let params = AutoProtectParams {
            content_role: Some("decision"),
            confidence: None,
            ..base_auto_protect_params()
        };
        assert!(!is_auto_protected(&params));
    }

    // ── ScoreBreakdown tests ──────────────────────────────────────────────

    #[test]
    fn test_breakdown_components_sum_to_total() {
        let params = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 30.0,
            access_count: 10,
            days_since_access: Some(5.0),
            inbound_edges: 3,
            confidence: Some(0.8),
            is_superseded: false,
        };

        let breakdown = compute_retention_score_breakdown(&params);
        let sum = breakdown.d_score + breakdown.a_score + breakdown.g_score + breakdown.c_score;
        assert!(approx_eq(breakdown.total, sum, 0.01),
            "total ({}) should equal sum of components ({})", breakdown.total, sum);
    }

    #[test]
    fn test_breakdown_superseded_penalty() {
        let base = RetentionParams {
            memory_type: MemoryType::Episodic,
            days_since_creation: 10.0,
            access_count: 5,
            days_since_access: Some(2.0),
            inbound_edges: 2,
            confidence: Some(0.8),
            is_superseded: false,
        };

        let superseded = RetentionParams {
            is_superseded: true,
            ..base
        };

        let base_bd = compute_retention_score_breakdown(&base);
        let sup_bd = compute_retention_score_breakdown(&superseded);

        // Superseded total should be roughly half of base total
        // (components may differ due to alpha change for Structural)
        assert!(sup_bd.total < base_bd.total,
            "superseded ({}) should be lower than base ({})", sup_bd.total, base_bd.total);
    }

    #[test]
    fn test_breakdown_matches_compute_retention_score() {
        let params = RetentionParams {
            memory_type: MemoryType::Decisional,
            days_since_creation: 45.0,
            access_count: 20,
            days_since_access: Some(3.0),
            inbound_edges: 5,
            confidence: Some(0.9),
            is_superseded: false,
        };

        let score = compute_retention_score(&params);
        let breakdown = compute_retention_score_breakdown(&params);

        assert!(approx_eq(score, breakdown.total, 0.001),
            "compute_retention_score ({}) should match breakdown.total ({})", score, breakdown.total);
    }
}
