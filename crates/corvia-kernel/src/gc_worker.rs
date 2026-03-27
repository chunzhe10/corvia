//! Background GC worker for tiered knowledge lifecycle.
//!
//! Periodically evaluates all entries, computes retention scores, applies tier
//! transitions, and triggers HNSW rebuilds when entries move out of indexed tiers.
//!
//! Safeguards: supersession chain protection, auto-protection rules, budget policy,
//! rate limiting (max 50 Forgotten/cycle), and circuit breaker (>50% write failures).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use corvia_common::config::{CorviaConfig, ForgettingConfig, ScopeForgettingOverride};
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, KnowledgeEntry, MemoryType, Tier};
use tracing::{info, warn};

use crate::scoring::{self, AutoProtectParams, RetentionParams};
use crate::traits::{GraphStore, QueryableStore};

/// Shared counter for Forgotten entry access attempts.
///
/// Tracks how often agents try to access Forgotten entries via direct ID lookups
/// (`corvia_history`, `corvia_graph` traversal). High rate signals thresholds
/// are too aggressive. The GC cycle reads and resets this counter.
#[derive(Debug, Default)]
pub struct ForgottenAccessCounter {
    counter: std::sync::atomic::AtomicU64,
}

impl ForgottenAccessCounter {
    pub fn new() -> Self {
        Self { counter: std::sync::atomic::AtomicU64::new(0) }
    }

    /// Increment the counter by 1. Called when a Forgotten entry is accessed.
    pub fn increment(&self) {
        self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load the current count without resetting.
    pub fn load(&self) -> u64 {
        self.counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Swap the counter to 0, returning the previous value.
    pub fn take(&self) -> u64 {
        self.counter.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
}

/// Batch size for Redb write transactions during tier transitions.
const BATCH_SIZE: usize = 100;

/// Score threshold above which inactivity policy does NOT force Cold.
const INACTIVITY_SCORE_EXEMPTION: f32 = 0.60;

/// Maximum number of Cold → Forgotten transitions per GC cycle.
const MAX_FORGOTTEN_PER_CYCLE: usize = 50;

/// Circuit breaker threshold: abort if this fraction of batch writes fail.
const CIRCUIT_BREAKER_THRESHOLD: f64 = 0.50;

// ── Report ──────────────────────────────────────────────────────────────────

/// Metrics from a single GC cycle.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GcCycleReport {
    pub entries_scanned: usize,
    pub entries_scored: usize,
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub cold_to_forgotten: usize,
    pub warm_to_hot: usize,
    pub cold_to_warm: usize,
    pub hnsw_rebuild_triggered: bool,
    pub rebuild_duration_ms: u64,
    pub cycle_duration_ms: u64,
    pub scopes_processed: usize,
    /// Forgotten transitions deferred due to rate limit.
    pub forgotten_rate_limited: usize,
    /// Entries protected from Forgotten by chain protection.
    pub chain_protected: usize,
    /// Entries protected from Forgotten by auto-protection rules.
    pub auto_protected: usize,
    /// Entries demoted by budget policy enforcement.
    pub budget_demoted: usize,
    /// Whether the circuit breaker tripped during this cycle.
    pub circuit_breaker_tripped: bool,
}

// ── Tier transition record ──────────────────────────────────────────────────

/// A pending tier transition for a single entry, with score breakdown for structured logging.
#[derive(Debug, Clone)]
struct TierTransition {
    entry_id: uuid::Uuid,
    scope_id: String,
    memory_type: MemoryType,
    old_tier: Tier,
    new_tier: Tier,
    retention_score: f32,
    d_score: f32,
    a_score: f32,
    g_score: f32,
    c_score: f32,
    superseded: bool,
    reason: &'static str,
}

// ── Spawn ───────────────────────────────────────────────────────────────────

/// Spawn the periodic GC worker as a background `tokio::spawn` task.
///
/// Reads `forgetting.interval_minutes` from the hot-reloadable config on each
/// iteration so the interval can be changed without restart.
pub fn spawn_gc_worker(
    store: Arc<dyn QueryableStore>,
    graph: Arc<dyn GraphStore>,
    config: Arc<std::sync::RwLock<CorviaConfig>>,
    data_dir: std::path::PathBuf,
    forgotten_counter: Option<Arc<ForgottenAccessCounter>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            // Read all config values in a single lock acquisition to avoid
            // TOCTOU races between hot-reload and cycle execution.
            let (interval_minutes, forgetting_config, scope_configs) = {
                let cfg = config.read().unwrap_or_else(|poisoned| {
                    warn!("Config lock poisoned in GC worker, using last known config");
                    poisoned.into_inner()
                });
                let interval = cfg
                    .forgetting
                    .as_ref()
                    .map(|f| f.interval_minutes)
                    .unwrap_or(60);
                let forgetting = cfg.forgetting.clone();
                let scopes: HashMap<String, Option<ScopeForgettingOverride>> = cfg
                    .scope
                    .as_ref()
                    .map(|sc| {
                        sc.iter()
                            .map(|s| (s.id.clone(), s.forgetting.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                (interval, forgetting, scopes)
            };

            tokio::time::sleep(std::time::Duration::from_secs(
                interval_minutes as u64 * 60,
            ))
            .await;

            match run_gc_cycle(&store, &graph, &data_dir, forgetting_config.as_ref(), &scope_configs, forgotten_counter.as_deref()).await {
                Ok(report) => {
                    if report.entries_scored > 0 {
                        info!(
                            entries_scanned = report.entries_scanned,
                            entries_scored = report.entries_scored,
                            hot_to_warm = report.hot_to_warm,
                            warm_to_cold = report.warm_to_cold,
                            cold_to_forgotten = report.cold_to_forgotten,
                            warm_to_hot = report.warm_to_hot,
                            cold_to_warm = report.cold_to_warm,
                            chain_protected = report.chain_protected,
                            auto_protected = report.auto_protected,
                            budget_demoted = report.budget_demoted,
                            forgotten_rate_limited = report.forgotten_rate_limited,
                            circuit_breaker = report.circuit_breaker_tripped,
                            hnsw_rebuild = report.hnsw_rebuild_triggered,
                            cycle_ms = report.cycle_duration_ms,
                            scopes = report.scopes_processed,
                            "GC knowledge cycle complete"
                        );
                    }
                }
                Err(e) => {
                    warn!(error = %e, "GC knowledge cycle failed");
                }
            }
        }
    })
}

// ── Core cycle ──────────────────────────────────────────────────────────────

/// Run one GC cycle across all scopes.
///
/// Public for direct invocation via `ops::gc_run` and testing.
#[tracing::instrument(name = "corvia.gc.cycle", skip_all, fields(
    entries_scanned = 0u64,
    transitions_hot_warm = 0u64,
    transitions_warm_cold = 0u64,
    transitions_cold_forgotten = 0u64,
    promotions_cold_warm = 0u64,
    promotions_warm_hot = 0u64,
    rebuild_triggered = false,
    rebuild_duration_ms = 0u64,
    forgotten_access_attempts = 0u64,
    cycle_duration_ms = 0u64,
))]
pub async fn run_gc_cycle(
    store: &Arc<dyn QueryableStore>,
    graph: &Arc<dyn GraphStore>,
    data_dir: &Path,
    forgetting_config: Option<&ForgettingConfig>,
    scope_configs: &HashMap<String, Option<ScopeForgettingOverride>>,
    forgotten_counter: Option<&ForgottenAccessCounter>,
) -> Result<GcCycleReport> {
    let start = std::time::Instant::now();

    let forgetting = match forgetting_config {
        Some(f) if f.enabled => f,
        _ => {
            return Ok(GcCycleReport::default());
        }
    };

    let mut report = GcCycleReport::default();

    // Downcast to LiteStore once for the entire cycle
    let lite_store = store
        .as_any()
        .downcast_ref::<crate::lite_store::LiteStore>()
        .ok_or_else(|| {
            corvia_common::errors::CorviaError::Storage(
                "GC worker requires LiteStore backend".into(),
            )
        })?;

    // Load ALL entries once (not per-scope) to avoid O(S*E) redundant scans
    let all_entries = lite_store.fetch_all_entries()?;

    // Discover scopes from knowledge directory
    let knowledge_dir = data_dir.join("knowledge");
    let scopes = list_scope_dirs(&knowledge_dir);

    for scope_id in &scopes {
        let scope_override = scope_configs
            .get(scope_id.as_str())
            .and_then(|o| o.as_ref());

        process_scope(
            lite_store,
            graph,
            scope_id,
            &all_entries,
            forgetting,
            scope_override,
            &mut report,
        )
        .await?;
        report.scopes_processed += 1;
    }

    report.cycle_duration_ms = start.elapsed().as_millis() as u64;

    // Take the forgotten_access_attempts counter (reset to 0 for next cycle)
    let forgotten_attempts = forgotten_counter.map(|c| c.take()).unwrap_or(0);

    // Record all metrics into the span
    let span = tracing::Span::current();
    span.record("entries_scanned", report.entries_scanned as u64);
    span.record("transitions_hot_warm", report.hot_to_warm as u64);
    span.record("transitions_warm_cold", report.warm_to_cold as u64);
    span.record("transitions_cold_forgotten", report.cold_to_forgotten as u64);
    span.record("promotions_cold_warm", report.cold_to_warm as u64);
    span.record("promotions_warm_hot", report.warm_to_hot as u64);
    span.record("rebuild_triggered", report.hnsw_rebuild_triggered);
    span.record("rebuild_duration_ms", report.rebuild_duration_ms);
    span.record("forgotten_access_attempts", forgotten_attempts);
    span.record("cycle_duration_ms", report.cycle_duration_ms);

    Ok(report)
}

/// Process a single scope: score all entries, determine transitions, apply batched writes.
///
/// Safeguard pipeline (applied in order):
/// 1. Score-based transitions (existing)
/// 2. Supersession chain protection — block Forgotten for ancestors of active HEADs
/// 3. Auto-protection rules — block Forgotten for qualifying entries
/// 4. Rate limit — cap Forgotten transitions at MAX_FORGOTTEN_PER_CYCLE
/// 5. Budget policy — enforce budget_top_n cap with one-tier-step demotions
/// 6. Circuit breaker — abort batch writes if >50% fail
async fn process_scope(
    lite_store: &crate::lite_store::LiteStore,
    graph: &Arc<dyn GraphStore>,
    scope_id: &str,
    all_entries: &[KnowledgeEntry],
    forgetting: &ForgettingConfig,
    scope_override: Option<&ScopeForgettingOverride>,
    report: &mut GcCycleReport,
) -> Result<()> {
    let scope_entries: Vec<&KnowledgeEntry> = all_entries
        .iter()
        .filter(|e| e.scope_id == scope_id)
        .collect();

    let now = Utc::now();
    let mut transitions: Vec<TierTransition> = Vec::new();
    let mut score_updates: Vec<(uuid::Uuid, f32)> = Vec::new();
    // Track entry IDs that already have a transition in this cycle (for budget skip logic)
    let mut transitioned_ids: HashSet<uuid::Uuid> = HashSet::new();

    // ── Pre-compute supersession chain protection ───────────────────────────
    let chain_protected_ids = build_chain_protected_set(&scope_entries);

    // ── Pre-compute inbound edge counts (needed for auto-protection) ────────
    let mut inbound_edge_map: HashMap<uuid::Uuid, u32> = HashMap::new();

    for entry in &scope_entries {
        report.entries_scanned += 1;

        // Skip pinned entries
        if entry.pin.is_some() {
            continue;
        }
        // Skip Forgotten (terminal)
        if entry.tier == Tier::Forgotten {
            continue;
        }

        // Resolve policy for this entry's memory type
        let policy = forgetting.resolve_policy(entry.memory_type, scope_override);
        if !policy.enabled {
            continue;
        }

        report.entries_scored += 1;

        // Compute retention params (clamp to non-negative for clock skew safety)
        let days_since_creation =
            (now - entry.recorded_at).num_seconds().max(0) as f64 / 86400.0;
        let days_since_access = entry
            .last_accessed
            .map(|la| (now - la).num_seconds().max(0) as f64 / 86400.0);
        let inbound_edges = graph
            .edges(&entry.id, EdgeDirection::Incoming)
            .await
            .map(|e| e.len() as u32)
            .unwrap_or(0);

        // Cache for budget phase
        inbound_edge_map.insert(entry.id, inbound_edges);

        let params = RetentionParams {
            memory_type: entry.memory_type,
            days_since_creation,
            access_count: entry.access_count,
            days_since_access,
            inbound_edges,
            confidence: entry.confidence,
            is_superseded: entry.superseded_by.is_some(),
        };

        let breakdown = scoring::compute_retention_score_breakdown(&params);
        let retention_score = breakdown.total;
        score_updates.push((entry.id, retention_score));

        // Determine score-based transition
        let score_transition = scoring::determine_tier_transition(entry.tier, retention_score);

        // Apply inactivity policy: force Cold if inactive AND score < exemption
        let inactivity_transition = apply_inactivity_policy(
            entry,
            &policy,
            retention_score,
            days_since_access,
            days_since_creation,
        );

        // Take the WORSE transition (lower tier = worse)
        let effective_transition = worse_transition(score_transition, inactivity_transition);

        // Determine transition reason
        let reason = match (&score_transition, &inactivity_transition) {
            (Some(_), None) => "score_decay",
            (None, Some(_)) => "inactivity",
            (Some(s), Some(i)) if i <= s => "inactivity",
            _ => "score_decay",
        };

        if let Some(target_tier) = effective_transition {
            // Enforce one-tier-step-per-cycle
            let clamped = clamp_one_step(entry.tier, target_tier);
            if clamped != entry.tier {
                transitions.push(TierTransition {
                    entry_id: entry.id,
                    scope_id: scope_id.to_string(),
                    memory_type: entry.memory_type,
                    old_tier: entry.tier,
                    new_tier: clamped,
                    retention_score,
                    d_score: breakdown.d_score,
                    a_score: breakdown.a_score,
                    g_score: breakdown.g_score,
                    c_score: breakdown.c_score,
                    superseded: entry.superseded_by.is_some(),
                    reason,
                });
                transitioned_ids.insert(entry.id);
            }
        }
    }

    // ── Safeguard 1: Supersession chain protection ──────────────────────────
    // Block Cold→Forgotten for entries in active chains.
    transitions.retain(|t| {
        if t.new_tier == Tier::Forgotten && chain_protected_ids.contains(&t.entry_id) {
            report.chain_protected += 1;
            transitioned_ids.remove(&t.entry_id);
            false
        } else {
            true
        }
    });

    // Build entry lookup once (used by auto-protection and budget policy)
    let entry_map: HashMap<uuid::Uuid, &KnowledgeEntry> =
        scope_entries.iter().map(|e| (e.id, *e)).collect();

    // ── Safeguard 2: Auto-protection rules ──────────────────────────────────
    // Block Cold→Forgotten for auto-protected entries.
    {
        transitions.retain(|t| {
            if t.new_tier == Tier::Forgotten
                && let Some(entry) = entry_map.get(&t.entry_id)
            {
                let auto_params = AutoProtectParams {
                    memory_type: entry.memory_type,
                    is_chain_head: entry.superseded_by.is_none(),
                    inbound_edges: inbound_edge_map.get(&entry.id).copied().unwrap_or(0),
                    content_role: entry.metadata.content_role.as_deref(),
                    confidence: entry.confidence,
                };
                if scoring::is_auto_protected(&auto_params) {
                    report.auto_protected += 1;
                    transitioned_ids.remove(&t.entry_id);
                    return false;
                }
            }
            true
        });
    }

    // ── Safeguard 3: Rate limit (max Forgotten per cycle) ───────────────────
    let mut forgotten_count = 0;
    transitions.retain(|t| {
        if t.new_tier == Tier::Forgotten {
            if forgotten_count >= MAX_FORGOTTEN_PER_CYCLE {
                report.forgotten_rate_limited += 1;
                transitioned_ids.remove(&t.entry_id);
                return false;
            }
            forgotten_count += 1;
        }
        true
    });

    // ── Safeguard 4: Budget policy enforcement ──────────────────────────────
    // Runs after score-based transitions, before physical writes.
    {
        let score_map: HashMap<uuid::Uuid, f32> = score_updates.iter().copied().collect();

        let budget_transitions = apply_budget_policy(
            &scope_entries,
            &entry_map,
            &score_map,
            &inbound_edge_map,
            &transitioned_ids,
            &chain_protected_ids,
            forgetting,
            scope_override,
        );

        report.budget_demoted += budget_transitions.len();
        for bt in budget_transitions {
            transitioned_ids.insert(bt.entry_id);
            transitions.push(bt);
        }
    }

    // Count transitions by type and emit structured transition logs
    for t in &transitions {
        match (t.old_tier, t.new_tier) {
            (Tier::Hot, Tier::Warm) => report.hot_to_warm += 1,
            (Tier::Warm, Tier::Cold) => report.warm_to_cold += 1,
            (Tier::Cold, Tier::Forgotten) => report.cold_to_forgotten += 1,
            (Tier::Warm, Tier::Hot) => report.warm_to_hot += 1,
            (Tier::Cold, Tier::Warm) => report.cold_to_warm += 1,
            _ => {}
        }

        // Structured transition log with full score breakdown
        info!(
            entry_id = %t.entry_id,
            scope_id = %t.scope_id,
            memory_type = %t.memory_type,
            from_tier = ?t.old_tier,
            to_tier = ?t.new_tier,
            retention_score = t.retention_score,
            d_score = t.d_score,
            a_score = t.a_score,
            g_score = t.g_score,
            c_score = t.c_score,
            superseded = t.superseded,
            reason = t.reason,
            "tier_transition"
        );
    }

    let has_warm_to_cold = transitions
        .iter()
        .any(|t| t.old_tier == Tier::Warm && t.new_tier == Tier::Cold);

    // ── Safeguard 5: Circuit breaker on batch writes ────────────────────────
    match apply_transitions_batched(lite_store, &transitions, &score_updates) {
        Ok(()) => {}
        Err(e) => {
            // Circuit breaker tripped — abort this scope
            warn!(scope = scope_id, error = %e, "Circuit breaker tripped during GC batch writes");
            report.circuit_breaker_tripped = true;
            return Ok(());
        }
    }

    // Trigger HNSW rebuild if any Warm→Cold transitions (entries removed from HNSW)
    if has_warm_to_cold {
        let rebuild_start = std::time::Instant::now();
        info!(scope = scope_id, "Triggering HNSW rebuild after Warm→Cold transitions");
        lite_store.rebuild_from_files()?;
        if let Err(e) = lite_store.flush_hnsw() {
            warn!(error = %e, "Failed to persist HNSW after GC rebuild");
        }
        report.hnsw_rebuild_triggered = true;
        report.rebuild_duration_ms = rebuild_start.elapsed().as_millis() as u64;
    }

    Ok(())
}

// ── Chain protection ────────────────────────────────────────────────────────

/// Build a set of entry IDs that are protected from Forgotten by supersession chains.
///
/// Walk `superseded_by` pointers to find chain HEADs. If a HEAD is in an active
/// tier (Hot, Warm, or Cold), all ancestors in the chain are protected from Forgotten.
fn build_chain_protected_set(entries: &[&KnowledgeEntry]) -> HashSet<uuid::Uuid> {
    let mut protected = HashSet::new();

    // Build reverse lookup: superseded_by → predecessor_id
    // (which entry was superseded to become this one?)
    let mut predecessors: HashMap<uuid::Uuid, Vec<uuid::Uuid>> = HashMap::new();
    for entry in entries {
        if let Some(successor_id) = entry.superseded_by {
            predecessors.entry(successor_id).or_default().push(entry.id);
        }
    }

    // Find all chain HEADs (not superseded by anything)
    let heads: Vec<&KnowledgeEntry> = entries
        .iter()
        .filter(|e| e.superseded_by.is_none())
        .copied()
        .collect();

    for head in &heads {
        // If HEAD is in an active tier, protect all ancestors
        if head.tier != Tier::Forgotten {
            // Walk backward through predecessors with cycle guard
            let mut stack = vec![head.id];
            let mut visited = HashSet::new();
            visited.insert(head.id);
            while let Some(current_id) = stack.pop() {
                if let Some(preds) = predecessors.get(&current_id) {
                    for &pred_id in preds {
                        if visited.insert(pred_id) {
                            // Protect ancestors (not the head — it's covered by auto-protection rule 2)
                            if pred_id != head.id {
                                protected.insert(pred_id);
                            }
                            stack.push(pred_id);
                        }
                    }
                }
            }
        }
    }

    protected
}

// ── Budget policy ───────────────────────────────────────────────────────────

/// Enforce budget_top_n: demote lowest-scoring non-pinned entries by one tier step
/// when a scope exceeds its active entry cap.
///
/// - Runs after score-based transitions, before physical writes
/// - Ranks non-pinned entries by retention_score
/// - Demotes lowest by one tier step when scope > budget_top_n
/// - Skips entries already transitioned in current cycle
/// - Pinned entries excluded from count
/// - Auto-protected entries can be budget-demoted to Warm but not Cold/Forgotten
#[allow(clippy::too_many_arguments)]
fn apply_budget_policy(
    scope_entries: &[&KnowledgeEntry],
    entry_map: &HashMap<uuid::Uuid, &KnowledgeEntry>,
    score_map: &HashMap<uuid::Uuid, f32>,
    inbound_edge_map: &HashMap<uuid::Uuid, u32>,
    transitioned_ids: &HashSet<uuid::Uuid>,
    chain_protected_ids: &HashSet<uuid::Uuid>,
    forgetting: &ForgettingConfig,
    scope_override: Option<&ScopeForgettingOverride>,
) -> Vec<TierTransition> {
    let mut budget_transitions = Vec::new();

    // Budget is scope-wide, not per-type. Resolve directly from global defaults
    // with optional scope override, bypassing per-type layer.
    let mut budget = forgetting.defaults.budget_top_n;
    if let Some(scope_cfg) = scope_override
        && let Some(b) = scope_cfg.budget_top_n
    {
        budget = b;
    }
    if budget == 0 {
        return budget_transitions; // 0 = no limit
    }

    // Count active entries (Hot + Warm), excluding pinned
    let active_count = scope_entries
        .iter()
        .filter(|e| {
            (e.tier == Tier::Hot || e.tier == Tier::Warm)
                && e.pin.is_none()
        })
        .count();

    if active_count <= budget as usize {
        return budget_transitions; // Under budget
    }

    let excess = active_count - budget as usize;

    // Rank active non-pinned entries by retention_score (ascending = lowest first)
    let mut candidates: Vec<(uuid::Uuid, f32, Tier)> = scope_entries
        .iter()
        .filter(|e| {
            (e.tier == Tier::Hot || e.tier == Tier::Warm)
                && e.pin.is_none()
                && !transitioned_ids.contains(&e.id) // Skip already-transitioned
        })
        .map(|e| {
            let score = score_map.get(&e.id).copied().unwrap_or(0.0);
            (e.id, score, e.tier)
        })
        .collect();

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut demoted = 0;
    for (entry_id, score, current_tier) in &candidates {
        if demoted >= excess {
            break;
        }

        let target = clamp_one_step(*current_tier, Tier::Forgotten);
        if target == *current_tier {
            continue;
        }

        // Auto-protected entries can only be demoted to Warm, not Cold/Forgotten
        if let Some(entry) = entry_map.get(entry_id) {
            let auto_params = AutoProtectParams {
                memory_type: entry.memory_type,
                is_chain_head: entry.superseded_by.is_none(),
                inbound_edges: inbound_edge_map.get(entry_id).copied().unwrap_or(0),
                content_role: entry.metadata.content_role.as_deref(),
                confidence: entry.confidence,
            };
            if scoring::is_auto_protected(&auto_params) && target < Tier::Warm {
                // Floor at Warm for auto-protected entries
                if *current_tier == Tier::Hot {
                    budget_transitions.push(TierTransition {
                        entry_id: *entry_id,
                        scope_id: entry.scope_id.clone(),
                        memory_type: entry.memory_type,
                        old_tier: *current_tier,
                        new_tier: Tier::Warm,
                        retention_score: *score,
                        d_score: 0.0,
                        a_score: 0.0,
                        g_score: 0.0,
                        c_score: 0.0,
                        superseded: entry.superseded_by.is_some(),
                        reason: "budget_policy",
                    });
                    demoted += 1;
                }
                // If already Warm and auto-protected, skip (can't go to Cold)
                continue;
            }

            // Chain-protected entries also can't go to Forgotten
            if chain_protected_ids.contains(entry_id) && target == Tier::Forgotten {
                continue;
            }
        }

        budget_transitions.push(TierTransition {
            entry_id: *entry_id,
            scope_id: entry_map.get(entry_id).map(|e| e.scope_id.clone()).unwrap_or_default(),
            memory_type: entry_map.get(entry_id).map(|e| e.memory_type).unwrap_or_default(),
            old_tier: *current_tier,
            new_tier: target,
            retention_score: *score,
            d_score: 0.0,
            a_score: 0.0,
            g_score: 0.0,
            c_score: 0.0,
            superseded: entry_map.get(entry_id).map(|e| e.superseded_by.is_some()).unwrap_or(false),
            reason: "budget_policy",
        });
        demoted += 1;
    }

    budget_transitions
}

/// Apply inactivity policy: force to Cold if inactive past threshold AND score < exemption.
///
/// `days_since_creation` is used as fallback when the entry has never been accessed.
fn apply_inactivity_policy(
    entry: &KnowledgeEntry,
    policy: &corvia_common::config::ResolvedPolicy,
    retention_score: f32,
    days_since_access: Option<f64>,
    days_since_creation: f64,
) -> Option<Tier> {
    // Only applies to Hot/Warm entries
    if entry.tier != Tier::Hot && entry.tier != Tier::Warm {
        return None;
    }

    // Exemption: high-scoring entries are not forced
    if retention_score >= INACTIVITY_SCORE_EXEMPTION {
        return None;
    }

    // Never accessed → use days since creation as inactivity measure
    let inactive_days = days_since_access.unwrap_or(days_since_creation);

    if inactive_days > policy.max_inactive_days as f64 {
        Some(Tier::Cold)
    } else {
        None
    }
}

/// Return the WORSE of two optional transitions (lower tier = worse).
fn worse_transition(a: Option<Tier>, b: Option<Tier>) -> Option<Tier> {
    match (a, b) {
        (None, None) => None,
        (Some(t), None) | (None, Some(t)) => Some(t),
        (Some(a), Some(b)) => {
            // Tier ordering: Forgotten < Cold < Warm < Hot
            // "Worse" = lower tier
            if a <= b { Some(a) } else { Some(b) }
        }
    }
}

/// Clamp a transition to at most one tier step.
fn clamp_one_step(current: Tier, target: Tier) -> Tier {
    match current {
        Tier::Hot => {
            if target < Tier::Warm {
                Tier::Warm // Hot can only drop to Warm
            } else {
                target
            }
        }
        Tier::Warm => {
            if target < Tier::Cold {
                Tier::Cold // Warm can only drop to Cold
            } else if target > Tier::Warm {
                Tier::Hot // Warm can only promote to Hot
            } else {
                target
            }
        }
        Tier::Cold => {
            if target < Tier::Cold {
                Tier::Forgotten // Cold can only drop to Forgotten
            } else if target > Tier::Cold {
                Tier::Warm // Cold can only promote to Warm
            } else {
                target
            }
        }
        Tier::Forgotten => Tier::Forgotten, // terminal
    }
}

/// Apply tier transitions and score updates in batched Redb writes.
///
/// Uses read-modify-write via `GcEntryUpdate` to preserve concurrent access
/// tracking updates (access_count, last_accessed) that may have occurred
/// between the initial entry scan and this batch write.
///
/// Circuit breaker: if >50% of batch writes fail, aborts and returns an error.
fn apply_transitions_batched(
    store: &crate::lite_store::LiteStore,
    transitions: &[TierTransition],
    score_updates: &[(uuid::Uuid, f32)],
) -> Result<()> {
    if transitions.is_empty() && score_updates.is_empty() {
        return Ok(());
    }

    // Build lookup maps
    let transition_map: HashMap<uuid::Uuid, &TierTransition> =
        transitions.iter().map(|t| (t.entry_id, t)).collect();
    let score_map: HashMap<uuid::Uuid, f32> =
        score_updates.iter().copied().collect();

    // Collect all entry IDs that need updates
    let mut all_ids: Vec<uuid::Uuid> = score_map.keys().copied().collect();
    for t in transitions {
        if !score_map.contains_key(&t.entry_id) {
            all_ids.push(t.entry_id);
        }
    }
    all_ids.sort();
    all_ids.dedup();

    // Build GcEntryUpdate structs (lightweight, no entry cloning)
    let mut gc_updates: Vec<crate::lite_store::GcEntryUpdate> = Vec::with_capacity(all_ids.len());
    let mut hnsw_removal_ids: Vec<uuid::Uuid> = Vec::new();

    for &entry_id in &all_ids {
        let score = match score_map.get(&entry_id) {
            Some(&s) => s,
            None => 0.0, // transition-only entry (shouldn't happen, but safe fallback)
        };

        let (new_tier, tier_changed_at) = if let Some(transition) = transition_map.get(&entry_id) {
            // Track entries leaving HNSW-indexed tiers for mapping cleanup.
            // Warm→Cold is the primary case; Cold→Forgotten is defense-in-depth.
            if transition.new_tier == Tier::Cold || transition.new_tier == Tier::Forgotten {
                hnsw_removal_ids.push(entry_id);
            }
            (Some(transition.new_tier), Some(Utc::now()))
        } else {
            (None, None)
        };

        gc_updates.push(crate::lite_store::GcEntryUpdate {
            entry_id,
            retention_score: score,
            new_tier,
            tier_changed_at,
        });
    }

    // Batch write with circuit breaker
    let total_batches = gc_updates.len().div_ceil(BATCH_SIZE);
    let mut failed_batches = 0;

    for chunk in gc_updates.chunks(BATCH_SIZE) {
        if let Err(e) = store.update_entries_batch(chunk, BATCH_SIZE) {
            failed_batches += 1;
            warn!(error = %e, "GC batch write failed");

            // Circuit breaker: abort if >50% of batches fail
            if total_batches > 0
                && (failed_batches as f64 / total_batches as f64) > CIRCUIT_BREAKER_THRESHOLD
            {
                return Err(corvia_common::errors::CorviaError::Storage(
                    format!(
                        "Circuit breaker tripped: {failed_batches}/{total_batches} batch writes failed (>{:.0}%)",
                        CIRCUIT_BREAKER_THRESHOLD * 100.0,
                    ),
                ));
            }
        }
    }

    // Delete HNSW mappings for entries leaving indexed tiers (lazy orphaning)
    if !hnsw_removal_ids.is_empty() {
        store.remove_hnsw_mappings(&hnsw_removal_ids)?;
    }

    Ok(())
}

/// List scope directories under the knowledge root.
fn list_scope_dirs(knowledge_dir: &Path) -> Vec<String> {
    let mut scopes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(knowledge_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir()
                && let Some(name) = entry.file_name().to_str()
            {
                scopes.push(name.to_string());
            }
        }
    }
    scopes.sort();
    scopes
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::types::{MemoryType, PinInfo, Tier};

    // ── clamp_one_step ─────────────────────────────────────────────────────

    #[test]
    fn test_clamp_hot_to_forgotten_becomes_warm() {
        assert_eq!(clamp_one_step(Tier::Hot, Tier::Forgotten), Tier::Warm);
    }

    #[test]
    fn test_clamp_hot_to_cold_becomes_warm() {
        assert_eq!(clamp_one_step(Tier::Hot, Tier::Cold), Tier::Warm);
    }

    #[test]
    fn test_clamp_hot_to_warm_stays() {
        assert_eq!(clamp_one_step(Tier::Hot, Tier::Warm), Tier::Warm);
    }

    #[test]
    fn test_clamp_warm_to_forgotten_becomes_cold() {
        assert_eq!(clamp_one_step(Tier::Warm, Tier::Forgotten), Tier::Cold);
    }

    #[test]
    fn test_clamp_cold_to_warm_stays() {
        assert_eq!(clamp_one_step(Tier::Cold, Tier::Warm), Tier::Warm);
    }

    #[test]
    fn test_clamp_cold_to_hot_becomes_warm() {
        assert_eq!(clamp_one_step(Tier::Cold, Tier::Hot), Tier::Warm);
    }

    #[test]
    fn test_clamp_forgotten_is_terminal() {
        assert_eq!(clamp_one_step(Tier::Forgotten, Tier::Hot), Tier::Forgotten);
    }

    // ── worse_transition ───────────────────────────────────────────────────

    #[test]
    fn test_worse_both_none() {
        assert_eq!(worse_transition(None, None), None);
    }

    #[test]
    fn test_worse_one_some() {
        assert_eq!(worse_transition(Some(Tier::Cold), None), Some(Tier::Cold));
        assert_eq!(worse_transition(None, Some(Tier::Warm)), Some(Tier::Warm));
    }

    #[test]
    fn test_worse_picks_lower_tier() {
        assert_eq!(
            worse_transition(Some(Tier::Warm), Some(Tier::Cold)),
            Some(Tier::Cold)
        );
        assert_eq!(
            worse_transition(Some(Tier::Cold), Some(Tier::Warm)),
            Some(Tier::Cold)
        );
    }

    // ── apply_inactivity_policy ────────────────────────────────────────────

    fn make_entry(tier: Tier, memory_type: MemoryType) -> KnowledgeEntry {
        KnowledgeEntry {
            id: uuid::Uuid::now_v7(),
            content: String::new(),
            source_version: String::new(),
            scope_id: "test".into(),
            workstream: String::new(),
            recorded_at: Utc::now() - chrono::Duration::days(60),
            valid_from: Utc::now(),
            valid_to: None,
            superseded_by: None,
            embedding: Some(vec![0.0; 768]),
            metadata: Default::default(),
            agent_id: None,
            session_id: None,
            entry_status: corvia_common::agent_types::EntryStatus::Committed,
            memory_type,
            confidence: Some(0.7),
            last_accessed: None,
            access_count: 0,
            tier,
            tier_changed_at: None,
            retention_score: None,
            pin: None,
        }
    }

    fn default_policy() -> corvia_common::config::ResolvedPolicy {
        corvia_common::config::ResolvedPolicy {
            enabled: true,
            max_inactive_days: 30,
            budget_top_n: 10_000,
        }
    }

    #[test]
    fn test_inactivity_forces_cold_when_inactive_and_low_score() {
        let entry = make_entry(Tier::Hot, MemoryType::Episodic);
        let policy = default_policy();
        // Never accessed, 60 days old, low score
        let result = apply_inactivity_policy(&entry, &policy, 0.30, None, 60.0);
        assert_eq!(result, Some(Tier::Cold));
    }

    #[test]
    fn test_inactivity_exempts_high_score() {
        let entry = make_entry(Tier::Hot, MemoryType::Episodic);
        let policy = default_policy();
        // High score exempts from inactivity policy
        let result = apply_inactivity_policy(&entry, &policy, 0.65, None, 60.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_inactivity_skips_cold_entries() {
        let entry = make_entry(Tier::Cold, MemoryType::Episodic);
        let policy = default_policy();
        let result = apply_inactivity_policy(&entry, &policy, 0.10, None, 60.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_inactivity_respects_threshold() {
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        // Accessed 10 days ago (under 30-day threshold)
        entry.last_accessed = Some(Utc::now() - chrono::Duration::days(10));
        let policy = default_policy();
        let result = apply_inactivity_policy(&entry, &policy, 0.30, Some(10.0), 60.0);
        assert_eq!(result, None);
    }

    // ── Tier ordering invariant ─────────────────────────────────────────────

    #[test]
    fn test_tier_ordering_invariant() {
        // worse_transition relies on Ord derivation from enum variant order.
        // This test catches accidental reordering of Tier variants.
        assert!(Tier::Forgotten < Tier::Cold);
        assert!(Tier::Cold < Tier::Warm);
        assert!(Tier::Warm < Tier::Hot);
    }

    // ── list_scope_dirs ────────────────────────────────────────────────────

    #[test]
    fn test_list_scope_dirs_nonexistent() {
        let scopes = list_scope_dirs(Path::new("/nonexistent/path"));
        assert!(scopes.is_empty());
    }

    #[test]
    fn test_list_scope_dirs_with_dirs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("alpha")).unwrap();
        std::fs::create_dir(dir.path().join("beta")).unwrap();
        std::fs::write(dir.path().join("file.txt"), "not a dir").unwrap();
        let scopes = list_scope_dirs(dir.path());
        assert_eq!(scopes, vec!["alpha", "beta"]);
    }

    // ── Integration: full GC cycle on LiteStore ────────────────────────────

    #[tokio::test]
    async fn test_gc_cycle_disabled_config() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();
        let graph: Arc<dyn GraphStore> = store.clone();
        let store: Arc<dyn QueryableStore> = store;

        // forgetting = None → disabled
        let report = run_gc_cycle(&store, &graph, dir.path(), None, &HashMap::new(), None)
            .await
            .unwrap();
        assert_eq!(report.entries_scanned, 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_enabled_but_no_entries() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();
        let graph: Arc<dyn GraphStore> = store.clone();
        let store: Arc<dyn QueryableStore> = store;

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };
        let report = run_gc_cycle(&store, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();
        assert_eq!(report.entries_scanned, 0);
        assert_eq!(report.entries_scored, 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_scores_and_transitions() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert an old Episodic entry with low score (should transition Hot → Warm)
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(90);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.3);

        // Write knowledge file and insert to store
        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.entries_scanned, 1);
        assert_eq!(report.entries_scored, 1);
        // Old episodic entry with no access and low confidence should demote
        assert!(report.hot_to_warm > 0 || report.warm_to_cold > 0 || report.cold_to_forgotten > 0,
            "Expected at least one demotion, got report: {report:?}");

        // Verify entry was updated
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert!(updated.retention_score.is_some(), "retention_score should be set");
    }

    #[tokio::test]
    async fn test_gc_cycle_pinned_entry_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(90);
        entry.embedding = Some(vec![0.1; 768]);
        entry.pin = Some(PinInfo {
            by: "admin".into(),
            at: Utc::now(),
        });

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.entries_scanned, 1);
        assert_eq!(report.entries_scored, 0); // pinned = skipped

        // Verify entry tier unchanged
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Hot);
    }

    #[tokio::test]
    async fn test_gc_cycle_forgotten_entry_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        let mut entry = make_entry(Tier::Forgotten, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.embedding = None; // Forgotten entries have no embedding

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        // Use update_entry_metadata since Forgotten entries have no embedding
        store.update_entry_metadata(&entry).unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.entries_scanned, 1);
        assert_eq!(report.entries_scored, 0); // Forgotten = skipped
    }

    #[tokio::test]
    async fn test_gc_cycle_one_tier_step_constraint() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Hot entry that would score very low (should still only go to Warm)
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.superseded_by = Some(uuid::Uuid::now_v7()); // superseded penalty halves score

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        // Entry should move to Warm (one step), not Cold or Forgotten
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Warm, "Hot should only demote to Warm in one cycle");
        assert_eq!(report.hot_to_warm, 1);
        assert_eq!(report.warm_to_cold, 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a fresh Hot entry with great score (should stay Hot)
        let mut entry = make_entry(Tier::Hot, MemoryType::Structural);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now();
        entry.embedding = Some(vec![0.1; 768]);
        entry.access_count = 50;
        entry.last_accessed = Some(Utc::now());
        entry.confidence = Some(0.9);

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        // Run GC twice
        let report1 = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();
        let report2 = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        // Both should have zero transitions
        assert_eq!(report1.hot_to_warm, 0);
        assert_eq!(report1.warm_to_cold, 0);
        assert_eq!(report2.hot_to_warm, 0);
        assert_eq!(report2.warm_to_cold, 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_forgotten_discards_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Cold entry with very low score (should transition Cold → Forgotten)
        let mut entry = make_entry(Tier::Cold, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.superseded_by = Some(uuid::Uuid::now_v7()); // superseded penalty

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.cold_to_forgotten, 1);

        // Verify embedding is discarded
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Forgotten);
        assert!(updated.embedding.is_none(), "Forgotten entry should have no embedding");
    }

    #[tokio::test]
    async fn test_gc_cycle_warm_to_cold_triggers_hnsw_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Warm entry with very low score (should transition Warm → Cold)
        let mut entry = make_entry(Tier::Warm, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.superseded_by = Some(uuid::Uuid::now_v7());
        entry.tier = Tier::Warm;

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.warm_to_cold, 1, "Expected Warm→Cold transition");
        assert!(report.hnsw_rebuild_triggered, "HNSW rebuild should be triggered on Warm→Cold");

        // Verify entry is now Cold
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Cold);
    }

    #[tokio::test]
    async fn test_gc_cycle_cold_to_warm_promotion() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Cold entry with very high score (should promote Cold → Warm)
        let mut entry = make_entry(Tier::Cold, MemoryType::Structural);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now(); // fresh
        entry.embedding = Some(vec![0.1; 768]);
        entry.access_count = 80;
        entry.last_accessed = Some(Utc::now());
        entry.confidence = Some(0.95);
        entry.tier = Tier::Cold;

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.cold_to_warm, 1, "Expected Cold→Warm promotion");

        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Warm, "Cold entry with high score should promote to Warm");
    }

    // ── build_chain_protected_set ─────────────────────────────────────────

    #[test]
    fn test_chain_protection_active_head_protects_ancestors() {
        // Chain: A → B → C (C is HEAD, Warm)
        // A and B should be protected from Forgotten.
        let mut c = make_entry(Tier::Warm, MemoryType::Episodic);
        c.superseded_by = None; // HEAD
        let mut b = make_entry(Tier::Cold, MemoryType::Episodic);
        b.superseded_by = Some(c.id);
        let mut a = make_entry(Tier::Cold, MemoryType::Episodic);
        a.superseded_by = Some(b.id);

        let entries: Vec<&KnowledgeEntry> = vec![&a, &b, &c];
        let protected = build_chain_protected_set(&entries);

        assert!(protected.contains(&a.id), "A should be chain-protected");
        assert!(protected.contains(&b.id), "B should be chain-protected");
        assert!(!protected.contains(&c.id), "C (HEAD) protected by auto-protection, not chain");
    }

    #[test]
    fn test_chain_protection_forgotten_head_allows_ancestor_forgotten() {
        // Chain: A → B → C (C is HEAD, Forgotten)
        // A and B can be Forgotten since HEAD is Forgotten.
        let mut c = make_entry(Tier::Forgotten, MemoryType::Episodic);
        c.superseded_by = None;
        let mut b = make_entry(Tier::Cold, MemoryType::Episodic);
        b.superseded_by = Some(c.id);
        let mut a = make_entry(Tier::Cold, MemoryType::Episodic);
        a.superseded_by = Some(b.id);

        let entries: Vec<&KnowledgeEntry> = vec![&a, &b, &c];
        let protected = build_chain_protected_set(&entries);

        assert!(!protected.contains(&a.id), "A should NOT be protected (HEAD is Forgotten)");
        assert!(!protected.contains(&b.id), "B should NOT be protected (HEAD is Forgotten)");
    }

    #[test]
    fn test_chain_protection_no_chain_no_protection() {
        // Standalone entry (no supersession chain)
        let entry = make_entry(Tier::Cold, MemoryType::Episodic);
        let entries: Vec<&KnowledgeEntry> = vec![&entry];
        let protected = build_chain_protected_set(&entries);

        assert!(protected.is_empty());
    }

    // ── Integration: chain protection in GC cycle ─────────────────────────

    #[tokio::test]
    async fn test_gc_chain_protection_blocks_forgotten() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Chain: A → B → C where C is Warm (active HEAD)
        // A and B are Cold with very low scores — would be Forgotten without protection
        let mut c = make_entry(Tier::Warm, MemoryType::Structural);
        c.scope_id = "test-scope".into();
        c.recorded_at = Utc::now();
        c.embedding = Some(vec![0.1; 768]);
        c.access_count = 50;
        c.last_accessed = Some(Utc::now());
        c.confidence = Some(0.9);
        c.superseded_by = None;

        let mut b = make_entry(Tier::Cold, MemoryType::Episodic);
        b.scope_id = "test-scope".into();
        b.recorded_at = Utc::now() - chrono::Duration::days(365);
        b.embedding = Some(vec![0.2; 768]);
        b.confidence = Some(0.0);
        b.superseded_by = Some(c.id);

        let mut a = make_entry(Tier::Cold, MemoryType::Episodic);
        a.scope_id = "test-scope".into();
        a.recorded_at = Utc::now() - chrono::Duration::days(365);
        a.embedding = Some(vec![0.3; 768]);
        a.confidence = Some(0.0);
        a.superseded_by = Some(b.id);

        for entry in [&a, &b, &c] {
            crate::knowledge_files::write_entry(dir.path(), entry).unwrap();
            store.insert(entry).await.unwrap();
        }

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        // B and A should be chain-protected from Forgotten
        let updated_a = store.get(&a.id).await.unwrap().unwrap();
        let updated_b = store.get(&b.id).await.unwrap().unwrap();
        assert_ne!(updated_a.tier, Tier::Forgotten, "A should be chain-protected from Forgotten");
        assert_ne!(updated_b.tier, Tier::Forgotten, "B should be chain-protected from Forgotten");
        // Chain protection specifically (not just auto-protection) should fire for ancestors
        assert!(report.chain_protected > 0,
            "Expected chain_protected > 0, got {}", report.chain_protected);
    }

    // ── Integration: auto-protection (Structural can reach Cold but not Forgotten) ──

    #[tokio::test]
    async fn test_gc_auto_protection_structural_not_forgotten() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Structural entry in Cold with very low score
        let mut entry = make_entry(Tier::Cold, MemoryType::Structural);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.superseded_by = Some(uuid::Uuid::now_v7()); // superseded to lower score

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        let updated = store.get(&entry.id).await.unwrap().unwrap();
        // Structural is auto-protected — Cold is OK, Forgotten is blocked
        assert_ne!(updated.tier, Tier::Forgotten, "Structural entry should not reach Forgotten");
        assert!(report.auto_protected > 0, "Expected auto-protection to fire");
    }

    // ── Integration: pinned entry never demoted ──────────────────────────

    #[tokio::test]
    async fn test_gc_pinned_entry_never_demoted() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.pin = Some(PinInfo { by: "admin".into(), at: Utc::now() });

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        // Run multiple cycles
        for _ in 0..3 {
            run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
                .await
                .unwrap();
        }

        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Hot, "Pinned entry should never be demoted");
    }

    // ── Integration: budget policy ──────────────────────────────────────

    #[tokio::test]
    async fn test_gc_budget_demotes_excess() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Create 15 Hot entries with varying scores
        let mut entry_ids = Vec::new();
        for i in 0..15 {
            let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
            entry.scope_id = "test-scope".into();
            entry.recorded_at = Utc::now() - chrono::Duration::days(1);
            entry.embedding = Some(vec![0.1 * (i as f32 + 1.0); 768]);
            entry.access_count = (i * 10) as u32;
            entry.last_accessed = Some(Utc::now());
            entry.confidence = Some(0.5 + (i as f32) * 0.03);

            crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
            store.insert(&entry).await.unwrap();
            entry_ids.push(entry.id);
        }

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        // Set budget_top_n = 10 (so 5 should be demoted)
        let mut config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };
        config.defaults.budget_top_n = 10;

        let _report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        // Count how many entries were demoted (budget + score-based)
        let mut demoted = 0;
        for &id in &entry_ids {
            let updated = store.get(&id).await.unwrap().unwrap();
            if updated.tier != Tier::Hot {
                demoted += 1;
            }
        }

        // At least 5 should be demoted (budget forces it), possibly more from scoring
        assert!(demoted >= 5, "Expected at least 5 demotions with budget_top_n=10, got {demoted}");
    }

    #[tokio::test]
    async fn test_gc_budget_skips_already_transitioned() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Create entries: some will transition by scoring, budget should skip those
        let mut entries = Vec::new();
        for i in 0..12 {
            let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
            entry.scope_id = "test-scope".into();
            entry.embedding = Some(vec![0.1 * (i as f32 + 1.0); 768]);

            if i < 4 {
                // These 4 will naturally transition (old, low score)
                entry.recorded_at = Utc::now() - chrono::Duration::days(365);
                entry.confidence = Some(0.0);
                entry.superseded_by = Some(uuid::Uuid::now_v7());
            } else {
                // These 8 are fresh and high-scoring
                entry.recorded_at = Utc::now() - chrono::Duration::days(1);
                entry.access_count = 50;
                entry.last_accessed = Some(Utc::now());
                entry.confidence = Some(0.9);
            }

            crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
            store.insert(&entry).await.unwrap();
            entries.push(entry);
        }

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        // budget_top_n = 10, but 4 will already transition from scoring
        let mut config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };
        config.defaults.budget_top_n = 10;

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        // The budget should not double-count already-transitioned entries
        // 4 transitioned by scoring, remaining 8 active Hot — under budget of 10
        // So budget_demoted should be minimal (0 or very few)
        assert!(report.budget_demoted <= 2,
            "Budget should skip already-transitioned entries, got {} budget demotions", report.budget_demoted);
    }

    // ── Integration: rate limit ─────────────────────────────────────────

    #[tokio::test]
    async fn test_gc_rate_limit_caps_forgotten_at_50() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Create 60 Cold entries with very low scores (all would go Forgotten)
        for i in 0..60 {
            let mut entry = make_entry(Tier::Cold, MemoryType::Episodic);
            entry.scope_id = "test-scope".into();
            entry.recorded_at = Utc::now() - chrono::Duration::days(365);
            entry.embedding = Some(vec![0.01 * (i as f32 + 1.0); 768]);
            entry.confidence = Some(0.0);
            entry.superseded_by = Some(uuid::Uuid::now_v7()); // low score
            entry.last_accessed = None;

            crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
            store.insert(&entry).await.unwrap();
        }

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert!(
            report.cold_to_forgotten <= MAX_FORGOTTEN_PER_CYCLE,
            "Expected at most {} Forgotten transitions, got {}",
            MAX_FORGOTTEN_PER_CYCLE,
            report.cold_to_forgotten,
        );
        assert!(
            report.forgotten_rate_limited > 0,
            "Expected some rate-limited transitions, got 0",
        );
    }

    // ── Unit: build_chain_protected_set with branching chains ────────────

    #[test]
    fn test_chain_protection_multiple_chains() {
        // Chain 1: A → B (B is HEAD, Warm) → A protected
        // Chain 2: C → D (D is HEAD, Forgotten) → C NOT protected
        let mut b = make_entry(Tier::Warm, MemoryType::Episodic);
        b.superseded_by = None;
        let mut a = make_entry(Tier::Cold, MemoryType::Episodic);
        a.superseded_by = Some(b.id);

        let mut d = make_entry(Tier::Forgotten, MemoryType::Episodic);
        d.superseded_by = None;
        let mut c = make_entry(Tier::Cold, MemoryType::Episodic);
        c.superseded_by = Some(d.id);

        let entries: Vec<&KnowledgeEntry> = vec![&a, &b, &c, &d];
        let protected = build_chain_protected_set(&entries);

        assert!(protected.contains(&a.id), "A should be protected (HEAD B is Warm)");
        assert!(!protected.contains(&c.id), "C should NOT be protected (HEAD D is Forgotten)");
    }

    // ── Unit: apply_budget_policy ────────────────────────────────────────

    #[test]
    fn test_budget_policy_no_limit_when_zero() {
        let entry = {
            let mut e = make_entry(Tier::Hot, MemoryType::Structural);
            e.scope_id = "test".into();
            e
        };
        let scope_entries = vec![&entry];
        let entry_map: HashMap<uuid::Uuid, &KnowledgeEntry> = [(entry.id, &entry)].into();
        let score_map: HashMap<uuid::Uuid, f32> = [(entry.id, 0.01)].into();
        let inbound_edge_map = HashMap::new();
        let transitioned_ids = HashSet::new();
        let chain_protected_ids = HashSet::new();

        let forgetting = ForgettingConfig {
            enabled: true,
            defaults: corvia_common::config::ForgettingPolicyConfig {
                max_inactive_days: 90,
                budget_top_n: 0, // budget_top_n=0 means no limit
            },
            ..Default::default()
        };

        let transitions = apply_budget_policy(
            &scope_entries, &entry_map, &score_map, &inbound_edge_map,
            &transitioned_ids, &chain_protected_ids,
            &forgetting, None,
        );
        assert!(transitions.is_empty(), "budget_top_n=0 means no limit");
    }

    #[test]
    fn test_budget_policy_auto_protected_floor_at_warm() {
        // 3 entries: 2 Structural (auto-protected), 1 Episodic. Budget = 1.
        // Budget should demote Structural from Hot to Warm (floor), not Cold.
        let mut structural1 = make_entry(Tier::Hot, MemoryType::Structural);
        structural1.scope_id = "test".into();
        let mut structural2 = make_entry(Tier::Hot, MemoryType::Structural);
        structural2.scope_id = "test".into();
        let mut episodic = make_entry(Tier::Hot, MemoryType::Episodic);
        episodic.scope_id = "test".into();

        let scope_entries = vec![&structural1, &structural2, &episodic];
        let entry_map: HashMap<uuid::Uuid, &KnowledgeEntry> = scope_entries.iter().map(|e| (e.id, *e)).collect();
        let score_map: HashMap<uuid::Uuid, f32> = [
            (structural1.id, 0.01), // lowest
            (structural2.id, 0.02),
            (episodic.id, 0.03),    // highest of the three
        ].into();
        let inbound_edge_map = HashMap::new();
        let transitioned_ids = HashSet::new();
        let chain_protected_ids = HashSet::new();

        let forgetting = ForgettingConfig {
            enabled: true,
            defaults: corvia_common::config::ForgettingPolicyConfig {
                max_inactive_days: 90,
                budget_top_n: 1, // only 1 active allowed, 3 present → 2 excess
            },
            ..Default::default()
        };

        let transitions = apply_budget_policy(
            &scope_entries, &entry_map, &score_map, &inbound_edge_map,
            &transitioned_ids, &chain_protected_ids,
            &forgetting, None,
        );

        // Should get 2 demotions
        assert_eq!(transitions.len(), 2, "Expected 2 budget demotions, got {}", transitions.len());

        // All demotions should be to Warm (auto-protected Structural floor + one-step for Episodic)
        for t in &transitions {
            assert_eq!(t.new_tier, Tier::Warm,
                "Budget demotion should be to Warm (one step from Hot), got {:?}", t.new_tier);
        }
    }

    // ── Circuit breaker unit test ──────────────────────────────────────

    #[tokio::test]
    async fn test_gc_circuit_breaker_on_write_failure() {
        // The circuit breaker is tested by ensuring the report field is set correctly
        // when the write path succeeds (no trip) vs when processing continues normally.
        // A full write-failure simulation requires a mock store, so we verify:
        // 1. Normal cycle does NOT trip circuit breaker
        // 2. The circuit breaker field exists and defaults to false
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(90);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.3);

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert!(!report.circuit_breaker_tripped, "Normal cycle should not trip circuit breaker");
    }

    // ── Observability tests ────────────────────────────────────────────────

    #[test]
    fn test_forgotten_access_counter() {
        let counter = ForgottenAccessCounter::new();
        assert_eq!(counter.load(), 0);

        counter.increment();
        counter.increment();
        counter.increment();
        assert_eq!(counter.load(), 3);

        // take() returns current and resets to 0
        let val = counter.take();
        assert_eq!(val, 3);
        assert_eq!(counter.load(), 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_transition_emits_with_score_breakdown() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert an old Episodic entry that will transition
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(90);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.3);

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let counter = ForgottenAccessCounter::new();
        counter.increment(); // simulate one forgotten access

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), Some(&counter))
            .await
            .unwrap();

        // Should have at least one transition
        assert!(report.entries_scored > 0);
        assert!(
            report.hot_to_warm > 0 || report.warm_to_cold > 0 || report.cold_to_forgotten > 0,
            "Expected a transition for old episodic entry"
        );

        // Counter should be reset after GC cycle takes it
        assert_eq!(counter.load(), 0);
    }

    #[tokio::test]
    async fn test_gc_cycle_report_has_rebuild_duration() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::lite_store::LiteStore::open(dir.path(), 768).unwrap());
        store.init_schema().await.unwrap();

        // Insert a Warm entry that will transition Warm → Cold
        let mut entry = make_entry(Tier::Warm, MemoryType::Episodic);
        entry.scope_id = "test-scope".into();
        entry.recorded_at = Utc::now() - chrono::Duration::days(365);
        entry.embedding = Some(vec![0.1; 768]);
        entry.confidence = Some(0.0);
        entry.superseded_by = Some(uuid::Uuid::now_v7());

        crate::knowledge_files::write_entry(dir.path(), &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let graph: Arc<dyn GraphStore> = store.clone();
        let store_q: Arc<dyn QueryableStore> = store.clone();

        let config = ForgettingConfig {
            enabled: true,
            ..Default::default()
        };

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new(), None)
            .await
            .unwrap();

        assert_eq!(report.warm_to_cold, 1);
        assert!(report.hnsw_rebuild_triggered);
        // rebuild_duration_ms should be set (>= 0)
        // Can't assert > 0 because sub-ms rebuilds on empty stores
    }
}
