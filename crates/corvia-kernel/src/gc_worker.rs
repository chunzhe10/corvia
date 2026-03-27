//! Background GC worker for tiered knowledge lifecycle.
//!
//! Periodically evaluates all entries, computes retention scores, applies tier
//! transitions, and triggers HNSW rebuilds when entries move out of indexed tiers.
//!
//! This is the skeleton — safeguards (chain protection, rate limit, circuit breaker)
//! are in a follow-up issue.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use corvia_common::config::{CorviaConfig, ForgettingConfig, ScopeForgettingOverride};
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, KnowledgeEntry, Tier};
use tracing::{info, info_span, warn, Instrument};

use crate::scoring::{self, RetentionParams};
use crate::traits::{GraphStore, QueryableStore};

/// Batch size for Redb write transactions during tier transitions.
const BATCH_SIZE: usize = 100;

/// Score threshold above which inactivity policy does NOT force Cold.
const INACTIVITY_SCORE_EXEMPTION: f32 = 0.60;

// ── Report ──────────────────────────────────────────────────────────────────

/// Metrics from a single GC cycle.
#[derive(Debug, Clone, Default)]
pub struct GcCycleReport {
    pub entries_scanned: usize,
    pub entries_scored: usize,
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub cold_to_forgotten: usize,
    pub warm_to_hot: usize,
    pub cold_to_warm: usize,
    pub hnsw_rebuild_triggered: bool,
    pub cycle_duration_ms: u64,
    pub scopes_processed: usize,
}

// ── Tier transition record ──────────────────────────────────────────────────

/// A pending tier transition for a single entry.
#[derive(Debug, Clone)]
struct TierTransition {
    entry_id: uuid::Uuid,
    old_tier: Tier,
    new_tier: Tier,
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
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let interval_minutes = {
                let cfg = config.read().unwrap();
                cfg.forgetting
                    .as_ref()
                    .map(|f| f.interval_minutes)
                    .unwrap_or(60)
            };
            tokio::time::sleep(std::time::Duration::from_secs(
                interval_minutes as u64 * 60,
            ))
            .await;

            let forgetting_config = {
                let cfg = config.read().unwrap();
                cfg.forgetting.clone()
            };
            let scope_configs: HashMap<String, Option<ScopeForgettingOverride>> = {
                let cfg = config.read().unwrap();
                cfg.scope
                    .as_ref()
                    .map(|scopes| {
                        scopes
                            .iter()
                            .map(|s| (s.id.clone(), s.forgetting.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            };

            match run_gc_cycle(&store, &graph, &data_dir, forgetting_config.as_ref(), &scope_configs).await {
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
pub async fn run_gc_cycle(
    store: &Arc<dyn QueryableStore>,
    graph: &Arc<dyn GraphStore>,
    data_dir: &Path,
    forgetting_config: Option<&ForgettingConfig>,
    scope_configs: &HashMap<String, Option<ScopeForgettingOverride>>,
) -> Result<GcCycleReport> {
    let start = std::time::Instant::now();

    let forgetting = match forgetting_config {
        Some(f) if f.enabled => f,
        _ => {
            return Ok(GcCycleReport::default());
        }
    };

    let span = info_span!("corvia.gc.cycle");
    async move {
        let mut report = GcCycleReport::default();

        // Discover scopes from knowledge directory
        let knowledge_dir = data_dir.join("knowledge");
        let scopes = list_scope_dirs(&knowledge_dir);

        for scope_id in &scopes {
            let scope_override = scope_configs
                .get(scope_id.as_str())
                .and_then(|o| o.as_ref());

            process_scope(
                store,
                graph,
                scope_id,
                forgetting,
                scope_override,
                &mut report,
            )
            .await?;
            report.scopes_processed += 1;
        }

        report.cycle_duration_ms = start.elapsed().as_millis() as u64;
        Ok(report)
    }
    .instrument(span)
    .await
}

/// Process a single scope: score all entries, determine transitions, apply batched writes.
async fn process_scope(
    store: &Arc<dyn QueryableStore>,
    graph: &Arc<dyn GraphStore>,
    scope_id: &str,
    forgetting: &ForgettingConfig,
    scope_override: Option<&ScopeForgettingOverride>,
    report: &mut GcCycleReport,
) -> Result<()> {
    // Load all entries from Redb via the store (downcast to LiteStore).
    let lite_store = store
        .as_any()
        .downcast_ref::<crate::lite_store::LiteStore>()
        .ok_or_else(|| {
            corvia_common::errors::CorviaError::Storage(
                "GC worker requires LiteStore backend".into(),
            )
        })?;

    let all_entries = lite_store.fetch_all_entries()?;
    let scope_entries: Vec<&KnowledgeEntry> = all_entries
        .iter()
        .filter(|e| e.scope_id == scope_id)
        .collect();

    let now = Utc::now();
    let mut transitions: Vec<TierTransition> = Vec::new();
    let mut score_updates: Vec<(uuid::Uuid, f32)> = Vec::new();

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

        // Compute retention params
        let days_since_creation = (now - entry.recorded_at).num_seconds() as f64 / 86400.0;
        let days_since_access = entry
            .last_accessed
            .map(|la| (now - la).num_seconds() as f64 / 86400.0);
        let inbound_edges = graph
            .edges(&entry.id, EdgeDirection::Incoming)
            .await
            .map(|e| e.len() as u32)
            .unwrap_or(0);

        let params = RetentionParams {
            memory_type: entry.memory_type,
            days_since_creation,
            access_count: entry.access_count,
            days_since_access,
            inbound_edges,
            confidence: entry.confidence,
            is_superseded: entry.superseded_by.is_some(),
        };

        let retention_score = scoring::compute_retention_score(&params);
        score_updates.push((entry.id, retention_score));

        // Determine score-based transition
        let score_transition = scoring::determine_tier_transition(entry.tier, retention_score);

        // Apply inactivity policy: force Cold if inactive AND score < exemption
        let inactivity_transition = apply_inactivity_policy(
            entry,
            &policy,
            retention_score,
            days_since_access,
        );

        // Take the WORSE transition (lower tier = worse)
        let effective_transition = worse_transition(score_transition, inactivity_transition);

        if let Some(target_tier) = effective_transition {
            // Enforce one-tier-step-per-cycle
            let clamped = clamp_one_step(entry.tier, target_tier);
            if clamped != entry.tier {
                transitions.push(TierTransition {
                    entry_id: entry.id,
                    old_tier: entry.tier,
                    new_tier: clamped,
                });
            }
        }
    }

    // Count transitions by type
    for t in &transitions {
        match (t.old_tier, t.new_tier) {
            (Tier::Hot, Tier::Warm) => report.hot_to_warm += 1,
            (Tier::Warm, Tier::Cold) => report.warm_to_cold += 1,
            (Tier::Cold, Tier::Forgotten) => report.cold_to_forgotten += 1,
            (Tier::Warm, Tier::Hot) => report.warm_to_hot += 1,
            (Tier::Cold, Tier::Warm) => report.cold_to_warm += 1,
            _ => {}
        }
    }

    let has_warm_to_cold = transitions
        .iter()
        .any(|t| t.old_tier == Tier::Warm && t.new_tier == Tier::Cold);

    // Apply transitions + score updates in batches
    apply_transitions_batched(lite_store, &transitions, &score_updates, &all_entries)?;

    // Trigger HNSW rebuild if any Warm→Cold transitions (entries removed from HNSW)
    if has_warm_to_cold {
        info!(scope = scope_id, "Triggering HNSW rebuild after Warm→Cold transitions");
        lite_store.rebuild_from_files()?;
        if let Err(e) = lite_store.flush_hnsw() {
            warn!(error = %e, "Failed to persist HNSW after GC rebuild");
        }
        report.hnsw_rebuild_triggered = true;
    }

    Ok(())
}

/// Apply inactivity policy: force to Cold if inactive past threshold AND score < exemption.
fn apply_inactivity_policy(
    entry: &KnowledgeEntry,
    policy: &corvia_common::config::ResolvedPolicy,
    retention_score: f32,
    days_since_access: Option<f64>,
) -> Option<Tier> {
    // Only applies to Hot/Warm entries
    if entry.tier != Tier::Hot && entry.tier != Tier::Warm {
        return None;
    }

    // Exemption: high-scoring entries are not forced
    if retention_score >= INACTIVITY_SCORE_EXEMPTION {
        return None;
    }

    let inactive_days = days_since_access.unwrap_or_else(|| {
        // Never accessed — use days since creation
        let now = Utc::now();
        (now - entry.recorded_at).num_seconds() as f64 / 86400.0
    });

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
fn apply_transitions_batched(
    store: &crate::lite_store::LiteStore,
    transitions: &[TierTransition],
    score_updates: &[(uuid::Uuid, f32)],
    all_entries: &[KnowledgeEntry],
) -> Result<()> {
    if transitions.is_empty() && score_updates.is_empty() {
        return Ok(());
    }

    // Build lookup maps
    let transition_map: HashMap<uuid::Uuid, &TierTransition> =
        transitions.iter().map(|t| (t.entry_id, t)).collect();
    let score_map: HashMap<uuid::Uuid, f32> =
        score_updates.iter().copied().collect();

    // Build entry lookup
    let entry_map: HashMap<uuid::Uuid, &KnowledgeEntry> =
        all_entries.iter().map(|e| (e.id, e)).collect();

    // Collect all entry IDs that need updates
    let mut all_ids: Vec<uuid::Uuid> = score_map.keys().copied().collect();
    for t in transitions {
        if !score_map.contains_key(&t.entry_id) {
            all_ids.push(t.entry_id);
        }
    }
    all_ids.sort();
    all_ids.dedup();

    // Build updated entries
    let mut updated_entries: Vec<KnowledgeEntry> = Vec::with_capacity(all_ids.len());
    let mut warm_to_cold_ids: Vec<uuid::Uuid> = Vec::new();

    for &entry_id in &all_ids {
        let base = match entry_map.get(&entry_id) {
            Some(e) => e,
            None => {
                warn!(entry_id = %entry_id, "Entry not found during GC batch update, skipping");
                continue;
            }
        };

        let mut entry = (*base).clone();
        let mut changed = false;

        // Apply score update
        if let Some(&score) = score_map.get(&entry_id) {
            entry.retention_score = Some(score);
            changed = true;
        }

        // Apply tier transition
        if let Some(transition) = transition_map.get(&entry_id) {
            entry.tier = transition.new_tier;
            entry.tier_changed_at = Some(Utc::now());

            // Track Warm→Cold for HNSW mapping cleanup
            if transition.old_tier == Tier::Warm && transition.new_tier == Tier::Cold {
                warm_to_cold_ids.push(entry_id);
            }

            // Forgotten = discard embedding
            if transition.new_tier == Tier::Forgotten {
                entry.embedding = None;
            }
            changed = true;
        }

        if changed {
            updated_entries.push(entry);
        }
    }

    // Batch write to Redb + JSON files
    let refs: Vec<&KnowledgeEntry> = updated_entries.iter().collect();
    store.update_entries_batch(&refs, BATCH_SIZE)?;

    // Delete HNSW mappings for Warm→Cold transitions (lazy orphaning)
    if !warm_to_cold_ids.is_empty() {
        store.remove_hnsw_mappings(&warm_to_cold_ids)?;
    }

    Ok(())
}

/// List scope directories under the knowledge root.
fn list_scope_dirs(knowledge_dir: &Path) -> Vec<String> {
    let mut scopes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(knowledge_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    scopes.push(name.to_string());
                }
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
        let result = apply_inactivity_policy(&entry, &policy, 0.30, None);
        assert_eq!(result, Some(Tier::Cold));
    }

    #[test]
    fn test_inactivity_exempts_high_score() {
        let entry = make_entry(Tier::Hot, MemoryType::Episodic);
        let policy = default_policy();
        // High score exempts from inactivity policy
        let result = apply_inactivity_policy(&entry, &policy, 0.65, None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_inactivity_skips_cold_entries() {
        let entry = make_entry(Tier::Cold, MemoryType::Episodic);
        let policy = default_policy();
        let result = apply_inactivity_policy(&entry, &policy, 0.10, None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_inactivity_respects_threshold() {
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        // Accessed 10 days ago (under 30-day threshold)
        entry.last_accessed = Some(Utc::now() - chrono::Duration::days(10));
        let policy = default_policy();
        let result = apply_inactivity_policy(&entry, &policy, 0.30, Some(10.0));
        assert_eq!(result, None);
    }

    // ── Pinned entries ─────────────────────────────────────────────────────

    #[test]
    fn test_pinned_entry_has_pin_info() {
        let mut entry = make_entry(Tier::Hot, MemoryType::Episodic);
        entry.pin = Some(PinInfo {
            by: "admin".into(),
            at: Utc::now(),
        });
        assert!(entry.pin.is_some());
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
        let report = run_gc_cycle(&store, &graph, dir.path(), None, &HashMap::new())
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
        let report = run_gc_cycle(&store, &graph, dir.path(), Some(&config), &HashMap::new())
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

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
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

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
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

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
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

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
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
        let report1 = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
            .await
            .unwrap();
        let report2 = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
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

        let report = run_gc_cycle(&store_q, &graph, dir.path(), Some(&config), &HashMap::new())
            .await
            .unwrap();

        assert_eq!(report.cold_to_forgotten, 1);

        // Verify embedding is discarded
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.tier, Tier::Forgotten);
        assert!(updated.embedding.is_none(), "Forgotten entry should have no embedding");
    }
}
