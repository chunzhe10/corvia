# GC Worker Skeleton — Design

**Issue:** chunzhe10/corvia#19
**Date:** 2026-03-27
**Status:** Approved (self-approved per dev-loop autonomous protocol)

## Problem

corvia has all the building blocks for tiered knowledge lifecycle — retention scoring
(#15), access tracking (#16), ArcSwap HNSW (#17), and forgetting policy config (#18) —
but no background worker to orchestrate them. Entries accumulate in Hot tier indefinitely.

## Design

### Architecture

A single `tokio::spawn` periodic task inside `corvia serve`, reading `ForgettingConfig`
from the hot-reloadable config. The worker is a pure orchestration layer that composes
existing modules:

```
gc_worker::run_cycle()
  ├── config.rs::resolve_policy()     — 3-level policy hierarchy
  ├── scoring.rs::compute_retention_score() — pure scoring
  ├── scoring.rs::determine_tier_transition() — hysteresis thresholds
  └── lite_store.rs::update_tiers_batch() — batched Redb writes
       └── rebuild_from_files()       — blue-green HNSW swap (if Warm→Cold)
```

### GC Cycle Algorithm

```
1. Read ForgettingConfig from hot-reloadable config
2. If forgetting.enabled == false → skip, sleep, repeat
3. For each scope directory in data_dir/knowledge/:
   a. Load all entries from Redb (fetch_all_entries filtered by scope)
   b. Resolve policy for this scope (global → per-type → per-scope)
   c. For each entry:
      - Skip if pinned (entry.pin.is_some())
      - Skip if Forgotten (terminal)
      - Compute RetentionParams from entry fields + graph inbound edge count
      - Compute retention_score via scoring::compute_retention_score()
      - Determine score-based transition via scoring::determine_tier_transition()
      - Apply inactivity policy: if days_since_access > max_inactive_days
        AND score < 0.60 → force Cold (if currently Hot/Warm)
      - Take WORSE of score-based and inactivity transitions
      - Enforce one-tier-step-per-cycle (Hot can only → Warm, not → Cold)
      - Record transition if any
   d. Apply transitions in batched Redb writes (100 entries/txn)
   e. Update retention_score on all evaluated entries
4. If any Warm→Cold transitions occurred: trigger HNSW rebuild (blue-green)
5. Emit tracing span with cycle metrics
```

### Key Design Decisions

1. **Redb-first, not file-first**: Read entries from Redb (fast, in-process) rather
   than JSON files. JSON files are the source of truth for rebuilds, but Redb is the
   hot path for reads.

2. **Batch size = 100**: Matches the issue spec. Keeps write transactions short to
   avoid blocking concurrent access tracking writes. Each batch is an independent
   Redb write transaction.

3. **One-tier-step-per-cycle**: Even if score = 0.01 (below Cold→Forgotten), a Hot
   entry only moves to Warm this cycle. This prevents data loss from scoring spikes.

4. **Inactivity policy is additive**: It can only demote, never promote. It produces
   a "worst-case" tier that is combined with score-based transitions.

5. **HNSW rebuild only on Warm→Cold**: Hot→Warm doesn't change HNSW membership.
   Cold→Forgotten doesn't change HNSW either (Cold already excluded). Only Warm→Cold
   removes entries from HNSW, requiring a rebuild.

6. **Graph edge count via LiteGraphStore**: Count inbound edges synchronously per
   entry. For the skeleton, this is acceptable — batched graph queries can be optimized
   in a follow-up.

7. **Forgotten = discard embedding**: Set `embedding = None` on the entry. The entry
   stays in JSON files but without its vector. This is irreversible at the entry level
   but the JSON file persists for audit.

8. **Knowledge file updates**: Tier changes and retention_score updates are persisted
   to both Redb AND JSON files, keeping the dual-write contract intact.

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `crates/corvia-kernel/src/gc_worker.rs` | CREATE | GC cycle logic, `spawn_gc_worker()`, `run_gc_cycle()` |
| `crates/corvia-kernel/src/lite_store.rs` | MODIFY | Add `update_tiers_batch()`, `entries_by_scope()` methods |
| `crates/corvia-kernel/src/lib.rs` | MODIFY | `pub mod gc_worker;` |
| `crates/corvia-cli/src/main.rs` | MODIFY | Spawn GC worker in `cmd_serve()` |
| `crates/corvia-kernel/src/ops.rs` | MODIFY | Update `gc_run` to also trigger a knowledge GC cycle |

### GcCycleReport struct

```rust
pub struct GcCycleReport {
    pub entries_scanned: usize,
    pub entries_scored: usize,       // non-pinned, non-forgotten
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub cold_to_forgotten: usize,
    pub warm_to_hot: usize,          // promotions
    pub cold_to_warm: usize,         // promotions
    pub hnsw_rebuild_triggered: bool,
    pub cycle_duration_ms: u64,
    pub scopes_processed: usize,
}
```

### Tracing

Single `corvia.gc.cycle` span wrapping each cycle with all report fields as attributes.

### Configuration

Uses existing `ForgettingConfig` from `config.rs`:
- `forgetting.enabled` — master switch
- `forgetting.interval_minutes` — cycle frequency (default from config)
- `forgetting.defaults.max_inactive_days` — inactivity threshold
- Per-type and per-scope overrides via `resolve_policy()`

### Test Plan

1. **Unit: scoring + transition** — already covered in `scoring.rs` tests
2. **Unit: one-tier-step constraint** — Hot at score 0.04 → Warm, not Cold
3. **Unit: inactivity policy** — force Cold when inactive + score < 0.60
4. **Unit: pinned entries skipped** — no transition regardless of score
5. **Unit: Forgotten is terminal** — no further transitions
6. **Unit: embedding discarded on Cold→Forgotten**
7. **Unit: idempotent** — running GC twice with no changes produces zero transitions
8. **Integration: full cycle on mixed scope** — entries at various tiers, verify correct transitions
