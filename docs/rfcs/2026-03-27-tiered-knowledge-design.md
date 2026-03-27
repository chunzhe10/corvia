# Tiered Knowledge with Access Tracking and Decay

**Date:** 2026-03-27
**Status:** Design Complete
**Depends on:** M3 (GraphStore, TemporalStore), M3.2 (RAG Pipeline)
**Related issues:** chunzhe10/corvia#12 (buffered access writes), chunzhe10/corvia#13 (multi-channel RAG)

## Summary

Add tiered knowledge lifecycle management to corvia: track how entries are accessed,
classify them by memory type, compute retention scores via a composite formula, and
move entries through Hot → Warm → Cold → Forgotten tiers. Includes forgetting policies
(inactivity-based, budget-based), HNSW index management for tier transitions, and a
background GC worker. Designed to keep the active knowledge store lean while preserving
all knowledge for audit (archive-only compaction in v1).

## Motivation

1. **Unbounded growth.** Every ingestion and agent write adds entries. Without lifecycle
   management, the knowledge store grows monotonically. At 100K+ entries, HNSW search
   quality degrades and retrieval latency increases.

2. **Signal-to-noise ratio.** Systems that actively forget outperform systems that
   remember everything (Redis Agent Memory Server, SAGE). A debugging note from 3 months
   ago should not compete equally with a design decision made yesterday.

3. **Differentiated decay.** Not all knowledge ages the same way. A build instruction
   (Procedural) stays relevant for years. A session discovery (Episodic) fades in days.
   The system needs per-type decay profiles.

4. **Access-aware retention.** Knowledge that gets used should be strengthened. Knowledge
   that goes unused should fade. This is the spaced repetition principle applied to
   organizational memory.

5. **Forward compatibility.** The `memory_type` field enables future multi-channel RAG
   (one retrieval pipeline per type) without retrofit.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Agent Context Window (managed by agent harness)             │
├─────────────────────────────────────────────────────────────┤
│ Retrieval Layer (query-time scoring, session/agent boosts)  │
├─────────┬───────────┬───────────────────────────────────────┤
│ Hot     │ Warm      │ Cold              │ Forgotten         │
│ HNSW ✓  │ HNSW ✓    │ HNSW ✗            │ Archived          │
│ Full    │ Depriori- │ Brute-force       │ Embedding         │
│ search  │ tized     │ fallback only     │ discarded         │
├─────────┴───────────┴───────────────────┴───────────────────┤
│ Retention Score = 0.35×D + 0.30×A + 0.20×G + 0.15×C       │
├─────────────────────────────────────────────────────────────┤
│ Structural │ Decisional │ Episodic │ Analytical │ Procedural│
│ α=0        │ α=0.15     │ α=0.60   │ α=0.30     │ α=0.10   │
└─────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

| Layer | Responsibility | When it runs |
|-------|---------------|-------------|
| **Tier classification** | Storage lifecycle — where entries physically live | Background GC worker (periodic) |
| **Retrieval ranking** | Query-time scoring — which entries answer this query | Every search/context/ask call |
| **Working memory curation** | What enters the agent's context window | Agent harness responsibility |

These are three separate systems. Tier classification feeds into retrieval ranking
(via tier_weight multiplier), but they are not the same system.

## Memory Types

Five types, each with a distinct decay profile:

| Memory Type | What lives here | Decay α | content_role mapping |
|-------------|----------------|---------|---------------------|
| **Structural** | Code signatures, API shapes from ingestion | 0 (no decay) | `code` |
| **Decisional** | Design decisions, architectural choices, rationale | 0.15 | `design`, `decision`, `plan` |
| **Episodic** | Session discoveries, agent activity, debugging | 0.60 | `memory`, `learning` |
| **Analytical** | Synthesized findings, health checks, patterns | 0.30 | `finding` |
| **Procedural** | Instructions, workflows, how-to patterns | 0.10 | `instruction` |

### Auto-classification

Existing entries without `memory_type` are auto-classified from `content_role` using the
mapping above. Unrecognized `content_role` values (including `None`) default to Episodic
(fastest decay — safe default for unclassified knowledge).

A `corvia_reason` health check ("unmapped_content_role") warns when entries have
`content_role` values not in the mapping table, so operators can detect and reclassify
them. Log a `warn!()` on first encounter:

```rust
warn!(
    entry_id = %entry.id,
    content_role = %role,
    default_type = "Episodic",
    "unmapped_content_role"
);
```

## Tier Lifecycle

### Tiers

| Tier | HNSW indexed | Searchable | Embedding preserved | Reversible |
|------|-------------|-----------|-------------------|-----------|
| **Hot** | Yes | Default search | Yes | — |
| **Warm** | Yes | Default search (deprioritized) | Yes | — |
| **Cold** | No | Only with `include_cold=true` or brute-force fallback | Yes | Yes → promotes via next GC cycle after access |
| **Forgotten** | No | No (explicit history query only) | No (discarded) | Archive only in v1 |

### Tier Thresholds (with hysteresis)

| Transition | Threshold | Direction |
|-----------|-----------|-----------|
| Hot → Warm | score < 0.50 | Demotion |
| Warm → Cold | score < 0.25 | Demotion |
| Cold → Forgotten | score < 0.05 | Demotion |
| Cold → Warm | score ≥ 0.35 | Promotion |
| Warm → Hot | score ≥ 0.60 | Promotion |

The 0.10 gap between demotion and promotion thresholds prevents oscillation at boundaries.

### Cold is Reversible

Cold entries retain their embeddings in Redb. When accessed via brute-force fallback,
the entry is returned to the caller immediately and its `last_accessed` / `access_count`
are updated. The next GC cycle recomputes the retention score; if it crosses the
promotion threshold (≥ 0.35), the entry is promoted to Warm and re-inserted into HNSW
during the cycle's rebuild phase.

Promotion is **deferred to the GC cycle**, not synchronous. The entry is already
reachable via brute-force scan (adding ~1-3ms), so the HNSW insert is a performance
optimization, not a correctness requirement. This avoids mutation complexity with the
ArcSwap-based read path. This is hibernation, not death.

## Retention Score

### Composite Formula (Additive)

```
retention_score = 0.35 × D(t, α_type)
               + 0.30 × A(access_count, days_since_access)
               + 0.20 × G(inbound_edge_count)
               + 0.15 × C(confidence)
```

### Component Definitions

**D(t, α) — Time Decay (power-law)**
```
D(t) = (1 + t_days)^(-α)
```
Where `t_days` = days since entry creation (UTC). Negative `t_days` values from clock
skew are clamped to 0. Power-law chosen over exponential for its fat tail — old entries
retain nonzero scores, enabling Cold tier reversibility.

**A(access_count, days_since_access) — Access Signal**
```
A = frequency × recency
frequency = log(1 + access_count) / FREQ_NORMALIZER
recency   = (1 + days_since_last_access)^(-0.3)

where FREQ_NORMALIZER = log(101.0) = 4.615  (precomputed constant)
```
Log-scaled frequency with diminishing returns — an entry accessed ~50 times reaches
~85% of maximum frequency signal. The normalizer of 100 means an entry accessed 100+
times saturates at 1.0. Power-law recency with its own fat tail.
For never-accessed entries: A = 0.

**G(inbound_edges) — Graph Connectivity**
```
G = min(inbound_edges / 10, 1.0)
```
Inbound edges only (entries that point TO this entry). Entries with ≥10 inbound edges
score maximum. Uses degree centrality (O(1) per entry), not PageRank.

**C(confidence) — Confidence Signal**
```
C = confidence if set, else 0.7 (default)
```
Applies to all entry types. Decisional and Analytical entries should set explicit values.
The 0.7 default reflects that most entries pass through ingestion or agent merge pipelines
which provide baseline quality assurance — we prefer false retention over false forgetting
in v1.

### Supersession Penalty

Entries with `superseded_by` set (a newer version exists) receive a 0.5× multiplier
on the final retention score:

```
if entry.superseded_by.is_some() {
    retention_score *= 0.5;
}
```

This ensures stale-but-accessed entries decay faster than their successors. A superseded
entry needs twice the access/graph signal to maintain the same tier.

Additionally, Structural entries with `superseded_by` set are treated as Episodic for
decay purposes (α=0.60 instead of α=0). Superseded code signatures are stale artifacts
from prior ingestion and should decay normally.

### Properties

- **Additive, not multiplicative**: Access and graph signals can rescue old entries from decay.
  A 6-month-old decision that is frequently accessed and well-connected stays Hot.
- **New entries start Warm** (~0.45-0.50 score): Stored with `tier: Hot` initially (full
  HNSW participation), but the first GC cycle reclassifies them to Warm based on score.
  Must earn sustained Hot status through access or connections. Session affinity boosts
  for current-session entries are a retrieval-layer concern.
- **Max score = 1.0**: All components normalized to [0, 1], weights sum to 1.0.
- **Superseded entries decay faster**: The 0.5× penalty makes it hard for stale entries
  to compete with their successors, even if still accessed by agents unaware of the update.
- **Expected steady-state distribution**: ~2-5% Hot, ~25-35% Warm, ~50-60% Cold,
  ~10-15% Forgotten. A small Hot tier is by design — Hot means actively used and
  well-connected, not just "recently created." If operators see a different distribution,
  use `corvia gc status` and the `forgotten_access_attempts` counter to calibrate.

### Worked Examples

**Episodic entry, 14 days old, accessed twice (last access 1 day ago), no edges, confidence 0.6:**
```
D = (1+14)^(-0.6) = 15^(-0.6) = 1/5.08     = 0.197
A = (log(3)/log(101)) × (1+1)^(-0.3)
  = 0.238 × 0.812                            = 0.193
G = 0, C = 0.6

score = 0.35×0.197 + 0.30×0.193 + 0 + 0.15×0.6
      = 0.069 + 0.058 + 0 + 0.090            = 0.217 → Cold
```
Episodic entries decay fast (α=0.60). Two accesses in 14 days is low engagement — Cold
is correct. The entry auto-promotes if accessed again.

**Decisional entry, 180 days old, accessed 30 times (last access 2 days ago), 6 inbound edges, confidence 0.9:**
```
D = (1+180)^(-0.15) = 181^(-0.15) = 1/2.181 = 0.458
A = (log(31)/log(101)) × (1+2)^(-0.3)
  = 0.744 × 0.719                            = 0.535
G = min(6/10, 1.0) = 0.6, C = 0.9

score = 0.35×0.458 + 0.30×0.535 + 0.20×0.6 + 0.15×0.9
      = 0.160 + 0.161 + 0.120 + 0.135        = 0.576 → Hot (stays Hot)
```
Score 0.576 is above the Hot→Warm demotion threshold (0.50), so the entry remains Hot.
It would not *promote* to Hot from Warm (needs ≥ 0.60), but it doesn't get demoted either.
A 6-month-old decision stays Hot because it is frequently accessed and well-connected.

**Episodic entry, 60 days old, never accessed, no edges, confidence 0.5:**
```
D = (1+60)^(-0.6) = 61^(-0.6) = 1/11.78     = 0.085
A = 0, G = 0, C = 0.5

score = 0.35×0.085 + 0 + 0 + 0.15×0.5
      = 0.030 + 0 + 0 + 0.075                = 0.105 → Cold
```
Unused episodic entry at 60 days — Cold. Still retrievable via brute-force fallback.

## Schema Changes

### New fields on KnowledgeEntry

```rust
pub enum MemoryType {
    Structural,
    Decisional,
    Episodic,
    Analytical,
    Procedural,
}

pub enum Tier {
    Hot,
    Warm,
    Cold,
    Forgotten,
}

// Added to KnowledgeEntry:
pub memory_type: MemoryType,          // inferred from content_role if not set
pub confidence: Option<f32>,          // 0.0-1.0, defaults to 0.7 in scoring
pub last_accessed: Option<DateTime<Utc>>,
pub access_count: u32,                // saturating increment
pub tier: Tier,                       // computed by GC worker, stored for fast filtering
pub tier_changed_at: Option<DateTime<Utc>>,  // when tier last transitioned
pub retention_score: Option<f32>,     // last computed score, for debugging/dashboard
pub pinned: bool,                     // explicit pin, skips all forgetting
pub pinned_by: Option<String>,        // agent or user who pinned
pub pinned_at: Option<DateTime<Utc>>, // when pinned
```

### Backward Compatibility

Existing entries without the new fields get defaults via `#[serde(default)]`:
- `memory_type`: inferred from `content_role` (see mapping table). Default: `Episodic`
- `confidence`: None (scored as 0.7). Default: `None`
- `last_accessed`: None (scored as never-accessed). Default: `None`
- `access_count`: 0. Default: `0`
- `tier`: Hot (all existing entries start Hot, GC worker reclassifies on first run). Default: `Hot`
- `tier_changed_at`: None. Default: `None`
- `retention_score`: None (computed on first GC cycle). Default: `None`
- `pinned`: false. Default: `false`
- `pinned_by`: None. Default: `None`
- `pinned_at`: None. Default: `None`

All new fields use `#[serde(default)]` with explicit `Default` impls to ensure
backward-compatible deserialization of existing Redb entries and JSON knowledge files.
The first GC cycle after upgrade may take longer as it reclassifies all entries.

## Access Tracking

### What counts as an access

**Tier 1 (strong signal):** Entry returned in a retrieval result set delivered to an agent
via `corvia_search`, `corvia_context`, or `corvia_ask`. Updates `last_accessed` and
increments `access_count`.

**Tier 2 (not tracked in v1):** Entry was a candidate but filtered out, or traversed
during graph operations. Deferred — no clear signal value at current scale.

### Write path

Direct Redb write transaction in the retrieval path. All access updates for a single
retrieval call are batched into one write transaction (not per-entry). No buffering
beyond that for v1 — overhead is acceptable at current scale (~5K entries, dozens of
searches per session). Buffered writes via DashMap deferred to chunzhe10/corvia#12.

## Forgetting Policies

### Two policies for v1

**1. Inactivity-based (primary)**

Force entry to Cold if `last_accessed` exceeds `max_inactive_days` AND retention_score
is below 0.60. For entries that have never been accessed (`last_accessed: None`),
use `recorded_at` as the fallback — the entry is considered inactive since creation.

This prevents the cliff-edge scenario where a well-connected, high-confidence entry
(e.g., a foundational design decision known by heart but never explicitly queried)
is forced Cold purely due to inactivity. Entries with `retention_score >= 0.60` are
exempt from inactivity-based demotion — their graph connectivity and confidence signals
are strong enough to justify retention even without recent access.

**2. Budget-based (capacity)**

Per-scope cap on total active entries (Hot + Warm). Pinned entries are excluded from the
count — the effective capacity is `budget_top_n + pinned_count`. When the non-pinned
active count exceeds `budget_top_n`, rank all non-pinned entries by retention_score,
demote the lowest by one tier step. Auto-protected entries can be budget-demoted to
Warm but not to Cold or Forgotten.

### Configuration

```toml
[forgetting]
enabled = true
interval_minutes = 60

[forgetting.defaults]
max_inactive_days = 90
budget_top_n = 10000

[forgetting.by_type.episodic]
max_inactive_days = 14    # safety net; score-based demotion handles most episodic
                          # entries by ~day 4 (alpha=0.60 drops score below 0.25)

[forgetting.by_type.structural]
enabled = false  # refreshed by ingestion, not forgotten

[forgetting.by_type.decisional]
max_inactive_days = 365

[forgetting.by_type.procedural]
max_inactive_days = 180
```

Per-scope overrides:
```toml
[scopes.compliance.forgetting]
max_inactive_days = 3650  # 10 years
budget_top_n = 0          # no limit
```

Config hierarchy: global defaults → per-type overrides → per-scope overrides.

### Pinning

**Explicit pin:** `pinned: bool` field. Skips all forgetting evaluation. Requires
manual unpin.

**Auto-protection** (prevents Cold → Forgotten only, not demotion):
- `memory_type == Structural`
- HEAD of supersession chain (not superseded by anything)
- `inbound_edges >= 5`
- `content_role == "decision"` AND `confidence >= 0.9`

## Compaction Strategy (v1)

**Archive only.** When an entry hits Forgotten (score < 0.05):
1. Mark tier as Forgotten
2. Discard embedding (set to None)
3. Exclude from all search paths
4. JSON file preserved in `.corvia/knowledge/` for audit

No LLM summarization. No chain merging. Storage cost is negligible at current scale
(~2KB per archived entry). Supersession chain compaction deferred to v2.

## HNSW Index Management

### Demotion (Warm → Cold)

Lazy orphaning: delete Redb `UUID_TO_HNSW` / `HNSW_TO_UUID` mappings during GC cycle.
Orphaned vectors remain in HNSW graph (filtered out on lookup miss). Single blue-green
rebuild at end of GC cycle if any demotions occurred.

### Promotion (Cold → Warm)

Deferred to GC cycle. When a Cold entry is accessed via brute-force, only `last_accessed`
and `access_count` are updated in Redb. The next GC cycle recomputes the score; if it
crosses the promotion threshold, the entry is included in the rebuild's HNSW insert batch.

This avoids the ArcSwap mutation problem (`hnsw_rs::insert()` requires `&mut Hnsw`, but
`ArcSwap::load()` returns read-only guards). The trade-off is up to 60 minutes where the
promoted entry is only reachable via brute-force, not HNSW. At corvia's scale, brute-force
adds ~1-3ms — acceptable for the architectural simplicity gained.

### Concurrency: ArcSwap for Lock-Free HNSW Access

Replace `Arc<Mutex<Hnsw>>` with `Arc<ArcSwap<Hnsw>>` (from the `arc-swap` crate).
Search calls `load()` (wait-free, no lock). GC rebuild builds a new `Hnsw` instance
without holding any lock, then calls `store(Arc::new(new_hnsw))` for an atomic pointer
swap. The old index is dropped when the last reader releases its guard — outside any
lock, on a background thread via `tokio::task::spawn_blocking(move || drop(old_hnsw))`.

This eliminates the query blackout during HNSW rebuild. At 100K entries, the old index
is ~300MB; deferring its deallocation avoids a 10-50ms stall under the mutex.

### Rebuild triggers

- End of GC cycle (if any Warm → Cold transitions)
- After `delete_scope`
- After manual `corvia_gc_run`
- When orphan ratio exceeds 20% of total HNSW entries
- Blue-green pattern: build new HNSW in background, atomic swap via ArcSwap

### Cold entry search

Brute-force cosine scan over Redb-stored embeddings. Returns top-K results (same K as
the HNSW search limit), merged with Hot+Warm results by score before final truncation.
Uses `rayon::par_iter()` for parallel cosine computation when cold entry count > 1,000.

Performance at corvia's scale:

| Cold entries | Scan time (sequential) | Scan time (rayon, 4 cores) |
|-------------|----------------------|---------------------------|
| 1,000 | ~1ms | ~1ms (overhead dominates) |
| 5,000 | ~3ms | ~1.5ms |
| 10,000 | ~5-10ms | ~3-5ms |

**Note:** These estimates assume embeddings are read from Redb via a single snapshot read
transaction (batch iteration), not individual lookups. If cold entries grow beyond 10K,
consider a dedicated `COLD_EMBEDDINGS` Redb table for sequential-scan-friendly storage,
or enable the secondary cold HNSW index.

Secondary cold HNSW index deferred — only needed above 50K cold entries.

### PostgresStore

Partial HNSW indexes with WHERE clause:
```sql
CREATE INDEX idx_hot ON knowledge USING hnsw (embedding vector_cosine_ops)
  WHERE tier IN ('hot', 'warm');
```
Tier changes handled automatically by PostgreSQL. No manual rebuild needed.

## Background Worker

### Execution

In-process `tokio::spawn` task inside `corvia serve`. Default interval: 60 minutes.
Configurable via `forgetting.interval_minutes`.

### GC Cycle

```
1. Load entries per scope (paginated, 1000/page)
2. For each entry:
   a. Skip if pinned
   b. Compute retention_score (with supersession penalty)
   c. Determine tier transition: take the WORSE of:
      - Inactivity policy: if last_accessed > max_inactive_days AND score < 0.60 → Cold
      - Score-based: compare against tier thresholds (with hysteresis)
   d. Record tier transition if changed (max one tier step per cycle per entry)
3. Apply budget policy: if scope > budget_top_n, rank non-pinned entries by
   retention_score, demote lowest by one tier step. Skip entries already
   transitioned in step 2.
4. Apply all transitions in batched Redb write transactions (100 entries/txn):
   a. Hot → Warm: update tier field
   b. Warm → Cold: update tier + delete HNSW mappings
   c. Cold → Forgotten: update tier + discard embedding + mark archived
5. If any Warm → Cold transitions: trigger HNSW rebuild (blue-green via ArcSwap)
6. Emit metrics and transition log (see Observability section)
```

**Key constraints:**
- **One tier step per cycle per entry**: An entry cannot go Hot → Cold in a single cycle.
  This preserves the hysteresis design and prevents budget policy from undoing promotions.
- **Budget after scoring, before physical transitions**: Budget-forced demotions are
  merged with score-based demotions into a single batch of Redb writes and one rebuild.
- **Batched write transactions**: GC uses short-lived Redb write txns (100 entries/txn)
  to avoid blocking concurrent access-tracking writes from the retrieval path.

### Safeguards

- **Pinned entries**: Skip entirely (pinned entries excluded from budget count)
- **Auto-protected**: Can demote but cannot transition to Forgotten
- **Supersession chain protection**: All entries in an active supersession chain (HEAD
  is Hot/Warm/Cold) are auto-protected from Forgotten. Only chains where the HEAD itself
  is Forgotten can have their ancestors archived.
- **Rate limit**: Max 50 Forgotten transitions per cycle
- **Circuit breaker**: Abort if >50% Redb writes fail in a cycle
- **Idempotent**: Crash-safe — next run re-evaluates all entries

### Estimated runtime

- 5K entries: <1 second scoring + 1-3s rebuild = under 5 seconds total
- 100K entries: 1-5 seconds scoring (snapshot read txn) + 30-90s rebuild = under 2 minutes total

## Observability

### Entry-Level Fields

Every entry exposes its lifecycle state for debugging and dashboard display:

- `tier`: Current tier (Hot/Warm/Cold/Forgotten) — included in all search results
- `tier_changed_at`: Timestamp of last tier transition
- `retention_score`: Last computed composite score (updated each GC cycle)
- `pinned_by` / `pinned_at`: Audit trail for explicit pins

### Span Constants (corvia-telemetry)

New span constants following the `corvia.{subsystem}.{operation}` convention:

```rust
// In crates/corvia-telemetry/src/lib.rs, add to pub mod spans:
// GC_CYCLE is a child span of the existing GC_RUN (corvia.gc.run).
// GC_RUN is the top-level trigger (manual or periodic); GC_CYCLE is one
// evaluation pass over a scope.
pub const GC_CYCLE: &str = "corvia.gc.cycle";
pub const GC_SCORE: &str = "corvia.gc.score";
pub const GC_TRANSITION: &str = "corvia.gc.transition";
pub const GC_REBUILD: &str = "corvia.gc.rebuild";
pub const ACCESS_RECORD: &str = "corvia.access.record";
```

### Transition Log

Each tier transition is emitted as a structured `info!()` event inside the
`corvia.gc.cycle` span, following corvia's existing pattern (named fields,
`%` for Display types, message as last argument):

```rust
info!(
    entry_id = %entry.id,
    scope_id = %entry.scope_id,
    memory_type = %entry.memory_type,
    from_tier = %old_tier,
    to_tier = %new_tier,
    retention_score,
    d_score,
    a_score,
    g_score,
    c_score,
    superseded,
    reason,        // "score_decay" | "inactivity" | "budget"
    "tier_transition"
);
```

These events are captured by the `DashboardTraceLayer` (which filters to `corvia.*`
spans) and written to `corvia-traces.log` as JSON lines, making them queryable from
the dashboard's trace viewer.

### GC Cycle Instrumentation

The GC cycle function uses `#[tracing::instrument]` with the span constant:

```rust
#[tracing::instrument(name = "corvia.gc.cycle", skip(self), fields(
    scope_id,
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
async fn run_gc_cycle(&self, scope_id: &str) -> Result<()> { ... }
```

Fields are updated via `tracing::Span::current().record()` as the cycle progresses.
The `forgotten_access_attempts` counter tracks how often agents attempt to access
Forgotten entries (via direct ID lookups such as `corvia_history` or `corvia_graph`
traversal hitting an archived entry). A high rate signals thresholds are too aggressive.
Normal search paths exclude Forgotten entries, so this counter only fires on explicit
lookups.

### MCP Tool Changes

- `corvia_search` / `corvia_context` / `corvia_ask`: Results include `tier` and
  `retention_score` fields. New `include_cold: bool` parameter (default false).
- `corvia_system_status`: Includes per-scope tier distribution
  (`{hot: N, warm: N, cold: N, forgotten: N}`).
- New `corvia_pin` MCP tool: Pin an entry by ID. Requires `agent_id` for audit trail.
  Returns confirmation with `pinned_by` and `pinned_at`.
- New `corvia_unpin` MCP tool: Unpin an entry by ID. **Requires confirmation** (consistent
  with existing MCP patterns for `corvia_gc_run`, `corvia_agent_suspend`), since unpinning
  makes the entry eligible for forgetting.

### Dashboard

- **Tier distribution chart**: Per-scope breakdown of Hot/Warm/Cold/Forgotten counts
- **Recent transitions**: Table of recent tier transitions with score breakdown
- **Pinned entries**: List of all pinned entries per scope with pin metadata
- **GC history**: Timeline of GC cycle runs with metrics

### CLI Commands

```bash
corvia gc status                 # Show tier distribution per scope
corvia gc run                    # Trigger manual GC cycle
corvia gc history                # Show recent GC cycle metrics
corvia pin <entry-id>            # Pin an entry (prevents forgetting)
corvia unpin <entry-id>          # Unpin an entry
corvia inspect <entry-id>        # Show entry with full lifecycle metadata
                                 # (tier, score, components, access history)
```

### Rollback

If the feature needs to be disabled or reverted:

1. **Set `forgetting.enabled = false`**: Stops all GC cycles. All non-Forgotten entries
   are treated as Hot for retrieval purposes (tier_weight = 1.0 for all).
2. **Hot/Warm/Cold entries**: Fully intact. Re-enabling forgetting resumes normal scoring.
3. **Forgotten entries**: Embeddings were discarded. To restore, run `corvia rebuild`
   which re-reads JSON knowledge files (Git-as-truth), re-generates embeddings via
   inference, and re-inserts into HNSW. This requires the inference server to be running.
4. **Access metadata**: `last_accessed`, `access_count`, `retention_score` remain on
   entries but are ignored when forgetting is disabled. No data loss.

## RAG Integration Points

Two small changes to the existing retrieval path:

**1. Tier-aware scoring (retriever.rs)**
```rust
let tier_weight = match entry.tier {
    Tier::Hot => 1.0,
    Tier::Warm => 0.7,
    Tier::Cold => 0.3,
    Tier::Forgotten => 0.0,  // excluded from search
};
final_score = cosine_similarity * tier_weight;
```

**2. Access recording (after retrieval returns)**
```rust
for entry in &results {
    record_access(entry.id, agent_id, &write_txn)?;
}
```

**3. RetrievalOpts extension**
```rust
pub include_cold: bool,  // default false — search cold tier via brute-force
```

## Forward Compatibility: Multi-Channel RAG

The `memory_type` field enables a future architecture where each type gets a dedicated
retrieval channel with specialized strategy (chunzhe10/corvia#13):

| Channel | Strategy |
|---------|---------|
| Structural | Keyword/exact match |
| Decisional | Graph-heavy + vector |
| Episodic | Recency-weighted vector |
| Analytical | Vector + confidence reranking |
| Procedural | Vector + access-frequency boost |

No changes to this design are needed to support this. The existing `Retriever` trait
accommodates per-type implementations. A future `MultiChannelRetriever` wraps all
channels and fuses results via Reciprocal Rank Fusion.

## Deferred Work

| Item | Reason | Tracked |
|------|--------|---------|
| Buffered access writes (DashMap) | Premature optimization at current scale | chunzhe10/corvia#12 |
| Multi-channel RAG | Separate feature, forward-compatible | chunzhe10/corvia#13 |
| Agent breadth tracking (`access_agents`) | No consumer in v1, add when multi-channel RAG needs it | — |
| Age-based forgetting policy | Redundant with power-law decay | — |
| Supersession chain compaction | Knowledge loss risk, needs LLM dependency | v2 |
| Cluster compaction | Research problem, poor risk/reward | — |
| Secondary cold HNSW index | Only needed above 50K cold entries | — |
| PageRank for graph scoring | O(E×iterations), degree centrality sufficient | — |
| Outbound edge scoring | Inbound-only captures structural importance | — |
| TIER_INDEX Redb table | GC already does full scan; use in-memory counters instead | — |

## Research References

| System/Paper | Key Contribution |
|---|---|
| Redis Agent Memory Server | 5-stage lifecycle, 4 forgetting policies, background compaction |
| Hindsight (arxiv 2512.12818) | 4-network architecture, TEMPR parallel retrieval, 91.4% LongMemEval |
| MAGMA (arxiv 2601.03236) | Multi-graph decomposition (semantic/temporal/causal/entity) |
| SAGE (arxiv 2409.00872) | Ebbinghaus curve as active pruning policy |
| Knowledge Objects (arxiv 2603.17781) | Anti-compaction: 60% knowledge loss from cascading summarization |
| Kahana & Adler (2002, UPenn) | Human aggregate forgetting follows power-law, not exponential |
| Tsinghua Memory Survey (arxiv 2512.13564) | 4-tier cognitive taxonomy (working/episodic/semantic/procedural) |
| Caffeine/Moka | W-TinyLFU, striped ring buffer, lossy access tracking |
