# Dashboard Index Coverage & Staleness Detection

**Date**: 2026-03-26
**Status**: Draft (rev 2 — post 5-persona review)
**Issue**: corvia-workspace#18

## Problem

`GET /api/dashboard/status` does not report whether the HNSW index is stale or
how much of the knowledge store is indexed. Users and automation (e.g., Taskfile
post-start hooks) have no API-level signal that search results may be incomplete
after a partial ingest or index corruption.

Three divergence scenarios exist across the storage layers:

1. **Partial ingest** — knowledge JSON files written to disk but never inserted
   into Redb/HNSW (e.g., embedding server down mid-ingest). Disk > Redb = HNSW.
2. **Index corruption/rebuild** — HNSW wiped or rebuilt but Redb and disk files
   remain intact. Disk = Redb > HNSW.
3. **Orphaned index entries** — entries deleted from disk (manual or GC) but
   still present in Redb/HNSW. Redb or HNSW > Disk.

## Design

### Approach

Standalone `IndexCoverageCache` struct in a new `dashboard/coverage.rs` module.
Computed on server startup, cached with a configurable TTL (default 60s),
refreshable via explicit endpoint. No background tasks, no filesystem watchers.

Three-layer comparison: **disk files vs Redb store vs HNSW vector index**. This
detects all three divergence scenarios above, not just partial ingest.

### Data Model

New fields on `DashboardStatusResponse` (`corvia-common/src/dashboard.rs`):

```rust
/// Coverage ratio of HNSW-indexed entries vs knowledge files on disk.
/// null when disk_count == 0 (fresh workspace, no data yet).
/// Always serialized (as null, not omitted) for frontend ergonomics.
pub index_coverage: Option<f64>,  // 0.0-1.0, null if no files on disk

/// true when coverage < configured threshold.
/// null when index_coverage is null (no data to be stale about).
pub index_stale: Option<bool>,

/// Raw counts for debugging and actionable UI messages.
/// e.g., "73 of 100 entries indexed" is more useful than "73% coverage".
pub index_disk_count: u64,        // .json files in knowledge/{scope_id}/
pub index_store_count: u64,       // entries in Redb SCOPE_INDEX
pub index_hnsw_count: u64,        // entries in Redb HNSW_TO_UUID table

/// The configured staleness threshold, so consumers can display it.
/// e.g., "Coverage 85% (threshold: 90%)"
pub index_stale_threshold: f64,

/// ISO 8601 timestamp of when coverage was last computed.
/// Lets consumers show data freshness: "checked 45s ago".
pub index_coverage_checked_at: Option<String>,
```

Note: `index_coverage` and `index_stale` are always present in the JSON (as
`null` when not applicable), never omitted. This avoids absent-field vs `null`
ambiguity in TypeScript consumers.

New cache struct (`dashboard/coverage.rs`):

```rust
pub struct IndexCoverageCache {
    disk_count: u64,
    store_count: u64,
    hnsw_count: u64,
    coverage: Option<f64>,
    stale: Option<bool>,
    last_computed: Option<Instant>,
    last_computed_wall: Option<chrono::DateTime<chrono::Utc>>,
    ttl: Duration,
    threshold: f64,
}
```

Wrapped in `Arc<tokio::sync::Mutex<IndexCoverageCache>>` on `AppState`. Using
`tokio::sync::Mutex` (not `std::sync::Mutex`) because `get()` performs async
I/O (`store.count()`) and we must not hold a sync lock across `.await` points.

### Config

New optional section in `corvia.toml`:

```toml
[dashboard]
stale_threshold = 0.9      # coverage below this = index_stale: true
coverage_ttl_secs = 60     # cache TTL for coverage recomputation
```

In `corvia-common/src/config.rs`:

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DashboardSection {
    #[serde(default = "default_stale_threshold")]
    pub stale_threshold: f64,       // default: 0.9, validated: 0.0..=1.0
    #[serde(default = "default_coverage_ttl")]
    pub coverage_ttl_secs: u64,     // default: 60, minimum: 5
}

impl Default for DashboardSection {
    fn default() -> Self {
        Self { stale_threshold: 0.9, coverage_ttl_secs: 60 }
    }
}
```

Wired into the top-level `CorviaConfig`:

```rust
#[serde(default)]
pub dashboard: DashboardSection,
```

**Validation** (at config load time):
- `stale_threshold` must be in `0.0..=1.0`. Values outside this range produce a
  config load error.
- `coverage_ttl_secs` has a floor of 5 seconds. Values below 5 are clamped to 5
  with a warning log, to prevent excessive disk scanning.

**Hot-reload**: Add `"dashboard"` to the hot-reload allowlist in
`crates/corvia-server/src/mcp.rs`. Safe because the cache reads config values on
each recompute, so new TTL/threshold values take effect within one TTL cycle.

### Computation Logic

`IndexCoverageCache::get()` method:

```rust
pub async fn get(
    &self,
    data_dir: &Path,
    scope_id: &str,
    store: &dyn QueryableStore,
) -> CoverageSnapshot
```

Returns a `CoverageSnapshot` struct with all fields needed by the response.

**Steps:**

1. If `last_computed` is `Some(t)` and `Instant::now() - t < ttl`:
   return cached snapshot. Log `tracing::debug!` with cached values.

2. Otherwise recompute:

   a. **Disk count** — via `tokio::task::spawn_blocking` to avoid blocking the
      async runtime. Uses `knowledge_files::scope_dir(data_dir, scope_id)` to
      construct the path (inherits `validate_scope_id` for path traversal safety).
      - Directory does not exist → `disk_count = 0` (fresh workspace).
      - `read_dir` fails (permissions, NFS) → `disk_count = 0`, log `tracing::warn!`.
      - Count `.json` files only (non-recursive, flat directory — matches
        `knowledge_files::write_entry` layout).

   b. **Store count** — `store.count(scope_id).await`. This is the Redb
      `SCOPE_INDEX` entry count.

   c. **HNSW count** — count entries in the Redb `HNSW_TO_UUID` table. This
      requires exposing a new method on `LiteStore` (see Files Changed). For
      `PostgresStore`, return `store_count` (pgvector manages its own index).

   d. **Coverage** — `hnsw_count` vs `disk_count`:
      - If `disk_count == 0`: `coverage = None`, `stale = None`.
      - Else: `coverage = Some((hnsw_count as f64 / disk_count as f64).min(1.0))`.
      - `stale = coverage.map(|c| c < threshold)`.

   e. **Orphan detection** — if `hnsw_count > disk_count` or
      `store_count > disk_count`, log `tracing::warn!` with counts. Coverage is
      still clamped to 1.0 (orphaned entries don't reduce search quality), but
      the raw counts in the response let consumers detect this condition.

   f. **Observability** — `tracing::info!` on every recompute:
      ```
      index coverage recomputed: disk={disk_count} store={store_count}
      hnsw={hnsw_count} coverage={coverage:?} stale={stale:?}
      compute_ms={elapsed_ms}
      ```

   g. Update `last_computed` and `last_computed_wall`.

3. Return snapshot.

### Refresh Endpoint

`POST /api/dashboard/status/refresh-coverage`

Path follows existing patterns (`/gc/run`, `/merge/retry` — action nested under
its parent resource). Recomputes immediately (not lazy) and returns the fresh
snapshot:

```json
{
  "index_coverage": 0.73,
  "index_stale": true,
  "index_disk_count": 100,
  "index_store_count": 100,
  "index_hnsw_count": 73,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z"
}
```

Returns `200 OK` with body. Single round-trip, no race conditions.

### Integration Points

- **Startup**: Create `IndexCoverageCache::new(threshold, ttl)` during `AppState`
  construction. Perform initial `get()` to populate cache.
- **`status_handler`**: Call `coverage_cache.lock().await.get(...)` and populate
  response fields. The `tokio::sync::Mutex` ensures correct async behavior.
- **Router**: Add refresh route to dashboard router.
- **PostgresStore**: `index_hnsw_count` returns same as `index_store_count`
  (pgvector manages its own index consistency). `index_coverage` compares
  `store_count` vs `disk_count` — but for PostgresStore there are no disk files,
  so `coverage = None`, `stale = None`. This is correct: PostgresStore has no
  separate disk layer to diverge from.

### Response Example

**Healthy (fully indexed):**
```json
{
  "services": [...],
  "entry_count": 100,
  "index_coverage": 1.0,
  "index_stale": false,
  "index_disk_count": 100,
  "index_store_count": 100,
  "index_hnsw_count": 100,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z",
  ...
}
```

**Stale (partial ingest):**
```json
{
  "index_coverage": 0.73,
  "index_stale": true,
  "index_disk_count": 100,
  "index_store_count": 73,
  "index_hnsw_count": 73,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z"
}
```

**HNSW divergence (index corruption):**
```json
{
  "index_coverage": 0.0,
  "index_stale": true,
  "index_disk_count": 100,
  "index_store_count": 100,
  "index_hnsw_count": 0,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z"
}
```

**Fresh workspace (no data):**
```json
{
  "index_coverage": null,
  "index_stale": null,
  "index_disk_count": 0,
  "index_store_count": 0,
  "index_hnsw_count": 0,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z"
}
```

**Orphaned entries (store > disk):**
```json
{
  "index_coverage": 1.0,
  "index_stale": false,
  "index_disk_count": 80,
  "index_store_count": 100,
  "index_hnsw_count": 100,
  "index_stale_threshold": 0.9,
  "index_coverage_checked_at": "2026-03-26T12:34:56Z"
}
```
Note: coverage is clamped to 1.0 but `store_count > disk_count` signals orphans.
A server-side `tracing::warn!` is also emitted.

## Files Changed

| File | Change |
|------|--------|
| `crates/corvia-common/src/config.rs` | Add `DashboardSection` struct with defaults and validation, wire into `CorviaConfig` |
| `crates/corvia-common/src/dashboard.rs` | Add 7 new fields to `DashboardStatusResponse` |
| `crates/corvia-server/src/dashboard/coverage.rs` | New file: `IndexCoverageCache`, `CoverageSnapshot` |
| `crates/corvia-server/src/dashboard/mod.rs` | Wire cache into `status_handler`, add refresh route |
| `crates/corvia-server/src/rest.rs` | Add `coverage_cache` field to `AppState` |
| `crates/corvia-server/src/mcp.rs` | Add `"dashboard"` to hot-reload allowlist |
| `crates/corvia-kernel/src/lite_store.rs` | Add `hnsw_entry_count()` method (count `HNSW_TO_UUID` table) |

## Tests

### Unit tests (`coverage.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_fresh_workspace_no_files` | disk=0 → coverage=None, stale=None |
| `test_knowledge_dir_missing` | directory does not exist → disk=0 (distinct from empty dir) |
| `test_full_coverage` | disk=N, hnsw=N → coverage=1.0, stale=false |
| `test_partial_coverage` | disk=100, hnsw=73 → coverage=0.73, stale=true |
| `test_threshold_boundary_exact` | coverage == threshold → stale=false (not strictly less) |
| `test_threshold_boundary_below` | coverage just below threshold → stale=true |
| `test_hnsw_gt_disk_orphaned` | hnsw > disk → coverage clamped to 1.0, stale=false |
| `test_store_gt_disk_orphaned` | store > disk → coverage clamped, warning logged |
| `test_ttl_returns_cached` | second call within TTL returns same values without recompute |
| `test_ttl_expired_recomputes` | call after TTL triggers fresh disk scan |
| `test_invalid_json_in_dir` | non-deserializable .json file inflates disk_count (documented behavior) |
| `test_read_dir_permission_error` | unreadable directory → disk=0, warning logged |
| `test_concurrent_access` | multiple tasks calling get() concurrently — no deadlocks |

### Config tests (`config.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_dashboard_defaults` | omitted `[dashboard]` section → threshold=0.9, ttl=60 |
| `test_dashboard_partial_override` | only threshold set → ttl gets default |
| `test_threshold_out_of_range` | value > 1.0 or < 0.0 → config load error |
| `test_ttl_below_minimum` | ttl=1 → clamped to 5 with warning |

### Integration tests (`dashboard/mod.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_status_includes_coverage_fields` | all 7 new fields present with correct types |
| `test_status_coverage_values` | create N knowledge files, index M → coverage=M/N, stale matches |
| `test_refresh_endpoint_returns_fresh_values` | POST refresh returns computed snapshot, not empty |
| `test_refresh_forces_recompute` | POST refresh then GET status → updated `checked_at` |

### LiteStore tests (`lite_store.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_hnsw_entry_count` | count matches inserted entries |
| `test_hnsw_entry_count_after_delete` | count decreases after entry removal |

## Design Notes

- **Coverage is best-effort**: disk count and HNSW count are read sequentially,
  not atomically. An ingest completing between the two reads could produce a
  transiently inaccurate ratio. The TTL cache makes this a ~60s eventual
  consistency window, which is acceptable for a dashboard metric.
- **Flat directory assumption**: `knowledge_files::write_entry` writes to
  `knowledge/{scope_id}/{uuid}.json` (flat, no subdirectories). If the layout
  changes to support sharding, the disk count logic must be updated.
- **Invalid JSON files**: a manually placed or corrupt `.json` file in the
  knowledge directory inflates `disk_count`, causing coverage to appear lower
  than reality. This is documented behavior — removing the file fixes it.

## Non-Goals

- Frontend warning banner (tracked in corvia-workspace#19, blocked on this).
- Automatic ingest triggering when stale.
- Per-scope coverage (single default scope only for now).
- Forward-compatible ingest trigger endpoint (future work).
- PostgresStore coverage (pgvector is self-consistent; returns null coverage).
