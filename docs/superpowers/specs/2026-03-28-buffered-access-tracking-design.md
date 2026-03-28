# Buffered Access Tracking (DashMap + Periodic Flush)

**Issue:** #12
**Date:** 2026-03-28
**Status:** Design

## Problem

LiteStore's `record_access()` performs a Redb write transaction on every retrieval
call. Each write does a full entry deserialize-modify-serialize round-trip costing
~1-5ms. Under high concurrency (many agents, hundreds of searches per minute), this
becomes a measurable bottleneck on search p99 latency.

## Design

Replace direct Redb writes with an in-memory `DashMap` buffer that accumulates access
events and flushes them to Redb periodically.

### Data Structures

```rust
// New file: crates/corvia-kernel/src/access_buffer.rs

use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AccessBuffer {
    pending: DashMap<uuid::Uuid, BufferedAccess>,
    last_flush: AtomicU64,       // epoch seconds
    flush_interval_secs: u64,    // default: 60
    max_pending: usize,          // default: 256
}

struct BufferedAccess {
    count: u32,                  // accumulated access count since last flush
    last_ts: u64,                // most recent access timestamp (epoch secs)
    agents: HashSet<String>,     // unique agent IDs that accessed (for future analytics)
}
```

### Buffer Placement

The buffer lives **inside LiteStore** as a new field, not at the retriever level:

- `record_access()` on the `QueryableStore` trait stays unchanged.
- The LiteStore impl of `record_access()` writes to the DashMap instead of Redb.
- PostgresStore's `record_access()` is unaffected (keeps its direct SQL UPDATE).
- No changes to the retriever or pipeline call sites.

### Flush Triggers

Two flush triggers, whichever fires first:

1. **Time-based:** Every `flush_interval_secs` (default 60s). Checked on each
   `record_access()` call using `AtomicU64` compare.
2. **Size-based:** When `pending.len() >= max_pending` (default 256). Checked on
   each `record_access()` call after insert.

Flush is synchronous within the `record_access()` call that triggers it. This is
acceptable because:
- `record_access()` is already fire-and-forget via `tokio::spawn`
- Only one caller triggers the flush; others continue inserting into DashMap
- Flush frequency is low (at most once per 60s or 256 entries)

A `try_flush()` method uses `AtomicU64` CAS to ensure only one concurrent flush runs.
If CAS fails, the caller returns immediately (another task is already flushing).

### Flush Implementation

```rust
impl AccessBuffer {
    fn try_flush(&self, db: &Database) {
        let now = epoch_secs();
        let last = self.last_flush.load(Ordering::Relaxed);
        let time_trigger = now.saturating_sub(last) >= self.flush_interval_secs;
        let size_trigger = self.pending.len() >= self.max_pending;

        if !time_trigger && !size_trigger {
            return;
        }

        // CAS to claim the flush (only one task flushes at a time)
        if self.last_flush.compare_exchange(last, now, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return; // Another task is flushing
        }

        // Drain all pending entries
        let batch: Vec<(uuid::Uuid, BufferedAccess)> =
            self.pending.iter().map(|r| (*r.key(), r.value().clone())).collect();
        // Remove drained entries
        for (id, _) in &batch {
            self.pending.remove(id);
        }

        // Single Redb write transaction for the batch
        self.flush_to_redb(db, &batch);
    }
}
```

### Lossy Under Pressure

If `pending.len() >= max_pending` and the flush CAS fails (another flush is
in progress), the new access event is **dropped silently**. Access tracking is
optimization, not correctness. Logged at `trace!` level for observability.

### Graceful Shutdown

LiteStore already has `impl Drop` that flushes HNSW. Add an `AccessBuffer::flush_sync()`
call in the same Drop impl to flush any pending access events before shutdown.

### LiteStore Changes

```rust
pub struct LiteStore {
    // ... existing fields ...
    access_buffer: AccessBuffer,  // NEW
}
```

In `LiteStore::open()`:
```rust
access_buffer: AccessBuffer::new(60, 256),
```

In `record_access()`:
```rust
async fn record_access(&self, entry_ids: &[uuid::Uuid]) -> Result<()> {
    self.access_buffer.record(entry_ids);
    self.access_buffer.try_flush(&self.db);
    Ok(())
}
```

In `Drop for LiteStore`:
```rust
fn drop(&mut self) {
    self.access_buffer.flush_sync(&self.db);
    // ... existing HNSW flush ...
}
```

### No Background Task

Deliberately no separate `tokio::spawn` interval task for flushing. The buffer
flushes opportunistically when `record_access()` is called. Rationale:

- Avoids requiring a tokio runtime handle in LiteStore (it's currently runtime-agnostic)
- Simplifies lifecycle (no task to cancel on shutdown)
- `record_access()` is already called frequently enough from search traffic
- Drop handler ensures no data is lost on shutdown even with no traffic

### Performance Expectations

| Metric | Before (v1) | After (v2) |
|--------|-------------|------------|
| Per-access latency | ~1-5ms (Redb write) | ~50-200ns (DashMap insert) |
| Per-flush latency | N/A | <50ms (single Redb txn, N=100) |
| Memory overhead | 0 | ~100 bytes x 256 max = ~25KB |
| Redb write frequency | Every search | Every 60s or 256 accesses |

### Test Plan

1. **Unit: buffer accumulation** -- insert entries, verify pending count
2. **Unit: flush triggers** -- verify time-based and size-based triggers
3. **Unit: concurrent inserts** -- multi-threaded DashMap inserts
4. **Unit: CAS flush exclusion** -- only one flush runs at a time
5. **Unit: lossy drop** -- verify events dropped when buffer full + flush in progress
6. **Integration: record_access round-trip** -- existing tests still pass
7. **Integration: Drop flush** -- verify pending events flushed on LiteStore drop

### Files Changed

| File | Change |
|------|--------|
| `crates/corvia-kernel/Cargo.toml` | Add `dashmap = "6"` dependency |
| `crates/corvia-kernel/src/access_buffer.rs` | New: AccessBuffer + BufferedAccess |
| `crates/corvia-kernel/src/lib.rs` | Add `mod access_buffer;` |
| `crates/corvia-kernel/src/lite_store.rs` | Add buffer field, rewrite `record_access()`, update Drop |

### Approach Rejected

**Mutex<HashMap>**: Higher contention under concurrent access. DashMap's sharded
locking provides better throughput for the write-heavy access pattern.

**mpsc channel + receiver task**: More complex lifecycle (task cancellation on shutdown),
requires tokio runtime handle, no easy "peek at pending count" for size-based trigger.

**Background interval task**: Adds lifecycle complexity. The opportunistic flush on
`record_access()` is sufficient since the buffer is only relevant when search traffic
exists. No traffic = nothing to flush = no task needed.

## Verification Criteria

1. `cargo test --workspace` passes (all existing tests)
2. `cargo clippy -- -D warnings` clean
3. New unit tests for AccessBuffer (accumulation, flush triggers, concurrency, lossy drop)
4. Existing `test_record_access_*` tests in lite_store.rs continue to pass
5. Drop handler flushes pending events (verified by test)
6. Memory overhead stays under 25KB at max_pending=256
