# Buffered Access Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LiteStore's direct Redb writes for access tracking with an in-memory DashMap buffer that flushes periodically, reducing per-access cost from ~1-5ms to ~50-200ns.

**Architecture:** New `access_buffer.rs` module containing `AccessBuffer` with `DashMap<Uuid, BufferedAccess>`. LiteStore gains an `access_buffer` field. `record_access()` inserts into the DashMap and opportunistically flushes to Redb when time or size thresholds are met. Drop handler ensures no data loss on shutdown.

**Tech Stack:** dashmap 6, std::sync::atomic, redb 2, chrono, uuid

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/corvia-kernel/Cargo.toml` | Modify | Add `dashmap = "6"` dependency |
| `crates/corvia-kernel/src/access_buffer.rs` | Create | AccessBuffer struct, BufferedAccess, record/flush/drop logic |
| `crates/corvia-kernel/src/lib.rs` | Modify | Add `pub mod access_buffer;` |
| `crates/corvia-kernel/src/lite_store.rs` | Modify | Add buffer field, rewrite `record_access()`, update `Drop` |

---

### Task 1: Add dashmap dependency

**Files:**
- Modify: `crates/corvia-kernel/Cargo.toml`

- [ ] **Step 1: Add dashmap to dependencies**

In `crates/corvia-kernel/Cargo.toml`, add after the `chrono` line:

```toml
dashmap = "6"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p corvia-kernel`
Expected: compiles with no errors

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-kernel/Cargo.toml
git commit -m "feat(kernel): add dashmap dependency for buffered access tracking"
```

---

### Task 2: Create AccessBuffer module

**Files:**
- Create: `crates/corvia-kernel/src/access_buffer.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

- [ ] **Step 1: Write unit tests for AccessBuffer**

Create `crates/corvia-kernel/src/access_buffer.rs` with the full module including tests at the bottom. The tests should cover:

```rust
//! In-memory buffer for access tracking writes.
//!
//! Accumulates `record_access()` events in a `DashMap` and flushes them to Redb
//! periodically. Reduces per-access cost from ~1-5ms (Redb write) to ~50-200ns
//! (DashMap insert). Access tracking is optimization, not correctness — events
//! may be dropped under pressure.

use dashmap::DashMap;
use redb::Database;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, trace, warn};

/// Buffered access event for a single entry.
#[derive(Debug, Clone)]
struct BufferedAccess {
    /// Accumulated access count since last flush.
    count: u32,
    /// Most recent access timestamp (epoch seconds).
    last_ts: u64,
}

/// In-memory buffer that batches access tracking writes for periodic Redb flush.
///
/// # Design
///
/// - `record()` inserts into a `DashMap` (~50-200ns per call).
/// - `try_flush()` drains the map and writes a single Redb transaction.
/// - Flush triggers: time-based (every `flush_interval_secs`) or size-based
///   (when pending entries >= `max_pending`).
/// - `AtomicBool` CAS ensures only one flush runs at a time.
/// - Lossy under pressure: if buffer is full and flush is in progress, events are dropped.
pub struct AccessBuffer {
    pending: DashMap<uuid::Uuid, BufferedAccess>,
    last_flush: AtomicU64,
    flush_interval_secs: u64,
    max_pending: usize,
    flushing: AtomicBool,
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl AccessBuffer {
    /// Create a new buffer with the given flush interval and max pending count.
    pub fn new(flush_interval_secs: u64, max_pending: usize) -> Self {
        Self {
            pending: DashMap::new(),
            last_flush: AtomicU64::new(epoch_secs()),
            flush_interval_secs,
            max_pending,
            flushing: AtomicBool::new(false),
        }
    }

    /// Record access for a batch of entry IDs.
    ///
    /// Inserts into the DashMap. If the buffer is full and a flush is already
    /// in progress, the event is silently dropped (lossy under pressure).
    pub fn record(&self, entry_ids: &[uuid::Uuid]) {
        let now = epoch_secs();
        for id in entry_ids {
            if self.pending.len() >= self.max_pending && self.flushing.load(Ordering::Relaxed) {
                trace!(entry_id = %id, "access buffer full + flush in progress, dropping event");
                continue;
            }
            self.pending
                .entry(*id)
                .and_modify(|ba| {
                    ba.count = ba.count.saturating_add(1);
                    ba.last_ts = now;
                })
                .or_insert(BufferedAccess { count: 1, last_ts: now });
        }
    }

    /// Attempt to flush pending access events to Redb.
    ///
    /// Only flushes if time or size thresholds are met. Uses AtomicBool CAS
    /// to ensure only one concurrent flush. Returns the number of entries flushed.
    pub fn try_flush(&self, db: &Arc<Database>) -> usize {
        let now = epoch_secs();
        let last = self.last_flush.load(Ordering::Relaxed);
        let time_trigger = now.saturating_sub(last) >= self.flush_interval_secs;
        let size_trigger = self.pending.len() >= self.max_pending;

        if !time_trigger && !size_trigger {
            return 0;
        }

        // CAS to claim the flush lock.
        if self.flushing.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return 0;
        }

        let flushed = self.flush_inner(db);

        self.last_flush.store(epoch_secs(), Ordering::Relaxed);
        self.flushing.store(false, Ordering::Release);

        flushed
    }

    /// Synchronous flush of all pending events. Called from LiteStore::Drop.
    pub fn flush_sync(&self, db: &Arc<Database>) -> usize {
        if self.pending.is_empty() {
            return 0;
        }
        self.flush_inner(db)
    }

    /// Number of pending (unflushed) entries.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Internal flush implementation shared by try_flush and flush_sync.
    fn flush_inner(&self, db: &Arc<Database>) -> usize {
        use crate::lite_store::ENTRIES;
        use corvia_common::types::KnowledgeEntry;
        use redb::ReadableTable;

        // Drain all pending entries atomically (per-key).
        let batch: Vec<(uuid::Uuid, BufferedAccess)> = self
            .pending
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect();

        if batch.is_empty() {
            return 0;
        }

        // Remove drained entries from the map.
        for (id, _) in &batch {
            self.pending.remove(id);
        }

        let count = batch.len();

        let write_txn = match db.begin_write() {
            Ok(txn) => txn,
            Err(e) => {
                warn!(error = %e, "AccessBuffer: failed to begin write txn for flush");
                return 0;
            }
        };

        {
            let mut entries_table = match write_txn.open_table(ENTRIES) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "AccessBuffer: failed to open ENTRIES for flush");
                    return 0;
                }
            };

            for (id, ba) in &batch {
                let uuid_str = id.to_string();
                let entry_bytes = match entries_table.get(uuid_str.as_str()) {
                    Ok(Some(val)) => val.value().to_vec(),
                    Ok(None) => continue,
                    Err(e) => {
                        warn!(entry_id = %id, error = %e, "AccessBuffer: failed to read entry");
                        continue;
                    }
                };

                let mut entry: KnowledgeEntry = match serde_json::from_slice(&entry_bytes) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(entry_id = %id, error = %e, "AccessBuffer: failed to deserialize");
                        continue;
                    }
                };

                // Apply buffered access: add accumulated count, set latest timestamp.
                entry.access_count = entry.access_count.saturating_add(ba.count);
                entry.last_accessed = Some(
                    chrono::DateTime::from_timestamp(ba.last_ts as i64, 0)
                        .unwrap_or_else(chrono::Utc::now),
                );

                let updated_bytes = match serde_json::to_vec(&entry) {
                    Ok(b) => b,
                    Err(e) => {
                        warn!(entry_id = %id, error = %e, "AccessBuffer: failed to serialize");
                        continue;
                    }
                };

                if let Err(e) = entries_table.insert(uuid_str.as_str(), updated_bytes.as_slice()) {
                    warn!(entry_id = %id, error = %e, "AccessBuffer: failed to write entry");
                }
            }
        }

        if let Err(e) = write_txn.commit() {
            warn!(error = %e, "AccessBuffer: failed to commit flush transaction");
            return 0;
        }

        debug!(count, "AccessBuffer: flushed access events to Redb");
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_accumulates_counts() {
        let buf = AccessBuffer::new(3600, 1000); // high thresholds so no auto-flush
        let id1 = uuid::Uuid::now_v7();
        let id2 = uuid::Uuid::now_v7();

        buf.record(&[id1, id2]);
        buf.record(&[id1]); // id1 accessed twice

        assert_eq!(buf.pending_count(), 2);

        let ba1 = buf.pending.get(&id1).unwrap();
        assert_eq!(ba1.count, 2);

        let ba2 = buf.pending.get(&id2).unwrap();
        assert_eq!(ba2.count, 1);
    }

    #[test]
    fn test_record_empty_input() {
        let buf = AccessBuffer::new(60, 256);
        buf.record(&[]);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_try_flush_no_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());
        let buf = AccessBuffer::new(3600, 1000); // very high thresholds
        let id = uuid::Uuid::now_v7();
        buf.record(&[id]);

        // Neither time nor size trigger should fire.
        let flushed = buf.try_flush(&db);
        assert_eq!(flushed, 0);
        assert_eq!(buf.pending_count(), 1); // still pending
    }

    #[test]
    fn test_try_flush_size_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        // Create ENTRIES table.
        {
            let txn = db.begin_write().unwrap();
            txn.open_table(crate::lite_store::ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 2); // max_pending = 2

        let id1 = uuid::Uuid::now_v7();
        let id2 = uuid::Uuid::now_v7();
        buf.record(&[id1, id2]);

        // Size trigger fires (2 >= 2).
        let flushed = buf.try_flush(&db);
        // Entries don't exist in Redb so they're skipped, but flush still runs.
        assert_eq!(flushed, 2);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_try_flush_time_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        // Create ENTRIES table.
        {
            let txn = db.begin_write().unwrap();
            txn.open_table(crate::lite_store::ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(0, 1000); // flush_interval = 0 means always triggers
        let id = uuid::Uuid::now_v7();
        buf.record(&[id]);

        let flushed = buf.try_flush(&db);
        assert_eq!(flushed, 1);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_flush_sync_drains_all() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        // Create ENTRIES table.
        {
            let txn = db.begin_write().unwrap();
            txn.open_table(crate::lite_store::ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 1000);
        for _ in 0..10 {
            buf.record(&[uuid::Uuid::now_v7()]);
        }
        assert_eq!(buf.pending_count(), 10);

        let flushed = buf.flush_sync(&db);
        assert_eq!(flushed, 10);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_flush_sync_empty_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());
        let buf = AccessBuffer::new(60, 256);

        let flushed = buf.flush_sync(&db);
        assert_eq!(flushed, 0);
    }

    #[test]
    fn test_concurrent_inserts() {
        let buf = Arc::new(AccessBuffer::new(3600, 100_000));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let buf = Arc::clone(&buf);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    buf.record(&[uuid::Uuid::now_v7()]);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(buf.pending_count(), 800);
    }

    #[test]
    fn test_lossy_drop_when_full_and_flushing() {
        let buf = AccessBuffer::new(3600, 2); // max_pending = 2
        let id1 = uuid::Uuid::now_v7();
        let id2 = uuid::Uuid::now_v7();
        let id3 = uuid::Uuid::now_v7();

        buf.record(&[id1, id2]);
        assert_eq!(buf.pending_count(), 2);

        // Simulate flush in progress.
        buf.flushing.store(true, Ordering::Relaxed);

        // This should be dropped because buffer is full + flush in progress.
        buf.record(&[id3]);
        assert_eq!(buf.pending_count(), 2);
        assert!(buf.pending.get(&id3).is_none());

        // Cleanup.
        buf.flushing.store(false, Ordering::Relaxed);
    }
}
```

- [ ] **Step 2: Register the module in lib.rs**

In `crates/corvia-kernel/src/lib.rs`, add after `pub mod gc_worker;` (line 100):

```rust
pub mod access_buffer;
```

- [ ] **Step 3: Make ENTRIES table constant accessible**

The `ENTRIES` table definition in `lite_store.rs` is currently `const` (private). The `access_buffer` module needs to access it. Change line 17 from:

```rust
const ENTRIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
```

to:

```rust
pub(crate) const ENTRIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-kernel access_buffer`
Expected: All 8 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/corvia-kernel/src/access_buffer.rs crates/corvia-kernel/src/lib.rs crates/corvia-kernel/src/lite_store.rs
git commit -m "feat(kernel): add AccessBuffer module with DashMap + periodic flush"
```

---

### Task 3: Integrate AccessBuffer into LiteStore

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs`

- [ ] **Step 1: Add access_buffer field to LiteStore struct**

In `lite_store.rs`, add to the `LiteStore` struct (after the `tantivy` field at line 65):

```rust
    /// Buffered access tracking. Accumulates record_access() events and flushes
    /// to Redb periodically. See `access_buffer::AccessBuffer` for design.
    access_buffer: crate::access_buffer::AccessBuffer,
```

- [ ] **Step 2: Initialize buffer in LiteStore::open()**

In the `LiteStore::open()` method, add `access_buffer` to the struct construction (the `Ok(Self { ... })` block):

```rust
            access_buffer: crate::access_buffer::AccessBuffer::new(60, 256),
```

- [ ] **Step 3: Rewrite record_access() to use buffer**

Replace the current `record_access()` implementation (lines 1074-1137) with:

```rust
    #[tracing::instrument(name = "corvia.access.record", skip(self), fields(count = entry_ids.len()))]
    async fn record_access(&self, entry_ids: &[uuid::Uuid]) -> Result<()> {
        if entry_ids.is_empty() {
            return Ok(());
        }
        self.access_buffer.record(entry_ids);
        self.access_buffer.try_flush(&self.db);
        Ok(())
    }
```

- [ ] **Step 4: Update Drop to flush access buffer**

In `impl Drop for LiteStore` (line 896), add buffer flush before HNSW flush:

```rust
impl Drop for LiteStore {
    fn drop(&mut self) {
        let flushed = self.access_buffer.flush_sync(&self.db);
        if flushed > 0 {
            tracing::info!(count = flushed, "LiteStore: flushed pending access events on drop");
        }
        if let Err(e) = self.flush_hnsw() {
            tracing::warn!(error = %e, "LiteStore: failed to flush HNSW on drop");
        } else {
            tracing::info!("LiteStore: HNSW index persisted to disk");
        }
    }
}
```

- [ ] **Step 5: Run existing access tracking tests**

Run: `cargo test -p corvia-kernel test_record_access`
Expected: All 3 existing tests pass (test_record_access_updates_fields, test_record_access_batch, test_record_access_nonexistent_entry_graceful)

**IMPORTANT:** The existing tests call `record_access()` then immediately `get()` to verify. With buffering, the writes are deferred. The tests need the buffer to flush. Two options:
- The tests insert real entries and the buffer doesn't auto-flush (thresholds not met).
- We need to force a flush in tests by dropping the store or calling `flush_sync` directly.

The simplest fix: since `record_access()` calls `try_flush()`, and the default thresholds are 60s/256, small tests won't trigger a flush. We need to either:
(a) Add a `flush_access_buffer()` public method on LiteStore for test use, or
(b) Lower the thresholds, or
(c) Let the Drop handle it.

**Approach:** Add a `pub fn flush_access_buffer(&self)` method on LiteStore that calls `self.access_buffer.flush_sync(&self.db)`. Existing tests call this after `record_access()`.

```rust
impl LiteStore {
    /// Force-flush the access buffer to Redb. Used by tests and graceful shutdown.
    pub fn flush_access_buffer(&self) -> usize {
        self.access_buffer.flush_sync(&self.db)
    }
}
```

Then update the existing tests to call `store.flush_access_buffer()` after `record_access()`:

In `test_record_access_updates_fields`:
```rust
store.record_access(&[entry_id]).await.unwrap();
store.flush_access_buffer();
// ... then get and assert ...
```

In `test_record_access_batch`:
```rust
store.record_access(&ids).await.unwrap();
store.flush_access_buffer();
// ... then get and assert ...
```

`test_record_access_nonexistent_entry_graceful` doesn't check values after, so no change needed.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p corvia-kernel`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/corvia-kernel/src/lite_store.rs
git commit -m "feat(kernel): integrate AccessBuffer into LiteStore record_access()"
```

---

### Task 4: Full workspace build and clippy

**Files:** (none new)

- [ ] **Step 1: Build workspace**

Run: `cargo build --workspace`
Expected: Compiles with no errors

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings or errors

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 4: Fix any issues found**

If clippy or tests fail, fix issues and re-run.

- [ ] **Step 5: Commit fixes if any**

```bash
git add -u
git commit -m "fix(kernel): address clippy warnings in access buffer"
```
