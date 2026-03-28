//! In-memory buffer for access tracking writes.
//!
//! Accumulates `record_access()` events in a [`DashMap`] and flushes them to Redb
//! periodically. Reduces per-access cost from ~1-5ms (Redb write) to ~50-200ns
//! (DashMap insert). Access tracking is optimization, not correctness — events
//! may be dropped under pressure.

use dashmap::DashMap;
use redb::{Database, ReadableTable};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, trace, warn};

use crate::lite_store::ENTRIES;
use corvia_common::types::KnowledgeEntry;

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
/// - [`record()`](Self::record) inserts into a `DashMap` (~50-200ns per call).
/// - [`try_flush()`](Self::try_flush) drains the map and writes a single Redb transaction.
/// - Flush triggers: time-based (every `flush_interval_secs`) or size-based
///   (when pending entries >= `max_pending`).
/// - [`AtomicBool`] CAS ensures only one flush runs at a time.
/// - Lossy under pressure: if buffer is full and flush is in progress, events are dropped.
///
/// # Concurrency note
///
/// `DashMap::len()` is approximate under contention (sums per-shard counts), so
/// the `max_pending` threshold may fire slightly early or late. This is acceptable
/// since access tracking is best-effort telemetry.
pub(crate) struct AccessBuffer {
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
    pub(crate) fn new(flush_interval_secs: u64, max_pending: usize) -> Self {
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
    pub(crate) fn record(&self, entry_ids: &[uuid::Uuid]) {
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
                .or_insert(BufferedAccess {
                    count: 1,
                    last_ts: now,
                });
        }
    }

    /// Attempt to flush pending access events to Redb.
    ///
    /// Only flushes if time or size thresholds are met. Uses [`AtomicBool`] CAS
    /// to ensure only one concurrent flush. Returns the number of entries
    /// successfully written to Redb.
    pub(crate) fn try_flush(&self, db: &Arc<Database>) -> usize {
        let now = epoch_secs();
        let last = self.last_flush.load(Ordering::Acquire);
        let time_trigger = now.saturating_sub(last) >= self.flush_interval_secs;
        let size_trigger = self.pending.len() >= self.max_pending;

        if !time_trigger && !size_trigger {
            return 0;
        }

        // CAS to claim the flush lock.
        if self
            .flushing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return 0;
        }

        let flushed = self.flush_inner(db);

        self.last_flush.store(now, Ordering::Release);
        self.flushing.store(false, Ordering::Release);

        flushed
    }

    /// Synchronous flush of all pending events. Called from `LiteStore::Drop`
    /// and `LiteStore::flush_access_buffer()`.
    ///
    /// Acquires the flush lock to prevent concurrent `try_flush` from racing.
    /// Spins briefly if another flush is in progress (unlikely in practice
    /// since this is called during shutdown).
    pub(crate) fn flush_sync(&self, db: &Arc<Database>) -> usize {
        if self.pending.is_empty() {
            return 0;
        }

        // Spin until we acquire the flush lock. In practice, contention only
        // happens if try_flush is mid-flight during Drop, which is rare.
        loop {
            match self.flushing.compare_exchange(
                false,
                true,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => std::hint::spin_loop(),
            }
        }

        let flushed = self.flush_inner(db);

        self.flushing.store(false, Ordering::Release);

        flushed
    }

    /// Number of pending (unflushed) entries. Used for testing and diagnostics.
    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Internal flush implementation shared by `try_flush` and `flush_sync`.
    ///
    /// Uses atomic per-key `DashMap::remove` to drain entries, avoiding the
    /// race where a concurrent `record()` inserts between collect and remove.
    /// Returns the number of entries successfully written to Redb.
    fn flush_inner(&self, db: &Arc<Database>) -> usize {
        // Atomically drain each entry via remove(). Any concurrent record()
        // that arrives after remove() for a given key will re-insert it,
        // and that entry will be picked up by the next flush cycle.
        let keys: Vec<uuid::Uuid> = self.pending.iter().map(|r| *r.key()).collect();

        if keys.is_empty() {
            return 0;
        }

        let batch: Vec<(uuid::Uuid, BufferedAccess)> = keys
            .into_iter()
            .filter_map(|id| self.pending.remove(&id))
            .collect();

        if batch.is_empty() {
            return 0;
        }

        let write_txn = match db.begin_write() {
            Ok(txn) => txn,
            Err(e) => {
                warn!(error = %e, "AccessBuffer: failed to begin write txn for flush");
                // Re-insert entries so they aren't lost.
                for (id, ba) in batch {
                    self.pending.entry(id).or_insert(ba);
                }
                return 0;
            }
        };

        let mut success_count = 0usize;

        {
            let mut entries_table = match write_txn.open_table(ENTRIES) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "AccessBuffer: failed to open ENTRIES for flush");
                    for (id, ba) in batch {
                        self.pending.entry(id).or_insert(ba);
                    }
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
                        warn!(
                            entry_id = %id,
                            error = %e,
                            "AccessBuffer: failed to deserialize"
                        );
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
                        warn!(
                            entry_id = %id,
                            error = %e,
                            "AccessBuffer: failed to serialize"
                        );
                        continue;
                    }
                };

                if let Err(e) =
                    entries_table.insert(uuid_str.as_str(), updated_bytes.as_slice())
                {
                    warn!(
                        entry_id = %id,
                        error = %e,
                        "AccessBuffer: failed to write entry"
                    );
                    continue;
                }

                success_count += 1;
            }
        }

        if let Err(e) = write_txn.commit() {
            warn!(
                error = %e,
                "AccessBuffer: failed to commit flush transaction"
            );
            // Re-insert entries so they aren't lost on commit failure.
            for (id, ba) in batch {
                self.pending
                    .entry(id)
                    .and_modify(|existing| {
                        existing.count = existing.count.saturating_add(ba.count);
                        if ba.last_ts > existing.last_ts {
                            existing.last_ts = ba.last_ts;
                        }
                    })
                    .or_insert(ba);
            }
            return 0;
        }

        debug!(
            success_count,
            total = batch.len(),
            "AccessBuffer: flushed access events to Redb"
        );
        success_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_accumulates_counts() {
        let buf = AccessBuffer::new(3600, 1000);
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
        let buf = AccessBuffer::new(3600, 1000);
        let id = uuid::Uuid::now_v7();
        buf.record(&[id]);

        let flushed = buf.try_flush(&db);
        assert_eq!(flushed, 0);
        assert_eq!(buf.pending_count(), 1);
    }

    #[test]
    fn test_try_flush_size_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        // Create ENTRIES table.
        {
            let txn = db.begin_write().unwrap();
            txn.open_table(ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 2);
        let id1 = uuid::Uuid::now_v7();
        let id2 = uuid::Uuid::now_v7();
        buf.record(&[id1, id2]);

        let flushed = buf.try_flush(&db);
        // Entries don't exist in Redb so success_count is 0, but drain still runs.
        assert_eq!(flushed, 0);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_try_flush_time_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        {
            let txn = db.begin_write().unwrap();
            txn.open_table(ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        // flush_interval_secs = 0 means time trigger always fires.
        let buf = AccessBuffer::new(0, 1000);
        let id = uuid::Uuid::now_v7();
        buf.record(&[id]);

        let flushed = buf.try_flush(&db);
        // Entry doesn't exist in Redb, so success_count is 0.
        assert_eq!(flushed, 0);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_flush_sync_drains_all() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        {
            let txn = db.begin_write().unwrap();
            txn.open_table(ENTRIES).unwrap();
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 1000);
        for _ in 0..10 {
            buf.record(&[uuid::Uuid::now_v7()]);
        }
        assert_eq!(buf.pending_count(), 10);

        let flushed = buf.flush_sync(&db);
        // Entries don't exist in Redb, but drain still runs.
        assert_eq!(flushed, 0);
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
        let buf = AccessBuffer::new(3600, 2);
        let id1 = uuid::Uuid::now_v7();
        let id2 = uuid::Uuid::now_v7();
        let id3 = uuid::Uuid::now_v7();

        buf.record(&[id1, id2]);
        assert_eq!(buf.pending_count(), 2);

        // Simulate flush in progress.
        buf.flushing.store(true, Ordering::Relaxed);

        // All events are dropped when buffer full + flushing (new and existing keys).
        buf.record(&[id3]);
        assert_eq!(buf.pending_count(), 2);
        assert!(buf.pending.get(&id3).is_none());

        buf.record(&[id1]); // existing key also dropped
        let ba1 = buf.pending.get(&id1).unwrap();
        assert_eq!(ba1.count, 1); // unchanged

        buf.flushing.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_flush_updates_redb_entries() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        // Create ENTRIES table and insert a test entry.
        let entry =
            KnowledgeEntry::new("test content".into(), "scope".into(), "v1".into());
        let entry_id = entry.id;
        {
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(ENTRIES).unwrap();
                let bytes = serde_json::to_vec(&entry).unwrap();
                table
                    .insert(entry_id.to_string().as_str(), bytes.as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 1000);
        buf.record(&[entry_id]);
        buf.record(&[entry_id]);
        buf.record(&[entry_id]);

        let flushed = buf.flush_sync(&db);
        assert_eq!(flushed, 1); // 1 unique entry successfully written

        // Read entry back from Redb and verify.
        let read_txn = db.begin_read().unwrap();
        let table = read_txn.open_table(ENTRIES).unwrap();
        let val = table.get(entry_id.to_string().as_str()).unwrap().unwrap();
        let updated: KnowledgeEntry = serde_json::from_slice(val.value()).unwrap();

        assert_eq!(updated.access_count, 3);
        assert!(updated.last_accessed.is_some());
    }

    #[test]
    fn test_multiple_flushes_accumulate_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());

        let entry =
            KnowledgeEntry::new("accumulation test".into(), "scope".into(), "v1".into());
        let entry_id = entry.id;
        {
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(ENTRIES).unwrap();
                let bytes = serde_json::to_vec(&entry).unwrap();
                table
                    .insert(entry_id.to_string().as_str(), bytes.as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        let buf = AccessBuffer::new(3600, 1000);

        // First flush: 3 accesses.
        buf.record(&[entry_id, entry_id, entry_id]);
        let flushed = buf.flush_sync(&db);
        assert_eq!(flushed, 1);

        // Second flush: 2 more accesses.
        buf.record(&[entry_id, entry_id]);
        let flushed = buf.flush_sync(&db);
        assert_eq!(flushed, 1);

        // Verify accumulated count in Redb.
        let read_txn = db.begin_read().unwrap();
        let table = read_txn.open_table(ENTRIES).unwrap();
        let val = table.get(entry_id.to_string().as_str()).unwrap().unwrap();
        let updated: KnowledgeEntry = serde_json::from_slice(val.value()).unwrap();

        assert_eq!(updated.access_count, 5); // 3 + 2
    }

    #[test]
    fn test_flush_sync_acquires_lock() {
        // Verify flush_sync waits for an in-progress flush to complete.
        let buf = Arc::new(AccessBuffer::new(3600, 1000));
        buf.record(&[uuid::Uuid::now_v7()]);

        // Simulate flush in progress.
        buf.flushing.store(true, Ordering::Relaxed);

        let buf_clone = Arc::clone(&buf);
        let handle = std::thread::spawn(move || {
            // This should spin until flushing is released.
            let dir = tempfile::tempdir().unwrap();
            let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());
            {
                let txn = db.begin_write().unwrap();
                txn.open_table(ENTRIES).unwrap();
                txn.commit().unwrap();
            }
            buf_clone.flush_sync(&db)
        });

        // Give the thread time to start spinning.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Release the lock.
        buf.flushing.store(false, Ordering::Release);

        // The thread should now complete.
        let result = handle.join().unwrap();
        assert_eq!(result, 0); // Entry doesn't exist in that db
        assert_eq!(buf.pending_count(), 0);
    }
}
