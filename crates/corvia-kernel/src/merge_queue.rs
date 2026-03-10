use corvia_common::agent_types::MergeQueueEntry;
use corvia_common::errors::{CorviaError, Result};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

const MERGE_QUEUE: TableDefinition<&str, &[u8]> = TableDefinition::new("merge_queue");

/// FIFO merge queue backed by Redb.
/// Entries are keyed by UUID string and sorted by enqueued_at for dequeue ordering.
pub struct MergeQueue {
    db: std::sync::Arc<Database>,
}

impl MergeQueue {
    /// Create from an existing shared coordination database handle.
    pub fn from_db(db: std::sync::Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        { let _ = write_txn.open_table(MERGE_QUEUE); }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to init merge_queue table: {e}")))?;
        Ok(Self { db })
    }

    /// Enqueue an entry for merge processing.
    pub fn enqueue(
        &self,
        entry_id: Uuid,
        agent_id: &str,
        session_id: &str,
        scope_id: &str,
    ) -> Result<()> {
        let entry = MergeQueueEntry {
            entry_id,
            agent_id: agent_id.into(),
            session_id: session_id.into(),
            scope_id: scope_id.into(),
            enqueued_at: chrono::Utc::now(),
            retry_count: 0,
            last_error: None,
        };

        let bytes = serde_json::to_vec(&entry)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize queue entry: {e}")))?;
        let key = entry_id.to_string();

        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(MERGE_QUEUE)
                .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;
            table.insert(key.as_str(), bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to enqueue: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit enqueue: {e}")))?;
        Ok(())
    }

    /// List a batch of entries, sorted by enqueued_at (oldest first). Read-only (non-destructive).
    pub fn list(&self, limit: usize) -> Result<Vec<MergeQueueEntry>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(MERGE_QUEUE)
            .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;

        let mut entries = Vec::new();
        for item in table.iter()
            .map_err(|e| CorviaError::Agent(format!("Failed to iterate merge_queue: {e}")))?
        {
            let (_key, val) = item
                .map_err(|e| CorviaError::Agent(format!("Failed to read queue entry: {e}")))?;
            let bytes: &[u8] = val.value();
            let entry: MergeQueueEntry = serde_json::from_slice(bytes)
                .map_err(|e| CorviaError::Agent(format!("Failed to deserialize queue entry: {e}")))?;
            entries.push(entry);
        }

        // Sort by enqueued_at (oldest first) and take limit
        entries.sort_by_key(|e| e.enqueued_at);
        entries.truncate(limit);
        Ok(entries)
    }

    /// Mark an entry as successfully merged — removes it from the queue.
    pub fn mark_complete(&self, entry_id: &Uuid) -> Result<()> {
        let key = entry_id.to_string();
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(MERGE_QUEUE)
                .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;
            table.remove(key.as_str())
                .map_err(|e| CorviaError::Agent(format!("Failed to remove queue entry: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit dequeue: {e}")))?;
        Ok(())
    }

    /// Mark an entry as failed — increments retry_count and records the error.
    pub fn mark_failed(&self, entry_id: &Uuid, error: &str) -> Result<()> {
        let key = entry_id.to_string();
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(MERGE_QUEUE)
            .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;

        let val = table.get(key.as_str())
            .map_err(|e| CorviaError::Agent(format!("Failed to get queue entry: {e}")))?
            .ok_or_else(|| CorviaError::NotFound(format!("Queue entry {entry_id} not found")))?;
        let bytes: &[u8] = val.value();
        let mut entry: MergeQueueEntry = serde_json::from_slice(bytes)
            .map_err(|e| CorviaError::Agent(format!("Failed to deserialize queue entry: {e}")))?;

        drop(val);
        drop(table);
        drop(read_txn);

        entry.retry_count += 1;
        entry.last_error = Some(error.into());

        let updated_bytes = serde_json::to_vec(&entry)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize queue entry: {e}")))?;

        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(MERGE_QUEUE)
                .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;
            table.insert(key.as_str(), updated_bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to update queue entry: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit failed entry: {e}")))?;
        Ok(())
    }

    /// Return the number of entries in the queue.
    pub fn depth(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(MERGE_QUEUE)
            .map_err(|e| CorviaError::Agent(format!("Failed to open merge_queue: {e}")))?;
        Ok(table.len()
            .map_err(|e| CorviaError::Agent(format!("Failed to get queue depth: {e}")))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_queue() -> MergeQueue {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        MergeQueue::from_db(db).unwrap()
    }

    #[test]
    fn test_enqueue_and_dequeue() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "test::agent/sess-abc", "scope").unwrap();
        let entries = queue.list(10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_id, id);
    }

    #[test]
    fn test_dequeue_empty() {
        let queue = test_queue();
        let entries = queue.list(10).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_mark_complete() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "sess", "scope").unwrap();
        queue.mark_complete(&id).unwrap();
        assert!(queue.list(10).unwrap().is_empty());
    }

    #[test]
    fn test_mark_failed_with_retry() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "sess", "scope").unwrap();
        queue.mark_failed(&id, "Ollama down").unwrap();
        let entries = queue.list(10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].retry_count, 1);
        assert_eq!(entries[0].last_error, Some("Ollama down".into()));
    }

    #[test]
    fn test_queue_depth() {
        let queue = test_queue();
        for _ in 0..5 {
            queue.enqueue(uuid::Uuid::now_v7(), "a", "s", "scope").unwrap();
        }
        assert_eq!(queue.depth().unwrap(), 5);
    }
}
