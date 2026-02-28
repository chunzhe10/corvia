use corvia_common::agent_types::{EntryStatus, MergeQueueEntry};
use corvia_common::config::MergeConfig;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::KnowledgeEntry;
use std::sync::Arc;
use tracing::{info, warn};

use crate::merge_queue::MergeQueue;
use crate::session_manager::SessionManager;
use crate::staging::StagingManager;
use crate::traits::{InferenceEngine, QueryableStore};

/// Merge worker that processes the merge queue.
///
/// For each queued entry:
/// 1. Detect conflict: search for similar Merged entries in the same scope
/// 2. No conflict → auto-merge (move staging file to knowledge/, update status)
/// 3. Conflict → LLM merge via Ollama `/api/chat`, re-embed, replace both entries
/// 4. On failure → retry with exponential backoff, mark Rejected after max retries
pub struct MergeWorker {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
    queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    session_mgr: Arc<SessionManager>,
    merge_config: MergeConfig,
    ollama_url: String,
}

impl MergeWorker {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        engine: Arc<dyn InferenceEngine>,
        queue: Arc<MergeQueue>,
        staging: Arc<StagingManager>,
        session_mgr: Arc<SessionManager>,
        merge_config: MergeConfig,
        ollama_url: String,
    ) -> Self {
        Self {
            store,
            engine,
            queue,
            staging,
            session_mgr,
            merge_config,
            ollama_url,
        }
    }

    /// Process a single merge queue entry.
    pub async fn process_one(&self, queue_entry: &MergeQueueEntry) -> Result<()> {
        // Check max retries
        if queue_entry.retry_count >= self.merge_config.max_retries {
            warn!(
                entry_id = %queue_entry.entry_id,
                retries = queue_entry.retry_count,
                "max retries exhausted, marking entry as rejected"
            );
            // Update entry status to Rejected in store
            if let Ok(Some(mut entry)) = self.store.get(&queue_entry.entry_id).await {
                entry.entry_status = EntryStatus::Rejected;
                let _ = self.store.insert(&entry).await;
            }
            self.queue.mark_complete(&queue_entry.entry_id)?;
            return Ok(());
        }

        // Get the entry from the store
        let entry = self.store.get(&queue_entry.entry_id).await?
            .ok_or_else(|| CorviaError::NotFound(
                format!("Entry {} not found in store", queue_entry.entry_id)
            ))?;

        // Detect conflict
        match self.detect_conflict(&entry).await? {
            None => {
                // No conflict — auto-merge
                self.auto_merge(&entry, &queue_entry.session_id).await?;
                self.queue.mark_complete(&queue_entry.entry_id)?;
                self.session_mgr.increment_merged(&queue_entry.session_id).ok();
                info!(entry_id = %queue_entry.entry_id, "auto_merged");
            }
            Some(conflict) => {
                // Conflict detected — attempt LLM merge
                match self.llm_merge(&entry, &conflict).await {
                    Ok(merged_entry) => {
                        // Insert merged entry, update original's status
                        self.store.insert(&merged_entry).await?;

                        // Move staging file to knowledge if exists
                        self.move_to_knowledge_if_staged(&entry, &queue_entry.session_id)?;

                        self.queue.mark_complete(&queue_entry.entry_id)?;
                        self.session_mgr.increment_merged(&queue_entry.session_id).ok();
                        info!(
                            entry_id = %queue_entry.entry_id,
                            conflict_id = %conflict.id,
                            "llm_merged"
                        );
                    }
                    Err(e) => {
                        warn!(
                            entry_id = %queue_entry.entry_id,
                            error = %e,
                            retry = queue_entry.retry_count + 1,
                            "llm_merge_failed"
                        );
                        self.queue.mark_failed(&queue_entry.entry_id, &e.to_string())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Detect if a queued entry conflicts with any existing Merged entry.
    /// Returns the conflicting entry if similarity > threshold in the same scope.
    pub async fn detect_conflict(
        &self,
        entry: &KnowledgeEntry,
    ) -> Result<Option<KnowledgeEntry>> {
        let embedding = entry.embedding.as_ref()
            .ok_or_else(|| CorviaError::Agent("Entry has no embedding for conflict detection".into()))?;

        // Search for similar entries in the same scope
        let results = self.store.search(embedding, &entry.scope_id, 5).await?;

        for result in results {
            // Skip self
            if result.entry.id == entry.id {
                continue;
            }
            // Only compare against Merged entries
            if result.entry.entry_status != EntryStatus::Merged {
                continue;
            }
            // Check similarity threshold
            if result.score > self.merge_config.similarity_threshold {
                return Ok(Some(result.entry));
            }
        }
        Ok(None)
    }

    /// Auto-merge: move staging file to knowledge/, update entry status to Merged.
    async fn auto_merge(&self, entry: &KnowledgeEntry, session_id: &str) -> Result<()> {
        // Move staging file to knowledge/
        self.move_to_knowledge_if_staged(entry, session_id)?;

        // Update entry status to Merged in the store
        let mut updated = entry.clone();
        updated.entry_status = EntryStatus::Merged;
        self.store.insert(&updated).await?;

        Ok(())
    }

    /// Move a staging file to the knowledge directory if the session has staging.
    fn move_to_knowledge_if_staged(&self, entry: &KnowledgeEntry, session_id: &str) -> Result<()> {
        if let Ok(Some(session)) = self.session_mgr.get(session_id) {
            if let Some(ref staging_dir_str) = session.staging_dir {
                let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
                // Ignore errors — file may already have been moved (idempotent)
                let _ = self.staging.move_to_knowledge(&staging_dir, &entry.id, &entry.scope_id);
            }
        }
        Ok(())
    }

    /// LLM merge: call Ollama `/api/chat` to merge two conflicting entries.
    async fn llm_merge(
        &self,
        new_entry: &KnowledgeEntry,
        existing: &KnowledgeEntry,
    ) -> Result<KnowledgeEntry> {
        let prompt = format!(
            "You are merging two knowledge entries that conflict. \
             Produce a single merged entry that preserves all important information from both.\n\n\
             Entry A (existing):\n{}\n\n\
             Entry B (new):\n{}\n\n\
             Merged entry:",
            existing.content, new_entry.content
        );

        let request_body = serde_json::json!({
            "model": self.merge_config.model,
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "stream": false
        });

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/api/chat", self.ollama_url))
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| CorviaError::Agent(format!("LLM merge request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Agent(format!(
                "LLM merge failed with status {status}: {body}"
            )));
        }

        let body: serde_json::Value = response.json().await
            .map_err(|e| CorviaError::Agent(format!("Failed to parse LLM response: {e}")))?;

        let merged_content = body["message"]["content"]
            .as_str()
            .ok_or_else(|| CorviaError::Agent("LLM response missing message.content".into()))?
            .to_string();

        // Re-embed the merged content
        let embedding = self.engine.embed(&merged_content).await?;

        // Create merged entry based on the new entry (preserves agent attribution)
        let merged = KnowledgeEntry::new(
            merged_content,
            new_entry.scope_id.clone(),
            new_entry.source_version.clone(),
        )
        .with_embedding(embedding)
        .with_agent(
            new_entry.agent_id.clone().unwrap_or_default(),
            new_entry.session_id.clone().unwrap_or_default(),
        );
        // Override status to Merged
        let mut merged = merged;
        merged.entry_status = EntryStatus::Merged;

        Ok(merged)
    }

    /// Run the merge worker loop: dequeue batch, process each, sleep if empty.
    pub async fn run(&self) {
        loop {
            match self.queue.dequeue_batch(10) {
                Ok(entries) if entries.is_empty() => {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                Ok(entries) => {
                    for entry in &entries {
                        if let Err(e) = self.process_one(entry).await {
                            warn!(entry_id = %entry.entry_id, error = %e, "merge_worker_error");
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "merge_worker_dequeue_error");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::merge_queue::MergeQueue;
    use crate::session_manager::SessionManager;
    use crate::staging::StagingManager;
    use corvia_common::agent_types::SessionState;

    struct MockEngine;
    #[async_trait::async_trait]
    impl crate::traits::InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize { 3 }
    }

    fn setup(dir: &std::path::Path) -> (MergeWorker, Arc<dyn QueryableStore>, Arc<MergeQueue>, Arc<SessionManager>) {
        let db = std::sync::Arc::new(
            redb::Database::create(dir.join("coordination.redb")).unwrap()
        );
        let store = Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let queue = Arc::new(MergeQueue::from_db(db.clone()).unwrap());
        let staging = Arc::new(StagingManager::new(dir));
        let session_mgr = Arc::new(SessionManager::from_db(db).unwrap());
        let config = MergeConfig::default();

        let worker = MergeWorker::new(
            store.clone(),
            engine,
            queue.clone(),
            staging,
            session_mgr.clone(),
            config,
            "http://127.0.0.1:11434".into(), // won't be called for non-conflict tests
        );

        (worker, store, queue, session_mgr)
    }

    #[tokio::test]
    async fn test_no_conflict_auto_merges() {
        let dir = tempfile::tempdir().unwrap();
        let (worker, store, queue, session_mgr) = setup(dir.path());
        store.init_schema().await.unwrap();

        // Insert entry A as Merged (different embedding direction)
        let mut entry_a = KnowledgeEntry::new("auth system design".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        entry_a.entry_status = EntryStatus::Merged;
        store.insert(&entry_a).await.unwrap();

        // Create a session and write entry B (very different embedding)
        let session = session_mgr.create("test::agent", true).unwrap();
        session_mgr.transition(&session.session_id, SessionState::Active).unwrap();

        let entry_b = KnowledgeEntry::new("database schema".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0])
            .with_agent("test::agent".into(), session.session_id.clone());
        store.insert(&entry_b).await.unwrap();

        // Enqueue entry B
        queue.enqueue(entry_b.id, "test::agent", &session.session_id, "scope-a").unwrap();

        let queue_entry = queue.dequeue_batch(1).unwrap().into_iter().next().unwrap();
        worker.process_one(&queue_entry).await.unwrap();

        // Entry B should now be Merged
        let updated = store.get(&entry_b.id).await.unwrap().unwrap();
        assert_eq!(updated.entry_status, EntryStatus::Merged);

        // Queue should be empty
        assert_eq!(queue.depth().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_conflict_detected_above_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let (worker, store, _queue, _session_mgr) = setup(dir.path());
        store.init_schema().await.unwrap();

        // Insert entry A as Merged
        let mut entry_a = KnowledgeEntry::new("auth system".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        entry_a.entry_status = EntryStatus::Merged;
        store.insert(&entry_a).await.unwrap();

        // Entry B: very similar embedding (cosine similarity ~0.998)
        let entry_b = KnowledgeEntry::new("auth updated".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![0.95, 0.05, 0.0])
            .with_agent("test::agent".into(), "sess".into());
        store.insert(&entry_b).await.unwrap();

        // Detect conflict
        let conflict = worker.detect_conflict(&entry_b).await.unwrap();
        assert!(conflict.is_some(), "Should detect conflict for similar embeddings");
        assert_eq!(conflict.unwrap().id, entry_a.id);
    }

    #[tokio::test]
    async fn test_failed_merge_retries() {
        let dir = tempfile::tempdir().unwrap();
        // Use an unreachable LLM URL
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let queue = Arc::new(MergeQueue::from_db(db.clone()).unwrap());
        let staging = Arc::new(StagingManager::new(dir.path()));
        let session_mgr = Arc::new(SessionManager::from_db(db).unwrap());
        let config = MergeConfig::default();

        let worker = MergeWorker::new(
            store.clone(), engine, queue.clone(), staging, session_mgr.clone(),
            config,
            "http://127.0.0.1:99999".into(), // unreachable
        );
        store.init_schema().await.unwrap();

        // Insert entry A as Merged
        let mut entry_a = KnowledgeEntry::new("auth system".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        entry_a.entry_status = EntryStatus::Merged;
        store.insert(&entry_a).await.unwrap();

        // Create session and entry B (similar — triggers conflict → LLM merge → fail)
        let session = session_mgr.create("test::agent", false).unwrap();
        let entry_b = KnowledgeEntry::new("auth updated".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![0.95, 0.05, 0.0])
            .with_agent("test::agent".into(), session.session_id.clone());
        store.insert(&entry_b).await.unwrap();

        queue.enqueue(entry_b.id, "test::agent", &session.session_id, "scope-a").unwrap();

        let queue_entry = queue.dequeue_batch(1).unwrap().into_iter().next().unwrap();
        worker.process_one(&queue_entry).await.unwrap();

        // Entry should still be in queue with retry_count=1
        let entries = queue.dequeue_batch(1).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].retry_count, 1);
        assert!(entries[0].last_error.is_some());
    }

    #[tokio::test]
    async fn test_max_retries_exhausted() {
        let dir = tempfile::tempdir().unwrap();
        let (worker, store, queue, _session_mgr) = setup(dir.path());
        store.init_schema().await.unwrap();

        // Create an entry in the store
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_agent("test::agent".into(), "sess".into());
        store.insert(&entry).await.unwrap();

        // Enqueue and manually set retry_count to max
        queue.enqueue(entry.id, "test::agent", "sess", "scope").unwrap();
        // Mark failed 3 times to reach max_retries (default 3)
        queue.mark_failed(&entry.id, "fail 1").unwrap();
        queue.mark_failed(&entry.id, "fail 2").unwrap();
        queue.mark_failed(&entry.id, "fail 3").unwrap();

        let queue_entry = queue.dequeue_batch(1).unwrap().into_iter().next().unwrap();
        assert_eq!(queue_entry.retry_count, 3);

        worker.process_one(&queue_entry).await.unwrap();

        // Queue should be empty (entry removed after max retries)
        assert_eq!(queue.depth().unwrap(), 0);

        // Entry should be Rejected in store
        let updated = store.get(&entry.id).await.unwrap().unwrap();
        assert_eq!(updated.entry_status, EntryStatus::Rejected);
    }
}
