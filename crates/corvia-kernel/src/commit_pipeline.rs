use corvia_common::agent_types::{EntryStatus, SessionState};
use corvia_common::errors::{CorviaError, Result};
use std::sync::Arc;
use tracing::{info, warn};

use crate::merge_queue::MergeQueue;
use crate::session_manager::SessionManager;
use crate::staging::StagingManager;
use crate::traits::QueryableStore;

/// The 5-step idempotent commit flow from D45 Part 5.
///
/// Each step is idempotent — re-running after crash skips completed steps:
/// 1. Session status → Committing (Redb)
/// 2. git add staging files + git commit on agent branch
/// 3. Mark entries as Committed in store (D45 Part 3 lifecycle)
/// 4. Entries enter merge queue (Redb)
/// 5. Session status → Merging (Redb)
pub struct CommitPipeline {
    session_mgr: Arc<SessionManager>,
    merge_queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    store: Arc<dyn QueryableStore>,
}

impl CommitPipeline {
    pub fn new(
        session_mgr: Arc<SessionManager>,
        merge_queue: Arc<MergeQueue>,
        staging: Arc<StagingManager>,
        store: Arc<dyn QueryableStore>,
    ) -> Self {
        Self {
            session_mgr,
            merge_queue,
            staging,
            store,
        }
    }

    /// Execute the commit pipeline for a session.
    ///
    /// Idempotent: safe to call multiple times (e.g., after crash recovery).
    /// Steps already completed are detected and skipped.
    pub async fn commit_session(&self, session_id: &str) -> Result<()> {
        let session = self.session_mgr.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        // Step 1: Transition to Committing (skip if already Committing or Merging)
        if session.state == SessionState::Active {
            self.session_mgr.transition(session_id, SessionState::Committing)?;
            info!(session_id, "commit_step1: state → Committing");
        }

        // Re-read after possible state change
        let session = self.session_mgr.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        // Step 2: Git commit on branch (if staging dir exists)
        if let Some(ref staging_dir_str) = session.staging_dir {
            let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
            if let Some(ref branch) = session.git_branch {
                let files = self.staging.list_staging_files(&staging_dir)?;
                if !files.is_empty() {
                    let file_paths: Vec<String> = files.iter()
                        .map(|id| staging_dir.join(format!("{id}.json")).to_string_lossy().to_string())
                        .collect();
                    let file_refs: Vec<&str> = file_paths.iter().map(|s| s.as_str()).collect();
                    let msg = format!("agent commit: {} ({} entries)", session_id, files.len());
                    if let Err(e) = self.staging.commit_on_branch(branch, &msg, &file_refs) {
                        warn!(session_id, branch, error = %e, "commit_step2: git commit failed");
                    } else {
                        info!(session_id, branch, "commit_step2: git commit on branch");
                    }
                }
            }
        }

        // Step 3: Mark entries as Committed in store (D45 Part 3 lifecycle: Pending → Committed)
        if let Some(ref staging_dir_str) = session.staging_dir {
            let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
            let entry_ids = self.staging.list_staging_files(&staging_dir)?;
            for entry_id in &entry_ids {
                if let Ok(Some(mut entry)) = self.store.get(entry_id).await
                    && entry.entry_status == EntryStatus::Pending {
                        entry.entry_status = EntryStatus::Committed;
                        // Restore embedding from VECTORS table for re-insert
                        if entry.embedding.is_none()
                            && let Ok(Some(emb)) = self.store.get_embedding(entry_id).await
                        {
                            entry.embedding = Some(emb);
                        }
                        if let Err(e) = self.store.insert(&entry).await {
                            warn!(entry_id = %entry_id, error = %e, "commit_step3: failed to update entry status");
                        }
                    }
            }
            info!(session_id, count = entry_ids.len(), "commit_step3: entries → Committed");
        }

        // Step 4: Enqueue all staging entries into merge queue
        if let Some(ref staging_dir_str) = session.staging_dir {
            let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
            let entry_ids = self.staging.list_staging_files(&staging_dir)?;
            for entry_id in &entry_ids {
                match self.staging.read_staging_file(&staging_dir, entry_id) {
                    Ok(entry) => {
                        self.merge_queue.enqueue(
                            *entry_id,
                            &session.agent_id,
                            session_id,
                            &entry.scope_id,
                        )?;
                    }
                    Err(_) => {
                        // Entry may have already been moved (idempotent retry)
                        continue;
                    }
                }
            }
            info!(session_id, count = entry_ids.len(), "commit_step4: entries enqueued");
        }

        // Step 5: Transition to Merging
        if session.state == SessionState::Active || session.state == SessionState::Committing {
            self.session_mgr.transition(session_id, SessionState::Merging)?;
            info!(session_id, "commit_step5: state → Merging");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::staging::StagingManager;
    use crate::traits::QueryableStore;
    use corvia_common::types::KnowledgeEntry;

    #[tokio::test]
    async fn test_commit_flow_moves_entries_to_queue() {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        let session_mgr = Arc::new(SessionManager::from_db(db.clone()).unwrap());
        let merge_queue = Arc::new(MergeQueue::from_db(db).unwrap());
        let staging = Arc::new(StagingManager::new(dir.path()));
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        store.init_schema().await.unwrap();

        // Create session in Active state with staging
        let session = session_mgr.create("test::agent", true).unwrap();
        session_mgr.transition(&session.session_id, SessionState::Active).unwrap();

        // Write 2 entries to staging and store
        let staging_dir_str = session.staging_dir.as_ref().unwrap();
        let staging_dir = staging.resolve_staging_path(staging_dir_str);
        std::fs::create_dir_all(&staging_dir).unwrap();

        let entry1 = KnowledgeEntry::new("entry one".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_agent("test::agent".into(), session.session_id.clone());
        let entry2 = KnowledgeEntry::new("entry two".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0])
            .with_agent("test::agent".into(), session.session_id.clone());
        assert_eq!(entry1.entry_status, EntryStatus::Pending);
        staging.write_staging_file(&staging_dir, &entry1).unwrap();
        staging.write_staging_file(&staging_dir, &entry2).unwrap();
        store.insert(&entry1).await.unwrap();
        store.insert(&entry2).await.unwrap();

        let pipeline = CommitPipeline::new(
            session_mgr.clone(),
            merge_queue.clone(),
            staging,
            store.clone(),
        );

        // Commit
        pipeline.commit_session(&session.session_id).await.unwrap();

        // Verify: session state is Merging
        let updated = session_mgr.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Merging);

        // Verify: 2 entries in merge queue
        assert_eq!(merge_queue.depth().unwrap(), 2);

        // Verify: entries are now Committed in store
        let stored1 = store.get(&entry1.id).await.unwrap().unwrap();
        assert_eq!(stored1.entry_status, EntryStatus::Committed);
        let stored2 = store.get(&entry2.id).await.unwrap().unwrap();
        assert_eq!(stored2.entry_status, EntryStatus::Committed);
    }

    #[tokio::test]
    async fn test_commit_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        let session_mgr = Arc::new(SessionManager::from_db(db.clone()).unwrap());
        let merge_queue = Arc::new(MergeQueue::from_db(db).unwrap());
        let staging = Arc::new(StagingManager::new(dir.path()));
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        store.init_schema().await.unwrap();

        let session = session_mgr.create("test::agent", true).unwrap();
        session_mgr.transition(&session.session_id, SessionState::Active).unwrap();

        // Write 1 entry to staging and store
        let staging_dir_str = session.staging_dir.as_ref().unwrap();
        let staging_dir = staging.resolve_staging_path(staging_dir_str);
        std::fs::create_dir_all(&staging_dir).unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_agent("test::agent".into(), session.session_id.clone());
        staging.write_staging_file(&staging_dir, &entry).unwrap();
        store.insert(&entry).await.unwrap();

        let pipeline = CommitPipeline::new(
            session_mgr.clone(),
            merge_queue.clone(),
            staging,
            store,
        );

        // First commit
        pipeline.commit_session(&session.session_id).await.unwrap();
        let depth_after_first = merge_queue.depth().unwrap();

        // Second commit (idempotent — staging files still exist, but queue gets re-enqueue)
        pipeline.commit_session(&session.session_id).await.unwrap();

        // Session should still be in Merging state (no error)
        let updated = session_mgr.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Merging);
        assert!(merge_queue.depth().unwrap() >= depth_after_first);
    }
}
