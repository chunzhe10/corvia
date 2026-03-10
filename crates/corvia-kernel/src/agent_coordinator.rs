use corvia_common::agent_types::*;
use corvia_common::config::{AgentLifecycleConfig, MergeConfig};
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{KnowledgeEntry, SearchResult};
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

use crate::agent_registry::AgentRegistry;
use crate::agent_writer::AgentWriter;
use crate::commit_pipeline::CommitPipeline;
use crate::context_builder::ContextBuilder;
use crate::merge_queue::MergeQueue;
use crate::merge_worker::MergeWorker;
use crate::session_manager::SessionManager;
use crate::staging::StagingManager;
use crate::traits::{GenerationEngine, InferenceEngine, QueryableStore};

/// Response from `connect()` — shows the agent's active and recoverable sessions.
#[derive(Debug, Clone)]
pub struct ConnectResponse {
    pub agent_id: String,
    pub recoverable_sessions: Vec<SessionRecord>,
    pub active_sessions: Vec<SessionRecord>,
}

/// Report from garbage collection sweep.
#[derive(Debug, Clone, Default)]
pub struct GcReport {
    pub orphans_rolled_back: usize,
    pub closed_sessions_cleaned: usize,
    pub inactive_agents_cleaned: usize,
}

/// The top-level orchestrator that ties all agent coordination components together.
///
/// Constructed from a shared coordination Redb database, a store, and an engine.
/// Provides the full agent lifecycle: register → connect → create session → write → commit → merge → GC.
pub struct AgentCoordinator {
    pub registry: Arc<AgentRegistry>,
    pub sessions: Arc<SessionManager>,
    pub writer: Arc<AgentWriter>,
    pub commit_pipeline: Arc<CommitPipeline>,
    pub merge_worker: Arc<MergeWorker>,
    pub context: Arc<ContextBuilder>,
    pub merge_queue: Arc<MergeQueue>,
    pub staging: Arc<StagingManager>,
    pub config: AgentLifecycleConfig,
}

impl AgentCoordinator {
    /// Construct the coordinator and all sub-components from shared resources.
    pub fn new(
        store: Arc<dyn QueryableStore>,
        engine: Arc<dyn InferenceEngine>,
        data_dir: &Path,
        lifecycle_config: AgentLifecycleConfig,
        merge_config: MergeConfig,
        gen_engine: Arc<dyn GenerationEngine>,
    ) -> Result<Self> {
        // Open or create the shared coordination database
        let registry = AgentRegistry::open(data_dir)?;
        let db = registry.db().clone();

        let sessions = Arc::new(SessionManager::from_db(db.clone())?);
        let merge_queue = Arc::new(MergeQueue::from_db(db)?);
        let staging = Arc::new(StagingManager::new(data_dir));

        let writer = Arc::new(AgentWriter::new(
            store.clone(),
            engine.clone(),
            StagingManager::new(data_dir),
        ));

        let commit_pipeline = Arc::new(CommitPipeline::new(
            sessions.clone(),
            merge_queue.clone(),
            staging.clone(),
            store.clone(),
        ));

        let merge_worker = Arc::new(MergeWorker::new(
            store.clone(),
            engine.clone(),
            merge_queue.clone(),
            staging.clone(),
            sessions.clone(),
            merge_config,
            gen_engine,
        ));

        let context = Arc::new(ContextBuilder::new(store, engine));
        let registry = Arc::new(registry);

        Ok(Self {
            registry,
            sessions,
            writer,
            commit_pipeline,
            merge_worker,
            context,
            merge_queue,
            staging,
            config: lifecycle_config,
        })
    }

    /// Register an agent from an identity.
    pub fn register_agent(
        &self,
        identity: &AgentIdentity,
        display_name: &str,
        permissions: AgentPermission,
    ) -> Result<AgentRecord> {
        let identity_type = match identity {
            AgentIdentity::Registered { .. } => IdentityType::Registered,
            AgentIdentity::McpClient { .. } => IdentityType::McpClient,
            AgentIdentity::Anonymous => {
                return Err(CorviaError::Agent("Cannot register anonymous agents".into()));
            }
        };
        let agent_id = identity.effective_agent_id();
        self.registry.register(&agent_id, display_name, identity_type, permissions)
    }

    /// Connect an agent — returns active sessions and any recoverable (orphaned) sessions.
    pub fn connect(&self, agent_id: &str) -> Result<ConnectResponse> {
        // Touch last_seen
        let _ = self.registry.touch(agent_id);

        let all_sessions = self.sessions.list_by_agent(agent_id)?;

        let active_sessions: Vec<SessionRecord> = all_sessions.iter()
            .filter(|s| matches!(s.state, SessionState::Active | SessionState::Created))
            .cloned()
            .collect();

        let recoverable_sessions: Vec<SessionRecord> = all_sessions.iter()
            .filter(|s| matches!(s.state, SessionState::Orphaned | SessionState::Stale))
            .cloned()
            .collect();

        Ok(ConnectResponse {
            agent_id: agent_id.into(),
            active_sessions,
            recoverable_sessions,
        })
    }

    /// Create a new session for an agent.
    pub fn create_session(&self, agent_id: &str, with_staging: bool) -> Result<SessionRecord> {
        let session = self.sessions.create(agent_id, with_staging)?;
        self.sessions.transition(&session.session_id, SessionState::Active)?;

        // Create staging directory if needed
        if let Some(ref staging_dir_str) = session.staging_dir {
            let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
            std::fs::create_dir_all(&staging_dir)
                .map_err(|e| CorviaError::Agent(format!("Failed to create staging dir: {e}")))?;
        }

        // Create git branch if needed
        if let Some(ref branch) = session.git_branch {
            let _ = self.staging.create_branch(branch);
        }

        Ok(session)
    }

    /// Write a knowledge entry within a session.
    /// Checks RBAC: the session's agent must have write permission for the scope.
    pub async fn write_entry(
        &self,
        session_id: &str,
        content: &str,
        scope_id: &str,
        source_version: &str,
    ) -> Result<KnowledgeEntry> {
        let session = self.sessions.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        // RBAC check: verify agent has write permission for this scope
        if let Some(agent) = self.registry.get(&session.agent_id)? {
            if !agent.permissions.can_write_scope(scope_id) {
                return Err(CorviaError::Agent(format!(
                    "Agent '{}' does not have write permission for scope '{scope_id}'",
                    session.agent_id
                )));
            }
        }

        let staging_dir = session.staging_dir.as_ref().map(|s| {
            self.staging.resolve_staging_path(s)
        });

        let entry = self.writer.write(
            content,
            scope_id,
            source_version,
            &session.agent_id,
            session_id,
            staging_dir.as_deref(),
        ).await?;

        self.sessions.increment_written(session_id)?;
        Ok(entry)
    }

    /// Commit a session — transitions through the commit pipeline.
    pub async fn commit_session(&self, session_id: &str) -> Result<()> {
        self.commit_pipeline.commit_session(session_id).await
    }

    /// Rollback a session — cleanup staging and close.
    pub fn rollback_session(&self, session_id: &str) -> Result<()> {
        let session = self.sessions.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        // Cleanup staging directory
        if let Some(ref staging_dir_str) = session.staging_dir {
            let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
            self.staging.cleanup_staging_dir(&staging_dir)?;
        }

        // Delete git branch
        if let Some(ref branch) = session.git_branch {
            let _ = self.staging.delete_branch(branch);
        }

        // Close session
        self.sessions.transition(session_id, SessionState::Closed)?;
        info!(session_id, "session_rolled_back");
        Ok(())
    }

    /// Heartbeat for a session.
    pub fn heartbeat(&self, session_id: &str) -> Result<()> {
        self.sessions.heartbeat(session_id)
    }

    /// Recover an orphaned session.
    pub async fn recover(&self, session_id: &str, action: RecoveryAction) -> Result<()> {
        let session = self.sessions.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        match action {
            RecoveryAction::Resume => {
                // Transition back to Active
                self.sessions.transition(session_id, SessionState::Active)?;
                info!(session_id, "session_recovered_resume");
            }
            RecoveryAction::Commit => {
                // Transition to Committing and run commit pipeline
                self.sessions.transition(session_id, SessionState::Committing)?;
                self.commit_pipeline.commit_session(session_id).await?;
                info!(session_id, "session_recovered_commit");
            }
            RecoveryAction::Rollback => {
                // Cleanup staging and close
                if let Some(ref staging_dir_str) = session.staging_dir {
                    let staging_dir = self.staging.resolve_staging_path(staging_dir_str);
                    self.staging.cleanup_staging_dir(&staging_dir)?;
                }
                if let Some(ref branch) = session.git_branch {
                    let _ = self.staging.delete_branch(branch);
                }
                self.sessions.transition(session_id, SessionState::Closed)?;
                info!(session_id, "session_recovered_rollback");
            }
        }
        Ok(())
    }

    /// Search with visibility and RBAC filtering.
    pub async fn search(
        &self,
        query: &str,
        scope_id: &str,
        limit: usize,
        visibility: &VisibilityMode,
        agent_id: Option<&str>,
        permissions: Option<&AgentPermission>,
    ) -> Result<Vec<SearchResult>> {
        self.context.search(query, scope_id, limit, visibility, agent_id, permissions).await
    }

    /// Run garbage collection: Active past timeout → Stale, Stale past grace → Orphaned → rollback.
    pub async fn gc(&self) -> Result<GcReport> {
        let mut report = GcReport::default();

        // Step 1: Find Active sessions past heartbeat timeout → mark Stale
        let stale_timeout = std::time::Duration::from_secs(self.config.stale_timeout_secs);
        let stale = self.sessions.find_stale(stale_timeout)?;
        for session in &stale {
            if let Err(e) = self.sessions.transition(&session.session_id, SessionState::Stale) {
                warn!(session_id = %session.session_id, error = %e, "gc_stale_transition_failed");
            }
        }

        // Step 2: Find Stale sessions past orphan grace → mark Orphaned
        let orphan_grace = std::time::Duration::from_secs(self.config.gc_orphan_after_secs);
        let stale_past_grace = self.sessions.find_orphaned(orphan_grace)?;
        for session in &stale_past_grace {
            if let Err(e) = self.sessions.transition(&session.session_id, SessionState::Orphaned) {
                warn!(session_id = %session.session_id, error = %e, "gc_orphan_transition_failed");
                continue;
            }
            // Step 3: Rollback orphaned sessions
            match self.recover(&session.session_id, RecoveryAction::Rollback).await {
                Ok(()) => report.orphans_rolled_back += 1,
                Err(e) => warn!(
                    session_id = %session.session_id,
                    error = %e,
                    "gc_orphan_rollback_failed"
                ),
            }
        }

        info!(?report, "gc_sweep_complete");
        Ok(report)
    }

    /// Process one batch of merge queue entries, then close completed sessions.
    /// Can be called manually or from a background polling loop (see `MergeWorker::run()`).
    pub async fn process_merge_queue(&self) -> Result<()> {
        let entries = self.merge_queue.list(10)?;
        // Collect unique session IDs for post-processing
        let mut session_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in &entries {
            session_ids.insert(entry.session_id.clone());
            self.merge_worker.process_one(entry).await?;
        }

        // Check if any sessions are now complete (all entries merged)
        for session_id in &session_ids {
            if let Ok(Some(session)) = self.sessions.get(session_id) {
                if session.state == SessionState::Merging
                    && session.entries_merged >= session.entries_written
                    && session.entries_written > 0
                {
                    self.sessions.transition(session_id, SessionState::Closed).ok();
                    info!(session_id, "session_closed_all_merged");
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;

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

    struct MockGenerationEngine;
    #[async_trait::async_trait]
    impl crate::traits::GenerationEngine for MockGenerationEngine {
        fn name(&self) -> &str { "mock" }
        async fn generate(&self, _system_prompt: &str, user_message: &str) -> corvia_common::errors::Result<crate::traits::GenerationResult> {
            Ok(crate::traits::GenerationResult {
                text: format!("merged: {user_message}"),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize { 4096 }
    }

    async fn setup_coordinator(dir: &Path) -> AgentCoordinator {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();
        let gen_engine = Arc::new(MockGenerationEngine) as Arc<dyn GenerationEngine>;

        AgentCoordinator::new(
            store,
            engine,
            dir,
            AgentLifecycleConfig::default(),
            MergeConfig::default(),
            gen_engine,
        ).unwrap()
    }

    #[tokio::test]
    async fn test_full_agent_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        // 1. Register agent
        let identity = AgentIdentity::Registered {
            agent_id: "test::indexer".into(),
            api_key: None,
        };
        let record = coord.register_agent(
            &identity,
            "Code Indexer",
            AgentPermission::ReadWrite { scopes: vec!["test".into()] },
        ).unwrap();
        assert_eq!(record.agent_id, "test::indexer");

        // 2. Create session
        let session = coord.create_session("test::indexer", true).unwrap();
        assert!(session.staging_dir.is_some());

        // 3. Write 2 entries
        let e1 = coord.write_entry(&session.session_id, "knowledge one", "test", "v1").await.unwrap();
        let e2 = coord.write_entry(&session.session_id, "knowledge two", "test", "v1").await.unwrap();
        assert_eq!(e1.entry_status, EntryStatus::Pending);
        assert_eq!(e2.entry_status, EntryStatus::Pending);

        // 4. Commit session
        coord.commit_session(&session.session_id).await.unwrap();

        // 5. Verify entries in merge queue
        assert_eq!(coord.merge_queue.depth().unwrap(), 2);
    }

    #[tokio::test]
    async fn test_connect_with_recoverable_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        // Register agent and create session
        let identity = AgentIdentity::Registered {
            agent_id: "test::agent".into(),
            api_key: None,
        };
        coord.register_agent(&identity, "Agent", AgentPermission::ReadOnly).unwrap();
        let session = coord.create_session("test::agent", true).unwrap();

        // Simulate crash: Active → Stale → Orphaned
        coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();
        coord.sessions.transition(&session.session_id, SessionState::Orphaned).unwrap();

        // Connect again
        let response = coord.connect("test::agent").unwrap();
        assert_eq!(response.recoverable_sessions.len(), 1);
        assert_eq!(response.recoverable_sessions[0].session_id, session.session_id);
    }

    #[tokio::test]
    async fn test_recover_resume() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        let identity = AgentIdentity::Registered {
            agent_id: "test::agent".into(),
            api_key: None,
        };
        coord.register_agent(&identity, "Agent", AgentPermission::ReadOnly).unwrap();
        let session = coord.create_session("test::agent", true).unwrap();

        // Simulate crash
        coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();
        coord.sessions.transition(&session.session_id, SessionState::Orphaned).unwrap();

        // Recover: resume
        coord.recover(&session.session_id, RecoveryAction::Resume).await.unwrap();

        let updated = coord.sessions.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Active);
    }

    #[tokio::test]
    async fn test_recover_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        let identity = AgentIdentity::Registered {
            agent_id: "test::agent".into(),
            api_key: None,
        };
        coord.register_agent(&identity, "Agent", AgentPermission::ReadOnly).unwrap();
        let session = coord.create_session("test::agent", true).unwrap();

        // Verify staging dir was created
        let staging_dir_str = session.staging_dir.as_ref().unwrap();
        let staging_dir = coord.staging.resolve_staging_path(staging_dir_str);
        assert!(staging_dir.exists());

        // Simulate crash
        coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();
        coord.sessions.transition(&session.session_id, SessionState::Orphaned).unwrap();

        // Recover: rollback
        coord.recover(&session.session_id, RecoveryAction::Rollback).await.unwrap();

        let updated = coord.sessions.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Closed);
        assert!(!staging_dir.exists());
    }

    #[tokio::test]
    async fn test_gc_cleans_old_orphans() {
        let dir = tempfile::tempdir().unwrap();
        // Use a config with 0-second GC orphan timeout for testing
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();

        let mut lifecycle_config = AgentLifecycleConfig::default();
        lifecycle_config.gc_orphan_after_secs = 0; // Instant orphan cleanup

        let gen_engine = Arc::new(MockGenerationEngine) as Arc<dyn GenerationEngine>;
        let coord = AgentCoordinator::new(
            store, engine, dir.path(),
            lifecycle_config,
            MergeConfig::default(),
            gen_engine,
        ).unwrap();

        let identity = AgentIdentity::Registered {
            agent_id: "test::agent".into(),
            api_key: None,
        };
        coord.register_agent(&identity, "Agent", AgentPermission::ReadOnly).unwrap();
        let session = coord.create_session("test::agent", true).unwrap();

        // Mark as Stale (bypassing Active → Stale transition since it's already Active)
        coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();

        // Run GC — with 0-second grace, Stale sessions should be found by find_orphaned
        let report = coord.gc().await.unwrap();
        assert_eq!(report.orphans_rolled_back, 1);

        let updated = coord.sessions.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Closed);
    }

    #[tokio::test]
    async fn test_write_rbac_rejects_wrong_scope() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        // Register agent with write access to "scope-a" only
        let identity = AgentIdentity::Registered {
            agent_id: "test::writer".into(),
            api_key: None,
        };
        coord.register_agent(
            &identity,
            "Writer",
            AgentPermission::ReadWrite { scopes: vec!["scope-a".into()] },
        ).unwrap();
        let session = coord.create_session("test::writer", true).unwrap();

        // Writing to scope-a should succeed
        let result = coord.write_entry(&session.session_id, "ok", "scope-a", "v1").await;
        assert!(result.is_ok());

        // Writing to scope-b should be rejected
        let result = coord.write_entry(&session.session_id, "denied", "scope-b", "v1").await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("write permission"), "Expected RBAC error, got: {err_msg}");
    }

    #[tokio::test]
    async fn test_write_rbac_readonly_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let coord = setup_coordinator(dir.path()).await;

        let identity = AgentIdentity::Registered {
            agent_id: "test::reader".into(),
            api_key: None,
        };
        coord.register_agent(&identity, "Reader", AgentPermission::ReadOnly).unwrap();
        let session = coord.create_session("test::reader", false).unwrap();

        // ReadOnly agents cannot write to any scope
        let result = coord.write_entry(&session.session_id, "denied", "scope-a", "v1").await;
        assert!(result.is_err());
    }
}
