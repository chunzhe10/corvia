//! End-to-end integration tests for M2: Agent Coordination.
//!
//! These tests use LiteStore + MockEngine (no external services required).
//!
//! Run with: cargo test --test agent_e2e_test -- --nocapture

use async_trait::async_trait;
use corvia_common::agent_types::*;
use corvia_common::config::{AgentLifecycleConfig, MergeConfig};
use corvia_kernel::agent_coordinator::AgentCoordinator;
use corvia_kernel::lite_store::LiteStore;
use corvia_kernel::traits::{ChatEngine, InferenceEngine, QueryableStore};
use std::sync::Arc;

struct MockEngine;
#[async_trait]
impl InferenceEngine for MockEngine {
    async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
        Ok(vec![1.0, 0.0, 0.0])
    }
    async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
    }
    fn dimensions(&self) -> usize { 3 }
}

struct MockChatEngine;
#[async_trait]
impl ChatEngine for MockChatEngine {
    async fn chat(&self, messages: &[corvia_common::types::ChatMessage], _model: &str) -> corvia_common::errors::Result<String> {
        Ok(format!("merged: {}", messages.last().map(|m| m.content.as_str()).unwrap_or("")))
    }
}

async fn setup_coordinator(dir: &std::path::Path) -> (AgentCoordinator, Arc<dyn QueryableStore>) {
    let store = Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
    store.init_schema().await.unwrap();
    let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
    let chat_engine = Arc::new(MockChatEngine) as Arc<dyn ChatEngine>;

    let coord = AgentCoordinator::new(
        store.clone(),
        engine,
        dir,
        AgentLifecycleConfig::default(),
        MergeConfig {
            // Set threshold above max cosine similarity (1.0) so all entries auto-merge
            // without attempting LLM merge (no Ollama in tests)
            similarity_threshold: 2.0,
            max_retries: 3,
            ..Default::default()
        },
        chat_engine,
    ).unwrap();

    (coord, store)
}

#[tokio::test]
async fn test_full_agent_lifecycle_e2e() {
    let dir = tempfile::tempdir().unwrap();
    let (coord, store) = setup_coordinator(dir.path()).await;

    // 1. Register agent
    let identity = AgentIdentity::Registered {
        agent_id: "test::indexer".into(),
        api_key: None,
    };
    let agent = coord.register_agent(
        &identity,
        "Test Indexer",
        AgentPermission::ReadWrite { scopes: vec!["project-a".into()] },
    ).unwrap();
    assert_eq!(agent.agent_id, "test::indexer");

    // 2. Create session — returns SessionRecord directly
    let session = coord.create_session(&agent.agent_id, true).unwrap();
    let session_id = session.session_id.clone();

    // 3. Write 3 knowledge entries
    let e1 = coord.write_entry(&session_id, "Authentication uses JWT tokens", "project-a", "v1").await.unwrap();
    let e2 = coord.write_entry(&session_id, "Database uses PostgreSQL 16", "project-a", "v1").await.unwrap();
    let e3 = coord.write_entry(&session_id, "API follows REST conventions", "project-a", "v1").await.unwrap();

    assert_eq!(e1.entry_status, EntryStatus::Pending);
    assert_eq!(e2.entry_status, EntryStatus::Pending);
    assert_eq!(e3.entry_status, EntryStatus::Pending);

    // 4. Search — entries visible (all embeddings are identical, HNSW may not return all)
    let embedding = vec![1.0, 0.0, 0.0];
    let results = store.search(&embedding, "project-a", 10).await.unwrap();
    assert!(!results.is_empty(), "Expected some results from store");

    // 5. Commit session
    coord.commit_session(&session_id).await.unwrap();

    // Verify session is in Merging state
    let session = coord.sessions.get(&session_id).unwrap().unwrap();
    assert!(
        matches!(session.state, SessionState::Merging),
        "Expected Merging, got {:?}", session.state
    );

    // 6. Process merge queue (no conflicts expected — MockEngine)
    coord.process_merge_queue().await.unwrap();

    // 7. Verify entries have Merged status
    let results = store.search(&embedding, "project-a", 10).await.unwrap();
    let merged_count = results.iter()
        .filter(|r| r.entry.entry_status == EntryStatus::Merged)
        .count();
    assert!(merged_count >= 1, "Expected at least 1 merged entry, got {merged_count}");

    // 8. Session should be Closed after all entries merged
    let session = coord.sessions.get(&session_id).unwrap().unwrap();
    assert!(
        matches!(session.state, SessionState::Closed),
        "Expected Closed, got {:?}", session.state
    );
}

#[tokio::test]
async fn test_two_agents_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let (coord, store) = setup_coordinator(dir.path()).await;

    // Register two agents
    let id_a = AgentIdentity::Registered { agent_id: "test::agent-a".into(), api_key: None };
    let id_b = AgentIdentity::Registered { agent_id: "test::agent-b".into(), api_key: None };

    coord.register_agent(&id_a, "Agent A", AgentPermission::ReadWrite { scopes: vec!["shared".into()] }).unwrap();
    coord.register_agent(&id_b, "Agent B", AgentPermission::ReadWrite { scopes: vec!["shared".into()] }).unwrap();

    // Both create sessions
    let session_a = coord.create_session("test::agent-a", true).unwrap();
    let session_b = coord.create_session("test::agent-b", true).unwrap();
    let sess_a = &session_a.session_id;
    let sess_b = &session_b.session_id;

    // Both write entries to same scope
    coord.write_entry(sess_a, "Agent A: auth module docs", "shared", "v1").await.unwrap();
    coord.write_entry(sess_a, "Agent A: config system docs", "shared", "v1").await.unwrap();
    coord.write_entry(sess_b, "Agent B: testing guide", "shared", "v1").await.unwrap();
    coord.write_entry(sess_b, "Agent B: deployment docs", "shared", "v1").await.unwrap();

    // Both commit
    coord.commit_session(sess_a).await.unwrap();
    coord.commit_session(sess_b).await.unwrap();

    // Process merge queue
    coord.process_merge_queue().await.unwrap();

    // Verify entries merged (with HNSW approximation, may not return all)
    let embedding = vec![1.0, 0.0, 0.0];
    let results = store.search(&embedding, "shared", 20).await.unwrap();
    let merged = results.iter().filter(|r| r.entry.entry_status == EntryStatus::Merged).count();
    assert!(merged >= 1, "Expected at least 1 merged entry from both agents, got {merged}");

    // Both sessions should be Closed
    let sa = coord.sessions.get(sess_a).unwrap().unwrap();
    let sb = coord.sessions.get(sess_b).unwrap().unwrap();
    assert!(matches!(sa.state, SessionState::Closed));
    assert!(matches!(sb.state, SessionState::Closed));
}

#[tokio::test]
async fn test_crash_recovery_resume() {
    let dir = tempfile::tempdir().unwrap();
    let (coord, _store) = setup_coordinator(dir.path()).await;

    // Register agent, create session, write entries
    let identity = AgentIdentity::Registered { agent_id: "test::crasher".into(), api_key: None };
    coord.register_agent(&identity, "Crasher", AgentPermission::ReadWrite { scopes: vec!["recovery".into()] }).unwrap();
    let session = coord.create_session("test::crasher", true).unwrap();
    let session_id = session.session_id.clone();

    coord.write_entry(&session_id, "Important data before crash", "recovery", "v1").await.unwrap();
    coord.write_entry(&session_id, "More data before crash", "recovery", "v1").await.unwrap();

    // Simulate crash: mark session Stale then Orphaned
    coord.sessions.transition(&session_id, SessionState::Stale).unwrap();
    coord.sessions.transition(&session_id, SessionState::Orphaned).unwrap();

    // Connect again — should see recoverable session
    let reconnect = coord.connect("test::crasher").unwrap();
    assert_eq!(reconnect.recoverable_sessions.len(), 1);
    assert_eq!(reconnect.recoverable_sessions[0].session_id, session_id);

    // Resume the orphaned session
    coord.recover(&session_id, RecoveryAction::Resume).await.unwrap();

    // Session should be Active again
    let session = coord.sessions.get(&session_id).unwrap().unwrap();
    assert!(matches!(session.state, SessionState::Active));

    // Can continue writing
    coord.write_entry(&session_id, "New data after recovery", "recovery", "v1").await.unwrap();

    // Commit and merge
    coord.commit_session(&session_id).await.unwrap();
    coord.process_merge_queue().await.unwrap();

    // Session closed
    let session = coord.sessions.get(&session_id).unwrap().unwrap();
    assert!(matches!(session.state, SessionState::Closed));
}

#[tokio::test]
async fn test_crash_recovery_rollback() {
    let dir = tempfile::tempdir().unwrap();
    let (coord, store) = setup_coordinator(dir.path()).await;

    // Register and create session
    let identity = AgentIdentity::Registered { agent_id: "test::roller".into(), api_key: None };
    coord.register_agent(&identity, "Roller", AgentPermission::ReadWrite { scopes: vec!["rollback".into()] }).unwrap();
    let session = coord.create_session("test::roller", true).unwrap();
    let session_id = session.session_id.clone();

    coord.write_entry(&session_id, "Data to be rolled back", "rollback", "v1").await.unwrap();

    // Mark as orphaned
    coord.sessions.transition(&session_id, SessionState::Stale).unwrap();
    coord.sessions.transition(&session_id, SessionState::Orphaned).unwrap();

    // Recover with rollback
    coord.recover(&session_id, RecoveryAction::Rollback).await.unwrap();

    // Session should be Closed
    let session = coord.sessions.get(&session_id).unwrap().unwrap();
    assert!(matches!(session.state, SessionState::Closed));

    // The rolled-back entries remain in the store with Pending status
    // (LiteStore doesn't support physical deletion of individual entries yet)
    // But no new entries should appear as Merged
    let embedding = vec![1.0, 0.0, 0.0];
    let results = store.search(&embedding, "rollback", 10).await.unwrap();
    let merged = results.iter().filter(|r| r.entry.entry_status == EntryStatus::Merged).count();
    assert_eq!(merged, 0, "No entries should be merged after rollback");
}
