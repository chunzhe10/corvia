//! M2.1 Workspace integration test
//! Tests: config → scaffold → multi-namespace storage → cross-namespace search
//!
//! These tests use LiteStore + MockEngine (no external services required).
//!
//! Run with: cargo test --test workspace_e2e_test -- --nocapture

use corvia_common::config::{CorviaConfig, RepoConfig, WorkspaceConfig};
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::lite_store::LiteStore;
use corvia_kernel::traits::QueryableStore;
use tempfile::tempdir;

fn make_workspace_config(repos: Vec<RepoConfig>) -> CorviaConfig {
    let mut config = CorviaConfig::default();
    config.project.name = "test-workspace".into();
    config.project.scope_id = "test".into();
    config.embedding.dimensions = 3;
    config.workspace = Some(WorkspaceConfig {
        repos_dir: "repos".into(),
        repos,
        docs: None,
    });
    config
}

#[tokio::test]
async fn test_workspace_config_roundtrip_and_detection() {
    let dir = tempdir().unwrap();
    let config = make_workspace_config(vec![RepoConfig {
        name: "repo-a".into(),
        url: "https://github.com/org/repo-a".into(),
        local: None,
        namespace: "alpha".into(),
    }]);

    assert!(config.is_workspace());
    let path = dir.path().join("corvia.toml");
    config.save(&path).unwrap();

    let loaded = CorviaConfig::load(&path).unwrap();
    assert!(loaded.is_workspace());
    let ws = loaded.workspace.unwrap();
    assert_eq!(ws.repos.len(), 1);
    assert_eq!(ws.repos[0].namespace, "alpha");
}

#[tokio::test]
async fn test_workspace_multi_namespace_ingest_and_search() {
    let dir = tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    // Simulate ingesting from two repos with different workstreams
    let mut entry_a = KnowledgeEntry::new(
        "Authentication handler for user login".into(),
        "test".into(),
        "abc123".into(),
    )
    .with_embedding(vec![1.0, 0.0, 0.0]);
    entry_a.workstream = "backend".into();

    let mut entry_b = KnowledgeEntry::new(
        "Login form component with validation".into(),
        "test".into(),
        "def456".into(),
    )
    .with_embedding(vec![0.9, 0.1, 0.0]);
    entry_b.workstream = "frontend".into();

    store.insert(&entry_a).await.unwrap();
    store.insert(&entry_b).await.unwrap();

    // Search returns results from both namespaces
    // HNSW quirk: at <10 entries, approximate recall is unreliable — use >= assertions
    let results = store.search(&[1.0, 0.0, 0.0], "test", 10).await.unwrap();
    assert!(results.len() >= 2, "Expected at least 2 results, got {}", results.len());

    // Verify we can distinguish results by workstream
    let workstreams: Vec<&str> = results.iter().map(|r| r.entry.workstream.as_str()).collect();
    assert!(workstreams.contains(&"backend"));
    assert!(workstreams.contains(&"frontend"));
}

#[tokio::test]
async fn test_workspace_backward_compatible_no_workspace_section() {
    // A config without [workspace] still works as single-project
    let config = CorviaConfig::default();
    assert!(!config.is_workspace());
    assert_eq!(config.project.name, "default");
}
