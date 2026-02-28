//! M3 Integration Tests: Temporal + Graph + Reasoning end-to-end.
//!
//! Tests the full M3 pipeline using LiteStore (no external services required).
//! Covers temporal history/evolution, graph edges/traversal, reasoner findings,
//! and a self-dogfooding test that exercises ingest -> store -> graph -> reason.
//!
//! Run with: cargo test --test m3_e2e_test -- --nocapture

use chrono::{Duration, Utc};
use corvia_common::types::{EdgeDirection, KnowledgeEntry};
use corvia_kernel::knowledge_files;
use corvia_kernel::lite_store::LiteStore;
use corvia_kernel::reasoner::{CheckType, Reasoner};
use corvia_kernel::traits::{GraphStore, QueryableStore, TemporalStore};
use tempfile::tempdir;
use uuid::Uuid;

/// Helper: create a KnowledgeEntry with a 3-dim embedding.
fn entry(content: &str, scope: &str, version: &str) -> KnowledgeEntry {
    KnowledgeEntry::new(content.into(), scope.into(), version.into())
        .with_embedding(vec![1.0, 0.0, 0.0])
}

// ---------------------------------------------------------------------------
// Test 1: Temporal history and evolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_temporal_history_and_evolution() {
    let dir = tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    let before_insert = Utc::now();

    // 1. Insert original entry
    let entry_a = entry("Authentication uses JWT tokens", "project", "v1");
    let entry_a_id = entry_a.id;
    store.insert(&entry_a).await.unwrap();

    // 2. Insert superseding entry
    let entry_b = entry("Authentication uses OAuth2 + JWT refresh tokens", "project", "v2");
    let entry_b_id = entry_b.id;
    store.insert(&entry_b).await.unwrap();

    // 3. Mark entry_a as superseded by entry_b (uses the LiteStore::supersede method)
    store.supersede(&entry_a_id, &entry_b_id).await.unwrap();

    // 4. Insert an unrelated entry
    let entry_c = entry("Database uses PostgreSQL 16", "project", "v1");
    let entry_c_id = entry_c.id;
    store.insert(&entry_c).await.unwrap();

    let after_insert = Utc::now();

    // --- Verify history ---
    // history(entry_b) should return a chain of 2: entry_b (newest) -> entry_a (oldest)
    let history = store.history(&entry_b_id).await.unwrap();
    assert_eq!(
        history.len(),
        2,
        "Expected history chain of 2, got {}",
        history.len()
    );
    assert_eq!(history[0].id, entry_b_id, "Newest entry should be first");
    assert_eq!(history[1].id, entry_a_id, "Oldest entry should be second");

    // history(entry_c) should return just itself (no chain)
    let history_c = store.history(&entry_c_id).await.unwrap();
    assert_eq!(
        history_c.len(),
        1,
        "Unrelated entry should have history of 1"
    );

    // --- Verify evolution ---
    // All entries were inserted in the time range [before_insert, after_insert]
    let evolution = store
        .evolution("project", before_insert, after_insert)
        .await
        .unwrap();
    assert!(
        evolution.len() >= 3,
        "Expected at least 3 entries in evolution, got {}",
        evolution.len()
    );

    // Query a range before any insertion — should be empty
    let old_evolution = store
        .evolution(
            "project",
            before_insert - Duration::hours(2),
            before_insert - Duration::hours(1),
        )
        .await
        .unwrap();
    assert_eq!(
        old_evolution.len(),
        0,
        "Evolution before insertions should be empty"
    );

    // --- Verify as_of ---
    // Query at a point after all insertions — should see non-superseded entries
    let as_of_results = store
        .as_of("project", after_insert, 10)
        .await
        .unwrap();
    // entry_a has valid_to set (superseded), so it may be excluded depending
    // on the timestamp precision. entry_b and entry_c should be present.
    let result_ids: Vec<Uuid> = as_of_results.iter().map(|e| e.id).collect();
    assert!(
        result_ids.contains(&entry_b_id),
        "as_of should include the current (superseding) entry"
    );
    assert!(
        result_ids.contains(&entry_c_id),
        "as_of should include the unrelated entry"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Graph edges and traversal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_graph_edges_and_traversal() {
    let dir = tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    // Insert 3 entries with distinct embeddings for reliable retrieval
    let e_a = KnowledgeEntry::new("Module A: auth handler".into(), "graph".into(), "v1".into())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    let e_b = KnowledgeEntry::new("Module B: database layer".into(), "graph".into(), "v1".into())
        .with_embedding(vec![0.0, 1.0, 0.0]);
    let e_c = KnowledgeEntry::new("Module C: API router".into(), "graph".into(), "v1".into())
        .with_embedding(vec![0.0, 0.0, 1.0]);

    let id_a = e_a.id;
    let id_b = e_b.id;
    let id_c = e_c.id;

    store.insert(&e_a).await.unwrap();
    store.insert(&e_b).await.unwrap();
    store.insert(&e_c).await.unwrap();

    // Create edges: A -imports-> B, B -calls-> C
    store
        .relate(&id_a, "imports", &id_b, None)
        .await
        .unwrap();
    store
        .relate(&id_b, "calls", &id_c, None)
        .await
        .unwrap();

    // --- Verify edges(A, Outgoing) returns 1 edge to B ---
    let a_out = store.edges(&id_a, EdgeDirection::Outgoing).await.unwrap();
    assert_eq!(a_out.len(), 1, "A should have 1 outgoing edge");
    assert_eq!(a_out[0].to, id_b);
    assert_eq!(a_out[0].relation, "imports");

    // --- Verify edges(B, Both) returns 2 edges (incoming from A, outgoing to C) ---
    let b_both = store.edges(&id_b, EdgeDirection::Both).await.unwrap();
    assert_eq!(
        b_both.len(),
        2,
        "B should have 2 edges (1 incoming, 1 outgoing), got {}",
        b_both.len()
    );

    // --- Verify traverse(A, None, Outgoing, 3) reaches B and C ---
    let traversed = store
        .traverse(&id_a, None, EdgeDirection::Outgoing, 3)
        .await
        .unwrap();
    assert_eq!(
        traversed.len(),
        2,
        "Traversal from A should reach B and C, got {}",
        traversed.len()
    );
    let traversed_ids: Vec<Uuid> = traversed.iter().map(|e| e.id).collect();
    assert!(traversed_ids.contains(&id_b), "Traversal should include B");
    assert!(traversed_ids.contains(&id_c), "Traversal should include C");

    // --- Verify shortest_path(A, C) returns a path of 3 entries: A -> B -> C ---
    let path = store.shortest_path(&id_a, &id_c).await.unwrap();
    assert!(path.is_some(), "A path from A to C should exist");
    let path = path.unwrap();
    assert_eq!(path.len(), 3, "Shortest path A->B->C should have 3 entries");
    assert_eq!(path[0].id, id_a);
    assert_eq!(path[1].id, id_b);
    assert_eq!(path[2].id, id_c);

    // No reverse path (directed graph)
    let reverse = store.shortest_path(&id_c, &id_a).await.unwrap();
    assert!(reverse.is_none(), "No reverse path should exist in a directed graph");
}

// ---------------------------------------------------------------------------
// Test 3: Reasoner finds issues
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_reasoner_finds_issues() {
    let dir = tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    // 1. Create an orphaned node (no edges)
    let orphan = entry("Lonely module with no connections", "reason", "v1");
    store.insert(&orphan).await.unwrap();

    // 2. Create a dangling import (edge to nonexistent target)
    let importer = entry("Module that imports a ghost", "reason", "v1");
    let ghost_id = Uuid::now_v7();
    store.insert(&importer).await.unwrap();
    store
        .relate(&importer.id, "imports", &ghost_id, None)
        .await
        .unwrap();

    // 3. Create a stale entry (valid_to set but no superseded_by)
    let mut stale = entry("Outdated documentation", "reason", "v1");
    stale.valid_to = Some(Utc::now());
    // No superseded_by set — this is stale
    store.insert(&stale).await.unwrap();

    // 4. Create connected entries (should NOT be flagged as orphans)
    let connected_a = entry("Connected module A", "reason", "v1");
    let connected_b = entry("Connected module B", "reason", "v1");
    store.insert(&connected_a).await.unwrap();
    store.insert(&connected_b).await.unwrap();
    store
        .relate(&connected_a.id, "imports", &connected_b.id, None)
        .await
        .unwrap();

    // Collect all entries for the reasoner
    let entries = vec![
        orphan.clone(),
        importer.clone(),
        stale.clone(),
        connected_a.clone(),
        connected_b.clone(),
    ];

    // Run all checks
    let reasoner = Reasoner::new(&store, &store);
    let findings = reasoner.run_all(&entries, "reason").await.unwrap();

    // Verify OrphanedNode findings
    let orphan_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.check_type == CheckType::OrphanedNode)
        .collect();
    // The orphan entry has no edges. The stale entry also has no edges.
    assert!(
        orphan_findings.len() >= 1,
        "Expected at least 1 orphaned node finding, got {}",
        orphan_findings.len()
    );
    let orphan_ids: Vec<Uuid> = orphan_findings
        .iter()
        .flat_map(|f| f.target_ids.clone())
        .collect();
    assert!(
        orphan_ids.contains(&orphan.id),
        "The orphan entry should be flagged"
    );

    // Verify DanglingImport findings
    let dangling_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.check_type == CheckType::DanglingImport)
        .collect();
    assert_eq!(
        dangling_findings.len(),
        1,
        "Expected 1 dangling import finding, got {}",
        dangling_findings.len()
    );
    assert!(
        dangling_findings[0].target_ids.contains(&importer.id),
        "Dangling finding should reference the importer entry"
    );
    assert!(
        dangling_findings[0].target_ids.contains(&ghost_id),
        "Dangling finding should reference the ghost target"
    );

    // Verify StaleEntry findings
    let stale_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.check_type == CheckType::StaleEntry)
        .collect();
    assert_eq!(
        stale_findings.len(),
        1,
        "Expected 1 stale entry finding, got {}",
        stale_findings.len()
    );
    assert_eq!(stale_findings[0].target_ids, vec![stale.id]);

    // Also verify run_check for a single type
    let single = reasoner
        .run_check(&entries, "reason", CheckType::StaleEntry)
        .await
        .unwrap();
    assert_eq!(
        single.len(),
        1,
        "run_check(StaleEntry) should find exactly 1"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Ingest + Store + Graph + Reason (self-dogfood test)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_ingest_and_reason() {
    use corvia_adapter_git::git::GitAdapter;

    // 1. Create a small temp "repo" with 2 Rust files
    let repo_dir = tempdir().unwrap();
    let src_dir = repo_dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    std::fs::write(
        src_dir.join("lib.rs"),
        r#"
use std::collections::HashMap;

pub trait Store {
    fn get(&self, key: &str) -> Option<String>;
}

pub struct MemStore {
    data: HashMap<String, String>,
}

impl Store for MemStore {
    fn get(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
}
"#,
    )
    .unwrap();

    std::fs::write(
        src_dir.join("main.rs"),
        r#"
use crate::lib::MemStore;

fn main() {
    let store = MemStore { data: Default::default() };
    println!("{:?}", store.get("hello"));
}
"#,
    )
    .unwrap();

    // 2. Use GitAdapter to ingest with relations
    let adapter = GitAdapter::new();
    let result = adapter
        .ingest_with_relations(repo_dir.path().to_str().unwrap())
        .await
        .unwrap();

    assert!(
        !result.entries.is_empty(),
        "Ingestion should produce entries from the Rust files"
    );

    // 3. Set up LiteStore and insert entries with embeddings
    let store_dir = tempdir().unwrap();
    let store = LiteStore::open(store_dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    let mut stored_entries = Vec::new();
    for (i, mut e) in result.entries.into_iter().enumerate() {
        // Assign deterministic embeddings (vary by index for diversity)
        let x = if i % 3 == 0 { 1.0 } else { 0.0 };
        let y = if i % 3 == 1 { 1.0 } else { 0.0 };
        let z = if i % 3 == 2 { 1.0 } else { 0.0 };
        e = e.with_embedding(vec![x, y, z]);
        store.insert(&e).await.unwrap();
        stored_entries.push(e);
    }

    // 4. Write entries to knowledge files (already done by store.insert, but verify)
    for e in &stored_entries {
        let file_path = store_dir
            .path()
            .join("knowledge")
            .join(&e.scope_id)
            .join(format!("{}.json", e.id));
        assert!(
            file_path.exists(),
            "Knowledge file should exist for entry {}",
            e.id
        );
    }

    // 5. Wire relations into graph
    let mut edge_count = 0;
    for rel in &result.relations {
        if rel.from_chunk_index < stored_entries.len() {
            let from_id = stored_entries[rel.from_chunk_index].id;
            // For "imports" relations, create edges to a deterministic target.
            // In a real pipeline, the target would be resolved by file name.
            // For this test, just create a graph edge to the first entry
            // to exercise the graph pipeline.
            let to_id = stored_entries[0].id;
            if from_id != to_id {
                store
                    .relate(&from_id, &rel.relation, &to_id, None)
                    .await
                    .unwrap();
                edge_count += 1;
            }
        }
    }

    // 6. Run Reasoner
    let scope = &stored_entries[0].scope_id;
    let reasoner = Reasoner::new(&store, &store);
    let findings = reasoner.run_all(&stored_entries, scope).await.unwrap();

    // 7. Verify non-empty: entries were ingested, some graph edges exist
    assert!(
        stored_entries.len() >= 2,
        "Should have at least 2 entries from 2 Rust files"
    );
    assert!(
        edge_count >= 1,
        "Should have wired at least 1 graph edge from relations"
    );

    // The reasoner should produce findings (some entries will be orphaned
    // since we only wired partial relations)
    // This is the self-dogfood validation: the full pipeline works end-to-end.
    eprintln!(
        "Self-dogfood: {} entries, {} edges, {} findings",
        stored_entries.len(),
        edge_count,
        findings.len()
    );

    // Verify entries are searchable in the store
    let search_results = store
        .search(&[1.0, 0.0, 0.0], scope, 10)
        .await
        .unwrap();
    assert!(
        !search_results.is_empty(),
        "Entries should be searchable after ingestion"
    );

    // Verify knowledge files are readable
    let read_back = knowledge_files::read_all(store_dir.path()).unwrap();
    assert_eq!(
        read_back.len(),
        stored_entries.len(),
        "All entries should be readable from knowledge files"
    );
}
