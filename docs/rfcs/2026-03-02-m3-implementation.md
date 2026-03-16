# M3: Temporal + Graph + Reasoning — Implementation Plan

> **Status:** Shipped (v0.3.0)

**Goal:** Add bi-temporal queries, graph traversal, and algorithmic reasoning to Corvia's kernel — making knowledge time-aware, relationship-aware, and self-analyzing.

**Architecture:** Three new traits (`TemporalStore`, `GraphStore`) implemented by both LiteStore and SurrealStore, plus a `Reasoner` compute module. Graph populated automatically from tree-sitter ingestion and manually via CLI/API. `corvia upgrade` provides LiteStore→FullStore migration.

**Tech Stack:** Rust, petgraph (new dep), redb (existing), hnsw_rs (existing), surrealdb (existing), tree-sitter (via corvia-adapter-git)

**Design doc:** `docs/rfcs/2026-03-02-m3-temporal-graph-reasoning-design.md`

---

## Task Group A: Temporal Store (Tasks 1-4)

### Task 1: TemporalStore trait + GraphStore trait + types

Define the new traits and types that all subsequent tasks build on.

**Files:**
- Modify: `crates/corvia-kernel/src/traits.rs`
- Modify: `crates/corvia-common/src/types.rs`

**Step 1: Add graph/temporal types to `types.rs`**

Add after the `SearchResult` struct (line 87):

```rust
/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub relation: String,
    pub metadata: Option<serde_json::Value>,
}

/// Direction for graph edge queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}
```

**Step 2: Add TemporalStore and GraphStore traits to `traits.rs`**

Add after the `IngestionAdapter` trait (line 52):

```rust
/// Temporal query interface for bi-temporal knowledge (D38).
/// Implemented by LiteStore (Redb range scans) and SurrealStore (SurrealQL).
#[async_trait]
pub trait TemporalStore: Send + Sync {
    /// Return entries valid at a point in time within a scope.
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Follow the supersession chain for an entry (newest → oldest).
    async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>>;

    /// Return entries that changed within a time range (created, superseded, or expired).
    async fn evolution(
        &self,
        scope_id: &str,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<KnowledgeEntry>>;
}

/// Knowledge graph interface (D37).
/// Implemented by LiteStore (petgraph + Redb) and SurrealStore (RELATE).
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Create a directed edge between two entries.
    async fn relate(
        &self,
        from: &uuid::Uuid,
        relation: &str,
        to: &uuid::Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()>;

    /// Get all edges from/to an entry.
    async fn edges(
        &self,
        entry_id: &uuid::Uuid,
        direction: corvia_common::types::EdgeDirection,
    ) -> Result<Vec<corvia_common::types::GraphEdge>>;

    /// BFS traversal from a starting node, optionally filtering by relation type.
    async fn traverse(
        &self,
        start: &uuid::Uuid,
        relation: Option<&str>,
        direction: corvia_common::types::EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Shortest path between two entries (returns the entries along the path).
    async fn shortest_path(
        &self,
        from: &uuid::Uuid,
        to: &uuid::Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>>;

    /// Delete all edges involving an entry.
    async fn remove_edges(&self, entry_id: &uuid::Uuid) -> Result<()>;
}
```

**Step 3: Add necessary imports to `traits.rs`**

Add `chrono` and `GraphEdge`/`EdgeDirection` to the use statements at the top.

**Step 4: Verify compilation**

Run: `cargo check -p corvia-kernel`
Expected: Compiles (traits defined but not yet implemented).

**Step 5: Commit**

```
git add crates/corvia-kernel/src/traits.rs crates/corvia-common/src/types.rs
git commit -m "feat(m3): define TemporalStore and GraphStore traits + graph types"
```

---

### Task 2: LiteStore temporal index — Redb table + insert integration

Add the `TEMPORAL_INDEX` Redb table and populate it during `insert()`.

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs`

**Step 1: Write failing test**

Add to `mod tests` in `lite_store.rs`:

```rust
#[tokio::test]
async fn test_temporal_index_populated_on_insert() {
    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    let entry = KnowledgeEntry::new("temporal test".into(), "scope".into(), "v1".into())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    let id = entry.id;
    let valid_from = entry.valid_from;
    store.insert(&entry).await.unwrap();

    // Verify temporal index was written
    let results = store.as_of("scope", valid_from + chrono::Duration::seconds(1), 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, id);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-kernel -- test_temporal_index_populated_on_insert`
Expected: FAIL — `as_of` method doesn't exist yet.

**Step 3: Add TEMPORAL_INDEX table definition**

Add after the existing table definitions (line 18 of `lite_store.rs`):

```rust
/// Temporal index: compound key (scope_id, valid_from_millis, entry_id) → (valid_to_millis, recorded_at_millis)
/// Enables O(log n) range scans for bi-temporal queries via Redb B-tree.
const TEMPORAL_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("temporal_index");
```

Key format: `"{scope_id}:{valid_from_millis:020}:{entry_id}"` (zero-padded millis for lexicographic ordering).
Value format: JSON `{"valid_to_millis": Option<i64>, "recorded_at_millis": i64}`.

**Step 4: Write temporal index during insert**

In the `insert()` method, after `self.persist_next_id()?;`, add:

```rust
// 4. Write temporal index entry
self.write_temporal_index(entry)?;
```

Implement `write_temporal_index`:

```rust
fn write_temporal_index(&self, entry: &KnowledgeEntry) -> Result<()> {
    let valid_from_millis = entry.valid_from.timestamp_millis();
    let recorded_at_millis = entry.recorded_at.timestamp_millis();
    let valid_to_millis = entry.valid_to.map(|t| t.timestamp_millis());

    let key = format!(
        "{}:{:020}:{}",
        entry.scope_id, valid_from_millis, entry.id
    );
    let value = serde_json::to_vec(&serde_json::json!({
        "valid_to_millis": valid_to_millis,
        "recorded_at_millis": recorded_at_millis,
    }))
    .map_err(|e| CorviaError::Storage(format!("Failed to serialize temporal value: {e}")))?;

    let write_txn = self.db.begin_write()
        .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
    {
        let mut table = write_txn.open_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;
        table.insert(key.as_str(), value.as_slice())
            .map_err(|e| CorviaError::Storage(format!("Failed to insert temporal index: {e}")))?;
    }
    write_txn.commit()
        .map_err(|e| CorviaError::Storage(format!("Failed to commit temporal index: {e}")))?;
    Ok(())
}
```

**Step 5: Add TEMPORAL_INDEX to `init_schema`**

In `init_schema()`, add alongside the other table creations:

```rust
let _ = write_txn.open_table(TEMPORAL_INDEX)
    .map_err(|e| CorviaError::Storage(format!("Failed to create TEMPORAL_INDEX: {e}")))?;
```

**Step 6: Implement `as_of` on LiteStore** (minimal, to make the test pass)

```rust
impl TemporalStore for LiteStore {
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        let ts_millis = timestamp.timestamp_millis();
        // Range scan: all keys from "{scope_id}:00000000000000000000:" to "{scope_id}:{ts_millis:020}:\xff"
        let range_start = format!("{}:", scope_id);
        let range_end_key = format!("{}:{:020}:\xff", scope_id, ts_millis);

        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;

        let entries_table = read_txn.open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

        let range = table.range(range_start.as_str()..=range_end_key.as_str())
            .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

        let mut results = Vec::new();
        for item in range {
            let (key, value) = item
                .map_err(|e| CorviaError::Storage(format!("Failed to read range item: {e}")))?;
            let key_str = key.value();

            // Parse valid_to from value to check if entry is still valid at timestamp
            let val_json: serde_json::Value = serde_json::from_slice(value.value())
                .map_err(|e| CorviaError::Storage(format!("Failed to parse temporal value: {e}")))?;

            if let Some(valid_to_millis) = val_json["valid_to_millis"].as_i64() {
                if valid_to_millis <= ts_millis {
                    continue; // Entry expired before the query timestamp
                }
            }

            // Extract entry_id from compound key
            let parts: Vec<&str> = key_str.rsplitn(2, ':').collect();
            if parts.len() < 2 { continue; }
            let entry_uuid_str = parts[0];

            // Look up entry
            if let Ok(Some(entry_val)) = entries_table.get(entry_uuid_str) {
                if let Ok(entry) = serde_json::from_slice::<KnowledgeEntry>(entry_val.value()) {
                    results.push(entry);
                    if results.len() >= limit { break; }
                }
            }
        }

        Ok(results)
    }

    async fn history(&self, _entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>> {
        todo!("Implemented in Task 3")
    }

    async fn evolution(
        &self, _scope_id: &str,
        _from: chrono::DateTime<chrono::Utc>,
        _to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<KnowledgeEntry>> {
        todo!("Implemented in Task 3")
    }
}
```

**Step 7: Run test to verify it passes**

Run: `cargo test -p corvia-kernel -- test_temporal_index_populated_on_insert`
Expected: PASS

**Step 8: Commit**

```
git add crates/corvia-kernel/src/lite_store.rs
git commit -m "feat(m3): temporal index in LiteStore — Redb table + as_of query"
```

---

### Task 3: LiteStore TemporalStore — history + evolution

Complete the remaining TemporalStore methods.

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs`

**Step 1: Write failing tests**

```rust
#[tokio::test]
async fn test_temporal_history_follows_chain() {
    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    // Create original entry
    let mut e1 = KnowledgeEntry::new("version 1".into(), "scope".into(), "v1".into())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    let e1_id = e1.id;
    store.insert(&e1).await.unwrap();

    // Create superseding entry
    let e2 = KnowledgeEntry::new("version 2".into(), "scope".into(), "v2".into())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    let e2_id = e2.id;
    store.insert(&e2).await.unwrap();

    // Mark e1 as superseded by e2
    store.supersede(&e1_id, &e2_id).await.unwrap();

    let history = store.history(&e2_id).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].id, e2_id); // newest first
    assert_eq!(history[1].id, e1_id);
}

#[tokio::test]
async fn test_temporal_evolution_time_range() {
    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 3).unwrap();
    store.init_schema().await.unwrap();

    let before = chrono::Utc::now();
    let e1 = KnowledgeEntry::new("entry 1".into(), "scope".into(), "v1".into())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    store.insert(&e1).await.unwrap();
    let after = chrono::Utc::now();

    let results = store.evolution("scope", before, after).await.unwrap();
    assert!(results.len() >= 1);

    // Query a range before the entry was created — should be empty
    let old_results = store.evolution(
        "scope",
        before - chrono::Duration::hours(2),
        before - chrono::Duration::hours(1),
    ).await.unwrap();
    assert_eq!(old_results.len(), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p corvia-kernel -- test_temporal_history test_temporal_evolution`
Expected: FAIL — `supersede` doesn't exist, `history`/`evolution` are `todo!()`.

**Step 3: Implement `supersede` helper**

Add a public method on `LiteStore`:

```rust
/// Mark an entry as superseded by another. Updates valid_to, superseded_by,
/// and the temporal index for the old entry.
pub async fn supersede(&self, old_id: &Uuid, new_id: &Uuid) -> Result<()> {
    let now = chrono::Utc::now();

    // Read old entry, update fields, write back
    let mut old_entry = self.get(old_id).await?
        .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", old_id)))?;

    old_entry.valid_to = Some(now);
    old_entry.superseded_by = Some(*new_id);

    // Update ENTRIES table
    let entry_json = serde_json::to_vec(&old_entry)
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize: {e}")))?;
    let write_txn = self.db.begin_write()
        .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
    {
        let mut table = write_txn.open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;
        table.insert(old_id.to_string().as_str(), entry_json.as_slice())
            .map_err(|e| CorviaError::Storage(format!("Failed to update entry: {e}")))?;
    }
    write_txn.commit()
        .map_err(|e| CorviaError::Storage(format!("Failed to commit supersede: {e}")))?;

    // Update temporal index for old entry
    self.write_temporal_index(&old_entry)?;

    // Update knowledge file
    knowledge_files::write_entry(&self.data_dir, &old_entry)?;

    Ok(())
}
```

**Step 4: Implement `history`**

Replace the `todo!()` in the `TemporalStore` impl:

```rust
async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>> {
    let mut chain = Vec::new();
    let mut current_id = *entry_id;

    // Walk forward from this entry to find the newest in the chain
    loop {
        let entry = self.get(&current_id).await?
            .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found in chain", current_id)))?;
        chain.push(entry.clone());

        // Follow superseded_by backward: find who superseded the current entry's predecessor
        // Actually, we need to walk the chain. Start from given entry, find older entries.
        break;
    }

    // Now walk backward via superseded_by from the starting entry
    chain.clear();
    let start = self.get(entry_id).await?
        .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", entry_id)))?;
    chain.push(start.clone());

    // Walk backward: find entries whose superseded_by == current
    // Since superseded_by points forward (old → new), we need to find entries that were superseded BY this one
    // Actually: old_entry.superseded_by = new_id. So to go backward from new_id, we search for entries where superseded_by == entry_id

    // For history starting from newest: find older entries by scanning ENTRIES for superseded_by chains
    // More efficient: use entry's own data. Start entry might have no superseded_by (it's newest).
    // Walk: find entry where superseded_by == entry_id (that's the predecessor).

    // Simple approach: scan entries looking for superseded_by matches
    // For better perf, could add a reverse index. For now, scan knowledge files.
    let all_entries = knowledge_files::read_all(&self.data_dir)?;
    let mut by_superseded: std::collections::HashMap<Uuid, KnowledgeEntry> = all_entries
        .into_iter()
        .filter_map(|e| e.superseded_by.map(|s| (s, e)))
        .collect();

    let mut current = entry_id;
    loop {
        match by_superseded.remove(current) {
            Some(predecessor) => {
                let pred_id = predecessor.id;
                chain.push(predecessor);
                current = &chain.last().unwrap().id;
                let _ = pred_id; // just to use the binding
            }
            None => break,
        }
    }

    Ok(chain)
}
```

Note: This is a simple scan-based approach. For large stores, a `SUPERSEDED_BY_REVERSE` Redb index would be better, but YAGNI — optimize when needed.

**Step 5: Implement `evolution`**

Replace the `todo!()`:

```rust
async fn evolution(
    &self,
    scope_id: &str,
    from: chrono::DateTime<chrono::Utc>,
    to: chrono::DateTime<chrono::Utc>,
) -> Result<Vec<KnowledgeEntry>> {
    let from_millis = from.timestamp_millis();
    let to_millis = to.timestamp_millis();

    let range_start = format!("{}:{:020}:", scope_id, from_millis);
    let range_end = format!("{}:{:020}:\xff", scope_id, to_millis);

    let read_txn = self.db.begin_read()
        .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
    let table = read_txn.open_table(TEMPORAL_INDEX)
        .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;
    let entries_table = read_txn.open_table(ENTRIES)
        .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

    let range = table.range(range_start.as_str()..=range_end.as_str())
        .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

    let mut results = Vec::new();
    for item in range {
        let (key, _value) = item
            .map_err(|e| CorviaError::Storage(format!("Failed to read range item: {e}")))?;
        let key_str = key.value();
        let parts: Vec<&str> = key_str.rsplitn(2, ':').collect();
        if parts.len() < 2 { continue; }
        let entry_uuid_str = parts[0];

        if let Ok(Some(entry_val)) = entries_table.get(entry_uuid_str) {
            if let Ok(entry) = serde_json::from_slice::<KnowledgeEntry>(entry_val.value()) {
                results.push(entry);
            }
        }
    }

    Ok(results)
}
```

**Step 6: Run tests**

Run: `cargo test -p corvia-kernel -- test_temporal`
Expected: All temporal tests PASS.

**Step 7: Commit**

```
git add crates/corvia-kernel/src/lite_store.rs
git commit -m "feat(m3): LiteStore temporal — history (supersession chain) + evolution (time range)"
```

---

### Task 4: SurrealStore TemporalStore implementation

Implement TemporalStore for the SurrealDB backend.

**Files:**
- Modify: `crates/corvia-kernel/src/knowledge_store.rs`

**Step 1: Write tests** (gated on SurrealDB availability)

```rust
#[tokio::test]
async fn test_surreal_as_of() {
    // Skip if SurrealDB not available
    let store = match connect_test_store().await {
        Ok(s) => s,
        Err(_) => { eprintln!("Skipping: SurrealDB not available"); return; }
    };
    // ... similar to LiteStore temporal test
}
```

**Step 2: Implement TemporalStore for SurrealStore**

Use SurrealQL queries:
- `as_of`: `SELECT * FROM knowledge WHERE scope_id = $scope AND valid_from <= $ts AND (valid_to > $ts OR valid_to IS NONE) LIMIT $limit`
- `history`: Recursive chain following `superseded_by` via multiple GETs
- `evolution`: `SELECT * FROM knowledge WHERE scope_id = $scope AND valid_from >= $from AND valid_from <= $to`

**Step 3: Run tests, commit**

```
git commit -m "feat(m3): SurrealStore temporal — as_of, history, evolution via SurrealQL"
```

---

## Task Group B: Graph Store (Tasks 5-8)

### Task 5: Add petgraph dependency + GraphStore LiteStore scaffolding

**Files:**
- Modify: `crates/corvia-kernel/Cargo.toml`
- Create: `crates/corvia-kernel/src/graph_store.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Add petgraph to Cargo.toml**

Add to `[dependencies]`:
```toml
petgraph = "0.7"
```

**Step 2: Create `graph_store.rs` module**

Scaffold the module with the Redb table, petgraph struct, and stub implementations:

```rust
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry};
use petgraph::graph::{DiGraph, NodeIndex};
use redb::{Database, TableDefinition};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Redb table for persistent graph edges.
/// Key: "{from_uuid}:{relation}:{to_uuid}", Value: edge metadata JSON
const GRAPH_EDGES: TableDefinition<&str, &[u8]> = TableDefinition::new("graph_edges");

/// In-memory graph backed by petgraph, persisted to Redb.
pub struct LiteGraphStore {
    db: Arc<Database>,
    graph: Mutex<DiGraph<Uuid, String>>,
    node_map: Mutex<HashMap<Uuid, NodeIndex>>,
}
```

**Step 3: Implement constructor + rebuild from Redb**

```rust
impl LiteGraphStore {
    pub fn new(db: Arc<Database>) -> Result<Self> {
        let store = Self {
            db,
            graph: Mutex::new(DiGraph::new()),
            node_map: Mutex::new(HashMap::new()),
        };
        store.rebuild_from_redb()?;
        Ok(store)
    }

    fn rebuild_from_redb(&self) -> Result<()> {
        // Read all edges from GRAPH_EDGES table, populate petgraph
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read: {e}")))?;
        let table = match read_txn.open_table(GRAPH_EDGES) {
            Ok(t) => t,
            Err(_) => return Ok(()), // Table doesn't exist yet
        };

        let mut graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Graph lock poisoned: {e}")))?;
        let mut node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Node map lock poisoned: {e}")))?;

        for item in table.iter().map_err(|e| CorviaError::Storage(format!("Failed to iterate: {e}")))? {
            let (key, _value) = item.map_err(|e| CorviaError::Storage(format!("Failed to read: {e}")))?;
            let key_str = key.value();
            // Parse key: "{from}:{relation}:{to}"
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() != 3 { continue; }

            let from = match Uuid::parse_str(parts[0]) { Ok(u) => u, Err(_) => continue };
            let to = match Uuid::parse_str(parts[2]) { Ok(u) => u, Err(_) => continue };
            let relation = parts[1].to_string();

            let from_idx = *node_map.entry(from).or_insert_with(|| graph.add_node(from));
            let to_idx = *node_map.entry(to).or_insert_with(|| graph.add_node(to));
            graph.add_edge(from_idx, to_idx, relation);
        }

        Ok(())
    }
}
```

**Step 4: Register module in `lib.rs`**

Add: `pub mod graph_store;`

**Step 5: Verify compilation**

Run: `cargo check -p corvia-kernel`

**Step 6: Commit**

```
git commit -m "feat(m3): graph_store module — petgraph + Redb scaffolding"
```

---

### Task 6: LiteGraphStore — GraphStore trait implementation

Implement `relate`, `edges`, `traverse`, `shortest_path`, `remove_edges`.

**Files:**
- Modify: `crates/corvia-kernel/src/graph_store.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use redb::Database;

    fn test_db() -> Arc<Database> {
        let dir = tempfile::tempdir().unwrap();
        Arc::new(Database::create(dir.path().join("test.redb")).unwrap())
    }

    #[tokio::test]
    async fn test_relate_and_edges() {
        let db = test_db();
        // Create GRAPH_EDGES table
        { let w = db.begin_write().unwrap(); w.open_table(GRAPH_EDGES).unwrap(); w.commit().unwrap(); }
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();

        let out_edges = store.edges(&a, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].relation, "imports");
        assert_eq!(out_edges[0].to, b);

        let in_edges = store.edges(&b, EdgeDirection::Incoming).await.unwrap();
        assert_eq!(in_edges.len(), 1);
        assert_eq!(in_edges[0].from, a);
    }

    #[tokio::test]
    async fn test_traverse_bfs() {
        let db = test_db();
        { let w = db.begin_write().unwrap(); w.open_table(GRAPH_EDGES).unwrap(); w.commit().unwrap(); }
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();
        store.relate(&b, "imports", &c, None).await.unwrap();

        // Traverse from a, depth 2 — should reach b and c
        // Note: traverse returns UUIDs found, but entry lookup is caller's job
        // For this test, just check the traversal finds the right node count
        let results = store.traverse_ids(&a, None, EdgeDirection::Outgoing, 2).unwrap();
        assert_eq!(results.len(), 2); // b and c
    }

    #[tokio::test]
    async fn test_remove_edges() {
        let db = test_db();
        { let w = db.begin_write().unwrap(); w.open_table(GRAPH_EDGES).unwrap(); w.commit().unwrap(); }
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();
        assert_eq!(store.edges(&a, EdgeDirection::Outgoing).await.unwrap().len(), 1);

        store.remove_edges(&a).await.unwrap();
        assert_eq!(store.edges(&a, EdgeDirection::Outgoing).await.unwrap().len(), 0);
    }
}
```

**Step 2: Implement `relate`**

Write to both Redb and petgraph:

```rust
async fn relate(&self, from: &Uuid, relation: &str, to: &Uuid, metadata: Option<serde_json::Value>) -> Result<()> {
    let key = format!("{}:{}:{}", from, relation, to);
    let value = serde_json::to_vec(&metadata.unwrap_or(serde_json::Value::Null))
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize: {e}")))?;

    // Persist to Redb
    let write_txn = self.db.begin_write()
        .map_err(|e| CorviaError::Storage(format!("Failed to begin write: {e}")))?;
    {
        let mut table = write_txn.open_table(GRAPH_EDGES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open GRAPH_EDGES: {e}")))?;
        table.insert(key.as_str(), value.as_slice())
            .map_err(|e| CorviaError::Storage(format!("Failed to insert edge: {e}")))?;
    }
    write_txn.commit()
        .map_err(|e| CorviaError::Storage(format!("Failed to commit edge: {e}")))?;

    // Update in-memory graph
    let mut graph = self.graph.lock().map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
    let mut node_map = self.node_map.lock().map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
    let from_idx = *node_map.entry(*from).or_insert_with(|| graph.add_node(*from));
    let to_idx = *node_map.entry(*to).or_insert_with(|| graph.add_node(*to));
    graph.add_edge(from_idx, to_idx, relation.to_string());

    Ok(())
}
```

**Step 3: Implement `edges`**

Query petgraph for edges in the specified direction.

**Step 4: Implement `traverse_ids` (internal BFS helper) and `traverse`**

Use `petgraph::visit::Bfs` for BFS traversal with max_depth.

**Step 5: Implement `shortest_path`**

Use `petgraph::algo::astar` or `dijkstra` with unit weights.

**Step 6: Implement `remove_edges`**

Remove from both Redb (scan for matching keys) and petgraph (remove edges from graph).

**Step 7: Run tests, commit**

```
git commit -m "feat(m3): LiteGraphStore — relate, edges, traverse, shortest_path, remove"
```

---

### Task 7: Integrate GraphStore into LiteStore

Wire the `LiteGraphStore` into the `LiteStore` struct so it shares the same Redb database.

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs`
- Modify: `crates/corvia-kernel/src/graph_store.rs`

**Step 1:** Refactor `LiteStore` to hold `Arc<Database>` instead of `Database` (so it can share with `LiteGraphStore`).

**Step 2:** Add a `graph: LiteGraphStore` field to `LiteStore`.

**Step 3:** Delegate `GraphStore` trait methods from `LiteStore` to `self.graph`.

**Step 4:** Add `GRAPH_EDGES` table creation to `LiteStore::init_schema()`.

**Step 5:** Run all existing tests + new graph integration test, commit.

```
git commit -m "feat(m3): integrate GraphStore into LiteStore — shared Redb, delegated trait"
```

---

### Task 8: SurrealStore GraphStore implementation

Implement GraphStore for SurrealDB using `RELATE` statements.

**Files:**
- Modify: `crates/corvia-kernel/src/knowledge_store.rs`

Similar pattern to Task 4 — SurrealQL for each trait method. Commit:

```
git commit -m "feat(m3): SurrealStore graph — RELATE-based edges, traversal via SurrealQL"
```

---

## Task Group C: Adapter Relation Extraction (Tasks 9-10)

### Task 9: Extract structural relations from tree-sitter AST

Extend the git adapter to emit relations alongside code chunks during ingestion.

**Files:**
- Modify: `corvia-adapter-git/src/treesitter.rs`
- Modify: `corvia-adapter-git/src/git.rs`

**Step 1:** Define `CodeRelation` struct:

```rust
pub struct CodeRelation {
    pub from_chunk_index: usize,  // index into the chunks vec
    pub relation: String,         // "imports", "calls", "implements", "contains"
    pub to_file: String,          // resolved file path (for cross-file relations)
    pub to_name: Option<String>,  // symbol name (for within-file relations)
}
```

**Step 2:** Add `extract_relations()` to `treesitter.rs` — parse `use`/`import` statements, `impl` blocks.

**Step 3:** Update `GitAdapter::ingest()` return type to include relations (or add a separate method).

**Step 4:** Write tests for Rust `use` statement extraction, `impl` block extraction.

**Step 5:** Commit.

```
git commit -m "feat(m3): tree-sitter relation extraction — imports, implements, contains"
```

---

### Task 10: Wire relation extraction into ingestion pipeline

Connect the adapter's relation output to `GraphStore::relate()` during `corvia ingest`.

**Files:**
- Modify: `crates/corvia-cli/src/main.rs` (cmd_ingest)
- Modify: `crates/corvia-cli/src/workspace.rs` (ingest_workspace)

**Step 1:** After inserting entries, resolve `CodeRelation` chunk indices to UUIDs and call `store.relate()`.

**Step 2:** Test with actual codebase ingestion (workspace ingest).

**Step 3:** Commit.

```
git commit -m "feat(m3): wire relation extraction into ingest pipeline — auto-populate graph"
```

---

## Task Group D: Reasoner (Tasks 11-13)

### Task 11: Reasoner module — algorithmic checks

**Files:**
- Create: `crates/corvia-kernel/src/reasoner.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1:** Define `Finding` struct and `ReasonerCheck` enum.

**Step 2:** Implement algorithmic checks:
- `check_stale_entries()` — entries with `valid_to` but no `superseded_by` target
- `check_broken_chains()` — `superseded_by` pointing to nonexistent entry
- `check_orphaned_nodes()` — entries with zero graph edges
- `check_dangling_imports()` — `imports` edges to nonexistent entries
- `check_cycles()` — circular `depends_on` via petgraph cycle detection

**Step 3:** Each check returns `Vec<Finding>`. Findings are converted to `KnowledgeEntry` with `chunk_type = "finding"`.

**Step 4:** Test each check in isolation with crafted test data.

**Step 5:** Commit.

```
git commit -m "feat(m3): reasoner module — stale, broken chains, orphans, dangling, cycles"
```

---

### Task 12: LLM-powered findings (opt-in)

**Files:**
- Modify: `crates/corvia-kernel/src/reasoner.rs`
- Modify: `crates/corvia-common/src/config.rs`

**Step 1:** Add `[reasoning]` config section (optional).

**Step 2:** When configured, add LLM-based checks that call `InferenceEngine::embed()` for similarity analysis.

**Step 3:** Ensure all LLM checks gracefully skip when `[reasoning]` is not configured.

**Step 4:** Commit.

```
git commit -m "feat(m3): opt-in LLM reasoning — semantic gaps and contradiction detection"
```

---

### Task 13: CLI commands — history, graph, relate, reason

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

**Step 1:** Add `History`, `Graph`, `Relate`, `Reason` to `Commands` enum.

**Step 2:** Implement handlers:
- `history`: call `TemporalStore::history()`, print chain
- `evolution`: call `TemporalStore::evolution()`, print entries with timestamps
- `graph`: call `GraphStore::edges()`, print edges table
- `relate`: call `GraphStore::relate()`, print confirmation
- `reason`: call `Reasoner::run()`, print findings summary

**Step 3:** Test CLI output manually.

**Step 4:** Commit.

```
git commit -m "feat(m3): CLI commands — history, evolution, graph, relate, reason"
```

---

## Task Group E: Upgrade + Rebuild + MCP (Tasks 14-17)

### Task 14: `corvia upgrade` command

**Files:**
- Create: `crates/corvia-cli/src/upgrade.rs`
- Modify: `crates/corvia-cli/src/main.rs`

**Step 1:** Read knowledge files, connect to SurrealDB, bulk insert, migrate edges, verify counts, update config.

**Step 2:** Test with integration test (requires SurrealDB).

**Step 3:** Commit.

```
git commit -m "feat(m3): corvia upgrade — LiteStore to SurrealDB migration"
```

---

### Task 15: Enhanced `corvia rebuild`

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs`

**Step 1:** Extend `rebuild_from_files()` to also:
- Rebuild `TEMPORAL_INDEX` from entry `valid_from`/`valid_to` fields
- Re-extract graph edges via tree-sitter (or read from a persisted edge cache)
- Rebuild petgraph from Redb edges

**Step 2:** Test: insert entries, corrupt temporal/graph indexes, rebuild, verify.

**Step 3:** Commit.

```
git commit -m "feat(m3): enhanced rebuild — temporal index + graph reconstruction"
```

---

### Task 16: MCP + REST endpoints for temporal/graph

**Files:**
- Modify: `crates/corvia-server/src/mcp.rs`
- Modify: `crates/corvia-server/src/rest.rs`

**Step 1:** Flesh out `corvia_history` MCP tool (currently stubbed).

**Step 2:** Add `corvia_graph` and `corvia_reason` MCP tools.

**Step 3:** Add REST endpoints:
- `GET /v1/entries/{id}/history`
- `GET /v1/entries/{id}/edges`
- `POST /v1/edges`
- `GET /v1/evolution?scope={scope}&since={duration}`
- `POST /v1/reason?scope={scope}`

**Step 4:** Commit.

```
git commit -m "feat(m3): MCP + REST endpoints for temporal, graph, and reasoning"
```

---

### Task 17: Integration tests + self-dogfooding

**Files:**
- Create: `tests/integration/m3_e2e_test.rs`

**Step 1:** End-to-end test: ingest code → verify edges extracted → run reasoner → verify findings.

**Step 2:** Self-dogfood test: ingest Corvia's own codebase, run `corvia reason`, verify non-empty output.

**Step 3:** Run full test suite, verify no regressions.

**Step 4:** Commit.

```
git commit -m "test(m3): integration tests + self-dogfooding validation"
```

---

## Summary

| Group | Tasks | Focus |
|-------|-------|-------|
| A | 1-4 | Temporal Store (traits, LiteStore, SurrealStore) |
| B | 5-8 | Graph Store (petgraph, LiteStore, SurrealStore) |
| C | 9-10 | Adapter relation extraction + ingestion wiring |
| D | 11-13 | Reasoner (algorithmic + LLM opt-in) + CLI |
| E | 14-17 | Upgrade, rebuild, MCP/REST, integration tests |

**Total: 17 tasks, ~3,000 LOC estimated.**

Dependencies: A and B are independent (can parallelize). C depends on B. D depends on A+B. E depends on all.
