use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use redb::{Database, ReadableTable, TableDefinition};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Redb table for persistent graph edges.
/// Key: "{from_uuid}:{relation}:{to_uuid}", Value: edge metadata JSON
pub const GRAPH_EDGES: TableDefinition<&str, &[u8]> = TableDefinition::new("graph_edges");

/// In-memory graph backed by petgraph, persisted to Redb.
/// Provides BFS/DFS/shortest-path at graph-traversal speed.
/// Redb is the persistence layer; petgraph is the compute layer.
pub struct LiteGraphStore {
    db: Arc<Database>,
    graph: Mutex<DiGraph<Uuid, String>>,
    node_map: Mutex<HashMap<Uuid, NodeIndex>>,
}

impl LiteGraphStore {
    /// Create a new LiteGraphStore backed by the given Redb database.
    /// Rebuilds the in-memory petgraph from persisted edges on construction.
    pub fn new(db: Arc<Database>) -> Result<Self> {
        let store = Self {
            db,
            graph: Mutex::new(DiGraph::new()),
            node_map: Mutex::new(HashMap::new()),
        };
        store.rebuild_from_redb()?;
        Ok(store)
    }

    /// Rebuild the in-memory petgraph from all edges stored in Redb GRAPH_EDGES table.
    fn rebuild_from_redb(&self) -> Result<()> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read: {e}")))?;
        let table = match read_txn.open_table(GRAPH_EDGES) {
            Ok(t) => t,
            Err(_) => return Ok(()), // Table doesn't exist yet — no edges to rebuild
        };

        let mut graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Graph lock poisoned: {e}")))?;
        let mut node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Node map lock poisoned: {e}")))?;

        for item in table.iter().map_err(|e| CorviaError::Storage(format!("Failed to iterate: {e}")))? {
            let (key, _value) = item.map_err(|e| CorviaError::Storage(format!("Failed to read: {e}")))?;
            let key_str = key.value();
            // Parse key: "{from_uuid}:{relation}:{to_uuid}"
            // UUIDs contain hyphens but not colons, so splitn(3, ':') safely separates the 3 parts
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() != 3 {
                tracing::warn!("Malformed GRAPH_EDGES key: {}", key_str);
                continue;
            }

            let from = match Uuid::parse_str(parts[0]) { Ok(u) => u, Err(_) => continue };
            let to = match Uuid::parse_str(parts[2]) { Ok(u) => u, Err(_) => continue };
            let relation = parts[1].to_string();

            let from_idx = *node_map.entry(from).or_insert_with(|| graph.add_node(from));
            let to_idx = *node_map.entry(to).or_insert_with(|| graph.add_node(to));
            graph.add_edge(from_idx, to_idx, relation);
        }

        let edge_count = graph.edge_count();
        if edge_count > 0 {
            tracing::info!("Rebuilt graph with {} edges from Redb", edge_count);
        }

        Ok(())
    }

    /// BFS traversal returning UUIDs (no entry lookup). Used by trait method and tests.
    pub fn traverse_ids(
        &self,
        start: &Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<Uuid>> {
        let graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;

        let start_idx = match node_map.get(start) {
            Some(idx) => *idx,
            None => return Ok(Vec::new()),
        };

        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut results = Vec::new();

        visited.insert(start_idx);
        queue.push_back((start_idx, 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let petgraph_dir = match direction {
                EdgeDirection::Outgoing => petgraph::Direction::Outgoing,
                EdgeDirection::Incoming => petgraph::Direction::Incoming,
                EdgeDirection::Both => petgraph::Direction::Outgoing, // Handle Both below
            };

            let mut process_neighbors = |dir: petgraph::Direction| {
                for edge in graph.edges_directed(current, dir) {
                    let neighbor = if dir == petgraph::Direction::Outgoing {
                        edge.target()
                    } else {
                        edge.source()
                    };

                    // Filter by relation if specified
                    if let Some(rel) = relation {
                        if edge.weight() != rel {
                            continue;
                        }
                    }

                    if visited.insert(neighbor) {
                        results.push(graph[neighbor]);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            };

            process_neighbors(petgraph_dir);
            if direction == EdgeDirection::Both {
                process_neighbors(petgraph::Direction::Incoming);
            }
        }

        Ok(results)
    }

    /// Shortest path returning UUIDs. Used by trait method and tests.
    pub fn shortest_path_ids(&self, from: &Uuid, to: &Uuid) -> Result<Option<Vec<Uuid>>> {
        let graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;

        let from_idx = match node_map.get(from) {
            Some(idx) => *idx,
            None => return Ok(None),
        };
        let to_idx = match node_map.get(to) {
            Some(idx) => *idx,
            None => return Ok(None),
        };

        // A* with unit weight and zero heuristic = Dijkstra
        let result = petgraph::algo::astar(
            &*graph,
            from_idx,
            |n| n == to_idx,
            |_| 1u32,
            |_| 0u32,
        );

        match result {
            Some((_cost, path)) => {
                let uuid_path: Vec<Uuid> = path.iter().map(|idx| graph[*idx]).collect();
                Ok(Some(uuid_path))
            }
            None => Ok(None),
        }
    }
}

#[async_trait]
impl crate::traits::GraphStore for LiteGraphStore {
    async fn relate(
        &self,
        from: &Uuid,
        relation: &str,
        to: &Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        // Validate relation name: colons would corrupt the composite key format
        if relation.contains(':') {
            return Err(CorviaError::Validation(
                format!("Relation name must not contain ':': {relation}")
            ));
        }
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
        let mut graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let mut node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let from_idx = *node_map.entry(*from).or_insert_with(|| graph.add_node(*from));
        let to_idx = *node_map.entry(*to).or_insert_with(|| graph.add_node(*to));
        graph.add_edge(from_idx, to_idx, relation.to_string());

        Ok(())
    }

    async fn edges(
        &self,
        entry_id: &Uuid,
        direction: EdgeDirection,
    ) -> Result<Vec<GraphEdge>> {
        let graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;

        let node_idx = match node_map.get(entry_id) {
            Some(idx) => *idx,
            None => return Ok(Vec::new()),
        };

        let mut edges = Vec::new();

        match direction {
            EdgeDirection::Outgoing | EdgeDirection::Both => {
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
                    let to_uuid = graph[edge.target()];
                    edges.push(GraphEdge {
                        from: *entry_id,
                        to: to_uuid,
                        relation: edge.weight().clone(),
                        metadata: None,
                    });
                }
            }
            _ => {}
        }

        match direction {
            EdgeDirection::Incoming | EdgeDirection::Both => {
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let from_uuid = graph[edge.source()];
                    edges.push(GraphEdge {
                        from: from_uuid,
                        to: *entry_id,
                        relation: edge.weight().clone(),
                        metadata: None,
                    });
                }
            }
            _ => {}
        }

        Ok(edges)
    }

    async fn traverse(
        &self,
        start: &Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        // Returns empty for now — entry lookup requires QueryableStore (wired in Task 7)
        // traverse_ids is the real implementation
        let _ids = self.traverse_ids(start, relation, direction, max_depth)?;
        Ok(Vec::new())
    }

    async fn shortest_path(
        &self,
        from: &Uuid,
        to: &Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>> {
        // Returns None for now — entry lookup requires QueryableStore (wired in Task 7)
        // shortest_path_ids is the real implementation
        let _ids = self.shortest_path_ids(from, to)?;
        Ok(None)
    }

    async fn remove_edges(&self, entry_id: &Uuid) -> Result<()> {
        let entry_str = entry_id.to_string();

        // Remove from Redb: scan for keys containing this UUID
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write: {e}")))?;
        {
            let mut table = write_txn.open_table(GRAPH_EDGES)
                .map_err(|e| CorviaError::Storage(format!("Failed to open GRAPH_EDGES: {e}")))?;

            // Collect keys to remove (can't mutate while iterating)
            let keys_to_remove: Vec<String> = table.iter()
                .map_err(|e| CorviaError::Storage(format!("Failed to iterate: {e}")))?
                .filter_map(|item| {
                    let (key, _) = item.ok()?;
                    let k = key.value().to_string();
                    if k.starts_with(&format!("{}:", entry_str))
                        || k.ends_with(&format!(":{}", entry_str))
                    {
                        Some(k)
                    } else {
                        None
                    }
                })
                .collect();

            for key in &keys_to_remove {
                table.remove(key.as_str())
                    .map_err(|e| CorviaError::Storage(format!("Failed to remove edge: {e}")))?;
            }
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit remove: {e}")))?;

        // Remove from petgraph
        let mut graph = self.graph.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;
        let node_map = self.node_map.lock()
            .map_err(|e| CorviaError::Storage(format!("Lock: {e}")))?;

        if let Some(&node_idx) = node_map.get(entry_id) {
            // Collect all edge indices (outgoing and incoming) to remove
            let outgoing: Vec<_> = graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .map(|e| e.id())
                .collect();
            let incoming: Vec<_> = graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .map(|e| e.id())
                .collect();

            for idx in outgoing.into_iter().chain(incoming) {
                graph.remove_edge(idx);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::GraphStore;

    fn test_db() -> Arc<Database> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let db = Arc::new(Database::create(path).unwrap());
        // Create GRAPH_EDGES table
        {
            let w = db.begin_write().unwrap();
            w.open_table(GRAPH_EDGES).unwrap();
            w.commit().unwrap();
        }
        std::mem::forget(dir); // keep tempdir alive
        db
    }

    #[tokio::test]
    async fn test_relate_and_edges() {
        let db = test_db();
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

        // Both direction
        let both = store.edges(&a, EdgeDirection::Both).await.unwrap();
        assert_eq!(both.len(), 1);
    }

    #[tokio::test]
    async fn test_traverse_bfs() {
        let db = test_db();
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();
        store.relate(&b, "imports", &c, None).await.unwrap();

        // Traverse from a, depth 2 — should reach b and c
        let results = store.traverse_ids(&a, None, EdgeDirection::Outgoing, 2).unwrap();
        assert_eq!(results.len(), 2); // b and c (not a itself)
    }

    #[tokio::test]
    async fn test_shortest_path_ids() {
        let db = test_db();
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();
        store.relate(&b, "calls", &c, None).await.unwrap();

        let path = store.shortest_path_ids(&a, &c).unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3); // a -> b -> c
        assert_eq!(path[0], a);
        assert_eq!(path[1], b);
        assert_eq!(path[2], c);

        // No path in reverse (directed graph)
        let reverse = store.shortest_path_ids(&c, &a).unwrap();
        assert!(reverse.is_none());
    }

    #[tokio::test]
    async fn test_remove_edges() {
        let db = test_db();
        let store = LiteGraphStore::new(db).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        store.relate(&a, "imports", &b, None).await.unwrap();
        assert_eq!(
            store.edges(&a, EdgeDirection::Outgoing).await.unwrap().len(),
            1
        );

        store.remove_edges(&a).await.unwrap();
        assert_eq!(
            store.edges(&a, EdgeDirection::Outgoing).await.unwrap().len(),
            0
        );
        // b's incoming edge should also be gone
        assert_eq!(
            store.edges(&b, EdgeDirection::Incoming).await.unwrap().len(),
            0
        );
    }
}
