use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EdgeDirection, EntryMetadata, GraphEdge, KnowledgeEntry, SearchResult};
use serde::Deserialize;
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::opt::auth::Root;
use surrealdb::Surreal;
use tracing::info;

// Required for TemporalStore impl to call QueryableStore::get on self
use crate::traits::QueryableStore;

/// Helper struct for deserializing KNN search results from SurrealDB v3.
/// Avoids the RecordId deserialization issue with serde_json::Value.
#[derive(Debug, Deserialize)]
struct SearchHit {
    content: String,
    scope_id: String,
    #[serde(default)]
    source_version: String,
    #[serde(default)]
    metadata: EntryMetadata,
    distance: f64,
}

pub struct SurrealStore {
    db: Surreal<Client>,
    dimensions: usize,
}

/// SELECT field list that avoids SurrealDB v3 Record type deserialization issues.
/// Uses `record::id(id) AS id` to convert the record ID to a plain string,
/// which the Rust SDK can deserialize as serde_json::Value.
/// Must include ALL fields that KnowledgeEntry expects for deserialization.
/// SELECT field list that avoids SurrealDB v3 SDK deserialization issues:
/// - `record::id(id)` converts Record type to string (SDK can't deserialize Record as JSON)
/// - `IF type::is_datetime(f) THEN type::string(f) ELSE f END` handles native datetimes
///   (e.g., values set via time::now() in UPDATE) while preserving NULL/NONE and strings.
const KNOWLEDGE_FIELDS: &str = "\
    record::id(id) AS id, content, scope_id, source_version, workstream, \
    IF type::is_datetime(recorded_at) THEN type::string(recorded_at) ELSE recorded_at END AS recorded_at, \
    IF type::is_datetime(valid_from) THEN type::string(valid_from) ELSE valid_from END AS valid_from, \
    IF type::is_datetime(valid_to) THEN type::string(valid_to) ELSE valid_to END AS valid_to, \
    embedding, superseded_by, metadata, agent_id, session_id, entry_status";

impl SurrealStore {
    pub async fn connect(
        url: &str,
        ns: &str,
        db_name: &str,
        user: &str,
        pass: &str,
        dimensions: usize,
    ) -> Result<Self> {
        let db = Surreal::new::<Ws>(url)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to connect to SurrealDB: {e}")))?;

        db.signin(Root {
            username: user.to_string(),
            password: pass.to_string(),
        })
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to sign in: {e}")))?;

        // SurrealDB v3 requires explicit namespace/database creation before use.
        // Without this, use_ns/use_db may silently fail or use the wrong scope.
        db.query(format!("DEFINE NAMESPACE IF NOT EXISTS `{ns}`;"))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to define namespace: {e}")))?;

        db.use_ns(ns)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to select namespace: {e}")))?;

        db.query(format!("DEFINE DATABASE IF NOT EXISTS `{db_name}`;"))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to define database: {e}")))?;

        db.use_db(db_name)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to select database: {e}")))?;

        info!("Connected to SurrealDB at {url} (ns={ns}, db={db_name})");

        Ok(Self { db, dimensions })
    }
}

/// Convert a KnowledgeEntry to a serde_json::Value for SurrealDB storage.
fn entry_to_json(entry: &KnowledgeEntry) -> serde_json::Value {
    serde_json::to_value(entry).unwrap_or_default()
}

/// Attempt to parse a serde_json::Value into a KnowledgeEntry.
fn json_to_entry(v: serde_json::Value) -> Option<KnowledgeEntry> {
    match serde_json::from_value(v) {
        Ok(entry) => Some(entry),
        Err(e) => {
            tracing::warn!("Failed to deserialize KnowledgeEntry from SurrealDB: {e}");
            None
        }
    }
}

#[async_trait]
impl super::traits::QueryableStore for SurrealStore {
    async fn init_schema(&self) -> Result<()> {
        let dim = self.dimensions;
        self.db
            .query(format!(
                "DEFINE TABLE IF NOT EXISTS knowledge SCHEMALESS;\
                 DEFINE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge \
                    FIELDS embedding HNSW DIMENSION {dim} DIST COSINE;\
                 DEFINE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge \
                    FIELDS scope_id;"
            ))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to init schema: {e}")))?;

        // Graph edges table for GraphStore
        self.db
            .query(
                "DEFINE TABLE IF NOT EXISTS edges SCHEMALESS; \
                 DEFINE INDEX IF NOT EXISTS idx_edges_from ON edges FIELDS from_id; \
                 DEFINE INDEX IF NOT EXISTS idx_edges_to ON edges FIELDS to_id;",
            )
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create edges schema: {e}")))?;

        info!("Schema initialized (embedding dim={dim})");
        Ok(())
    }

    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()> {
        let id_str = entry.id.to_string();
        let data = entry_to_json(entry);
        // Use backtick-quoted record literal — type::record() parameter binding
        // is unreliable in the SurrealDB v3 Rust SDK.
        let query = format!("CREATE knowledge:`{id_str}` CONTENT $data;");
        let mut response = self.db
            .query(query)
            .bind(("data", data))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to insert entry: {e}")))?;

        // Check for query-level errors (the SDK doesn't surface these automatically)
        let errors = response.take_errors();
        if !errors.is_empty() {
            let err_msgs: Vec<String> = errors.into_values().map(|e| e.to_string()).collect();
            return Err(CorviaError::Storage(format!("Insert query error: {}", err_msgs.join("; "))));
        }
        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        scope_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // SurrealDB v3 KNN operator requires literal values (no parameter binding)
        // for both K and the vector operand. Inline embedding as a literal array.
        let embedding_literal: String = format!(
            "[{}]",
            embedding.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
        );

        let query = format!(
            "SELECT content, scope_id, source_version, metadata, \
             vector::distance::knn() AS distance \
             FROM knowledge \
             WHERE embedding <|{limit},COSINE|> {embedding_literal} \
             AND scope_id = $scope_id \
             ORDER BY distance;"
        );

        let mut response = self
            .db
            .query(query)
            .bind(("scope_id", scope_id.to_string()))
            .await
            .map_err(|e| CorviaError::Storage(format!("Search failed: {e}")))?;

        // SurrealDB v3's .take() requires the SurrealValue trait, which serde_json::Value
        // implements as a universal bridge. Deserialize to JSON first, then parse into
        // our SearchHit struct via serde to avoid needing a manual SurrealValue impl.
        let raw_results: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse search results: {e}")))?;

        let search_results = raw_results
            .into_iter()
            .filter_map(|v| {
                let hit: SearchHit = serde_json::from_value(v).ok()?;
                let score = 1.0 - hit.distance as f32; // cosine distance to similarity
                let entry = KnowledgeEntry::new(
                    hit.content,
                    hit.scope_id,
                    hit.source_version,
                ).with_metadata(hit.metadata);
                Some(SearchResult { entry, score })
            })
            .collect();

        Ok(search_results)
    }

    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>> {
        let id_str = id.to_string();
        // Use record::id(id) for string comparison — the SDK cannot construct
        // SurrealDB v3 Record types via type::record() in parameter binding.
        let query = format!("SELECT {KNOWLEDGE_FIELDS} FROM knowledge WHERE record::id(id) = $id;");
        let mut response = self
            .db
            .query(query)
            .bind(("id", id_str))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to get entry: {e}")))?;

        let results: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse get result: {e}")))?;

        Ok(results.into_iter().next().and_then(json_to_entry))
    }

    async fn count(&self, scope_id: &str) -> Result<u64> {
        let mut response = self
            .db
            .query(
                "SELECT count() FROM knowledge WHERE scope_id = $scope_id GROUP ALL;",
            )
            .bind(("scope_id", scope_id.to_string()))
            .await
            .map_err(|e| CorviaError::Storage(format!("Count failed: {e}")))?;

        let result: Option<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse count: {e}")))?;

        match result {
            Some(v) => Ok(v.get("count").and_then(|c| c.as_u64()).unwrap_or(0)),
            None => Ok(0),
        }
    }

    async fn delete_scope(&self, scope_id: &str) -> Result<()> {
        self.db
            .query("DELETE FROM knowledge WHERE scope_id = $scope_id;")
            .bind(("scope_id", scope_id.to_string()))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to delete scope: {e}")))?;
        info!("Deleted all entries for scope '{scope_id}'");
        Ok(())
    }
}

#[async_trait]
impl crate::traits::TemporalStore for SurrealStore {
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        let ts = timestamp.to_rfc3339();
        // SurrealDB may not support parameter binding for LIMIT, so inline it.
        let query = format!(
            "SELECT {KNOWLEDGE_FIELDS} FROM knowledge \
             WHERE scope_id = $scope \
             AND type::datetime(valid_from) <= type::datetime($ts) \
             AND (valid_to > type::datetime($ts) OR valid_to IS NULL) \
             LIMIT {limit}"
        );
        let mut response = self.db
            .query(query)
            .bind(("scope", scope_id.to_string()))
            .bind(("ts", ts))
            .await
            .map_err(|e| CorviaError::Storage(format!("as_of query failed: {e}")))?;

        let results: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse as_of results: {e}")))?;

        Ok(results.into_iter().filter_map(json_to_entry).collect())
    }

    async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>> {
        let mut chain = Vec::new();

        // Start with the given entry
        let start = self.get(entry_id).await?
            .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", entry_id)))?;
        chain.push(start);

        // Walk backward: find entries whose superseded_by matches current
        let mut current_id = *entry_id;
        loop {
            let query = format!("SELECT {KNOWLEDGE_FIELDS} FROM knowledge WHERE superseded_by = type::string($id)");
            let mut response = self.db
                .query(query)
                .bind(("id", current_id.to_string()))
                .await
                .map_err(|e| CorviaError::Storage(format!("history query failed: {e}")))?;

            let results: Vec<serde_json::Value> = response
                .take(0)
                .map_err(|e| CorviaError::Storage(format!("Failed to parse history: {e}")))?;

            match results.into_iter().next().and_then(json_to_entry) {
                Some(predecessor) => {
                    current_id = predecessor.id;
                    chain.push(predecessor);
                }
                None => break,
            }
        }

        Ok(chain)
    }

    async fn evolution(
        &self,
        scope_id: &str,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<KnowledgeEntry>> {
        let from_ts = from.to_rfc3339();
        let to_ts = to.to_rfc3339();
        let query = format!(
            "SELECT {KNOWLEDGE_FIELDS} FROM knowledge \
             WHERE scope_id = $scope \
             AND type::datetime(valid_from) >= type::datetime($from) \
             AND type::datetime(valid_from) <= type::datetime($to)"
        );
        let mut response = self.db
            .query(query)
            .bind(("scope", scope_id.to_string()))
            .bind(("from", from_ts))
            .bind(("to", to_ts))
            .await
            .map_err(|e| CorviaError::Storage(format!("evolution query failed: {e}")))?;

        let results: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse evolution: {e}")))?;

        Ok(results.into_iter().filter_map(json_to_entry).collect())
    }
}

#[async_trait]
impl crate::traits::GraphStore for SurrealStore {
    async fn relate(
        &self,
        from: &uuid::Uuid,
        relation: &str,
        to: &uuid::Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let from_str = from.to_string();
        let to_str = to.to_string();
        let meta = metadata.unwrap_or(serde_json::Value::Null);

        let mut response = self
            .db
            .query(
                "CREATE edges SET from_id = $from_id, to_id = $to_id, \
                 relation = $relation, metadata = $metadata;",
            )
            .bind(("from_id", from_str))
            .bind(("to_id", to_str))
            .bind(("relation", relation.to_string()))
            .bind(("metadata", meta))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create edge: {e}")))?;

        let errors = response.take_errors();
        if !errors.is_empty() {
            let err_msgs: Vec<String> = errors.into_values().map(|e| e.to_string()).collect();
            return Err(CorviaError::Storage(format!(
                "Relate query error: {}",
                err_msgs.join("; ")
            )));
        }

        Ok(())
    }

    async fn edges(
        &self,
        entry_id: &uuid::Uuid,
        direction: EdgeDirection,
    ) -> Result<Vec<GraphEdge>> {
        let id_str = entry_id.to_string();

        let query = match direction {
            EdgeDirection::Outgoing => {
                "SELECT from_id, to_id, relation, metadata FROM edges WHERE from_id = $id"
            }
            EdgeDirection::Incoming => {
                "SELECT from_id, to_id, relation, metadata FROM edges WHERE to_id = $id"
            }
            EdgeDirection::Both => {
                "SELECT from_id, to_id, relation, metadata FROM edges \
                 WHERE from_id = $id OR to_id = $id"
            }
        };

        let mut response = self
            .db
            .query(query)
            .bind(("id", id_str))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to query edges: {e}")))?;

        let results: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse edges: {e}")))?;

        let mut edges = Vec::new();
        for row in results {
            let from_str = row
                .get("from_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let to_str = row
                .get("to_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let relation = row
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let metadata = row.get("metadata").cloned().and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(v)
                }
            });

            let from = match uuid::Uuid::parse_str(from_str) {
                Ok(u) => u,
                Err(_) => continue,
            };
            let to = match uuid::Uuid::parse_str(to_str) {
                Ok(u) => u,
                Err(_) => continue,
            };

            edges.push(GraphEdge {
                from,
                to,
                relation,
                metadata,
            });
        }

        Ok(edges)
    }

    async fn traverse(
        &self,
        start: &uuid::Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        // Application-level BFS (similar to LiteStore approach)
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut results = Vec::new();

        visited.insert(*start);
        queue.push_back((*start, 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Get edges from current node in the requested direction
            let current_edges = self.edges(&current, direction).await?;

            for edge in current_edges {
                // Filter by relation if specified
                if let Some(rel) = relation {
                    if edge.relation != rel {
                        continue;
                    }
                }

                // Determine the neighbor: depends on direction and which side we are on
                let neighbor = match direction {
                    EdgeDirection::Outgoing => edge.to,
                    EdgeDirection::Incoming => edge.from,
                    EdgeDirection::Both => {
                        if edge.from == current {
                            edge.to
                        } else {
                            edge.from
                        }
                    }
                };

                if visited.insert(neighbor) {
                    if let Some(entry) = self.get(&neighbor).await? {
                        results.push(entry);
                    }
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        Ok(results)
    }

    async fn shortest_path(
        &self,
        from: &uuid::Uuid,
        to: &uuid::Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>> {
        // BFS from source to target (directed, outgoing edges)
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        // parent map: child -> parent (to reconstruct path)
        let mut parent: std::collections::HashMap<uuid::Uuid, uuid::Uuid> =
            std::collections::HashMap::new();

        visited.insert(*from);
        queue.push_back(*from);

        let mut found = false;

        while let Some(current) = queue.pop_front() {
            if current == *to {
                found = true;
                break;
            }

            let current_edges = self.edges(&current, EdgeDirection::Outgoing).await?;

            for edge in current_edges {
                if visited.insert(edge.to) {
                    parent.insert(edge.to, current);
                    queue.push_back(edge.to);
                }
            }
        }

        if !found {
            return Ok(None);
        }

        // Reconstruct path from `to` back to `from`
        let mut path_ids = Vec::new();
        let mut current = *to;
        path_ids.push(current);
        while current != *from {
            match parent.get(&current) {
                Some(&p) => {
                    path_ids.push(p);
                    current = p;
                }
                None => return Ok(None), // Should not happen
            }
        }
        path_ids.reverse();

        // Look up entries
        let mut entries = Vec::new();
        for id in path_ids {
            if let Some(entry) = self.get(&id).await? {
                entries.push(entry);
            }
        }

        Ok(Some(entries))
    }

    async fn remove_edges(&self, entry_id: &uuid::Uuid) -> Result<()> {
        let id_str = entry_id.to_string();

        self.db
            .query("DELETE FROM edges WHERE from_id = $id OR to_id = $id;")
            .bind(("id", id_str))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to remove edges: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{GraphStore, QueryableStore, TemporalStore};
    use corvia_common::types::EdgeDirection;

    /// Embedding dimension for SurrealDB tests.
    /// MUST match the dimension used by e2e tests (768) because SurrealDB v3 has a
    /// server-level HNSW caching bug: once an HNSW index is created with dimension N,
    /// all subsequent indexes with the same name on the same server enforce that
    /// dimension regardless of the schema definition.
    const TEST_DIM: usize = 768;

    /// Create a 768-dim test embedding with distinctive values at the first 3 positions.
    /// The remaining positions are zero. This lets us write readable test vectors like
    /// `test_embedding(1.0, 0.0, 0.0)` while satisfying the 768-dim HNSW index.
    fn test_embedding(x: f32, y: f32, z: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; TEST_DIM];
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v
    }

    /// Per-process unique ID to isolate concurrent test runs against the same SurrealDB.
    /// Uses std::sync::LazyLock so the ID is computed once and shared across all tests
    /// in the same binary invocation, but different binary invocations get different IDs.
    static TEST_RUN_ID: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        format!("{:x}", std::process::id())
    });

    /// Each test gets its own SurrealDB database to avoid parallel test interference.
    /// The database name includes a per-process run ID so concurrent `cargo test`
    /// invocations against the same SurrealDB server don't collide.
    async fn connect_test_store(db_suffix: &str) -> Result<SurrealStore> {
        let db_name = format!("test_{db_suffix}_{}", *TEST_RUN_ID);
        let store = SurrealStore::connect(
            "127.0.0.1:8000", "test", &db_name, "root", "root", TEST_DIM,
        ).await?;
        // Clean slate: drop tables then recreate schema
        let _ = store.db.query("REMOVE TABLE IF EXISTS knowledge;").await;
        let _ = store.db.query("REMOVE TABLE IF EXISTS edges;").await;
        store.init_schema().await?;
        Ok(store)
    }

    #[tokio::test]
    async fn test_surreal_as_of() {
        let store = match connect_test_store("as_of").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let entry = KnowledgeEntry::new(
            "temporal test".into(),
            "test-temporal".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let id = entry.id;
        let valid_from = entry.valid_from;
        store.insert(&entry).await.unwrap();

        let results = store
            .as_of(
                "test-temporal",
                valid_from + chrono::Duration::seconds(1),
                10,
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);

        // Cleanup
        store.delete_scope("test-temporal").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_history() {
        let store = match connect_test_store("history").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new(
            "version 1".into(),
            "test-temporal".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e1_id = e1.id;
        store.insert(&e1).await.unwrap();

        let e2 = KnowledgeEntry::new(
            "version 2".into(),
            "test-temporal".into(),
            "v2".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2_id = e2.id;
        store.insert(&e2).await.unwrap();

        // Manually update e1 to be superseded by e2
        store
            .db
            .query(
                "UPDATE type::record('knowledge', $id) \
                 SET superseded_by = $new_id, valid_to = time::now()",
            )
            .bind(("id", e1_id.to_string()))
            .bind(("new_id", e2_id.to_string()))
            .await
            .unwrap();

        let history = store.history(&e2_id).await.unwrap();
        assert!(history.len() >= 1); // At least the entry itself
        assert_eq!(history[0].id, e2_id);

        store.delete_scope("test-temporal").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_evolution() {
        let store = match connect_test_store("evolution").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let before = chrono::Utc::now();
        let e1 = KnowledgeEntry::new(
            "evolution test".into(),
            "test-temporal".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        store.insert(&e1).await.unwrap();
        let after = chrono::Utc::now();

        let results = store.evolution("test-temporal", before, after).await.unwrap();
        assert!(results.len() >= 1);

        store.delete_scope("test-temporal").await.unwrap();
    }

    // ── Graph tests ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_surreal_relate_and_edges() {
        let store = match connect_test_store("edges").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("module A".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("module B".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();

        let out_edges = store.edges(&e1.id, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].relation, "imports");
        assert_eq!(out_edges[0].to, e2.id);

        let in_edges = store.edges(&e2.id, EdgeDirection::Incoming).await.unwrap();
        assert_eq!(in_edges.len(), 1);
        assert_eq!(in_edges[0].from, e1.id);

        // Both direction
        let both = store.edges(&e1.id, EdgeDirection::Both).await.unwrap();
        assert_eq!(both.len(), 1);

        store.delete_scope("test-graph").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_remove_edges() {
        let store = match connect_test_store("rm_edges").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("node A".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("node B".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();
        assert_eq!(
            store.edges(&e1.id, EdgeDirection::Outgoing).await.unwrap().len(),
            1
        );

        store.remove_edges(&e1.id).await.unwrap();
        assert_eq!(
            store.edges(&e1.id, EdgeDirection::Outgoing).await.unwrap().len(),
            0
        );
        // Incoming edge on e2 should also be gone
        assert_eq!(
            store.edges(&e2.id, EdgeDirection::Incoming).await.unwrap().len(),
            0
        );

        store.delete_scope("test-graph").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_traverse() {
        let store = match connect_test_store("traverse").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("module A".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("module B".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        let e3 = KnowledgeEntry::new("module C".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 0.0, 1.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        store.insert(&e3).await.unwrap();

        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();
        store.relate(&e2.id, "imports", &e3.id, None).await.unwrap();

        // Traverse from e1, depth 2 — should reach e2 and e3
        let results = store
            .traverse(&e1.id, None, EdgeDirection::Outgoing, 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // Traverse depth 1 — only e2
        let results = store
            .traverse(&e1.id, None, EdgeDirection::Outgoing, 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "module B");

        // Traverse with relation filter
        store.relate(&e1.id, "calls", &e3.id, None).await.unwrap();
        let results = store
            .traverse(&e1.id, Some("calls"), EdgeDirection::Outgoing, 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "module C");

        store.delete_scope("test-graph").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_shortest_path() {
        let store = match connect_test_store("path").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("module A".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("module B".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        let e3 = KnowledgeEntry::new("module C".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 0.0, 1.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        store.insert(&e3).await.unwrap();

        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();
        store.relate(&e2.id, "calls", &e3.id, None).await.unwrap();

        // Path from e1 to e3: e1 -> e2 -> e3
        let path = store.shortest_path(&e1.id, &e3.id).await.unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].content, "module A");
        assert_eq!(path[1].content, "module B");
        assert_eq!(path[2].content, "module C");

        // No path in reverse (directed graph)
        let reverse = store.shortest_path(&e3.id, &e1.id).await.unwrap();
        assert!(reverse.is_none());

        store.delete_scope("test-graph").await.unwrap();
    }

    #[tokio::test]
    async fn test_surreal_relate_with_metadata() {
        let store = match connect_test_store("meta").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: SurrealDB not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("src A".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("src B".into(), "test-graph".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        let meta = serde_json::json!({"weight": 0.95, "kind": "strong"});
        store
            .relate(&e1.id, "depends_on", &e2.id, Some(meta.clone()))
            .await
            .unwrap();

        let edges = store.edges(&e1.id, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, "depends_on");
        // Metadata should be present
        assert!(edges[0].metadata.is_some());
        let m = edges[0].metadata.as_ref().unwrap();
        assert_eq!(m.get("kind").and_then(|v| v.as_str()), Some("strong"));

        store.delete_scope("test-graph").await.unwrap();
    }
}
