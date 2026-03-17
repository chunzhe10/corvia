use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EdgeDirection, EntryMetadata, GraphEdge, KnowledgeEntry, SearchResult};
use pgvector::Vector;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use tracing::info;
use uuid::Uuid;

pub struct PostgresStore {
    pool: PgPool,
    dimensions: usize,
}

impl PostgresStore {
    pub async fn connect(url: &str, dimensions: usize) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to connect to PostgreSQL: {e}")))?;

        info!("Connected to PostgreSQL at {url}");
        Ok(Self { pool, dimensions })
    }

    /// Fetch all knowledge entries (used for migration export).
    pub async fn fetch_all_entries(&self) -> Result<Vec<KnowledgeEntry>> {
        let rows = sqlx::query("SELECT * FROM knowledge")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to fetch all entries: {e}")))?;

        Ok(rows.iter().filter_map(row_to_entry).collect())
    }

    /// Fetch all graph edges (used for migration export).
    pub async fn fetch_all_edges(&self) -> Result<Vec<GraphEdge>> {
        let rows = sqlx::query("SELECT from_id, to_id, relation, metadata FROM edges")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to fetch all edges: {e}")))?;

        let mut edges = Vec::new();
        for row in &rows {
            let from: Uuid = match row.try_get("from_id") {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Skipping edge with invalid from_id: {e}");
                    continue;
                }
            };
            let to: Uuid = match row.try_get("to_id") {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Skipping edge with invalid to_id: {e}");
                    continue;
                }
            };
            let relation: String = row.try_get("relation").unwrap_or_default();
            let metadata: Option<serde_json::Value> = row.try_get("metadata").ok().flatten();

            edges.push(GraphEdge {
                from,
                to,
                relation,
                metadata,
            });
        }

        Ok(edges)
    }
}

/// Map a sqlx Row to a KnowledgeEntry.
fn row_to_entry(row: &sqlx::postgres::PgRow) -> Option<KnowledgeEntry> {
    use sqlx::Row;

    let id: Uuid = row.try_get("id").ok()?;
    let content: String = row.try_get("content").ok()?;
    let source_version: String = row.try_get("source_version").ok().unwrap_or_default();
    let scope_id: String = row.try_get("scope_id").ok()?;
    let workstream: String = row.try_get("workstream").unwrap_or_else(|_| "main".into());
    let recorded_at: chrono::DateTime<chrono::Utc> = row.try_get("recorded_at").ok()?;
    let valid_from: chrono::DateTime<chrono::Utc> = row.try_get("valid_from").ok()?;
    let valid_to: Option<chrono::DateTime<chrono::Utc>> = row.try_get("valid_to").ok().flatten();
    let superseded_by: Option<Uuid> = row.try_get("superseded_by").ok().flatten();
    let embedding_vec: Option<Vector> = row.try_get("embedding").ok().flatten();
    let embedding = embedding_vec.map(|v| v.to_vec());
    let metadata: serde_json::Value = row
        .try_get("metadata")
        .ok()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    let metadata: EntryMetadata =
        serde_json::from_value(metadata).unwrap_or_default();
    let agent_id: Option<String> = row.try_get("agent_id").ok().flatten();
    let session_id: Option<String> = row.try_get("session_id").ok().flatten();
    let entry_status_str: String = row
        .try_get("entry_status")
        .unwrap_or_else(|_| "merged".into());
    let entry_status: corvia_common::agent_types::EntryStatus =
        serde_json::from_value(serde_json::Value::String(entry_status_str))
            .unwrap_or_default();

    Some(KnowledgeEntry {
        id,
        content,
        source_version,
        scope_id,
        workstream,
        recorded_at,
        valid_from,
        valid_to,
        superseded_by,
        embedding,
        metadata,
        agent_id,
        session_id,
        entry_status,
    })
}

#[async_trait]
impl crate::traits::QueryableStore for PostgresStore {
    async fn init_schema(&self) -> Result<()> {
        let dim = self.dimensions;
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create vector extension: {e}")))?;

        let schema = format!(
            r#"
            CREATE TABLE IF NOT EXISTS knowledge (
                id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                source_version TEXT NOT NULL DEFAULT '',
                scope_id TEXT NOT NULL,
                workstream TEXT NOT NULL DEFAULT '',
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_to TIMESTAMPTZ,
                superseded_by UUID,
                embedding vector({dim}),
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                agent_id TEXT,
                session_id TEXT,
                entry_status TEXT NOT NULL DEFAULT 'merged'
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge(scope_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_valid ON knowledge(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS edges (
                from_id UUID NOT NULL,
                relation TEXT NOT NULL,
                to_id UUID NOT NULL,
                metadata JSONB,
                PRIMARY KEY (from_id, relation, to_id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
            "#
        );

        sqlx::query(&schema)
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to init schema: {e}")))?;

        // HNSW index creation must be separate (CREATE INDEX IF NOT EXISTS with USING
        // requires its own statement in some pg versions).
        let hnsw = format!(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge \
             USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        );
        sqlx::query(&hnsw)
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create HNSW index: {e}")))?;

        info!("PostgreSQL schema initialized (embedding dim={dim})");
        Ok(())
    }

    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()> {
        let embedding: Option<Vector> = entry
            .embedding
            .as_ref()
            .map(|v| Vector::from(v.clone()));

        let metadata = serde_json::to_value(&entry.metadata).unwrap_or_default();
        let entry_status = serde_json::to_value(&entry.entry_status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "merged".into());

        sqlx::query(
            r#"INSERT INTO knowledge
               (id, content, source_version, scope_id, workstream,
                recorded_at, valid_from, valid_to, superseded_by,
                embedding, metadata, agent_id, session_id, entry_status)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)"#,
        )
        .bind(entry.id)
        .bind(&entry.content)
        .bind(&entry.source_version)
        .bind(&entry.scope_id)
        .bind(&entry.workstream)
        .bind(entry.recorded_at)
        .bind(entry.valid_from)
        .bind(entry.valid_to)
        .bind(entry.superseded_by)
        .bind(embedding)
        .bind(metadata)
        .bind(&entry.agent_id)
        .bind(&entry.session_id)
        .bind(&entry_status)
        .execute(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to insert entry: {e}")))?;

        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        scope_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_vec = Vector::from(embedding.to_vec());

        let rows = sqlx::query(
            r#"SELECT *, embedding <=> $1 AS distance
               FROM knowledge
               WHERE scope_id = $2 AND embedding IS NOT NULL
               ORDER BY embedding <=> $1
               LIMIT $3"#,
        )
        .bind(&query_vec)
        .bind(scope_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Search failed: {e}")))?;

        let results = rows
            .iter()
            .filter_map(|row| {
                let distance: f64 = row.try_get("distance").ok()?;
                let score = 1.0 - distance as f32;
                let entry = row_to_entry(row)?;
                Some(SearchResult { entry, score })
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, id: &Uuid) -> Result<Option<KnowledgeEntry>> {
        let row = sqlx::query("SELECT * FROM knowledge WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to get entry: {e}")))?;

        Ok(row.as_ref().and_then(row_to_entry))
    }

    async fn count(&self, scope_id: &str) -> Result<u64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM knowledge WHERE scope_id = $1")
            .bind(scope_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Count failed: {e}")))?;

        let count: i64 = row.try_get("count").unwrap_or(0);
        Ok(count as u64)
    }

    async fn delete_scope(&self, scope_id: &str) -> Result<()> {
        // Delete edges referencing entries in this scope
        sqlx::query(
            "DELETE FROM edges WHERE from_id IN (SELECT id FROM knowledge WHERE scope_id = $1) \
             OR to_id IN (SELECT id FROM knowledge WHERE scope_id = $1)",
        )
        .bind(scope_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to delete scope edges: {e}")))?;

        sqlx::query("DELETE FROM knowledge WHERE scope_id = $1")
            .bind(scope_id)
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to delete scope: {e}")))?;

        info!("Deleted all entries for scope '{scope_id}'");
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl crate::traits::TemporalStore for PostgresStore {
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        let rows = sqlx::query(
            r#"SELECT * FROM knowledge
               WHERE scope_id = $1
               AND valid_from <= $2
               AND (valid_to IS NULL OR valid_to > $2)
               LIMIT $3"#,
        )
        .bind(scope_id)
        .bind(timestamp)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("as_of query failed: {e}")))?;

        Ok(rows.iter().filter_map(row_to_entry).collect())
    }

    async fn history(&self, entry_id: &Uuid) -> Result<Vec<KnowledgeEntry>> {
        // Recursive CTE: walk the superseded_by chain starting from the given entry.
        // First walk forward (find entries this one was superseded by),
        // then walk backward (find entries that were superseded to reach this one).
        //
        // Walk backward (find predecessors): start from entry_id, then walk backward via superseded_by.
        let mut chain = Vec::new();

        // Start with the given entry
        let start = crate::traits::QueryableStore::get(self, entry_id)
            .await?
            .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", entry_id)))?;
        chain.push(start);

        // Walk backward: find entries whose superseded_by matches current
        let mut current_id = *entry_id;
        loop {
            let row = sqlx::query(
                "SELECT * FROM knowledge WHERE superseded_by = $1",
            )
            .bind(current_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("history query failed: {e}")))?;

            match row.as_ref().and_then(row_to_entry) {
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
        let rows = sqlx::query(
            r#"SELECT * FROM knowledge
               WHERE scope_id = $1
               AND valid_from >= $2
               AND valid_from <= $3"#,
        )
        .bind(scope_id)
        .bind(from)
        .bind(to)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("evolution query failed: {e}")))?;

        Ok(rows.iter().filter_map(row_to_entry).collect())
    }
}

#[async_trait]
impl crate::traits::GraphStore for PostgresStore {
    async fn relate(
        &self,
        from: &Uuid,
        relation: &str,
        to: &Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO edges (from_id, relation, to_id, metadata) \
             VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
        )
        .bind(from)
        .bind(relation)
        .bind(to)
        .bind(&metadata)
        .execute(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to create edge: {e}")))?;

        Ok(())
    }

    async fn edges(
        &self,
        entry_id: &Uuid,
        direction: EdgeDirection,
    ) -> Result<Vec<GraphEdge>> {
        let query = match direction {
            EdgeDirection::Outgoing => {
                "SELECT from_id, to_id, relation, metadata FROM edges WHERE from_id = $1"
            }
            EdgeDirection::Incoming => {
                "SELECT from_id, to_id, relation, metadata FROM edges WHERE to_id = $1"
            }
            EdgeDirection::Both => {
                "SELECT from_id, to_id, relation, metadata FROM edges \
                 WHERE from_id = $1 OR to_id = $1"
            }
        };

        let rows = sqlx::query(query)
            .bind(entry_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to query edges: {e}")))?;

        let mut edges = Vec::new();
        for row in &rows {
            let from: Uuid = match row.try_get("from_id") {
                Ok(v) => v,
                Err(_) => continue,
            };
            let to: Uuid = match row.try_get("to_id") {
                Ok(v) => v,
                Err(_) => continue,
            };
            let relation: String = row.try_get("relation").unwrap_or_default();
            let metadata: Option<serde_json::Value> = row.try_get("metadata").ok().flatten();

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
        start: &Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        // App-level BFS traversal
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut results = Vec::new();

        visited.insert(*start);
        queue.push_back((*start, 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let current_edges = crate::traits::GraphStore::edges(self, &current, direction).await?;

            for edge in current_edges {
                if let Some(rel) = relation {
                    if edge.relation != rel {
                        continue;
                    }
                }

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
                    if let Some(entry) = crate::traits::QueryableStore::get(self, &neighbor).await? {
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
        from: &Uuid,
        to: &Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>> {
        // BFS from source to target (directed, outgoing edges)
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent: std::collections::HashMap<Uuid, Uuid> =
            std::collections::HashMap::new();

        visited.insert(*from);
        queue.push_back(*from);

        let mut found = false;

        while let Some(current) = queue.pop_front() {
            if current == *to {
                found = true;
                break;
            }

            let current_edges =
                crate::traits::GraphStore::edges(self, &current, EdgeDirection::Outgoing).await?;

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
                None => return Ok(None),
            }
        }
        path_ids.reverse();

        let mut entries = Vec::new();
        for id in path_ids {
            if let Some(entry) = crate::traits::QueryableStore::get(self, &id).await? {
                entries.push(entry);
            }
        }

        Ok(Some(entries))
    }

    async fn remove_edges(&self, entry_id: &Uuid) -> Result<()> {
        sqlx::query("DELETE FROM edges WHERE from_id = $1 OR to_id = $1")
            .bind(entry_id)
            .execute(&self.pool)
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

    const TEST_DIM: usize = 768;

    fn test_embedding(x: f32, y: f32, z: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; TEST_DIM];
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v
    }

    static TEST_RUN_ID: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| format!("{:x}", std::process::id()));

    /// Connect to a test-specific database. Creates the database fresh, drops
    /// existing tables, and initializes the schema.
    async fn connect_test_store(suffix: &str) -> Result<PostgresStore> {
        let base_url = "postgres://corvia:corvia@127.0.0.1:5432/postgres";
        let db_name = format!("corvia_test_{}_{}", *TEST_RUN_ID, suffix);

        // Connect to the default `postgres` database to create our test DB
        let admin_pool = PgPoolOptions::new()
            .max_connections(2)
            .connect(base_url)
            .await
            .map_err(|e| CorviaError::Storage(format!("PostgreSQL not available: {e}")))?;

        // Drop and recreate the test database for a clean slate
        let _ = sqlx::query(&format!("DROP DATABASE IF EXISTS \"{db_name}\""))
            .execute(&admin_pool)
            .await;
        sqlx::query(&format!("CREATE DATABASE \"{db_name}\""))
            .execute(&admin_pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create test DB: {e}")))?;
        admin_pool.close().await;

        let test_url = format!("postgres://corvia:corvia@127.0.0.1:5432/{db_name}");
        let store = PostgresStore::connect(&test_url, TEST_DIM).await?;
        store.init_schema().await?;
        Ok(store)
    }

    /// Drop the test database during teardown.
    async fn cleanup_test_db(suffix: &str) {
        let base_url = "postgres://corvia:corvia@127.0.0.1:5432/postgres";
        let db_name = format!("corvia_test_{}_{}", *TEST_RUN_ID, suffix);
        if let Ok(pool) = PgPoolOptions::new()
            .max_connections(2)
            .connect(base_url)
            .await
        {
            let _ = sqlx::query(&format!("DROP DATABASE IF EXISTS \"{db_name}\""))
                .execute(&pool)
                .await;
            pool.close().await;
        }
    }

    #[tokio::test]
    async fn test_pg_insert_and_get() {
        let store = match connect_test_store("insert_get").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let entry = KnowledgeEntry::new("test content".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let id = entry.id;
        store.insert(&entry).await.unwrap();

        let got = store.get(&id).await.unwrap();
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.id, id);
        assert_eq!(got.content, "test content");
        assert_eq!(got.scope_id, "test-scope");
        assert_eq!(got.source_version, "v1");

        store.delete_scope("test-scope").await.unwrap();
        cleanup_test_db("insert_get").await;
    }

    #[tokio::test]
    async fn test_pg_search() {
        let store = match connect_test_store("search").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("alpha content".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("beta content".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        let results = store
            .search(&test_embedding(1.0, 0.0, 0.0), "test-scope", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());
        // The closest match should be e1
        assert_eq!(results[0].entry.content, "alpha content");

        store.delete_scope("test-scope").await.unwrap();
        cleanup_test_db("search").await;
    }

    #[tokio::test]
    async fn test_pg_count() {
        let store = match connect_test_store("count").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        assert_eq!(store.count("test-scope").await.unwrap(), 0);

        let e1 = KnowledgeEntry::new("one".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let e2 = KnowledgeEntry::new("two".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(0.0, 1.0, 0.0));
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        assert_eq!(store.count("test-scope").await.unwrap(), 2);
        assert_eq!(store.count("other-scope").await.unwrap(), 0);

        store.delete_scope("test-scope").await.unwrap();
        cleanup_test_db("count").await;
    }

    #[tokio::test]
    async fn test_pg_delete_scope() {
        let store = match connect_test_store("delete_scope").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let e1 = KnowledgeEntry::new("entry".into(), "test-scope".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        store.insert(&e1).await.unwrap();
        assert_eq!(store.count("test-scope").await.unwrap(), 1);

        store.delete_scope("test-scope").await.unwrap();
        assert_eq!(store.count("test-scope").await.unwrap(), 0);

        cleanup_test_db("delete_scope").await;
    }

    #[tokio::test]
    async fn test_pg_temporal_as_of() {
        let store = match connect_test_store("as_of").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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

        store.delete_scope("test-temporal").await.unwrap();
        cleanup_test_db("as_of").await;
    }

    #[tokio::test]
    async fn test_pg_temporal_history() {
        let store = match connect_test_store("history").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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

        // Mark e1 as superseded by e2
        sqlx::query(
            "UPDATE knowledge SET superseded_by = $1, valid_to = now() WHERE id = $2",
        )
        .bind(e2_id)
        .bind(e1_id)
        .execute(&store.pool)
        .await
        .unwrap();

        let history = store.history(&e2_id).await.unwrap();
        assert!(history.len() >= 1);
        assert_eq!(history[0].id, e2_id);

        store.delete_scope("test-temporal").await.unwrap();
        cleanup_test_db("history").await;
    }

    #[tokio::test]
    async fn test_pg_temporal_evolution() {
        let store = match connect_test_store("evolution").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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

        let results = store
            .evolution("test-temporal", before, after)
            .await
            .unwrap();
        assert!(results.len() >= 1);

        store.delete_scope("test-temporal").await.unwrap();
        cleanup_test_db("evolution").await;
    }

    #[tokio::test]
    async fn test_pg_graph_relate_and_edges() {
        let store = match connect_test_store("edges").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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

        let both = store.edges(&e1.id, EdgeDirection::Both).await.unwrap();
        assert_eq!(both.len(), 1);

        store.delete_scope("test-graph").await.unwrap();
        cleanup_test_db("edges").await;
    }

    #[tokio::test]
    async fn test_pg_graph_traverse() {
        let store = match connect_test_store("traverse").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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
        cleanup_test_db("traverse").await;
    }

    #[tokio::test]
    async fn test_pg_graph_shortest_path() {
        let store = match connect_test_store("path").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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
        cleanup_test_db("path").await;
    }

    #[tokio::test]
    async fn test_pg_graph_remove_edges() {
        let store = match connect_test_store("rm_edges").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
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
            store
                .edges(&e1.id, EdgeDirection::Outgoing)
                .await
                .unwrap()
                .len(),
            1
        );

        store.remove_edges(&e1.id).await.unwrap();
        assert_eq!(
            store
                .edges(&e1.id, EdgeDirection::Outgoing)
                .await
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            store
                .edges(&e2.id, EdgeDirection::Incoming)
                .await
                .unwrap()
                .len(),
            0
        );

        store.delete_scope("test-graph").await.unwrap();
        cleanup_test_db("rm_edges").await;
    }
}
