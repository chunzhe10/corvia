use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{EdgeDirection, EntryMetadata, GraphEdge, KnowledgeEntry, SearchResult};
use pgvector::Vector;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use tracing::{info, warn};
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

    /// Search including cold-tier entries (may fall back to sequential scan).
    pub async fn search_with_cold(
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
                 AND tier IN ('hot', 'warm', 'cold')
               ORDER BY embedding <=> $1
               LIMIT $3"#,
        )
        .bind(&query_vec)
        .bind(scope_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Cold search failed: {e}")))?;

        let results = rows
            .iter()
            .filter_map(|row| {
                let distance: f64 = row.try_get("distance").ok()?;
                let score = 1.0 - distance as f32;
                let entry = row_to_entry(row)?;
                let tier = entry.tier;
                let retention_score = entry.retention_score;
                Some(SearchResult { entry, score, tier, retention_score })
            })
            .collect();

        Ok(results)
    }

    /// Transition an entry to a new tier, updating tier_changed_at.
    pub async fn update_tier(&self, entry_id: &Uuid, new_tier: corvia_common::types::Tier) -> Result<()> {
        let tier_str = new_tier.to_string();
        sqlx::query(
            "UPDATE knowledge SET tier = $1, tier_changed_at = NOW() WHERE id = $2",
        )
        .bind(&tier_str)
        .bind(entry_id)
        .execute(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to update tier: {e}")))?;

        Ok(())
    }

    /// Query pinned entries in a scope.
    pub async fn get_pinned(&self, scope_id: &str) -> Result<Vec<KnowledgeEntry>> {
        let rows = sqlx::query(
            "SELECT * FROM knowledge WHERE scope_id = $1 AND pinned = true",
        )
        .bind(scope_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to query pinned entries: {e}")))?;

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
    let last_accessed: Option<chrono::DateTime<chrono::Utc>> =
        row.try_get("last_accessed").ok().flatten();
    let access_count: i32 = row.try_get("access_count").unwrap_or(0);

    // Tiered knowledge fields
    let memory_type_str: String = row
        .try_get("memory_type")
        .unwrap_or_else(|_| "episodic".into());
    let memory_type = match memory_type_str.as_str() {
        "structural" => corvia_common::types::MemoryType::Structural,
        "decisional" => corvia_common::types::MemoryType::Decisional,
        "analytical" => corvia_common::types::MemoryType::Analytical,
        "procedural" => corvia_common::types::MemoryType::Procedural,
        _ => corvia_common::types::MemoryType::Episodic,
    };
    let confidence: Option<f64> = row.try_get("confidence").ok().flatten();
    let tier_str: String = row.try_get("tier").unwrap_or_else(|_| "hot".into());
    let tier = match tier_str.as_str() {
        "warm" => corvia_common::types::Tier::Warm,
        "cold" => corvia_common::types::Tier::Cold,
        "forgotten" => corvia_common::types::Tier::Forgotten,
        _ => corvia_common::types::Tier::Hot,
    };
    let tier_changed_at: Option<chrono::DateTime<chrono::Utc>> =
        row.try_get("tier_changed_at").ok().flatten();
    let retention_score_f64: Option<f64> = row.try_get("retention_score").ok().flatten();
    let pinned: bool = row.try_get("pinned").unwrap_or(false);
    let pin = if pinned {
        let pinned_by: String = row.try_get("pinned_by").unwrap_or_else(|_| String::new());
        let pinned_at: chrono::DateTime<chrono::Utc> = row
            .try_get("pinned_at")
            .unwrap_or_else(|_| chrono::Utc::now());
        Some(corvia_common::types::PinInfo {
            by: pinned_by,
            at: pinned_at,
        })
    } else {
        None
    };

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
        memory_type,
        confidence: confidence.map(|c| c as f32),
        last_accessed,
        access_count: access_count as u32,
        tier,
        tier_changed_at,
        retention_score: retention_score_f64.map(|r| r as f32),
        pin,
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
                entry_status TEXT NOT NULL DEFAULT 'merged',
                last_accessed TIMESTAMPTZ,
                access_count INTEGER NOT NULL DEFAULT 0
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
            CREATE INDEX IF NOT EXISTS idx_knowledge_source_version ON knowledge(scope_id, source_version);
            "#
        );

        sqlx::query(&schema)
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to init schema: {e}")))?;

        // Idempotent tiered-knowledge column migrations.
        let tiered_columns = [
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS memory_type TEXT NOT NULL DEFAULT 'episodic'",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS confidence FLOAT",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'hot'",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS tier_changed_at TIMESTAMPTZ",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS retention_score FLOAT",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT false",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS pinned_by TEXT",
            "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS pinned_at TIMESTAMPTZ",
        ];
        for ddl in &tiered_columns {
            sqlx::query(ddl)
                .execute(&self.pool)
                .await
                .map_err(|e| CorviaError::Storage(format!("Failed to add tiered column: {e}")))?;
        }

        // Replace the full HNSW index with a partial index on active tiers.
        // DROP + CREATE is needed because CREATE INDEX IF NOT EXISTS won't update
        // an existing index's WHERE clause.
        sqlx::query("DROP INDEX IF EXISTS idx_knowledge_embedding")
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to drop old HNSW index: {e}")))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_embedding_active ON knowledge \
             USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64) \
             WHERE tier IN ('hot', 'warm')"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to create partial HNSW index: {e}")))?;

        // Tier index for fast scope+tier filtering.
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_tier ON knowledge(scope_id, tier)"
        )
        .execute(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to create tier index: {e}")))?;

        info!("PostgreSQL schema initialized (embedding dim={dim})");
        Ok(())
    }

    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()> {
        let embedding: Option<Vector> = entry
            .embedding
            .as_ref()
            .map(|v| Vector::from(v.clone()));

        let metadata = serde_json::to_value(&entry.metadata).unwrap_or_default();
        let entry_status = serde_json::to_value(entry.entry_status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "merged".into());

        let memory_type_str = entry.memory_type.to_string();
        let tier_str = entry.tier.to_string();
        let confidence_f64 = entry.confidence.map(|c| c as f64);
        let retention_score_f64 = entry.retention_score.map(|r| r as f64);
        let pinned = entry.pin.is_some();
        let pinned_by = entry.pin.as_ref().map(|p| p.by.clone());
        let pinned_at = entry.pin.as_ref().map(|p| p.at);

        sqlx::query(
            r#"INSERT INTO knowledge
               (id, content, source_version, scope_id, workstream,
                recorded_at, valid_from, valid_to, superseded_by,
                embedding, metadata, agent_id, session_id, entry_status,
                memory_type, confidence, last_accessed, access_count,
                tier, tier_changed_at, retention_score,
                pinned, pinned_by, pinned_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                       $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)"#,
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
        .bind(&memory_type_str)
        .bind(confidence_f64)
        .bind(entry.last_accessed)
        .bind(entry.access_count as i32)
        .bind(&tier_str)
        .bind(entry.tier_changed_at)
        .bind(retention_score_f64)
        .bind(pinned)
        .bind(&pinned_by)
        .bind(pinned_at)
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
        // Default search: only Hot + Warm entries (uses partial HNSW index).
        let query_vec = Vector::from(embedding.to_vec());

        let rows = sqlx::query(
            r#"SELECT *, embedding <=> $1 AS distance
               FROM knowledge
               WHERE scope_id = $2 AND embedding IS NOT NULL
                 AND tier IN ('hot', 'warm')
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
                let tier = entry.tier;
                let retention_score = entry.retention_score;
                Some(SearchResult { entry, score, tier, retention_score })
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

    async fn get_by_source_version(
        &self,
        scope_id: &str,
        source_version: &str,
    ) -> Result<Option<KnowledgeEntry>> {
        if source_version.is_empty() {
            return Ok(None);
        }
        let row = sqlx::query(
            "SELECT * FROM knowledge WHERE scope_id = $1 AND source_version = $2 LIMIT 1",
        )
        .bind(scope_id)
        .bind(source_version)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to get by source_version: {e}")))?;

        Ok(row.as_ref().and_then(row_to_entry))
    }

    #[tracing::instrument(name = "corvia.access.record", skip(self), fields(count = entry_ids.len()))]
    async fn record_access(&self, entry_ids: &[uuid::Uuid]) -> Result<()> {
        if entry_ids.is_empty() {
            return Ok(());
        }

        // SQL integer addition is safe here: access_count is INTEGER (max 2^31-1).
        // Overflow is unreachable in practice (would require billions of accesses per entry).
        if let Err(e) = sqlx::query(
            r#"UPDATE knowledge
               SET last_accessed = NOW(),
                   access_count = access_count + 1
               WHERE id = ANY($1)"#,
        )
        .bind(entry_ids)
        .execute(&self.pool)
        .await
        {
            warn!(error = %e, "Failed to record access in PostgresStore");
        }

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
                if let Some(rel) = relation
                    && edge.relation != rel
                {
                    continue;
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

    #[tokio::test]
    async fn test_postgres_get_by_source_version() {
        let store = match connect_test_store("sv_lookup").await {
            Ok(s) => s,
            Err(_) => return, // PostgreSQL not available
        };

        let entry = KnowledgeEntry::new(
            "session turn 1".into(),
            "user-history".into(),
            "ses-pg-abc:turn-1".into(),
        )
        .with_embedding(test_embedding(0.1, 0.2, 0.3));
        let expected_id = entry.id;
        store.insert(&entry).await.unwrap();

        // Found by exact scope + source_version
        let result = store
            .get_by_source_version("user-history", "ses-pg-abc:turn-1")
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, expected_id);

        // Not found: wrong scope
        let result = store
            .get_by_source_version("other-scope", "ses-pg-abc:turn-1")
            .await
            .unwrap();
        assert!(result.is_none());

        // Not found: wrong source_version
        let result = store
            .get_by_source_version("user-history", "ses-nonexistent:turn-1")
            .await
            .unwrap();
        assert!(result.is_none());

        store.delete_scope("user-history").await.unwrap();
        cleanup_test_db("sv_lookup").await;
    }

    #[tokio::test]
    async fn test_pg_tiered_columns_defaults() {
        let store = match connect_test_store("tiered_defaults").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let entry = KnowledgeEntry::new("tiered test".into(), "test-tiered".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let id = entry.id;
        store.insert(&entry).await.unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.memory_type, corvia_common::types::MemoryType::Episodic);
        assert_eq!(got.tier, corvia_common::types::Tier::Hot);
        assert_eq!(got.access_count, 0);
        assert!(got.confidence.is_none());
        assert!(got.tier_changed_at.is_none());
        assert!(got.retention_score.is_none());
        assert!(got.pin.is_none());

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("tiered_defaults").await;
    }

    #[tokio::test]
    async fn test_pg_insert_tiered_fields_roundtrip() {
        let store = match connect_test_store("tiered_roundtrip").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let mut entry = KnowledgeEntry::new(
            "decisional entry".into(),
            "test-tiered".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        entry.memory_type = corvia_common::types::MemoryType::Decisional;
        entry.confidence = Some(0.85);
        entry.tier = corvia_common::types::Tier::Warm;
        entry.tier_changed_at = Some(chrono::Utc::now());
        entry.retention_score = Some(0.72);
        entry.pin = Some(corvia_common::types::PinInfo {
            by: "test-agent".into(),
            at: chrono::Utc::now(),
        });
        let id = entry.id;
        store.insert(&entry).await.unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.memory_type, corvia_common::types::MemoryType::Decisional);
        assert!((got.confidence.unwrap() - 0.85).abs() < 0.01);
        assert_eq!(got.tier, corvia_common::types::Tier::Warm);
        assert!(got.tier_changed_at.is_some());
        assert!((got.retention_score.unwrap() - 0.72).abs() < 0.01);
        assert!(got.pin.is_some());
        assert_eq!(got.pin.unwrap().by, "test-agent");

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("tiered_roundtrip").await;
    }

    #[tokio::test]
    async fn test_pg_search_filters_by_tier() {
        let store = match connect_test_store("tier_search").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        // Insert a hot entry and a cold entry
        let mut hot_entry = KnowledgeEntry::new(
            "hot content".into(),
            "test-tiered".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        hot_entry.tier = corvia_common::types::Tier::Hot;

        let mut cold_entry = KnowledgeEntry::new(
            "cold content".into(),
            "test-tiered".into(),
            "v2".into(),
        )
        .with_embedding(test_embedding(0.9, 0.1, 0.0));
        cold_entry.tier = corvia_common::types::Tier::Cold;

        store.insert(&hot_entry).await.unwrap();
        store.insert(&cold_entry).await.unwrap();

        // Default search should only return the hot entry
        let results = store
            .search(&test_embedding(1.0, 0.0, 0.0), "test-tiered", 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.content, "hot content");

        // Cold search should return both
        let results = store
            .search_with_cold(&test_embedding(1.0, 0.0, 0.0), "test-tiered", 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("tier_search").await;
    }

    #[tokio::test]
    async fn test_pg_update_tier() {
        let store = match connect_test_store("update_tier").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let entry = KnowledgeEntry::new("tier test".into(), "test-tiered".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let id = entry.id;
        store.insert(&entry).await.unwrap();

        // Verify starts as Hot
        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.tier, corvia_common::types::Tier::Hot);
        assert!(got.tier_changed_at.is_none());

        // Transition to Warm
        store
            .update_tier(&id, corvia_common::types::Tier::Warm)
            .await
            .unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.tier, corvia_common::types::Tier::Warm);
        assert!(got.tier_changed_at.is_some());

        // Transition to Cold — should be excluded from default search
        store
            .update_tier(&id, corvia_common::types::Tier::Cold)
            .await
            .unwrap();

        let results = store
            .search(&test_embedding(1.0, 0.0, 0.0), "test-tiered", 10)
            .await
            .unwrap();
        assert!(results.is_empty());

        // But visible in cold search
        let results = store
            .search_with_cold(&test_embedding(1.0, 0.0, 0.0), "test-tiered", 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.tier, corvia_common::types::Tier::Cold);

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("update_tier").await;
    }

    #[tokio::test]
    async fn test_pg_pinned_entries() {
        let store = match connect_test_store("pinned").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let mut pinned_entry = KnowledgeEntry::new(
            "pinned content".into(),
            "test-tiered".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        pinned_entry.pin = Some(corvia_common::types::PinInfo {
            by: "admin".into(),
            at: chrono::Utc::now(),
        });

        let unpinned_entry = KnowledgeEntry::new(
            "unpinned content".into(),
            "test-tiered".into(),
            "v2".into(),
        )
        .with_embedding(test_embedding(0.0, 1.0, 0.0));

        store.insert(&pinned_entry).await.unwrap();
        store.insert(&unpinned_entry).await.unwrap();

        let pinned = store.get_pinned("test-tiered").await.unwrap();
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].content, "pinned content");
        assert_eq!(pinned[0].pin.as_ref().unwrap().by, "admin");

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("pinned").await;
    }

    #[tokio::test]
    async fn test_pg_record_access_updates() {
        let store = match connect_test_store("access_track").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let entry = KnowledgeEntry::new("access test".into(), "test-tiered".into(), "v1".into())
            .with_embedding(test_embedding(1.0, 0.0, 0.0));
        let id = entry.id;
        store.insert(&entry).await.unwrap();

        // Verify initial state
        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.access_count, 0);
        assert!(got.last_accessed.is_none());

        // Record access
        store.record_access(&[id]).await.unwrap();

        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.access_count, 1);
        assert!(got.last_accessed.is_some());

        // Record access again
        store.record_access(&[id]).await.unwrap();
        let got = store.get(&id).await.unwrap().unwrap();
        assert_eq!(got.access_count, 2);

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("access_track").await;
    }

    #[tokio::test]
    async fn test_pg_warm_entries_in_default_search() {
        let store = match connect_test_store("warm_search").await {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: PostgreSQL not available");
                return;
            }
        };

        let mut warm_entry = KnowledgeEntry::new(
            "warm content".into(),
            "test-tiered".into(),
            "v1".into(),
        )
        .with_embedding(test_embedding(1.0, 0.0, 0.0));
        warm_entry.tier = corvia_common::types::Tier::Warm;
        store.insert(&warm_entry).await.unwrap();

        // Warm entries should appear in default search
        let results = store
            .search(&test_embedding(1.0, 0.0, 0.0), "test-tiered", 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.tier, corvia_common::types::Tier::Warm);

        store.delete_scope("test-tiered").await.unwrap();
        cleanup_test_db("warm_search").await;
    }
}
