use corvia_common::errors::Result;
use corvia_common::types::KnowledgeEntry;
use std::path::Path;
use std::sync::Arc;

use crate::staging::StagingManager;
use crate::traits::{InferenceEngine, QueryableStore};

/// Atomic write path per D45 Part 4.
/// 1. Write JSON to staging dir (if registered agent) — filesystem, recoverable
/// 2. Embed content via engine — idempotent
/// 3. Insert into store (Redb + HNSW)
pub struct AgentWriter {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub staging: StagingManager,
}

impl AgentWriter {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        engine: Arc<dyn InferenceEngine>,
        staging: StagingManager,
    ) -> Self {
        Self { store, engine, staging }
    }

    /// Write a knowledge entry with agent attribution.
    ///
    /// - Creates a `KnowledgeEntry` tagged with agent_id/session_id and `EntryStatus::Pending`
    /// - Embeds the content
    /// - Writes to staging dir (if provided — Registered agents only)
    /// - Inserts into the shared store (immediately searchable via HNSW)
    pub async fn write(
        &self,
        content: &str,
        scope_id: &str,
        source_version: &str,
        agent_id: &str,
        session_id: &str,
        staging_dir: Option<&Path>,
        content_role: Option<String>,
        source_origin: Option<String>,
    ) -> Result<KnowledgeEntry> {
        // 1. Create entry with agent attribution
        let mut entry = KnowledgeEntry::new(
            content.to_string(),
            scope_id.to_string(),
            source_version.to_string(),
        )
        .with_agent(agent_id.to_string(), session_id.to_string());

        // 1b. Set docs-workflow metadata if provided
        entry.metadata.content_role = content_role;
        entry.metadata.source_origin = source_origin;

        // 2. Embed content
        let embedding = self.engine.embed(content).await?;
        let entry = entry.with_embedding(embedding);

        // 3. Write staging file (if Registered agent with staging dir)
        if let Some(dir) = staging_dir {
            self.staging.write_staging_file(dir, &entry)?;
        }

        // 4. Insert into store (HNSW + Redb — immediately searchable)
        self.store.insert(&entry).await?;

        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::traits::QueryableStore;
    use corvia_common::agent_types::EntryStatus;

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

    #[tokio::test]
    async fn test_write_entry_for_registered_agent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine);
        let staging = StagingManager::new(dir.path());
        let writer = AgentWriter::new(
            Arc::new(store),
            engine,
            staging,
        );

        let staging_dir = writer.staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = writer.write(
            "test knowledge",
            "test-scope",
            "v1",
            "test::agent",
            "test::agent/sess-abc",
            Some(&staging_dir),
            None,
            None,
        ).await.unwrap();

        assert_eq!(entry.agent_id, Some("test::agent".into()));
        assert_eq!(entry.entry_status, EntryStatus::Pending);
        // Entry is in HNSW (searchable immediately)
        let results = writer.store.search(&[1.0, 0.0, 0.0], "test-scope", 5).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_write_entry_for_mcp_agent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine);
        let staging = StagingManager::new(dir.path());
        let writer = AgentWriter::new(
            Arc::new(store),
            engine,
            staging,
        );

        // MCP agents have no staging dir
        let entry = writer.write(
            "mcp knowledge",
            "test-scope",
            "v1",
            "crewai::advisor",
            "crewai::advisor/sess-xyz",
            None,
            None,
            None,
        ).await.unwrap();

        assert_eq!(entry.agent_id, Some("crewai::advisor".into()));
    }
}
