use corvia_common::agent_types::{AgentPermission, EntryStatus, VisibilityMode};
use corvia_common::errors::Result;
use corvia_common::types::SearchResult;
use std::sync::Arc;

use crate::traits::{InferenceEngine, QueryableStore};

/// Context builder that wraps search with visibility mode filtering and RBAC scope enforcement.
///
/// Visibility modes (D43):
/// - `Own`: Merged entries + agent's own Pending entries
/// - `All`: Merged entries + all agents' Pending entries
/// - `Explicit(agents)`: Merged entries + named agents' Pending entries
///
/// RBAC (D39): Agents can only see entries in scopes they have access to.
pub struct ContextBuilder {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
}

impl ContextBuilder {
    pub fn new(
        store: Arc<dyn QueryableStore>,
        engine: Arc<dyn InferenceEngine>,
    ) -> Self {
        Self { store, engine }
    }

    /// Search with visibility filtering and RBAC enforcement.
    ///
    /// 1. Embeds the query
    /// 2. Searches store with 2x limit (to allow for post-filtering)
    /// 3. Filters by visibility mode + entry_status
    /// 4. Filters by RBAC permissions (scope access)
    /// 5. Returns up to `limit` results
    pub async fn search(
        &self,
        query: &str,
        scope_id: &str,
        limit: usize,
        visibility: &VisibilityMode,
        agent_id: Option<&str>,
        permissions: Option<&AgentPermission>,
    ) -> Result<Vec<SearchResult>> {
        // RBAC check: does the agent have access to this scope?
        // - Admin: can read all scopes
        // - ReadOnly: can read all scopes (D39 — isolation is for correctness, not security)
        // - ReadWrite: can only read scopes in their allowed list
        if let Some(perms) = permissions {
            if let AgentPermission::ReadWrite { scopes } = perms {
                if !scopes.iter().any(|s| s == scope_id || s == "*") {
                    return Ok(Vec::new());
                }
            }
        }

        // Embed the query
        let embedding = self.engine.embed(query).await?;

        // Search with 2x limit to allow for post-filtering
        let fetch_limit = std::cmp::max(limit * 2, 10);
        let results = self.store.search(&embedding, scope_id, fetch_limit).await?;

        // Post-filter by visibility mode
        let filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| {
                match r.entry.entry_status {
                    EntryStatus::Merged => true, // Always visible
                    EntryStatus::Pending => {
                        match visibility {
                            VisibilityMode::Own => {
                                // Only show agent's own Pending entries
                                match (agent_id, &r.entry.agent_id) {
                                    (Some(aid), Some(entry_aid)) => aid == entry_aid,
                                    _ => false,
                                }
                            }
                            VisibilityMode::All => true, // Show all Pending
                            VisibilityMode::Explicit(agents) => {
                                // Show Pending from named agents
                                match &r.entry.agent_id {
                                    Some(entry_aid) => agents.iter().any(|a| a == entry_aid),
                                    None => false,
                                }
                            }
                        }
                    }
                    EntryStatus::Committed => {
                        // Committed entries are in transition — treat like Pending
                        match visibility {
                            VisibilityMode::All => true,
                            _ => {
                                match (agent_id, &r.entry.agent_id) {
                                    (Some(aid), Some(entry_aid)) => aid == entry_aid,
                                    _ => false,
                                }
                            }
                        }
                    }
                    EntryStatus::Rejected => false, // Never visible
                }
            })
            .take(limit)
            .collect();

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use corvia_common::types::KnowledgeEntry;

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

    async fn setup_with_entries(dir: &std::path::Path) -> (ContextBuilder, Arc<dyn QueryableStore>) {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();

        // Use very similar but non-identical embeddings — HNSW needs distance variation
        // to build a well-connected graph. Identical vectors cause degenerate structure.
        // All are close to query [1,0,0] (cosine ~0.999+), ensuring HNSW returns them.
        let mut idx = 0_usize;
        let mut next_emb = || {
            idx += 1;
            let offset = idx as f32 * 0.001;
            vec![1.0, offset, 0.0]
        };

        // Insert 2 Merged entries
        let mut m1 = KnowledgeEntry::new("merged one".into(), "scope-a".into(), "v1".into())
            .with_embedding(next_emb());
        m1.entry_status = EntryStatus::Merged;
        store.insert(&m1).await.unwrap();

        let mut m2 = KnowledgeEntry::new("merged two".into(), "scope-a".into(), "v1".into())
            .with_embedding(next_emb());
        m2.entry_status = EntryStatus::Merged;
        store.insert(&m2).await.unwrap();

        // Insert 1 Pending by agent-A
        let p_a = KnowledgeEntry::new("pending by A".into(), "scope-a".into(), "v1".into())
            .with_embedding(next_emb())
            .with_agent("agent-A".into(), "sess-A".into());
        store.insert(&p_a).await.unwrap();

        // Insert 1 Pending by agent-B
        let p_b = KnowledgeEntry::new("pending by B".into(), "scope-a".into(), "v1".into())
            .with_embedding(next_emb())
            .with_agent("agent-B".into(), "sess-B".into());
        store.insert(&p_b).await.unwrap();

        // Add filler entries to improve HNSW graph connectivity at small scale
        for i in 0..6 {
            let emb = next_emb();
            let mut filler = KnowledgeEntry::new(
                format!("filler-{i}"), "scope-a".into(), "v1".into(),
            ).with_embedding(emb);
            filler.entry_status = EntryStatus::Merged;
            store.insert(&filler).await.unwrap();
        }

        let builder = ContextBuilder::new(store.clone(), engine);
        (builder, store)
    }

    #[tokio::test]
    async fn test_own_visibility_includes_agent_pending() {
        let dir = tempfile::tempdir().unwrap();
        let (builder, _store) = setup_with_entries(dir.path()).await;

        let results = builder.search(
            "test query",
            "scope-a",
            10,
            &VisibilityMode::Own,
            Some("agent-A"),
            None,
        ).await.unwrap();

        // Should include Merged entries + agent-A's Pending, NOT agent-B's Pending
        let has_merged = results.iter().any(|r| r.entry.entry_status == EntryStatus::Merged);
        let has_a_pending = results.iter().any(|r| {
            r.entry.entry_status == EntryStatus::Pending
            && r.entry.agent_id.as_deref() == Some("agent-A")
        });
        let has_b_pending = results.iter().any(|r| {
            r.entry.entry_status == EntryStatus::Pending
            && r.entry.agent_id.as_deref() == Some("agent-B")
        });

        assert!(has_merged);
        assert!(has_a_pending);
        assert!(!has_b_pending);
    }

    #[tokio::test]
    async fn test_all_visibility_includes_all_pending() {
        let dir = tempfile::tempdir().unwrap();
        let (builder, _store) = setup_with_entries(dir.path()).await;

        let results = builder.search(
            "test query",
            "scope-a",
            10,
            &VisibilityMode::All,
            Some("agent-A"),
            None,
        ).await.unwrap();

        // 10 entries (8 Merged + 2 Pending). HNSW is approximate, so use >= threshold.
        // Key behavioral assertion: All visibility includes BOTH Merged and Pending.
        assert!(results.len() >= 4, "Expected at least 4 results, got {}", results.len());
        let has_merged = results.iter().any(|r| r.entry.entry_status == EntryStatus::Merged);
        let has_pending = results.iter().any(|r| r.entry.entry_status == EntryStatus::Pending);
        assert!(has_merged, "All visibility should include Merged entries");
        assert!(has_pending, "All visibility should include Pending entries");
    }

    #[tokio::test]
    async fn test_explicit_visibility() {
        let dir = tempfile::tempdir().unwrap();
        let (builder, _store) = setup_with_entries(dir.path()).await;

        let results = builder.search(
            "test query",
            "scope-a",
            10,
            &VisibilityMode::Explicit(vec!["agent-A".into()]),
            None,
            None,
        ).await.unwrap();

        // Should include Merged + agent-A's Pending only
        let has_b_pending = results.iter().any(|r| {
            r.entry.entry_status == EntryStatus::Pending
            && r.entry.agent_id.as_deref() == Some("agent-B")
        });
        assert!(!has_b_pending);

        let has_a_pending = results.iter().any(|r| {
            r.entry.entry_status == EntryStatus::Pending
            && r.entry.agent_id.as_deref() == Some("agent-A")
        });
        assert!(has_a_pending);
    }

    #[tokio::test]
    async fn test_rbac_scope_filtering() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();

        // Insert entries in project-a
        let mut e = KnowledgeEntry::new("project-a data".into(), "project-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        e.entry_status = EntryStatus::Merged;
        store.insert(&e).await.unwrap();

        let builder = ContextBuilder::new(store.clone(), engine);

        // Agent with access to project-a only
        let perms = AgentPermission::ReadWrite { scopes: vec!["project-a".into()] };

        // Can search project-a
        let results = builder.search(
            "query", "project-a", 10, &VisibilityMode::Own, Some("agent"), Some(&perms),
        ).await.unwrap();
        assert_eq!(results.len(), 1);

        // Cannot search project-b (no access)
        let results = builder.search(
            "query", "project-b", 10, &VisibilityMode::Own, Some("agent"), Some(&perms),
        ).await.unwrap();
        assert_eq!(results.len(), 0);
    }
}
