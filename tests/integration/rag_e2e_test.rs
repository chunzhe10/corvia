//! RAG Pipeline Integration Test: Retrieve → Augment → Trace end-to-end.
//!
//! Exercises the full pipeline factory + context mode using LiteStore
//! and a mock embedding engine (no external services required).
//!
//! Run with: cargo test --test rag_e2e_test -- --nocapture

use corvia_common::agent_types::EntryStatus;
use corvia_common::config::CorviaConfig;
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::traits::{GraphStore, InferenceEngine, QueryableStore};
use corvia_kernel::lite_store::LiteStore;
use std::sync::Arc;
use tempfile::tempdir;

struct MockEngine;

#[async_trait::async_trait]
impl InferenceEngine for MockEngine {
    async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
        Ok(vec![1.0, 0.0, 0.0])
    }
    async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
    }
    fn dimensions(&self) -> usize {
        3
    }
}

// ---------------------------------------------------------------------------
// Test 1: Context mode — retrieve + augment, no generation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_pipeline_context_mode_e2e() {
    let dir = tempdir().unwrap();
    let config = CorviaConfig::default();

    let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
    store.init_schema().await.unwrap();

    let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;

    // Insert 10 entries with varied embeddings for HNSW connectivity.
    let mut idx = 0_usize;
    let mut next_emb = || {
        idx += 1;
        vec![1.0, idx as f32 * 0.001, 0.0]
    };

    for i in 0..10 {
        let mut e = KnowledgeEntry::new(
            format!("Knowledge about topic {i}"),
            "test-scope".into(),
            "v1".into(),
        )
        .with_embedding(next_emb());
        e.entry_status = EntryStatus::Merged;
        store.insert(&e).await.unwrap();
    }

    // Create pipeline via factory — no graph, no generator.
    let pipeline = corvia_kernel::create_rag_pipeline(
        store.clone() as Arc<dyn QueryableStore>,
        engine,
        None,
        None,
        &config,
    ).await;

    // Without graph, should use VectorRetriever.
    assert_eq!(pipeline.retriever_name(), "vector");

    // Test context() mode.
    let response = pipeline
        .context("topic 0", "test-scope", None)
        .await
        .unwrap();

    assert!(response.answer.is_none(), "context mode produces no answer");
    assert!(
        response.context.metrics.sources_included > 0,
        "should include at least one source"
    );
    assert!(
        response.context.context.contains("[1]"),
        "context should have citation markers"
    );
    assert_eq!(
        response.trace.retrieval.retriever_name, "vector",
        "trace should record retriever name"
    );
    assert!(
        response.trace.generation.is_none(),
        "no generation trace in context mode"
    );
    assert!(
        response.trace.total_latency_ms < 5000,
        "pipeline should complete in reasonable time"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Graph-expanded retriever via factory
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_pipeline_with_graph_e2e() {
    let dir = tempdir().unwrap();
    let config = CorviaConfig::default();

    let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
    store.init_schema().await.unwrap();

    let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;

    // Insert entries with edges to exercise graph expansion.
    let mut idx = 0_usize;
    let mut next_emb = || {
        idx += 1;
        vec![1.0, idx as f32 * 0.001, 0.0]
    };

    let mut ids = Vec::new();
    for i in 0..10 {
        let mut e = KnowledgeEntry::new(
            format!("Module {i} implementation"),
            "graph-scope".into(),
            "v1".into(),
        )
        .with_embedding(next_emb());
        e.entry_status = EntryStatus::Merged;
        let id = e.id;
        store.insert(&e).await.unwrap();
        ids.push(id);
    }

    // Create edges: 0→1, 1→2, 2→3
    for w in ids.windows(2).take(3) {
        store.relate(&w[0], "imports", &w[1], None).await.unwrap();
    }

    // Create pipeline WITH graph — should use GraphExpandRetriever.
    let pipeline = corvia_kernel::create_rag_pipeline(
        store.clone() as Arc<dyn QueryableStore>,
        engine,
        Some(store.clone() as Arc<dyn GraphStore>),
        None,
        &config,
    ).await;

    assert_eq!(pipeline.retriever_name(), "graph_expand");

    let response = pipeline
        .context("module implementation", "graph-scope", None)
        .await
        .unwrap();

    assert!(response.answer.is_none());
    assert!(
        response.context.metrics.sources_included > 0,
        "graph-expanded retrieval should return sources"
    );
    assert_eq!(response.trace.retrieval.retriever_name, "graph_expand");

    // Trace should record the graph_expanded metric.
    eprintln!(
        "graph_expanded: {} entries",
        response.trace.retrieval.graph_expanded
    );
}

// ---------------------------------------------------------------------------
// Test 3: ask() without generator returns clear error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_pipeline_ask_without_generator_e2e() {
    let dir = tempdir().unwrap();
    let config = CorviaConfig::default();

    let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
    store.init_schema().await.unwrap();

    let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;

    let pipeline = corvia_kernel::create_rag_pipeline(
        store.clone() as Arc<dyn QueryableStore>,
        engine,
        None,
        None,
        &config,
    ).await;

    let result = pipeline.ask("question", "empty-scope", None).await;
    assert!(result.is_err(), "ask() without generator should fail");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("GenerationEngine"),
        "error should mention GenerationEngine, got: {err}"
    );
}
