//! End-to-end integration tests for M1.
//!
//! SurrealDB tests require: Docker running, SurrealDB on port 8000, vLLM on port 8001.
//! Ollama tests require: Ollama running on port 11434 with nomic-embed-text model.
//! LiteStore-only tests (test_lite_store_*) run without any external services.
//!
//! Run with: cargo test --test e2e_test -- --nocapture

use corvia_common::config::CorviaConfig;
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::embedding_pipeline::VllmEngine;
use corvia_kernel::knowledge_store::SurrealStore;
use corvia_kernel::lite_store::LiteStore;
use corvia_kernel::ollama_engine::OllamaEngine;
use corvia_kernel::traits::{InferenceEngine, QueryableStore};
use corvia_kernel::introspect::{Introspect, IntrospectConfig, IntrospectMeta, CanonicalQuery, IntrospectReport};

/// Per-process unique ID to isolate concurrent test runs against the same SurrealDB.
static TEST_RUN_ID: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!("{:x}", std::process::id())
});

#[tokio::test]
async fn test_write_and_search() {
    // Use test namespace to avoid polluting real data
    let config = CorviaConfig::default();

    let db_name = format!("e2e_{}", *TEST_RUN_ID);
    let store = match SurrealStore::connect(
        config.storage.surrealdb_url.as_deref().unwrap_or("127.0.0.1:8000"),
        "corvia_test",
        &db_name,
        config.storage.surrealdb_user.as_deref().unwrap_or("root"),
        config.storage.surrealdb_pass.as_deref().unwrap_or("root"),
        config.embedding.dimensions,
    ).await {
        Ok(s) => s,
        Err(_) => {
            eprintln!("SKIPPING test_write_and_search: SurrealDB not reachable");
            return;
        }
    };

    store.init_schema().await.expect("Failed to init schema");

    let engine = VllmEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    // Embed and store a test entry
    let content = "fn authenticate(token: &str) -> Result<User> { verify_jwt(token) }";
    let embedding = engine.embed(content).await
        .expect("Failed to embed — is vLLM running with nomic-embed-text-v1.5?");

    assert_eq!(embedding.len(), config.embedding.dimensions);

    let entry = KnowledgeEntry::new(
        content.to_string(),
        "test-repo".to_string(),
        "abc123".to_string(),
    ).with_embedding(embedding);

    store.insert(&entry).await.expect("Failed to insert");

    // Search for it
    let query_embedding = engine.embed("how does authentication work?").await.unwrap();
    let results = store.search(&query_embedding, "test-repo", 5).await.unwrap();

    assert!(!results.is_empty(), "Expected at least one search result");
    assert!(results[0].entry.content.contains("authenticate"));

    println!("E2E test passed: wrote 1 entry, searched, found it.");
    println!("Score: {:.3}", results[0].score);
}

#[tokio::test]
async fn test_introspect_self_query() {
    let config = CorviaConfig::default();

    let db_name = format!("introspect_{}", *TEST_RUN_ID);
    let store = match SurrealStore::connect(
        config.storage.surrealdb_url.as_deref().unwrap_or("127.0.0.1:8000"),
        "corvia_introspect_test",
        &db_name,
        config.storage.surrealdb_user.as_deref().unwrap_or("root"),
        config.storage.surrealdb_pass.as_deref().unwrap_or("root"),
        config.embedding.dimensions,
    ).await {
        Ok(s) => s,
        Err(_) => {
            eprintln!("SKIPPING test_introspect_self_query: SurrealDB not reachable");
            return;
        }
    };

    store.init_schema().await.expect("Failed to init schema");

    let engine = VllmEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    let adapter = corvia_adapter_git::GitAdapter::new();

    let introspect_config = IntrospectConfig {
        config: IntrospectMeta {
            default_min_score: 0.50,
            scope_id: "introspect-e2e".into(),
        },
        query: vec![
            CanonicalQuery {
                text: "what CLI commands are available?".into(),
                expect_file: "crates/corvia-cli/src/main.rs".into(),
                min_score: None,
            },
        ],
    };

    let introspect = Introspect::new(introspect_config);

    let chunks = introspect.ingest_self(".", &adapter, &engine, &store).await
        .expect("Failed to ingest self");
    assert!(chunks > 0, "Expected at least one chunk ingested");

    let results = introspect.query_self(&engine, &store).await
        .expect("Failed to query self");

    let report = IntrospectReport {
        results,
        chunks_ingested: chunks,
    };

    println!("Introspect E2E: {}/{} passed, avg score: {:.3}",
        report.pass_count(), report.results.len(), report.avg_score());

    for r in &report.results {
        println!("  [{}] \"{}\" -> {} (score: {:.3})",
            if r.passed() { "PASS" } else { "FAIL" },
            r.query_text,
            r.actual_file.as_deref().unwrap_or("none"),
            r.score);
    }

    assert!(report.pass_count() > 0, "Expected at least one passing query");
}

#[tokio::test]
async fn test_lite_store_write_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), 768).unwrap();
    store.init_schema().await.unwrap();

    // Create entry with a known embedding
    let entry = KnowledgeEntry::new(
        "fn authenticate(token: &str) -> Result<User> { verify_jwt(token) }".into(),
        "test-repo".into(),
        "abc123".into(),
    ).with_embedding(vec![0.1f32; 768]);

    store.insert(&entry).await.unwrap();

    // Verify get
    let retrieved = store.get(&entry.id).await.unwrap();
    assert!(retrieved.is_some());
    assert!(retrieved.unwrap().content.contains("authenticate"));

    // Verify count
    let count = store.count("test-repo").await.unwrap();
    assert_eq!(count, 1);

    // Verify search (same embedding should match)
    let query_embedding = vec![0.1f32; 768];
    let results = store.search(&query_embedding, "test-repo", 5).await.unwrap();
    assert!(!results.is_empty(), "Expected at least one search result");
    assert!(results[0].entry.content.contains("authenticate"));
    assert!(results[0].score > 0.9, "Same embedding should have high similarity");

    // Verify knowledge file was written
    let knowledge_dir = dir.path().join("knowledge").join("test-repo");
    assert!(knowledge_dir.exists(), "Knowledge directory should exist");

    println!("LiteStore E2E test passed: insert, get, count, search all work.");
}

#[tokio::test]
async fn test_lite_store_rebuild() {
    let dir = tempfile::tempdir().unwrap();

    // Phase 1: Insert entries
    {
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        for i in 0..5 {
            let entry = KnowledgeEntry::new(
                format!("entry {i}"), "rebuild-test".into(), "v1".into(),
            ).with_embedding(vec![i as f32 * 0.1, 0.5, 0.5]);
            store.insert(&entry).await.unwrap();
        }
        assert_eq!(store.count("rebuild-test").await.unwrap(), 5);
    }
    // LiteStore dropped here — simulates process restart

    // Phase 2: Open fresh and rebuild from knowledge files
    {
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let rebuilt = store.rebuild_from_files().unwrap();
        assert_eq!(rebuilt, 5);
        assert_eq!(store.count("rebuild-test").await.unwrap(), 5);

        // Search should work after rebuild
        let results = store.search(&[0.0, 0.5, 0.5], "rebuild-test", 1).await.unwrap();
        assert!(!results.is_empty(), "Search should work after rebuild");
    }

    println!("Rebuild integration test passed.");
}

/// E2E test: OllamaEngine + LiteStore — mirrors test_write_and_search but with zero-Docker stack.
/// Requires: Ollama running on port 11434 with nomic-embed-text model pulled.
#[tokio::test]
async fn test_ollama_lite_store_write_and_search() {
    let config = CorviaConfig::default();

    // Check Ollama is reachable
    if !OllamaEngine::check_health(&config.embedding.url).await {
        eprintln!(
            "SKIPPING test_ollama_lite_store_write_and_search: \
             Ollama not reachable at {}",
            config.embedding.url
        );
        return;
    }

    let engine = OllamaEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), config.embedding.dimensions).unwrap();
    store.init_schema().await.unwrap();

    // Embed and store a test entry
    let content = "fn authenticate(token: &str) -> Result<User> { verify_jwt(token) }";
    let embedding = engine.embed(content).await
        .expect("Failed to embed — is Ollama running with nomic-embed-text?");

    assert_eq!(embedding.len(), config.embedding.dimensions);

    let entry = KnowledgeEntry::new(
        content.to_string(),
        "test-repo".to_string(),
        "abc123".to_string(),
    ).with_embedding(embedding);

    store.insert(&entry).await.expect("Failed to insert");

    // Search for it
    let query_embedding = engine.embed("how does authentication work?").await.unwrap();
    let results = store.search(&query_embedding, "test-repo", 5).await.unwrap();

    assert!(!results.is_empty(), "Expected at least one search result");
    assert!(results[0].entry.content.contains("authenticate"));

    // Verify knowledge file was written
    let knowledge_dir = dir.path().join("knowledge").join("test-repo");
    assert!(knowledge_dir.exists(), "Knowledge directory should exist");

    println!("Ollama + LiteStore E2E test passed.");
    println!("Score: {:.3}", results[0].score);
}

/// E2E test: OllamaEngine + LiteStore Introspect — mirrors test_introspect_self_query.
/// Requires: Ollama running on port 11434 with nomic-embed-text model pulled.
#[tokio::test]
async fn test_ollama_lite_store_introspect() {
    let config = CorviaConfig::default();

    // Check Ollama is reachable
    if !OllamaEngine::check_health(&config.embedding.url).await {
        eprintln!(
            "SKIPPING test_ollama_lite_store_introspect: \
             Ollama not reachable at {}",
            config.embedding.url
        );
        return;
    }

    let engine = OllamaEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    let dir = tempfile::tempdir().unwrap();
    let store = LiteStore::open(dir.path(), config.embedding.dimensions).unwrap();
    store.init_schema().await.unwrap();

    let adapter = corvia_adapter_git::GitAdapter::new();

    let introspect_config = IntrospectConfig {
        config: IntrospectMeta {
            default_min_score: 0.50,
            scope_id: "introspect-lite-e2e".into(),
        },
        query: vec![
            CanonicalQuery {
                text: "what CLI commands are available?".into(),
                expect_file: "crates/corvia-cli/src/main.rs".into(),
                min_score: None,
            },
        ],
    };

    let introspect = Introspect::new(introspect_config);

    let chunks = introspect.ingest_self(".", &adapter, &engine, &store).await
        .expect("Failed to ingest self");
    assert!(chunks > 0, "Expected at least one chunk ingested");

    let results = introspect.query_self(&engine, &store).await
        .expect("Failed to query self");

    let report = IntrospectReport {
        results,
        chunks_ingested: chunks,
    };

    println!("Ollama+LiteStore Introspect: {}/{} passed, avg score: {:.3}",
        report.pass_count(), report.results.len(), report.avg_score());

    for r in &report.results {
        println!("  [{}] \"{}\" -> {} (score: {:.3})",
            if r.passed() { "PASS" } else { "FAIL" },
            r.query_text,
            r.actual_file.as_deref().unwrap_or("none"),
            r.score);
    }

    assert!(report.pass_count() > 0, "Expected at least one passing query");
}
