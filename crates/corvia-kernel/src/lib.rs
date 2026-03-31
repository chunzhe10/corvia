//! Knowledge storage, agent coordination, graph reasoning, and temporal queries
//! for the Corvia organizational memory system.
//!
//! `corvia-kernel` is the primary library crate — it contains all storage backends,
//! the agent coordination layer, and the reasoning engine.
//!
//! # Storage Tiers
//!
//! Corvia offers two storage tiers behind the same [`QueryableStore`](traits::QueryableStore) trait:
//!
//! - **[`LiteStore`](lite_store::LiteStore)** (default) — zero-Docker, uses hnsw_rs for
//!   vector search, petgraph for graph traversal, and Redb for metadata. Knowledge is
//!   persisted as Git-tracked JSON files in `.corvia/knowledge/`.
//! - **`PostgresStore`** (opt-in, `--features postgres`) — PostgreSQL + pgvector store
//!   with HNSW vector indexes, recursive CTE temporal queries, and relational graph edges.
//!
//! # Core Traits
//!
//! | Trait | Purpose |
//! |-------|---------|
//! | [`QueryableStore`](traits::QueryableStore) | Insert, search, get, count, delete |
//! | [`InferenceEngine`](traits::InferenceEngine) | Text → embedding vectors |
//! | [`TemporalStore`](traits::TemporalStore) | Point-in-time and evolution queries |
//! | [`GraphStore`](traits::GraphStore) | Edges, traversal, shortest path |
//! | [`IngestionAdapter`](traits::IngestionAdapter) | Source → knowledge entries |
//!
//! # Agent Coordination
//!
//! Multi-agent writes are isolated through staging branches
//! ([`staging`]), coordinated by the [`agent_coordinator::AgentCoordinator`],
//! and merged via an LLM-assisted [`merge_worker`].
//!
//! # Reasoning
//!
//! The [`reasoner`] module provides five deterministic health checks (staleness,
//! fragmentation, contradiction signals, coverage gaps, relationship density) plus
//! two LLM-powered checks (semantic contradiction detection and insight generation).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use corvia_common::config::CorviaConfig;
//! use corvia_kernel::{create_engine, create_store};
//!
//! # async fn example() -> corvia_common::errors::Result<()> {
//! let config = CorviaConfig::default(); // LiteStore + Ollama
//! let engine = create_engine(&config);
//! let store = create_store(&config).await?;
//! store.init_schema().await?;
//! # Ok(())
//! # }
//! ```
//!
//! See the [project architecture](https://github.com/corvia/corvia/blob/master/ARCHITECTURE.md)
//! for the full layered design.

pub mod traits;
pub mod embedding_pipeline;
pub mod ollama_engine;
pub mod ollama_provisioner;
pub mod knowledge_files;
pub mod lite_store;
pub mod docker;
pub mod introspect;
pub mod agent_registry;
pub mod session_manager;
pub mod staging;
pub mod agent_writer;
pub mod merge_queue;
pub mod commit_pipeline;
pub mod merge_worker;
pub mod ollama_chat;
pub mod context_builder;
pub mod agent_coordinator;
pub mod graph_store;
pub mod reasoner;
pub mod token_estimator;
pub mod rag_types;
pub mod retriever;
pub mod augmenter;
pub mod rag_pipeline;
pub mod chunking_strategy;
pub mod chunking_fallback;
pub mod chunking_pipeline;
pub mod chunking_markdown;
pub mod chunking_config_fmt;
pub mod chunking_pdf;
pub mod semantic_split;
pub mod adapter_protocol;
pub mod adapter_discovery;
pub mod process_adapter;
pub mod ops;
pub mod event_bus;
pub mod grpc_engine;
pub mod grpc_chat;
pub mod inference_provisioner;
pub mod pipeline;
pub mod tantivy_index;
pub mod skill_registry;
pub mod ingest;
pub mod scoring;
pub mod gc_worker;
pub mod spoke;
pub(crate) mod access_buffer;
#[cfg(feature = "postgres")]
pub mod postgres_store;

use corvia_common::config::{CorviaConfig, InferenceProvider, StoreType};
use corvia_common::errors::Result;
use std::sync::Arc;
use tracing::{info, warn};

/// Create the appropriate InferenceEngine based on config.
pub fn create_engine(config: &CorviaConfig) -> Arc<dyn traits::InferenceEngine> {
    match config.embedding.provider {
        InferenceProvider::Ollama => Arc::new(ollama_engine::OllamaEngine::new(
            &config.embedding.url,
            &config.embedding.model,
            config.embedding.dimensions,
        )),
        InferenceProvider::Vllm => Arc::new(embedding_pipeline::VllmEngine::new(
            &config.embedding.url,
            &config.embedding.model,
            config.embedding.dimensions,
        )),
        InferenceProvider::Corvia => Arc::new(grpc_engine::GrpcInferenceEngine::new(
            &config.embedding.url,
            &config.embedding.model,
            config.embedding.dimensions,
        )),
    }
}

/// Internal: connect to PostgreSQL with config defaults.
#[cfg(feature = "postgres")]
async fn connect_postgres(config: &CorviaConfig) -> Result<postgres_store::PostgresStore> {
    let url = config
        .storage
        .postgres_url
        .as_deref()
        .unwrap_or("postgres://corvia:corvia@127.0.0.1:5432/corvia");
    postgres_store::PostgresStore::connect(url, config.embedding.dimensions).await
}

/// Create a store at a specific directory path (for workspace support).
///
/// For LiteStore, opens the store at the given `data_dir` path instead of
/// reading `config.storage.data_dir`. For PostgresStore, delegates to `create_store`.
pub async fn create_store_at(
    config: &CorviaConfig,
    data_dir: &std::path::Path,
) -> Result<Box<dyn traits::QueryableStore>> {
    match config.storage.store_type {
        StoreType::Lite => {
            let store = lite_store::LiteStore::open(data_dir, config.embedding.dimensions)?;
            Ok(Box::new(store))
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            let store = connect_postgres(config).await?;
            Ok(Box::new(store))
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            Err(corvia_common::errors::CorviaError::Config(
                "PostgresStore requires --features postgres".into(),
            ))
        }
    }
}

/// Create the appropriate QueryableStore based on config.
pub async fn create_store(config: &CorviaConfig) -> Result<Box<dyn traits::QueryableStore>> {
    match config.storage.store_type {
        StoreType::Lite => {
            let store = lite_store::LiteStore::open(
                std::path::Path::new(&config.storage.data_dir),
                config.embedding.dimensions,
            )?;
            Ok(Box::new(store))
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            let store = connect_postgres(config).await?;
            Ok(Box::new(store))
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            Err(corvia_common::errors::CorviaError::Config(
                "PostgresStore requires --features postgres".into(),
            ))
        }
    }
}

/// Create store with both QueryableStore and GraphStore access.
/// Both Arcs point to the same underlying store instance (single-writer safe).
pub async fn create_store_with_graph(
    config: &CorviaConfig,
) -> Result<(Arc<dyn traits::QueryableStore>, Arc<dyn traits::GraphStore>)> {
    match config.storage.store_type {
        StoreType::Lite => {
            let store = Arc::new(lite_store::LiteStore::open(
                std::path::Path::new(&config.storage.data_dir),
                config.embedding.dimensions,
            )?);
            Ok((store.clone() as Arc<dyn traits::QueryableStore>, store as Arc<dyn traits::GraphStore>))
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            let store = Arc::new(connect_postgres(config).await?);
            Ok((store.clone() as Arc<dyn traits::QueryableStore>, store as Arc<dyn traits::GraphStore>))
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            Err(corvia_common::errors::CorviaError::Config(
                "PostgresStore requires --features postgres".into(),
            ))
        }
    }
}

/// Create store with QueryableStore, GraphStore, AND TemporalStore access.
/// All three Arcs point to the same underlying store instance.
pub async fn create_full_store(
    config: &CorviaConfig,
) -> Result<(Arc<dyn traits::QueryableStore>, Arc<dyn traits::GraphStore>, Arc<dyn traits::TemporalStore>)> {
    match config.storage.store_type {
        StoreType::Lite => {
            let store = Arc::new(lite_store::LiteStore::open(
                std::path::Path::new(&config.storage.data_dir),
                config.embedding.dimensions,
            )?);
            Ok((
                store.clone() as Arc<dyn traits::QueryableStore>,
                store.clone() as Arc<dyn traits::GraphStore>,
                store as Arc<dyn traits::TemporalStore>,
            ))
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            let store = Arc::new(connect_postgres(config).await?);
            Ok((
                store.clone() as Arc<dyn traits::QueryableStore>,
                store.clone() as Arc<dyn traits::GraphStore>,
                store as Arc<dyn traits::TemporalStore>,
            ))
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            Err(corvia_common::errors::CorviaError::Config(
                "PostgresStore requires --features postgres".into(),
            ))
        }
    }
}

/// Create a RAG pipeline with auto-selected retriever.
///
/// Uses [`GraphExpandRetriever`](retriever::GraphExpandRetriever) when a
/// [`GraphStore`](traits::GraphStore) is available,
/// [`VectorRetriever`](retriever::VectorRetriever) otherwise.
///
/// When `skills_enabled` is true in the config, loads skill files from
/// `skills_dirs` and embeds their descriptions for query-time matching.
///
/// This is the main entry point for constructing a ready-to-use
/// [`RagPipeline`](rag_pipeline::RagPipeline) from the kernel's factory layer.
pub async fn create_rag_pipeline(
    store: Arc<dyn traits::QueryableStore>,
    engine: Arc<dyn traits::InferenceEngine>,
    graph: Option<Arc<dyn traits::GraphStore>>,
    generator: Option<Arc<dyn traits::GenerationEngine>>,
    config: &CorviaConfig,
) -> rag_pipeline::RagPipeline {
    // When [rag.pipeline] is configured, use the composable pipeline.
    // Otherwise, fall back to legacy monolithic retrievers.
    let ret: Arc<dyn retriever::Retriever> = if let Some(ref pipeline_cfg) = config.rag.pipeline {
        build_pipeline_retriever(
            store.clone(),
            engine.clone(),
            graph.clone(),
            config,
            pipeline_cfg,
        )
    } else {
        build_legacy_retriever(store.clone(), engine.clone(), graph, config)
    };

    let system_prompt = if config.rag.system_prompt.is_empty() {
        String::new()
    } else {
        config.rag.system_prompt.clone()
    };

    // Load skill registry when enabled.
    let skill_reg = if config.rag.skills_enabled {
        match skill_registry::SkillRegistry::load(&config.rag.skills_dirs, engine.clone()).await {
            Ok(reg) if !reg.is_empty() => {
                info!(count = reg.len(), "skill registry loaded");
                Some(Arc::new(reg))
            }
            Ok(_) => {
                info!("skills enabled but no skill files found");
                None
            }
            Err(e) => {
                warn!(error = %e, "failed to load skills, continuing without");
                None
            }
        }
    } else {
        None
    };

    let aug: Arc<dyn augmenter::Augmenter> = match skill_reg {
        Some(reg) => {
            let prompt = if system_prompt.is_empty() {
                "You are a knowledge assistant. Answer questions using only the provided context. \
                 Cite sources using [N] notation.".to_string()
            } else {
                system_prompt
            };
            Arc::new(augmenter::StructuredAugmenter::with_skills(prompt, reg))
        }
        None => {
            if system_prompt.is_empty() {
                Arc::new(augmenter::StructuredAugmenter::new())
            } else {
                Arc::new(augmenter::StructuredAugmenter::with_system_prompt(system_prompt))
            }
        }
    };

    rag_pipeline::RagPipeline::new(ret, aug, generator, config.rag.clone())
}

/// Create a [`ChunkingPipeline`] pre-loaded with kernel default strategies.
///
/// The returned pipeline has [`FallbackChunker`], [`MarkdownChunker`],
/// [`ConfigChunker`], and [`PdfChunker`] registered. Adapters can register
/// additional strategies (e.g., [`AstChunker`]) via [`ChunkingPipeline::registry_mut()`].
pub fn create_chunking_pipeline(config: &CorviaConfig) -> chunking_pipeline::ChunkingPipeline {
    chunking_pipeline::ChunkingPipeline::with_kernel_defaults(config.chunking.clone())
}

/// Same as `create_store_with_graph` but with explicit data_dir (for workspace support).
pub async fn create_store_at_with_graph(
    config: &CorviaConfig,
    data_dir: &std::path::Path,
) -> Result<(Arc<dyn traits::QueryableStore>, Arc<dyn traits::GraphStore>)> {
    match config.storage.store_type {
        StoreType::Lite => {
            let store = Arc::new(lite_store::LiteStore::open(data_dir, config.embedding.dimensions)?);
            Ok((store.clone() as Arc<dyn traits::QueryableStore>, store as Arc<dyn traits::GraphStore>))
        }
        StoreType::Postgres => {
            create_store_with_graph(config).await
        }
    }
}

/// Build a legacy monolithic retriever (backward compat when [rag.pipeline] is absent).
#[allow(deprecated)]
fn build_legacy_retriever(
    store: Arc<dyn traits::QueryableStore>,
    engine: Arc<dyn traits::InferenceEngine>,
    graph: Option<Arc<dyn traits::GraphStore>>,
    config: &CorviaConfig,
) -> Arc<dyn retriever::Retriever> {
    match config.rag.retriever.as_str() {
        "vector" => Arc::new(retriever::VectorRetriever::new(store, engine)),
        "graph_expand" => {
            match graph {
                Some(g) => Arc::new(retriever::GraphExpandRetriever::new(
                    store, engine, g, config.rag.graph_alpha,
                )),
                None => {
                    info!(fallback = "vector", "graph store unavailable, falling back to vector retriever");
                    Arc::new(retriever::VectorRetriever::new(store, engine))
                }
            }
        }
        other => {
            warn!(
                retriever = other,
                "unknown retriever '{other}', falling back to graph_expand. \
                 Valid options: \"vector\", \"graph_expand\""
            );
            match graph {
                Some(g) => Arc::new(retriever::GraphExpandRetriever::new(
                    store, engine, g, config.rag.graph_alpha,
                )),
                None => Arc::new(retriever::VectorRetriever::new(store, engine)),
            }
        }
    }
}

/// Rebuild a composable pipeline retriever from updated config.
///
/// This is the hot-swap entry point: called when `config_set` modifies
/// `rag.pipeline.*` fields. Validates the config and builds a new pipeline.
/// Returns `Err` if the config is invalid (caller should keep the old pipeline).
pub fn rebuild_pipeline_retriever(
    store: Arc<dyn traits::QueryableStore>,
    engine: Arc<dyn traits::InferenceEngine>,
    graph: Option<Arc<dyn traits::GraphStore>>,
    generator: Option<Arc<dyn traits::GenerationEngine>>,
    config: &CorviaConfig,
) -> Result<rag_pipeline::RagPipeline> {
    let pipeline_cfg = config.rag.pipeline.as_ref().ok_or_else(|| {
        corvia_common::errors::CorviaError::Config(
            "[rag.pipeline] section not found in config".into(),
        )
    })?;

    // Reuse existing tantivy from LiteStore if already initialized (OnceLock),
    // otherwise open a new one. Reusing avoids tantivy file lock conflicts when
    // the old pipeline's tantivy is being dropped in a background thread.
    let fts: Option<Arc<dyn traits::FullTextSearchable>> =
        if pipeline_cfg.searchers.iter().any(|s| s == "bm25") {
            if let Some(ls) = store.as_any().downcast_ref::<lite_store::LiteStore>() {
                if let Some(existing) = ls.tantivy() {
                    // Reuse the tantivy index already wired into LiteStore.
                    Some(Arc::clone(existing) as Arc<dyn traits::FullTextSearchable>)
                } else {
                    // First time: open tantivy and wire it into LiteStore.
                    let cache_dir = std::path::Path::new(&config.storage.data_dir)
                        .join("cache")
                        .join("tantivy");
                    match tantivy_index::TantivyIndex::open(&cache_dir, ls.db().clone(), store.clone()) {
                        Ok(idx) => {
                            let idx = Arc::new(idx);
                            ls.set_tantivy(Arc::clone(&idx));
                            Some(idx as Arc<dyn traits::FullTextSearchable>)
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to open TantivyIndex during hot-swap");
                            None
                        }
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

    let deps = pipeline::ComponentDeps {
        store: Some(store.clone()),
        engine: Some(engine.clone()),
        graph: graph.clone(),
        fts,
        rrf_k: pipeline_cfg.rrf.k,
        channels: pipeline_cfg.channels.clone(),
        searcher_timeout_ms: pipeline_cfg.searcher_timeout_ms,
    };

    let retriever = pipeline::PipelineRegistry::validate_and_build(
        pipeline_cfg,
        &deps,
        graph.clone(),
        config.rag.graph_alpha,
    )?;

    let system_prompt = if config.rag.system_prompt.is_empty() {
        String::new()
    } else {
        config.rag.system_prompt.clone()
    };

    let aug: Arc<dyn augmenter::Augmenter> = if system_prompt.is_empty() {
        Arc::new(augmenter::StructuredAugmenter::new())
    } else {
        Arc::new(augmenter::StructuredAugmenter::with_system_prompt(system_prompt))
    };

    Ok(rag_pipeline::RagPipeline::new(
        Arc::new(retriever),
        aug,
        generator,
        config.rag.clone(),
    ))
}

/// Build a composable pipeline retriever from [rag.pipeline] config.
fn build_pipeline_retriever(
    store: Arc<dyn traits::QueryableStore>,
    engine: Arc<dyn traits::InferenceEngine>,
    graph: Option<Arc<dyn traits::GraphStore>>,
    config: &CorviaConfig,
    pipeline_cfg: &corvia_common::config::PipelineConfig,
) -> Arc<dyn retriever::Retriever> {
    use pipeline::{ComponentDeps, PipelineRegistry};

    // Initialize tantivy if "bm25" or "multichannel" is in the searchers list and store is LiteStore.
    // MultiChannelSearcher may use BM25 for the structural channel.
    let fts: Option<Arc<dyn traits::FullTextSearchable>> =
        if pipeline_cfg.searchers.iter().any(|s| s == "bm25" || s == "multichannel") {
            if let Some(ls) = store.as_any().downcast_ref::<lite_store::LiteStore>() {
                let cache_dir = std::path::Path::new(&config.storage.data_dir)
                    .join("cache")
                    .join("tantivy");
                match tantivy_index::TantivyIndex::open(&cache_dir, ls.db().clone(), store.clone()) {
                    Ok(idx) => {
                        let idx = Arc::new(idx);
                        info!("TantivyIndex initialized at {}", cache_dir.display());
                        // Wire tantivy back into LiteStore for insert/delete mirroring.
                        ls.set_tantivy(Arc::clone(&idx));
                        // Check staleness and rebuild if needed.
                        match idx.is_stale() {
                            Ok(true) => {
                                info!("Tantivy index is stale, rebuild will be triggered on first use or via 'corvia rebuild'");
                            }
                            Ok(false) => {
                                info!("Tantivy index is up-to-date");
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to check tantivy staleness");
                            }
                        }
                        Some(idx as Arc<dyn traits::FullTextSearchable>)
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to open TantivyIndex, BM25 searcher will be unavailable");
                        None
                    }
                }
            } else {
                info!("BM25 searcher requested but store is not LiteStore, skipping");
                None
            }
        } else {
            None
        };

    let registry = PipelineRegistry::with_defaults();
    let deps = ComponentDeps {
        store: Some(store.clone()),
        engine: Some(engine.clone()),
        graph: graph.clone(),
        fts,
        rrf_k: pipeline_cfg.rrf.k,
        channels: pipeline_cfg.channels.clone(),
        searcher_timeout_ms: pipeline_cfg.searcher_timeout_ms,
    };

    // Build searchers.
    let mut searchers: Vec<Arc<dyn pipeline::Searcher>> = Vec::new();
    for name in &pipeline_cfg.searchers {
        match registry.searchers.create(name, &deps) {
            Ok(s) => searchers.push(s),
            Err(e) => {
                warn!(searcher = name.as_str(), error = %e, "failed to create searcher, skipping");
            }
        }
    }
    if searchers.is_empty() {
        warn!("no searchers configured, falling back to vector");
        if let Ok(s) = registry.searchers.create("vector", &deps) {
            searchers.push(s);
        }
    }

    // Build fusion.
    let fusion = registry
        .fusions
        .create(&pipeline_cfg.fusion, &deps)
        .unwrap_or_else(|e| {
            warn!(fusion = pipeline_cfg.fusion.as_str(), error = %e, "falling back to passthrough");
            registry.fusions.create("passthrough", &deps).expect("built-in 'passthrough' fusion must be registered")
        });

    // Build expander (falls back to noop if graph store unavailable).
    let expander = match (pipeline_cfg.expander.as_str(), graph) {
        ("graph", Some(g)) => {
            Arc::new(pipeline::expander::GraphExpander::new(
                store.clone(),
                g,
                config.rag.graph_alpha,
            )) as Arc<dyn pipeline::Expander>
        }
        ("graph", None) => {
            info!("graph expander requested but no graph store, falling back to noop");
            registry.expanders.create("noop", &deps).expect("noop expander is always registered")
        }
        (name, _) => {
            registry
                .expanders
                .create(name, &deps)
                .unwrap_or_else(|e| {
                    warn!(expander = name, error = %e, "falling back to noop");
                    registry.expanders.create("noop", &deps).expect("noop expander is always registered")
                })
        }
    };

    // Build reranker.
    let reranker = registry
        .rerankers
        .create(&pipeline_cfg.reranker, &deps)
        .unwrap_or_else(|e| {
            warn!(reranker = pipeline_cfg.reranker.as_str(), error = %e, "falling back to identity");
            registry.rerankers.create("identity", &deps).expect("built-in 'identity' reranker must be registered")
        });

    Arc::new(pipeline::RetrievalPipeline::new(
        searchers,
        fusion,
        expander,
        reranker,
        engine,
        store,
        pipeline_cfg.searcher_timeout_ms,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::config::InferenceProvider;

    #[test]
    fn test_create_engine_ollama() {
        let config = CorviaConfig::default();
        assert_eq!(config.embedding.provider, InferenceProvider::Ollama);
        let engine = create_engine(&config);
        assert_eq!(engine.dimensions(), 768);
    }

    #[test]
    fn test_create_engine_vllm() {
        let mut config = CorviaConfig::default();
        config.embedding.provider = InferenceProvider::Vllm;
        config.embedding.url = "http://127.0.0.1:8001".into();
        config.embedding.model = "nomic-ai/nomic-embed-text-v1.5".into();
        let engine = create_engine(&config);
        assert_eq!(engine.dimensions(), 768);
    }

    #[tokio::test]
    async fn test_create_store_lite() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = CorviaConfig::default();
        config.storage.data_dir = dir.path().to_string_lossy().to_string();

        let store = create_store(&config).await.unwrap();
        store.init_schema().await.unwrap();
        assert_eq!(store.count("test").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_create_rag_pipeline_without_graph() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = CorviaConfig::default();
        config.storage.data_dir = dir.path().to_string_lossy().to_string();

        let store = create_store(&config).await.unwrap();
        let store: Arc<dyn traits::QueryableStore> = Arc::from(store);
        store.init_schema().await.unwrap();
        let engine: Arc<dyn traits::InferenceEngine> = Arc::new(
            ollama_engine::OllamaEngine::new(
                &config.embedding.url,
                &config.embedding.model,
                config.embedding.dimensions,
            ),
        );

        let pipeline = create_rag_pipeline(store, engine, None, None, &config).await;
        assert_eq!(pipeline.retriever_name(), "vector");
    }

    #[tokio::test]
    async fn test_create_rag_pipeline_with_graph() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = CorviaConfig::default();
        config.storage.data_dir = dir.path().to_string_lossy().to_string();

        let (store, graph) = create_store_with_graph(&config).await.unwrap();
        store.init_schema().await.unwrap();
        let engine: Arc<dyn traits::InferenceEngine> = Arc::new(
            ollama_engine::OllamaEngine::new(
                &config.embedding.url,
                &config.embedding.model,
                config.embedding.dimensions,
            ),
        );

        let pipeline = create_rag_pipeline(store, engine, Some(graph), None, &config).await;
        assert_eq!(pipeline.retriever_name(), "graph_expand");
    }

    /// Verify that `rebuild_pipeline_retriever` reuses an existing tantivy index
    /// from LiteStore's OnceLock instead of trying to open a new one (which would
    /// fail due to file lock conflicts).
    #[tokio::test]
    async fn test_rebuild_pipeline_reuses_existing_tantivy() {
        use corvia_common::config::PipelineConfig;

        let dir = tempfile::tempdir().unwrap();
        let mut config = CorviaConfig::default();
        config.storage.data_dir = dir.path().to_string_lossy().to_string();

        // Set up pipeline config with both vector and bm25 searchers.
        let mut pipeline_cfg = PipelineConfig::default();
        pipeline_cfg.searchers = vec!["vector".into(), "bm25".into()];
        pipeline_cfg.fusion = "rrf".into();
        config.rag.pipeline = Some(pipeline_cfg);

        // Create a LiteStore and initialize it.
        let store = create_store(&config).await.unwrap();
        store.init_schema().await.unwrap();
        let store: Arc<dyn traits::QueryableStore> = Arc::from(store);

        // Open a TantivyIndex and wire it into LiteStore via set_tantivy().
        let ls = store.as_any().downcast_ref::<lite_store::LiteStore>().unwrap();
        let cache_dir = dir.path().join("cache").join("tantivy");
        let tantivy = tantivy_index::TantivyIndex::open(&cache_dir, ls.db().clone(), store.clone()).unwrap();
        ls.set_tantivy(Arc::new(tantivy));

        let engine: Arc<dyn traits::InferenceEngine> = Arc::new(
            ollama_engine::OllamaEngine::new(
                &config.embedding.url,
                &config.embedding.model,
                config.embedding.dimensions,
            ),
        );

        // First hot-swap: should reuse the OnceLock tantivy, not open a new one.
        let result = rebuild_pipeline_retriever(
            store.clone(),
            engine.clone(),
            None,
            None,
            &config,
        );
        assert!(result.is_ok(), "First rebuild_pipeline_retriever should succeed");

        // Second hot-swap: should still reuse the same OnceLock tantivy.
        let result = rebuild_pipeline_retriever(
            store.clone(),
            engine.clone(),
            None,
            None,
            &config,
        );
        assert!(result.is_ok(), "Second rebuild_pipeline_retriever should succeed (OnceLock reuse)");
    }
}
