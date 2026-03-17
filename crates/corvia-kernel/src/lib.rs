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
pub mod adapter_protocol;
pub mod adapter_discovery;
pub mod process_adapter;
pub mod ops;
pub mod grpc_engine;
pub mod grpc_chat;
pub mod inference_provisioner;
pub mod skill_registry;
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
    let ret: Arc<dyn retriever::Retriever> = match graph {
        Some(g) => Arc::new(retriever::GraphExpandRetriever::new(
            store.clone(),
            engine.clone(),
            g,
            config.rag.graph_alpha,
        )),
        None => Arc::new(retriever::VectorRetriever::new(store.clone(), engine.clone())),
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
}
