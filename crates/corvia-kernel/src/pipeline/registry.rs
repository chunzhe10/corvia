//! Component registry for the composable pipeline.
//!
//! [`ComponentRegistry`] is a generic factory registry that maps names to
//! component constructors. [`PipelineRegistry`] bundles registries for all
//! four stage types and provides `with_defaults()` for built-in components.

use std::collections::HashMap;
use std::sync::Arc;

use corvia_common::errors::{CorviaError, Result};

use super::expander::{Expander, NoOpExpander};
use super::fusion::{Fusion, PassThrough, RRFusion};
use super::searcher::{BM25Searcher, Searcher, VectorSearcher};
use super::{IdentityReranker, Reranker};
use crate::traits::{FullTextSearchable, GraphStore, InferenceEngine, QueryableStore};

/// Shared dependencies available to component factories.
#[derive(Clone)]
pub struct ComponentDeps {
    pub store: Option<Arc<dyn QueryableStore>>,
    pub engine: Option<Arc<dyn InferenceEngine>>,
    pub graph: Option<Arc<dyn GraphStore>>,
    /// Full-text search backend. Required by BM25Searcher.
    pub fts: Option<Arc<dyn FullTextSearchable>>,
    /// RRF smoothing constant k. Read from `[rag.pipeline.rrf]` config.
    pub rrf_k: usize,
}

/// Error variants for pipeline component operations.
#[derive(Debug)]
pub enum PipelineError {
    /// Requested component name not found in registry.
    UnknownComponent(String),
    /// Factory failed to create the component.
    FactoryError(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownComponent(name) => write!(f, "unknown pipeline component: {name}"),
            Self::FactoryError(msg) => write!(f, "pipeline factory error: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<PipelineError> for CorviaError {
    fn from(e: PipelineError) -> Self {
        CorviaError::Config(e.to_string())
    }
}

/// Type-erased factory function for pipeline components.
type Factory<T> = Box<dyn Fn(&ComponentDeps) -> Result<Arc<T>> + Send + Sync>;

/// Generic component registry mapping names to factory functions.
pub struct ComponentRegistry<T: ?Sized> {
    factories: HashMap<String, Factory<T>>,
}

impl<T: ?Sized> ComponentRegistry<T> {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a named factory.
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn(&ComponentDeps) -> Result<Arc<T>> + Send + Sync + 'static,
    {
        self.factories.insert(name.into(), Box::new(factory));
    }

    /// Create a component by name using the given dependencies.
    pub fn create(&self, name: &str, deps: &ComponentDeps) -> Result<Arc<T>> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| PipelineError::UnknownComponent(name.to_string()))?;
        factory(deps).map_err(|e| PipelineError::FactoryError(e.to_string()).into())
    }

    /// Check if a component name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// List registered component names.
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl<T: ?Sized> Default for ComponentRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle of registries for all four pipeline stages.
pub struct PipelineRegistry {
    pub searchers: ComponentRegistry<dyn Searcher>,
    pub fusions: ComponentRegistry<dyn Fusion>,
    pub expanders: ComponentRegistry<dyn Expander>,
    pub rerankers: ComponentRegistry<dyn Reranker>,
}

impl PipelineRegistry {
    /// Validate pipeline config and build a complete `RetrievalPipeline`.
    ///
    /// Returns `Err` if the config references unknown components or required
    /// dependencies are missing. Used by the hot-swap path to validate before
    /// atomically swapping the live pipeline.
    pub fn validate_and_build(
        config: &corvia_common::config::PipelineConfig,
        deps: &ComponentDeps,
        graph: Option<Arc<dyn GraphStore>>,
        graph_alpha: f32,
    ) -> Result<super::RetrievalPipeline> {
        let registry = Self::with_defaults();

        // Build searchers.
        let mut searchers: Vec<Arc<dyn Searcher>> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        for name in &config.searchers {
            match registry.searchers.create(name, deps) {
                Ok(s) => searchers.push(s),
                Err(e) => errors.push(format!("searcher '{name}': {e}")),
            }
        }
        if searchers.is_empty() {
            return Err(CorviaError::Config(format!(
                "no valid searchers in config: {}",
                errors.join("; ")
            )));
        }

        // Build fusion.
        let fusion = registry.fusions.create(&config.fusion, deps)
            .map_err(|e| CorviaError::Config(format!("fusion '{}': {e}", config.fusion)))?;

        // Build expander.
        let expander: Arc<dyn super::Expander> = match (config.expander.as_str(), graph) {
            ("graph", Some(g)) => {
                let store = deps.store.clone().ok_or_else(|| {
                    CorviaError::Config("GraphExpander requires a store".into())
                })?;
                Arc::new(super::expander::GraphExpander::new(store, g, graph_alpha))
            }
            ("graph", None) => {
                registry.expanders.create("noop", deps)
                    .expect("noop expander is always registered")
            }
            (name, _) => {
                registry.expanders.create(name, deps)
                    .map_err(|e| CorviaError::Config(format!("expander '{name}': {e}")))?
            }
        };

        // Build reranker.
        let reranker = registry.rerankers.create(&config.reranker, deps)
            .map_err(|e| CorviaError::Config(format!("reranker '{}': {e}", config.reranker)))?;

        let engine = deps.engine.clone().ok_or_else(|| {
            CorviaError::Config("Pipeline requires an InferenceEngine".into())
        })?;
        let store = deps.store.clone().ok_or_else(|| {
            CorviaError::Config("Pipeline requires a store".into())
        })?;

        if !errors.is_empty() {
            tracing::warn!(
                skipped = ?errors,
                "some searchers failed to create, continuing with available ones"
            );
        }

        Ok(super::RetrievalPipeline::new(
            searchers,
            fusion,
            expander,
            reranker,
            engine,
            store,
            config.searcher_timeout_ms,
        ))
    }

    /// Create a registry pre-loaded with built-in components.
    ///
    /// Registers:
    /// - Searchers: `"vector"`, `"bm25"`
    /// - Fusions: `"passthrough"`
    /// - Expanders: `"graph"`, `"noop"`
    /// - Rerankers: `"identity"`
    pub fn with_defaults() -> Self {
        let mut searchers = ComponentRegistry::new();
        searchers.register("vector", |deps: &ComponentDeps| {
            let store = deps.store.clone().ok_or_else(|| {
                CorviaError::Config("VectorSearcher requires a store".into())
            })?;
            let engine = deps.engine.clone().ok_or_else(|| {
                CorviaError::Config("VectorSearcher requires an engine".into())
            })?;
            Ok(Arc::new(VectorSearcher::new(store, engine)) as Arc<dyn Searcher>)
        });

        searchers.register("bm25", |deps: &ComponentDeps| {
            let fts = deps.fts.clone().ok_or_else(|| {
                CorviaError::Config(
                    "BM25Searcher requires FullTextSearchable (run `corvia rebuild` to build text index)".into(),
                )
            })?;
            Ok(Arc::new(BM25Searcher::new(fts)) as Arc<dyn Searcher>)
        });

        let mut fusions = ComponentRegistry::new();
        fusions.register("passthrough", |_deps: &ComponentDeps| {
            Ok(Arc::new(PassThrough) as Arc<dyn Fusion>)
        });
        fusions.register("rrf", |deps: &ComponentDeps| {
            Ok(Arc::new(RRFusion::new(deps.rrf_k)) as Arc<dyn Fusion>)
        });

        let mut expanders = ComponentRegistry::new();
        // Note: "graph" expander is NOT registered here because it requires
        // config-driven alpha. It is constructed directly in build_pipeline_retriever.
        expanders.register("noop", |_deps: &ComponentDeps| {
            Ok(Arc::new(NoOpExpander) as Arc<dyn Expander>)
        });

        let mut rerankers = ComponentRegistry::new();
        rerankers.register("identity", |_deps: &ComponentDeps| {
            Ok(Arc::new(IdentityReranker) as Arc<dyn Reranker>)
        });

        Self {
            searchers,
            fusions,
            expanders,
            rerankers,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_unknown_name() {
        let registry = ComponentRegistry::<dyn Searcher>::new();
        let deps = ComponentDeps {
            store: None,
            engine: None,
            graph: None,
            fts: None,
            rrf_k: 60,
        };
        let result = registry.create("nonexistent", &deps);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_factory_missing_deps() {
        let reg = PipelineRegistry::with_defaults();
        let deps = ComponentDeps {
            store: None,
            engine: None,
            graph: None,
            fts: None,
            rrf_k: 60,
        };
        // VectorSearcher requires store + engine.
        let result = reg.searchers.create("vector", &deps);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_names() {
        let reg = PipelineRegistry::with_defaults();
        assert!(reg.searchers.contains("vector"));
        assert!(reg.searchers.contains("bm25"));
        assert!(reg.fusions.contains("passthrough"));
        assert!(reg.fusions.contains("rrf"));
        // "graph" expander is NOT in the registry; it requires config-driven alpha
        // and is constructed directly in build_pipeline_retriever.
        assert!(!reg.expanders.contains("graph"));
        assert!(reg.expanders.contains("noop"));
        assert!(reg.rerankers.contains("identity"));
    }

    #[test]
    fn test_passthrough_fusion_creates() {
        let reg = PipelineRegistry::with_defaults();
        let deps = ComponentDeps {
            store: None,
            engine: None,
            graph: None,
            fts: None,
            rrf_k: 60,
        };
        let fusion = reg.fusions.create("passthrough", &deps);
        assert!(fusion.is_ok());
    }

    #[test]
    fn test_identity_reranker_creates() {
        let reg = PipelineRegistry::with_defaults();
        let deps = ComponentDeps {
            store: None,
            engine: None,
            graph: None,
            fts: None,
            rrf_k: 60,
        };
        let reranker = reg.rerankers.create("identity", &deps);
        assert!(reranker.is_ok());
    }
}
