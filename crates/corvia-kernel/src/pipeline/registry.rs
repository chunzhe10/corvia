//! Component registry for the composable pipeline.
//!
//! [`ComponentRegistry`] is a generic factory registry that maps names to
//! component constructors. [`PipelineRegistry`] bundles registries for all
//! four stage types and provides `with_defaults()` for built-in components.

use std::collections::HashMap;
use std::sync::Arc;

use corvia_common::errors::{CorviaError, Result};

use super::expander::{Expander, NoOpExpander};
use super::fusion::{Fusion, PassThrough};
use super::searcher::{Bm25Searcher, Searcher, VectorSearcher};
use super::{IdentityReranker, Reranker};
use crate::traits::{FullTextSearchable, GraphStore, InferenceEngine, QueryableStore};

/// Shared dependencies available to component factories.
#[derive(Clone)]
pub struct ComponentDeps {
    pub store: Option<Arc<dyn QueryableStore>>,
    pub engine: Option<Arc<dyn InferenceEngine>>,
    pub graph: Option<Arc<dyn GraphStore>>,
    pub fts: Option<Arc<dyn FullTextSearchable>>,
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
    /// Create a registry pre-loaded with built-in components.
    ///
    /// Registers:
    /// - Searchers: `"vector"`
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
            let store = deps.store.clone().ok_or_else(|| {
                CorviaError::Config("Bm25Searcher requires a store".into())
            })?;
            let fts = deps.fts.clone().ok_or_else(|| {
                CorviaError::Config("Bm25Searcher requires a FullTextSearchable backend".into())
            })?;
            Ok(Arc::new(Bm25Searcher::new(store, fts)) as Arc<dyn Searcher>)
        });

        let mut fusions = ComponentRegistry::new();
        fusions.register("passthrough", |_deps: &ComponentDeps| {
            Ok(Arc::new(PassThrough) as Arc<dyn Fusion>)
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
        };
        // VectorSearcher requires store + engine.
        let result = reg.searchers.create("vector", &deps);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_names() {
        let reg = PipelineRegistry::with_defaults();
        assert!(reg.searchers.contains("vector"));
        assert!(reg.fusions.contains("passthrough"));
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
        };
        let reranker = reg.rerankers.create("identity", &deps);
        assert!(reranker.is_ok());
    }
}
