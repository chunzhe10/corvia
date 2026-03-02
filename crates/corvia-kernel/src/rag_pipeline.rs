//! RagPipeline orchestrator for the R->A->G pipeline (D61).
//!
//! Wires [`Retriever`] -> [`Augmenter`] -> [`GenerationEngine`] together
//! with two modes:
//!
//! - [`RagPipeline::context()`] — retrieve + augment only (no generation).
//! - [`RagPipeline::ask()`] — retrieve + augment + generate.
//!
//! The pipeline builds a [`PipelineTrace`] for observability (D62 Layer B)
//! and respects token budgets (D64) with model-aware auto-sizing.

use corvia_common::config::RagConfig;
use corvia_common::errors::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
use uuid::Uuid;

use crate::augmenter::Augmenter;
use crate::rag_types::*;
use crate::retriever::Retriever;
use crate::traits::GenerationEngine;

/// Main RAG pipeline orchestrator.
///
/// Holds references to the three stages (retriever, augmenter, generator)
/// and a [`RagConfig`] for default parameters. The generator is optional
/// to support context-only mode.
pub struct RagPipeline {
    retriever: Arc<dyn Retriever>,
    augmenter: Arc<dyn Augmenter>,
    generator: Option<Arc<dyn GenerationEngine>>,
    config: RagConfig,
}

impl RagPipeline {
    /// Create a new pipeline with the given stages and config.
    pub fn new(
        retriever: Arc<dyn Retriever>,
        augmenter: Arc<dyn Augmenter>,
        generator: Option<Arc<dyn GenerationEngine>>,
        config: RagConfig,
    ) -> Self {
        Self {
            retriever,
            augmenter,
            generator,
            config,
        }
    }

    /// Get the name of the active retriever (for config verification / tests).
    pub fn retriever_name(&self) -> &str {
        self.retriever.name()
    }

    /// Context-only mode: retrieve + augment, no generation.
    pub async fn context(
        &self,
        query: &str,
        scope_id: &str,
        opts: Option<RetrievalOpts>,
    ) -> Result<RagResponse> {
        self.run_pipeline(query, scope_id, opts, false).await
    }

    /// Full RAG: retrieve + augment + generate.
    pub async fn ask(
        &self,
        query: &str,
        scope_id: &str,
        opts: Option<RetrievalOpts>,
    ) -> Result<RagResponse> {
        // Validate generator exists before running the pipeline.
        self.generator.as_ref().ok_or_else(|| {
            corvia_common::errors::CorviaError::Config(
                "GenerationEngine not configured — cannot use ask() mode".into(),
            )
        })?;
        self.run_pipeline(query, scope_id, opts, true).await
    }

    /// Internal pipeline executor shared by `context()` and `ask()`.
    async fn run_pipeline(
        &self,
        query: &str,
        scope_id: &str,
        opts: Option<RetrievalOpts>,
        generate: bool,
    ) -> Result<RagResponse> {
        let trace_id = Uuid::now_v7();
        let pipeline_start = Instant::now();
        let mode = if generate { "ask" } else { "context" };

        info!(query, scope_id, mode, "rag_query_started");

        // --- Stage 1: Retrieval ---

        let retrieval_opts = opts.unwrap_or_else(|| RetrievalOpts {
            limit: self.config.default_limit,
            expand_graph: self.config.graph_expand,
            graph_depth: self.config.graph_depth,
            ..Default::default()
        });

        let retrieval = self
            .retriever
            .retrieve(query, scope_id, &retrieval_opts)
            .await?;

        // --- Stage 2: Augmentation ---

        // Build token budget: explicit config > model-aware auto-sizing > default.
        let max_context_tokens = if self.config.max_context_tokens > 0 {
            Some(self.config.max_context_tokens)
        } else if let Some(ref generator_ref) = self.generator {
            // Model-aware auto-sizing: context_window * (1 - reserve_for_answer).
            let window = generator_ref.context_window();
            Some((window as f32 * (1.0 - self.config.reserve_for_answer)) as usize)
        } else {
            // No explicit config and no generator — let the augmenter use its default.
            None
        };

        let budget = TokenBudget {
            max_context_tokens,
            reserve_for_answer: self.config.reserve_for_answer,
        };

        let augmented = self
            .augmenter
            .augment(query, &retrieval.results, &budget)?;

        // --- Stage 3: Generation (optional) ---

        let (answer, generation_metrics) = if generate {
            let generator_ref = self.generator.as_ref().expect(
                "generator checked in ask() before run_pipeline",
            );
            let gen_start = Instant::now();
            let result = generator_ref
                .generate(&augmented.system_prompt, &augmented.context)
                .await?;
            let gen_latency = gen_start.elapsed().as_millis() as u64;

            let metrics = GenerationMetrics {
                latency_ms: gen_latency,
                model: result.model.clone(),
                input_tokens: result.input_tokens,
                output_tokens: result.output_tokens,
            };

            (Some(result.text), Some(metrics))
        } else {
            (None, None)
        };

        // --- Build trace and response ---

        let total_latency_ms = pipeline_start.elapsed().as_millis() as u64;

        let trace = PipelineTrace {
            trace_id,
            retrieval: retrieval.metrics,
            augmentation: augmented.metrics.clone(),
            generation: generation_metrics,
            total_latency_ms,
        };

        info!(total_latency_ms, mode, "rag_query_completed");

        Ok(RagResponse {
            answer,
            context: augmented,
            trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::augmenter::StructuredAugmenter;
    use crate::lite_store::LiteStore;
    use crate::retriever::VectorRetriever;
    use crate::traits::{GenerationEngine, InferenceEngine, QueryableStore};
    use async_trait::async_trait;
    use corvia_common::agent_types::EntryStatus;
    use corvia_common::types::KnowledgeEntry;

    /// Mock embedding engine that returns a fixed 3-dim vector.
    struct MockEngine;

    #[async_trait]
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

    /// Mock generation engine for testing ask() mode.
    struct MockGenerator;

    #[async_trait]
    impl GenerationEngine for MockGenerator {
        fn name(&self) -> &str {
            "mock"
        }
        async fn generate(
            &self,
            _system: &str,
            user: &str,
        ) -> corvia_common::errors::Result<GenerationResult> {
            Ok(GenerationResult {
                text: format!("Generated answer for: {user}"),
                model: "mock-v1".into(),
                input_tokens: user.len() / 4,
                output_tokens: 20,
            })
        }
        fn context_window(&self) -> usize {
            4096
        }
    }

    /// Build a pipeline with 10 test entries and optional generator.
    async fn setup_pipeline(dir: &std::path::Path, with_generator: bool) -> RagPipeline {
        let store =
            Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();

        // Insert 10 entries with varied embeddings for HNSW connectivity.
        let mut idx = 0_usize;
        let mut next_emb = || {
            idx += 1;
            vec![1.0, idx as f32 * 0.001, 0.0]
        };

        for i in 0..10 {
            let mut e = KnowledgeEntry::new(
                format!("knowledge entry {i}"),
                "proj".into(),
                "v1".into(),
            )
            .with_embedding(next_emb());
            e.entry_status = EntryStatus::Merged;
            store.insert(&e).await.unwrap();
        }

        let retriever =
            Arc::new(VectorRetriever::new(store, engine)) as Arc<dyn Retriever>;
        let augmenter =
            Arc::new(StructuredAugmenter::new()) as Arc<dyn Augmenter>;
        let generator: Option<Arc<dyn GenerationEngine>> = if with_generator {
            Some(Arc::new(MockGenerator))
        } else {
            None
        };

        RagPipeline::new(retriever, augmenter, generator, RagConfig::default())
    }

    /// Context-only mode: no answer, non-empty context, sources included,
    /// no generation trace, correct stage names.
    #[tokio::test]
    async fn test_context_mode_returns_no_answer() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = setup_pipeline(dir.path(), false).await;

        let resp = pipeline.context("test query", "proj", None).await.unwrap();

        assert!(resp.answer.is_none(), "context mode should not produce an answer");
        assert!(
            !resp.context.context.is_empty(),
            "context should not be empty"
        );
        assert!(
            resp.context.metrics.sources_included > 0,
            "should include at least one source"
        );
        assert!(
            resp.trace.generation.is_none(),
            "generation trace should be None in context mode"
        );
        assert_eq!(
            pipeline.retriever_name(),
            "vector",
            "retriever name should be VectorRetriever"
        );
        assert_eq!(
            resp.context.metrics.augmenter_name, "structured",
            "augmenter name should be StructuredAugmenter"
        );
    }

    /// Ask mode: answer contains generated text, generation trace has model info.
    #[tokio::test]
    async fn test_ask_mode_returns_generated_answer() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = setup_pipeline(dir.path(), true).await;

        let resp = pipeline.ask("test query", "proj", None).await.unwrap();

        let answer = resp.answer.expect("ask mode should produce an answer");
        assert!(
            answer.contains("Generated answer for:"),
            "answer should contain mock generator output, got: {answer}"
        );

        let gen_trace = resp
            .trace
            .generation
            .expect("generation trace should be present in ask mode");
        assert_eq!(gen_trace.model, "mock-v1", "generation model should be mock-v1");
    }

    /// Ask without a generator configured should return an error.
    #[tokio::test]
    async fn test_ask_without_generator_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = setup_pipeline(dir.path(), false).await;

        let result = pipeline.ask("test query", "proj", None).await;
        assert!(result.is_err(), "ask() without generator should fail");
    }

    /// Each pipeline call should get a unique trace_id (UUID v7).
    #[tokio::test]
    async fn test_pipeline_trace_has_uuid() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = setup_pipeline(dir.path(), false).await;

        let resp1 = pipeline.context("query 1", "proj", None).await.unwrap();
        let resp2 = pipeline.context("query 2", "proj", None).await.unwrap();

        assert_ne!(
            resp1.trace.trace_id, resp2.trace.trace_id,
            "each call should produce a unique trace_id"
        );
    }
}
