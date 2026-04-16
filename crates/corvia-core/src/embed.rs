//! Embedding via fastembed-rs.
//! Default model: nomic-embed-text-v1.5 (768d).
//! Lite mode: all-MiniLM-L6-v2 (384d).

use std::path::Path;

use anyhow::Result;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankResult, RerankerModel, TextEmbedding,
    TextRerank,
};

/// Wrapper around fastembed's TextEmbedding and TextRerank models.
///
/// Provides a simple interface for embedding text and reranking documents.
pub struct Embedder {
    embedding: TextEmbedding,
    reranker: TextRerank,
}

/// Resolve a config string to the fastembed `RerankerModel` enum.
///
/// Defaults to `JINARerankerV1TurboEn` for unrecognized names.
pub fn resolve_reranker(name: &str) -> RerankerModel {
    match name {
        "bge-base" => RerankerModel::BGERerankerBase,
        "bge-v2-m3" => RerankerModel::BGERerankerV2M3,
        "jina-v2-multilingual" => RerankerModel::JINARerankerV2BaseMultiligual,
        _ => RerankerModel::JINARerankerV1TurboEn, // default
    }
}

/// Resolve a config string to the fastembed `EmbeddingModel` enum.
///
/// Maps common model name strings to their fastembed variants.
/// Defaults to `NomicEmbedTextV15` for unrecognized names.
pub fn resolve_embedding_model(name: &str) -> EmbeddingModel {
    match name {
        "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
        "nomic-embed-text-v1.5-q" => EmbeddingModel::NomicEmbedTextV15Q,
        "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
        "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
        "all-MiniLM-L6-v2-q" => EmbeddingModel::AllMiniLML6V2Q,
        "all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
        "all-MiniLM-L12-v2-q" => EmbeddingModel::AllMiniLML12V2Q,
        "bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "bge-small-en-v1.5-q" => EmbeddingModel::BGESmallENV15Q,
        "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "bge-base-en-v1.5-q" => EmbeddingModel::BGEBaseENV15Q,
        "bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "bge-large-en-v1.5-q" => EmbeddingModel::BGELargeENV15Q,
        "mxbai-embed-large-v1" => EmbeddingModel::MxbaiEmbedLargeV1,
        "mxbai-embed-large-v1-q" => EmbeddingModel::MxbaiEmbedLargeV1Q,
        "gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15,
        "gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15,
        "multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
        "multilingual-e5-base" => EmbeddingModel::MultilingualE5Base,
        "multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
        _ => EmbeddingModel::NomicEmbedTextV15, // default
    }
}

impl Embedder {
    /// Initialize embedding and reranker models.
    ///
    /// Downloads models on first use with progress output.
    /// Pass `cache_dir` to control where model files are stored.
    /// Pass `embedding_model_name` to select the embedding model (see [`resolve_embedding_model`]).
    /// Pass `reranker_name` to select the reranker model (see [`resolve_reranker`]).
    pub fn new(cache_dir: Option<&Path>, embedding_model_name: &str, reranker_name: &str) -> Result<Self> {
        let embedding_model = resolve_embedding_model(embedding_model_name);
        let reranker_model = resolve_reranker(reranker_name);

        let mut embed_opts =
            InitOptions::new(embedding_model).with_show_download_progress(true);
        let mut rerank_opts =
            RerankInitOptions::new(reranker_model).with_show_download_progress(true);

        if let Some(dir) = cache_dir {
            embed_opts = embed_opts.with_cache_dir(dir.to_path_buf());
            rerank_opts = rerank_opts.with_cache_dir(dir.to_path_buf());
        }

        let embedding = TextEmbedding::try_new(embed_opts)?;
        let reranker = TextRerank::try_new(rerank_opts)?;

        Ok(Self {
            embedding,
            reranker,
        })
    }

    /// Embed a single text, returning its vector representation.
    #[tracing::instrument(name = "corvia.embed", skip(self, text), fields(text_len = text.len()))]
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embedding.embed(vec![text], None)?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding produced no output"))
    }

    /// Embed multiple texts in a single batch call.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let owned: Vec<&str> = texts.to_vec();
        self.embedding.embed(owned, None)
    }

    /// Rerank documents against a query, returning the top_n results sorted by relevance.
    #[tracing::instrument(name = "corvia.rerank", skip(self, query, documents), fields(candidate_count = documents.len(), top_n))]
    pub fn rerank(&self, query: &str, documents: &[&str], top_n: usize) -> Result<Vec<RerankResult>> {
        let docs: Vec<&str> = documents.to_vec();
        let mut results = self.reranker.rerank(query, docs, true, None)?;
        results.truncate(top_n);
        Ok(results)
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns 0.0 if either vector has zero magnitude.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_embedding_model_known_names() {
        assert_eq!(
            resolve_embedding_model("nomic-embed-text-v1.5"),
            EmbeddingModel::NomicEmbedTextV15,
        );
        assert_eq!(
            resolve_embedding_model("all-MiniLM-L6-v2"),
            EmbeddingModel::AllMiniLML6V2,
        );
        assert_eq!(
            resolve_embedding_model("bge-small-en-v1.5"),
            EmbeddingModel::BGESmallENV15,
        );
        assert_eq!(
            resolve_embedding_model("mxbai-embed-large-v1"),
            EmbeddingModel::MxbaiEmbedLargeV1,
        );
    }

    #[test]
    fn resolve_embedding_model_unknown_defaults() {
        assert_eq!(
            resolve_embedding_model("unknown-model"),
            EmbeddingModel::NomicEmbedTextV15,
        );
        assert_eq!(
            resolve_embedding_model(""),
            EmbeddingModel::NomicEmbedTextV15,
        );
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = Embedder::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected ~1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = Embedder::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected ~0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(Embedder::cosine_similarity(&a, &zero), 0.0);
        assert_eq!(Embedder::cosine_similarity(&zero, &a), 0.0);
        assert_eq!(Embedder::cosine_similarity(&zero, &zero), 0.0);
    }

    #[test]
    #[ignore]
    fn embed_produces_correct_dimensions() {
        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");
        let vec = embedder.embed("test text").expect("failed to embed");
        assert_eq!(vec.len(), 768, "nomic-embed-text-v1.5 should produce 768d vectors");
    }

    #[test]
    #[ignore]
    fn embed_batch_produces_correct_count() {
        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");
        let texts = &["first text", "second text", "third text"];
        let vecs = embedder.embed_batch(texts).expect("failed to embed batch");
        assert_eq!(vecs.len(), 3, "should produce one vector per input text");
    }

    #[test]
    #[ignore]
    fn similar_texts_have_high_similarity() {
        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");
        let rust_prog = embedder.embed("Rust programming").unwrap();
        let rust_sys = embedder.embed("Rust systems").unwrap();
        let chocolate = embedder.embed("chocolate cake recipe").unwrap();

        let sim_related = Embedder::cosine_similarity(&rust_prog, &rust_sys);
        let sim_unrelated = Embedder::cosine_similarity(&rust_prog, &chocolate);

        assert!(
            sim_related > sim_unrelated,
            "related texts ({sim_related}) should be more similar than unrelated ({sim_unrelated})"
        );
    }
}
