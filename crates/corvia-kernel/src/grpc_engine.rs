use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_proto::embedding_service_client::EmbeddingServiceClient;
use corvia_proto::{EmbedBatchRequest, EmbedRequest};
use tonic::transport::Channel;
use tracing::warn;

const MAX_EMBED_CHARS: usize = 4000;

pub struct GrpcInferenceEngine {
    endpoint: String,
    model: String,
    dimensions: usize,
}

impl GrpcInferenceEngine {
    pub fn new(endpoint: &str, model: &str, dimensions: usize) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    async fn connect(&self) -> Result<EmbeddingServiceClient<Channel>> {
        let url = if self.endpoint.starts_with("http") {
            self.endpoint.clone()
        } else {
            format!("http://{}", self.endpoint)
        };
        EmbeddingServiceClient::connect(url)
            .await
            .map_err(|e| CorviaError::Embedding(format!("gRPC connect failed: {e}")))
    }

    fn truncate(text: &str) -> String {
        if text.len() > MAX_EMBED_CHARS {
            warn!("Truncating input from {} to {} bytes", text.len(), MAX_EMBED_CHARS);
            // Find the last char boundary at or before MAX_EMBED_CHARS to avoid
            // panicking on multi-byte UTF-8 (e.g. CJK, emoji).
            let mut end = MAX_EMBED_CHARS;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text[..end].to_string()
        } else {
            text.to_string()
        }
    }
}

#[async_trait]
impl super::traits::InferenceEngine for GrpcInferenceEngine {
    #[tracing::instrument(name = "corvia.entry.embed", skip(self, text))]
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut client = self.connect().await?;
        let request = tonic::Request::new(EmbedRequest {
            model: self.model.clone(),
            text: Self::truncate(text),
        });
        let response = client.embed(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC Embed failed: {e}")))?;
        Ok(response.into_inner().embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut client = self.connect().await?;
        let request = tonic::Request::new(EmbedBatchRequest {
            model: self.model.clone(),
            texts: texts.iter().map(|t| Self::truncate(t)).collect(),
        });
        let response = client.embed_batch(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC EmbedBatch failed: {e}")))?;
        let mut embeddings: Vec<_> = response.into_inner().embeddings;
        embeddings.sort_by_key(|e| e.index);
        Ok(embeddings.into_iter().map(|e| e.values).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::InferenceEngine;

    #[test]
    fn test_grpc_engine_dimensions() {
        let engine = GrpcInferenceEngine::new("http://127.0.0.1:8030", "test-model", 768);
        assert_eq!(engine.dimensions(), 768);
    }

    #[test]
    fn test_truncate_short_text() {
        let text = "hello world";
        assert_eq!(GrpcInferenceEngine::truncate(text), "hello world");
    }

    #[test]
    fn test_truncate_long_text() {
        let text = "a".repeat(5000);
        let truncated = GrpcInferenceEngine::truncate(&text);
        assert_eq!(truncated.len(), MAX_EMBED_CHARS);
    }

    #[test]
    fn test_truncate_multibyte_utf8() {
        // Each CJK character is 3 bytes. Build a string where the byte limit
        // falls in the middle of a character — must not panic.
        let text = "\u{4e16}".repeat(2000); // 6000 bytes of 3-byte chars
        let truncated = GrpcInferenceEngine::truncate(&text);
        assert!(truncated.len() <= MAX_EMBED_CHARS);
        // Must be a multiple of 3 (each char is 3 bytes) — proves we didn't split mid-char
        assert_eq!(truncated.len() % 3, 0);
    }
}
