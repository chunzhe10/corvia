use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Maximum character length for embedding input. nomic-embed-text has a 2048-token
/// context window; ~4 chars/token gives a safe ceiling.
const MAX_EMBED_CHARS: usize = 4000;

pub struct OllamaEngine {
    url: String,
    model: String,
    dimensions: usize,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
    index: usize,
}

impl OllamaEngine {
    pub fn new(url: &str, model: &str, dimensions: usize) -> Self {
        Self {
            url: url.to_string(),
            model: model.to_string(),
            dimensions,
            client: reqwest::Client::new(),
        }
    }

    /// Check if Ollama is reachable at the configured URL.
    /// Ollama returns "Ollama is running" at GET /.
    pub async fn check_health(url: &str) -> bool {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_default();
        client.get(url).send().await.is_ok()
    }
}

#[async_trait]
impl super::traits::InferenceEngine for OllamaEngine {
    #[tracing::instrument(name = "corvia.entry.embed", skip(self, text))]
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results.into_iter().next()
            .ok_or_else(|| CorviaError::Embedding("Empty embedding response from Ollama".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.url);
        let input: Vec<String> = texts.iter().map(|t| {
            if t.len() > MAX_EMBED_CHARS {
                warn!("Truncating input from {} to {} chars for embedding", t.len(), MAX_EMBED_CHARS);
                t[..MAX_EMBED_CHARS].to_string()
            } else {
                t.clone()
            }
        }).collect();
        let request = EmbedRequest {
            model: self.model.clone(),
            input,
        };

        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| CorviaError::Embedding(format!(
                "Ollama request failed (is Ollama running at {}?): {e}", self.url
            )))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Embedding(format!("Ollama returned {status}: {body}")));
        }

        let embed_response: EmbedResponse = response.json().await
            .map_err(|e| CorviaError::Embedding(format!("Failed to parse Ollama response: {e}")))?;

        // Sort by index to ensure correct ordering
        let mut data = embed_response.data;
        data.sort_by_key(|d| d.index);

        Ok(data.into_iter().map(|d| d.embedding).collect())
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
    fn test_ollama_engine_dimensions() {
        let engine = OllamaEngine::new("http://localhost:11434", "nomic-embed-text", 768);
        assert_eq!(engine.dimensions(), 768);
    }
}
