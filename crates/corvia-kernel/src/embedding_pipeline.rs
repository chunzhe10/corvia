use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use tracing::warn;

/// Maximum character length for embedding input. nomic-embed-text has a 2048-token
/// context window; ~4 chars/token gives a safe ceiling.
const MAX_EMBED_CHARS: usize = 4000;

pub struct VllmEngine {
    url: String,
    model: String,
    dimensions: usize,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<Cow<'a, str>>,
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

impl VllmEngine {
    pub fn new(url: &str, model: &str, dimensions: usize) -> Self {
        Self {
            url: url.to_string(),
            model: model.to_string(),
            dimensions,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl super::traits::InferenceEngine for VllmEngine {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results.into_iter().next()
            .ok_or_else(|| CorviaError::Embedding("Empty embedding response".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.url);
        let input: Vec<Cow<'_, str>> = texts.iter().map(|t| {
            if t.len() > MAX_EMBED_CHARS {
                warn!("Truncating input from {} to {} chars for embedding", t.len(), MAX_EMBED_CHARS);
                Cow::Owned(t[..MAX_EMBED_CHARS].to_string())
            } else {
                Cow::Borrowed(t.as_str())
            }
        }).collect();
        let request = EmbedRequest {
            model: &self.model,
            input,
        };

        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| CorviaError::Embedding(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Embedding(format!("vLLM returned {status}: {body}")));
        }

        let embed_response: EmbedResponse = response.json().await
            .map_err(|e| CorviaError::Embedding(format!("Failed to parse response: {e}")))?;

        // Sort by index to ensure correct ordering
        let mut data = embed_response.data;
        data.sort_by_key(|d| d.index);

        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
