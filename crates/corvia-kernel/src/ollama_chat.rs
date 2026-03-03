//! OllamaChatEngine — GenerationEngine implementation that calls Ollama's `/api/chat` endpoint.
//!
//! Extracted from the inline HTTP logic in `merge_worker.rs` to support trait-based
//! generation providers (D63). This is the default GenerationEngine for LiteStore deployments.

use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};

use crate::traits::{GenerationEngine, GenerationResult};

/// GenerationEngine backed by Ollama's HTTP `/api/chat` endpoint.
pub struct OllamaChatEngine {
    url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaChatEngine {
    pub fn new(url: &str, model: &str) -> Self {
        Self {
            url: url.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl GenerationEngine for OllamaChatEngine {
    fn name(&self) -> &str { "ollama_chat" }

    async fn generate(&self, system_prompt: &str, user_message: &str) -> Result<GenerationResult> {
        let messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
            serde_json::json!({"role": "user", "content": user_message}),
        ];

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false
        });

        let response = self.client
            .post(format!("{}/api/chat", self.url))
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| CorviaError::Agent(format!("LLM chat request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Agent(format!("LLM chat failed with status {status}: {body}")));
        }

        let body: serde_json::Value = response.json().await
            .map_err(|e| CorviaError::Agent(format!("Failed to parse LLM response: {e}")))?;

        let text = body["message"]["content"]
            .as_str()
            .ok_or_else(|| CorviaError::Agent("LLM response missing message.content".into()))?
            .to_string();

        Ok(GenerationResult {
            text,
            model: self.model.clone(),
            input_tokens: 0,  // Ollama doesn't always report token counts
            output_tokens: 0,
        })
    }

    fn context_window(&self) -> usize { 4096 }
}
