//! OllamaChatEngine — ChatEngine implementation that calls Ollama's `/api/chat` endpoint.
//!
//! Extracted from the inline HTTP logic in `merge_worker.rs` to support trait-based
//! chat providers (D60). This is the default ChatEngine for LiteStore deployments.

use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::ChatMessage;

/// ChatEngine backed by Ollama's HTTP `/api/chat` endpoint.
pub struct OllamaChatEngine {
    url: String,
    client: reqwest::Client,
}

impl OllamaChatEngine {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl super::traits::ChatEngine for OllamaChatEngine {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String> {
        let api_messages: Vec<serde_json::Value> = messages.iter().map(|m| {
            serde_json::json!({ "role": m.role, "content": m.content })
        }).collect();

        let request_body = serde_json::json!({
            "model": model,
            "messages": api_messages,
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

        body["message"]["content"]
            .as_str()
            .ok_or_else(|| CorviaError::Agent("LLM response missing message.content".into()))
            .map(|s| s.to_string())
    }
}
