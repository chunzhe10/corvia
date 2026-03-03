use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::ChatRequest;
use tonic::transport::Channel;

use crate::traits::{GenerationEngine, GenerationResult};

/// Default temperature for merge/reasoning chat calls.
const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Default max tokens for merge/reasoning chat calls.
const DEFAULT_MAX_TOKENS: u32 = 2048;

pub struct GrpcChatEngine {
    endpoint: String,
    model: String,
}

impl GrpcChatEngine {
    pub fn new(endpoint: &str, model: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            model: model.to_string(),
        }
    }

    async fn connect(&self) -> Result<ChatServiceClient<Channel>> {
        let url = if self.endpoint.starts_with("http") {
            self.endpoint.clone()
        } else {
            format!("http://{}", self.endpoint)
        };
        ChatServiceClient::connect(url)
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC chat connect failed: {e}")))
    }
}

#[async_trait]
impl GenerationEngine for GrpcChatEngine {
    fn name(&self) -> &str { "grpc_chat" }

    async fn generate(&self, system_prompt: &str, user_message: &str) -> Result<GenerationResult> {
        let mut client = self.connect().await?;
        let proto_messages = vec![
            corvia_proto::ChatMessage {
                role: "system".into(),
                content: system_prompt.into(),
            },
            corvia_proto::ChatMessage {
                role: "user".into(),
                content: user_message.into(),
            },
        ];

        let request = tonic::Request::new(ChatRequest {
            model: self.model.clone(),
            messages: proto_messages,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
        });

        let response = client.chat(request).await
            .map_err(|e| CorviaError::Infra(format!("gRPC Chat failed: {e}")))?;

        let msg = response.into_inner().message
            .ok_or_else(|| CorviaError::Infra("gRPC Chat response missing message".into()))?;

        Ok(GenerationResult {
            text: msg.content,
            model: self.model.clone(),
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    fn context_window(&self) -> usize { 4096 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_chat_engine_creates() {
        let _engine = GrpcChatEngine::new("http://127.0.0.1:8030", "test-model");
    }

    #[test]
    fn test_grpc_chat_engine_endpoint_formatting() {
        let engine = GrpcChatEngine::new("127.0.0.1:8030", "test-model");
        // Verify endpoint is stored as-is; http:// prefix added in connect()
        assert_eq!(engine.endpoint, "127.0.0.1:8030");
    }
}
