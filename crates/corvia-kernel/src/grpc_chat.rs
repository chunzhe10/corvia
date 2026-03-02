use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::ChatMessage;
use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::ChatRequest;
use tonic::transport::Channel;

/// Default temperature for merge/reasoning chat calls.
const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Default max tokens for merge/reasoning chat calls.
const DEFAULT_MAX_TOKENS: u32 = 2048;

pub struct GrpcChatEngine {
    endpoint: String,
}

impl GrpcChatEngine {
    pub fn new(endpoint: &str) -> Self {
        Self { endpoint: endpoint.to_string() }
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
impl super::traits::ChatEngine for GrpcChatEngine {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String> {
        let mut client = self.connect().await?;
        let proto_messages: Vec<corvia_proto::ChatMessage> = messages.iter().map(|m| {
            corvia_proto::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            }
        }).collect();

        let request = tonic::Request::new(ChatRequest {
            model: model.to_string(),
            messages: proto_messages,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
        });

        let response = client.chat(request).await
            .map_err(|e| CorviaError::Infra(format!("gRPC Chat failed: {e}")))?;

        let msg = response.into_inner().message
            .ok_or_else(|| CorviaError::Infra("gRPC Chat response missing message".into()))?;
        Ok(msg.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_chat_engine_creates() {
        let _engine = GrpcChatEngine::new("http://127.0.0.1:8030");
    }

    #[test]
    fn test_grpc_chat_engine_endpoint_formatting() {
        let engine = GrpcChatEngine::new("127.0.0.1:8030");
        // Verify endpoint is stored as-is; http:// prefix added in connect()
        assert_eq!(engine.endpoint, "127.0.0.1:8030");
    }
}
