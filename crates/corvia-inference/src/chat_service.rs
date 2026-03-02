use corvia_proto::chat_service_server::ChatService;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

/// Stub ChatService implementation.
/// Currently returns stub responses for actual model inference.
/// The candle GGUF backend will be implemented in a future task.
#[derive(Clone)]
pub struct ChatServiceImpl {
    models: Arc<RwLock<HashMap<String, ChatModelEntry>>>,
}

struct ChatModelEntry {
    _name: String,
}

impl ChatServiceImpl {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn load_model(&self, name: &str) -> Result<(), Status> {
        tracing::info!(model = %name, "Loading chat model (stub)...");
        // TODO: Implement candle GGUF model loading
        // 1. Resolve HuggingFace repo
        // 2. Download via hf-hub if not cached
        // 3. Load GGUF into candle
        // 4. Load tokenizer
        let mut models = self.models.write().await;
        models.insert(
            name.to_string(),
            ChatModelEntry {
                _name: name.to_string(),
            },
        );
        tracing::info!(model = %name, "Chat model registered (stub — no real inference yet)");
        Ok(())
    }
}

#[tonic::async_trait]
impl ChatService for ChatServiceImpl {
    async fn chat(&self, req: Request<ChatRequest>) -> Result<Response<ChatResponse>, Status> {
        let req = req.into_inner();
        let models = self.models.read().await;
        if !models.contains_key(&req.model) {
            return Err(Status::not_found(format!(
                "Chat model '{}' not loaded",
                req.model
            )));
        }

        // TODO: Implement real candle inference
        // For now, return a stub response indicating the service is scaffolded
        let last_user_msg = req
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default();

        Ok(Response::new(ChatResponse {
            message: Some(ChatMessage {
                role: "assistant".to_string(),
                content: format!(
                    "[stub] Chat inference not yet implemented. Received: {}",
                    &last_user_msg[..last_user_msg.len().min(100)]
                ),
            }),
            prompt_tokens: 0,
            completion_tokens: 0,
        }))
    }

    type ChatStreamStream = tokio_stream::wrappers::ReceiverStream<Result<ChatChunk, Status>>;

    async fn chat_stream(
        &self,
        req: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        let req = req.into_inner();
        let models = self.models.read().await;
        if !models.contains_key(&req.model) {
            return Err(Status::not_found(format!(
                "Chat model '{}' not loaded",
                req.model
            )));
        }

        // TODO: Implement real streaming candle inference
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(ChatChunk {
                    delta: "[stub] Streaming chat not yet implemented".to_string(),
                    done: true,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                }))
                .await;
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }
}
