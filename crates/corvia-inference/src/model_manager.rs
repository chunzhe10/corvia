use corvia_proto::model_manager_server::ModelManager;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::chat_service::ChatServiceImpl;
use crate::embedding_service::EmbeddingServiceImpl;

#[derive(Clone)]
pub struct ModelEntry {
    pub name: String,
    pub model_type: String,
    pub loaded: bool,
}

pub struct ModelManagerService {
    /// Registry tracking all loaded models (for health/list).
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,
    /// Delegates embedding model loads to the actual EmbeddingService.
    embed_svc: EmbeddingServiceImpl,
    /// Delegates chat model loads to the actual ChatService.
    chat_svc: ChatServiceImpl,
}

impl ModelManagerService {
    pub fn new(embed_svc: EmbeddingServiceImpl, chat_svc: ChatServiceImpl) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            embed_svc,
            chat_svc,
        }
    }
}

#[tonic::async_trait]
impl ModelManager for ModelManagerService {
    async fn health(
        &self,
        _req: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let models = self.models.read().await;
        Ok(Response::new(HealthResponse {
            healthy: true,
            models_loaded: models.values().filter(|m| m.loaded).count() as u32,
        }))
    }

    async fn list_models(
        &self,
        _req: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let models = self.models.read().await;
        let statuses = models
            .values()
            .map(|m| ModelStatus {
                name: m.name.clone(),
                model_type: m.model_type.clone(),
                loaded: m.loaded,
                memory_bytes: 0,
            })
            .collect();
        Ok(Response::new(ListModelsResponse { models: statuses }))
    }

    async fn load_model(
        &self,
        req: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = req.into_inner();
        tracing::info!(model = %req.name, model_type = %req.model_type, "load_model requested");

        // Delegate to the appropriate service based on model_type
        let result = match req.model_type.as_str() {
            "embedding" => self.embed_svc.load_model(&req.name).await,
            "chat" => self.chat_svc.load_model(&req.name).await,
            other => Err(Status::invalid_argument(format!(
                "Unknown model_type: '{other}'. Expected 'embedding' or 'chat'."
            ))),
        };

        match result {
            Ok(()) => {
                let mut models = self.models.write().await;
                models.insert(
                    req.name.clone(),
                    ModelEntry {
                        name: req.name,
                        model_type: req.model_type,
                        loaded: true,
                    },
                );
                Ok(Response::new(LoadModelResponse {
                    success: true,
                    error: String::new(),
                    actual_device: String::new(),
                    actual_backend: String::new(),
                }))
            }
            Err(status) => Ok(Response::new(LoadModelResponse {
                success: false,
                error: status.message().to_string(),
                actual_device: String::new(),
                actual_backend: String::new(),
            })),
        }
    }

    async fn unload_model(
        &self,
        req: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        let name = req.into_inner().name;
        let mut models = self.models.write().await;
        models.remove(&name);
        Ok(Response::new(UnloadModelResponse { success: true }))
    }
}
