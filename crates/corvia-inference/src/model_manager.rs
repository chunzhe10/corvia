use crate::backend::{self, GpuCapabilities, ModelType};
use crate::chat_service::ChatServiceImpl;
use crate::embedding_service::EmbeddingServiceImpl;
use corvia_proto::model_manager_server::ModelManager;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

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
    /// Probed GPU capabilities (cached at startup).
    gpu: GpuCapabilities,
}

impl ModelManagerService {
    pub fn new(embed_svc: EmbeddingServiceImpl, chat_svc: ChatServiceImpl, gpu: GpuCapabilities) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            embed_svc,
            chat_svc,
            gpu,
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
        tracing::info!(model = %req.name, model_type = %req.model_type,
            device = %req.device, backend = %req.backend, "load_model requested");

        let model_type = match req.model_type.as_str() {
            "embedding" => ModelType::Embedding,
            "chat" => ModelType::Chat,
            other => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: format!("Unknown model_type: '{other}'. Expected 'embedding' or 'chat'."),
                    actual_device: String::new(),
                    actual_backend: String::new(),
                }));
            }
        };

        // Resolve backend
        let resolved = match backend::resolve_backend(&req.device, &req.backend, model_type, &self.gpu) {
            Ok(r) => r,
            Err(e) => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: e,
                    actual_device: String::new(),
                    actual_backend: String::new(),
                }));
            }
        };

        let actual_device = resolved.device.to_string();
        let actual_backend = resolved.backend.to_string();

        if resolved.fallback_used {
            tracing::warn!(
                model = %req.name,
                requested_device = %req.device,
                actual_device = %actual_device,
                actual_backend = %actual_backend,
                "GPU not available, fell back to CPU"
            );
        }

        // Delegate to appropriate service
        let result = match model_type {
            ModelType::Embedding => self.embed_svc.load_model(&req.name, resolved).await,
            ModelType::Chat => self.chat_svc.load_model(&req.name, resolved).await,
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
                    actual_device,
                    actual_backend,
                }))
            }
            Err(status) => Ok(Response::new(LoadModelResponse {
                success: false,
                error: status.message().to_string(),
                actual_device,
                actual_backend,
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
