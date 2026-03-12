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
    pub device: String,
    pub backend: String,
}

pub struct ModelManagerService {
    /// Registry tracking all loaded models (for health/list).
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,
    /// Delegates embedding model loads to the actual EmbeddingService.
    embed_svc: EmbeddingServiceImpl,
    /// Delegates chat model loads to the actual ChatService.
    chat_svc: ChatServiceImpl,
    /// Probed GPU capabilities (re-probeable at runtime).
    gpu: Arc<std::sync::RwLock<GpuCapabilities>>,
}

impl ModelManagerService {
    pub fn new(embed_svc: EmbeddingServiceImpl, chat_svc: ChatServiceImpl, gpu: Arc<std::sync::RwLock<GpuCapabilities>>) -> Self {
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
                device: m.device.clone(),
                backend: m.backend.clone(),
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

        // Resolve backend (scope lock before any .await)
        let resolved = {
            let gpu = self.gpu.read().map_err(|e| Status::internal(format!("GPU lock poisoned: {e}")))?;
            match backend::resolve_backend(&req.device, &req.backend, model_type, &gpu) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(Response::new(LoadModelResponse {
                        success: false,
                        error: e,
                        actual_device: String::new(),
                        actual_backend: String::new(),
                    }));
                }
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
                        device: actual_device.clone(),
                        backend: actual_backend.clone(),
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

    async fn reload_models(
        &self,
        req: Request<ReloadModelsRequest>,
    ) -> Result<Response<ReloadModelsResponse>, Status> {
        let req = req.into_inner();
        tracing::info!(device = %req.device, backend = %req.backend, reprobe = req.reprobe_gpu, "reload_models requested");

        // Optionally re-probe GPU capabilities
        if req.reprobe_gpu {
            let new_gpu = GpuCapabilities::probe();
            tracing::info!(cuda = new_gpu.cuda_available, openvino = new_gpu.openvino_available, "GPU capabilities re-probed");
            let mut gpu = self.gpu.write().map_err(|e| Status::internal(format!("GPU lock poisoned: {e}")))?;
            *gpu = new_gpu;
        }

        // Snapshot currently loaded models
        let snapshot: Vec<ModelEntry> = {
            let models = self.models.read().await;
            models.values().filter(|m| m.loaded).cloned().collect()
        };

        if snapshot.is_empty() {
            return Ok(Response::new(ReloadModelsResponse {
                success: true,
                error: String::new(),
                results: vec![],
            }));
        }

        let mut results = Vec::new();
        let mut all_success = true;

        for entry in &snapshot {
            let model_type = match entry.model_type.as_str() {
                "embedding" => ModelType::Embedding,
                "chat" => ModelType::Chat,
                _ => {
                    results.push(ModelReloadResult {
                        name: entry.name.clone(),
                        model_type: entry.model_type.clone(),
                        success: false,
                        error: format!("Unknown model_type: '{}'", entry.model_type),
                        actual_device: String::new(),
                        actual_backend: String::new(),
                    });
                    all_success = false;
                    continue;
                }
            };

            let resolved = {
                let gpu = self.gpu.read().map_err(|e| Status::internal(format!("GPU lock poisoned: {e}")))?;
                match backend::resolve_backend(&req.device, &req.backend, model_type, &gpu) {
                    Ok(r) => r,
                    Err(e) => {
                        results.push(ModelReloadResult {
                            name: entry.name.clone(),
                            model_type: entry.model_type.clone(),
                            success: false,
                            error: e,
                            actual_device: String::new(),
                            actual_backend: String::new(),
                        });
                        all_success = false;
                        continue;
                    }
                }
            };

            let actual_device = resolved.device.to_string();
            let actual_backend = resolved.backend.to_string();

            tracing::info!(model = %entry.name, device = %actual_device, backend = %actual_backend, "Reloading model...");

            let load_result = match model_type {
                ModelType::Embedding => self.embed_svc.load_model(&entry.name, resolved).await,
                ModelType::Chat => self.chat_svc.load_model(&entry.name, resolved).await,
            };

            match load_result {
                Ok(()) => {
                    let mut models = self.models.write().await;
                    models.insert(
                        entry.name.clone(),
                        ModelEntry {
                            name: entry.name.clone(),
                            model_type: entry.model_type.clone(),
                            loaded: true,
                            device: actual_device.clone(),
                            backend: actual_backend.clone(),
                        },
                    );
                    results.push(ModelReloadResult {
                        name: entry.name.clone(),
                        model_type: entry.model_type.clone(),
                        success: true,
                        error: String::new(),
                        actual_device,
                        actual_backend,
                    });
                }
                Err(status) => {
                    all_success = false;
                    results.push(ModelReloadResult {
                        name: entry.name.clone(),
                        model_type: entry.model_type.clone(),
                        success: false,
                        error: status.message().to_string(),
                        actual_device,
                        actual_backend,
                    });
                }
            }
        }

        Ok(Response::new(ReloadModelsResponse {
            success: all_success,
            error: if all_success { String::new() } else { "Some models failed to reload".into() },
            results,
        }))
    }
}
