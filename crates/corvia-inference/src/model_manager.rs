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
    pub models: Arc<RwLock<HashMap<String, ModelEntry>>>,
}

impl ModelManagerService {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
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
        }))
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
