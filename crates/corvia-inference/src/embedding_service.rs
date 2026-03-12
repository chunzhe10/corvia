use crate::backend::{BackendKind, ResolvedBackend};
use corvia_proto::embedding_service_server::EmbeddingService;
use corvia_proto::*;
use ort::execution_providers::{CUDAExecutionProvider, OpenVINOExecutionProvider, ExecutionProviderDispatch};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Build ONNX Runtime execution providers based on resolved backend.
fn build_execution_providers(backend: &ResolvedBackend) -> Vec<ExecutionProviderDispatch> {
    match backend.backend {
        BackendKind::Cuda => vec![CUDAExecutionProvider::default().build()],
        BackendKind::OpenVino => vec![
            OpenVINOExecutionProvider::default()
                .with_device_type("GPU")
                .build(),
        ],
        BackendKind::Cpu => vec![], // ort defaults to CPU EP
    }
}

/// A loaded embedding model with its enum variant for metadata lookups.
struct LoadedModel {
    engine: fastembed::TextEmbedding,
    variant: fastembed::EmbeddingModel,
    backend: ResolvedBackend,
}

#[derive(Clone)]
pub struct EmbeddingServiceImpl {
    // std::sync::Mutex (not tokio) so we can hold the lock inside spawn_blocking.
    // fastembed's embed() is CPU-bound and requires &mut self.
    models: Arc<std::sync::Mutex<HashMap<String, LoadedModel>>>,
}

impl EmbeddingServiceImpl {
    pub fn new() -> Self {
        Self {
            models: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Resolve a model name to a fastembed EmbeddingModel enum variant.
    pub fn resolve_model(name: &str) -> Result<fastembed::EmbeddingModel, Status> {
        match name {
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                Ok(fastembed::EmbeddingModel::NomicEmbedTextV15)
            }
            "nomic-embed-text-v1" | "nomic-ai/nomic-embed-text-v1" => {
                Ok(fastembed::EmbeddingModel::NomicEmbedTextV1)
            }
            "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => {
                Ok(fastembed::EmbeddingModel::BGESmallENV15)
            }
            "bge-base-en-v1.5" | "BAAI/bge-base-en-v1.5" => {
                Ok(fastembed::EmbeddingModel::BGEBaseENV15)
            }
            "bge-large-en-v1.5" | "BAAI/bge-large-en-v1.5" => {
                Ok(fastembed::EmbeddingModel::BGELargeENV15)
            }
            "all-MiniLM-L6-v2" | "sentence-transformers/all-MiniLM-L6-v2" => {
                Ok(fastembed::EmbeddingModel::AllMiniLML6V2)
            }
            "all-MiniLM-L12-v2" | "sentence-transformers/all-MiniLM-L12-v2" => {
                Ok(fastembed::EmbeddingModel::AllMiniLML12V2)
            }
            other => Err(Status::not_found(format!("Unknown embedding model: {other}"))),
        }
    }

    /// Load an embedding model by name. Downloads from HuggingFace if not cached.
    pub async fn load_model(&self, name: &str, backend: ResolvedBackend) -> Result<(), Status> {
        let model_enum = Self::resolve_model(name)?;
        let name_owned = name.to_string();
        tracing::info!(model = %name_owned, device = %backend.device, backend_kind = %backend.backend, "Loading embedding model...");

        let eps = build_execution_providers(&backend);
        let model_enum_for_spawn = model_enum.clone();
        let engine = tokio::task::spawn_blocking(move || {
            fastembed::TextEmbedding::try_new(
                fastembed::InitOptions::new(model_enum_for_spawn)
                    .with_show_download_progress(true)
                    .with_execution_providers(eps),
            )
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        self.models
            .lock()
            .map_err(|e| Status::internal(format!("Lock poisoned: {e}")))?
            .insert(
                name_owned.clone(),
                LoadedModel {
                    engine,
                    variant: model_enum,
                    backend,
                },
            );
        tracing::info!(model = %name_owned, "Embedding model loaded");
        Ok(())
    }

    /// Get the resolved backend for a loaded model.
    pub fn get_backend(&self, name: &str) -> Option<ResolvedBackend> {
        self.models
            .lock()
            .ok()?
            .get(name)
            .map(|m| m.backend.clone())
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed(&self, req: Request<EmbedRequest>) -> Result<Response<EmbedResponse>, Status> {
        let req = req.into_inner();
        let models = self.models.clone();
        let model_name = req.model;
        let text = req.text;

        // Run the CPU-intensive ONNX inference on a blocking thread
        let embedding = tokio::task::spawn_blocking(move || {
            let mut guard = models
                .lock()
                .map_err(|e| Status::internal(format!("Lock poisoned: {e}")))?;
            let loaded = guard
                .get_mut(&model_name)
                .ok_or_else(|| Status::not_found(format!("Model '{model_name}' not loaded")))?;

            let texts = vec![text.as_str()];
            let embeddings = loaded
                .engine
                .embed(texts, None)
                .map_err(|e| Status::internal(format!("Embed failed: {e}")))?;

            embeddings
                .into_iter()
                .next()
                .ok_or_else(|| Status::internal("Empty embedding result"))
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        ?;

        let dimensions = embedding.len() as u32;
        Ok(Response::new(EmbedResponse {
            embedding,
            dimensions,
        }))
    }

    async fn embed_batch(
        &self,
        req: Request<EmbedBatchRequest>,
    ) -> Result<Response<EmbedBatchResponse>, Status> {
        let req = req.into_inner();
        let models = self.models.clone();
        let model_name = req.model;
        let texts = req.texts;

        // Run the CPU-intensive ONNX inference on a blocking thread
        let embeddings_raw = tokio::task::spawn_blocking(move || {
            let mut guard = models
                .lock()
                .map_err(|e| Status::internal(format!("Lock poisoned: {e}")))?;
            let loaded = guard
                .get_mut(&model_name)
                .ok_or_else(|| Status::not_found(format!("Model '{model_name}' not loaded")))?;

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            loaded
                .engine
                .embed(text_refs, None)
                .map_err(|e| Status::internal(format!("EmbedBatch failed: {e}")))
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        ?;

        let embeddings = embeddings_raw
            .into_iter()
            .enumerate()
            .map(|(i, values)| Embedding {
                values,
                index: i as u32,
            })
            .collect();

        Ok(Response::new(EmbedBatchResponse { embeddings }))
    }

    async fn model_info(
        &self,
        req: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        let name = req.into_inner().model;
        let models = self
            .models
            .lock()
            .map_err(|e| Status::internal(format!("Lock poisoned: {e}")))?;
        let loaded = models.contains_key(&name);

        let dimensions = if let Some(entry) = models.get(&name) {
            fastembed::TextEmbedding::get_model_info(&entry.variant)
                .map(|info| info.dim as u32)
                .unwrap_or(0)
        } else {
            0
        };

        Ok(Response::new(ModelInfoResponse {
            model: name,
            dimensions,
            loaded,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, Device, ResolvedBackend};

    #[test]
    fn test_build_execution_providers_cpu() {
        let backend = ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert!(eps.is_empty(), "CPU backend should produce no EPs (ort defaults to CPU)");
    }

    #[test]
    fn test_build_execution_providers_cuda() {
        let backend = ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::Cuda,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert_eq!(eps.len(), 1, "CUDA backend should produce exactly one EP");
    }

    #[test]
    fn test_build_execution_providers_openvino() {
        let backend = ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::OpenVino,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert_eq!(eps.len(), 1, "OpenVINO backend should produce exactly one EP");
    }
}
