use crate::backend::{BackendKind, ResolvedBackend};
use corvia_proto::embedding_service_server::EmbeddingService;
use corvia_proto::*;
use ort::ep::ExecutionProvider;
use ort::execution_providers::{self, CUDAExecutionProvider, OpenVINOExecutionProvider, ExecutionProviderDispatch};
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

/// Check if the requested EP's provider shared library can be loaded at runtime.
///
/// ORT's `GetAvailableProviders()` only reports *compiled-in* providers — it does
/// NOT detect dynamically-loadable providers like OpenVINO EP (which are loaded
/// via dlopen at session creation time). So we check for the provider .so directly.
fn verify_ep_available(backend: &BackendKind) -> bool {
    match backend {
        BackendKind::Cuda => {
            // CUDA EP: check compiled-in first (cu12 prebuilt includes it),
            // then fall back to checking the provider .so.
            if execution_providers::CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
            {
                return true;
            }
            can_dlopen("libonnxruntime_providers_cuda.so")
                || can_find_library("libonnxruntime_providers_cuda.so")
        }
        BackendKind::OpenVino => {
            // OpenVINO EP is NEVER in GetAvailableProviders() for the cu12 prebuilt.
            // It's loaded at runtime via dlopen — check the .so exists on the library path.
            // We use file-existence check instead of dlopen because the provider's init
            // code references Provider_GetHost (from providers_shared.so), which isn't
            // available until ORT loads it during session creation.
            can_find_library("libonnxruntime_providers_openvino.so")
        }
        BackendKind::Cpu => true,
    }
}

/// Check if a shared library can be loaded via dlopen (same mechanism ORT uses).
fn can_dlopen(lib_name: &str) -> bool {
    use std::ffi::CString;
    let name = CString::new(lib_name).unwrap();
    // RTLD_LAZY: resolve symbols on first use (matches ORT behavior)
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        false
    } else {
        unsafe { libc::dlclose(handle) };
        true
    }
}

/// Check if a shared library exists on the standard library search paths.
/// Safer than dlopen for libraries with init code that depends on other libraries
/// (e.g., providers_openvino.so requires Provider_GetHost from providers_shared.so).
fn can_find_library(lib_name: &str) -> bool {
    // Check common library paths
    let search_paths = [
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/local/bin",
    ];
    for dir in &search_paths {
        if std::path::Path::new(dir).join(lib_name).exists() {
            return true;
        }
    }
    // Also check LD_LIBRARY_PATH
    if let Ok(paths) = std::env::var("LD_LIBRARY_PATH") {
        for dir in paths.split(':') {
            if std::path::Path::new(dir).join(lib_name).exists() {
                return true;
            }
        }
    }
    false
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
    pub async fn load_model(&self, name: &str, mut backend: ResolvedBackend) -> Result<(), Status> {
        let model_enum = Self::resolve_model(name)?;
        let name_owned = name.to_string();
        tracing::info!(model = %name_owned, device = %backend.device, backend_kind = %backend.backend, "Loading embedding model...");

        // Verify the requested EP is actually available in the ORT runtime.
        // ORT silently falls back to CPU if the EP provider library is missing
        // (e.g., libonnxruntime_providers_openvino.so). Detect this upfront.
        if !verify_ep_available(&backend.backend) {
            tracing::warn!(
                model = %name_owned,
                requested_backend = %backend.backend,
                "ONNX Runtime does NOT have the {} EP provider library. \
                 Embedding will fall back to CPU. To fix: install the matching \
                 libonnxruntime_providers_{}.so alongside the binary.",
                backend.backend,
                match backend.backend {
                    BackendKind::Cuda => "cuda",
                    BackendKind::OpenVino => "openvino",
                    BackendKind::Cpu => "cpu",
                }
            );
            backend.backend = BackendKind::Cpu;
            backend.device = crate::backend::Device::Cpu;
            backend.fallback_used = true;
        }

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
        tracing::info!(model = %name_owned, "Embedding model loaded — running canary test...");

        // Run canary embedding to verify the execution provider actually works.
        // This catches silent CPU fallback, broken EP libraries, and other runtime
        // issues that only surface when inference is attempted.
        let canary_models = self.models.clone();
        let canary_name = name_owned.clone();
        let canary_result = tokio::task::spawn_blocking(move || {
            let canary_start = std::time::Instant::now();
            let mut guard = canary_models
                .lock()
                .map_err(|e| format!("Lock poisoned: {e}"))?;
            let loaded = guard
                .get_mut(&canary_name)
                .ok_or_else(|| format!("Model '{canary_name}' disappeared after insert"))?;

            let expected_dims = fastembed::TextEmbedding::get_model_info(&loaded.variant)
                .map(|info| info.dim)
                .unwrap_or(0);

            let result = loaded
                .engine
                .embed(vec!["corvia canary test"], None)
                .map_err(|e| format!("Canary embed failed: {e}"))?;

            let embedding = result
                .into_iter()
                .next()
                .ok_or_else(|| "Canary returned empty result".to_string())?;

            let latency = canary_start.elapsed();
            Ok::<_, String>((embedding, expected_dims, latency))
        })
        .await
        .map_err(|e| Status::internal(format!("Canary spawn failed: {e}")));

        match canary_result {
            Ok(Ok((embedding, expected_dims, latency))) => {
                // Verify the embedding is non-zero (valid)
                let is_nonzero = embedding.iter().any(|&v| v != 0.0);
                let actual_dims = embedding.len();

                if !is_nonzero {
                    tracing::warn!(
                        model = %name_owned,
                        "Canary embedding is all zeros — EP may not be working correctly"
                    );
                }

                if expected_dims > 0 && actual_dims != expected_dims {
                    tracing::warn!(
                        model = %name_owned,
                        expected_dims,
                        actual_dims,
                        "Canary embedding dimensions mismatch"
                    );
                }

                if latency.as_secs() > 5 {
                    tracing::warn!(
                        model = %name_owned,
                        latency_ms = latency.as_millis() as u64,
                        "Canary embedding latency >5s — possible CPU fallback or cold start"
                    );
                }

                tracing::info!(
                    model = %name_owned,
                    latency_ms = latency.as_millis() as u64,
                    dimensions = actual_dims,
                    nonzero = is_nonzero,
                    "Canary embedding verified"
                );
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    model = %name_owned,
                    error = %e,
                    "Canary embedding failed — model loaded but inference may not work"
                );
            }
            Err(e) => {
                tracing::warn!(
                    model = %name_owned,
                    error = %e,
                    "Canary embedding task failed — model loaded but inference may not work"
                );
            }
        }

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

    #[test]
    fn test_verify_ep_available_cpu_always_true() {
        assert!(verify_ep_available(&BackendKind::Cpu));
    }

    #[test]
    fn test_verify_ep_available_cuda() {
        // CUDA EP: the cu12 prebuilt includes the provider .so.
        // Don't assert true/false — just verify it doesn't panic.
        let _ = verify_ep_available(&BackendKind::Cuda);
    }

    #[test]
    fn test_verify_ep_available_openvino() {
        // OpenVINO EP availability depends on whether
        // libonnxruntime_providers_openvino.so is installed on the system.
        // Don't assert true/false — just verify it doesn't panic.
        let _ = verify_ep_available(&BackendKind::OpenVino);
    }

    #[test]
    fn test_can_dlopen_nonexistent() {
        assert!(!can_dlopen("libdoes_not_exist_12345.so"));
    }

    #[test]
    fn test_can_dlopen_libc() {
        // libc.so.6 is always present on Linux
        assert!(can_dlopen("libc.so.6"));
    }

    #[test]
    fn test_can_find_library_existing() {
        // libc is always in /usr/lib/x86_64-linux-gnu/
        assert!(can_find_library("libc.so.6"));
    }

    #[test]
    fn test_can_find_library_nonexistent() {
        assert!(!can_find_library("libdoes_not_exist_12345.so"));
    }
}
