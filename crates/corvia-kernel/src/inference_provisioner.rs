use corvia_common::errors::{CorviaError, Result};
use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::{HealthRequest, ListModelsRequest, LoadModelRequest, ReloadModelsRequest};
use tracing::info;

/// Resolved HF coordinates for a chat model.
#[derive(Debug)]
pub struct ChatModelCoords {
    pub name: String,
    pub hf_repo: String,
    pub hf_filename: String,
}

/// Provisions the corvia-inference gRPC server.
/// Consistent with OllamaProvisioner: install → start → wait → load models.
pub struct InferenceProvisioner {
    grpc_addr: String,
}

impl InferenceProvisioner {
    pub fn new(grpc_addr: &str) -> Self {
        Self {
            grpc_addr: grpc_addr.to_string(),
        }
    }

    /// Check if the `corvia-inference` binary is installed.
    pub fn is_installed() -> bool {
        std::process::Command::new("corvia-inference")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn endpoint_url(&self) -> String {
        if self.grpc_addr.starts_with("http") {
            self.grpc_addr.clone()
        } else {
            format!("http://{}", self.grpc_addr)
        }
    }

    /// Check if the inference server is reachable via gRPC health.
    pub async fn is_running(&self) -> bool {
        let Ok(mut client) = ModelManagerClient::connect(self.endpoint_url()).await else {
            return false;
        };
        client
            .health(tonic::Request::new(HealthRequest {}))
            .await
            .map(|r| r.into_inner().healthy)
            .unwrap_or(false)
    }

    /// Start the inference server as a background process.
    pub fn start(&self) -> Result<()> {
        info!("Starting corvia-inference server...");
        let port = self.grpc_addr.split(':').last().unwrap_or("8030");
        std::process::Command::new("corvia-inference")
            .arg("serve")
            .arg("--port")
            .arg(port)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| CorviaError::Infra(format!("Failed to start corvia-inference: {e}")))?;
        info!("corvia-inference server started");
        Ok(())
    }

    /// Wait for the server to become healthy.
    pub async fn wait_ready(&self, timeout_secs: u64) -> Result<()> {
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        while tokio::time::Instant::now() < deadline {
            if self.is_running().await {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(CorviaError::Infra(format!(
            "corvia-inference did not become ready within {timeout_secs}s at {}",
            self.grpc_addr
        )))
    }

    /// Load models on the server.
    pub async fn load_models(
        &self,
        embed_model: &str,
        chat_model: Option<&ChatModelCoords>,
        device: &str,
        backend: &str,
        embedding_backend: &str,
        kv_quant: &str,
        flash_attention: bool,
    ) -> Result<()> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        // Use embedding_backend when set, otherwise fall back to the global backend.
        let effective_embed_backend = if embedding_backend.is_empty() { backend } else { embedding_backend };

        // Load embedding model
        let resp = client
            .load_model(tonic::Request::new(LoadModelRequest {
                name: embed_model.to_string(),
                model_type: "embedding".to_string(),
                device: device.to_string(),
                backend: effective_embed_backend.to_string(),
                kv_quant: kv_quant.to_string(),
                flash_attention,
                hf_repo: String::new(),
                hf_filename: String::new(),
            }))
            .await
            .map_err(|e| CorviaError::Infra(format!("LoadModel failed: {e}")))?;
        let resp = resp.into_inner();
        if !resp.success {
            return Err(CorviaError::Infra(format!(
                "Failed to load embed model: {}",
                resp.error
            )));
        }
        info!(
            model = embed_model,
            device = resp.actual_device,
            backend = resp.actual_backend,
            "Loaded embedding model"
        );

        // Load chat model (optional — skipped when [merge] is not configured)
        if let Some(coords) = chat_model {
            let resp = client
                .load_model(tonic::Request::new(LoadModelRequest {
                    name: coords.name.clone(),
                    model_type: "chat".to_string(),
                    device: device.to_string(),
                    backend: backend.to_string(),
                    kv_quant: kv_quant.to_string(),
                    flash_attention,
                    hf_repo: coords.hf_repo.clone(),
                    hf_filename: coords.hf_filename.clone(),
                }))
                .await
                .map_err(|e| CorviaError::Infra(format!("LoadModel failed: {e}")))?;
            let resp = resp.into_inner();
            if !resp.success {
                return Err(CorviaError::Infra(format!(
                    "Failed to load chat model: {}",
                    resp.error
                )));
            }
            info!(
                model = %coords.name,
                device = resp.actual_device,
                backend = resp.actual_backend,
                "Loaded chat model"
            );
        }

        Ok(())
    }

    /// Reload loaded models with a new device/backend.
    /// If `model_name` is Some, only that model is reloaded; otherwise all.
    pub async fn reload_models(&self, device: &str, backend: &str, kv_quant: &str, flash_attention: bool, model_name: Option<&str>) -> Result<()> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        let resp = client
            .reload_models(tonic::Request::new(ReloadModelsRequest {
                device: device.to_string(),
                backend: backend.to_string(),
                reprobe_gpu: true,
                name: model_name.unwrap_or_default().to_string(),
                kv_quant: kv_quant.to_string(),
                flash_attention,
            }))
            .await
            .map_err(|e| CorviaError::Infra(format!("ReloadModels failed: {e}")))?;
        let resp = resp.into_inner();
        if !resp.success {
            return Err(CorviaError::Infra(format!(
                "Reload failed: {}",
                resp.error
            )));
        }
        for r in &resp.results {
            if r.success {
                info!(model = %r.name, device = %r.actual_device, backend = %r.actual_backend, "Reloaded");
            } else {
                tracing::error!(model = %r.name, error = %r.error, "Reload failed");
            }
        }
        Ok(())
    }

    /// List currently loaded models with their device/backend info.
    pub async fn list_models(&self) -> Result<Vec<corvia_proto::ModelStatus>> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        let resp = client
            .list_models(tonic::Request::new(ListModelsRequest {}))
            .await
            .map_err(|e| CorviaError::Infra(format!("ListModels failed: {e}")))?;
        Ok(resp.into_inner().models)
    }

    /// Full provisioning: start if not running → wait → load models.
    pub async fn ensure_ready(
        &self,
        embed_model: &str,
        chat_model: Option<&ChatModelCoords>,
        device: &str,
        backend: &str,
        embedding_backend: &str,
        kv_quant: &str,
        flash_attention: bool,
    ) -> Result<()> {
        if !self.is_running().await {
            if !Self::is_installed() {
                return Err(CorviaError::Infra(
                    "corvia-inference not found. Install it or switch to provider = \"ollama\""
                        .into(),
                ));
            }
            self.start()?;
            self.wait_ready(15).await?;
        }
        self.load_models(embed_model, chat_model, device, backend, embedding_backend, kv_quant, flash_attention).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provisioner_construction() {
        let p = InferenceProvisioner::new("127.0.0.1:8030");
        assert_eq!(p.grpc_addr, "127.0.0.1:8030");
    }

    #[test]
    fn test_provisioner_endpoint_url_no_scheme() {
        let p = InferenceProvisioner::new("127.0.0.1:8030");
        assert_eq!(p.endpoint_url(), "http://127.0.0.1:8030");
    }

    #[test]
    fn test_provisioner_endpoint_url_with_scheme() {
        let p = InferenceProvisioner::new("http://localhost:8030");
        assert_eq!(p.endpoint_url(), "http://localhost:8030");
    }

    #[test]
    fn test_provisioner_is_installed_check() {
        // Just verifies the check doesn't panic (corvia-inference likely not installed in test env)
        let _ = InferenceProvisioner::is_installed();
    }
}
