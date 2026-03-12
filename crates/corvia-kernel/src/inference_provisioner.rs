use corvia_common::errors::{CorviaError, Result};
use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::{HealthRequest, ListModelsRequest, LoadModelRequest, ReloadModelsRequest};
use tracing::info;

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
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
    ) -> Result<()> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        // Load embedding model
        let resp = client
            .load_model(tonic::Request::new(LoadModelRequest {
                name: embed_model.to_string(),
                model_type: "embedding".to_string(),
                device: device.to_string(),
                backend: backend.to_string(),
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
        if let Some(chat_model) = chat_model {
            let resp = client
                .load_model(tonic::Request::new(LoadModelRequest {
                    name: chat_model.to_string(),
                    model_type: "chat".to_string(),
                    device: device.to_string(),
                    backend: backend.to_string(),
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
                model = chat_model,
                device = resp.actual_device,
                backend = resp.actual_backend,
                "Loaded chat model"
            );
        }

        Ok(())
    }

    /// Reload all currently loaded models with a new device/backend.
    pub async fn reload_models(&self, device: &str, backend: &str) -> Result<()> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        let resp = client
            .reload_models(tonic::Request::new(ReloadModelsRequest {
                device: device.to_string(),
                backend: backend.to_string(),
                reprobe_gpu: true,
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
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
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
        self.load_models(embed_model, chat_model, device, backend).await?;
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
