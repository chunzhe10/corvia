use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, StartContainerOptions};
use bollard::image::CreateImageOptions;
use bollard::models::{HostConfig, PortBinding};
use bollard::Docker;
use corvia_common::errors::{CorviaError, Result};
use futures_util::TryStreamExt;
use serde_json;
use std::collections::HashMap;
use tracing::info;

const VLLM_CONTAINER_NAME: &str = "corvia-vllm";
const VLLM_IMAGE: &str = "vllm/vllm-openai:latest";
const VLLM_PORT: u16 = 8001;

const OLLAMA_CONTAINER_NAME: &str = "corvia-ollama";
const OLLAMA_IMAGE: &str = "ollama/ollama:latest";
const OLLAMA_PORT: u16 = 11434;

pub struct DockerProvisioner {
    docker: Docker,
}

impl DockerProvisioner {
    pub fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| CorviaError::Docker(format!("Failed to connect to Docker: {e}")))?;
        Ok(Self { docker })
    }

    /// Check if vLLM container is already running.
    pub async fn is_vllm_running(&self) -> Result<bool> {
        let filters: HashMap<String, Vec<String>> = HashMap::from([
            ("name".into(), vec![VLLM_CONTAINER_NAME.into()]),
        ]);
        let options = ListContainersOptions {
            filters,
            ..Default::default()
        };
        let containers = self.docker.list_containers(Some(options)).await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to list containers: {e}")))?;
        Ok(!containers.is_empty())
    }

    /// Start vLLM container with the given model. Pulls image if needed.
    pub async fn start_vllm(&self, model: &str) -> Result<()> {
        if self.is_vllm_running().await? {
            info!("vLLM container already running");
            return Ok(());
        }

        // Pull image (idempotent)
        info!("Pulling vLLM image: {VLLM_IMAGE}");
        let options = CreateImageOptions {
            from_image: VLLM_IMAGE,
            ..Default::default()
        };
        self.docker.create_image(Some(options), None, None)
            .try_collect::<Vec<_>>().await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to pull image: {e}")))?;
        info!("vLLM image pulled successfully");

        let port_str = format!("{VLLM_PORT}/tcp");
        let host_config = HostConfig {
            port_bindings: Some(HashMap::from([(
                port_str.clone(),
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".into()),
                    host_port: Some(VLLM_PORT.to_string()),
                }]),
            )])),
            ..Default::default()
        };

        let config: Config<String> = Config {
            image: Some(VLLM_IMAGE.to_string()),
            cmd: Some(vec![
                "--model".into(),
                model.into(),
                "--port".into(),
                VLLM_PORT.to_string(),
            ]),
            exposed_ports: Some(HashMap::from([(port_str, HashMap::new())])),
            host_config: Some(host_config),
            ..Default::default()
        };

        let create_options = CreateContainerOptions {
            name: VLLM_CONTAINER_NAME.to_string(),
            ..Default::default()
        };

        // Remove existing stopped container if present
        let _ = self.docker.remove_container(VLLM_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await;

        info!("Creating vLLM container: {VLLM_CONTAINER_NAME}");
        self.docker.create_container(Some(create_options), config).await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to create container: {e}")))?;

        self.docker.start_container(VLLM_CONTAINER_NAME, None::<StartContainerOptions<String>>).await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to start container: {e}")))?;

        info!("vLLM started on port {VLLM_PORT}");

        // Wait for vLLM model to load
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;

        Ok(())
    }

    /// Stop and remove the vLLM container.
    pub async fn stop_vllm(&self) -> Result<()> {
        info!("Stopping vLLM container");
        self.docker.stop_container(VLLM_CONTAINER_NAME, None::<bollard::container::StopContainerOptions>).await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to stop container: {e}")))?;
        self.docker.remove_container(VLLM_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await
            .map_err(|e| CorviaError::Docker(format!("vLLM: Failed to remove container: {e}")))?;
        Ok(())
    }

    /// Check if Ollama container is already running.
    pub async fn is_ollama_running(&self) -> Result<bool> {
        let filters: HashMap<String, Vec<String>> = HashMap::from([
            ("name".into(), vec![OLLAMA_CONTAINER_NAME.into()]),
        ]);
        let options = ListContainersOptions {
            filters,
            ..Default::default()
        };
        let containers = self.docker.list_containers(Some(options)).await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to list containers: {e}")))?;
        Ok(!containers.is_empty())
    }

    /// Start Ollama container and pull the specified model.
    pub async fn start_ollama(&self, model: &str) -> Result<()> {
        if self.is_ollama_running().await? {
            info!("Ollama container already running");
            return Ok(());
        }

        // Pull image
        info!("Pulling Ollama image: {OLLAMA_IMAGE}");
        let options = CreateImageOptions {
            from_image: OLLAMA_IMAGE,
            ..Default::default()
        };
        self.docker.create_image(Some(options), None, None)
            .try_collect::<Vec<_>>().await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to pull image: {e}")))?;
        info!("Ollama image pulled successfully");

        let port_str = format!("{OLLAMA_PORT}/tcp");
        let host_config = HostConfig {
            port_bindings: Some(HashMap::from([(
                port_str.clone(),
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".into()),
                    host_port: Some(OLLAMA_PORT.to_string()),
                }]),
            )])),
            ..Default::default()
        };

        let config: Config<String> = Config {
            image: Some(OLLAMA_IMAGE.to_string()),
            exposed_ports: Some(HashMap::from([(port_str, HashMap::new())])),
            host_config: Some(host_config),
            ..Default::default()
        };

        let create_options = CreateContainerOptions {
            name: OLLAMA_CONTAINER_NAME.to_string(),
            ..Default::default()
        };

        // Remove existing stopped container if present
        let _ = self.docker.remove_container(OLLAMA_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await;

        info!("Creating Ollama container: {OLLAMA_CONTAINER_NAME}");
        self.docker.create_container(Some(create_options), config).await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to create container: {e}")))?;

        self.docker.start_container(OLLAMA_CONTAINER_NAME, None::<StartContainerOptions<String>>).await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to start container: {e}")))?;

        info!("Ollama started on port {OLLAMA_PORT}");

        // Wait for Ollama to be ready
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        // Pull the embedding model via Ollama's API
        info!("Pulling model {model} via Ollama API...");
        let pull_url = format!("http://127.0.0.1:{OLLAMA_PORT}/api/pull");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| CorviaError::Docker(format!("Failed to create HTTP client: {e}")))?;
        let resp = client.post(&pull_url)
            .json(&serde_json::json!({"name": model}))
            .send()
            .await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to pull model {model}: {e}")))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(CorviaError::Docker(format!("Ollama: Model pull failed: {body}")));
        }

        // The pull response is streamed — consume the full body to wait for completion
        let _ = resp.bytes().await;
        info!("Model {model} pulled successfully");

        Ok(())
    }

    /// Stop and remove the Ollama container.
    pub async fn stop_ollama(&self) -> Result<()> {
        info!("Stopping Ollama container");
        self.docker.stop_container(OLLAMA_CONTAINER_NAME, None::<bollard::container::StopContainerOptions>).await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to stop container: {e}")))?;
        self.docker.remove_container(OLLAMA_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await
            .map_err(|e| CorviaError::Docker(format!("Ollama: Failed to remove container: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_container_config() {
        assert_eq!(VLLM_CONTAINER_NAME, "corvia-vllm");
        assert_eq!(VLLM_PORT, 8001);
        assert!(!VLLM_IMAGE.is_empty());
    }
}
