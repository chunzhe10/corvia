use corvia_common::errors::{CorviaError, Result};
use tracing::info;

/// Provisions Ollama natively (no Docker). Mirrors DockerProvisioner's
/// install → start → pull model workflow, but for the zero-Docker LiteStore path.
pub struct OllamaProvisioner {
    url: String,
}

impl OllamaProvisioner {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
        }
    }

    /// Check if the `ollama` binary is installed.
    pub fn is_installed() -> bool {
        std::process::Command::new("ollama")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Check if Ollama is reachable (serving requests).
    pub async fn is_running(&self) -> bool {
        crate::ollama_engine::OllamaEngine::check_health(&self.url).await
    }

    /// Install Ollama using the official install script.
    pub fn install() -> Result<()> {
        info!("Installing Ollama...");

        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg("curl -fsSL https://ollama.com/install.sh | sh")
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
            .map_err(|e| CorviaError::Infra(format!("Failed to run Ollama installer: {e}")))?;

        if !status.success() {
            return Err(CorviaError::Infra(
                "Ollama installation failed. Install manually: https://ollama.com".into(),
            ));
        }

        info!("Ollama installed successfully");
        Ok(())
    }

    /// Start the Ollama server in the background.
    pub fn start() -> Result<()> {
        info!("Starting Ollama server...");

        // Spawn `ollama serve` as a background process
        std::process::Command::new("ollama")
            .arg("serve")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| CorviaError::Infra(format!("Failed to start Ollama: {e}")))?;

        info!("Ollama server started");
        Ok(())
    }

    /// Wait for Ollama to become healthy (up to timeout_secs).
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
            "Ollama did not become ready within {timeout_secs}s at {}",
            self.url
        )))
    }

    /// Pull a model if not already available.
    pub async fn pull_model(&self, model: &str) -> Result<()> {
        // Check if model is already available via the tags API
        let tags_url = format!("{}/api/tags", self.url);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        if let Ok(resp) = client.get(&tags_url).send().await {
            if let Ok(body) = resp.text().await {
                if body.contains(model) {
                    info!("Model {model} already available");
                    return Ok(());
                }
            }
        }

        info!("Pulling model {model}...");
        let status = std::process::Command::new("ollama")
            .arg("pull")
            .arg(model)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
            .map_err(|e| CorviaError::Infra(format!("Failed to pull model {model}: {e}")))?;

        if !status.success() {
            return Err(CorviaError::Infra(format!(
                "Failed to pull model {model}. Run manually: ollama pull {model}"
            )));
        }

        info!("Model {model} pulled successfully");
        Ok(())
    }

    /// Full provisioning: install → start → wait → pull model.
    /// Skips steps that are already satisfied.
    pub async fn ensure_ready(&self, model: &str) -> Result<()> {
        // Step 1: Install if needed
        if !Self::is_installed() {
            Self::install()?;
        }

        // Step 2: Start if not running
        if !self.is_running().await {
            Self::start()?;
            self.wait_ready(10).await?;
        }

        // Step 3: Pull model if needed
        self.pull_model(model).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provisioner_is_installed() {
        // Just verifies the check doesn't panic — result depends on environment
        let _ = OllamaProvisioner::is_installed();
    }
}
