//! Integration tests for the corvia-inference gRPC server.
//!
//! These tests auto-start the server and gracefully skip if the binary
//! isn't available — following the same pattern as the Ollama e2e tests.
//!
//! Run with: cargo test --test grpc_inference_test -- --nocapture

use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::*;
use std::process::{Child, Command, Stdio};
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::OnceCell;

/// Fixed port for the shared test server. Unique enough to avoid collisions.
const TEST_PORT: u16 = 18030;

/// Shared server process — started once, reused by all tests.
/// We use OnceCell so only the first test to run actually starts the server.
static SERVER: LazyLock<OnceCell<Option<ServerGuard>>> = LazyLock::new(OnceCell::new);

/// RAII guard that kills the server process on drop.
struct ServerGuard {
    child: std::sync::Mutex<Child>,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            child.kill().ok();
            child.wait().ok();
        }
    }
}

/// Try to find the corvia-inference binary. Prefer the pre-built binary in
/// target/debug, fall back to PATH.
fn find_binary() -> Option<String> {
    // CARGO_MANIFEST_DIR points to the crate that owns this test (corvia-cli).
    // Walk up to the workspace root to find the target/ directory.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("Cargo.lock").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // Check target/debug first (built by `cargo build --workspace`)
    let debug_path = workspace_root.join("target/debug/corvia-inference");
    if debug_path.exists() {
        return Some(debug_path.to_string_lossy().to_string());
    }
    // Check target/release
    let release_path = workspace_root.join("target/release/corvia-inference");
    if release_path.exists() {
        return Some(release_path.to_string_lossy().to_string());
    }
    // Check PATH
    if Command::new("corvia-inference")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some("corvia-inference".to_string());
    }
    None
}

async fn wait_for_server(addr: &str, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    let url = format!("http://{addr}");
    while tokio::time::Instant::now() < deadline {
        if let Ok(mut client) = ModelManagerClient::connect(url.clone()).await {
            if let Ok(resp) = client.health(tonic::Request::new(HealthRequest {})).await {
                if resp.into_inner().healthy {
                    return true;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Ensure the shared test server is running. Returns the gRPC URL, or None
/// if the binary isn't available (test should skip).
async fn ensure_server() -> Option<String> {
    let addr = format!("127.0.0.1:{TEST_PORT}");
    let url = format!("http://{addr}");

    // Check if something is already listening (e.g. previous test run, dev server)
    if wait_for_server(&addr, Duration::from_millis(500)).await {
        return Some(url);
    }

    let guard = SERVER
        .get_or_init(|| async {
            let Some(binary) = find_binary() else {
                eprintln!(
                    "SKIPPING gRPC tests: corvia-inference binary not found. \
                     Run `cargo build -p corvia-inference` first."
                );
                return None;
            };

            let child = Command::new(&binary)
                .args(["serve", "--port", &TEST_PORT.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();

            match child {
                Ok(child) => Some(ServerGuard {
                    child: std::sync::Mutex::new(child),
                }),
                Err(e) => {
                    eprintln!("SKIPPING gRPC tests: failed to start {binary}: {e}");
                    None
                }
            }
        })
        .await;

    if guard.is_none() {
        return None;
    }

    // Wait for it to be healthy
    if !wait_for_server(&addr, Duration::from_secs(30)).await {
        eprintln!("SKIPPING gRPC tests: server did not become healthy within 30s");
        return None;
    }

    Some(url)
}

#[tokio::test]
async fn test_grpc_health_check() {
    let Some(url) = ensure_server().await else {
        return;
    };

    let mut mgr = ModelManagerClient::connect(url).await.unwrap();
    let resp = mgr
        .health(tonic::Request::new(HealthRequest {}))
        .await
        .unwrap();
    let health = resp.into_inner();
    assert!(health.healthy);
}

#[tokio::test]
async fn test_grpc_model_lifecycle() {
    let Some(url) = ensure_server().await else {
        return;
    };

    let mut mgr = ModelManagerClient::connect(url).await.unwrap();

    // Load a model via ModelManager
    let resp = mgr
        .load_model(tonic::Request::new(LoadModelRequest {
            name: "test-embed-model".into(),
            model_type: "embedding".into(),
        }))
        .await
        .unwrap();
    assert!(resp.into_inner().success);

    // Verify model is listed
    let resp = mgr
        .list_models(tonic::Request::new(ListModelsRequest {}))
        .await
        .unwrap();
    let models = resp.into_inner().models;
    assert!(models.iter().any(|m| m.name == "test-embed-model"));

    // Unload model
    let resp = mgr
        .unload_model(tonic::Request::new(UnloadModelRequest {
            name: "test-embed-model".into(),
        }))
        .await
        .unwrap();
    assert!(resp.into_inner().success);

    // Verify it's gone
    let resp = mgr
        .list_models(tonic::Request::new(ListModelsRequest {}))
        .await
        .unwrap();
    let models = resp.into_inner().models;
    assert!(!models.iter().any(|m| m.name == "test-embed-model"));
}

#[tokio::test]
async fn test_grpc_chat_stub() {
    let Some(url) = ensure_server().await else {
        return;
    };

    // Register a chat model via ModelManager (the ChatService checks its own registry)
    // First register in the chat service's model map by calling the stub load
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();
    mgr.load_model(tonic::Request::new(LoadModelRequest {
        name: "test-chat".into(),
        model_type: "chat".into(),
    }))
    .await
    .unwrap();

    // Call chat — the stub currently checks its own internal model map,
    // not the ModelManager's. The chat stub should accept any model or
    // return not_found. Let's test the not_found path first.
    let mut chat = ChatServiceClient::connect(url.clone()).await.unwrap();
    let result = chat
        .chat(tonic::Request::new(ChatRequest {
            model: "nonexistent-model".into(),
            messages: vec![corvia_proto::ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            }],
            temperature: 0.7,
            max_tokens: 100,
        }))
        .await;

    // Should fail with NOT_FOUND for unknown model
    assert!(
        result.is_err(),
        "Expected NOT_FOUND error for unregistered chat model"
    );
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn test_grpc_embedding_model_info() {
    let Some(url) = ensure_server().await else {
        return;
    };

    let mut embed =
        corvia_proto::embedding_service_client::EmbeddingServiceClient::connect(url)
            .await
            .unwrap();

    // Query model_info for a model that isn't loaded
    let resp = embed
        .model_info(tonic::Request::new(ModelInfoRequest {
            model: "nomic-embed-text-v1.5".into(),
        }))
        .await
        .unwrap();
    let info = resp.into_inner();
    assert_eq!(info.model, "nomic-embed-text-v1.5");
    assert!(!info.loaded);
}
