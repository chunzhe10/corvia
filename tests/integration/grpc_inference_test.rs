//! Integration tests for the corvia-inference gRPC server.
//!
//! These tests auto-build and auto-start the server. If any setup step
//! fails (build, spawn, health check), the test panics with a clear
//! diagnostic rather than silently skipping.
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
static SERVER: LazyLock<OnceCell<ServerGuard>> = LazyLock::new(OnceCell::new);

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

/// Find the corvia-inference binary, building it if necessary.
///
/// Resolution order:
/// 1. target/debug/corvia-inference (already built)
/// 2. target/release/corvia-inference
/// 3. Build via `cargo build -p corvia-inference`, then return debug path
///
/// Panics if the binary cannot be found or built.
fn find_or_build_binary() -> String {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("Cargo.lock").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let debug_path = workspace_root.join("target/debug/corvia-inference");
    if debug_path.exists() {
        return debug_path.to_string_lossy().to_string();
    }

    let release_path = workspace_root.join("target/release/corvia-inference");
    if release_path.exists() {
        return release_path.to_string_lossy().to_string();
    }

    // Binary not found — build it
    eprintln!("corvia-inference binary not found, building...");
    let status = Command::new("cargo")
        .args(["build", "-p", "corvia-inference"])
        .current_dir(&workspace_root)
        .status()
        .expect("failed to run `cargo build -p corvia-inference`");

    assert!(
        status.success(),
        "cargo build -p corvia-inference failed with exit code: {:?}",
        status.code()
    );

    assert!(
        debug_path.exists(),
        "cargo build succeeded but binary not found at {}",
        debug_path.display()
    );

    debug_path.to_string_lossy().to_string()
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

/// Ensure the shared test server is running. Builds the binary if needed,
/// starts the server, and waits for it to become healthy.
///
/// Panics if any setup step fails.
async fn ensure_server() -> String {
    let addr = format!("127.0.0.1:{TEST_PORT}");
    let url = format!("http://{addr}");

    // Check if something is already listening (e.g. previous test run, dev server)
    if wait_for_server(&addr, Duration::from_millis(500)).await {
        return url;
    }

    SERVER
        .get_or_init(|| async {
            let binary = find_or_build_binary();

            let child = Command::new(&binary)
                .args(["serve", "--port", &TEST_PORT.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap_or_else(|e| panic!("failed to start {binary}: {e}"));

            ServerGuard {
                child: std::sync::Mutex::new(child),
            }
        })
        .await;

    // Wait for it to be healthy
    assert!(
        wait_for_server(&addr, Duration::from_secs(30)).await,
        "corvia-inference server did not become healthy within 30s on port {TEST_PORT}"
    );

    url
}

#[tokio::test]
async fn test_grpc_health_check() {
    let url = ensure_server().await;

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
    let url = ensure_server().await;

    // Use a real fastembed model name (smallest available).
    // Loading downloads from HuggingFace on first run (~33 MB).
    let model_name = "bge-small-en-v1.5";

    let mut mgr = ModelManagerClient::connect(url).await.unwrap();

    // Load a model via ModelManager
    let resp = mgr
        .load_model(tonic::Request::new(LoadModelRequest {
            name: model_name.into(),
            model_type: "embedding".into(),
            device: "auto".into(),
            backend: String::new(),
            kv_quant: String::new(),
            flash_attention: false,
        }))
        .await
        .unwrap();
    let load_resp = resp.into_inner();
    assert!(
        load_resp.success,
        "load_model failed: {}",
        load_resp.error
    );

    // Verify model is listed
    let resp = mgr
        .list_models(tonic::Request::new(ListModelsRequest {}))
        .await
        .unwrap();
    let models = resp.into_inner().models;
    assert!(models.iter().any(|m| m.name == model_name));

    // Unload model
    let resp = mgr
        .unload_model(tonic::Request::new(UnloadModelRequest {
            name: model_name.into(),
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
    assert!(!models.iter().any(|m| m.name == model_name));
}

#[tokio::test]
async fn test_grpc_chat_stub() {
    let url = ensure_server().await;

    // Register a chat model via ModelManager (the ChatService checks its own registry)
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();
    mgr.load_model(tonic::Request::new(LoadModelRequest {
        name: "test-chat".into(),
        model_type: "chat".into(),
        device: "auto".into(),
        backend: String::new(),
        kv_quant: String::new(),
        flash_attention: false,
    }))
    .await
    .unwrap();

    // Call chat with a nonexistent model — should return NOT_FOUND
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

    assert!(
        result.is_err(),
        "Expected NOT_FOUND error for unregistered chat model"
    );
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn test_grpc_embedding_model_info() {
    let url = ensure_server().await;

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
