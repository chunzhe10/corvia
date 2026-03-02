//! Integration test: start corvia-inference, connect as gRPC client, test embedding round-trip.
//!
//! This test is #[ignore] by default — it requires the corvia-inference binary to be built
//! and downloads an ONNX model (~50MB) on first run.
//!
//! Run with: cargo test --test grpc_inference_test -- --ignored

use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::*;
use std::time::Duration;

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

#[tokio::test]
#[ignore]
async fn test_grpc_health_check() {
    let port = 18030u16;
    let addr = format!("127.0.0.1:{port}");

    let mut child = std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "corvia-inference",
            "--",
            "serve",
            "--port",
            &port.to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start corvia-inference");

    let server_ready = wait_for_server(&addr, Duration::from_secs(60)).await;
    if !server_ready {
        child.kill().ok();
        panic!("corvia-inference server did not start within 60s");
    }

    // Health check
    let url = format!("http://{addr}");
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();
    let resp = mgr
        .health(tonic::Request::new(HealthRequest {}))
        .await
        .unwrap();
    let health = resp.into_inner();
    assert!(health.healthy);
    assert_eq!(health.models_loaded, 0);

    child.kill().ok();
}

#[tokio::test]
#[ignore]
async fn test_grpc_embed_roundtrip() {
    let port = 18031u16;
    let addr = format!("127.0.0.1:{port}");

    let mut child = std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "corvia-inference",
            "--",
            "serve",
            "--port",
            &port.to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start corvia-inference");

    let server_ready = wait_for_server(&addr, Duration::from_secs(60)).await;
    if !server_ready {
        child.kill().ok();
        panic!("corvia-inference server did not start within 60s");
    }

    let url = format!("http://{addr}");

    // Load embedding model (bge-small for fast test — 384 dims, ~50MB download)
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();

    // NOTE: The model_manager stub currently just registers the name.
    // The actual embedding service needs its own load_model call.
    // For now, test with the ModelManager stub load.
    let resp = mgr
        .load_model(tonic::Request::new(LoadModelRequest {
            name: "bge-small-en-v1.5".into(),
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
    assert!(!models.is_empty());
    assert!(models.iter().any(|m| m.name == "bge-small-en-v1.5"));

    // Health should show 1 model loaded
    let resp = mgr
        .health(tonic::Request::new(HealthRequest {}))
        .await
        .unwrap();
    assert_eq!(resp.into_inner().models_loaded, 1);

    // Unload model
    let resp = mgr
        .unload_model(tonic::Request::new(UnloadModelRequest {
            name: "bge-small-en-v1.5".into(),
        }))
        .await
        .unwrap();
    assert!(resp.into_inner().success);

    child.kill().ok();
}

#[tokio::test]
#[ignore]
async fn test_grpc_chat_stub() {
    let port = 18032u16;
    let addr = format!("127.0.0.1:{port}");

    let mut child = std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "corvia-inference",
            "--",
            "serve",
            "--port",
            &port.to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start corvia-inference");

    let server_ready = wait_for_server(&addr, Duration::from_secs(60)).await;
    if !server_ready {
        child.kill().ok();
        panic!("corvia-inference server did not start within 60s");
    }

    let url = format!("http://{addr}");

    // Register a chat model via ModelManager
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();
    mgr.load_model(tonic::Request::new(LoadModelRequest {
        name: "test-chat".into(),
        model_type: "chat".into(),
    }))
    .await
    .unwrap();

    // Call chat service (stub response expected)
    let mut chat =
        corvia_proto::chat_service_client::ChatServiceClient::connect(url.clone())
            .await
            .unwrap();
    let resp = chat
        .chat(tonic::Request::new(ChatRequest {
            model: "test-chat".into(),
            messages: vec![corvia_proto::ChatMessage {
                role: "user".into(),
                content: "Hello, world!".into(),
            }],
            temperature: 0.7,
            max_tokens: 100,
        }))
        .await
        .unwrap();

    let msg = resp.into_inner().message.unwrap();
    assert_eq!(msg.role, "assistant");
    assert!(msg.content.contains("[stub]"));

    child.kill().ok();
}
