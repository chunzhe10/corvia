//! E2E tests for chat inference via llama.cpp.
//!
//! These are **Tier 3** tests — they require downloading a ~800 MB GGUF model
//! from HuggingFace. They are gated behind `CORVIA_CHAT_TEST=1` and auto-skip
//! when the env var is unset.
//!
//! ```bash
//! CORVIA_CHAT_TEST=1 cargo test --package corvia-inference -- chat_e2e
//! ```

use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::*;
use tokio_stream::StreamExt;

/// Return true if the test should be skipped.
fn should_skip() -> bool {
    if std::env::var("CORVIA_CHAT_TEST").is_err() {
        eprintln!("Skipping: set CORVIA_CHAT_TEST=1 to enable (downloads ~800 MB model)");
        true
    } else {
        false
    }
}

/// Start a corvia-inference gRPC server on an ephemeral port and return the address.
async fn start_server() -> String {
    // Find a free port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let port = addr.port();
    drop(listener); // Release the port so the binary can bind to it
    let binary = env!("CARGO_BIN_EXE_corvia-inference");

    tokio::spawn(async move {
        let mut child = tokio::process::Command::new(binary)
            .args(["serve", "--port", &port.to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to start corvia-inference binary");

        child.wait().await.ok();
    });

    // Wait for the server to be ready
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        if ModelManagerClient::connect(endpoint.clone()).await.is_ok() {
            return endpoint;
        }
    }

    panic!("Server failed to start within 6 seconds");
}

/// Clean up: kill any corvia-inference process we spawned on the given port.
/// Best-effort — the OS will clean up on test process exit anyway.
fn cleanup_server(endpoint: &str) {
    // Extract port from endpoint
    if let Some(port) = endpoint.rsplit(':').next() {
        let _ = std::process::Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .output();
    }
}

const TEST_MODEL: &str = "llama3.2:1b";

#[tokio::test]
async fn test_chat_e2e_generates_response() {
    if should_skip() {
        return;
    }

    let endpoint = start_server().await;

    // Load the 1B model (smallest, ~800 MB download)
    let mut mgr = ModelManagerClient::connect(endpoint.clone())
        .await
        .expect("Failed to connect to model manager");

    let load_resp = mgr
        .load_model(LoadModelRequest {
            name: TEST_MODEL.to_string(),
            model_type: "chat".to_string(),
            device: "auto".to_string(),
            backend: String::new(),
            kv_quant: String::new(),
            flash_attention: false,
            hf_repo: String::new(),
            hf_filename: String::new(),
        })
        .await
        .expect("load_model RPC failed")
        .into_inner();

    assert!(
        load_resp.success,
        "Model load failed: {}",
        load_resp.error
    );

    // Send a chat request
    let mut chat = ChatServiceClient::connect(endpoint.clone())
        .await
        .expect("Failed to connect to chat service");

    let resp = chat
        .chat(ChatRequest {
            model: TEST_MODEL.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "What is 2 + 2? Reply with just the number.".to_string(),
            }],
            temperature: 0.0,
            max_tokens: 64,
        })
        .await
        .expect("chat RPC failed")
        .into_inner();

    let msg = resp.message.expect("Response should have a message");
    assert_eq!(msg.role, "assistant");
    assert!(!msg.content.is_empty(), "Response content should be non-empty");
    assert!(resp.prompt_tokens > 0, "Should report prompt tokens");
    assert!(resp.completion_tokens > 0, "Should report completion tokens");

    eprintln!("Chat response: {:?}", msg.content);

    cleanup_server(&endpoint);
}

#[tokio::test]
async fn test_chat_stream_e2e() {
    if should_skip() {
        return;
    }

    let endpoint = start_server().await;

    // Load model
    let mut mgr = ModelManagerClient::connect(endpoint.clone())
        .await
        .expect("Failed to connect");

    let load_resp = mgr
        .load_model(LoadModelRequest {
            name: TEST_MODEL.to_string(),
            model_type: "chat".to_string(),
            device: "auto".to_string(),
            backend: String::new(),
            kv_quant: String::new(),
            flash_attention: false,
            hf_repo: String::new(),
            hf_filename: String::new(),
        })
        .await
        .expect("load_model failed")
        .into_inner();

    assert!(load_resp.success, "Model load failed: {}", load_resp.error);

    // Stream a chat request
    let mut chat = ChatServiceClient::connect(endpoint.clone())
        .await
        .expect("Failed to connect to chat service");

    let mut stream = chat
        .chat_stream(ChatRequest {
            model: TEST_MODEL.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Say hello in one word.".to_string(),
            }],
            temperature: 0.0,
            max_tokens: 32,
        })
        .await
        .expect("chat_stream RPC failed")
        .into_inner();

    let mut chunks = Vec::new();
    let mut full_text = String::new();
    let mut saw_done = false;

    while let Some(result) = stream.next().await {
        let chunk = result.expect("Stream chunk error");
        if !chunk.delta.is_empty() {
            full_text.push_str(&chunk.delta);
        }
        if chunk.done {
            saw_done = true;
            assert!(
                chunk.completion_tokens > 0,
                "Final chunk should report completion tokens"
            );
        }
        chunks.push(chunk);
    }

    assert!(saw_done, "Stream should end with a done=true chunk");
    assert!(!full_text.is_empty(), "Concatenated stream text should be non-empty");
    assert!(chunks.len() >= 2, "Should have at least 2 chunks (content + done)");

    eprintln!(
        "Streaming response ({} chunks): {:?}",
        chunks.len(),
        full_text
    );

    cleanup_server(&endpoint);
}
