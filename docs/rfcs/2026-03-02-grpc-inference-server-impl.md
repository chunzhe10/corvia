# gRPC Inference Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a custom gRPC inference server (`corvia-inference`) that serves embeddings (ONNX Runtime) and chat (candle), replacing Ollama as the LiteStore default.

**Architecture:** Two new crates: `corvia-proto` (shared protobuf types) and `corvia-inference` (gRPC server binary). Three new engine implementations in `corvia-kernel`: `GrpcInferenceEngine`, `GrpcChatEngine`, `GrpcVllmEngine`. New `ChatEngine` trait formalizes merge chat. `InferenceProvisioner` auto-manages the server lifecycle.

**Tech Stack:** tonic (gRPC), prost (protobuf codegen), ort/fastembed (ONNX embeddings), candle-core/candle-transformers (chat/generation), hf-hub (model download)

**Design doc:** `docs/rfcs/2026-03-02-grpc-inference-server-design.md` (D60)

---

## Task 1: Create corvia-proto crate with proto files

**Files:**
- Create: `crates/corvia-proto/Cargo.toml`
- Create: `crates/corvia-proto/build.rs`
- Create: `crates/corvia-proto/src/lib.rs`
- Create: `crates/corvia-proto/proto/corvia/inference/v1/embedding.proto`
- Create: `crates/corvia-proto/proto/corvia/inference/v1/chat.proto`
- Create: `crates/corvia-proto/proto/corvia/inference/v1/model.proto`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Create proto files**

Create `crates/corvia-proto/proto/corvia/inference/v1/embedding.proto`:
```protobuf
syntax = "proto3";
package corvia.inference.v1;

service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
  rpc ModelInfo(ModelInfoRequest) returns (ModelInfoResponse);
}

message EmbedRequest {
  string model = 1;
  string text = 2;
}

message EmbedResponse {
  repeated float embedding = 1 [packed = true];
  uint32 dimensions = 2;
}

message EmbedBatchRequest {
  string model = 1;
  repeated string texts = 2;
}

message EmbedBatchResponse {
  repeated Embedding embeddings = 1;
}

message Embedding {
  repeated float values = 1 [packed = true];
  uint32 index = 2;
}

message ModelInfoRequest {
  string model = 1;
}

message ModelInfoResponse {
  string model = 1;
  uint32 dimensions = 2;
  bool loaded = 3;
}
```

Create `crates/corvia-proto/proto/corvia/inference/v1/chat.proto`:
```protobuf
syntax = "proto3";
package corvia.inference.v1;

service ChatService {
  rpc Chat(ChatRequest) returns (ChatResponse);
  rpc ChatStream(ChatRequest) returns (stream ChatChunk);
}

message ChatMessage {
  string role = 1;
  string content = 2;
}

message ChatRequest {
  string model = 1;
  repeated ChatMessage messages = 2;
  float temperature = 3;
  uint32 max_tokens = 4;
}

message ChatResponse {
  ChatMessage message = 1;
  uint32 prompt_tokens = 2;
  uint32 completion_tokens = 3;
}

message ChatChunk {
  string delta = 1;
  bool done = 2;
  uint32 prompt_tokens = 3;
  uint32 completion_tokens = 4;
}
```

Create `crates/corvia-proto/proto/corvia/inference/v1/model.proto`:
```protobuf
syntax = "proto3";
package corvia.inference.v1;

service ModelManager {
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
}

message ListModelsRequest {}
message ListModelsResponse {
  repeated ModelStatus models = 1;
}

message ModelStatus {
  string name = 1;
  string model_type = 2;
  bool loaded = 3;
  uint64 memory_bytes = 4;
}

message LoadModelRequest {
  string name = 1;
  string model_type = 2;
}
message LoadModelResponse {
  bool success = 1;
  string error = 2;
}

message UnloadModelRequest { string name = 1; }
message UnloadModelResponse { bool success = 1; }

message HealthRequest {}
message HealthResponse {
  bool healthy = 1;
  uint32 models_loaded = 2;
}
```

**Step 2: Create Cargo.toml and build.rs**

Create `crates/corvia-proto/Cargo.toml`:
```toml
[package]
name = "corvia-proto"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
tonic = "0.12"
prost = "0.13"

[build-dependencies]
tonic-build = "0.12"
```

Create `crates/corvia-proto/build.rs`:
```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/corvia/inference/v1/embedding.proto",
                "proto/corvia/inference/v1/chat.proto",
                "proto/corvia/inference/v1/model.proto",
            ],
            &["proto/"],
        )?;
    Ok(())
}
```

**Step 3: Create lib.rs re-exports**

Create `crates/corvia-proto/src/lib.rs`:
```rust
pub mod corvia {
    pub mod inference {
        pub mod v1 {
            tonic::include_proto!("corvia.inference.v1");
        }
    }
}

// Convenience re-export
pub use corvia::inference::v1::*;
```

**Step 4: Add to workspace Cargo.toml**

Add `"crates/corvia-proto"` to `[workspace].members` array in root `Cargo.toml`.
Add `corvia-proto = { path = "crates/corvia-proto" }` to `[workspace.dependencies]`.

**Step 5: Verify it compiles**

Run: `cargo build -p corvia-proto`
Expected: SUCCESS — tonic-build generates server/client stubs from proto files.

**Step 6: Commit**

```bash
git add crates/corvia-proto/ Cargo.toml
git commit -m "feat(d60): add corvia-proto crate with gRPC service definitions"
```

---

## Task 2: Add ChatEngine trait and ChatMessage type

**Files:**
- Modify: `crates/corvia-kernel/src/traits.rs`
- Modify: `crates/corvia-common/src/types.rs`
- Test: existing tests in `crates/corvia-kernel/src/merge_worker.rs`

**Step 1: Write ChatMessage type in corvia-common**

Add to `crates/corvia-common/src/types.rs`:
```rust
/// A chat message for LLM inference (used by ChatEngine trait).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}
```

**Step 2: Add ChatEngine trait to traits.rs**

Add to `crates/corvia-kernel/src/traits.rs`:
```rust
use corvia_common::types::ChatMessage;

/// Chat/reasoning provider for LLM inference (D60).
/// Used by MergeWorker for conflict resolution.
#[async_trait]
pub trait ChatEngine: Send + Sync {
    /// Send messages to a chat model and return the response text.
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String>;
}
```

**Step 3: Verify existing tests still pass**

Run: `cargo test -p corvia-kernel`
Expected: All 117 tests PASS (no behavioral changes yet).

**Step 4: Commit**

```bash
git add crates/corvia-common/src/types.rs crates/corvia-kernel/src/traits.rs
git commit -m "feat(d60): add ChatEngine trait and ChatMessage type"
```

---

## Task 3: Add InferenceProvider::Corvia to config

**Files:**
- Modify: `crates/corvia-common/src/config.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Write failing test for Corvia provider serde**

Add test to `crates/corvia-common/src/config.rs`:
```rust
#[test]
fn test_inference_provider_corvia_serde() {
    let json = serde_json::to_string(&InferenceProvider::Corvia).unwrap();
    assert_eq!(json, "\"corvia\"");
    let parsed: InferenceProvider = serde_json::from_str("\"corvia\"").unwrap();
    assert_eq!(parsed, InferenceProvider::Corvia);
}

#[test]
fn test_default_config_uses_corvia() {
    let config = CorviaConfig::default();
    assert_eq!(config.embedding.provider, InferenceProvider::Corvia);
    assert_eq!(config.embedding.url, "127.0.0.1:8030");
    assert_eq!(config.embedding.model, "nomic-embed-text-v1.5");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-common test_inference_provider_corvia_serde`
Expected: FAIL — `Corvia` variant doesn't exist yet.

**Step 3: Add Corvia variant to InferenceProvider enum**

In `crates/corvia-common/src/config.rs`, change:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceProvider {
    #[default]
    Corvia,
    Ollama,
    Vllm,
}
```

Update `CorviaConfig::default()` — change the `embedding` field:
```rust
embedding: EmbeddingConfig {
    provider: InferenceProvider::Corvia,
    model: "nomic-embed-text-v1.5".into(),
    url: "127.0.0.1:8030".into(),
    dimensions: 768,
},
```

Add `provider` field to `MergeConfig`:
```rust
pub struct MergeConfig {
    #[serde(default = "default_merge_model")]
    pub model: String,
    #[serde(default = "default_merge_provider")]
    pub provider: InferenceProvider,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_merge_provider() -> InferenceProvider { InferenceProvider::Corvia }
```

Update `MergeConfig::default()`:
```rust
impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            model: default_merge_model(),
            provider: default_merge_provider(),
            similarity_threshold: default_similarity_threshold(),
            max_retries: default_max_retries(),
        }
    }
}
```

**Step 4: Update create_engine in corvia-kernel/src/lib.rs**

Add arm to the match in `create_engine()`:
```rust
InferenceProvider::Corvia => {
    // TODO: Replace with GrpcInferenceEngine in Task 7
    // For now, fall back to Ollama so existing code doesn't break
    Box::new(ollama_engine::OllamaEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    ))
}
```

**Step 5: Update cmd_status display in corvia-cli/src/main.rs**

Add the `Corvia` arm to the match in `cmd_status()`:
```rust
match config.embedding.provider {
    corvia_common::config::InferenceProvider::Corvia => "Corvia Inference",
    corvia_common::config::InferenceProvider::Ollama => "Ollama",
    corvia_common::config::InferenceProvider::Vllm => "vLLM",
},
```

**Step 6: Fix existing test assertions**

Update test `test_default_config` in `crates/corvia-common/src/config.rs`:
```rust
assert_eq!(config.embedding.provider, InferenceProvider::Corvia);
```

Update test `test_lite_config_defaults`:
```rust
assert_eq!(config.embedding.provider, InferenceProvider::Corvia);
assert_eq!(config.embedding.url, "127.0.0.1:8030");
assert_eq!(config.embedding.model, "nomic-embed-text-v1.5");
```

Update test `test_create_engine_ollama` in `crates/corvia-kernel/src/lib.rs`:
```rust
#[test]
fn test_create_engine_corvia_fallback() {
    let config = CorviaConfig::default();
    assert_eq!(config.embedding.provider, InferenceProvider::Corvia);
    let engine = create_engine(&config);
    assert_eq!(engine.dimensions(), 768);
}
```

Fix any TOML test fixtures that hardcode `provider = "ollama"` for LiteStore configs.

**Step 7: Run all tests**

Run: `cargo test -p corvia-common && cargo test -p corvia-kernel`
Expected: All tests PASS.

**Step 8: Commit**

```bash
git add crates/corvia-common/src/config.rs crates/corvia-kernel/src/lib.rs crates/corvia-cli/src/main.rs
git commit -m "feat(d60): add InferenceProvider::Corvia, update defaults"
```

---

## Task 4: Refactor MergeWorker to use ChatEngine trait

**Files:**
- Modify: `crates/corvia-kernel/src/merge_worker.rs`
- Create: `crates/corvia-kernel/src/ollama_chat.rs` (extract Ollama chat logic)
- Modify: `crates/corvia-kernel/src/lib.rs` (add module)
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs` (pass ChatEngine)

**Step 1: Write failing test — MergeWorker accepts ChatEngine**

Add to `crates/corvia-kernel/src/merge_worker.rs` tests:
```rust
struct MockChatEngine;
#[async_trait::async_trait]
impl crate::traits::ChatEngine for MockChatEngine {
    async fn chat(&self, messages: &[corvia_common::types::ChatMessage], _model: &str) -> corvia_common::errors::Result<String> {
        // Return merged content from both messages
        Ok(format!("merged: {}", messages.last().map(|m| m.content.as_str()).unwrap_or("")))
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-kernel test_merge_worker_with_chat_engine`
Expected: FAIL — function doesn't exist yet.

**Step 3: Create OllamaChatEngine in new file**

Create `crates/corvia-kernel/src/ollama_chat.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::ChatMessage;

/// ChatEngine implementation that calls Ollama /api/chat.
/// Preserves the existing MergeWorker behavior as a ChatEngine impl.
pub struct OllamaChatEngine {
    url: String,
    client: reqwest::Client,
}

impl OllamaChatEngine {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl super::traits::ChatEngine for OllamaChatEngine {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String> {
        let api_messages: Vec<serde_json::Value> = messages.iter().map(|m| {
            serde_json::json!({ "role": m.role, "content": m.content })
        }).collect();

        let request_body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "stream": false
        });

        let response = self.client
            .post(format!("{}/api/chat", self.url))
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| CorviaError::Agent(format!("LLM chat request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Agent(format!("LLM chat failed with status {status}: {body}")));
        }

        let body: serde_json::Value = response.json().await
            .map_err(|e| CorviaError::Agent(format!("Failed to parse LLM response: {e}")))?;

        body["message"]["content"]
            .as_str()
            .ok_or_else(|| CorviaError::Agent("LLM response missing message.content".into()))
            .map(|s| s.to_string())
    }
}
```

**Step 4: Refactor MergeWorker to accept ChatEngine**

In `crates/corvia-kernel/src/merge_worker.rs`, replace `ollama_url: String` with `chat_engine: Arc<dyn ChatEngine>`:

```rust
pub struct MergeWorker {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
    chat_engine: Arc<dyn ChatEngine>,  // NEW — replaces ollama_url
    queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    session_mgr: Arc<SessionManager>,
    merge_config: MergeConfig,
}
```

Update constructor:
```rust
pub fn new(
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
    chat_engine: Arc<dyn ChatEngine>,  // NEW
    queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    session_mgr: Arc<SessionManager>,
    merge_config: MergeConfig,
) -> Self {
    Self { store, engine, chat_engine, queue, staging, session_mgr, merge_config }
}
```

Replace `llm_merge` method's HTTP call with:
```rust
async fn llm_merge(&self, new_entry: &KnowledgeEntry, existing: &KnowledgeEntry) -> Result<KnowledgeEntry> {
    let prompt = format!(
        "You are merging two knowledge entries that conflict. \
         Produce a single merged entry that preserves all important information from both.\n\n\
         Entry A (existing):\n{}\n\n\
         Entry B (new):\n{}\n\n\
         Merged entry:",
        existing.content, new_entry.content
    );

    let messages = vec![ChatMessage::user(prompt)];
    let merged_content = self.chat_engine.chat(&messages, &self.merge_config.model).await?;

    // Re-embed the merged content
    let embedding = self.engine.embed(&merged_content).await?;

    let merged = KnowledgeEntry::new(
        merged_content,
        new_entry.scope_id.clone(),
        new_entry.source_version.clone(),
    )
    .with_embedding(embedding)
    .with_agent(
        new_entry.agent_id.clone().unwrap_or_default(),
        new_entry.session_id.clone().unwrap_or_default(),
    );
    let mut merged = merged;
    merged.entry_status = EntryStatus::Merged;
    Ok(merged)
}
```

**Step 5: Update agent_coordinator.rs to pass ChatEngine**

In `crates/corvia-kernel/src/agent_coordinator.rs`, update `AgentCoordinator::new()` to accept and pass `Arc<dyn ChatEngine>` to `MergeWorker::new()`. Update all callers.

Check `crates/corvia-cli/src/main.rs` `cmd_serve()`: construct `OllamaChatEngine` and pass it.

**Step 6: Update existing MergeWorker tests**

In `crates/corvia-kernel/src/merge_worker.rs` tests, update `setup()`:
```rust
fn setup(dir: &std::path::Path) -> (MergeWorker, Arc<dyn QueryableStore>, Arc<MergeQueue>, Arc<SessionManager>) {
    // ... existing db/store/engine setup ...
    let chat_engine = Arc::new(MockChatEngine) as Arc<dyn ChatEngine>;
    let worker = MergeWorker::new(
        store.clone(), engine, chat_engine, queue.clone(), staging, session_mgr.clone(), config,
    );
    (worker, store, queue, session_mgr)
}
```

Update `test_failed_merge_retries` similarly (no more unreachable URL — use a FailingChatEngine mock).

**Step 7: Run all tests**

Run: `cargo test -p corvia-kernel`
Expected: All tests PASS.

**Step 8: Commit**

```bash
git add crates/corvia-kernel/src/merge_worker.rs crates/corvia-kernel/src/ollama_chat.rs \
  crates/corvia-kernel/src/lib.rs crates/corvia-kernel/src/agent_coordinator.rs \
  crates/corvia-cli/src/main.rs
git commit -m "refactor(d60): extract OllamaChatEngine, MergeWorker uses ChatEngine trait"
```

---

## Task 5: Implement GrpcInferenceEngine client

**Files:**
- Create: `crates/corvia-kernel/src/grpc_engine.rs`
- Modify: `crates/corvia-kernel/Cargo.toml`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Write failing test — GrpcInferenceEngine::new connects**

Add to `crates/corvia-kernel/src/grpc_engine.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_engine_dimensions() {
        let engine = GrpcInferenceEngine::new("http://127.0.0.1:8030", "test-model", 768);
        assert_eq!(engine.dimensions(), 768);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-kernel test_grpc_engine_dimensions`
Expected: FAIL — module doesn't exist.

**Step 3: Add corvia-proto dependency to corvia-kernel**

In `crates/corvia-kernel/Cargo.toml`, add:
```toml
corvia-proto.workspace = true
tonic = "0.12"
```

**Step 4: Implement GrpcInferenceEngine**

Create `crates/corvia-kernel/src/grpc_engine.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_proto::embedding_service_client::EmbeddingServiceClient;
use corvia_proto::{EmbedBatchRequest, EmbedRequest};
use tonic::transport::Channel;
use tracing::warn;

const MAX_EMBED_CHARS: usize = 4000;

pub struct GrpcInferenceEngine {
    endpoint: String,
    model: String,
    dimensions: usize,
}

impl GrpcInferenceEngine {
    pub fn new(endpoint: &str, model: &str, dimensions: usize) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            dimensions,
        }
    }

    async fn connect(&self) -> Result<EmbeddingServiceClient<Channel>> {
        let url = if self.endpoint.starts_with("http") {
            self.endpoint.clone()
        } else {
            format!("http://{}", self.endpoint)
        };
        EmbeddingServiceClient::connect(url)
            .await
            .map_err(|e| CorviaError::Embedding(format!("gRPC connect failed: {e}")))
    }

    fn truncate(text: &str) -> String {
        if text.len() > MAX_EMBED_CHARS {
            warn!("Truncating input from {} to {} chars", text.len(), MAX_EMBED_CHARS);
            text[..MAX_EMBED_CHARS].to_string()
        } else {
            text.to_string()
        }
    }
}

#[async_trait]
impl super::traits::InferenceEngine for GrpcInferenceEngine {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut client = self.connect().await?;
        let request = tonic::Request::new(EmbedRequest {
            model: self.model.clone(),
            text: Self::truncate(text),
        });
        let response = client.embed(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC Embed failed: {e}")))?;
        Ok(response.into_inner().embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut client = self.connect().await?;
        let request = tonic::Request::new(EmbedBatchRequest {
            model: self.model.clone(),
            texts: texts.iter().map(|t| Self::truncate(t)).collect(),
        });
        let response = client.embed_batch(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC EmbedBatch failed: {e}")))?;
        let mut embeddings: Vec<_> = response.into_inner().embeddings;
        embeddings.sort_by_key(|e| e.index);
        Ok(embeddings.into_iter().map(|e| e.values).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
```

**Step 5: Register module and update create_engine**

In `crates/corvia-kernel/src/lib.rs`, add:
```rust
pub mod grpc_engine;
```

Update `create_engine()`:
```rust
InferenceProvider::Corvia => Box::new(grpc_engine::GrpcInferenceEngine::new(
    &config.embedding.url,
    &config.embedding.model,
    config.embedding.dimensions,
)),
```

**Step 6: Run tests**

Run: `cargo test -p corvia-kernel`
Expected: All tests PASS.

**Step 7: Commit**

```bash
git add crates/corvia-kernel/src/grpc_engine.rs crates/corvia-kernel/Cargo.toml \
  crates/corvia-kernel/src/lib.rs
git commit -m "feat(d60): add GrpcInferenceEngine — gRPC client for InferenceEngine trait"
```

---

## Task 6: Implement GrpcChatEngine client

**Files:**
- Create: `crates/corvia-kernel/src/grpc_chat.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_chat_engine_creates() {
        let engine = GrpcChatEngine::new("http://127.0.0.1:8030");
        // Just verify construction doesn't panic
        assert!(true);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-kernel test_grpc_chat_engine_creates`
Expected: FAIL — module doesn't exist.

**Step 3: Implement GrpcChatEngine**

Create `crates/corvia-kernel/src/grpc_chat.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::ChatMessage;
use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::ChatRequest;
use tonic::transport::Channel;

pub struct GrpcChatEngine {
    endpoint: String,
}

impl GrpcChatEngine {
    pub fn new(endpoint: &str) -> Self {
        Self { endpoint: endpoint.to_string() }
    }

    async fn connect(&self) -> Result<ChatServiceClient<Channel>> {
        let url = if self.endpoint.starts_with("http") {
            self.endpoint.clone()
        } else {
            format!("http://{}", self.endpoint)
        };
        ChatServiceClient::connect(url)
            .await
            .map_err(|e| CorviaError::Agent(format!("gRPC chat connect failed: {e}")))
    }
}

#[async_trait]
impl super::traits::ChatEngine for GrpcChatEngine {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String> {
        let mut client = self.connect().await?;
        let proto_messages: Vec<corvia_proto::ChatMessage> = messages.iter().map(|m| {
            corvia_proto::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            }
        }).collect();

        let request = tonic::Request::new(ChatRequest {
            model: model.to_string(),
            messages: proto_messages,
            temperature: 0.7,
            max_tokens: 2048,
        });

        let response = client.chat(request).await
            .map_err(|e| CorviaError::Agent(format!("gRPC Chat failed: {e}")))?;

        let msg = response.into_inner().message
            .ok_or_else(|| CorviaError::Agent("gRPC Chat response missing message".into()))?;
        Ok(msg.content)
    }
}
```

**Step 4: Register module in lib.rs**

Add to `crates/corvia-kernel/src/lib.rs`:
```rust
pub mod grpc_chat;
```

**Step 5: Run tests**

Run: `cargo test -p corvia-kernel`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add crates/corvia-kernel/src/grpc_chat.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(d60): add GrpcChatEngine — gRPC client for ChatEngine trait"
```

---

## Task 7: Implement InferenceProvisioner

**Files:**
- Create: `crates/corvia-kernel/src/inference_provisioner.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provisioner_is_installed_check() {
        // Just verifies the check doesn't panic
        let _ = InferenceProvisioner::is_installed();
    }
}
```

**Step 2: Implement InferenceProvisioner**

Create `crates/corvia-kernel/src/inference_provisioner.rs`:
```rust
use corvia_common::errors::{CorviaError, Result};
use corvia_proto::model_manager_client::ModelManagerClient;
use corvia_proto::{HealthRequest, LoadModelRequest};
use tonic::transport::Channel;
use tracing::info;

/// Provisions the corvia-inference gRPC server.
/// Consistent with OllamaProvisioner: install → start → wait → load models.
pub struct InferenceProvisioner {
    grpc_addr: String,
}

impl InferenceProvisioner {
    pub fn new(grpc_addr: &str) -> Self {
        Self { grpc_addr: grpc_addr.to_string() }
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
        client.health(tonic::Request::new(HealthRequest {}))
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
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        while tokio::time::Instant::now() < deadline {
            if self.is_running().await {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(CorviaError::Infra(format!(
            "corvia-inference did not become ready within {timeout_secs}s at {}", self.grpc_addr
        )))
    }

    /// Load models on the server.
    pub async fn load_models(&self, embed_model: &str, chat_model: &str) -> Result<()> {
        let mut client = ModelManagerClient::connect(self.endpoint_url())
            .await
            .map_err(|e| CorviaError::Infra(format!("gRPC connect failed: {e}")))?;

        // Load embedding model
        let resp = client.load_model(tonic::Request::new(LoadModelRequest {
            name: embed_model.to_string(),
            model_type: "embedding".to_string(),
        })).await
            .map_err(|e| CorviaError::Infra(format!("LoadModel failed: {e}")))?;
        let resp = resp.into_inner();
        if !resp.success {
            return Err(CorviaError::Infra(format!("Failed to load embed model: {}", resp.error)));
        }
        info!("Loaded embedding model: {embed_model}");

        // Load chat model
        let resp = client.load_model(tonic::Request::new(LoadModelRequest {
            name: chat_model.to_string(),
            model_type: "chat".to_string(),
        })).await
            .map_err(|e| CorviaError::Infra(format!("LoadModel failed: {e}")))?;
        let resp = resp.into_inner();
        if !resp.success {
            return Err(CorviaError::Infra(format!("Failed to load chat model: {}", resp.error)));
        }
        info!("Loaded chat model: {chat_model}");

        Ok(())
    }

    /// Full provisioning: start if not running → wait → load models.
    pub async fn ensure_ready(&self, embed_model: &str, chat_model: &str) -> Result<()> {
        if !self.is_running().await {
            if !Self::is_installed() {
                return Err(CorviaError::Infra(
                    "corvia-inference not found. Install it or switch to provider = \"ollama\"".into()
                ));
            }
            self.start()?;
            self.wait_ready(15).await?;
        }
        self.load_models(embed_model, chat_model).await?;
        Ok(())
    }
}
```

**Step 3: Register module**

Add `pub mod inference_provisioner;` to `crates/corvia-kernel/src/lib.rs`.

**Step 4: Run tests**

Run: `cargo test -p corvia-kernel`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/corvia-kernel/src/inference_provisioner.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(d60): add InferenceProvisioner — auto-manages corvia-inference lifecycle"
```

---

## Task 8: Scaffold corvia-inference crate with health endpoint

**Files:**
- Create: `crates/corvia-inference/Cargo.toml`
- Create: `crates/corvia-inference/src/main.rs`
- Create: `crates/corvia-inference/src/model_manager.rs`
- Modify: `Cargo.toml` (workspace)

**Step 1: Create Cargo.toml**

Create `crates/corvia-inference/Cargo.toml`:
```toml
[package]
name = "corvia-inference"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "corvia-inference"
path = "src/main.rs"

[dependencies]
corvia-proto.workspace = true
tonic.workspace = true
prost = "0.13"
tokio.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
clap = { version = "4", features = ["derive"] }
serde.workspace = true
serde_json.workspace = true
```

**Step 2: Implement ModelManager service (health + list + load stubs)**

Create `crates/corvia-inference/src/model_manager.rs` with in-memory model registry:
```rust
use corvia_proto::model_manager_server::ModelManager;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct ModelEntry {
    pub name: String,
    pub model_type: String,
    pub loaded: bool,
}

pub struct ModelManagerService {
    pub models: Arc<RwLock<HashMap<String, ModelEntry>>>,
}

impl ModelManagerService {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[tonic::async_trait]
impl ModelManager for ModelManagerService {
    async fn health(&self, _req: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        let models = self.models.read().await;
        Ok(Response::new(HealthResponse {
            healthy: true,
            models_loaded: models.values().filter(|m| m.loaded).count() as u32,
        }))
    }

    async fn list_models(&self, _req: Request<ListModelsRequest>) -> Result<Response<ListModelsResponse>, Status> {
        let models = self.models.read().await;
        let statuses = models.values().map(|m| ModelStatus {
            name: m.name.clone(),
            model_type: m.model_type.clone(),
            loaded: m.loaded,
            memory_bytes: 0,
        }).collect();
        Ok(Response::new(ListModelsResponse { models: statuses }))
    }

    async fn load_model(&self, req: Request<LoadModelRequest>) -> Result<Response<LoadModelResponse>, Status> {
        let req = req.into_inner();
        tracing::info!(model = %req.name, model_type = %req.model_type, "load_model requested");
        // TODO: Actually load model in Tasks 9/10
        let mut models = self.models.write().await;
        models.insert(req.name.clone(), ModelEntry {
            name: req.name,
            model_type: req.model_type,
            loaded: true,
        });
        Ok(Response::new(LoadModelResponse { success: true, error: String::new() }))
    }

    async fn unload_model(&self, req: Request<UnloadModelRequest>) -> Result<Response<UnloadModelResponse>, Status> {
        let name = req.into_inner().name;
        let mut models = self.models.write().await;
        models.remove(&name);
        Ok(Response::new(UnloadModelResponse { success: true }))
    }
}
```

**Step 3: Create main.rs with tonic server**

Create `crates/corvia-inference/src/main.rs`:
```rust
mod model_manager;

use clap::Parser;
use corvia_proto::model_manager_server::ModelManagerServer;
use tonic::transport::Server;

#[derive(Parser)]
#[command(name = "corvia-inference")]
#[command(about = "Corvia inference server — gRPC embedding + chat")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Start the gRPC server
    Serve {
        #[arg(long, default_value = "8030")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corvia_inference=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port } => {
            let addr = format!("0.0.0.0:{port}").parse()?;
            let model_mgr = model_manager::ModelManagerService::new();

            tracing::info!("corvia-inference listening on {addr}");

            Server::builder()
                .add_service(ModelManagerServer::new(model_mgr))
                // TODO: add EmbeddingServiceServer in Task 9
                // TODO: add ChatServiceServer in Task 10
                .serve(addr)
                .await?;
        }
    }

    Ok(())
}
```

**Step 4: Add to workspace**

Add `"crates/corvia-inference"` to workspace members in root `Cargo.toml`.
Add `tonic = "0.12"` to `[workspace.dependencies]` if not already there.

**Step 5: Verify it compiles and runs**

Run: `cargo build -p corvia-inference`
Expected: SUCCESS — binary compiles.

Run: `cargo run -p corvia-inference -- serve --port 0` (in background, Ctrl-C)
Expected: Prints "corvia-inference listening on 0.0.0.0:0"

**Step 6: Commit**

```bash
git add crates/corvia-inference/ Cargo.toml
git commit -m "feat(d60): scaffold corvia-inference crate with ModelManager gRPC service"
```

---

## Task 9: Implement EmbeddingService with fastembed

**Files:**
- Create: `crates/corvia-inference/src/embedding_service.rs`
- Modify: `crates/corvia-inference/Cargo.toml`
- Modify: `crates/corvia-inference/src/main.rs`
- Modify: `crates/corvia-inference/src/model_manager.rs`

**Step 1: Add fastembed dependency**

In `crates/corvia-inference/Cargo.toml`:
```toml
fastembed = "5"
```

**Step 2: Implement EmbeddingService**

Create `crates/corvia-inference/src/embedding_service.rs`:
```rust
use corvia_proto::embedding_service_server::EmbeddingService;
use corvia_proto::*;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub struct EmbeddingServiceImpl {
    models: Arc<RwLock<HashMap<String, TextEmbedding>>>,
}

impl EmbeddingServiceImpl {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn models(&self) -> Arc<RwLock<HashMap<String, TextEmbedding>>> {
        self.models.clone()
    }

    /// Load an embedding model by name. Downloads from HuggingFace if not cached.
    pub async fn load_model(&self, name: &str) -> Result<(), Status> {
        let model_enum = match name {
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" =>
                EmbeddingModel::NomicEmbedTextV15,
            "nomic-embed-text-v1" | "nomic-ai/nomic-embed-text-v1" =>
                EmbeddingModel::NomicEmbedTextV1,
            "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" =>
                EmbeddingModel::BGESmallENV15,
            "bge-base-en-v1.5" | "BAAI/bge-base-en-v1.5" =>
                EmbeddingModel::BGEBaseENV15,
            "bge-large-en-v1.5" | "BAAI/bge-large-en-v1.5" =>
                EmbeddingModel::BGELargeENV15,
            other => return Err(Status::not_found(format!("Unknown embedding model: {other}"))),
        };

        tracing::info!(model = %name, "Loading embedding model...");
        let model = tokio::task::spawn_blocking(move || {
            TextEmbedding::try_new(InitOptions::new(model_enum).with_show_download_progress(true))
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        self.models.write().await.insert(name.to_string(), model);
        tracing::info!(model = %name, "Embedding model loaded");
        Ok(())
    }

    fn get_model_or_err<'a>(
        models: &'a HashMap<String, TextEmbedding>,
        name: &str,
    ) -> Result<&'a TextEmbedding, Status> {
        models.get(name).ok_or_else(|| Status::not_found(format!("Model '{name}' not loaded")))
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed(&self, req: Request<EmbedRequest>) -> Result<Response<EmbedResponse>, Status> {
        let req = req.into_inner();
        let models = self.models.read().await;
        let model = Self::get_model_or_err(&models, &req.model)?;

        let texts = vec![req.text];
        let embeddings = model.embed(texts, None)
            .map_err(|e| Status::internal(format!("Embed failed: {e}")))?;

        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| Status::internal("Empty embedding result"))?;
        let dimensions = embedding.len() as u32;

        Ok(Response::new(EmbedResponse { embedding, dimensions }))
    }

    async fn embed_batch(&self, req: Request<EmbedBatchRequest>) -> Result<Response<EmbedBatchResponse>, Status> {
        let req = req.into_inner();
        let models = self.models.read().await;
        let model = Self::get_model_or_err(&models, &req.model)?;

        let embeddings_raw = model.embed(req.texts, None)
            .map_err(|e| Status::internal(format!("EmbedBatch failed: {e}")))?;

        let embeddings = embeddings_raw.into_iter().enumerate().map(|(i, values)| {
            Embedding { values, index: i as u32 }
        }).collect();

        Ok(Response::new(EmbedBatchResponse { embeddings }))
    }

    async fn model_info(&self, req: Request<ModelInfoRequest>) -> Result<Response<ModelInfoResponse>, Status> {
        let name = req.into_inner().model;
        let models = self.models.read().await;
        let loaded = models.contains_key(&name);
        Ok(Response::new(ModelInfoResponse {
            model: name,
            dimensions: 768, // TODO: get from loaded model
            loaded,
        }))
    }
}
```

**Step 3: Wire into main.rs and model_manager**

In `crates/corvia-inference/src/main.rs`, add:
```rust
mod embedding_service;
use corvia_proto::embedding_service_server::EmbeddingServiceServer;
```

Update the server builder:
```rust
let embed_svc = embedding_service::EmbeddingServiceImpl::new();

Server::builder()
    .add_service(ModelManagerServer::new(model_mgr))
    .add_service(EmbeddingServiceServer::new(embed_svc))
    .serve(addr)
    .await?;
```

**Step 4: Verify it compiles**

Run: `cargo build -p corvia-inference`
Expected: SUCCESS.

**Step 5: Commit**

```bash
git add crates/corvia-inference/
git commit -m "feat(d60): implement EmbeddingService with fastembed ONNX backend"
```

---

## Task 10: Implement ChatService with candle

**Files:**
- Create: `crates/corvia-inference/src/chat_service.rs`
- Modify: `crates/corvia-inference/Cargo.toml`
- Modify: `crates/corvia-inference/src/main.rs`

**Step 1: Add candle dependencies**

In `crates/corvia-inference/Cargo.toml`:
```toml
candle-core = "0.8"
candle-transformers = "0.8"
candle-nn = "0.8"
hf-hub = "0.4"
tokenizers = "0.21"
```

Note: Check latest candle version on crates.io — use the current stable release.

**Step 2: Implement ChatService**

Create `crates/corvia-inference/src/chat_service.rs`:

This is the most complex component. The implementation needs to:
1. Load GGUF quantized models via candle
2. Tokenize input with HuggingFace tokenizers
3. Run autoregressive generation
4. Support both unary and streaming RPCs

Implementation outline:
```rust
use corvia_proto::chat_service_server::ChatService;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub struct ChatServiceImpl {
    // Chat models keyed by name
    // Each model entry holds the candle model + tokenizer
    models: Arc<RwLock<HashMap<String, ChatModelEntry>>>,
}

struct ChatModelEntry {
    // candle model weights + config
    // tokenizer
}

impl ChatServiceImpl {
    pub fn new() -> Self {
        Self { models: Arc::new(RwLock::new(HashMap::new())) }
    }

    pub async fn load_model(&self, name: &str) -> Result<(), Status> {
        // 1. Resolve HuggingFace repo (e.g., "llama3.2" → specific GGUF)
        // 2. Download via hf-hub if not cached
        // 3. Load GGUF into candle
        // 4. Load tokenizer
        // 5. Store in models map
        todo!("Implement candle GGUF model loading")
    }
}

#[tonic::async_trait]
impl ChatService for ChatServiceImpl {
    async fn chat(&self, req: Request<ChatRequest>) -> Result<Response<ChatResponse>, Status> {
        let req = req.into_inner();
        // 1. Get model from registry
        // 2. Format messages into prompt
        // 3. Tokenize
        // 4. Autoregressive generation loop
        // 5. Decode tokens to text
        // 6. Return ChatResponse
        todo!("Implement candle chat generation")
    }

    type ChatStreamStream = tokio_stream::wrappers::ReceiverStream<Result<ChatChunk, Status>>;

    async fn chat_stream(&self, req: Request<ChatRequest>) -> Result<Response<Self::ChatStreamStream>, Status> {
        // Same as chat() but yield tokens via channel
        todo!("Implement streaming chat")
    }
}
```

Note: The candle chat implementation is the most complex part of this plan. The exact code depends on candle's API for GGUF loading (which evolves). Reference the `candle-examples/examples/quantized/` directory in the candle repo for working GGUF inference code. Key pattern:

```rust
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file;
// Load: gguf_file::Content::read(&mut file) → ModelWeights::from_gguf(...)
// Generate: loop { model.forward(&input_ids, pos) → logits → sample → decode }
```

**Step 3: Add tokio-stream dependency**

In `crates/corvia-inference/Cargo.toml`:
```toml
tokio-stream = "0.1"
```

**Step 4: Wire into main.rs**

```rust
mod chat_service;
use corvia_proto::chat_service_server::ChatServiceServer;

// In server builder:
let chat_svc = chat_service::ChatServiceImpl::new();

Server::builder()
    .add_service(ModelManagerServer::new(model_mgr))
    .add_service(EmbeddingServiceServer::new(embed_svc))
    .add_service(ChatServiceServer::new(chat_svc))
    .serve(addr)
    .await?;
```

**Step 5: Verify it compiles**

Run: `cargo build -p corvia-inference`
Expected: SUCCESS (with todo!() stubs — they compile but panic at runtime).

**Step 6: Commit**

```bash
git add crates/corvia-inference/
git commit -m "feat(d60): scaffold ChatService with candle GGUF backend (stubs)"
```

---

## Task 11: Wire Corvia provider into CLI

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`
- Modify: `crates/corvia-cli/Cargo.toml`

**Step 1: Add corvia-proto dependency to CLI crate**

Add to `crates/corvia-cli/Cargo.toml`:
```toml
corvia-proto.workspace = true
```

**Step 2: Update cmd_init for Corvia provider**

In `cmd_init()`, add provisioning for the `Corvia` inference backend:
```rust
// After LiteStore initialization, provision inference
match config.embedding.provider {
    corvia_common::config::InferenceProvider::Corvia => {
        println!("  Provisioning Corvia inference server...");
        let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(
            &config.embedding.url,
        );
        provisioner.ensure_ready(&config.embedding.model, &config.merge.model).await?;
        println!("  Corvia inference ready (embed: {}, chat: {})",
            config.embedding.model, config.merge.model);
    }
    corvia_common::config::InferenceProvider::Ollama => {
        println!("  Provisioning Ollama...");
        let provisioner = OllamaProvisioner::new(&config.embedding.url);
        provisioner.ensure_ready(&config.embedding.model).await?;
        println!("  Ollama ready (model: {})", config.embedding.model);
    }
    corvia_common::config::InferenceProvider::Vllm => {
        // vLLM provisioned via Docker in full mode
    }
}
```

**Step 3: Update cmd_serve to construct correct ChatEngine**

In `cmd_serve()`, construct the appropriate `ChatEngine` based on merge config:
```rust
use corvia_kernel::traits::ChatEngine;

let chat_engine: Arc<dyn ChatEngine> = match config.merge.provider {
    corvia_common::config::InferenceProvider::Corvia => {
        Arc::new(corvia_kernel::grpc_chat::GrpcChatEngine::new(&config.embedding.url))
    }
    corvia_common::config::InferenceProvider::Ollama => {
        Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url))
    }
    corvia_common::config::InferenceProvider::Vllm => {
        // TODO: vLLM chat via gRPC
        Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url))
    }
};
```

Pass `chat_engine` to `AgentCoordinator::new()`.

**Step 4: Run tests**

Run: `cargo build -p corvia`
Expected: SUCCESS.

**Step 5: Commit**

```bash
git add crates/corvia-cli/
git commit -m "feat(d60): wire Corvia inference provider into CLI init/serve"
```

---

## Task 12: gRPC Integration Tests

**Files:**
- Create: `tests/integration/grpc_inference_test.rs`
- Modify: `crates/corvia-cli/Cargo.toml`

**Step 1: Write gRPC round-trip test**

Create `tests/integration/grpc_inference_test.rs`:
```rust
//! Integration test: start corvia-inference, connect as gRPC client, embed text.
//! Requires: `cargo build -p corvia-inference` first.
//! Gated behind real-inference feature (needs model download).

use corvia_proto::embedding_service_client::EmbeddingServiceClient;
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
#[ignore] // Run with: cargo test --test grpc_inference_test -- --ignored
async fn test_grpc_embed_roundtrip() {
    // Start server
    let port = 18030; // test port
    let addr = format!("127.0.0.1:{port}");
    let mut child = std::process::Command::new("cargo")
        .args(["run", "-p", "corvia-inference", "--", "serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start corvia-inference");

    assert!(wait_for_server(&addr, Duration::from_secs(30)).await, "Server failed to start");

    let url = format!("http://{addr}");

    // Load model
    let mut mgr = ModelManagerClient::connect(url.clone()).await.unwrap();
    let resp = mgr.load_model(tonic::Request::new(LoadModelRequest {
        name: "bge-small-en-v1.5".into(), // small model for fast test
        model_type: "embedding".into(),
    })).await.unwrap();
    assert!(resp.into_inner().success);

    // Embed single text
    let mut embed = EmbeddingServiceClient::connect(url.clone()).await.unwrap();
    let resp = embed.embed(tonic::Request::new(EmbedRequest {
        model: "bge-small-en-v1.5".into(),
        text: "hello world".into(),
    })).await.unwrap();
    let resp = resp.into_inner();
    assert_eq!(resp.dimensions, 384); // bge-small-en-v1.5 = 384 dims
    assert_eq!(resp.embedding.len(), 384);

    // Embed batch
    let resp = embed.embed_batch(tonic::Request::new(EmbedBatchRequest {
        model: "bge-small-en-v1.5".into(),
        texts: vec!["hello".into(), "world".into()],
    })).await.unwrap();
    let resp = resp.into_inner();
    assert_eq!(resp.embeddings.len(), 2);
    assert_eq!(resp.embeddings[0].index, 0);
    assert_eq!(resp.embeddings[1].index, 1);

    // Cleanup
    child.kill().ok();
}
```

**Step 2: Register test in Cargo.toml**

Add to `crates/corvia-cli/Cargo.toml`:
```toml
[[test]]
name = "grpc_inference_test"
path = "../../tests/integration/grpc_inference_test.rs"
```

**Step 3: Run test (manually, requires model download)**

Run: `cargo test --test grpc_inference_test -- --ignored`
Expected: PASS (downloads bge-small-en-v1.5 ONNX model ~50MB on first run).

**Step 4: Commit**

```bash
git add tests/integration/grpc_inference_test.rs crates/corvia-cli/Cargo.toml
git commit -m "test(d60): add gRPC inference integration test"
```

---

## Task 13: Implement GrpcVllmEngine (vLLM gRPC client)

**Files:**
- Create: `crates/corvia-proto/proto/vllm/vllm_engine.proto` (imported from vLLM repo)
- Modify: `crates/corvia-proto/build.rs`
- Modify: `crates/corvia-proto/src/lib.rs`
- Create: `crates/corvia-kernel/src/grpc_vllm_engine.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Import vLLM proto**

Download the proto from vLLM v0.14.0 repo:
`vllm/grpc/vllm_engine.proto` → `crates/corvia-proto/proto/vllm/vllm_engine.proto`

Note: The exact proto contents must be sourced from the vLLM repository at tag v0.14.0. If the proto is not publicly documented, this task should be deferred until the proto is confirmed.

**Step 2: Add to tonic-build**

In `crates/corvia-proto/build.rs`, add:
```rust
tonic_build::configure()
    .build_server(false)  // client only for vLLM
    .build_client(true)
    .compile_protos(
        &["proto/vllm/vllm_engine.proto"],
        &["proto/"],
    )?;
```

**Step 3: Implement GrpcVllmEngine**

Create `crates/corvia-kernel/src/grpc_vllm_engine.rs` implementing `InferenceEngine` using vLLM's gRPC embedding endpoint.

**Step 4: Update create_engine for Vllm provider**

In `crates/corvia-kernel/src/lib.rs`, update:
```rust
InferenceProvider::Vllm => Box::new(grpc_vllm_engine::GrpcVllmEngine::new(
    &config.embedding.url,
    &config.embedding.model,
    config.embedding.dimensions,
)),
```

**Step 5: Run tests, commit**

Run: `cargo test -p corvia-kernel`
Expected: All tests PASS.

```bash
git add crates/corvia-proto/ crates/corvia-kernel/
git commit -m "feat(d60): add GrpcVllmEngine — vLLM gRPC client for InferenceEngine"
```

---

## Task 14: Fill in candle chat implementation

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

This task replaces the `todo!()` stubs from Task 10 with working candle GGUF inference.

**Step 1: Implement model loading**

Use `hf-hub` to download GGUF files, `candle_core::quantized::gguf_file` to parse them, and `candle_transformers::models::quantized_llama::ModelWeights` to load weights.

**Step 2: Implement generate loop**

Standard autoregressive generation:
```
tokenize → forward → sample logits → append token → repeat until EOS or max_tokens
```

**Step 3: Implement streaming via channel**

For `ChatStream`, use a `tokio::sync::mpsc::channel` and send `ChatChunk` messages as tokens are generated.

**Step 4: Test with a small model**

Use a small GGUF model (e.g., TinyLlama 1.1B Q4) for testing.

**Step 5: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(d60): implement ChatService with candle GGUF generation"
```

---

## Summary: Dependency Order

```
Task 1: corvia-proto crate ─────────────────────────────────────┐
Task 2: ChatEngine trait ────────────────────────┐              │
Task 3: Config changes ─────────────────────────┐│              │
Task 4: Refactor MergeWorker ◄──────────────────┘│              │
Task 5: GrpcInferenceEngine ◄───────────────────────────────────┤
Task 6: GrpcChatEngine ◄───────────────────────────────────────┤
Task 7: InferenceProvisioner ◄──────────────────────────────────┤
Task 8: Scaffold corvia-inference ◄─────────────────────────────┤
Task 9: EmbeddingService (fastembed) ◄──────── Task 8          │
Task 10: ChatService scaffold ◄─────────────── Task 8          │
Task 11: CLI wiring ◄──── Tasks 3,4,5,6,7                     │
Task 12: Integration tests ◄── Tasks 5,6,8,9                  │
Task 13: GrpcVllmEngine ◄─────────────────────────────────────┘
Task 14: Candle chat impl ◄── Task 10
```

Tasks 1-4 are foundation (must be sequential).
Tasks 5-7 can be parallelized after Task 1.
Tasks 8-10 can be parallelized after Task 1.
Task 11 depends on Tasks 3-7.
Task 12 depends on Tasks 5,6,8,9.
Task 13 can be done anytime after Task 1.
Task 14 can be done anytime after Task 10.
