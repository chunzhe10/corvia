# D60: gRPC Inference Server Design

**Date:** 2026-03-02
**Status:** Approved
**Depends on:** D35 (Two-tier storage), D40 (Ollama default), D53 (Python SDK)

## Summary

Add a custom gRPC inference server (`corvia-inference`) that handles both embedding and chat/reasoning inference. This replaces Ollama as the default inference backend for LiteStore, making the zero-Docker promise truly dependency-free. gRPC (HTTP/2 + protobuf) replaces HTTP/JSON on all inference paths except legacy Ollama.

## Motivation

1. **Ollama is overhead for embeddings.** Ollama is a 500MB Go binary wrapping llama.cpp, designed for LLM serving. For embeddings (a single forward pass), most of its features are dead weight. The HTTP/JSON layer adds network latency and 2x serialization overhead for float vectors.

2. **Zero-Docker gap.** D35 promised "LiteStore = full product, zero Docker" but LiteStore still requires Ollama installed. Users must `ollama pull nomic-embed-text` before `corvia ingest` works. This is adoption friction.

3. **Multi-model future.** Different knowledge formats may need different embedding models. LLM reasoning (merge, summarization) needs chat models. A unified inference server supports multiple models simultaneously with shared GPU memory.

4. **Hardware optimization.** ONNX Runtime's execution providers (CoreML on Apple, OpenVINO on Intel, TensorRT on NVIDIA) give automatic hardware acceleration. Ollama's llama.cpp has narrower hardware support for encoder models.

5. **Protocol efficiency.** A 768-dim f32 vector is 3,072 bytes in packed protobuf vs ~6,000 bytes in JSON. gRPC's HTTP/2 multiplexing eliminates head-of-line blocking for concurrent embedding requests.

## Architecture

```
corvia-kernel (client side)
├── GrpcInferenceEngine  ──gRPC──►  corvia-inference:8030  (Corvia proto)
├── GrpcVllmEngine       ──gRPC──►  vLLM:8001              (vLLM proto)
└── OllamaEngine         ──HTTP──►  Ollama:11434            (OpenAI-compat JSON)
```

### Product Matrix

|                  | LiteStore (default)        | FullStore (opt-in)                  |
|------------------|----------------------------|-------------------------------------|
| Store            | HNSW + Redb + Git          | SurrealDB + Git                     |
| Inference        | corvia-inference (gRPC)    | vLLM (gRPC) or corvia-inference     |
| Chat (merge)     | corvia-inference (gRPC)    | vLLM/Ollama or corvia-inference     |
| Prerequisites    | None                       | Docker + SurrealDB                  |
| Zero-Docker      | Yes, truly                 | No (by design)                      |

### Three Inference Backends

| Backend            | Protocol | Default for | Best when                                    | Dependencies         |
|--------------------|----------|-------------|----------------------------------------------|----------------------|
| corvia-inference   | gRPC     | LiteStore   | Zero-dependency, single-machine, dev/teams   | None (auto-starts)   |
| vLLM               | gRPC     | FullStore   | Scale, multi-GPU, production inference       | Python + CUDA        |
| Ollama             | HTTP     | (opt-in)    | Already have Ollama, share GPU across apps   | Ollama binary        |

**Protocol principle:** gRPC everywhere. HTTP only for legacy Ollama compatibility.

## New Crates

### corvia-proto

Shared protobuf definitions and generated Rust code.

Contains:
- `proto/corvia/inference/v1/embedding.proto` — Corvia EmbeddingService
- `proto/corvia/inference/v1/chat.proto` — Corvia ChatService
- `proto/corvia/inference/v1/model.proto` — Corvia ModelManager
- `proto/vllm/vllm_engine.proto` — vLLM gRPC proto (imported from vLLM v0.14.0)

Dependencies: `tonic`, `prost`, `tonic-build`

### corvia-inference

Separate binary. gRPC server with ONNX Runtime (embeddings) + candle (chat).

Dependencies: `tonic`, `corvia-proto`, `ort`/`fastembed`, `candle-core`, `candle-transformers`, `hf-hub`, `tokenizers`

## Proto Definitions

### EmbeddingService

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

### ChatService

```protobuf
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

### ModelManager

```protobuf
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

## Integration with Existing Code

### Config Changes (corvia-common)

`InferenceProvider` enum gains `Corvia` variant:

```rust
pub enum InferenceProvider {
    #[default]
    Corvia,     // NEW default — gRPC to corvia-inference
    Ollama,     // Opt-in — HTTP to Ollama
    Vllm,       // FullStore — gRPC to vLLM
}
```

Default config changes:

```toml
[embedding]
provider = "corvia"
model = "nomic-embed-text-v1.5"
url = "127.0.0.1:8030"
dimensions = 768
```

New field in `MergeConfig`:

```toml
[merge]
model = "llama3.2"
provider = "corvia"   # NEW — chat model provider for merge
```

### New Trait: ChatEngine (corvia-kernel)

```rust
#[async_trait]
pub trait ChatEngine: Send + Sync {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<String>;
}
```

Implementations:
- `GrpcChatEngine` — speaks Corvia ChatService proto
- MergeWorker takes `Arc<dyn ChatEngine>` instead of `ollama_url: String`

### New Engines (corvia-kernel)

**GrpcInferenceEngine** — implements `InferenceEngine` via Corvia EmbeddingService gRPC client.

**GrpcVllmEngine** — implements `InferenceEngine` via vLLM's gRPC proto. Replaces the existing HTTP-based `VllmEngine`.

**GrpcChatEngine** — implements `ChatEngine` via Corvia ChatService gRPC client.

### InferenceProvisioner (corvia-kernel)

Consistent with `OllamaProvisioner` pattern:

```rust
impl InferenceProvisioner {
    pub fn is_installed() -> bool;           // check `which corvia-inference`
    pub async fn is_running(&self) -> bool;  // gRPC Health check
    pub fn start(&self) -> Result<()>;       // spawn `corvia-inference serve`
    pub async fn wait_ready(&self, timeout_secs: u64) -> Result<()>;
    pub async fn ensure_ready(&self, embed_model: &str, chat_model: &str) -> Result<()>;
}
```

## corvia-inference Server Internals

### Runtimes

- **Embedding models:** ONNX Runtime via `ort` crate (or `fastembed` for model management + tokenization). Execution providers: CPU (default), CUDA, CoreML, OpenVINO, TensorRT, DirectML.
- **Chat models:** candle (`candle-core` + `candle-transformers`). Loads GGUF quantized models. Supports llama, mistral, phi architectures.

### Model Registry

- Auto-downloads models from HuggingFace Hub on first `LoadModel` call
- Cache directory: `~/.cache/corvia-inference/`
- Multiple models loaded simultaneously (keyed by name)
- Embedding models: ONNX format from HuggingFace (~137MB for nomic-embed-text-v1.5)
- Chat models: GGUF format from HuggingFace (~2GB for llama3.2 3B Q4)

### Server Startup

```
corvia-inference serve [--port 8030] [--embed-model nomic-embed-text-v1.5] [--chat-model llama3.2]
```

1. Start tonic gRPC server on port 8030
2. Download and load default models if specified
3. Respond to Health, LoadModel, Embed, Chat RPCs

## Error Handling

| Scenario              | gRPC Status          | Client behavior                                  |
|-----------------------|----------------------|--------------------------------------------------|
| Model not loaded      | `NOT_FOUND`          | InferenceProvisioner calls LoadModel, retries     |
| Model download fails  | `UNAVAILABLE`        | Return CorviaError::Embedding with context        |
| Input too long        | `INVALID_ARGUMENT`   | Client truncation + server validation             |
| Server overloaded     | `RESOURCE_EXHAUSTED` | Backoff + retry                                   |
| Server crashed        | `UNAVAILABLE`        | InferenceProvisioner restarts, reconnects         |
| OOM during inference  | `INTERNAL`           | Return error, log memory usage                    |

Reconnection: tonic Channel handles automatic reconnection on server restart.

## Testing Strategy

1. **Unit tests (corvia-inference):** model loading, output shape validation, tokenization
2. **gRPC integration tests:** start server on random port, round-trip embed/chat calls, verify packed float encoding
3. **Existing tests unaffected:** all 117 tests use MockEngine
4. **Real model e2e tests:** CI-gated behind `--features real-inference`

## Crate Dependency Graph

```
corvia-proto  (shared protobuf types)
    ↑              ↑
corvia-kernel    corvia-inference
(gRPC client)   (gRPC server + ort + candle)
    ↑
corvia-server  corvia-cli
```

## Defaults

- Port: 8030
- Embedding model: nomic-embed-text-v1.5 (768 dimensions)
- Chat model: llama3.2 (3B, GGUF Q4)
- Model cache: ~/.cache/corvia-inference/
- Env overrides: `CORVIA_INFERENCE_URL`, `CORVIA_INFERENCE_PORT`
