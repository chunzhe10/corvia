# Chat Inference for corvia-inference

> **Status:** Shipped (v0.3.6)

> Replace the stub ChatService in corvia-inference with real LLM text generation
> powered by llama.cpp via the `llama-cpp-2` Rust crate.

## Context

corvia-inference serves embeddings via fastembed (ONNX Runtime) but its ChatService
is a stub returning placeholder text. The RAG pipeline's `ask()` mode and the merge
worker's conflict resolution both depend on a working GenerationEngine. The gRPC
client (`GrpcChatEngine` in corvia-kernel) and proto definitions are already wired —
only the server-side implementation is missing.

## Decision: Dual Runtime

- **Embeddings**: fastembed (ONNX Runtime) — unchanged, production-proven
- **Chat/generation**: llama-cpp-2 (llama.cpp) — new, best-in-class for GGUF inference

Rationale: fastembed is purpose-built for embeddings with ergonomic model management.
llama.cpp is the standard for local LLM inference with native chat template support.
Consolidating to a single runtime would sacrifice embedding ergonomics for marginal
simplification.

## Architecture

```
corvia-inference (gRPC server, port 8030)
├── EmbeddingService  — fastembed (ONNX Runtime), unchanged
├── ChatService       — llama-cpp-2 (llama.cpp), NEW
└── ModelManager      — loads both embedding + chat models
```

Proto definitions (`ChatRequest`, `ChatResponse`, `ChatChunk`) are unchanged.
The `GrpcChatEngine` client in corvia-kernel is unchanged. Only the server-side
`ChatServiceImpl` is rewritten.

## Chat Request Flow

1. Receive `ChatRequest` (model, messages, temperature, max_tokens)
2. Format messages using llama.cpp's native chat template support —
   `model.apply_chat_template()` extracts Jinja template from GGUF metadata
3. Tokenize the formatted prompt
4. Run inference on `tokio::task::spawn_blocking` (llama.cpp is synchronous)
5. Sample token-by-token with temperature/top-p
6. Return `ChatResponse` with generated text + token counts

### Streaming (chat_stream)

Same flow, but tokens sent via `tokio::sync::mpsc` channel as `ChatChunk` messages
back to the gRPC stream.

## Concurrency Model

- One `LlamaModel` per loaded model — `Arc`-wrapped, thread-safe (read-only weights)
- One `LlamaContext` per active request — created on blocking thread, not shared
- `tokio::task::spawn_blocking` for all inference work
- No explicit concurrency limit to start; tokio's blocking pool provides backpressure

## Model Resolution

Hardcoded lookup table (same pattern as fastembed's `resolve_model`):

| Short name     | HF repo                                    | Quant   | Size  |
|----------------|--------------------------------------------|---------|-------|
| `llama3.2`     | `bartowski/Llama-3.2-3B-Instruct-GGUF`    | Q4_K_M  | ~2GB  |
| `llama3.2:1b`  | `bartowski/Llama-3.2-1B-Instruct-GGUF`    | Q4_K_M  | ~0.8GB|

Models downloaded via `hf-hub` crate, cached in `~/.cache/huggingface/hub/`.

## New Dependencies

```toml
# corvia-inference/Cargo.toml
llama-cpp-2 = "0.1"       # llama.cpp Rust bindings
hf-hub = "0.4"             # HuggingFace model downloads
```

Build requirements: CMake + C++17 compiler (present in devcontainer).

## Error Handling

- Unknown model name: `Status::not_found`
- HF download failure: `Status::internal`
- OOM on context creation: `Status::resource_exhausted`
- Token limit exceeded: stop generation, return partial response
- EOS token: normal stop
- No chat model loaded: `Status::not_found` (existing behavior)

No retry logic, request queuing, or model eviction for v1.

## Testing

**Tier 1** (no external deps):
- Model name resolution tests
- Error case: chat with unloaded model

**Tier 3** (requires GGUF model download):
- Load llama3.2:1b, run chat, verify non-empty response
- Streaming: verify chunks arrive, final chunk has `done: true`
- Auto-skip when model unavailable

Existing RAG e2e tests (mock generators) unchanged for fast Tier 1 testing.

## Files Changed

| File | Change |
|------|--------|
| `corvia-inference/Cargo.toml` | Add `llama-cpp-2`, `hf-hub` |
| `corvia-inference/src/chat_service.rs` | Rewrite stub with llama-cpp-2 |
| `corvia-inference/src/model_manager.rs` | Minor: download progress logging |

## Out of Scope

- Embedding service changes (none)
- Proto changes (none)
- GrpcChatEngine client changes (none)
- Config changes (none)
- GPU acceleration (CPU-only for v1, llama.cpp supports CUDA/Metal via feature flags later)
- Model eviction / memory management
- Request queuing / rate limiting
