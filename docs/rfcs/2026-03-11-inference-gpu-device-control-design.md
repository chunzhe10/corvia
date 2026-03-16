# GPU/CPU Device Control for corvia-inference

**Date:** 2026-03-11
**Status:** Shipped (v0.4.2)
**Scope:** corvia-inference crate only (no kernel/server changes)

## Problem

corvia-inference runs both ONNX (fastembed) embeddings and llama.cpp chat inference
on CPU only, despite NVIDIA RTX 3060 and Intel iGPU being available in the
devcontainer. There is no mechanism to select CPU vs GPU at runtime.

## Design

### gRPC Protocol Changes

Extend `LoadModelRequest` with two optional fields for device control:

```protobuf
message LoadModelRequest {
  string name = 1;
  string model_type = 2;        // "embedding" | "chat"
  string device = 3;            // "auto" | "gpu" | "cpu" (default: "auto")
  string backend = 4;           // optional override: "cuda", "openvino", "vulkan", ""
}
```

Extend `LoadModelResponse` to report what actually happened (fallback-with-signal):

```protobuf
message LoadModelResponse {
  bool success = 1;
  string error = 2;
  string actual_device = 3;     // "gpu" | "cpu"
  string actual_backend = 4;    // "cuda", "openvino", "vulkan", "cpu"
}
```

- `device` defaults to `"auto"` (empty string treated as auto).
- `backend` defaults to empty (system picks best available).
- `actual_device`/`actual_backend` always report what was used, enabling the
  fallback-with-signal pattern: if GPU unavailable, load succeeds on CPU but
  the caller sees the mismatch.

### Backend Resolution Logic

A `resolve_backend()` function maps `(device, backend, model_type)` to a concrete
backend selection:

```
1. If backend is explicitly set ("cuda", "openvino", "vulkan", "cpu"):
   → validate availability, use it or return error
   → "openvino" is invalid for chat models (llama.cpp doesn't support it)

2. If device = "gpu":
   → try CUDA first (probe NVIDIA availability)
   → then OpenVINO (probe Intel GPU + libs) [embedding only]
   → then Vulkan (for llama-cpp-2 only) [future]
   → if none available: fallback to CPU, set fallback_used = true

3. If device = "cpu":
   → use CPU directly

4. If device = "auto" (default):
   → same as "gpu" (prefer GPU, fallback to CPU)
```

GPU availability is probed once at startup and cached (doesn't change mid-process).

```rust
struct ResolvedBackend {
    device: Device,         // Gpu | Cpu
    backend: BackendKind,   // Cuda | OpenVino | Vulkan | Cpu
    fallback_used: bool,    // true if fell back from requested device
}
```

### "auto" Behavior

When both NVIDIA and Intel GPUs are present, `"auto"` (and `"gpu"`) prefer
NVIDIA/CUDA. Intel GPU is only used when explicitly requested via
`backend: "openvino"` (embeddings) or `backend: "vulkan"` (chat, future).

Power users who want to split workloads (e.g., embeddings on Intel, chat on
NVIDIA) achieve this by passing explicit `backend` per `load_model` call.

### Embedding Service Changes (fastembed)

Switch from `fastembed::InitOptions` to `InitOptionsUserDefined` to unlock
`with_execution_providers()` for per-model EP control.

Define a model registry mapping friendly names to `UserDefinedEmbeddingModel`
structs (ONNX paths, tokenizer config, dimensions).

Build execution providers based on resolved backend:

```rust
fn build_execution_providers(backend: &ResolvedBackend) -> Vec<ExecutionProviderDispatch> {
    match backend.backend {
        BackendKind::Cuda => vec![CUDAExecutionProvider::default().build()],
        BackendKind::OpenVino => vec![
            OpenVINOExecutionProvider::default()
                .with_device_type("GPU_FP16")
                .build()
        ],
        BackendKind::Cpu => vec![],  // ort defaults to CPU EP
    }
}
```

`LoadedModel` gains a `backend: ResolvedBackend` field for reporting.

### Chat Service Changes (llama-cpp-2)

Configure `LlamaModelParams` GPU layer offloading based on resolved backend:

```rust
fn build_model_params(backend: &ResolvedBackend) -> LlamaModelParams {
    match backend.backend {
        BackendKind::Cuda | BackendKind::Vulkan => {
            LlamaModelParams::default().with_n_gpu_layers(999)
        }
        _ => LlamaModelParams::default(),  // CPU, n_gpu_layers = 0
    }
}
```

If `backend: "openvino"` is requested for a chat model, return a gRPC error
(llama.cpp does not support OpenVINO).

### Cargo.toml Changes

```toml
# corvia-inference/Cargo.toml
fastembed = { version = "5", features = ["cuda"] }
ort = { version = "=2.0.0-rc.11", features = ["openvino", "cuda"] }
llama-cpp-2 = { version = "0.1", features = ["cuda"] }
```

The `ort` direct dependency with matching pinned version ensures Cargo unifies
features so fastembed's internal ort gets both CUDA and OpenVINO.

Vulkan for llama-cpp-2 is deferred to a follow-up to limit compile complexity.

### Devcontainer Changes

NVIDIA GPU already works (`--gpus all`). Intel GPU is **opt-in** — documented
but commented out by default to keep the container starting on all hosts:

```jsonc
// devcontainer.json runArgs additions for Intel GPU (uncomment if available):
// "--device=/dev/dri",
// "--device=/dev/dxg",
// "--group-add", "video",
// "--group-add", "render"
```

Plus a WSL GPU driver mount:

```jsonc
// "source=/usr/lib/wsl,target=/usr/lib/wsl,type=bind,readonly"
```

Dockerfile additions for Intel GPU runtime (opt-in):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    && rm -rf /var/lib/apt/lists/*
```

## Change Summary

| Area | Change | Risk |
|------|--------|------|
| corvia-proto | Add device/backend to LoadModelRequest; actual_device/actual_backend to LoadModelResponse | Low |
| corvia-inference/backend.rs | New module: resolve_backend(), GPU probing, ResolvedBackend type | Medium |
| corvia-inference/embedding_service.rs | Switch to InitOptionsUserDefined, EP selection per backend | Medium |
| corvia-inference/chat_service.rs | build_model_params() with GPU layers, backend tracking | Low |
| corvia-inference/model_manager.rs | Thread device/backend through, return actual_* in response | Low |
| corvia-inference/Cargo.toml | Add ort direct dep, enable cuda/openvino features | Medium |
| Dockerfile | Intel GPU runtime packages (opt-in) | Low |
| devcontainer.json | Documented opt-in for Intel GPU device passthrough | Low |

## Out of Scope

- No changes to corvia.toml or kernel/server layers
- No hot-switching (changing device requires unload + reload)
- No NPU support (Intel NPU via OpenVINO deferred)
- Vulkan for llama-cpp-2 deferred to follow-up
