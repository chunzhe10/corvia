# GPU/CPU Device Control for corvia-inference — Implementation Plan

> **Status:** Shipped (v0.4.2)

**Goal:** Add runtime GPU/CPU device selection to corvia-inference so models can use CUDA (NVIDIA) or OpenVINO (Intel iGPU) execution providers instead of CPU-only inference.

**Architecture:** Extend the gRPC `LoadModelRequest` with `device`/`backend` fields. A new `backend.rs` module resolves these to a concrete `ResolvedBackend` (CUDA/OpenVINO/CPU) based on probed hardware. Embedding service passes execution providers to fastembed; chat service configures llama.cpp GPU layer offloading. Fallback-with-signal pattern: if GPU unavailable, load succeeds on CPU but response reports the mismatch.

**Tech Stack:** Rust, tonic/prost (gRPC), ort 2.0.0-rc.11 (ONNX Runtime EPs), fastembed 5.12 (embeddings), llama-cpp-2 0.1 (chat), protobuf

**Design spec:** `2026-03-11-inference-gpu-device-control-design.md`

**Deviation from spec:** The design spec calls for switching to `InitOptionsUserDefined`. Investigation shows `fastembed::InitOptions` (the standard `InitOptionsWithLength<EmbeddingModel>`) already has `with_execution_providers()`, so we use it directly — simpler and no BYOM plumbing needed.

**Compile-time note:** `ort` features (`cuda`, `openvino`) are runtime-loaded (no CUDA toolkit needed at build time). `llama-cpp-2/cuda` requires CUDA toolkit at build time, so it's gated behind a `cuda` cargo feature on `corvia-inference`.

**Hardware available (verified 2026-03-13):**
- **NVIDIA RTX 3060 Laptop GPU** — 6GB VRAM, CUDA 13.0, driver 580.126.09, compute 8.6. `nvidia-smi` works; CUDA probe returns `true`.
- **Intel Iris Xe (Alder Lake-P GT2, 0x46a6)** — `/dev/dri/card1` present, but OpenVINO libs not installed. Probe returns `openvino_available: false`. Install `intel-opencl-icd` + `level-zero` + OpenVINO runtime to enable.

**Status:** All implementation tasks complete. Config (`corvia.toml`) has `device = "auto"` which will auto-select CUDA on this host.

---

## Chunk 1: Protocol + Backend Types (Foundation)

### Task 1: Proto Changes

**Files:**
- Modify: `crates/corvia-proto/proto/corvia/inference/v1/model.proto`

-**Step 1: Add device/backend fields to LoadModelRequest and LoadModelResponse**

```protobuf
message LoadModelRequest {
  string name = 1;
  string model_type = 2;        // "embedding" | "chat"
  string device = 3;            // "auto" | "gpu" | "cpu" (default: "auto")
  string backend = 4;           // optional override: "cuda", "openvino", "vulkan", ""
}
message LoadModelResponse {
  bool success = 1;
  string error = 2;
  string actual_device = 3;     // "gpu" | "cpu"
  string actual_backend = 4;    // "cuda", "openvino", "vulkan", "cpu"
}
```

-**Step 2: Verify proto compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build -p corvia-proto 2>&1 | tail -5`
Expected: compiles successfully

-**Step 3: Verify workspace still builds**

Run: `cargo build -p corvia-inference 2>&1 | tail -10`
Expected: compiles (new fields have defaults, existing code ignores them)

-**Step 4: Commit**

```bash
git add crates/corvia-proto/proto/corvia/inference/v1/model.proto
git commit -m "feat(proto): add device/backend fields to LoadModelRequest/Response

Adds device (auto|gpu|cpu) and backend (cuda|openvino|vulkan) to LoadModelRequest.
Adds actual_device and actual_backend to LoadModelResponse for fallback-with-signal."
```

---

### Task 2: Backend Resolution Module

**Files:**
- Create: `crates/corvia-inference/src/backend.rs`
- Modify: `crates/corvia-inference/src/main.rs` (add `mod backend;`)

-**Step 1: Write failing tests for backend resolution**

Create `crates/corvia-inference/src/backend.rs` with the test module first:

```rust
/// GPU/CPU backend resolution for corvia-inference.
///
/// Probes hardware once at startup and caches availability.
/// `resolve_backend()` maps (device, backend, model_type) → ResolvedBackend.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    OpenVino,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Embedding,
    Chat,
}

#[derive(Debug, Clone)]
pub struct ResolvedBackend {
    pub device: Device,
    pub backend: BackendKind,
    pub fallback_used: bool,
}

#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub cuda_available: bool,
    pub openvino_available: bool,
}

impl GpuCapabilities {
    /// Probe actual hardware. Call once at startup.
    pub fn probe() -> Self {
        todo!()
    }

    /// Create with explicit values (for testing).
    pub fn new(cuda: bool, openvino: bool) -> Self {
        Self {
            cuda_available: cuda,
            openvino_available: openvino,
        }
    }
}

pub fn resolve_backend(
    device: &str,
    backend: &str,
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    todo!()
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Gpu => write!(f, "gpu"),
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "cuda"),
            BackendKind::OpenVino => write!(f, "openvino"),
            BackendKind::Cpu => write!(f, "cpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Explicit backend ---

    #[test]
    fn explicit_cuda_available() {
        let gpu = GpuCapabilities::new(true, false);
        let r = resolve_backend("", "cuda", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn explicit_cuda_unavailable() {
        let gpu = GpuCapabilities::new(false, false);
        let r = resolve_backend("", "cuda", ModelType::Embedding, &gpu);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("CUDA"));
    }

    #[test]
    fn explicit_openvino_embedding() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("", "openvino", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::OpenVino);
        assert_eq!(r.device, Device::Gpu);
    }

    #[test]
    fn explicit_openvino_chat_rejected() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("", "openvino", ModelType::Chat, &gpu);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("not support"));
    }

    #[test]
    fn explicit_cpu() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("", "cpu", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn explicit_unknown_backend() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("", "rocm", ModelType::Embedding, &gpu);
        assert!(r.is_err());
    }

    // --- Device-based resolution ---

    #[test]
    fn device_cpu() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("cpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
    }

    #[test]
    fn device_gpu_prefers_cuda() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn device_gpu_falls_back_to_openvino_for_embedding() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::OpenVino);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn device_gpu_no_gpu_falls_back_to_cpu() {
        let gpu = GpuCapabilities::new(false, false);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(r.fallback_used);
    }

    #[test]
    fn device_gpu_chat_skips_openvino() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("gpu", "", ModelType::Chat, &gpu).unwrap();
        // OpenVINO not valid for chat, so falls back to CPU
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(r.fallback_used);
    }

    #[test]
    fn device_auto_same_as_gpu() {
        let gpu = GpuCapabilities::new(true, false);
        let auto = resolve_backend("auto", "", ModelType::Embedding, &gpu).unwrap();
        let explicit = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(auto.backend, explicit.backend);
        assert_eq!(auto.device, explicit.device);
    }

    #[test]
    fn device_empty_same_as_auto() {
        let gpu = GpuCapabilities::new(true, false);
        let empty = resolve_backend("", "", ModelType::Embedding, &gpu).unwrap();
        let auto = resolve_backend("auto", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(empty.backend, auto.backend);
    }

    #[test]
    fn explicit_backend_overrides_device_field() {
        let gpu = GpuCapabilities::new(true, false);
        // Even though device says "cpu", explicit backend="cuda" takes priority
        let r = resolve_backend("cpu", "cuda", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
    }

    #[test]
    fn unknown_device() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("tpu", "", ModelType::Embedding, &gpu);
        assert!(r.is_err());
    }
}
```

-**Step 2: Add `mod backend;` to main.rs and run tests to see them fail**

Add `mod backend;` after the existing module declarations in `crates/corvia-inference/src/main.rs`.

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-inference backend::tests 2>&1 | tail -20`
Expected: FAIL — all tests panic with `todo!()`

-**Step 3: Implement resolve_backend and GpuCapabilities::probe**

Replace the `todo!()` bodies in `backend.rs`:

```rust
impl GpuCapabilities {
    /// Probe actual hardware. Call once at startup.
    pub fn probe() -> Self {
        let cuda_available = std::process::Command::new("nvidia-smi")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        let openvino_available = std::path::Path::new("/dev/dri").exists()
            && (std::path::Path::new("/usr/lib/x86_64-linux-gnu/libopenvino.so").exists()
                || std::env::var("INTEL_OPENVINO_DIR").is_ok());

        tracing::info!(cuda = cuda_available, openvino = openvino_available, "GPU capabilities probed");

        Self {
            cuda_available,
            openvino_available,
        }
    }
}

pub fn resolve_backend(
    device: &str,
    backend: &str,
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    // 1. Explicit backend override
    if !backend.is_empty() {
        return resolve_explicit_backend(backend, model_type, gpu);
    }

    // 2. Device-based resolution
    match device {
        "cpu" => Ok(ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        }),
        "gpu" | "auto" | "" => resolve_gpu_preferred(model_type, gpu),
        other => Err(format!("Unknown device: '{other}'. Expected 'auto', 'gpu', or 'cpu'.")),
    }
}

fn resolve_explicit_backend(
    backend: &str,
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    match backend {
        "cuda" => {
            if !gpu.cuda_available {
                return Err("CUDA requested but not available (nvidia-smi not found)".into());
            }
            Ok(ResolvedBackend {
                device: Device::Gpu,
                backend: BackendKind::Cuda,
                fallback_used: false,
            })
        }
        "openvino" => {
            if model_type == ModelType::Chat {
                return Err("OpenVINO does not support chat models (llama.cpp)".into());
            }
            if !gpu.openvino_available {
                return Err("OpenVINO requested but not available (no Intel GPU or libs)".into());
            }
            Ok(ResolvedBackend {
                device: Device::Gpu,
                backend: BackendKind::OpenVino,
                fallback_used: false,
            })
        }
        "cpu" => Ok(ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        }),
        other => Err(format!(
            "Unknown backend: '{other}'. Expected 'cuda', 'openvino', or 'cpu'."
        )),
    }
}

fn resolve_gpu_preferred(
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    // Prefer CUDA
    if gpu.cuda_available {
        return Ok(ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::Cuda,
            fallback_used: false,
        });
    }

    // Then OpenVINO (embedding only)
    if gpu.openvino_available && model_type == ModelType::Embedding {
        return Ok(ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::OpenVino,
            fallback_used: false,
        });
    }

    // Fallback to CPU
    Ok(ResolvedBackend {
        device: Device::Cpu,
        backend: BackendKind::Cpu,
        fallback_used: true,
    })
}
```

-**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-inference backend::tests -- --nocapture 2>&1 | tail -20`
Expected: all 14 tests pass

-**Step 5: Commit**

```bash
git add crates/corvia-inference/src/backend.rs crates/corvia-inference/src/main.rs
git commit -m "feat(inference): add backend resolution module

Introduces Device, BackendKind, ResolvedBackend types and resolve_backend()
for mapping (device, backend, model_type) to concrete GPU/CPU selection.
GpuCapabilities probes nvidia-smi and /dev/dri at startup.
14 unit tests covering all resolution paths."
```

---

## Chunk 2: Service Integration

### Task 3: Cargo.toml Changes

**Files:**
- Modify: `crates/corvia-inference/Cargo.toml`

-**Step 1: Add ort direct dependency with EP features and cuda feature gate**

Add after the existing `fastembed = "5"` line:

```toml
# Direct ort dep with EP features — Cargo unifies with fastembed's internal ort.
# CUDA and OpenVINO EPs are runtime-loaded (no toolkit needed at build time).
ort = { version = "=2.0.0-rc.11", features = ["openvino", "cuda"] }
```

Keep `llama-cpp-2 = "0.1"` as-is (CUDA is gated via `[features]` section below, not as a default feature).

Add a features section at the end of the file:

```toml
[features]
default = []
cuda = ["llama-cpp-2/cuda"]
```

Note: `llama-cpp-2/cuda` requires CUDA toolkit at build time, so it's opt-in. The ort CUDA/OpenVINO features are always available (runtime-loaded).

-**Step 2: Verify compilation**

Run: `cargo build -p corvia-inference 2>&1 | tail -10`
Expected: compiles (first build may be slow as ort recompiles with new features)

-**Step 3: Commit**

```bash
git add crates/corvia-inference/Cargo.toml
git commit -m "feat(inference): add ort EP features and cuda feature gate

Adds direct ort dependency pinned at =2.0.0-rc.11 with cuda+openvino features
(runtime-loaded, no toolkit at build). llama-cpp-2 cuda behind feature gate."
```

---

### Task 4: Embedding Service — EP Selection

**Files:**
- Modify: `crates/corvia-inference/src/embedding_service.rs`

-**Step 1: Write failing test for EP-aware load**

Add to embedding_service.rs, update the `LoadedModel` struct and `load_model` signature. First, add test:

```rust
// In #[cfg(test)] mod tests { ... } (create if not exists)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, GpuCapabilities, ModelType, ResolvedBackend, Device};

    #[test]
    fn test_build_execution_providers_cpu() {
        let backend = ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert!(eps.is_empty(), "CPU backend should produce no EPs (ort defaults to CPU)");
    }

    #[test]
    fn test_build_execution_providers_cuda() {
        let backend = ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::Cuda,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert_eq!(eps.len(), 1, "CUDA backend should produce exactly one EP");
    }

    #[test]
    fn test_build_execution_providers_openvino() {
        let backend = ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::OpenVino,
            fallback_used: false,
        };
        let eps = build_execution_providers(&backend);
        assert_eq!(eps.len(), 1, "OpenVINO backend should produce exactly one EP");
    }
}
```

-**Step 2: Run tests to see them fail**

Run: `cargo test -p corvia-inference embedding_service::tests 2>&1 | tail -10`
Expected: FAIL — `build_execution_providers` not found

-**Step 3: Implement EP builder and update load_model**

Add imports and helper function to `embedding_service.rs`:

```rust
use crate::backend::{BackendKind, ResolvedBackend};
use ort::ep::{ExecutionProviderDispatch, CUDA, OpenVINO};
```

Add the `build_execution_providers` function:

```rust
/// Build ONNX Runtime execution providers based on resolved backend.
fn build_execution_providers(backend: &ResolvedBackend) -> Vec<ExecutionProviderDispatch> {
    match backend.backend {
        BackendKind::Cuda => vec![CUDA::default().build()],
        BackendKind::OpenVino => vec![
            OpenVINO::default()
                .with_device_type("GPU")
                .build(),
        ],
        BackendKind::Cpu => vec![], // ort defaults to CPU EP
    }
}
```

Update `LoadedModel` to track backend:

```rust
struct LoadedModel {
    engine: fastembed::TextEmbedding,
    variant: fastembed::EmbeddingModel,
    backend: ResolvedBackend,
}
```

Change `load_model` signature to accept `ResolvedBackend`:

```rust
pub async fn load_model(&self, name: &str, backend: ResolvedBackend) -> Result<(), Status> {
    let model_enum = Self::resolve_model(name)?;
    let name_owned = name.to_string();
    tracing::info!(model = %name_owned, device = %backend.device, backend_kind = %backend.backend, "Loading embedding model...");

    let eps = build_execution_providers(&backend);
    let model_enum_for_spawn = model_enum.clone();
    let engine = tokio::task::spawn_blocking(move || {
        fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(model_enum_for_spawn)
                .with_show_download_progress(true)
                .with_execution_providers(eps),
        )
    })
    .await
    .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
    .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

    self.models
        .lock()
        .map_err(|e| Status::internal(format!("Lock poisoned: {e}")))?
        .insert(
            name_owned.clone(),
            LoadedModel {
                engine,
                variant: model_enum,
                backend,
            },
        );
    tracing::info!(model = %name_owned, "Embedding model loaded");
    Ok(())
}
```

Add a getter for the loaded model's backend:

```rust
/// Get the resolved backend for a loaded model.
pub fn get_backend(&self, name: &str) -> Option<ResolvedBackend> {
    self.models
        .lock()
        .ok()?
        .get(name)
        .map(|m| m.backend.clone())
}
```

-**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-inference embedding_service::tests 2>&1 | tail -10`
Expected: all 3 tests pass

-**Step 5: Commit**

```bash
git add crates/corvia-inference/src/embedding_service.rs
git commit -m "feat(inference): add EP selection to embedding service

build_execution_providers() maps ResolvedBackend to CUDA/OpenVINO/CPU EPs.
load_model() now accepts ResolvedBackend and passes EPs to fastembed.
LoadedModel tracks which backend was used for reporting."
```

---

### Task 5: Chat Service — GPU Layer Offloading

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

-**Step 1: Write failing test for GPU model params**

Add to existing `#[cfg(test)] mod tests` in chat_service.rs:

```rust
use crate::backend::{BackendKind, Device, ResolvedBackend};

#[test]
fn test_build_model_params_cpu() {
    let backend = ResolvedBackend {
        device: Device::Cpu,
        backend: BackendKind::Cpu,
        fallback_used: false,
    };
    let params = build_model_params(&backend);
    // CPU: default params (n_gpu_layers = 0)
    // We can't inspect n_gpu_layers directly, but we can verify it doesn't panic
    let _ = params;
}

#[test]
fn test_build_model_params_cuda() {
    let backend = ResolvedBackend {
        device: Device::Gpu,
        backend: BackendKind::Cuda,
        fallback_used: false,
    };
    let params = build_model_params(&backend);
    let _ = params;
}
```

-**Step 2: Run tests to see them fail**

Run: `cargo test -p corvia-inference chat_service::tests::test_build_model_params 2>&1 | tail -10`
Expected: FAIL — `build_model_params` not found

-**Step 3: Implement GPU layer offloading and update load_model**

Add import to `chat_service.rs`:

```rust
use crate::backend::{BackendKind, ResolvedBackend};
```

Add `build_model_params` function:

```rust
/// Configure llama.cpp model params based on resolved backend.
fn build_model_params(backend: &ResolvedBackend) -> LlamaModelParams {
    match backend.backend {
        BackendKind::Cuda => LlamaModelParams::default().with_n_gpu_layers(999),
        _ => LlamaModelParams::default(), // CPU: n_gpu_layers = 0
    }
}
```

Update `ChatModelEntry` to track backend:

```rust
struct ChatModelEntry {
    model: Arc<LlamaModel>,
    backend: ResolvedBackend,
}
```

Change `load_model` signature and implementation:

```rust
pub async fn load_model(&self, name: &str, backend: ResolvedBackend) -> Result<(), Status> {
    let resolved = resolve_model(name)?;
    let name_owned = name.to_string();
    tracing::info!(model = %name_owned, repo = %resolved.repo, file = %resolved.filename,
        device = %backend.device, backend_kind = %backend.backend, "Loading chat model...");

    let backend_clone = backend.clone();
    let model = tokio::task::spawn_blocking(move || -> Result<Arc<LlamaModel>, Status> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| Status::internal(format!("hf-hub API init failed: {e}")))?;
        let repo = api.model(resolved.repo.clone());
        let model_path: PathBuf = repo
            .get(&resolved.filename)
            .map_err(|e| Status::internal(format!("Model download failed: {e}")))?;

        tracing::info!(path = %model_path.display(), "GGUF file ready, loading into llama.cpp...");

        let model_params = build_model_params(&backend_clone);
        let model = LlamaModel::load_from_file(llama_backend(), &model_path, &model_params)
            .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

        Ok(Arc::new(model))
    })
    .await
    .map_err(|e| Status::internal(format!("spawn_blocking failed: {e}")))??;

    let mut models = self.models.write().await;
    models.insert(
        name.to_string(),
        ChatModelEntry { model, backend },
    );
    tracing::info!(model = %name, "Chat model loaded successfully");
    Ok(())
}
```

Rename the existing `backend()` function (which returns `&'static LlamaBackend`) to `llama_backend()` to avoid name collision with the `backend` module:

```rust
fn llama_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        let backend = LlamaBackend::init().expect("Failed to initialize llama.cpp backend");
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        backend
    })
}
```

Update ALL references to `backend()` → `llama_backend()`. There are exactly 3 call sites:

1. `ChatServiceImpl::new()` (line 82): `let _ = backend();` → `let _ = llama_backend();`
2. `generate_blocking()` (line 198): `.new_context(backend(), ctx_params)` → `.new_context(llama_backend(), ctx_params)`
3. `generate_streaming_blocking()` (line 305): `.new_context(backend(), ctx_params)` → `.new_context(llama_backend(), ctx_params)`

Verify no references remain: `grep -n 'backend()' crates/corvia-inference/src/chat_service.rs` should return zero matches (only `llama_backend()` hits).

Add a getter:

```rust
pub fn get_backend(&self, name: &str) -> Option<ResolvedBackend> {
    let models = self.models.try_read().ok()?;
    models.get(name).map(|e| e.backend.clone())
}
```

-**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-inference chat_service::tests 2>&1 | tail -10`
Expected: all tests pass (existing + 2 new)

-**Step 5: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(inference): add GPU layer offloading to chat service

build_model_params() configures llama.cpp GPU layers based on ResolvedBackend.
CUDA backend offloads all layers (n_gpu_layers=999), CPU uses default.
ChatModelEntry tracks backend for reporting."
```

---

### Task 6: Model Manager — Thread device/backend Through

**Files:**
- Modify: `crates/corvia-inference/src/model_manager.rs`

-**Step 1: Update ModelManagerService to hold GpuCapabilities and resolve backend**

```rust
use crate::backend::{self, GpuCapabilities, ModelType, ResolvedBackend};
```

Update constructor:

```rust
pub struct ModelManagerService {
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,
    embed_svc: EmbeddingServiceImpl,
    chat_svc: ChatServiceImpl,
    gpu: GpuCapabilities,
}

impl ModelManagerService {
    pub fn new(embed_svc: EmbeddingServiceImpl, chat_svc: ChatServiceImpl, gpu: GpuCapabilities) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            embed_svc,
            chat_svc,
            gpu,
        }
    }
}
```

-**Step 2: Update load_model to resolve backend and return actual_device/actual_backend**

```rust
async fn load_model(
    &self,
    req: Request<LoadModelRequest>,
) -> Result<Response<LoadModelResponse>, Status> {
    let req = req.into_inner();
    tracing::info!(model = %req.name, model_type = %req.model_type,
        device = %req.device, backend = %req.backend, "load_model requested");

    let model_type = match req.model_type.as_str() {
        "embedding" => ModelType::Embedding,
        "chat" => ModelType::Chat,
        other => {
            return Ok(Response::new(LoadModelResponse {
                success: false,
                error: format!("Unknown model_type: '{other}'. Expected 'embedding' or 'chat'."),
                actual_device: String::new(),
                actual_backend: String::new(),
            }));
        }
    };

    // Resolve backend
    let resolved = match backend::resolve_backend(&req.device, &req.backend, model_type, &self.gpu) {
        Ok(r) => r,
        Err(e) => {
            return Ok(Response::new(LoadModelResponse {
                success: false,
                error: e,
                actual_device: String::new(),
                actual_backend: String::new(),
            }));
        }
    };

    let actual_device = resolved.device.to_string();
    let actual_backend = resolved.backend.to_string();

    if resolved.fallback_used {
        tracing::warn!(
            model = %req.name,
            requested_device = %req.device,
            actual_device = %actual_device,
            actual_backend = %actual_backend,
            "GPU not available, fell back to CPU"
        );
    }

    // Delegate to appropriate service
    let result = match model_type {
        ModelType::Embedding => self.embed_svc.load_model(&req.name, resolved).await,
        ModelType::Chat => self.chat_svc.load_model(&req.name, resolved).await,
    };

    match result {
        Ok(()) => {
            let mut models = self.models.write().await;
            models.insert(
                req.name.clone(),
                ModelEntry {
                    name: req.name,
                    model_type: req.model_type,
                    loaded: true,
                },
            );
            Ok(Response::new(LoadModelResponse {
                success: true,
                error: String::new(),
                actual_device,
                actual_backend,
            }))
        }
        Err(status) => Ok(Response::new(LoadModelResponse {
            success: false,
            error: status.message().to_string(),
            actual_device,
            actual_backend,
        })),
    }
}
```

-**Step 3: Update main.rs to probe GPU and pass to model manager**

In `main.rs`, update the `Commands::Serve` handler:

```rust
Commands::Serve { port } => {
    let addr = format!("0.0.0.0:{port}").parse()?;

    // Probe GPU capabilities once at startup
    let gpu = backend::GpuCapabilities::probe();
    tracing::info!(
        cuda = gpu.cuda_available,
        openvino = gpu.openvino_available,
        "GPU capabilities"
    );

    let embed_svc = embedding_service::EmbeddingServiceImpl::new();
    let chat_svc = chat_service::ChatServiceImpl::new();
    let model_mgr = model_manager::ModelManagerService::new(
        embed_svc.clone(),
        chat_svc.clone(),
        gpu,
    );

    tracing::info!(port, "inference_server_starting");

    Server::builder()
        .add_service(ModelManagerServer::new(model_mgr))
        .add_service(EmbeddingServiceServer::with_interceptor(embed_svc, accept_trace))
        .add_service(ChatServiceServer::with_interceptor(chat_svc, accept_trace))
        .serve(addr)
        .await?;
}
```

-**Step 4: Verify full build**

Run: `cargo build -p corvia-inference 2>&1 | tail -10`
Expected: compiles cleanly

-**Step 5: Run all tests**

Run: `cargo test -p corvia-inference 2>&1 | tail -20`
Expected: all tests pass (backend::tests + chat_service::tests + embedding_service::tests)

-**Step 6: Commit**

```bash
git add crates/corvia-inference/src/model_manager.rs crates/corvia-inference/src/main.rs
git commit -m "feat(inference): wire GPU device control through model manager

ModelManagerService probes GPU at startup via GpuCapabilities.
load_model resolves device/backend from request, passes ResolvedBackend
to embedding/chat services, and returns actual_device/actual_backend
in the response (fallback-with-signal pattern)."
```

---

## Chunk 3: Infrastructure

### Task 7: Devcontainer + Dockerfile

**Files:**
- Modify: `.devcontainer/devcontainer.json`
- Modify: `.devcontainer/Dockerfile`

-**Step 1: Add Intel GPU opt-in comments to devcontainer.json**

After the existing GPU passthrough comment, add Intel GPU documentation:

```jsonc
// GPU passthrough: uncomment the next line if you have NVIDIA GPU + nvidia-container-toolkit.
// "runArgs": ["--gpus", "all"],
//
// Intel GPU passthrough (iGPU): uncomment the lines below if Intel GPU is available.
// Required for OpenVINO execution provider on Intel integrated/discrete GPUs.
// "runArgs": [
//     "--device=/dev/dri",
//     "--device=/dev/dxg",
//     "--group-add", "video",
//     "--group-add", "render"
// ],
```

Also add the WSL GPU driver mount as a comment in the mounts section:

```jsonc
// Uncomment for WSL2 GPU driver passthrough (Intel + NVIDIA):
// "source=/usr/lib/wsl,target=/usr/lib/wsl,type=bind,readonly",
```

-**Step 2: Add Intel GPU runtime packages to Dockerfile (opt-in section)**

Add after the system dependencies `RUN` block:

```dockerfile
# Intel GPU runtime (OpenVINO EP for ONNX Runtime).
# Uncomment if Intel iGPU/dGPU is available and passed through via devcontainer.json.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     intel-opencl-icd intel-level-zero-gpu level-zero \
#     && rm -rf /var/lib/apt/lists/*
```

-**Step 3: Commit**

```bash
git add .devcontainer/devcontainer.json .devcontainer/Dockerfile
git commit -m "docs(devcontainer): add Intel GPU passthrough opt-in

Documented Intel GPU device passthrough for OpenVINO EP support.
Includes runArgs, WSL driver mount, and Intel GPU runtime packages.
All commented out by default — uncomment if Intel GPU is available."
```

---

### Task 8: Full Workspace Verification

-**Step 1: Run full workspace tests**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: all tests pass (existing 433+ tests + new backend/EP tests)

-**Step 2: Run the inference server manually to verify startup**

Run: `cargo run -p corvia-inference -- serve --port 8031 &`
Expected: see log line with `cuda = false, openvino = false, "GPU capabilities"` (in devcontainer without GPU passthrough) and `inference_server_starting` with port 8031.

Then kill: `kill %1`

-**Step 3: Final commit (if any fixups needed)**

If any fixups were made during verification, commit them.

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `crates/corvia-proto/proto/corvia/inference/v1/model.proto` | Add device/backend to request, actual_device/actual_backend to response |
| `crates/corvia-inference/src/backend.rs` | **New:** ResolvedBackend types, GPU probing, resolve_backend() with 13 tests |
| `crates/corvia-inference/src/embedding_service.rs` | EP selection via fastembed's with_execution_providers(), backend tracking |
| `crates/corvia-inference/src/chat_service.rs` | GPU layer offloading via build_model_params(), backend tracking |
| `crates/corvia-inference/src/model_manager.rs` | GpuCapabilities, resolve backend from request, return actual_* in response |
| `crates/corvia-inference/src/main.rs` | Add `mod backend`, probe GPU at startup, pass to model manager |
| `crates/corvia-inference/Cargo.toml` | Add ort direct dep with cuda+openvino, cuda feature gate for llama-cpp-2 |
| `.devcontainer/devcontainer.json` | Intel GPU passthrough documentation (opt-in) |
| `.devcontainer/Dockerfile` | Intel GPU runtime packages (opt-in, commented) |
