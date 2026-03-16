# Inference Config & KV Cache Quantization Implementation Plan

> **Status:** Shipped (v0.4.2)

**Goal:** Add KV cache quantization, flash attention, and a unified `[inference]` config section with config-is-truth reload semantics.

**Architecture:** New `InferenceConfig` struct in `corvia-common` with `device`, `backend`, `kv_quant`, `flash_attention` fields. Proto gets new fields on existing messages. Chat service applies KV quant/flash attention at context creation time. CLI writes config then triggers gRPC reload (`--no-persist` for ephemeral).

**Tech Stack:** Rust, tonic/prost (gRPC), llama-cpp-2 (llama.cpp bindings), ONNX Runtime (via fastembed), TOML config.

**Spec:** `2026-03-12-inference-kv-cache-config-design.md`

---

## Chunk 1: Proto + Config Foundation

### Task 1: Proto changes — add kv_quant and flash_attention fields

**Files:**
- Modify: `crates/corvia-proto/proto/corvia/inference/v1/model.proto`

-**Step 1: Add fields to LoadModelRequest**

In `crates/corvia-proto/proto/corvia/inference/v1/model.proto`, add two fields to `LoadModelRequest` (after `backend = 4`):

```protobuf
message LoadModelRequest {
  string name = 1;
  string model_type = 2;        // "embedding" | "chat"
  string device = 3;            // "auto" | "gpu" | "cpu" (default: "auto")
  string backend = 4;           // optional override: "cuda", "openvino", "vulkan", ""
  string kv_quant = 5;          // "q8", "q4", "none", "" (use default)
  bool flash_attention = 6;     // enable flash attention
}
```

-**Step 2: Add fields to ReloadModelsRequest**

Add to `ReloadModelsRequest` (after `name = 4`):

```protobuf
message ReloadModelsRequest {
  string device = 1;
  string backend = 2;
  bool reprobe_gpu = 3;
  string name = 4;
  string kv_quant = 5;          // "q8", "q4", "none", ""
  bool flash_attention = 6;     // enable flash attention
}
```

-**Step 3: Add fields to ModelStatus**

Add to `ModelStatus` (after `backend = 6`):

```protobuf
message ModelStatus {
  string name = 1;
  string model_type = 2;
  bool loaded = 3;
  uint64 memory_bytes = 4;
  string device = 5;
  string backend = 6;
  string kv_quant = 7;          // current KV quant setting
  bool flash_attention = 8;     // current flash attention setting
}
```

-**Step 4: Build proto to verify**

Run: `cd repos/corvia && cargo build -p corvia-proto`
Expected: compiles successfully, generated Rust code includes new fields.

-**Step 5: Commit**

```bash
git add crates/corvia-proto/proto/corvia/inference/v1/model.proto
git commit -m "proto: add kv_quant and flash_attention fields to inference messages"
```

### Task 2: InferenceConfig struct in corvia-common

**Files:**
- Modify: `crates/corvia-common/src/config.rs`

-**Step 1: Write test for InferenceConfig defaults**

Add to the `#[cfg(test)] mod tests` block at the bottom of `crates/corvia-common/src/config.rs`:

```rust
    #[test]
    fn test_inference_config_defaults() {
        let config = InferenceConfig::default();
        assert_eq!(config.device, "auto");
        assert!(config.backend.is_empty());
        assert_eq!(config.kv_quant, "q8");
        assert!(config.flash_attention);
    }
```

-**Step 2: Run test to verify it fails**

Run: `cd repos/corvia && cargo test -p corvia-common test_inference_config_defaults`
Expected: FAIL — `InferenceConfig` not found.

-**Step 3: Add InferenceConfig struct and default functions**

In `crates/corvia-common/src/config.rs`, after the `default_device()` function (line 103), add:

```rust
fn default_kv_quant() -> String { "q8".into() }
fn default_flash_attention() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Device preference: "auto" (default), "gpu", or "cpu".
    #[serde(default = "default_device")]
    pub device: String,
    /// Backend override: "cuda", "openvino", or "" (auto-select).
    #[serde(default)]
    pub backend: String,
    /// KV cache quantization: "q8" (default), "q4", "none".
    #[serde(default = "default_kv_quant")]
    pub kv_quant: String,
    /// Enable flash attention (default: true).
    #[serde(default = "default_flash_attention")]
    pub flash_attention: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            backend: String::new(),
            kv_quant: default_kv_quant(),
            flash_attention: default_flash_attention(),
        }
    }
}
```

-**Step 4: Run test to verify it passes**

Run: `cd repos/corvia && cargo test -p corvia-common test_inference_config_defaults`
Expected: PASS

-**Step 5: Write test for InferenceConfig in CorviaConfig**

Add to tests:

```rust
    #[test]
    fn test_corvia_config_has_inference_defaults() {
        let config = CorviaConfig::default();
        assert_eq!(config.inference.device, "auto");
        assert_eq!(config.inference.kv_quant, "q8");
        assert!(config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_from_toml() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
data_dir = ".corvia"

[embedding]
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020

[inference]
device = "gpu"
backend = "cuda"
kv_quant = "q4"
flash_attention = false
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.inference.device, "gpu");
        assert_eq!(config.inference.backend, "cuda");
        assert_eq!(config.inference.kv_quant, "q4");
        assert!(!config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_omitted_still_parses() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
data_dir = ".corvia"

[embedding]
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.inference.device, "auto");
        assert_eq!(config.inference.kv_quant, "q8");
        assert!(config.inference.flash_attention);
    }

    #[test]
    fn test_inference_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.inference.kv_quant = "q4".into();
        config.inference.flash_attention = false;
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.inference.kv_quant, "q4");
        assert!(!loaded.inference.flash_attention);
    }
```

-**Step 6: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-common test_corvia_config_has_inference`
Expected: FAIL — `CorviaConfig` has no field `inference`.

-**Step 7: Add inference field to CorviaConfig**

In `crates/corvia-common/src/config.rs`, add to the `CorviaConfig` struct (after the `telemetry` field at line 72):

```rust
    #[serde(default)]
    pub inference: InferenceConfig,
```

Update `Default for CorviaConfig` (around line 376) to include:
```rust
            inference: InferenceConfig::default(),
```

-**Step 8: Run all new tests**

Run: `cd repos/corvia && cargo test -p corvia-common test_inference_config`
Expected: all 4 new tests PASS.

-**Step 9: Remove device and backend from EmbeddingConfig**

In `crates/corvia-common/src/config.rs`:

1. Remove these two fields from `EmbeddingConfig` (lines 112-117):
   ```rust
       /// Device preference: ...
       #[serde(default = "default_device")]
       pub device: String,
       /// Backend override: ...
       #[serde(default)]
       pub backend: String,
   ```

2. Remove `device` and `backend` from `Default for CorviaConfig` embedding block (lines 362-364):
   ```rust
                   device: "auto".into(),
                   backend: String::new(),
   ```

3. Remove `device` and `backend` from `full_default()` embedding block (lines 406-408).

4. Remove `device` and `backend` from `postgres_default()` embedding block (lines 430-432).

-**Step 10: Fix compile errors in callers**

After removing `device`/`backend` from `EmbeddingConfig`, callers that referenced `config.embedding.device` and `config.embedding.backend` must be updated to `config.inference.device` and `config.inference.backend`. Find them:

Run: `cd repos/corvia && cargo build --workspace 2>&1 | grep "no field"`

Expected locations:
- `crates/corvia-cli/src/main.rs` — lines 416-417, 1678-1679 (in `ensure_ready` calls)

Update each from `config.embedding.device` → `config.inference.device` and `config.embedding.backend` → `config.inference.backend`.

-**Step 11: Run workspace build**

Run: `cd repos/corvia && cargo build --workspace`
Expected: compiles successfully.

-**Step 12: Run all tests**

Run: `cd repos/corvia && cargo test -p corvia-common`
Expected: all tests PASS (some existing tests that hardcoded `device`/`backend` in TOML under `[embedding]` will still parse due to TOML ignoring unknown fields).

-**Step 13: Commit**

```bash
git add crates/corvia-common/src/config.rs crates/corvia-cli/src/main.rs
git commit -m "feat(config): add [inference] section, remove device/backend from [embedding]"
```

### Task 3: Telemetry span constants

**Files:**
- Modify: `crates/corvia-telemetry/src/lib.rs`

-**Step 1: Add span constants**

In `crates/corvia-telemetry/src/lib.rs`, add to the `spans` module (after `RAG_ASK` at line 26):

```rust
    pub const INFERENCE_LOAD: &str = "corvia.inference.load";
    pub const INFERENCE_RELOAD: &str = "corvia.inference.reload";
    pub const INFERENCE_CONFIG_RELOAD: &str = "corvia.inference.config_reload";
```

-**Step 2: Update the span constants test**

In `test_span_constants_are_dotted`, add to the `all` array (after `spans::RAG_ASK` at line 162):

```rust
            spans::INFERENCE_LOAD,
            spans::INFERENCE_RELOAD,
            spans::INFERENCE_CONFIG_RELOAD,
```

-**Step 3: Run test**

Run: `cd repos/corvia && cargo test -p corvia-telemetry test_span_constants_are_dotted`
Expected: PASS

-**Step 4: Commit**

```bash
git add crates/corvia-telemetry/src/lib.rs
git commit -m "feat(telemetry): add inference span constants"
```

### Task 4: Hot-reloadable config — add inference section

**Files:**
- Modify: `crates/corvia-kernel/src/ops.rs`

-**Step 1: Add "inference" to HOT_RELOADABLE_SECTIONS**

In `crates/corvia-kernel/src/ops.rs`, line 127, change:

```rust
const HOT_RELOADABLE_SECTIONS: &[&str] = &["agent_lifecycle", "merge", "rag", "chunking", "reasoning", "adapters"];
```

to:

```rust
const HOT_RELOADABLE_SECTIONS: &[&str] = &["agent_lifecycle", "merge", "rag", "chunking", "reasoning", "adapters", "inference"];
```

-**Step 2: Build to verify**

Run: `cd repos/corvia && cargo build -p corvia-kernel`
Expected: compiles.

-**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/ops.rs
git commit -m "feat(kernel): add inference to hot-reloadable config sections"
```

---

## Chunk 2: Inference Server — KV Cache + Flash Attention

### Task 5: resolve_kv_quant in backend.rs

**Files:**
- Modify: `crates/corvia-inference/src/backend.rs`

-**Step 1: Write tests for resolve_kv_quant**

Add to `#[cfg(test)] mod tests` at the bottom of `crates/corvia-inference/src/backend.rs`:

```rust
    // --- KV quant resolution ---

    #[test]
    fn kv_quant_q8() {
        let r = resolve_kv_quant("q8").unwrap();
        assert_eq!(r, KvCacheType::Q8_0);
    }

    #[test]
    fn kv_quant_q8_0() {
        let r = resolve_kv_quant("q8_0").unwrap();
        assert_eq!(r, KvCacheType::Q8_0);
    }

    #[test]
    fn kv_quant_q4() {
        let r = resolve_kv_quant("q4").unwrap();
        assert_eq!(r, KvCacheType::Q4_0);
    }

    #[test]
    fn kv_quant_q4_0() {
        let r = resolve_kv_quant("q4_0").unwrap();
        assert_eq!(r, KvCacheType::Q4_0);
    }

    #[test]
    fn kv_quant_none() {
        let r = resolve_kv_quant("none").unwrap();
        assert_eq!(r, KvCacheType::F16);
    }

    #[test]
    fn kv_quant_f16() {
        let r = resolve_kv_quant("f16").unwrap();
        assert_eq!(r, KvCacheType::F16);
    }

    #[test]
    fn kv_quant_empty_defaults_to_f16() {
        let r = resolve_kv_quant("").unwrap();
        assert_eq!(r, KvCacheType::F16);
    }

    #[test]
    fn kv_quant_case_insensitive() {
        let r = resolve_kv_quant("Q8").unwrap();
        assert_eq!(r, KvCacheType::Q8_0);
    }

    #[test]
    fn kv_quant_unknown() {
        let r = resolve_kv_quant("q2");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("Unknown kv_quant"));
    }
```

-**Step 2: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-inference kv_quant`
Expected: FAIL — `resolve_kv_quant` not found, `KvCacheType` not imported.

-**Step 3: Implement resolve_kv_quant**

In `crates/corvia-inference/src/backend.rs`, add the import at the top of the file (after line 4):

```rust
use llama_cpp_2::context::params::KvCacheType;
```

Then add this function after `resolve_gpu_preferred` (before the `impl Display` blocks, around line 161):

```rust
/// Resolve a KV cache quantization string to its llama-cpp-2 type.
///
/// Accepts: "q8"/"q8_0" → Q8_0, "q4"/"q4_0" → Q4_0, "none"/"f16"/"" → F16.
pub fn resolve_kv_quant(raw: &str) -> Result<KvCacheType, String> {
    match raw.to_lowercase().as_str() {
        "q8" | "q8_0" => Ok(KvCacheType::Q8_0),
        "q4" | "q4_0" => Ok(KvCacheType::Q4_0),
        "none" | "f16" | "" => Ok(KvCacheType::F16),
        other => Err(format!("Unknown kv_quant: '{other}'. Expected 'q8', 'q4', or 'none'.")),
    }
}
```

Also add `KvCacheType` to the test module's imports:

```rust
    use llama_cpp_2::context::params::KvCacheType;
```

-**Step 4: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-inference kv_quant`
Expected: all 9 tests PASS.

-**Step 5: Commit**

```bash
git add crates/corvia-inference/src/backend.rs
git commit -m "feat(inference): add resolve_kv_quant for KV cache type resolution"
```

### Task 6: ChatModelEntry + load_model with kv_quant/flash_attention

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

-**Step 1: Update ChatModelEntry struct**

In `crates/corvia-inference/src/chat_service.rs`, replace the `ChatModelEntry` struct (lines 75-78):

```rust
struct ChatModelEntry {
    model: Arc<LlamaModel>,
    backend: ResolvedBackend,
}
```

with:

```rust
struct ChatModelEntry {
    model: Arc<LlamaModel>,
    backend: ResolvedBackend,
    kv_cache_type: KvCacheType,
    flash_attention: bool,
}
```

-**Step 2: Add KvCacheType and flash attention imports**

Add to the imports at the top of the file (after line 11):

```rust
use llama_cpp_2::context::params::KvCacheType;
```

Also add `llama-cpp-sys-2` as a dependency in `crates/corvia-inference/Cargo.toml` (after `llama-cpp-2 = "0.1"` on line 29):

```toml
llama-cpp-sys-2 = "0.1"
```

Then add this import to `chat_service.rs`:

```rust
use llama_cpp_sys_2::{LLAMA_FLASH_ATTN_TYPE_ENABLED, LLAMA_FLASH_ATTN_TYPE_DISABLED};
```

-**Step 3: Update load_model signature and implementation**

Change the `load_model` method signature (line 99) from:

```rust
    pub async fn load_model(&self, name: &str, backend: ResolvedBackend) -> Result<(), Status> {
```

to:

```rust
    pub async fn load_model(&self, name: &str, backend: ResolvedBackend, kv_cache_type: KvCacheType, flash_attention: bool) -> Result<(), Status> {
```

Update the `models.insert` call (lines 128-131) to include the new fields:

```rust
        let mut models = self.models.write().await;
        models.insert(
            name.to_string(),
            ChatModelEntry { model, backend, kv_cache_type, flash_attention },
        );
```

-**Step 4: Add get_kv_settings method**

Add after the `get_backend` method (after line 140):

```rust
    /// Get the KV cache settings for a loaded model.
    pub async fn get_kv_settings(&self, name: &str) -> Option<(KvCacheType, bool)> {
        let models = self.models.read().await;
        models.get(name).map(|e| (e.kv_cache_type, e.flash_attention))
    }
```

-**Step 5: Update get_model to also return kv settings**

Replace `get_model` (lines 143-149):

```rust
    async fn get_model(&self, name: &str) -> Result<Arc<LlamaModel>, Status> {
        let models = self.models.read().await;
        models
            .get(name)
            .map(|e| Arc::clone(&e.model))
            .ok_or_else(|| Status::not_found(format!("Chat model '{}' not loaded", name)))
    }
```

with:

```rust
    /// Get a loaded model with its KV cache settings.
    async fn get_model_with_settings(&self, name: &str) -> Result<(Arc<LlamaModel>, KvCacheType, bool), Status> {
        let models = self.models.read().await;
        models
            .get(name)
            .map(|e| (Arc::clone(&e.model), e.kv_cache_type, e.flash_attention))
            .ok_or_else(|| Status::not_found(format!("Chat model '{}' not loaded", name)))
    }
```

-**Step 6: Update GenerateParams to include KV settings**

Replace `GenerateParams` (lines 157-162):

```rust
struct GenerateParams {
    model: Arc<LlamaModel>,
    prompt: String,
    temperature: f32,
    max_tokens: u32,
    kv_cache_type: KvCacheType,
    flash_attention: bool,
}
```

-**Step 7: Add build_context_params helper**

Add after `build_prompt` (after line 188):

```rust
/// Build LlamaContextParams with KV cache quantization and flash attention.
fn build_context_params(ctx_size: u32, kv_cache_type: KvCacheType, flash_attention: bool) -> LlamaContextParams {
    let flash_policy = if flash_attention && kv_cache_type == KvCacheType::Q4_0 {
        tracing::warn!("Flash attention not compatible with Q4 KV cache, disabling");
        LLAMA_FLASH_ATTN_TYPE_DISABLED
    } else if flash_attention {
        LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        LLAMA_FLASH_ATTN_TYPE_DISABLED
    };

    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size.max(512)))
        .with_n_batch(512)
        .with_type_k(kv_cache_type)
        .with_type_v(kv_cache_type)
        .with_flash_attention_policy(flash_policy)
}
```

Note: `with_flash_attention_policy` takes `llama_flash_attn_type` (a `c_int` from `llama-cpp-sys-2`), not a `bool`. The constants `LLAMA_FLASH_ATTN_TYPE_ENABLED` and `LLAMA_FLASH_ATTN_TYPE_DISABLED` are imported from `llama_cpp_sys_2` (added in Step 2).

-**Step 8: Update generate_blocking to use build_context_params**

In `generate_blocking` (around lines 210-213), replace:

```rust
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size.max(512)))
        .with_n_batch(512);
```

with:

```rust
    let ctx_params = build_context_params(ctx_size, params.kv_cache_type, params.flash_attention);
```

-**Step 9: Update generate_streaming_blocking to use build_context_params**

In `generate_streaming_blocking` (around lines 317-320), replace:

```rust
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size.max(512)))
        .with_n_batch(512);
```

with:

```rust
    let ctx_params = build_context_params(ctx_size, params.kv_cache_type, params.flash_attention);
```

-**Step 10: Update chat() to pass KV settings**

In the `chat` method (around lines 429-442), replace:

```rust
        let model = self.get_model(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let params = GenerateParams {
            model,
            prompt,
            temperature,
            max_tokens,
        };
```

with:

```rust
        let (model, kv_cache_type, flash_attention) = self.get_model_with_settings(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let params = GenerateParams {
            model,
            prompt,
            temperature,
            max_tokens,
            kv_cache_type,
            flash_attention,
        };
```

-**Step 11: Update chat_stream() to pass KV settings**

In the `chat_stream` method (around lines 464-479), replace:

```rust
        let model = self.get_model(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let params = GenerateParams {
                model,
                prompt,
                temperature,
                max_tokens,
            };
```

with:

```rust
        let (model, kv_cache_type, flash_attention) = self.get_model_with_settings(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let params = GenerateParams {
                model,
                prompt,
                temperature,
                max_tokens,
                kv_cache_type,
                flash_attention,
            };
```

-**Step 12: Build to check compile**

Run: `cd repos/corvia && cargo build -p corvia-inference 2>&1 | head -30`
Expected: compile errors from `model_manager.rs` callers (they pass wrong number of args to `load_model`). We'll fix those in the next task.

-**Step 13: Commit (partial — chat_service done)**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(inference): wire KV cache quantization and flash attention into chat service"
```

### Task 7: ModelManager — pass kv_quant/flash_attention through gRPC

**Files:**
- Modify: `crates/corvia-inference/src/model_manager.rs`

-**Step 1: Add kv_quant and flash_attention to ModelEntry**

In `crates/corvia-inference/src/model_manager.rs`, replace the `ModelEntry` struct (lines 12-18):

```rust
#[derive(Clone)]
pub struct ModelEntry {
    pub name: String,
    pub model_type: String,
    pub loaded: bool,
    pub device: String,
    pub backend: String,
    pub kv_quant: String,
    pub flash_attention: bool,
}
```

-**Step 2: Add resolve_kv_quant import**

Add to the imports at line 1:

```rust
use crate::backend::{self, GpuCapabilities, ModelType, resolve_kv_quant};
use llama_cpp_2::context::params::KvCacheType;
```

(Replace the existing `use crate::backend::{self, GpuCapabilities, ModelType};` line.)

-**Step 3: Update list_models to include new fields**

In `list_models` (around lines 62-68), update the `ModelStatus` construction:

```rust
            .map(|m| ModelStatus {
                name: m.name.clone(),
                model_type: m.model_type.clone(),
                loaded: m.loaded,
                memory_bytes: 0,
                device: m.device.clone(),
                backend: m.backend.clone(),
                kv_quant: m.kv_quant.clone(),
                flash_attention: m.flash_attention,
            })
```

-**Step 4: Update load_model to resolve and pass kv_quant**

In `load_model`, after the backend resolution block (after line 109), add:

```rust
        // Resolve KV quant
        let kv_cache_type = match resolve_kv_quant(&req.kv_quant) {
            Ok(t) => t,
            Err(e) => {
                return Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: e,
                    actual_device: String::new(),
                    actual_backend: String::new(),
                }));
            }
        };

        // Log if KV quant set on embedding model (ignored by ONNX)
        if model_type == ModelType::Embedding && kv_cache_type != KvCacheType::F16 {
            tracing::debug!(model = %req.name, kv_quant = %req.kv_quant, "KV quant ignored for embedding model");
        }
```

Update the delegate call (around lines 125-128) to pass KV settings to chat models:

```rust
        let result = match model_type {
            ModelType::Embedding => self.embed_svc.load_model(&req.name, resolved).await,
            ModelType::Chat => self.chat_svc.load_model(&req.name, resolved, kv_cache_type, req.flash_attention).await,
        };
```

Update the `ModelEntry` insertion (around lines 135-141) to include new fields:

```rust
                models.insert(
                    req.name.clone(),
                    ModelEntry {
                        name: req.name,
                        model_type: req.model_type,
                        loaded: true,
                        device: actual_device.clone(),
                        backend: actual_backend.clone(),
                        kv_quant: req.kv_quant,
                        flash_attention: req.flash_attention,
                    },
                );
```

-**Step 5: Update reload_models to pass kv_quant**

In `reload_models`, after the backend resolution block (around line 240), add KV quant resolution:

```rust
            let kv_cache_type = match resolve_kv_quant(&req.kv_quant) {
                Ok(t) => t,
                Err(e) => {
                    results.push(ModelReloadResult {
                        name: entry.name.clone(),
                        model_type: entry.model_type.clone(),
                        success: false,
                        error: e,
                        actual_device: String::new(),
                        actual_backend: String::new(),
                    });
                    all_success = false;
                    continue;
                }
            };
```

Update the load delegate (around lines 247-250):

```rust
            let load_result = match model_type {
                ModelType::Embedding => self.embed_svc.load_model(&entry.name, resolved).await,
                ModelType::Chat => self.chat_svc.load_model(&entry.name, resolved, kv_cache_type, req.flash_attention).await,
            };
```

Update the `ModelEntry` insertion in the success branch (around lines 257-263):

```rust
                    models.insert(
                        entry.name.clone(),
                        ModelEntry {
                            name: entry.name.clone(),
                            model_type: entry.model_type.clone(),
                            loaded: true,
                            device: actual_device.clone(),
                            backend: actual_backend.clone(),
                            kv_quant: req.kv_quant.clone(),
                            flash_attention: req.flash_attention,
                        },
                    );
```

-**Step 6: Build the inference crate**

Run: `cd repos/corvia && cargo build -p corvia-inference`
Expected: compiles successfully.

-**Step 7: Run inference tests**

Run: `cd repos/corvia && cargo test -p corvia-inference`
Expected: all tests PASS.

-**Step 8: Commit**

```bash
git add crates/corvia-inference/src/model_manager.rs
git commit -m "feat(inference): wire kv_quant and flash_attention through model manager"
```

---

## Chunk 3: Provisioner + CLI

### Task 8: InferenceProvisioner — pass kv_quant/flash_attention

**Files:**
- Modify: `crates/corvia-kernel/src/inference_provisioner.rs`

-**Step 1: Update load_models signature**

Change `load_models` (line 83) from:

```rust
    pub async fn load_models(
        &self,
        embed_model: &str,
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
    ) -> Result<()> {
```

to:

```rust
    pub async fn load_models(
        &self,
        embed_model: &str,
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
        kv_quant: &str,
        flash_attention: bool,
    ) -> Result<()> {
```

-**Step 2: Update LoadModelRequest construction in load_models**

For the embedding model load (around lines 96-101), add the new fields:

```rust
            .load_model(tonic::Request::new(LoadModelRequest {
                name: embed_model.to_string(),
                model_type: "embedding".to_string(),
                device: device.to_string(),
                backend: backend.to_string(),
                kv_quant: kv_quant.to_string(),
                flash_attention,
            }))
```

For the chat model load (around lines 121-126), add the new fields:

```rust
                .load_model(tonic::Request::new(LoadModelRequest {
                    name: chat_model.to_string(),
                    model_type: "chat".to_string(),
                    device: device.to_string(),
                    backend: backend.to_string(),
                    kv_quant: kv_quant.to_string(),
                    flash_attention,
                }))
```

-**Step 3: Update reload_models signature**

Change `reload_models` (line 149) from:

```rust
    pub async fn reload_models(&self, device: &str, backend: &str, model_name: Option<&str>) -> Result<()> {
```

to:

```rust
    pub async fn reload_models(&self, device: &str, backend: &str, kv_quant: &str, flash_attention: bool, model_name: Option<&str>) -> Result<()> {
```

Update the `ReloadModelsRequest` construction (around lines 155-160):

```rust
        let resp = client
            .reload_models(tonic::Request::new(ReloadModelsRequest {
                device: device.to_string(),
                backend: backend.to_string(),
                reprobe_gpu: true,
                name: model_name.unwrap_or_default().to_string(),
                kv_quant: kv_quant.to_string(),
                flash_attention,
            }))
```

-**Step 4: Update ensure_ready signature**

Change `ensure_ready` (line 194) from:

```rust
    pub async fn ensure_ready(
        &self,
        embed_model: &str,
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
    ) -> Result<()> {
```

to:

```rust
    pub async fn ensure_ready(
        &self,
        embed_model: &str,
        chat_model: Option<&str>,
        device: &str,
        backend: &str,
        kv_quant: &str,
        flash_attention: bool,
    ) -> Result<()> {
```

Update the `load_models` call (line 211):

```rust
        self.load_models(embed_model, chat_model, device, backend, kv_quant, flash_attention).await?;
```

-**Step 5: Build to find callers**

Run: `cd repos/corvia && cargo build --workspace 2>&1 | grep "error" | head -20`
Expected: compile errors in CLI where `ensure_ready` and `reload_models` are called with wrong number of args.

-**Step 6: Fix CLI callers of ensure_ready**

In `crates/corvia-cli/src/main.rs`, update both `ensure_ready` call sites.

First call site (around line 413):
```rust
                    provisioner.ensure_ready(
                        &config.embedding.model,
                        chat_model,
                        &config.inference.device,
                        &config.inference.backend,
                        &config.inference.kv_quant,
                        config.inference.flash_attention,
                    ).await?;
```

Second call site (around line 1675):
```rust
            provisioner.ensure_ready(
                &config.embedding.model,
                chat_model,
                &config.inference.device,
                &config.inference.backend,
                &config.inference.kv_quant,
                config.inference.flash_attention,
            ).await?;
```

-**Step 7: Fix CLI caller of reload_models**

In `cmd_inference_reload` (around line 1607), update the call. This will be fully reworked in Task 9, but for now make it compile:

```rust
    provisioner.reload_models(device, backend, &config.inference.kv_quant, config.inference.flash_attention, model).await?;
```

-**Step 8: Build and test**

Run: `cd repos/corvia && cargo build --workspace && cargo test --workspace`
Expected: compiles and tests pass.

-**Step 9: Commit**

```bash
git add crates/corvia-kernel/src/inference_provisioner.rs crates/corvia-cli/src/main.rs
git commit -m "feat(kernel): pass kv_quant and flash_attention through provisioner and CLI"
```

### Task 9: CLI — config-is-truth reload + new flags

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

-**Step 1: Update InferenceCommands::Reload with new flags**

Replace the `Reload` variant in `InferenceCommands` (lines 219-230):

```rust
    /// Reload loaded models with a different device/backend/kv-quant
    Reload {
        /// Device: "auto", "gpu", or "cpu"
        #[arg(long)]
        device: Option<String>,
        /// Backend override: "cuda", "openvino", or "" (auto-select)
        #[arg(long)]
        backend: Option<String>,
        /// Reload only this model (omit to reload all)
        #[arg(long)]
        model: Option<String>,
        /// KV cache quantization: "q8", "q4", "none"
        #[arg(long)]
        kv_quant: Option<String>,
        /// Enable/disable flash attention
        #[arg(long)]
        flash_attention: Option<bool>,
        /// Don't persist changes to corvia.toml
        #[arg(long)]
        no_persist: bool,
    },
```

-**Step 2: Update the match arm for Reload**

In the `Commands::Inference { command }` match (around line 331), update:

```rust
            InferenceCommands::Reload { device, backend, model, kv_quant, flash_attention, no_persist } =>
                cmd_inference_reload(device.as_deref(), backend.as_deref(), model.as_deref(), kv_quant.as_deref(), flash_attention, no_persist).await?,
```

-**Step 3: Rewrite cmd_inference_reload with config-is-truth semantics**

Replace the `cmd_inference_reload` function (around line 1593) with:

```rust
async fn cmd_inference_reload(
    device: Option<&str>,
    backend: Option<&str>,
    model: Option<&str>,
    kv_quant: Option<&str>,
    flash_attention: Option<bool>,
    no_persist: bool,
) -> Result<()> {
    let mut config = load_config()?;
    let grpc_url = match config.embedding.provider {
        corvia_common::config::InferenceProvider::Corvia => config.embedding.url.clone(),
        _ => anyhow::bail!("inference reload requires provider = \"corvia\" in corvia.toml"),
    };

    // Apply overrides to config
    if let Some(d) = device {
        config.inference.device = d.to_string();
    }
    if let Some(b) = backend {
        config.inference.backend = b.to_string();
    }
    if let Some(kv) = kv_quant {
        config.inference.kv_quant = kv.to_string();
    }
    if let Some(fa) = flash_attention {
        config.inference.flash_attention = fa;
    }

    // Persist to config file unless --no-persist
    if !no_persist {
        let config_path = corvia_common::config::CorviaConfig::config_path();
        config.save(&config_path)?;
        println!("Updated corvia.toml [inference] section");
    }

    // Trigger gRPC reload
    let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(&grpc_url);
    if !provisioner.is_running().await {
        anyhow::bail!("corvia-inference is not running at {grpc_url}");
    }
    provisioner.reload_models(
        &config.inference.device,
        &config.inference.backend,
        &config.inference.kv_quant,
        config.inference.flash_attention,
        model,
    ).await?;
    println!("Reload complete.");
    Ok(())
}
```

-**Step 4: Update cmd_inference_status to show new fields**

Replace the table header and row formatting in `cmd_inference_status` (around lines 1629-1635):

```rust
    println!("{:<30} {:<10} {:<8} {:<10} {:<8} {:<6}", "MODEL", "TYPE", "DEVICE", "BACKEND", "KV_QUANT", "FLASH");
    println!("{}", "-".repeat(72));
    for m in &models {
        println!(
            "{:<30} {:<10} {:<8} {:<10} {:<8} {:<6}",
            truncate_str(&m.name, 29),
            m.model_type,
            m.device,
            m.backend,
            if m.kv_quant.is_empty() { "f16" } else { &m.kv_quant },
            if m.flash_attention { "on" } else { "off" },
        );
    }
```

-**Step 5: Build and test**

Run: `cd repos/corvia && cargo build --workspace && cargo test --workspace`
Expected: compiles and tests pass.

-**Step 6: Commit**

```bash
git add crates/corvia-cli/src/main.rs
git commit -m "feat(cli): config-is-truth reload with --kv-quant, --flash-attention, --no-persist"
```

### Task 10: Update workspace corvia.toml

**Files:**
- Modify: `corvia.toml` (workspace root)

-**Step 1: Add [inference] section to corvia.toml**

Add after the `[embedding]` section:

```toml
[inference]
device = "auto"
kv_quant = "q8"
flash_attention = true
```

-**Step 2: Remove device/backend from [embedding] if present**

If `corvia.toml` has `device` or `backend` under `[embedding]`, remove them (they're now ignored anyway).

-**Step 3: Verify config loads**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo run -p corvia-cli -- config get inference 2>/dev/null || echo "binary not built yet"`

-**Step 4: Commit**

```bash
git add corvia.toml
git commit -m "chore: add [inference] section to workspace corvia.toml"
```

### Task 11: Final workspace build + test

-**Step 1: Full workspace build**

Run: `cd repos/corvia && cargo build --workspace`
Expected: clean compile.

-**Step 2: Full workspace test**

Run: `cd repos/corvia && cargo test --workspace`
Expected: all tests pass (pre-existing `staging::tests::test_git_branch_lifecycle` failure is known and unrelated).

-**Step 3: Verify proto fields exist**

Run: `cd repos/corvia && cargo doc -p corvia-proto --no-deps 2>&1 | tail -3`
Expected: compiles. Verify `LoadModelRequest` docs mention `kv_quant` and `flash_attention`.
