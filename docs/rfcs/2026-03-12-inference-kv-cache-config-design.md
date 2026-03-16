# Inference Config & KV Cache Quantization Design

**Date**: 2026-03-12
**Status**: Shipped (v0.4.2)
**Scope**: corvia-inference, corvia-common, corvia-kernel, corvia-cli, corvia-proto, corvia-telemetry

## Problem

corvia-inference supports runtime GPU/CPU switching via `ReloadModels` RPC, but three gaps remain:

1. **KV cache quantization**: llama.cpp supports Q8/Q4 KV cache quantization and flash attention, which reduce VRAM usage by 50-75% with minimal quality loss. These are not wired into corvia-inference.
2. **Config drift**: Runtime changes via CLI/gRPC are ephemeral — they don't persist to `corvia.toml`, so the next server start reverts to old settings.
3. **Config location**: `device` and `backend` fields live in `[embedding]`, but they control both embedding and chat model hardware. This is misleading.

## Design

### 1. New `[inference]` Config Section

Create a unified `[inference]` section in `corvia.toml` that controls hardware for all model types (embedding and chat). Remove `device`/`backend` from `[embedding]`.

```toml
[inference]
device = "auto"          # "auto" | "gpu" | "cpu"
backend = ""             # "" | "cuda" | "openvino" | "cpu"
kv_quant = "q8"          # "none"/"f16" | "q8"/"q8_0" | "q4"/"q4_0"
flash_attention = true   # enable flash attention (requires compatible hardware)
```

**Rationale**: Embedding and chat are both inference workloads. A single `[inference]` section avoids the false separation between "embedding hardware config" and "chat hardware config" when both share the same GPU.

**Breaking change**: Remove `device` and `backend` from `EmbeddingConfig` struct. Acceptable in prerelease (v0.x). Note: the `toml` crate silently ignores unknown fields, so existing `corvia.toml` files with `[embedding].device` will still parse — the fields are simply ignored. Users should move them to `[inference]` to take effect. The `default_device()` function is retained since `InferenceConfig` reuses it.

**Implementation** (`corvia-common/src/config.rs`):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_device")]
    pub device: String,         // "auto" (default), "gpu", "cpu"
    #[serde(default)]
    pub backend: String,        // "", "cuda", "openvino", "cpu"
    #[serde(default = "default_kv_quant")]
    pub kv_quant: String,       // "q8" (default), "q4", "none"
    #[serde(default = "default_flash_attention")]
    pub flash_attention: bool,  // true (default)
}
```

Add `inference: InferenceConfig` to `CorviaConfig` with `#[serde(default)]`. Update `Default`, `full_default()`, `postgres_default()` to include the new section.

### 2. Config-is-Truth Reload Semantics

Runtime changes write to `corvia.toml` first, then trigger gRPC reload. This follows the Triton/PostgreSQL model: config file is the single source of truth.

**Flow**:
```
CLI: corvia inference reload --device gpu
  1. Read corvia.toml
  2. Update [inference].device = "gpu"
  3. Write corvia.toml
  4. Call ReloadModels gRPC with new values
  5. Report result

CLI: corvia inference reload --device gpu --no-persist
  1. Call ReloadModels gRPC directly (ephemeral)
  2. Report result
```

**Why**: Prevents config drift where the running state diverges from the config file. After a restart, the kernel reads `[inference]` from `corvia.toml` and passes the values via `load_models()` gRPC calls, so the server comes up with the last-applied settings.

**Note**: The inference server itself does not read `corvia.toml`. All config values flow through the kernel → provisioner → gRPC path. The config file is authoritative for the kernel side; the inference server is stateless and receives settings per-load/reload.

**Implementation** (`corvia-cli/src/main.rs`):
- `cmd_inference_reload` reads `corvia.toml`, updates the `[inference]` section, saves, then calls `reload_models()` on the provisioner.
- `--no-persist` flag skips the config write and calls gRPC directly.
- Add `--kv-quant` and `--flash-attention` flags to the `Reload` subcommand.

### 3. Proto Changes

Add KV cache and flash attention fields to the gRPC protocol.

**`LoadModelRequest`** — add fields:
```protobuf
string kv_quant = 5;          // "q8", "q4", "none", "" (use server default)
bool flash_attention = 6;     // enable flash attention
```

**`ReloadModelsRequest`** — add fields:
```protobuf
string kv_quant = 5;
bool flash_attention = 6;
```

**`ModelStatus`** — add fields:
```protobuf
string kv_quant = 7;          // current KV quant setting
bool flash_attention = 8;     // current flash attention setting
```

These fields are additive (new field numbers). No backward compatibility issues.

**Proto default for `flash_attention`**: Protobuf `bool` defaults to `false`, but the config default is `true`. The kernel/provisioner must always explicitly set this field from `[inference].flash_attention` when constructing gRPC requests — never rely on proto defaults. This is consistent with how `device` and `backend` are already handled (always populated from config, never left empty).

**KV quant applies to chat models only**. For embedding models, the fields are accepted but ignored (ONNX Runtime manages its own memory). The model manager logs a debug-level message if KV quant is set for an embedding model.

### 4. KV Cache Quantization Implementation

**KV quant value normalization** (`corvia-inference/src/backend.rs`):

Add a `resolve_kv_quant()` function that maps config strings to `llama_cpp_2::context::params::KvCacheType`:
```rust
use llama_cpp_2::context::params::KvCacheType;

pub fn resolve_kv_quant(raw: &str) -> Result<KvCacheType, String> {
    match raw.to_lowercase().as_str() {
        "q8" | "q8_0" => Ok(KvCacheType::Q8_0),
        "q4" | "q4_0" => Ok(KvCacheType::Q4_0),
        "none" | "f16" | "" => Ok(KvCacheType::F16),
        other => Err(format!("Unknown kv_quant: '{other}'. Expected 'q8', 'q4', or 'none'.")),
    }
}
```

This uses `KvCacheType` directly from llama-cpp-2 — no custom enum needed. Default is `"q8"` — near-lossless quality with ~50% VRAM reduction and actually faster inference due to reduced memory bandwidth.

**LlamaContextParams update** (`corvia-inference/src/chat_service.rs`):

Context creation happens per-request in both `generate_blocking()` (line 211) and `generate_streaming_blocking()` (line 318). Both sites must be updated identically:
```rust
use llama_cpp_2::context::params::LlamaFlashAttentionPolicy;

let flash_policy = if flash_attention {
    LlamaFlashAttentionPolicy::Enabled
} else {
    LlamaFlashAttentionPolicy::Disabled
};

let ctx_params = LlamaContextParams::default()
    .with_n_ctx(NonZeroU32::new(ctx_size.max(512)))
    .with_n_batch(512)
    .with_type_k(kv_cache_type)   // KvCacheType::Q8_0, Q4_0, or F16
    .with_type_v(kv_cache_type)
    .with_flash_attention_policy(flash_policy);
```

Note: the llama-cpp-2 API uses `with_flash_attention_policy()` (not `with_flash_attention()`), which takes a `LlamaFlashAttentionPolicy` enum. The `bool` config field maps to `Enabled`/`Disabled`.

Both `type_k` and `type_v` use the same quantization level. Asymmetric K/V quantization is not exposed — it's a niche optimization with limited benefit for single-user inference.

**ChatModelEntry struct update** (`corvia-inference/src/chat_service.rs`):

`ChatModelEntry` must store the resolved settings so they're available at context-creation time (every request):
```rust
struct ChatModelEntry {
    model: Arc<LlamaModel>,
    backend: ResolvedBackend,
    kv_cache_type: KvCacheType,       // resolved from config string
    flash_attention: bool,
}
```

**ModelEntry struct update** (`corvia-inference/src/model_manager.rs`):

`ModelEntry` must also track these fields for `list_models()` reporting:
```rust
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

**Flash attention + Q4 incompatibility**: When `flash_attention = true` and `kv_quant = "q4"`, the inference server (in `chat_service.rs` context creation) silently disables flash attention and logs at WARN level. This is server-side validation — the CLI does not need to validate this combination.

**InferenceProvisioner updates** (`corvia-kernel/src/inference_provisioner.rs`):

`load_models()` and `ensure_ready()` must accept and pass `kv_quant` and `flash_attention` parameters alongside `device` and `backend`. These are added to `LoadModelRequest` fields sent via gRPC.

### 5. Hot-Reloadable Config

Add `"inference"` to `HOT_RELOADABLE_SECTIONS` in `corvia-kernel/src/ops.rs`:
```rust
const HOT_RELOADABLE_SECTIONS: &[&str] = &[
    "agent_lifecycle", "merge", "rag", "chunking", "reasoning", "adapters", "inference"
];
```

Keep `"embedding"` in `RESTART_REQUIRED_SECTIONS` — the remaining `[embedding]` fields (`provider`, `model`, `url`, `dimensions`) still require restart. The device/backend field removal from `EmbeddingConfig` is orthogonal to section reload classification.

### 6. Telemetry

Add span constants to `corvia-telemetry/src/lib.rs`:
```rust
pub const INFERENCE_LOAD: &str = "corvia.inference.load";
pub const INFERENCE_RELOAD: &str = "corvia.inference.reload";
pub const INFERENCE_CONFIG_RELOAD: &str = "corvia.inference.config_reload";
```

Update the `test_span_constants_are_dotted` test to include the new constants.

**Structured log events** (emitted within spans):
- `corvia.inference.load`: `model`, `device`, `backend`, `kv_quant`, `flash_attention` at INFO
- `corvia.inference.reload`: `model`, `from_device`, `to_device`, `from_kv_quant`, `to_kv_quant` at INFO
- `corvia.inference.config_reload`: `section = "inference"`, `changed_keys` at INFO
- KV quant set on embedding model: `model`, `kv_quant = "ignored"` at DEBUG

### 7. CLI Changes

Update `corvia inference reload` with new flags:
```
corvia inference reload [OPTIONS]
  --device <DEVICE>         Device: auto, gpu, cpu
  --backend <BACKEND>       Backend: cuda, openvino, cpu
  --model <MODEL>           Reload only this model
  --kv-quant <KV_QUANT>     KV cache quantization: q8, q4, none
  --flash-attention <BOOL>  Enable/disable flash attention
  --no-persist              Don't write changes to corvia.toml
```

Update `corvia inference status` to show KV quant and flash attention for each loaded model.

## KV Cache Quantization — Technical Details

**What it does**: Quantizes the key-value cache from FP16 to Q8_0 (8-bit) or Q4_0 (4-bit). The KV cache stores attention state for all previous tokens and grows linearly with context length.

**Memory impact** (for a 3B model, 4K context):
- FP16 (none): ~1.5 GB KV cache
- Q8_0: ~0.75 GB (~50% reduction)
- Q4_0: ~0.375 GB (~75% reduction)

**Quality impact**: Q8 is near-lossless (imperceptible for most tasks). Q4 has slight quality degradation on long-context reasoning tasks but is acceptable for summarization, classification, and short-form generation.

**Performance impact**: KV cache quantization is actually faster, not slower, because smaller cache means less memory bandwidth consumed during attention computation.

**Flash attention**: Fused attention kernel that avoids materializing the full attention matrix. Requires Q8 or FP16 KV cache (not Q4). Provides additional speedup on supported hardware. When `flash_attention = true` and `kv_quant = "q4"`, flash attention is silently disabled with a WARN log.

## Migration

1. Remove `device` and `backend` from `[embedding]` in `corvia.toml`
2. Add `[inference]` section with desired settings
3. Existing configs without `[inference]` get defaults: `device = "auto"`, `kv_quant = "q8"`, `flash_attention = true`

## Verification

1. `cargo build -p corvia-proto` — proto compiles with new fields
2. `cargo build -p corvia-inference` — KV quant wired into context params
3. `cargo test -p corvia-common` — config roundtrip with `[inference]` section
4. `cargo test -p corvia-inference` — resolve_kv_quant tests
5. `cargo test --workspace` — no regressions
6. Manual: `corvia inference reload --kv-quant q4` persists to config and reloads model
7. Manual: `corvia inference status` shows kv_quant and flash_attention per model
