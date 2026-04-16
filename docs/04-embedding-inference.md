# Embedding Inference Engine

**Date**: 2026-04-14
**Decision**: fastembed-rs + ort (ONNX Runtime), nomic-embed-text-v1.5 default
**Updated**: Added cross-platform GPU deep dive

## V1 Architecture (What to Simplify)

V1 runs a separate gRPC inference server (corvia-inference crate) with:
- fastembed-rs v5 + ort v2.0.0-rc.11
- CUDA and OpenVINO execution providers
- llama-cpp-2 for chat models
- Separate process on port 8030

**V2 change**: Embed directly in the main binary. No separate server. No gRPC.
The embedding engine is a library call, not a network call.

## Cross-Platform GPU Support

### What's Actually Available (ort prebuilt binaries)

| Platform | CPU | CUDA | DirectML | CoreML | OpenVINO | WebGPU |
|----------|-----|------|----------|--------|----------|--------|
| Linux x86_64 | Prebuilt | Prebuilt (cu12/cu13) | N/A | N/A | Source build | Prebuilt |
| Linux ARM64 | Prebuilt | No | N/A | N/A | No | No |
| Windows x86_64 | Prebuilt | Prebuilt (cu12/cu13) | **Included in ALL builds** | N/A | Source build | Prebuilt |
| Windows ARM64 | Prebuilt | No | No | N/A | No | No |
| macOS ARM64 | Prebuilt | N/A | N/A | **Prebuilt** | N/A | Prebuilt |
| macOS x86_64 | **Discontinued** | N/A | N/A | N/A | N/A | N/A |

### The Key Finding: Windows and macOS Get Free GPU

**Windows**: DirectML is baked into ALL pyke-provided Windows builds. The ort build
script unconditionally links D3D12, DXGI, DXCORE, and DirectML on Windows targets.
Any DirectX 12 GPU (NVIDIA, AMD, Intel -- last 8+ years) gets acceleration with
**zero user-side installation**.

**macOS**: CoreML is a system framework on macOS 12+. The prebuilt aarch64-apple-darwin
binary supports it. Apple Neural Engine / GPU used automatically on M1/M2/M3/M4 with
**zero user-side installation**.

**Linux**: CPU-only by default. CUDA variant needs CUDA runtime + cuDNN installed.
Or point at Ollama/TEI for zero-hassle GPU.

### Runtime Fallback Chain

```rust
let session = Session::builder()?
    .with_execution_providers([
        CUDAExecutionProvider::default().build(),      // NVIDIA GPU (if available)
        DirectMLExecutionProvider::default().build(),   // Any DX12 GPU on Windows
        CoreMLExecutionProvider::default().build(),     // Apple Silicon
        CPUExecutionProvider::default().build(),        // Always works
    ])?
    .commit_from_file("model.onnx")?;
```

Each EP checks `is_available()` at runtime and silently falls back. A single
per-platform binary handles all GPU variants automatically.

### GPU Benchmarks for Embedding

From corvia v1 benchmarks (nomic-embed-text-v1.5):

| Backend | Latency/embed | Throughput | Speedup |
|---------|---------------|------------|---------|
| CPU (ONNX) | ~213ms | 4.7/s | baseline |
| OpenVINO (Intel iGPU) | 51ms | 19.8/s | 4.21x |
| CUDA (RTX 3060) | 56ms | 17.8/s | 3.81x |

Industry data: GPU provides 3-5x speedup for single items, 10-50x for batched
ingestion. GPU is most valuable during `corvia ingest` (thousands of chunks),
less important for individual `corvia search` queries.

### Distribution Strategy

**GitHub Release assets (5 binaries):**

| Asset | GPU Support | User Installs |
|-------|-------------|---------------|
| `corvia-linux-x86_64` | CPU only | Nothing |
| `corvia-linux-x86_64-cuda` | CUDA + CPU | CUDA runtime + cuDNN |
| `corvia-windows-x86_64.exe` | DirectML + CPU | **Nothing** |
| `corvia-windows-x86_64-cuda.exe` | CUDA + DirectML + CPU | CUDA runtime + cuDNN |
| `corvia-macos-arm64` | CoreML + CPU | **Nothing** |

3 out of 5 binaries get GPU acceleration with zero user dependencies.

### External Provider Fallback

For users who want GPU without dealing with CUDA:

```bash
corvia ingest                                    # CPU (default)
corvia ingest --embedding-provider ollama        # Ollama GPU (if installed)
corvia ingest --embedding-provider tei           # TEI Docker GPU (if running)
```

This offloads GPU complexity to maintained external tools. Ollama is already
common in the AI dev community.

### What NOT to Do

- ROCm: **Deprecated** from ONNX Runtime starting v1.23. Dead end.
- Vulkan: Does not exist in ONNX Runtime. Multiple requests, none implemented.
- candle: 9-10x slower on CPU than ONNX. Not worth it.
- llama.cpp for embeddings: Designed for generation. ONNX is 9.46x faster.
- CUDA in default binary: Runtime dependencies are a support nightmare.
  Ship CUDA as a separate variant.
- Single cross-platform binary: Not possible. Per-platform builds are standard.

## Runtime Comparison

| Approach | Startup | Latency/query | Memory | User installs |
|----------|---------|---------------|--------|---------------|
| **fastembed-rs (ONNX)** | 200-500ms | 15-40ms CPU | 100MB + model | Nothing |
| TEI (Docker) | 5-15s | 5-20ms GPU | 500MB-2GB | Docker |
| Ollama | 10-30s cold | 15-50ms warm | 500MB+ | Ollama |
| candle (Rust) | 200-500ms | 40-200ms CPU | 100MB + model | Nothing |
| API (OpenAI/Voyage) | 0ms | 200-800ms net | Minimal | API key |

## Model Comparison

| Model | Size | Dims | Context | MTEB | License |
|-------|------|------|---------|------|---------|
| all-MiniLM-L6-v2 | 46MB | 384 | 256 | 56.3 | Apache 2.0 |
| Snowflake arctic-embed-m | 227MB | 768 | 8192 | 60 | Apache 2.0 |
| **nomic-embed-text-v1.5** | **274MB** | **768** | **8192** | **62.4** | **Apache 2.0** |
| BGE-M3 | 1.2GB | 1024 | 8192 | 63 | MIT |

**Decision**: nomic-embed-text-v1.5 (default), all-MiniLM-L6-v2 (lite mode).
Download model on first run, not bundled (274MB binary too large).

## WebGPU (Future)

Prebuilt WebGPU binaries exist for all 3 platforms in ort's dist.txt. WebGPU
could eventually provide cross-platform GPU with no vendor dependencies. Still
experimental in ONNX Runtime. Worth watching for v2.1+.
