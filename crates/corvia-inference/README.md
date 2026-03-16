# corvia-inference

gRPC inference server for corvia. Provides embedding and chat completion services
using ONNX Runtime and llama.cpp, with support for multiple hardware backends.

## Features

- **Embedding** via [fastembed](https://github.com/Anush008/fastembed-rs) (ONNX Runtime)
  - OpenVINO execution provider for Intel iGPU acceleration
  - CUDA execution provider for NVIDIA GPU acceleration
  - CPU fallback with automatic detection
- **Chat completion** via [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs)
  - GGUF model loading from Hugging Face Hub
  - KV cache quantization and flash attention support
  - Optional CUDA GPU layer offloading (`--features cuda`)
- **Runtime backend switching** via `[inference]` config in `corvia.toml`

## Usage

```bash
# Start with default settings (reads from corvia.toml)
corvia-inference

# Override port
corvia-inference --port 8030

# With CUDA support (requires feature flag at build time)
cargo build -p corvia-inference --features cuda
```

## Protocol

gRPC service definitions are in `corvia-proto`. The server listens on port 8030
by default and implements `EmbeddingService` and `ChatService`.

## License

[AGPL-3.0-only](../../../LICENSE)
