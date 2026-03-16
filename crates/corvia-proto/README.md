# corvia-proto

Protocol Buffer definitions for corvia's gRPC inference service.

## Services

- **EmbeddingService** — text-to-vector embedding requests
- **ChatService** — LLM chat completion with streaming support

## Message types

- `EmbedRequest` / `EmbedResponse` — batch text embedding
- `LoadModelRequest` / `LoadModelResponse` — model loading with device/backend selection
- `ChatRequest` / `ChatResponse` — chat completion with configurable generation parameters

## Usage

This crate is a build dependency. It generates Rust types from `.proto` files
via `tonic-build` at compile time.

```toml
[dependencies]
corvia-proto = { path = "../corvia-proto" }
```

## License

[AGPL-3.0-only](../../../LICENSE)
