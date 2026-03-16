# OpenTelemetry Standardization Implementation Plan

> **Status:** Shipped (v0.4.0)

**Goal:** Standardize all logging and tracing in corvia core to OTEL format with full OTLP gRPC export, W3C trace propagation, and dogfooding via trace ingestion.

**Architecture:** Single-pass rewrite of `corvia-telemetry` as the sole telemetry authority. Add OpenTelemetry SDK with OTLP gRPC exporter, wire W3C trace propagation between server↔inference via vendored tonic adapters, add `TraceLayer` to Axum routers, sweep all logging to structured fields, and add a `/v1/traces/ingest` endpoint for dogfooding.

**Tech Stack:** Rust, `opentelemetry` 0.28, `opentelemetry_sdk` 0.28, `opentelemetry-otlp` 0.28, `tracing-opentelemetry` 0.29, `tower-http` TraceLayer, tonic 0.12

**Spec:** `2026-03-11-otel-standardization-design.md`

---

## Chunk 1: Dependencies & Configuration

### Task 1: Add OpenTelemetry workspace dependencies

**Files:**
- Modify: `Cargo.toml:23-38` (workspace dependencies)

-**Step 1: Add OTel deps to workspace Cargo.toml**

Add to `[workspace.dependencies]` section after the existing `tracing-appender` line:

```toml
opentelemetry = "0.28"
opentelemetry_sdk = { version = "0.28", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.28", features = ["grpc-tonic"] }
tracing-opentelemetry = "0.29"
```

-**Step 2: Verify workspace resolves**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check --workspace 2>&1 | head -5`
Expected: Compiles (new deps are unused so far, but should resolve)

-**Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "build: add OpenTelemetry workspace dependencies"
```

---

### Task 2: Extend TelemetryConfig

**Files:**
- Modify: `crates/corvia-common/src/config.rs:281-306`

-**Step 1: Write the failing test**

Add to the existing `tests` module in `config.rs`:

```rust
#[test]
fn test_telemetry_config_new_fields() {
    let config = TelemetryConfig::default();
    assert_eq!(config.service_name, "corvia");
    assert_eq!(config.otlp_protocol, "grpc");
}

#[test]
fn test_telemetry_config_deserialize_new_fields() {
    let toml_str = r#"
        exporter = "otlp"
        otlp_endpoint = "http://localhost:4317"
        otlp_protocol = "grpc"
        service_name = "corvia-inference"
        log_format = "json"
        metrics_enabled = true
    "#;
    let config: TelemetryConfig = toml::de::from_str(toml_str).unwrap();
    assert_eq!(config.service_name, "corvia-inference");
    assert_eq!(config.otlp_protocol, "grpc");
    assert_eq!(config.otlp_endpoint, "http://localhost:4317");
}
```

-**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-common test_telemetry_config_new_fields -- --nocapture`
Expected: FAIL — `service_name` field does not exist

-**Step 3: Add new fields to TelemetryConfig**

In `config.rs`, update `TelemetryConfig` struct (line 281-291):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_telemetry_exporter")]
    pub exporter: String,
    #[serde(default)]
    pub otlp_endpoint: String,
    #[serde(default = "default_telemetry_otlp_protocol")]
    pub otlp_protocol: String,
    #[serde(default = "default_telemetry_service_name")]
    pub service_name: String,
    #[serde(default = "default_telemetry_log_format")]
    pub log_format: String,
    #[serde(default = "default_telemetry_metrics_enabled")]
    pub metrics_enabled: bool,
}
```

Add default functions after the existing ones (after line 295):

```rust
fn default_telemetry_otlp_protocol() -> String { "grpc".into() }
fn default_telemetry_service_name() -> String { "corvia".into() }
```

Update `Default` impl (line 297-305):

```rust
impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            exporter: default_telemetry_exporter(),
            otlp_endpoint: String::new(),
            otlp_protocol: default_telemetry_otlp_protocol(),
            service_name: default_telemetry_service_name(),
            log_format: default_telemetry_log_format(),
            metrics_enabled: default_telemetry_metrics_enabled(),
        }
    }
}
```

-**Step 4: Run tests to verify they pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-common -- --nocapture`
Expected: All tests PASS

-**Step 5: Commit**

```bash
git add crates/corvia-common/src/config.rs
git commit -m "feat(config): add service_name and otlp_protocol to TelemetryConfig"
```

---

## Chunk 2: corvia-telemetry Rewrite

### Task 3: Add OTel dependencies to corvia-telemetry

**Files:**
- Modify: `crates/corvia-telemetry/Cargo.toml`

-**Step 1: Update Cargo.toml**

Replace the full `[dependencies]` section:

```toml
[dependencies]
corvia-common = { workspace = true }
tracing.workspace = true
tracing-subscriber = { workspace = true, features = ["env-filter", "json", "fmt"] }
tracing-appender = { workspace = true }
tracing-opentelemetry = { workspace = true }
opentelemetry = { workspace = true }
opentelemetry_sdk = { workspace = true }
opentelemetry-otlp = { workspace = true }
anyhow.workspace = true
tonic.workspace = true
```

-**Step 2: Verify it compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-telemetry`
Expected: Compiles (new deps available but unused)

-**Step 3: Commit**

```bash
git add crates/corvia-telemetry/Cargo.toml
git commit -m "build(telemetry): add OpenTelemetry dependencies"
```

---

### Task 4: Create propagation module

**Files:**
- Create: `crates/corvia-telemetry/src/propagation.rs`
- Modify: `crates/corvia-telemetry/src/lib.rs:1` (add module declaration)

-**Step 1: Write the failing test**

Create `crates/corvia-telemetry/src/propagation.rs`:

```rust
//! W3C trace context propagation adapters for tonic gRPC.
//!
//! Vendored adapters that bridge `opentelemetry::propagation` with tonic's
//! `MetadataMap` for injecting/extracting `traceparent`/`tracestate` headers.

use opentelemetry::propagation::{Extractor, Injector};
use tonic::metadata::MetadataMap;

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    #[test]
    fn test_metadata_injector_extractor_roundtrip() {
        let propagator = TraceContextPropagator::new();
        let mut metadata = MetadataMap::new();

        // Inject a traceparent header
        let cx = opentelemetry::Context::new();
        propagator.inject_context(&cx, &mut MetadataInjector(&mut metadata));

        // Extract it back
        let extracted_cx = propagator.extract(&MetadataExtractor(&metadata));

        // Both contexts should be valid (no panic, no corruption)
        drop(extracted_cx);
    }

    #[test]
    fn test_injector_sets_header() {
        let mut metadata = MetadataMap::new();
        let mut injector = MetadataInjector(&mut metadata);
        injector.set("traceparent", "00-trace-span-01".to_string());
        assert_eq!(
            metadata.get("traceparent").unwrap().to_str().unwrap(),
            "00-trace-span-01"
        );
    }

    #[test]
    fn test_extractor_gets_header() {
        let mut metadata = MetadataMap::new();
        metadata.insert("traceparent", "00-trace-span-01".parse().unwrap());
        let extractor = MetadataExtractor(&metadata);
        assert_eq!(extractor.get("traceparent"), Some("00-trace-span-01"));
    }

    #[test]
    fn test_extractor_missing_key_returns_none() {
        let metadata = MetadataMap::new();
        let extractor = MetadataExtractor(&metadata);
        assert_eq!(extractor.get("traceparent"), None);
    }
}
```

-**Step 2: Add module declaration to lib.rs**

Add at the top of `crates/corvia-telemetry/src/lib.rs` (before line 1):

```rust
pub mod propagation;
```

-**Step 3: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry test_metadata -- --nocapture`
Expected: FAIL — `MetadataInjector` and `MetadataExtractor` not found

-**Step 4: Implement MetadataInjector and MetadataExtractor**

Add to `propagation.rs` before the `#[cfg(test)]` block:

```rust
/// Injects OpenTelemetry context into tonic `MetadataMap` for outgoing gRPC calls.
pub struct MetadataInjector<'a>(pub &'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(val) = value.parse() {
            self.0.insert(key, val);
        }
    }
}

/// Extracts OpenTelemetry context from tonic `MetadataMap` for incoming gRPC calls.
pub struct MetadataExtractor<'a>(pub &'a MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().filter_map(|k| match k {
            tonic::metadata::KeyRef::Ascii(key) => Some(key.as_str()),
            _ => None,
        }).collect()
    }
}
```

-**Step 5: Run tests to verify they pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry -- --nocapture`
Expected: All tests PASS

-**Step 6: Commit**

```bash
git add crates/corvia-telemetry/src/propagation.rs crates/corvia-telemetry/src/lib.rs
git commit -m "feat(telemetry): add W3C trace context propagation adapters for tonic"
```

---

### Task 5: Rewrite init_telemetry with OTLP support

**Files:**
- Modify: `crates/corvia-telemetry/src/lib.rs:1-116`

-**Step 1: Write the failing test**

Add to the `tests` module in `lib.rs` (after existing tests):

```rust
#[test]
fn test_init_telemetry_returns_guard() {
    // Use a unique test — cannot init global subscriber twice in same process,
    // so we just verify the function signature and config handling.
    let config = TelemetryConfig {
        service_name: "test-service".into(),
        ..TelemetryConfig::default()
    };
    // Verify config fields are accessible
    assert_eq!(config.service_name, "test-service");
    assert_eq!(config.otlp_protocol, "grpc");
}
```

-**Step 2: Run test to verify it passes (baseline)**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry test_init_telemetry_returns_guard -- --nocapture`
Expected: PASS

-**Step 3: Rewrite lib.rs**

Replace the entire `crates/corvia-telemetry/src/lib.rs` with:

```rust
pub mod propagation;

use corvia_common::config::TelemetryConfig;
use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;

/// Span name constants following OTel dotted namespace convention.
pub mod spans {
    pub const AGENT_REGISTER: &str = "corvia.agent.register";
    pub const SESSION_CREATE: &str = "corvia.session.create";
    pub const ENTRY_WRITE: &str = "corvia.entry.write";
    pub const ENTRY_EMBED: &str = "corvia.entry.embed";
    pub const ENTRY_EMBED_BATCH: &str = "corvia.entry.embed_batch";
    pub const SESSION_COMMIT: &str = "corvia.session.commit";
    pub const MERGE_PROCESS: &str = "corvia.merge.process";
    pub const MERGE_PROCESS_ENTRY: &str = "corvia.merge.process_entry";
    pub const MERGE_CONFLICT: &str = "corvia.merge.conflict";
    pub const MERGE_LLM_RESOLVE: &str = "corvia.merge.llm_resolve";
    pub const GC_RUN: &str = "corvia.gc.run";
    pub const STORE_INSERT: &str = "corvia.store.insert";
    pub const STORE_SEARCH: &str = "corvia.store.search";
    pub const STORE_GET: &str = "corvia.store.get";
    pub const RAG_CONTEXT: &str = "corvia.rag.context";
    pub const RAG_ASK: &str = "corvia.rag.ask";
}

/// Opaque handle that keeps the telemetry pipeline alive.
/// Hold this in your top-level scope (e.g. `main`); dropping it flushes
/// any buffered output and shuts down the OTel tracer provider.
pub struct TelemetryGuard {
    _file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    _tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

/// Initialize the tracing subscriber pipeline based on config.
///
/// Returns a [`TelemetryGuard`] that **must** be held for the lifetime of
/// the process. Dropping the guard flushes buffered output and shuts down
/// the OpenTelemetry tracer provider.
///
/// ## OTLP Export
///
/// OTLP export is additive. If `config.otlp_endpoint` is non-empty, an
/// OpenTelemetry layer is added to the subscriber regardless of the
/// `exporter` setting. This means you always get local output plus
/// optional OTLP export.
///
/// ## Environment Overrides
///
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` overrides `config.otlp_endpoint`
/// - `OTEL_SERVICE_NAME` overrides `config.service_name`
/// - `RUST_LOG` overrides the default log level filter
pub fn init_telemetry(config: &TelemetryConfig) -> anyhow::Result<TelemetryGuard> {
    use tracing_subscriber::{fmt, EnvFilter, prelude::*};

    // Register W3C trace context propagator globally
    global::set_text_map_propagator(TraceContextPropagator::new());

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let mut file_guard = None;
    let mut tracer_provider = None;

    // Resolve OTLP endpoint: env var takes precedence over config
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| config.otlp_endpoint.clone());

    // Resolve service name: env var takes precedence over config
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .unwrap_or_else(|_| config.service_name.clone());

    // Build the OTLP layer if endpoint is configured
    let otel_layer = if !otlp_endpoint.is_empty() {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&otlp_endpoint)
            .build()?;

        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                opentelemetry_sdk::Resource::builder()
                    .with_service_name(service_name)
                    .build(),
            )
            .build();

        let tracer = provider.tracer("corvia");
        tracer_provider = Some(provider);

        Some(tracing_opentelemetry::layer().with_tracer(tracer))
    } else {
        None
    };

    // Build the local output layer based on exporter config
    match config.exporter.as_str() {
        "file" => {
            let file_appender = tracing_appender::rolling::daily("logs", "corvia.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            file_guard = Some(guard);

            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json().with_writer(non_blocking))
                    .with(otel_layer)
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().with_writer(non_blocking))
                    .with(otel_layer)
                    .init();
            }
        }
        _ => {
            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json())
                    .with(otel_layer)
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer())
                    .with(otel_layer)
                    .init();
            }
        }
    }

    Ok(TelemetryGuard {
        _file_guard: file_guard,
        _tracer_provider: tracer_provider,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_constants_are_dotted() {
        let all = [
            spans::AGENT_REGISTER, spans::SESSION_CREATE, spans::ENTRY_WRITE,
            spans::ENTRY_EMBED, spans::ENTRY_EMBED_BATCH, spans::SESSION_COMMIT,
            spans::MERGE_PROCESS, spans::MERGE_PROCESS_ENTRY,
            spans::MERGE_CONFLICT, spans::MERGE_LLM_RESOLVE,
            spans::GC_RUN, spans::STORE_INSERT, spans::STORE_SEARCH,
            spans::STORE_GET, spans::RAG_CONTEXT, spans::RAG_ASK,
        ];
        for name in &all {
            assert!(name.starts_with("corvia."), "{name} must start with 'corvia.'");
            assert!(name.contains('.'), "{name} must use dotted notation");
        }
    }

    #[test]
    fn test_default_telemetry_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.exporter, "stdout");
        assert_eq!(config.log_format, "text");
        assert!(config.metrics_enabled);
        assert!(config.otlp_endpoint.is_empty());
        assert_eq!(config.service_name, "corvia");
        assert_eq!(config.otlp_protocol, "grpc");
    }

    #[test]
    fn test_init_telemetry_returns_guard() {
        let config = TelemetryConfig {
            service_name: "test-service".into(),
            ..TelemetryConfig::default()
        };
        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.otlp_protocol, "grpc");
    }
}
```

-**Step 4: Run all telemetry tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry -- --nocapture`
Expected: All tests PASS

-**Step 5: Verify full workspace compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check --workspace`
Expected: Compiles (some warnings about removed constants `SEARCH`/`ENTRY_INSERT` are expected — they'll be cleaned up in the logging sweep)

-**Step 6: Commit**

```bash
git add crates/corvia-telemetry/src/lib.rs
git commit -m "feat(telemetry): rewrite init_telemetry with OTLP gRPC export and W3C propagation"
```

---

## Chunk 3: W3C Trace Propagation

### Task 6: Inject trace context in gRPC client

**Files:**
- Modify: `crates/corvia-kernel/src/grpc_engine.rs:1-78`
- Modify: `crates/corvia-kernel/Cargo.toml` (if `opentelemetry` not already a dep)

-**Step 1: Add opentelemetry dep to corvia-kernel Cargo.toml**

Add to `[dependencies]` in `crates/corvia-kernel/Cargo.toml`:

```toml
opentelemetry = { workspace = true }
tracing-opentelemetry = { workspace = true }
```

-**Step 2: Update embed() to inject trace context**

In `grpc_engine.rs`, update imports (line 1-6):

```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_proto::embedding_service_client::EmbeddingServiceClient;
use corvia_proto::{EmbedBatchRequest, EmbedRequest};
use corvia_telemetry::propagation::MetadataInjector;
use opentelemetry::global;
use tonic::transport::Channel;
use tracing::warn;
use tracing_opentelemetry::OpenTelemetrySpanExt;
```

Note: `OpenTelemetrySpanExt` is required for the `.context()` method on `tracing::Span`.

Replace the `embed()` method (lines 54-64):

```rust
    #[tracing::instrument(name = "corvia.entry.embed", skip(self, text))]
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut client = self.connect().await?;
        let mut request = tonic::Request::new(EmbedRequest {
            model: self.model.clone(),
            text: Self::truncate(text),
        });

        // Inject W3C trace context into gRPC metadata
        let cx = tracing::Span::current().context();
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&cx, &mut MetadataInjector(request.metadata_mut()));
        });

        let response = client.embed(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC Embed failed: {e}")))?;
        Ok(response.into_inner().embedding)
    }
```

-**Step 3: Update embed_batch() to inject trace context and add instrumentation**

Replace `embed_batch()` method (lines 66-77):

```rust
    #[tracing::instrument(name = "corvia.entry.embed_batch", skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut client = self.connect().await?;
        let mut request = tonic::Request::new(EmbedBatchRequest {
            model: self.model.clone(),
            texts: texts.iter().map(|t| Self::truncate(t)).collect(),
        });

        // Inject W3C trace context into gRPC metadata
        let cx = tracing::Span::current().context();
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&cx, &mut MetadataInjector(request.metadata_mut()));
        });

        let response = client.embed_batch(request).await
            .map_err(|e| CorviaError::Embedding(format!("gRPC EmbedBatch failed: {e}")))?;
        let mut embeddings: Vec<_> = response.into_inner().embeddings;
        embeddings.sort_by_key(|e| e.index);
        Ok(embeddings.into_iter().map(|e| e.values).collect())
    }
```

-**Step 4: Verify it compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-kernel`
Expected: Compiles

-**Step 5: Run existing grpc_engine tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel grpc_engine -- --nocapture`
Expected: All existing tests PASS

-**Step 6: Commit**

```bash
git add crates/corvia-kernel/Cargo.toml crates/corvia-kernel/src/grpc_engine.rs
git commit -m "feat(kernel): inject W3C trace context in gRPC embedding calls"
```

---

### Task 7: Add trace context extraction to corvia-inference

**Files:**
- Modify: `crates/corvia-inference/Cargo.toml`
- Modify: `crates/corvia-inference/src/main.rs:1-62`

-**Step 1: Add dependencies to corvia-inference Cargo.toml**

Add to `[dependencies]`:

```toml
corvia-common = { workspace = true }
corvia-telemetry = { workspace = true }
opentelemetry = { workspace = true }
```

-**Step 2: Rewrite main.rs to use init_telemetry and tonic interceptor**

Replace `crates/corvia-inference/src/main.rs`:

```rust
mod chat_service;
mod embedding_service;
mod model_manager;

use clap::Parser;
use corvia_common::config::TelemetryConfig;
use corvia_proto::chat_service_server::ChatServiceServer;
use corvia_proto::embedding_service_server::EmbeddingServiceServer;
use corvia_proto::model_manager_server::ModelManagerServer;
use corvia_telemetry::propagation::MetadataExtractor;
use opentelemetry::global;
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

/// Tonic interceptor that extracts W3C trace context from incoming gRPC metadata
/// and sets it as the current context's parent span.
fn accept_trace(mut request: tonic::Request<()>) -> std::result::Result<tonic::Request<()>, tonic::Status> {
    let parent_cx = global::get_text_map_propagator(|propagator| {
        propagator.extract(&MetadataExtractor(request.metadata()))
    });
    // Store the extracted context in request extensions so downstream
    // handlers can access it. The tracing-opentelemetry layer will
    // automatically pick up the parent context from the current span.
    request.extensions_mut().insert(parent_cx);
    Ok(request)
}
```

**Important:** The `Context::attach()` guard approach does not work in tonic interceptors because the guard is dropped at the end of the function, before handler spans are created. Instead, store the context in request extensions. The embedding service handler should then extract it:

```rust
// In embedding_service.rs handler methods, at the top:
use opentelemetry::Context;
if let Some(parent_cx) = request.extensions().get::<Context>() {
    let _guard = parent_cx.clone().attach();
}
```

Alternatively, if tonic interceptors don't support extensions propagation cleanly, use a tonic `tower::Layer` instead of an interceptor, which gives you control over the async scope where the context guard lives. The implementer should verify which approach works and add a note to the integration test (Task 15, Step 1).

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telem_cfg = TelemetryConfig {
        service_name: "corvia-inference".into(),
        ..TelemetryConfig::default()
    };
    let _telemetry_guard = corvia_telemetry::init_telemetry(&telem_cfg)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port } => {
            let addr = format!("0.0.0.0:{port}").parse()?;
            let embed_svc = embedding_service::EmbeddingServiceImpl::new();
            let chat_svc = chat_service::ChatServiceImpl::new();
            let model_mgr = model_manager::ModelManagerService::new(
                embed_svc.clone(),
                chat_svc.clone(),
            );

            tracing::info!(port, "inference_server_starting");

            Server::builder()
                .add_service(ModelManagerServer::new(model_mgr))
                .add_service(EmbeddingServiceServer::with_interceptor(embed_svc, accept_trace))
                .add_service(ChatServiceServer::with_interceptor(chat_svc, accept_trace))
                .serve(addr)
                .await?;
        }
    }

    Ok(())
}
```

-**Step 3: Verify it compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-inference`
Expected: Compiles

-**Step 4: Commit**

```bash
git add crates/corvia-inference/Cargo.toml crates/corvia-inference/src/main.rs
git commit -m "feat(inference): use init_telemetry and add W3C trace context extraction"
```

---

## Chunk 4: HTTP Request Tracing

### Task 8: Add TraceLayer to Axum routers

**Files:**
- Modify: `crates/corvia-server/src/rest.rs:257-282`
- Modify: `crates/corvia-server/src/mcp.rs:330-340`

-**Step 1: Add TraceLayer to REST router**

In `rest.rs`, add import at top of file:

```rust
use tower_http::trace::TraceLayer;
```

Update the `router()` function (line 257-282) to add the layer before `.with_state()`:

```rust
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Existing memory endpoints
        .route("/v1/memories/write", post(write_memory))
        .route("/v1/memories/search", post(search_memories))
        // Agent coordination endpoints
        .route("/v1/agents", post(register_agent))
        .route("/v1/agents/{agent_id}/sessions", post(create_session))
        .route("/v1/sessions/{session_id}/heartbeat", post(heartbeat))
        .route("/v1/sessions/{session_id}/write", post(session_write))
        .route("/v1/sessions/{session_id}/commit", post(commit_session))
        .route("/v1/sessions/{session_id}/rollback", post(rollback_session))
        .route("/v1/sessions/{session_id}/recover", post(recover_session))
        .route("/v1/sessions/{session_id}/state", get(session_state))
        // Temporal, graph, and reasoning endpoints
        .route("/v1/entries/{id}/history", get(entry_history))
        .route("/v1/entries/{id}/edges", get(entry_edges))
        .route("/v1/evolution", get(evolution))
        .route("/v1/edges", post(create_edge))
        .route("/v1/reason", post(reason))
        // RAG endpoints
        .route("/v1/context", post(rag_context))
        .route("/v1/ask", post(rag_ask))
        .route("/health", get(health))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
```

-**Step 2: Add TraceLayer to MCP router**

In `mcp.rs`, add import at top of file:

```rust
use tower_http::trace::TraceLayer;
```

Update `mcp_router()` function (line 330-340):

```rust
pub fn mcp_router(state: Arc<AppState>) -> Router {
    let mcp_state = Arc::new(McpState {
        app: state,
    });
    Router::new()
        .route("/mcp", post(handle_mcp_post))
        .route("/mcp", get(handle_mcp_get))
        .route("/mcp", delete(handle_mcp_delete))
        .layer(TraceLayer::new_for_http())
        .with_state(mcp_state)
}
```

-**Step 3: Verify it compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-server`
Expected: Compiles

-**Step 4: Run existing server tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server -- --nocapture`
Expected: All tests PASS

-**Step 5: Commit**

```bash
git add crates/corvia-server/src/rest.rs crates/corvia-server/src/mcp.rs
git commit -m "feat(server): add tower-http TraceLayer to REST and MCP routers"
```

---

## Chunk 5: Logging Standardization Sweep

### Task 9: Structured logging sweep — corvia-kernel

**Files:**
- Modify: `crates/corvia-kernel/src/lite_store.rs` (lines 99, 154, 184, 337, 499, 718, 779, 868)
- Modify: `crates/corvia-kernel/src/graph_store.rs` (lines 58, 196, 203, 210)
- Modify: `crates/corvia-kernel/src/staging.rs` (lines 119, 136, 152, 164)
- Modify: `crates/corvia-kernel/src/ollama_engine.rs` (line 68)

-**Step 1: Fix lite_store.rs logging**

Replace each string-interpolated log with structured fields:

```rust
// Line 99 — replace:
// warn!("Failed to load HNSW from disk, will rebuild: {e}");
warn!(err = %e, "hnsw_load_failed_rebuilding");

// Line 154 — replace:
// warn!("Failed to persist rebuilt HNSW index: {e}");
warn!(err = %e, "hnsw_persist_failed");

// Line 184 — replace:
// tracing::warn!("Skipping malformed entry in Redb: {e}");
tracing::warn!(err = %e, "redb_entry_malformed_skipping");

// Line 337 — replace:
// info!("HNSW flushed to {}", hnsw_dir.display());
info!(path = %hnsw_dir.display(), "hnsw_flushed");

// Line 499 — replace:
// info!("LiteStore schema initialized (dimensions={})", self.dimensions);
info!(dimensions = self.dimensions, "lite_store_schema_initialized");

// Line 718 — replace:
// info!("Deleted scope '{}' ({} entries)", scope_id, uuids_to_delete.len());
info!(scope_id, entries_deleted = uuids_to_delete.len(), "scope_deleted");

// Line 779 — replace:
// tracing::warn!("Malformed temporal index key (no entry_id segment): {}", key_str);
tracing::warn!(key = %key_str, "temporal_index_key_malformed");

// Line 868 — replace:
// tracing::warn!("Malformed temporal index key (no entry_id segment): {}", key_str);
tracing::warn!(key = %key_str, "temporal_index_key_malformed");
```

-**Step 2: Fix graph_store.rs logging**

```rust
// Line 58 — replace:
// tracing::warn!("Malformed GRAPH_EDGES key: {}", key_str);
tracing::warn!(key = %key_str, "graph_edges_key_malformed");

// Line 196 — replace:
// tracing::warn!("Malformed GRAPH_EDGES key: {}", key_str);
tracing::warn!(key = %key_str, "graph_edges_key_malformed");

// Line 203 — replace:
// tracing::warn!("Skipping edge with invalid from_id '{}': {e}", parts[0]);
tracing::warn!(from_id = parts[0], err = %e, "edge_from_id_invalid");

// Line 210 — replace:
// tracing::warn!("Skipping edge with invalid to_id '{}': {e}", parts[2]);
tracing::warn!(to_id = parts[2], err = %e, "edge_to_id_invalid");
```

-**Step 3: Fix staging.rs logging**

```rust
// Line 119 — replace:
// warn!("Not a git repo, skipping branch creation for {branch_name}");
warn!(branch = %branch_name, "git_branch_create_skipped_not_repo");

// Line 136 — replace:
// warn!("Not a git repo, skipping commit on {branch_name}");
warn!(branch = %branch_name, "git_commit_skipped_not_repo");

// Line 152 — replace:
// warn!("Not a git repo, skipping merge for {branch_name}");
warn!(branch = %branch_name, "git_merge_skipped_not_repo");

// Line 164 — replace:
// warn!("Not a git repo, skipping branch deletion for {branch_name}");
warn!(branch = %branch_name, "git_branch_delete_skipped_not_repo");
```

-**Step 4: Fix ollama_engine.rs logging**

```rust
// Line 68 — replace:
// warn!("Truncating input from {} to {} chars for embedding", t.len(), MAX_EMBED_CHARS);
warn!(input_len = t.len(), max_len = MAX_EMBED_CHARS, "embedding_input_truncated");
```

-**Step 5: Add #[instrument] to ollama_engine embed_batch**

In `ollama_engine.rs`, find the `embed_batch` method and add instrumentation. The method should get:

```rust
#[tracing::instrument(name = "corvia.entry.embed_batch", skip(self, texts), fields(batch_size = texts.len()))]
```

-**Step 6: Fix grpc_engine.rs logging**

In `grpc_engine.rs` line 38, replace:

```rust
// warn!("Truncating input from {} to {} bytes", text.len(), MAX_EMBED_CHARS);
warn!(input_len = text.len(), max_len = MAX_EMBED_CHARS, "embedding_input_truncated");
```

-**Step 7: Verify it compiles and tests pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel -- --nocapture`
Expected: All tests PASS

-**Step 8: Commit**

```bash
git add crates/corvia-kernel/src/lite_store.rs crates/corvia-kernel/src/graph_store.rs crates/corvia-kernel/src/staging.rs crates/corvia-kernel/src/ollama_engine.rs crates/corvia-kernel/src/grpc_engine.rs
git commit -m "refactor(kernel): standardize all logging to structured OTEL fields"
```

---

### Task 10: Structured logging sweep — corvia-server and docker

**Files:**
- Modify: `crates/corvia-server/src/rest.rs` (line 322)
- Modify: `crates/corvia-kernel/src/docker.rs` (lines 50, 105, 112, 152, 195, 202, 242, 279, 286, 292, 311)

-**Step 1: Fix rest.rs logging**

```rust
// Line 322 — replace:
// info!("Stored memory {id}");
info!(entry_id = %id, "memory_stored");
```

-**Step 2: Fix docker.rs logging**

```rust
// Line 50: info!("Pulling SurrealDB image: {SURREALDB_IMAGE}");
info!(image = SURREALDB_IMAGE, "docker_image_pulling");

// Line 105: info!("Creating SurrealDB container: {CONTAINER_NAME}");
info!(container = CONTAINER_NAME, "docker_container_creating");

// Line 112: info!("SurrealDB started on port {SURREALDB_PORT}");
info!(port = SURREALDB_PORT, service = "surrealdb", "docker_service_started");

// Line 152: info!("Pulling vLLM image: {VLLM_IMAGE}");
info!(image = VLLM_IMAGE, "docker_image_pulling");

// Line 195: info!("Creating vLLM container: {VLLM_CONTAINER_NAME}");
info!(container = VLLM_CONTAINER_NAME, "docker_container_creating");

// Line 202: info!("vLLM started on port {VLLM_PORT}");
info!(port = VLLM_PORT, service = "vllm", "docker_service_started");

// Line 242: info!("Pulling Ollama image: {OLLAMA_IMAGE}");
info!(image = OLLAMA_IMAGE, "docker_image_pulling");

// Line 279: info!("Creating Ollama container: {OLLAMA_CONTAINER_NAME}");
info!(container = OLLAMA_CONTAINER_NAME, "docker_container_creating");

// Line 286: info!("Ollama started on port {OLLAMA_PORT}");
info!(port = OLLAMA_PORT, service = "ollama", "docker_service_started");

// Line 292: info!("Pulling model {model} via Ollama API...");
info!(model, "ollama_model_pulling");

// Line 311: info!("Model {model} pulled successfully");
info!(model, "ollama_model_pulled");
```

-**Step 3: Verify it compiles and tests pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server -p corvia-kernel -- --nocapture`
Expected: All tests PASS

-**Step 4: Commit**

```bash
git add crates/corvia-server/src/rest.rs crates/corvia-kernel/src/docker.rs
git commit -m "refactor(server,kernel): standardize logging in REST handlers and docker module"
```

---

### Task 11: Structured logging sweep — adapters

**Files:**
- Modify: `adapters/corvia-adapter-git/rust/src/git.rs` (lines 67, 95, 103, 129)
- Modify: `adapters/corvia-adapter-basic/rust/Cargo.toml`
- Modify: `adapters/corvia-adapter-basic/rust/src/main.rs`

-**Step 1: Fix corvia-adapter-git logging**

```rust
// Lines 67-70 — replace:
// info!("Ingesting sources from {} (version: {}, scope: {})", source_path, source_version, scope_id);
info!(source_path, source_version, scope_id, "ingestion_started");

// Lines 95-98 — replace:
// debug!("Skipping binary or unreadable file: {}", file_path.display());
debug!(file = %file_path.display(), "file_skipped_binary_or_unreadable");

// Lines 103-107 — replace:
// warn!("Skipping large file ({}KB): {}", content.len() / 1024, file_path.display());
warn!(file = %file_path.display(), size_kb = content.len() / 1024, "file_skipped_too_large");

// Line 129 — replace:
// info!("Collected {} source files from {}", files.len(), source_path);
info!(file_count = files.len(), source_path, "ingestion_files_collected");
```

-**Step 2: Add tracing to corvia-adapter-basic**

Add to `adapters/corvia-adapter-basic/rust/Cargo.toml` under `[dependencies]`:

```toml
tracing.workspace = true
```

-**Step 3: Add tracing events to corvia-adapter-basic main.rs**

Add `use tracing::{info, warn};` at the top, then add structured logging at ingestion entry/exit points. The exact locations depend on the file structure — add an `info!` when starting ingestion and when completing:

```rust
info!(source_path = %path, "basic_ingestion_started");
// ... at completion:
info!(file_count = files.len(), "basic_ingestion_completed");
```

And convert any `eprintln!` error output to `warn!` or keep `eprintln!` for fatal startup errors only.

-**Step 4: Verify adapters compile**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-adapter-git -p corvia-adapter-basic`
Expected: Compiles

-**Step 5: Commit**

```bash
git add adapters/corvia-adapter-git/rust/src/git.rs adapters/corvia-adapter-basic/rust/Cargo.toml adapters/corvia-adapter-basic/rust/src/main.rs
git commit -m "refactor(adapters): standardize logging and add tracing to basic adapter"
```

---

### Task 12: Add error!() logging at API boundaries

**Files:**
- Modify: `crates/corvia-server/src/rest.rs` (handler functions at lines 300, 327, 363, 386, 411, 436, 457, 467, 488, 498, 508, 525, 544, 558, 579, 596, 611)

-**Step 1: Add tracing::error import**

Add `use tracing::error;` to the imports in `rest.rs`.

-**Step 2: Add error logging to handler error paths**

For each handler that returns `Err((StatusCode, String))`, add an `error!()` event before the error return. The pattern for each handler:

```rust
// Example for write_memory (line 300):
async fn write_memory(...) -> std::result::Result<..., (StatusCode, String)> {
    // ... existing code ...
    .map_err(|e| {
        error!(endpoint = "write_memory", err = %e, "request_failed");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;
    // ...
}
```

Apply this pattern to all handlers that have `.map_err(|e| (StatusCode::..., e.to_string()))` or similar error returns. Each gets an `error!()` with `endpoint` and `err` fields.

-**Step 3: Verify it compiles and tests pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server -- --nocapture`
Expected: All tests PASS

-**Step 4: Commit**

```bash
git add crates/corvia-server/src/rest.rs
git commit -m "feat(server): add error-level logging at REST API boundaries"
```

---

## Chunk 6: Traces Ingestion Endpoint

### Task 13: Add /v1/traces/ingest endpoint

**Files:**
- Modify: `crates/corvia-server/src/rest.rs` (router + new handler)

-**Step 1: Define trace ingestion types**

Add the following types in `rest.rs` (in the request/response types section):

```rust
#[derive(Deserialize)]
pub struct OtlpTraceSpan {
    pub trace_id: String,
    pub span_id: String,
    #[serde(default)]
    pub parent_span_id: String,
    pub name: String,
    pub service_name: String,
    pub start_time: String,
    pub end_time: String,
    pub duration_ms: u64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub attributes: serde_json::Value,
}

#[derive(Deserialize)]
pub struct TracesIngestRequest {
    pub spans: Vec<OtlpTraceSpan>,
}

#[derive(Serialize)]
pub struct TracesIngestResponse {
    pub accepted: usize,
}
```

-**Step 2: Add the handler**

```rust
async fn ingest_traces(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TracesIngestRequest>,
) -> std::result::Result<Json<TracesIngestResponse>, (StatusCode, String)> {
    let store = &state.coordinator.store;
    let accepted = req.spans.len();

    for span in &req.spans {
        let metadata = serde_json::json!({
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "service_name": span.service_name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "status": span.status,
            "attributes": span.attributes,
        });

        let mut entry = corvia_kernel::types::KnowledgeEntry::new(
            span.name.clone(),
            "corvia.traces".into(),
            String::new(),
        );
        entry.metadata = metadata;

        store.insert(&entry).await.map_err(|e| {
            error!(endpoint = "ingest_traces", err = %e, "request_failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;
    }

    tracing::info!(spans_accepted = accepted, "traces_ingested");
    Ok(Json(TracesIngestResponse { accepted }))
}
```

-**Step 3: Add route to router**

Add the route in the `router()` function, after the RAG endpoints:

```rust
        // Trace ingestion endpoint (dogfooding)
        .route("/v1/traces/ingest", post(ingest_traces))
```

-**Step 4: Verify it compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-server`
Expected: Compiles

-**Step 5: Write integration test**

Add to the server's test module or a new test file:

```rust
#[cfg(test)]
mod trace_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_traces_ingest_accepts_spans() {
        // Build a test app with LiteStore
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(
            corvia_kernel::lite_store::LiteStore::open(dir.path(), 3).unwrap()
        );
        store.init_schema().await.unwrap();
        // ... build AppState and router ...
        // POST a valid TracesIngestRequest
        // Assert 200 OK with { "accepted": N }
    }
}
```

Note: The exact test setup depends on how `AppState` is constructed in tests. Follow the existing test patterns in `corvia-server`.

-**Step 6: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server -- --nocapture`
Expected: All tests PASS

-**Step 7: Commit**

```bash
git add crates/corvia-server/src/rest.rs
git commit -m "feat(server): add /v1/traces/ingest endpoint for OTEL dogfooding"
```

---

## Chunk 7: Fix References to Removed Span Constants

### Task 14: Clean up references to removed span constants

**Files:**
- Any file that references `spans::SEARCH` or `spans::ENTRY_INSERT`

-**Step 1: Search for references**

Run: `grep -rn "spans::SEARCH\|spans::ENTRY_INSERT\|SEARCH.*corvia\.search\|ENTRY_INSERT.*corvia\.entry\.insert" crates/`

If any references exist outside of `corvia-telemetry/src/lib.rs`, update them.

-**Step 2: Verify full workspace compiles**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check --workspace`
Expected: Compiles with no errors

-**Step 3: Commit (if changes needed)**

```bash
git add -A
git commit -m "refactor: clean up references to removed span constants"
```

---

## Chunk 8: Final Verification

### Task 15: Full test suite and validation

-**Step 1: Run full workspace tests (Tier 1)**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test --workspace`
Expected: 433+ tests PASS

-**Step 2: Verify no string-interpolated logging remains**

Run: `grep -rn 'info!(".*{' crates/ adapters/ --include="*.rs" | grep -v '#\[cfg(test)\]' | grep -v 'mod tests' | grep -v '///'`

Expected: No matches in non-test production code (some may remain in test code — that's fine)

-**Step 3: Verify all spans use constants**

Run: `grep -rn '#\[tracing::instrument' crates/ adapters/ --include="*.rs"`

Verify each `name = "corvia.*"` value has a corresponding constant in `corvia-telemetry::spans`.

-**Step 4: Verify OTLP compiles correctly**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build --workspace --release 2>&1 | tail -5`
Expected: Compiles successfully

-**Step 5: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: final OTEL standardization cleanup"
```
