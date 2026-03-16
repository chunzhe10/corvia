# OpenTelemetry Standardization — Corvia Core

**Date:** 2026-03-11
**Status:** Shipped (v0.4.0)
**Scope:** corvia core (all crates in `repos/corvia/`)

## Context

Corvia uses the `tracing` crate with a centralized `corvia-telemetry` crate that
defines span constants and initializes the subscriber. The foundation is solid but
has gaps: no actual OpenTelemetry SDK integration, no OTLP export, missing HTTP
middleware traces, inconsistent logging patterns, and unused span constants.

This design standardizes all logging and tracing to OTEL format across the full
corvia core codebase in a single coordinated pass.

## Decisions

- **OTLP transport:** gRPC (preferred over HTTP)
- **Trace propagation:** Full W3C `traceparent`/`tracestate` between server ↔ inference
- **Collector:** External OTEL Collector managed by operator; SDK-side only (no bundled stack)
- **Dogfooding:** Traces routed back into corvia's knowledge store via OTEL Collector → `POST /v1/traces/ingest`
- **Approach:** Single-pass full rewrite (not phased)
- **Propagation adapters:** Vendored (~30 lines) in `corvia-telemetry::propagation`, not external crate

## Section 1: `corvia-telemetry` Rewrite — OTLP + OpenTelemetry SDK

The `corvia-telemetry` crate becomes the single telemetry authority for all corvia
services. The OTLP stub is replaced with a real exporter.

### New workspace dependencies (pinned for compatibility)

```toml
opentelemetry = "0.28"
opentelemetry_sdk = { version = "0.28", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.28", features = ["grpc-tonic"] }
tracing-opentelemetry = "0.29"
```

### `init_telemetry()` changes

- Adds an `OpenTelemetryLayer` to the `tracing_subscriber::registry()` stack when
  `otlp_endpoint` is non-empty
- Registers `TraceContextPropagator` globally for W3C propagation
- Sets `service.name` resource attribute from `TelemetryConfig.service_name`
- Returns an extended `TelemetryGuard` that holds the OTel `TracerProvider`
  (flushing on drop)
- Stdout/file exporters continue unchanged — OTLP is additive (dual-export)

### Span constants

Remove unused `SEARCH` and `ENTRY_INSERT` constants. `STORE_SEARCH` and
`STORE_INSERT` are the correct granularity — no unnecessary nesting.

## Section 2: W3C Trace Propagation — Server ↔ Inference

### Client side (`grpc_engine.rs`)

A vendored `MetadataInjector` struct in `corvia-telemetry::propagation` that
implements `opentelemetry::propagation::Injector` for tonic's `MetadataMap`.
Before each gRPC call in `embed()` and `embed_batch()`, inject the current
span's context into request metadata.

### Server side (`corvia-inference/src/main.rs`)

A vendored `MetadataExtractor` struct in `corvia-telemetry::propagation` that
implements `opentelemetry::propagation::Extractor`. Added as a tonic interceptor
on `Server::builder()`. Extracts `traceparent`/`tracestate` from gRPC metadata
and attaches the parent context.

### Inference telemetry unification

Replace the direct `tracing_subscriber::fmt()` call in `corvia-inference/src/main.rs`
with `corvia_telemetry::init_telemetry()`, passing `service_name: "corvia-inference"`.
Unifies filter defaults, format config, and OTLP export.

## Section 3: HTTP Request Tracing — Axum Middleware

Wire `tower_http::trace::TraceLayer` into the Axum router in both `rest.rs` and
`mcp.rs`. The `trace` feature is already enabled in `corvia-server/Cargo.toml`.

`TraceLayer::new_for_http()` produces spans with `http.method`, `http.uri`,
`http.status_code` following OTEL HTTP semantic conventions out of the box.

The MCP `mcp-session-id` header remains as application-level correlation alongside
W3C trace context.

## Section 4: Logging Standardization Sweep

### 4a. Structured fields everywhere

All `info!()`, `warn!()`, `debug!()` calls converted from string interpolation
to structured key-value fields:

```rust
// Before:
info!("Model {model} already available");

// After:
info!(model = %model, "model_already_available");
```

Event message convention: `snake_case` event names as message string. All
contextual data in structured fields, never in the message string.

### 4b. `error!()` events at API boundaries

Add `error!()` events in REST/MCP handlers when returning error responses:

```rust
error!(endpoint = "write_memory", status = %status, err = %e, "request_failed");
```

Keep kernel clean — errors propagate via `Result`, handlers log them.

### 4c. Remove unused span constants

- `SEARCH` — removed (`STORE_SEARCH` covers it)
- `ENTRY_INSERT` — removed (`STORE_INSERT` covers it)

### 4d. `embed_batch()` instrumentation

Add `#[tracing::instrument]` to `grpc_engine.rs:embed_batch()` and
`ollama_engine.rs` equivalent with `batch_size` field.

### 4e. Adapter parity

Add `tracing` dependency to `corvia-adapter-basic` and instrument its ingestion
path with entry/exit events matching `corvia-adapter-git`.

## Section 5: Dogfooding — Trace Ingestion via OTEL Collector

### Architecture

```
corvia-server ──OTLP/gRPC──→ OTEL Collector ──→ corvia (knowledge store)
corvia-inference ──OTLP/gRPC──→      ↑              ↓
                                     └── also: Jaeger/Tempo/etc.
```

### What we build

- `POST /v1/traces/ingest` endpoint in `corvia-server` accepting OTLP JSON-encoded
  trace data, writing spans as knowledge entries in `corvia.traces` scope
- The OTEL Collector is configured by the operator with an `otlphttp` exporter
  pointing at this endpoint

### Trace entry schema

```json
{
    "content": "corvia.entry.write",
    "scope_id": "corvia.traces",
    "metadata": {
        "trace_id": "abc123...",
        "span_id": "def456...",
        "parent_span_id": "...",
        "service_name": "corvia-server",
        "start_time": "2026-03-11T...",
        "end_time": "2026-03-11T...",
        "duration_ms": 42,
        "status": "ok",
        "attributes": {}
    }
}
```

### What we do NOT build

- No custom OTEL Collector exporter plugin
- No embedded collector inside corvia
- No Traces page frontend migration (separate task)

## Section 6: Configuration & Defaults

### Updated `TelemetryConfig`

```rust
pub struct TelemetryConfig {
    pub exporter: String,       // "stdout" (default) | "file" | "otlp"
    pub otlp_endpoint: String,  // e.g. "http://localhost:4317"
    pub otlp_protocol: String,  // "grpc" (default) | "http"
    pub service_name: String,   // "corvia-server" (default) | "corvia-inference"
    pub log_format: String,     // "text" (default) | "json"
    pub metrics_enabled: bool,  // true (default)
}
```

### Dual-output behavior

| `exporter` value | Local output | OTLP export |
|------------------|-------------|-------------|
| `"stdout"` | Console (text or json) | Only if `otlp_endpoint` set |
| `"file"` | Rolling file + console | Only if `otlp_endpoint` set |
| `"otlp"` | Console (fallback) | Always |

OTLP export is additive. `otlp_endpoint` enables it independently of `exporter`.

### Environment variable overrides (OTEL conventions)

- `OTEL_EXPORTER_OTLP_ENDPOINT` → overrides `otlp_endpoint`
- `OTEL_SERVICE_NAME` → overrides `service_name`
- `RUST_LOG` → already works via `EnvFilter`

## Section 7: Testing Strategy

### Unit tests (`corvia-telemetry`)

- `test_span_constants_are_dotted` — updated for removed constants
- `test_init_telemetry_stdout` — subscriber initializes without panic
- `test_init_telemetry_with_otlp_endpoint` — OTLP layer added when endpoint set
- `test_propagator_registered` — `TraceContextPropagator` is global after init
- `test_metadata_injector_extractor_roundtrip` — inject/extract preserves `traceparent`

### Unit tests (`corvia-server`)

- `test_rest_router_has_trace_layer` — request through router emits a span

### Integration tests

- `test_trace_propagation_grpc` — Tier 2+, verify shared trace ID across services
  (auto-skips when inference unreachable)
- `test_traces_ingest_endpoint` — POST OTLP JSON to `/v1/traces/ingest`, verify
  entries in store

### Out of scope

- No OTEL Collector routing tests (operator config)
- No load/performance tests for trace ingestion

## Files Touched (~15)

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | 4 new OpenTelemetry deps |
| `corvia-telemetry/Cargo.toml` | Add OTel deps |
| `corvia-telemetry/src/lib.rs` | Rewrite init, remove unused constants |
| `corvia-telemetry/src/propagation.rs` | New: `MetadataInjector`/`MetadataExtractor` |
| `corvia-common/src/config.rs` | Add `service_name`, `otlp_protocol` to `TelemetryConfig` |
| `corvia-kernel/src/grpc_engine.rs` | Inject W3C context before gRPC calls |
| `corvia-inference/src/main.rs` | Use `init_telemetry()`, add tonic interceptor |
| `corvia-inference/Cargo.toml` | Add `corvia-telemetry`, `corvia-common` deps |
| `corvia-server/src/rest.rs` | Add `TraceLayer`, error logging, traces ingest endpoint |
| `corvia-server/src/mcp.rs` | Add `TraceLayer` |
| `corvia-adapter-basic/Cargo.toml` | Add `tracing` dep |
| `corvia-adapter-basic/src/*.rs` | Add ingestion tracing events |
| ~10 kernel/adapter files | Structured field logging sweep |

## Gotchas

1. **Version pinning:** `tracing-opentelemetry` 0.29 requires `opentelemetry` 0.28.
   Pin all versions explicitly in workspace `Cargo.toml`.
2. **Traces API beta:** opentelemetry-rust traces are still beta. Pin, don't auto-upgrade.
3. **Async context guards:** `Context::attach()` guards must survive `.await` points.
   `#[instrument]` handles this; manual `attach()` in interceptors needs care.
4. **OTLP timeout units:** Changed from seconds to milliseconds in opentelemetry 0.29.
