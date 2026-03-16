# corvia-telemetry

Structured tracing initialization and observability utilities for corvia.

## Features

- **Telemetry initialization** from `TelemetryConfig` (stdout, file, OTLP exporters)
- **Span name constants** — canonical span names shared across all crates
- **OpenTelemetry integration** — OTLP gRPC export, W3C trace context propagation
- **Configurable exporters** — stdout (default), JSON file, OTLP endpoint

## Configuration

Configured via `[telemetry]` in `corvia.toml`:

```toml
[telemetry]
exporter = "stdout"        # stdout | file | otlp
otlp_endpoint = ""         # gRPC endpoint for OTLP
log_format = "json"        # json | pretty
metrics_enabled = true
```

## Usage

```rust
use corvia_telemetry::init_telemetry;

let config = TelemetryConfig::default();
let _guard = init_telemetry(&config)?;
```

## License

[AGPL-3.0-only](../../../LICENSE)
