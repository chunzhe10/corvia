//! Telemetry initialization: always-on OpenTelemetry + optional OTLP file exporter + optional gRPC export.
//!
//! OpenTelemetry instrumentation is always active. A TracerProvider is created
//! unconditionally so spans always have trace_id and span_id.
//!
//! When a `trace_file` path is supplied, the OTLP file exporter writes spans
//! as OTLP JSON lines to that absolute path. When `None`, the file exporter
//! is disabled — callers are expected to resolve the project root and pass an
//! absolute path, rather than let the exporter fall back to a cwd-relative
//! `.corvia/traces.jsonl` (which historically created stray `.corvia/` dirs).
//!
//! When `--otlp-endpoint` is provided or `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
//! spans are additionally exported via gRPC to the given collector.

use std::path::Path;

use anyhow::{Context, Result};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use corvia_core::trace::OtlpFileExporter;

/// Opaque handle that keeps the telemetry pipeline alive.
///
/// Hold this in your top-level scope (e.g. `main`). Dropping the guard
/// shuts down the OpenTelemetry provider, flushing any buffered spans.
pub struct TelemetryGuard {
    _provider: SdkTracerProvider,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Err(e) = self._provider.shutdown() {
            eprintln!("telemetry shutdown error: {e}");
        }
    }
}

/// Initialize the tracing/telemetry stack.
///
/// OpenTelemetry is always active. The TracerProvider has:
/// 1. File exporter (always): writes OTLP JSON lines to `trace_file` path
/// 2. gRPC exporter (optional): if `otlp_endpoint` is provided or env var is set
///
/// The subscriber stack has:
/// 1. `fmt` layer (always, to stderr)
/// 2. `tracing-opentelemetry` bridge (always, for trace_id/span_id propagation)
///
/// Both exporters receive the same span data in the same OTLP format via the
/// TracerProvider pipeline.
///
/// Returns a [`TelemetryGuard`] that must be held for the lifetime of the process.
pub fn init_telemetry(
    otlp_endpoint: Option<&str>,
    trace_file: Option<&Path>,
) -> Result<TelemetryGuard> {
    let endpoint = otlp_endpoint
        .map(String::from)
        .or_else(|| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok());

    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into());

    // Build the OpenTelemetry TracerProvider.
    let mut provider_builder = SdkTracerProvider::builder().with_resource(
        opentelemetry_sdk::Resource::builder()
            .with_service_name("corvia")
            .build(),
    );

    // Optional: file exporter (OTLP JSON). No cwd-relative default — if the
    // caller doesn't supply a path, the file exporter is simply not attached.
    // Uses simple_exporter (sync) so traces flush immediately on span close.
    if let Some(path) = trace_file {
        if let Ok(file_exporter) = OtlpFileExporter::new(path.to_path_buf()) {
            provider_builder = provider_builder.with_simple_exporter(file_exporter);
        }
    }

    // Optional: gRPC exporter for external collectors.
    if let Some(ref endpoint) = endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .context("failed to create OTLP span exporter")?;
        provider_builder = provider_builder.with_batch_exporter(exporter);
    }

    let provider = provider_builder.build();

    let tracer = provider.tracer("corvia");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    Ok(TelemetryGuard {
        _provider: provider,
    })
}
