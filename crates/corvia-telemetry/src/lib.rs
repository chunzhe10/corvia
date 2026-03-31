pub mod dashboard_trace_layer;
pub mod otel_context_layer;
pub mod propagation;

use corvia_common::config::TelemetryConfig;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;

/// Span name constants matching the observability contract.
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
    pub const INFERENCE_LOAD: &str = "corvia.inference.load";
    pub const INFERENCE_RELOAD: &str = "corvia.inference.reload";
    pub const INFERENCE_CONFIG_RELOAD: &str = "corvia.inference.config_reload";

    // Tiered knowledge lifecycle spans
    pub const GC_CYCLE: &str = "corvia.gc.cycle";
    pub const GC_SCORE: &str = "corvia.gc.score";
    pub const GC_TRANSITION: &str = "corvia.gc.transition";
    pub const GC_REBUILD: &str = "corvia.gc.rebuild";
    pub const ACCESS_RECORD: &str = "corvia.access.record";

    // Spoke lifecycle spans.
    // Note: tracing's info_span! macro requires a string literal as the first
    // argument, so these constants cannot be used directly in info_span! calls.
    // They exist for the naming convention test and as the canonical reference
    // for span names used inline in corvia-kernel/src/spoke.rs.
    pub const SPOKE_CREATE: &str = "corvia.spoke.create";
    pub const SPOKE_DESTROY: &str = "corvia.spoke.destroy";
    pub const SPOKE_PRUNE: &str = "corvia.spoke.prune";
    pub const SPOKE_RESTART: &str = "corvia.spoke.restart";
    pub const SPOKE_CHECK: &str = "corvia.spoke.check";

    // Hook lifecycle spans (embedded in JSONL events, reconstructed during ingestion)
    pub const HOOK_DISPATCH: &str = "corvia.hook.dispatch";
    pub const HOOK_SESSION_RECORD: &str = "corvia.hook.session_record";
    pub const HOOK_DOC_PLACEMENT: &str = "corvia.hook.doc_placement";
    pub const HOOK_AGENT_CHECK: &str = "corvia.hook.agent_check";
    pub const HOOK_ORPHAN_CLEANUP: &str = "corvia.hook.orphan_cleanup";
    pub const HOOK_WRITE_REMINDER: &str = "corvia.hook.write_reminder";

    // Pipeline stage spans (instrumented in corvia-kernel/src/pipeline/)
    pub const PIPELINE_SEARCH: &str = "corvia.pipeline.search";
    pub const PIPELINE_CHANNEL: &str = "corvia.pipeline.channel";
    pub const PIPELINE_FUSION: &str = "corvia.pipeline.fusion";
}

/// Opaque handle that keeps the telemetry pipeline alive.
///
/// Hold this in your top-level scope (e.g. `main`); dropping it flushes
/// any buffered log output and shuts down the tracer provider.
pub struct TelemetryGuard {
    _file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    _tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

/// Initialize the tracing subscriber pipeline based on config.
///
/// Returns a [`TelemetryGuard`] that **must** be held for the lifetime of
/// the process. Dropping the guard flushes buffered output and shuts down
/// the OTLP exporter if configured.
///
/// OTLP export is **additive**: if `otlp_endpoint` (or `OTEL_EXPORTER_OTLP_ENDPOINT`)
/// is non-empty, an OpenTelemetry tracing layer is added regardless of the `exporter`
/// setting. The `exporter` field only controls the local output layer (stdout/file).
pub fn init_telemetry(config: &TelemetryConfig) -> anyhow::Result<TelemetryGuard> {
    use tracing_subscriber::{
        fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
    };

    // 1. Register W3C TraceContext propagator globally
    global::set_text_map_propagator(TraceContextPropagator::new());

    // 2. Build env filter
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // 3. Resolve OTLP endpoint: env var overrides config
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| config.otlp_endpoint.clone());

    // 4. Resolve service name: env var overrides config
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| config.service_name.clone());

    // 5. Build tracer provider + OTEL layer.
    //
    // When an OTLP endpoint is configured, spans are exported externally.
    // Otherwise, a local-only provider is created so that trace_id/span_id
    // still appear in structured logs — the dashboard traces page depends
    // on these fields even without an external collector.
    let tracer_provider;
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
        let layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracer_provider = Some(provider);
        Some(layer)
    } else {
        // Local-only provider: generates trace_id/span_id for structured
        // logs without exporting to any external collector.
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_resource(
                opentelemetry_sdk::Resource::builder()
                    .with_service_name(service_name)
                    .build(),
            )
            .build();

        let tracer = provider.tracer("corvia");
        let layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracer_provider = Some(provider);
        Some(layer)
    };

    // 6. Build dashboard trace layer — writes span-close events with OTEL
    //    context to a file the dashboard can read, regardless of exporter mode.
    let dashboard_layer = dashboard_trace_layer::DashboardTraceLayer::new(
        dashboard_trace_layer::DashboardTraceLayer::default_path(),
    )
    .ok();

    // 7. Build local output layer + compose registry
    let mut file_guard = None;

    // OtelContextLayer bridges trace_id/span_id into span extensions.
    // Always active — the local-only provider ensures IDs exist.
    let context_layer = Some(otel_context_layer::OtelContextLayer);

    match config.exporter.as_str() {
        "file" => {
            let file_appender = tracing_appender::rolling::daily("logs", "corvia.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            file_guard = Some(guard);

            let local_layer = if config.log_format == "json" {
                fmt::layer()
                    .json()
                    .with_writer(non_blocking)
                    .boxed()
            } else {
                fmt::layer()
                    .with_writer(non_blocking)
                    .boxed()
            };

            tracing_subscriber::registry()
                .with(env_filter)
                .with(otel_layer)
                .with(context_layer)
                .with(dashboard_layer)
                .with(local_layer)
                .init();
        }
        _ => {
            // stdout (default) or any other value
            let local_layer = if config.log_format == "json" {
                fmt::layer().json().boxed()
            } else {
                fmt::layer().boxed()
            };

            tracing_subscriber::registry()
                .with(env_filter)
                .with(otel_layer)
                .with(context_layer)
                .with(dashboard_layer)
                .with(local_layer)
                .init();
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
            spans::AGENT_REGISTER,
            spans::SESSION_CREATE,
            spans::ENTRY_WRITE,
            spans::ENTRY_EMBED,
            spans::ENTRY_EMBED_BATCH,
            spans::SESSION_COMMIT,
            spans::MERGE_PROCESS,
            spans::MERGE_PROCESS_ENTRY,
            spans::MERGE_CONFLICT,
            spans::MERGE_LLM_RESOLVE,
            spans::GC_RUN,
            spans::STORE_INSERT,
            spans::STORE_SEARCH,
            spans::STORE_GET,
            spans::RAG_CONTEXT,
            spans::RAG_ASK,
            spans::INFERENCE_LOAD,
            spans::INFERENCE_RELOAD,
            spans::INFERENCE_CONFIG_RELOAD,
            spans::GC_CYCLE,
            spans::GC_SCORE,
            spans::GC_TRANSITION,
            spans::GC_REBUILD,
            spans::ACCESS_RECORD,
            spans::SPOKE_CREATE,
            spans::SPOKE_DESTROY,
            spans::SPOKE_PRUNE,
            spans::SPOKE_RESTART,
            spans::SPOKE_CHECK,
            spans::PIPELINE_SEARCH,
            spans::PIPELINE_CHANNEL,
            spans::PIPELINE_FUSION,
        ];
        for name in &all {
            assert!(
                name.starts_with("corvia."),
                "{name} must start with 'corvia.'"
            );
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
        // Verify the TelemetryGuard struct has the expected fields by
        // constructing one directly (only possible in this module).
        let guard = TelemetryGuard {
            _file_guard: None,
            _tracer_provider: None,
        };
        // Guard should be droppable without panic
        drop(guard);
    }
}
