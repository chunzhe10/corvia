pub mod propagation;

use corvia_common::config::TelemetryConfig;

/// Span name constants matching the D45 observability contract.
pub mod spans {
    pub const AGENT_REGISTER: &str = "corvia.agent.register";
    pub const SESSION_CREATE: &str = "corvia.session.create";
    pub const ENTRY_WRITE: &str = "corvia.entry.write";
    pub const ENTRY_EMBED: &str = "corvia.entry.embed";
    pub const ENTRY_INSERT: &str = "corvia.entry.insert";
    pub const SESSION_COMMIT: &str = "corvia.session.commit";
    pub const MERGE_PROCESS: &str = "corvia.merge.process";
    pub const MERGE_PROCESS_ENTRY: &str = "corvia.merge.process_entry";
    pub const MERGE_CONFLICT: &str = "corvia.merge.conflict";
    pub const MERGE_LLM_RESOLVE: &str = "corvia.merge.llm_resolve";
    pub const GC_RUN: &str = "corvia.gc.run";
    pub const SEARCH: &str = "corvia.search";
    pub const STORE_INSERT: &str = "corvia.store.insert";
    pub const STORE_SEARCH: &str = "corvia.store.search";
    pub const STORE_GET: &str = "corvia.store.get";
    pub const RAG_CONTEXT: &str = "corvia.rag.context";
    pub const RAG_ASK: &str = "corvia.rag.ask";
}

/// Opaque handle that keeps the telemetry pipeline alive.
/// Hold this in your top-level scope (e.g. `main`); dropping it flushes
/// any buffered log output before the process exits.
pub struct TelemetryGuard {
    // The NonBlocking writer's WorkerGuard. When dropped, it flushes
    // pending writes. Only `Some` when exporter = "file".
    _file_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
}

/// Initialize the tracing subscriber pipeline based on config.
///
/// Returns a [`TelemetryGuard`] that **must** be held for the lifetime of
/// the process. Dropping the guard flushes buffered file output.
pub fn init_telemetry(config: &TelemetryConfig) -> anyhow::Result<TelemetryGuard> {
    use tracing_subscriber::{fmt, EnvFilter, prelude::*};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let mut file_guard = None;

    match config.exporter.as_str() {
        "file" => {
            let file_appender = tracing_appender::rolling::daily("logs", "corvia.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            file_guard = Some(guard);

            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json().with_writer(non_blocking))
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().with_writer(non_blocking))
                    .init();
            }
        }
        "otlp" => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer())
                .init();
            tracing::warn!("OTLP exporter configured but not yet implemented; falling back to stdout");
        }
        _ => {
            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json())
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer())
                    .init();
            }
        }
    }

    Ok(TelemetryGuard { _file_guard: file_guard })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_constants_are_dotted() {
        let all = [
            spans::AGENT_REGISTER, spans::SESSION_CREATE, spans::ENTRY_WRITE,
            spans::ENTRY_EMBED, spans::ENTRY_INSERT, spans::SESSION_COMMIT,
            spans::MERGE_PROCESS, spans::MERGE_PROCESS_ENTRY,
            spans::MERGE_CONFLICT, spans::MERGE_LLM_RESOLVE,
            spans::GC_RUN, spans::SEARCH, spans::STORE_INSERT, spans::STORE_SEARCH,
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
    }
}
