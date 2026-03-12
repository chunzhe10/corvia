//! Dashboard REST API — `/api/dashboard/*` endpoints.

pub mod health;
pub mod traces;

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};

use corvia_common::dashboard::{
    DashboardConfig, DashboardStatusResponse, LogEntry, LogsResponse, TracesResponse,
};
use crate::rest::AppState;

/// Dashboard REST API router — mounts at /api/dashboard/*
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/dashboard/status", get(status_handler))
        .route("/api/dashboard/traces", get(traces_handler))
        .route("/api/dashboard/logs", get(logs_handler))
        .route("/api/dashboard/config", get(config_handler))
        .route("/api/dashboard/graph", get(graph_handler))
        .with_state(state)
}

/// Build a DashboardConfig summary from the live config.
fn config_summary(cfg: &corvia_common::config::CorviaConfig) -> DashboardConfig {
    DashboardConfig {
        embedding_provider: format!("{:?}", cfg.embedding.provider).to_lowercase(),
        merge_provider: cfg
            .merge
            .as_ref()
            .map(|m| format!("{:?}", m.provider).to_lowercase())
            .unwrap_or_else(|| "none".to_string()),
        storage: format!("{:?}", cfg.storage.store_type).to_lowercase(),
        workspace: cfg.project.name.clone(),
    }
}

/// GET /api/dashboard/status
/// Returns service health, store metrics, config summary, and optional traces.
async fn status_handler(
    State(state): State<Arc<AppState>>,
) -> Json<DashboardStatusResponse> {
    // Health check all services
    let services = health::check_all_services().await;

    // Store metrics
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or("corvia");

    let entry_count = state
        .store
        .count(scope_id)
        .await
        .unwrap_or(0);

    let agent_count = state
        .coordinator
        .registry
        .list_active()
        .map(|v| v.len())
        .unwrap_or(0);

    let session_count = state
        .coordinator
        .sessions
        .list_open()
        .map(|v| v.len())
        .unwrap_or(0);

    let merge_queue_depth = state
        .coordinator
        .merge_queue
        .depth()
        .unwrap_or(0);

    // Config summary
    let cfg = state.config.read().unwrap();
    let config = config_summary(&cfg);
    drop(cfg);

    // Traces from log files
    let log_dir = traces::log_dir();
    let traces_data = traces::collect_traces(&log_dir);
    let traces = if traces_data.spans.is_empty() && traces_data.recent_events.is_empty() {
        None
    } else {
        Some(traces_data)
    };

    Json(DashboardStatusResponse {
        services,
        entry_count,
        agent_count,
        merge_queue_depth,
        session_count,
        config,
        traces,
    })
}

/// GET /api/dashboard/traces
/// Returns span statistics and recent events from structured logs.
async fn traces_handler(
    State(_state): State<Arc<AppState>>,
) -> Json<TracesResponse> {
    let log_dir = traces::log_dir();
    let data = traces::collect_traces(&log_dir);

    Json(TracesResponse {
        spans: data.spans,
        recent_events: data.recent_events,
    })
}

/// Query params for /api/dashboard/logs
#[derive(Debug, serde::Deserialize)]
pub struct LogsQuery {
    /// Filter by service name (log file stem)
    pub service: Option<String>,
    /// Filter by module (agent, entry, merge, etc.)
    pub module: Option<String>,
    /// Filter by level (info, warn, error, debug)
    pub level: Option<String>,
    /// Max entries to return (default 100)
    pub limit: Option<usize>,
}

/// GET /api/dashboard/logs
/// Returns filtered structured log entries.
async fn logs_handler(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<LogsQuery>,
) -> Json<LogsResponse> {
    let log_dir_path = traces::log_dir();
    let limit = params.limit.unwrap_or(100);

    let mut entries: Vec<LogEntry> = Vec::new();

    if let Ok(dir_entries) = std::fs::read_dir(&log_dir_path) {
        for entry in dir_entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext == "log") {
                continue;
            }

            // Filter by service (file stem)
            if let Some(ref svc) = params.service {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem != svc.as_str() {
                        continue;
                    }
                }
            }

            let lines = traces::tail_lines(&path, 500);
            for line in &lines {
                let parsed = match traces::parse_trace_line(line) {
                    Some(p) => p,
                    None => continue,
                };

                let (timestamp, level, module, message) = match parsed {
                    traces::ParsedTrace::Span {
                        timestamp, level, span_name, elapsed_ms,
                    } => {
                        let module = traces::span_to_module(&span_name);
                        let msg = format!("{span_name} ({elapsed_ms:.1}ms)");
                        (timestamp, level, module.to_string(), msg)
                    }
                    traces::ParsedTrace::Event {
                        timestamp, level, msg, target,
                    } => {
                        let module = traces::target_to_module(&target);
                        (timestamp, level, module.to_string(), msg)
                    }
                };

                let norm_level = traces::normalize_level(&level);

                // Apply filters
                if let Some(ref filter_module) = params.module {
                    if module != *filter_module {
                        continue;
                    }
                }
                if let Some(ref filter_level) = params.level {
                    if norm_level != filter_level.as_str() {
                        continue;
                    }
                }

                entries.push(LogEntry {
                    timestamp: traces::short_timestamp(&timestamp),
                    level: norm_level.to_string(),
                    module,
                    message,
                });
            }
        }
    }

    // Sort by timestamp and limit
    entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    let total = entries.len();
    entries.truncate(limit);

    Json(LogsResponse { entries, total })
}

/// GET /api/dashboard/config
/// Returns current server config summary (read-only).
async fn config_handler(
    State(state): State<Arc<AppState>>,
) -> Json<DashboardConfig> {
    let cfg = state.config.read().unwrap();
    Json(config_summary(&cfg))
}

/// Query params for /api/dashboard/graph
#[derive(Debug, serde::Deserialize)]
pub struct GraphQuery {
    pub scope: Option<String>,
    pub entry_id: Option<String>,
}

/// GET /api/dashboard/graph
/// Returns knowledge graph edges. Thin wrapper around kernel GraphStore.
async fn graph_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GraphQuery>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let entry_id = match params.entry_id {
        Some(id) => id
            .parse::<uuid::Uuid>()
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("Invalid entry_id: {e}")))?,
        None => {
            return Ok(Json(serde_json::json!({ "edges": [] })));
        }
    };

    let edges = state
        .graph
        .edges(&entry_id, corvia_common::types::EdgeDirection::Both)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let edge_dtos: Vec<serde_json::Value> = edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "from": e.from.to_string(),
                "to": e.to.to_string(),
                "relation": e.relation,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "edges": edge_dtos })))
}
