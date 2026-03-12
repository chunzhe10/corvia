//! Dashboard REST API — `/api/dashboard/*` endpoints.

pub mod health;
pub mod traces;

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};

use corvia_common::dashboard::{
    DashboardConfig, DashboardStatusResponse, LogEntry, LogsResponse, ModuleStats, TracesResponse,
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
        .route("/api/dashboard/agents", get(agents_handler))
        .route("/api/dashboard/agents/{agent_id}/sessions", get(agent_sessions_handler))
        .route("/api/dashboard/merge/queue", get(merge_queue_handler))
        .route("/api/dashboard/merge/retry", post(merge_retry_handler))
        .route("/api/dashboard/health", get(health_handler))
        .route("/api/dashboard/rag/context", post(rag_context_handler))
        .route("/api/dashboard/rag/ask", post(rag_ask_handler))
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

    // Config summary (poisoned lock safe)
    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner());
    let config = config_summary(&cfg);
    drop(cfg);

    // Traces from log files (bounded: 500 lines/file, 50MB max file size)
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
/// Returns span statistics, recent events, and pre-aggregated module stats.
async fn traces_handler(
    State(_state): State<Arc<AppState>>,
) -> Json<TracesResponse> {
    let log_dir = traces::log_dir();
    let data = traces::collect_traces(&log_dir);

    // Pre-aggregate per-module stats server-side so the client doesn't
    // have to recompute this on every poll cycle.
    let mut modules = std::collections::HashMap::<String, ModuleStats>::new();
    for module in &["agent", "entry", "merge", "storage", "rag", "inference", "gc"] {
        modules.insert(module.to_string(), ModuleStats::default());
    }
    for (span_name, stats) in &data.spans {
        let module = traces::span_to_module(span_name);
        if let Some(ms) = modules.get_mut(module) {
            ms.count += stats.count;
            ms.count_1h += stats.count_1h;
            ms.avg_ms += stats.avg_ms * stats.count as f64;
            ms.errors += stats.errors;
            ms.span_count += 1;
        }
    }
    for ms in modules.values_mut() {
        if ms.count > 0 {
            ms.avg_ms = (ms.avg_ms / ms.count as f64).round();
        }
    }

    Json(TracesResponse {
        spans: data.spans,
        recent_events: data.recent_events,
        modules,
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
    let limit = params.limit.unwrap_or(100).min(1000);

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
    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner());
    Json(config_summary(&cfg))
}

/// Query params for /api/dashboard/graph
#[derive(Debug, serde::Deserialize)]
pub struct GraphQuery {
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

// ---------------------------------------------------------------------------
// Agent endpoints
// ---------------------------------------------------------------------------

/// GET /api/dashboard/agents
/// Returns all registered agents.
async fn agents_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let agents = corvia_kernel::ops::agents_list(&state.coordinator)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list agents: {e}")))?;

    Ok(Json(serde_json::json!(agents)))
}

/// GET /api/dashboard/agents/{agent_id}/sessions
/// Returns all sessions for a given agent.
async fn agent_sessions_handler(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sessions = corvia_kernel::ops::sessions_list(&state.coordinator, &agent_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list sessions: {e}")))?;

    Ok(Json(serde_json::json!(sessions)))
}

// ---------------------------------------------------------------------------
// Merge queue endpoints
// ---------------------------------------------------------------------------

/// Query params for /api/dashboard/merge/queue
#[derive(Debug, serde::Deserialize)]
pub struct MergeQueueQuery {
    /// Max entries to return (default 50)
    pub limit: Option<usize>,
}

/// GET /api/dashboard/merge/queue
/// Returns merge queue depth and entries.
async fn merge_queue_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MergeQueueQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let limit = params.limit.unwrap_or(50);
    let status = corvia_kernel::ops::merge_queue_status(&state.coordinator, limit)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get merge queue: {e}")))?;

    Ok(Json(serde_json::json!(status)))
}

/// Request body for /api/dashboard/merge/retry
#[derive(Debug, serde::Deserialize)]
pub struct MergeRetryRequest {
    pub entry_ids: Vec<String>,
}

/// POST /api/dashboard/merge/retry
/// Retry failed merge queue entries.
async fn merge_retry_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MergeRetryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let ids: Vec<uuid::Uuid> = req
        .entry_ids
        .iter()
        .map(|s| s.parse::<uuid::Uuid>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;

    let retried = corvia_kernel::ops::merge_retry(&state.coordinator, &ids)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Merge retry failed: {e}")))?;

    Ok(Json(serde_json::json!({ "retried": retried })))
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------

/// Query params for /api/dashboard/health
#[derive(Debug, serde::Deserialize)]
pub struct HealthQuery {
    /// Optional single check to run (e.g. "stale", "broken", "orphan", "dangling", "cycle")
    pub check: Option<String>,
}

/// GET /api/dashboard/health
/// Run reasoner health checks on the default scope.
async fn health_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HealthQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or("corvia");

    // Load entries from knowledge files (direct disk read)
    let entries = corvia_kernel::knowledge_files::read_scope(&state.data_dir, scope_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load entries: {e}")))?;

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*state.store, &*state.graph);

    let findings = if let Some(ref check_str) = params.check {
        let check_type = check_str.parse::<corvia_kernel::reasoner::CheckType>()
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        reasoner.run_check(&entries, scope_id, check_type).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Health check failed: {e}")))?
    } else {
        reasoner.run_all(&entries, scope_id).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Health check failed: {e}")))?
    };

    let items: Vec<serde_json::Value> = findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "check_type": f.check_type.as_str(),
                "confidence": f.confidence,
                "rationale": f.rationale,
                "target_ids": f.target_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "scope_id": scope_id,
        "findings": items,
        "count": items.len(),
    })))
}

// ---------------------------------------------------------------------------
// RAG endpoint
// ---------------------------------------------------------------------------

/// Request body for /api/dashboard/rag/ask
#[derive(Debug, serde::Deserialize)]
pub struct RagAskRequest {
    pub query: String,
    pub scope_id: String,
}

/// POST /api/dashboard/rag/context
/// Thin wrapper around the RAG pipeline context-only endpoint.
async fn rag_context_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagAskRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let rag = state.rag.as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit: 10,
        expand_graph: true,
        ..Default::default()
    };

    let response = rag.context(&req.query, &req.scope_id, Some(opts)).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("RAG context failed: {e}")))?;

    let sources: Vec<serde_json::Value> = response.context.sources.iter().map(|s| {
        serde_json::json!({
            "content": s.entry.content,
            "score": s.score,
            "source_file": s.entry.metadata.source_file,
            "language": s.entry.metadata.language,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "sources": sources,
        "trace": response.trace,
    })))
}

/// POST /api/dashboard/rag/ask
/// Thin wrapper around the RAG pipeline ask endpoint.
async fn rag_ask_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagAskRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let rag = state.rag.as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit: 10,
        expand_graph: true,
        ..Default::default()
    };

    let response = rag.ask(&req.query, &req.scope_id, Some(opts)).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("RAG failed: {e}")))?;

    let sources: Vec<serde_json::Value> = response.context.sources.iter().map(|s| {
        serde_json::json!({
            "content": s.entry.content,
            "score": s.score,
            "source_file": s.entry.metadata.source_file,
            "language": s.entry.metadata.language,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "answer": response.answer,
        "sources": sources,
        "trace": response.trace,
    })))
}
