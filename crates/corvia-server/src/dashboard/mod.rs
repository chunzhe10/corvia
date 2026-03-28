//! Dashboard REST API — `/api/dashboard/*` endpoints.

pub mod activity;
pub mod clustering;
pub mod coverage;
pub mod gpu;
pub mod health;
pub mod session_watcher;
pub mod traces;

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use corvia_common::constants::{DEFAULT_SCOPE_ID, RAG_ASK_TIMEOUT_SECS};
use corvia_common::dashboard::{
    DashboardConfig, DashboardStatusResponse, LogEntry, LogsResponse, ModuleStats, TracesResponse,
};
use corvia_kernel::agent_coordinator::GcReport;
use serde::{Deserialize, Serialize};
use crate::rest::AppState;

/// Helper: read knowledge files on a blocking thread to avoid starving the async runtime.
/// All dashboard handlers that need scope entries MUST use this instead of calling
/// `knowledge_files::read_scope` directly in async context.
async fn load_scope_entries(
    data_dir: &std::path::Path,
    scope_id: &str,
) -> Result<Vec<corvia_common::types::KnowledgeEntry>, (StatusCode, String)> {
    let data_dir = data_dir.to_path_buf();
    let scope_id = scope_id.to_string();
    tokio::task::spawn_blocking(move || {
        corvia_kernel::knowledge_files::read_scope(&data_dir, &scope_id)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join failed: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load entries: {e}")))
}

fn gc_report_to_dto(r: GcReport) -> corvia_common::dashboard::GcReportDto {
    corvia_common::dashboard::GcReportDto {
        orphans_rolled_back: r.orphans_rolled_back,
        duration_ms: r.duration_ms,
        stale_transitioned: r.stale_transitioned,
        closed_sessions_cleaned: r.closed_sessions_cleaned,
        agents_suspended: r.agents_suspended,
        entries_deduplicated: r.entries_deduplicated,
        started_at: r.started_at,
    }
}

/// Dashboard REST API router — mounts at /api/dashboard/*
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/dashboard/status", get(status_handler))
        .route("/api/dashboard/traces", get(traces_handler))
        .route("/api/dashboard/logs", get(logs_handler))
        .route("/api/dashboard/config", get(config_handler))
        .route("/api/dashboard/graph", get(graph_handler))
        .route("/api/dashboard/graph/scope", get(clustered_graph_handler))
        .route("/api/dashboard/agents", get(agents_handler))
        .route("/api/dashboard/agents/reconnectable", get(reconnectable_agents_handler))
        .route("/api/dashboard/agents/{agent_id}/sessions", get(agent_sessions_handler))
        .route("/api/dashboard/agents/{agent_id}/connect", post(connect_agent_handler))
        .route("/api/dashboard/agents/{agent_id}/refresh-summary", post(refresh_summary_handler))
        .route("/api/dashboard/merge/queue", get(merge_queue_handler))
        .route("/api/dashboard/merge/retry", post(merge_retry_handler))
        .route("/api/dashboard/health", get(health_handler))
        .route("/api/dashboard/rag/context", post(rag_context_handler))
        .route("/api/dashboard/rag/ask", post(rag_ask_handler))
        .route("/api/dashboard/entries/{entry_id}", get(entry_detail_handler))
        .route("/api/dashboard/entries/{entry_id}/history", get(entry_history_handler))
        .route("/api/dashboard/entries/{entry_id}/neighbors", get(entry_neighbors_handler))
        .route("/api/dashboard/activity", get(activity::activity_feed_handler))
        .route("/api/dashboard/gc", get(gc_status_handler))
        .route("/api/dashboard/gc/run", post(gc_run_handler))
        .route("/api/dashboard/tiers", get(tiers_handler))
        .route("/api/dashboard/tiers/transitions", get(tier_transitions_handler))
        .route("/api/dashboard/tiers/pinned", get(pinned_entries_handler))
        .route("/api/dashboard/tiers/gc-history", get(gc_knowledge_history_handler))
        .route("/api/dashboard/tiers/unpin/{entry_id}", post(dashboard_unpin_handler))
        .route("/api/dashboard/sessions/live", get(live_sessions_handler))
        .route("/api/dashboard/sessions/hook", get(hook_sessions_handler))
        .route("/api/dashboard/sessions/hook/stream", get(hook_sessions_stream))
        .route("/api/dashboard/traces/recent", get(recent_traces_handler))
        .route("/api/dashboard/gpu", get(gpu_status_handler))
        .route("/api/dashboard/status/refresh-coverage", post(refresh_coverage_handler))
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
        .unwrap_or(DEFAULT_SCOPE_ID);

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

    // Index coverage (sync read from cache — refreshed at startup and via refresh endpoint)
    let cov = state.coverage_cache.get_cached();

    Json(DashboardStatusResponse {
        services,
        entry_count,
        agent_count,
        merge_queue_depth,
        session_count,
        config,
        traces,
        index_coverage: cov.coverage,
        index_stale: cov.stale,
        index_disk_count: cov.disk_count,
        index_store_count: cov.store_count,
        index_hnsw_count: cov.hnsw_count,
        index_stale_threshold: cov.threshold,
        index_coverage_checked_at: cov.checked_at,
    })
}

/// POST /api/dashboard/status/refresh-coverage
/// Force-recompute coverage and return fresh snapshot.
async fn refresh_coverage_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::CoverageResponse> {
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or(DEFAULT_SCOPE_ID);

    let snapshot = state.coverage_cache
        .refresh(&state.data_dir, scope_id, &*state.store)
        .await;

    Json(snapshot.into())
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
            if path.extension().is_none_or(|ext| ext != "log") {
                continue;
            }

            // Filter by service (file stem)
            if let Some(ref svc) = params.service
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && stem != svc.as_str() {
                        continue;
                    }

            let lines = traces::tail_lines(&path, 500);
            for line in &lines {
                let parsed = match traces::parse_trace_line(line) {
                    Some(p) => p,
                    None => continue,
                };

                let (timestamp, level, module, message) = match parsed {
                    traces::ParsedTrace::Span {
                        timestamp, level, span_name, elapsed_ms, ..
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
                if let Some(ref filter_module) = params.module
                    && module != *filter_module {
                        continue;
                    }
                if let Some(ref filter_level) = params.level
                    && norm_level != filter_level.as_str() {
                        continue;
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

/// GET /api/dashboard/graph/scope
/// Returns all nodes and edges for the default scope's knowledge graph.
/// Uses spawn_blocking for the file I/O to avoid blocking the async runtime.
async fn graph_scope_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or(DEFAULT_SCOPE_ID)
        .to_string();

    let entries = load_scope_entries(&state.data_dir, &scope_id).await?;

    // Collect edges for every entry, deduplicating by from:relation:to
    let mut seen = std::collections::HashSet::<String>::new();
    let mut all_edges = Vec::new();
    let mut node_ids = std::collections::HashSet::<uuid::Uuid>::new();

    for entry in &entries {
        let edges = state
            .graph
            .edges(&entry.id, corvia_common::types::EdgeDirection::Both)
            .await
            .unwrap_or_default();

        for e in edges {
            let key = format!("{}:{}:{}", e.from, e.relation, e.to);
            if seen.insert(key) {
                node_ids.insert(e.from);
                node_ids.insert(e.to);
                all_edges.push(serde_json::json!({
                    "from": e.from.to_string(),
                    "relation": e.relation,
                    "to": e.to.to_string(),
                    "weight": e.metadata.as_ref()
                        .and_then(|m| m.get("weight"))
                        .and_then(|w| w.as_f64()),
                }));
            }
        }
    }

    // Build nodes with content previews
    let entry_map: std::collections::HashMap<uuid::Uuid, &corvia_common::types::KnowledgeEntry> =
        entries.iter().map(|e| (e.id, e)).collect();

    // Truncate a string at a char boundary, appending "…" if shortened.
    fn truncate_str(s: &str, max: usize) -> String {
        if s.len() <= max { return s.to_string(); }
        let end = s.floor_char_boundary(max);
        format!("{}…", &s[..end])
    }

    let nodes: Vec<serde_json::Value> = node_ids
        .iter()
        .map(|id| {
            let entry = entry_map.get(id);
            // Prefer source_file for the label, fall back to content preview
            let label = entry
                .and_then(|e| e.metadata.source_file.as_deref().map(|s| s.to_string()))
                .or_else(|| entry.map(|e| truncate_str(&e.content, 80)))
                .unwrap_or_else(|| truncate_str(&id.to_string(), 8));
            let preview = entry
                .map(|e| truncate_str(&e.content, 200))
                .unwrap_or_default();
            // Derive cluster group from source_file path
            let group = entry
                .and_then(|e| e.metadata.source_file.as_deref())
                .map(|sf| {
                    let parts: Vec<&str> = sf.split('/').collect();
                    if (parts.first() == Some(&"crates") || parts.first() == Some(&"adapters"))
                        && parts.len() >= 2
                    {
                        parts[1].to_string()
                    } else if parts.first() == Some(&"docs") {
                        "docs".to_string()
                    } else if parts.first() == Some(&"tests") {
                        "tests".to_string()
                    } else {
                        parts.first().unwrap_or(&"other").to_string()
                    }
                })
                .unwrap_or_else(|| "other".to_string());

            serde_json::json!({
                "id": id.to_string(),
                "label": label,
                "preview": preview,
                "source_file": entry.and_then(|e| e.metadata.source_file.as_deref()),
                "language": entry.and_then(|e| e.metadata.language.as_deref()),
                "group": group,
                "content_role": entry.and_then(|e| e.metadata.content_role.as_deref()),
                "source_origin": entry.and_then(|e| e.metadata.source_origin.as_deref()),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "nodes": nodes,
        "edges": all_edges,
    })))
}

// ---------------------------------------------------------------------------
// Clustered graph endpoint (LOD)
// ---------------------------------------------------------------------------

/// Query params for /api/dashboard/graph/scope with LOD support.
#[derive(Debug, Deserialize)]
pub struct ClusteredGraphParams {
    pub level: Option<u8>,
    pub parent: Option<String>,
}

/// GET /api/dashboard/graph/scope
/// When `level` is provided, returns clustered graph at the specified LOD level.
/// When no `level` is provided, falls through to legacy behavior (all nodes + edges).
async fn clustered_graph_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ClusteredGraphParams>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // No level param: legacy behavior (backward compatible)
    let level = match params.level {
        None => return graph_scope_handler(State(state)).await.map(|j| j.into_response()),
        Some(l) => l,
    };

    let hierarchy = match state.cluster_store.current() {
        Some(h) => h,
        None => {
            // Degraded mode: cluster store not yet computed.
            // Try an immediate computation.
            let scope_id = state.default_scope_id.as_deref().unwrap_or(DEFAULT_SCOPE_ID);
            let entries = load_scope_entries(&state.data_dir, scope_id).await?;
            let pairs: Vec<(String, Vec<f32>)> = entries
                .iter()
                .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.to_string(), emb.clone())))
                .collect();
            let label_map: std::collections::HashMap<String, String> = entries
                .iter()
                .map(|e| {
                    let label = e.metadata.source_file.clone()
                        .unwrap_or_else(|| e.content.chars().take(60).collect());
                    (e.id.to_string(), label)
                })
                .collect();
            if state.cluster_store.maybe_recompute_with_labels(&pairs, &label_map) {
                tracing::info!("Cluster hierarchy computed on-demand ({} entries)", pairs.len());
            }
            match state.cluster_store.current() {
                Some(h) => h,
                None => {
                    return Ok(Json(serde_json::json!({
                        "nodes": [], "edges": [], "degraded": true
                    })).into_response());
                }
            }
        }
    };

    match level {
        0 => {
            // Super-clusters as nodes
            let scope_id = state.default_scope_id.as_deref().unwrap_or(DEFAULT_SCOPE_ID);
            let entries = load_scope_entries(&state.data_dir, scope_id).await.unwrap_or_default();
            let entry_map: std::collections::HashMap<String, &corvia_common::types::KnowledgeEntry> =
                entries.iter().map(|e| (e.id.to_string(), e)).collect();

            let nodes: Vec<serde_json::Value> = hierarchy.super_clusters.iter().map(|sc| {
                // Find best label from the nearest-centroid entry
                let label = sc.entry_ids.iter()
                    .find_map(|id| entry_map.get(id.as_str()))
                    .and_then(|e| {
                        e.metadata.source_file.clone()
                            .or_else(|| Some(e.content.chars().take(60).collect()))
                    })
                    .unwrap_or_else(|| sc.label.clone());

                serde_json::json!({
                    "id": sc.cluster_id,
                    "label": label,
                    "level": 0,
                    "entry_count": sc.entry_ids.len(),
                })
            }).collect();

            // Compute inter-cluster edge counts
            let mut edges = Vec::new();
            for (i, sc_a) in hierarchy.super_clusters.iter().enumerate() {
                for sc_b in hierarchy.super_clusters.iter().skip(i + 1) {
                    // Count graph edges that cross between these two clusters
                    let mut cross_count = 0u32;
                    for entry_id_str in &sc_a.entry_ids {
                        if let Ok(uuid) = entry_id_str.parse::<uuid::Uuid>() {
                            let entry_edges = state
                                .graph
                                .edges(&uuid, corvia_common::types::EdgeDirection::Both)
                                .await
                                .unwrap_or_default();
                            for edge in &entry_edges {
                                let other_id = if edge.from == uuid {
                                    edge.to.to_string()
                                } else {
                                    edge.from.to_string()
                                };
                                if sc_b.entry_ids.contains(&other_id) {
                                    cross_count += 1;
                                }
                            }
                        }
                    }
                    if cross_count > 0 {
                        edges.push(serde_json::json!({
                            "from": sc_a.cluster_id,
                            "to": sc_b.cluster_id,
                            "weight": cross_count,
                        }));
                    }
                }
            }

            Ok(Json(serde_json::json!({ "nodes": nodes, "edges": edges })).into_response())
        }
        1 => {
            // Sub-clusters within a parent super-cluster
            let parent_id = params.parent.as_deref().unwrap_or("");
            let nodes: Vec<serde_json::Value> = hierarchy.sub_clusters.iter()
                .filter(|sc| sc.parent_id.as_deref() == Some(parent_id))
                .map(|sc| serde_json::json!({
                    "id": sc.cluster_id,
                    "label": sc.label,
                    "level": 1,
                    "entry_count": sc.entry_ids.len(),
                    "parent_id": sc.parent_id,
                }))
                .collect();
            Ok(Json(serde_json::json!({ "nodes": nodes, "edges": [] })).into_response())
        }
        2 => {
            // Individual entries within a parent cluster
            let parent_id = params.parent.as_deref().unwrap_or("");
            let cluster = hierarchy.sub_clusters.iter()
                .find(|sc| sc.cluster_id == parent_id)
                .or_else(|| hierarchy.super_clusters.iter().find(|sc| sc.cluster_id == parent_id));

            let scope_id = state.default_scope_id.as_deref().unwrap_or(DEFAULT_SCOPE_ID);
            let entries = load_scope_entries(&state.data_dir, scope_id).await.unwrap_or_default();
            let entry_map: std::collections::HashMap<String, &corvia_common::types::KnowledgeEntry> =
                entries.iter().map(|e| (e.id.to_string(), e)).collect();

            let entry_ids = cluster.map(|c| &c.entry_ids);
            let nodes: Vec<serde_json::Value> = entry_ids
                .into_iter()
                .flat_map(|ids| ids.iter())
                .filter_map(|id| {
                    entry_map.get(id.as_str()).map(|e| {
                        let label = e.metadata.source_file.clone()
                            .unwrap_or_else(|| e.content.chars().take(80).collect());
                        serde_json::json!({
                            "id": id,
                            "label": label,
                            "level": 2,
                            "preview": e.content.chars().take(200).collect::<String>(),
                            "source_file": e.metadata.source_file,
                            "language": e.metadata.language,
                            "content_role": e.metadata.content_role,
                            "source_origin": e.metadata.source_origin,
                        })
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({ "nodes": nodes, "edges": [], "level": 2 })).into_response())
        }
        _ => {
            Ok(Json(serde_json::json!({ "error": "Invalid level. Use 0, 1, or 2." })).into_response())
        }
    }
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
    /// Optional single check to run (e.g. "stale", "broken", "orphan", "dangling", "cycle",
    /// "misplaced", "temporal", "coverage")
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
        .unwrap_or(DEFAULT_SCOPE_ID);

    let entries = load_scope_entries(&state.data_dir, scope_id).await?;

    // Extract DocsRulesConfig from live config for MisplacedDoc checks
    let docs_rules = state
        .config
        .read()
        .ok()
        .and_then(|cfg| cfg.workspace.as_ref()?.docs.as_ref()?.rules.clone())
        .unwrap_or_default();

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*state.store, &*state.graph)
        .with_docs_rules(docs_rules);

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

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(RAG_ASK_TIMEOUT_SECS),
        rag.ask(&req.query, &req.scope_id, Some(opts)),
    ).await
        .map_err(|_| (StatusCode::GATEWAY_TIMEOUT, format!("RAG ask timed out after {RAG_ASK_TIMEOUT_SECS}s")))?
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

// ---------------------------------------------------------------------------
// Agent reconnect / connect / refresh-summary endpoints
// ---------------------------------------------------------------------------

/// GET /api/dashboard/agents/reconnectable
/// Returns agents with stale or orphaned sessions, sorted by last_seen.
async fn reconnectable_agents_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let agents = state.coordinator.list_reconnectable()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list reconnectable agents: {e}")))?;
    Ok(Json(serde_json::json!(agents)))
}

/// Request body for POST /api/dashboard/agents/{agent_id}/connect
#[derive(Debug, Deserialize)]
pub struct ConnectAgentRequest {
    pub description: Option<String>,
}

/// POST /api/dashboard/agents/{agent_id}/connect
/// Connect to an existing agent, optionally updating its description.
async fn connect_agent_handler(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
    Json(body): Json<ConnectAgentRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Update description if provided
    if let Some(ref desc) = body.description {
        let _ = state.coordinator.registry.set_description(&agent_id, desc);
    }
    let response = state.coordinator.connect(&agent_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to connect agent: {e}")))?;

    Ok(Json(serde_json::json!({
        "agent_id": response.agent_id,
        "active_sessions": response.active_sessions.len(),
        "recoverable_sessions": response.recoverable_sessions.len(),
    })))
}

/// POST /api/dashboard/agents/{agent_id}/refresh-summary
/// Recompute activity summary for an agent using ClusterStore topic tags.
async fn refresh_summary_handler(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Get existing agent record (needed for prior topic_tags to detect drift)
    let agent = state.coordinator.registry.get(&agent_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Agent {agent_id} not found")))?;

    // Load entries from knowledge files and compute topic tags
    let scope_id = state.default_scope_id.as_deref().unwrap_or(DEFAULT_SCOPE_ID);
    let entries = load_scope_entries(&state.data_dir, scope_id).await?;

    // Get embeddings for entries (all entries for now — filtering by agent would require metadata)
    let pairs: Vec<(String, Vec<f32>)> = entries.iter()
        .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.to_string(), emb.clone())))
        .collect();
    let label_map: std::collections::HashMap<String, String> = entries
        .iter()
        .map(|e| {
            let label = e.metadata.source_file.clone()
                .unwrap_or_else(|| e.content.chars().take(60).collect());
            (e.id.to_string(), label)
        })
        .collect();

    // Ensure cluster store is current
    state.cluster_store.maybe_recompute_with_labels(&pairs, &label_map);

    if let Some(hierarchy) = state.cluster_store.current() {
        // Compute topic tags from all embeddings (simplified — ideally filter by agent entries)
        let all_embeddings: Vec<Vec<f32>> = pairs.iter().map(|(_, e)| e.clone()).collect();
        let new_topic_tags = clustering::compute_topic_tags(&hierarchy, &all_embeddings);

        let sessions = state.coordinator.sessions.list_by_agent(&agent_id)
            .unwrap_or_default();

        // Detect drift: compare existing topic_tags with new last_topics
        let historical_tags = agent.activity_summary
            .as_ref()
            .map(|s| &s.topic_tags[..])
            .unwrap_or(&[]);
        let drifted = clustering::is_topic_drifted(historical_tags, &new_topic_tags);

        // Preserve historical topic_tags if they exist, update last_topics to current
        let topic_tags = if historical_tags.is_empty() {
            new_topic_tags.clone()
        } else {
            historical_tags.to_vec()
        };

        let summary = corvia_common::agent_types::ActivitySummary {
            entry_count: entries.len() as u64,
            topic_tags,
            last_topics: new_topic_tags,
            last_active: chrono::Utc::now(),
            session_count: sessions.len() as u64,
            drifted,
        };

        let _ = state.coordinator.registry.set_activity_summary(&agent_id, &summary);

        Ok(Json(serde_json::json!({
            "status": "refreshed",
            "agent_id": agent_id,
            "drifted": drifted,
        })))
    } else {
        Ok(Json(serde_json::json!({ "status": "skipped", "reason": "cluster store not yet computed" })))
    }
}

// ---------------------------------------------------------------------------
// GC endpoints
// ---------------------------------------------------------------------------

/// GET /api/dashboard/gc
/// Returns GC history, last run report, and schedule status.
async fn gc_status_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::GcStatusResponse> {
    let last_run = state.gc_history.last().map(gc_report_to_dto);
    let history: Vec<corvia_common::dashboard::GcReportDto> = state
        .gc_history
        .all()
        .into_iter()
        .map(gc_report_to_dto)
        .collect();

    Json(corvia_common::dashboard::GcStatusResponse {
        last_run,
        history,
        scheduled: false,
    })
}

/// POST /api/dashboard/gc/run
/// Trigger a manual GC sweep and return the report.
async fn gc_run_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::GcReportDto>, (StatusCode, String)> {
    let report = corvia_kernel::ops::gc_run(&state.coordinator)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("GC failed: {e}")))?;

    state.gc_history.push(report.clone());

    Ok(Json(gc_report_to_dto(report)))
}

// ---------------------------------------------------------------------------
// Tiered knowledge endpoints
// ---------------------------------------------------------------------------

fn gc_cycle_to_dto(r: &corvia_kernel::gc_worker::GcCycleReport) -> corvia_common::dashboard::GcCycleReportDto {
    corvia_common::dashboard::GcCycleReportDto {
        entries_scanned: r.entries_scanned,
        entries_scored: r.entries_scored,
        hot_to_warm: r.hot_to_warm,
        warm_to_cold: r.warm_to_cold,
        cold_to_forgotten: r.cold_to_forgotten,
        warm_to_hot: r.warm_to_hot,
        cold_to_warm: r.cold_to_warm,
        hnsw_rebuild_triggered: r.hnsw_rebuild_triggered,
        rebuild_duration_ms: r.rebuild_duration_ms,
        cycle_duration_ms: r.cycle_duration_ms,
        scopes_processed: r.scopes_processed,
        chain_protected: r.chain_protected,
        auto_protected: r.auto_protected,
        budget_demoted: r.budget_demoted,
        circuit_breaker_tripped: r.circuit_breaker_tripped,
    }
}

/// GET /api/dashboard/tiers
/// Returns tier distribution and forgetting config status.
async fn tiers_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::TiersResponse> {
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or(DEFAULT_SCOPE_ID);

    // Get tier distribution (lightweight — avoids full deserialization)
    let dist = if let Some(lite_store) = state.store.as_any().downcast_ref::<corvia_kernel::lite_store::LiteStore>() {
        lite_store.count_tiers_by_scope(scope_id).unwrap_or_default()
    } else {
        corvia_kernel::ops::TierDistribution::default()
    };

    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner());
    let forgetting_enabled = cfg.forgetting.as_ref().map(|f| f.enabled).unwrap_or(false);
    drop(cfg);

    let total = dist.hot + dist.warm + dist.cold + dist.forgotten;

    Json(corvia_common::dashboard::TiersResponse {
        hot: dist.hot,
        warm: dist.warm,
        cold: dist.cold,
        forgotten: dist.forgotten,
        total,
        forgetting_enabled,
    })
}

/// GET /api/dashboard/tiers/transitions
/// Returns recent tier transitions extracted from structured trace logs.
async fn tier_transitions_handler(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<TransitionsQuery>,
) -> Result<Json<corvia_common::dashboard::TierTransitionsResponse>, (StatusCode, String)> {
    let limit = params.limit.unwrap_or(50).min(200);
    let log_dir = traces::log_dir();
    let transitions = tokio::task::spawn_blocking(move || {
        parse_tier_transitions(&log_dir, limit)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join failed: {e}")))?;
    let count = transitions.len();
    Ok(Json(corvia_common::dashboard::TierTransitionsResponse { transitions, count }))
}

#[derive(Deserialize)]
struct TransitionsQuery {
    limit: Option<usize>,
}

/// Parse tier_transition events from structured trace logs.
fn parse_tier_transitions(
    log_dir: &std::path::Path,
    limit: usize,
) -> Vec<corvia_common::dashboard::TierTransitionDto> {
    let trace_file = log_dir.join("corvia-traces.log");
    let content = match std::fs::read_to_string(&trace_file) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut transitions = Vec::new();

    // Parse from end (most recent first) — iterate in reverse
    for line in content.lines().rev() {
        if transitions.len() >= limit {
            break;
        }
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Look for tier_transition events
        let msg = v
            .get("fields")
            .and_then(|f| f.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("");

        if msg != "tier_transition" {
            continue;
        }

        let fields = match v.get("fields") {
            Some(f) => f,
            None => continue,
        };

        let entry_id = fields.get("entry_id").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let scope_id = fields.get("scope_id").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let from_tier = fields.get("from_tier").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let to_tier = fields.get("to_tier").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let retention_score = fields.get("retention_score").and_then(|e| e.as_f64()).unwrap_or(0.0);
        let reason = fields.get("reason").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let timestamp = v.get("timestamp").and_then(|t| t.as_str()).unwrap_or("").to_string();

        transitions.push(corvia_common::dashboard::TierTransitionDto {
            entry_id,
            scope_id,
            from_tier,
            to_tier,
            retention_score,
            reason,
            timestamp,
        });
    }

    transitions
}

/// GET /api/dashboard/tiers/pinned
/// Returns all pinned entries for the default scope.
async fn pinned_entries_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::PinnedEntriesResponse>, (StatusCode, String)> {
    let scope_id = state
        .default_scope_id
        .as_deref()
        .unwrap_or(DEFAULT_SCOPE_ID);

    let entries = load_scope_entries(&state.data_dir, scope_id).await?;

    let pinned: Vec<corvia_common::dashboard::PinnedEntryDto> = entries
        .into_iter()
        .filter_map(|entry| {
            let pin = entry.pin.as_ref()?;
            let preview = if entry.content.len() > 120 {
                // Find the last char boundary at or before byte 120 to avoid
                // panicking on multi-byte UTF-8 sequences.
                let end = entry.content.char_indices()
                    .map(|(i, _)| i)
                    .take_while(|&i| i <= 120)
                    .last()
                    .unwrap_or(0);
                format!("{}...", &entry.content[..end])
            } else {
                entry.content.clone()
            };
            Some(corvia_common::dashboard::PinnedEntryDto {
                entry_id: entry.id.to_string(),
                content_preview: preview,
                pinned_by: pin.by.clone(),
                pinned_at: pin.at.to_rfc3339(),
                scope_id: entry.scope_id.clone(),
            })
        })
        .collect();

    let count = pinned.len();
    Ok(Json(corvia_common::dashboard::PinnedEntriesResponse { entries: pinned, count }))
}

/// GET /api/dashboard/tiers/gc-history
/// Returns knowledge GC cycle history (distinct from session-level GC).
async fn gc_knowledge_history_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::GcCycleHistoryResponse> {
    let cycles: Vec<corvia_common::dashboard::GcCycleReportDto> = state
        .gc_knowledge_history
        .all()
        .iter()
        .map(gc_cycle_to_dto)
        .collect();
    let count = cycles.len();
    Json(corvia_common::dashboard::GcCycleHistoryResponse { cycles, count })
}

/// POST /api/dashboard/tiers/unpin/{entry_id}
/// Unpin an entry (makes it eligible for forgetting again).
async fn dashboard_unpin_handler(
    State(state): State<Arc<AppState>>,
    Path(entry_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&entry_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;
    let result = corvia_kernel::ops::unpin_entry(&state.store, &uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Unpin failed: {e}")))?;
    let json = serde_json::to_value(result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization failed: {e}")))?;
    Ok(Json(json))
}

// ---------------------------------------------------------------------------
// Live sessions endpoint
// ---------------------------------------------------------------------------

/// GET /api/dashboard/sessions/live
/// Returns currently open sessions with staging state and summary metrics.
async fn live_sessions_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::LiveSessionsResponse>, (StatusCode, String)> {
    let open_sessions = state
        .coordinator
        .sessions
        .list_open()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list sessions: {e}")))?;

    let agents = corvia_kernel::ops::agents_list(&state.coordinator).unwrap_or_default();
    let agent_names: std::collections::HashMap<String, String> = agents
        .into_iter()
        .map(|a| (a.agent_id.clone(), a.display_name))
        .collect();

    let now = chrono::Utc::now();
    let mut total_active = 0usize;
    let mut total_stale = 0usize;
    let mut total_entries_pending = 0u64;

    let sessions: Vec<corvia_common::dashboard::LiveSession> = open_sessions
        .iter()
        .map(|s| {
            let state_str = format!("{:?}", s.state);
            let pending = s.entries_written.saturating_sub(s.entries_merged);
            let duration = (now - s.created_at).num_seconds().max(0) as u64;
            let has_staging = s
                .staging_dir
                .as_ref()
                .map(|d| std::path::Path::new(d).exists())
                .unwrap_or(false);

            match s.state {
                corvia_common::agent_types::SessionState::Active => total_active += 1,
                corvia_common::agent_types::SessionState::Stale => total_stale += 1,
                _ => {}
            }
            total_entries_pending += pending;

            corvia_common::dashboard::LiveSession {
                session_id: s.session_id.clone(),
                agent_id: s.agent_id.clone(),
                agent_name: agent_names
                    .get(&s.agent_id)
                    .cloned()
                    .unwrap_or_else(|| s.agent_id.clone()),
                state: state_str,
                started_at: s.created_at.to_rfc3339(),
                duration_secs: duration,
                entries_written: s.entries_written,
                entries_merged: s.entries_merged,
                pending_entries: pending,
                git_branch: s.git_branch.clone(),
                has_staging_dir: has_staging,
            }
        })
        .collect();

    Ok(Json(corvia_common::dashboard::LiveSessionsResponse {
        sessions,
        summary: corvia_common::dashboard::LiveSessionsSummary {
            total_active,
            total_stale,
            total_entries_pending,
        },
    }))
}

// ---------------------------------------------------------------------------
// Hook-observed Claude sessions (file watcher)
// ---------------------------------------------------------------------------

async fn hook_sessions_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::HookSessionsResponse>, (StatusCode, String)> {
    let sessions = state.hook_sessions.snapshot().await;
    let total_active = sessions
        .iter()
        .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Active)
        .count();
    let total_stale = sessions
        .iter()
        .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Stale)
        .count();
    Ok(Json(corvia_common::dashboard::HookSessionsResponse {
        sessions,
        total_active,
        total_stale,
    }))
}

async fn hook_sessions_stream(
    State(state): State<Arc<AppState>>,
) -> axum::response::sse::Sse<impl futures_core::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>
{
    let mut rx = state.hook_sessions.subscribe();
    let hook_sessions = state.hook_sessions.clone();

    let stream = async_stream::stream! {
        // Send initial snapshot.
        let sessions = hook_sessions.snapshot().await;
        let total_active = sessions.iter()
            .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Active)
            .count();
        let total_stale = sessions.iter()
            .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Stale)
            .count();
        let initial = corvia_common::dashboard::HookSessionsResponse {
            sessions,
            total_active,
            total_stale,
        };
        if let Ok(data) = serde_json::to_string(&initial) {
            yield Ok(axum::response::sse::Event::default()
                .event("snapshot")
                .data(data));
        }

        // Stream deltas.
        loop {
            match rx.recv().await {
                Ok(update) => {
                    if let Ok(data) = serde_json::to_string(&update) {
                        yield Ok(axum::response::sse::Event::default()
                            .event("update")
                            .data(data));
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(lagged = n, "SSE client lagged, sending fresh snapshot");
                    let sessions = hook_sessions.snapshot().await;
                    let total_active = sessions.iter()
                        .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Active)
                        .count();
                    let total_stale = sessions.iter()
                        .filter(|s| s.state == corvia_common::dashboard::HookSessionState::Stale)
                        .count();
                    let refresh = corvia_common::dashboard::HookSessionsResponse {
                        sessions,
                        total_active,
                        total_stale,
                    };
                    if let Ok(data) = serde_json::to_string(&refresh) {
                        yield Ok(axum::response::sse::Event::default()
                            .event("snapshot")
                            .data(data));
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    axum::response::sse::Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15)),
    )
}

// ---------------------------------------------------------------------------
// Recent traces endpoint
// ---------------------------------------------------------------------------

/// Query params for /api/dashboard/traces/recent
#[derive(Debug, Deserialize)]
pub struct RecentTracesQuery {
    pub limit: Option<usize>,
}

/// GET /api/dashboard/traces/recent
/// Returns recent OTEL traces as parent-child trees.
async fn recent_traces_handler(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<RecentTracesQuery>,
) -> Json<corvia_common::dashboard::RecentTracesResponse> {
    let limit = params.limit.unwrap_or(20).min(100);
    let log_dir = traces::log_dir();

    let mut all_lines = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&log_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                let lines = traces::tail_lines(&path, 2000);
                all_lines.extend(lines);
            }
        }
    }

    let line_refs: Vec<&str> = all_lines.iter().map(|s| s.as_str()).collect();
    let trees = traces::collect_trace_trees(&line_refs, limit);

    Json(corvia_common::dashboard::RecentTracesResponse { traces: trees })
}

// ---------------------------------------------------------------------------
// GPU status endpoint
// ---------------------------------------------------------------------------

/// GET /api/dashboard/gpu
/// Returns detected GPU hardware and inference backend configuration.
async fn gpu_status_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::GpuStatusResponse> {
    use std::time::Instant;

    let mut cache = state.gpu_cache.lock().await;

    if !cache.is_stale() {
        return Json(cache.last_result.clone());
    }

    if cache.refreshing && cache.refresh_started.elapsed() < std::time::Duration::from_secs(10) {
        // Another request is already refreshing — return stale data
        return Json(cache.last_result.clone());
    }

    cache.refreshing = true;
    cache.refresh_started = Instant::now();
    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    drop(cache); // release lock during shell-outs

    let mut status = gpu::collect_gpu_status(&cfg);
    status.inference_health = Some(
        health::check_inference_health("127.0.0.1:8030").await
            .unwrap_or(corvia_common::dashboard::InferenceHealthStatus {
                status: corvia_common::dashboard::InferenceProbeStatus::Unreachable,
                ep_name: String::new(),
                ep_requested: String::new(),
                fallback_used: false,
                baseline_us: 0,
                last_probe_us: 0,
                drift_pct: 0.0,
                models_loaded: 0,
                last_probe_at: None,
            })
    );

    let mut cache = state.gpu_cache.lock().await;
    cache.last_result = status.clone();
    cache.last_fetched = Instant::now();
    cache.refreshing = false;

    Json(status)
}

// ---------------------------------------------------------------------------
// Entry / History / Neighbors endpoints
// ---------------------------------------------------------------------------

/// DTO for a neighbor entry in the graph.
#[derive(Serialize)]
pub struct NeighborDto {
    pub id: String,
    pub content: String,
    pub relation: String,
    pub direction: String,
    pub score: Option<f32>,
    pub source_file: Option<String>,
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
}

/// Query params for /api/dashboard/entries/{entry_id}/neighbors
#[derive(Debug, serde::Deserialize)]
pub struct NeighborParams {
    pub depth: Option<usize>,
}

/// GET /api/dashboard/entries/{entry_id}/neighbors
/// Returns neighbor entries via graph traversal.
async fn entry_neighbors_handler(
    State(state): State<Arc<AppState>>,
    Path(entry_id): Path<String>,
    Query(params): Query<NeighborParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = entry_id
        .parse::<uuid::Uuid>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid entry_id: {e}")))?;
    let depth = params.depth.unwrap_or(1).min(3);

    // Use traverse for multi-hop BFS
    let traversed = state
        .graph
        .traverse(&uuid, None, corvia_common::types::EdgeDirection::Both, depth)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Get direct edges for relation info
    let edges = state
        .graph
        .edges(&uuid, corvia_common::types::EdgeDirection::Both)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Build map from neighbor_id -> (relation, direction)
    let mut edge_info: std::collections::HashMap<uuid::Uuid, (String, String)> =
        std::collections::HashMap::new();
    for edge in &edges {
        if edge.from == uuid {
            edge_info.insert(edge.to, (edge.relation.clone(), "outgoing".into()));
        } else {
            edge_info.insert(edge.from, (edge.relation.clone(), "incoming".into()));
        }
    }

    let mut neighbors = Vec::new();
    for entry in &traversed {
        if entry.id == uuid {
            continue;
        }
        let (relation, direction) = edge_info
            .get(&entry.id)
            .cloned()
            .unwrap_or(("transitive".into(), "outgoing".into()));
        neighbors.push(NeighborDto {
            id: entry.id.to_string(),
            content: entry.content.chars().take(200).collect(),
            relation,
            direction,
            score: None,
            source_file: entry.metadata.source_file.clone(),
            content_role: entry.metadata.content_role.clone(),
            source_origin: entry.metadata.source_origin.clone(),
        });
    }

    let count = neighbors.len();
    Ok(Json(serde_json::json!({
        "neighbors": neighbors,
        "count": count,
    })))
}

/// GET /api/dashboard/entries/{entry_id}
/// Returns a single entry by ID.
async fn entry_detail_handler(
    State(state): State<Arc<AppState>>,
    Path(entry_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = entry_id
        .parse::<uuid::Uuid>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid entry_id: {e}")))?;

    let entry = state
        .store
        .get(&uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get entry: {e}")))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Entry {entry_id} not found")))?;

    Ok(Json(serde_json::json!({
        "id": entry.id.to_string(),
        "content": entry.content,
        "scope_id": entry.scope_id,
        "recorded_at": entry.recorded_at.to_rfc3339(),
        "valid_from": entry.valid_from.to_rfc3339(),
        "valid_to": entry.valid_to.map(|t| t.to_rfc3339()),
        "superseded_by": entry.superseded_by.map(|id| id.to_string()),
        "metadata": {
            "source_file": entry.metadata.source_file,
            "language": entry.metadata.language,
            "content_role": entry.metadata.content_role,
            "source_origin": entry.metadata.source_origin,
        },
    })))
}

/// GET /api/dashboard/entries/{entry_id}/history
/// Returns the supersession chain for an entry.
async fn entry_history_handler(
    State(state): State<Arc<AppState>>,
    Path(entry_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = entry_id
        .parse::<uuid::Uuid>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid entry_id: {e}")))?;

    let chain = state
        .temporal
        .history(&uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get history: {e}")))?;

    let chain_json: Vec<serde_json::Value> = chain
        .iter()
        .map(|e| {
            serde_json::json!({
                "id": e.id.to_string(),
                "content": e.content,
                "recorded_at": e.recorded_at.to_rfc3339(),
                "valid_from": e.valid_from.to_rfc3339(),
                "valid_to": e.valid_to.map(|t| t.to_rfc3339()),
                "superseded_by": e.superseded_by.map(|id| id.to_string()),
                "is_current": e.superseded_by.is_none(),
                "metadata": {
                    "source_file": e.metadata.source_file,
                    "language": e.metadata.language,
                    "content_role": e.metadata.content_role,
                    "source_origin": e.metadata.source_origin,
                },
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "entry_id": entry_id,
        "chain": chain_json,
        "count": chain_json.len(),
    })))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_kernel::agent_coordinator::AgentCoordinator;
    use corvia_kernel::lite_store::LiteStore;
    use corvia_kernel::traits::{
        GenerationEngine, GenerationResult, GraphStore, InferenceEngine, QueryableStore,
        TemporalStore,
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    struct MockEngine;
    #[async_trait::async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(
            &self,
            texts: &[String],
        ) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize {
            3
        }
    }

    struct MockGenerationEngine;
    #[async_trait::async_trait]
    impl GenerationEngine for MockGenerationEngine {
        fn name(&self) -> &str {
            "mock"
        }
        async fn generate(
            &self,
            _system_prompt: &str,
            _user_message: &str,
        ) -> corvia_common::errors::Result<GenerationResult> {
            Ok(GenerationResult {
                text: "mock".into(),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize {
            4096
        }
    }

    async fn test_state(dir: &std::path::Path) -> Arc<crate::rest::AppState> {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen_engine =
            Arc::new(MockGenerationEngine) as Arc<dyn GenerationEngine>;
        let coordinator = Arc::new(
            AgentCoordinator::new(
                store.clone() as Arc<dyn QueryableStore>,
                engine.clone(),
                dir,
                corvia_common::config::AgentLifecycleConfig::default(),
                corvia_common::config::MergeConfig {
                    similarity_threshold: 2.0,
                    ..Default::default()
                },
                gen_engine,
            )
            .unwrap(),
        );
        Arc::new(crate::rest::AppState {
            store: store.clone() as Arc<dyn QueryableStore>,
            engine,
            coordinator,
            graph: store.clone() as Arc<dyn GraphStore>,
            temporal: store as Arc<dyn TemporalStore>,
            data_dir: dir.to_path_buf(),
            rag: None,
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            default_scope_id: None,
            config: Arc::new(std::sync::RwLock::new(
                corvia_common::config::CorviaConfig::default(),
            )),
            config_path: dir.join("corvia.toml"),
            cluster_store: Arc::new(super::clustering::ClusterStore::new()),
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
            session_ingest_lock: tokio::sync::Mutex::new(()),
            hook_sessions: super::session_watcher::SessionWatcherState::new().0,
            coverage_cache: Arc::new(
                super::coverage::IndexCoverageCache::new(0.9, 60),
            ),
            workspace_root: dir.to_path_buf(),
            ingest_status: Arc::new(std::sync::RwLock::new(corvia_kernel::ingest::IngestStatus::idle())),
            gpu_cache: std::sync::Arc::new(tokio::sync::Mutex::new(super::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: std::sync::Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: std::sync::Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
        })
    }

    /// Helper: send a GET request to the dashboard router and return (status, body JSON).
    /// If the response is not valid JSON (e.g. plain text error), wraps it as a JSON string value.
    async fn get_json(
        state: Arc<crate::rest::AppState>,
        uri: &str,
    ) -> (axum::http::StatusCode, serde_json::Value) {
        let app = router(state);
        let req = axum::http::Request::builder()
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned()));
        (status, json)
    }

    #[tokio::test]
    async fn test_status_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/status").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("entry_count").is_some(), "missing entry_count");
        assert!(json.get("agent_count").is_some(), "missing agent_count");
        assert!(json.get("merge_queue_depth").is_some(), "missing merge_queue_depth");
        assert!(json.get("session_count").is_some(), "missing session_count");
        assert!(json.get("config").is_some(), "missing config");
        assert!(json.get("services").is_some(), "missing services");

        // Verify values are zero for empty store
        assert_eq!(json["entry_count"], 0);
        assert_eq!(json["agent_count"], 0);
        assert_eq!(json["merge_queue_depth"], 0);

        // Verify coverage fields are present
        assert!(json.get("index_coverage").is_some(), "missing index_coverage");
        assert!(json.get("index_stale").is_some(), "missing index_stale");
        assert!(json.get("index_disk_count").is_some(), "missing index_disk_count");
        assert!(json.get("index_store_count").is_some(), "missing index_store_count");
        assert!(json.get("index_hnsw_count").is_some(), "missing index_hnsw_count");
        assert!(json.get("index_stale_threshold").is_some(), "missing index_stale_threshold");

        // Fresh workspace: coverage is null, counts are 0
        assert!(json["index_coverage"].is_null());
        assert!(json["index_stale"].is_null());
        assert_eq!(json["index_disk_count"], 0);
        assert_eq!(json["index_store_count"], 0);
        assert_eq!(json["index_hnsw_count"], 0);
        assert_eq!(json["index_stale_threshold"], 0.9);
    }

    #[tokio::test]
    async fn test_refresh_coverage_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let app = router(state);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/api/dashboard/status/refresh-coverage")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("index_coverage").is_some(), "refresh should return coverage");
        assert!(json.get("index_coverage_checked_at").is_some(), "refresh should return checked_at");
        // Fresh workspace: verify exact values
        assert!(json["index_coverage"].is_null());
        assert!(json["index_stale"].is_null());
        assert_eq!(json["index_disk_count"], 0);
        assert_eq!(json["index_store_count"], 0);
        assert_eq!(json["index_hnsw_count"], 0);
        assert_eq!(json["index_stale_threshold"], 0.9);
    }

    #[tokio::test]
    async fn test_recent_traces_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/traces/recent").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("traces").is_some(), "missing traces array");
        assert!(json["traces"].is_array(), "traces should be an array");
    }

    #[tokio::test]
    async fn test_agents_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/agents").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.is_array(), "response should be an array of agents");
        // Empty coordinator should produce an empty list
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_health_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/health").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("findings").is_some(), "missing findings");
        assert!(json["findings"].is_array(), "findings should be an array");
        assert!(json.get("count").is_some(), "missing count");
        assert!(json.get("scope_id").is_some(), "missing scope_id");
        // Empty store should have no findings (or a small deterministic set)
        let count = json["count"].as_u64().unwrap();
        assert_eq!(count, json["findings"].as_array().unwrap().len() as u64);
    }

    #[tokio::test]
    async fn test_graph_handler_no_entry_id() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        // Without entry_id, should return empty edges
        let (status, json) = get_json(state, "/api/dashboard/graph").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("edges").is_some(), "missing edges");
        assert!(json["edges"].is_array(), "edges should be an array");
        assert_eq!(json["edges"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_graph_handler_invalid_entry_id() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let app = router(state);
        let req = axum::http::Request::builder()
            .uri("/api/dashboard/graph?entry_id=not-a-uuid")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_traces_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/traces").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("spans").is_some(), "missing spans");
        assert!(json.get("recent_events").is_some(), "missing recent_events");
        assert!(json.get("modules").is_some(), "missing modules");
    }

    #[tokio::test]
    async fn test_config_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/config").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("embedding_provider").is_some(), "missing embedding_provider");
        assert!(json.get("storage").is_some(), "missing storage");
        assert!(json.get("workspace").is_some(), "missing workspace");
    }

    #[tokio::test]
    async fn test_logs_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/logs").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("entries").is_some(), "missing entries");
        assert!(json["entries"].is_array(), "entries should be an array");
        assert!(json.get("total").is_some(), "missing total");
    }

    #[tokio::test]
    async fn test_merge_queue_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/merge/queue").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        // merge_queue_status returns a serialized struct; check it parsed as valid JSON
        assert!(json.is_object() || json.is_array(), "response should be valid JSON");
    }

    #[tokio::test]
    async fn test_gc_status_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/gc").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("history").is_some(), "missing history");
        assert!(json["history"].is_array(), "history should be an array");
        assert_eq!(json["scheduled"], false);
    }

    #[tokio::test]
    async fn test_live_sessions_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/sessions/live").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("sessions").is_some(), "missing sessions");
        assert!(json["sessions"].is_array(), "sessions should be an array");
        assert!(json.get("summary").is_some(), "missing summary");
        assert_eq!(json["summary"]["total_active"], 0);
    }

    #[tokio::test]
    async fn test_gpu_status_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/gpu").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        // GPU handler returns a GpuStatusResponse; just verify it's valid JSON
        assert!(json.is_object(), "response should be a JSON object");
    }

    #[tokio::test]
    async fn test_status_coverage_with_data() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        // Insert 7 entries (creates knowledge JSON files on disk + store + HNSW)
        let scope_id = corvia_common::constants::DEFAULT_SCOPE_ID;
        for _ in 0..7 {
            let mut entry = corvia_common::types::KnowledgeEntry::new(
                "test content".into(), scope_id.into(), "test.rs".into(),
            );
            entry.embedding = Some(vec![0.1, 0.2, 0.3]);
            state.store.insert(&entry).await.unwrap();
        }
        // Add 3 extra JSON files to simulate partial ingest
        let knowledge_dir = tmp.path().join("knowledge").join(scope_id);
        for i in 0..3 {
            std::fs::write(
                knowledge_dir.join(format!("extra-{i}.json")),
                format!(r#"{{"extra":{i}}}"#),
            ).unwrap();
        }

        // Refresh coverage cache so status has data
        state.coverage_cache
            .refresh(&state.data_dir, scope_id, &*state.store)
            .await;

        let (status, json) = get_json(state, "/api/dashboard/status").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json["index_disk_count"], 10);
        assert_eq!(json["index_store_count"], 7);
        assert_eq!(json["index_hnsw_count"], 7);
        assert_eq!(json["index_coverage"], 0.7);
        assert_eq!(json["index_stale"], true);
        assert_eq!(json["index_stale_threshold"], 0.9);
        assert!(json["index_coverage_checked_at"].is_string(), "checked_at should be a string");
    }

    #[tokio::test]
    async fn test_refresh_populates_cache_for_status() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        // Status before any refresh — cache has defaults
        let (_, json_before) = get_json(state.clone(), "/api/dashboard/status").await;
        assert!(json_before["index_coverage"].is_null());
        assert!(json_before["index_coverage_checked_at"].is_null());

        // POST refresh-coverage to populate the cache
        let app = router(state.clone());
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/api/dashboard/status/refresh-coverage")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);

        // Status after refresh — checked_at should now be set
        let (_, json_after) = get_json(state, "/api/dashboard/status").await;
        assert!(
            json_after["index_coverage_checked_at"].is_string(),
            "after refresh, checked_at should be set"
        );
    }

    // --- Tiered knowledge endpoint tests ---

    #[tokio::test]
    async fn test_tiers_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/tiers").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json["hot"], 0);
        assert_eq!(json["warm"], 0);
        assert_eq!(json["cold"], 0);
        assert_eq!(json["forgotten"], 0);
        assert_eq!(json["total"], 0);
        assert_eq!(json["forgetting_enabled"], false);
    }

    #[tokio::test]
    async fn test_tiers_handler_forgetting_enabled() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        // Enable forgetting in config
        {
            let mut cfg = state.config.write().unwrap();
            cfg.forgetting = Some(corvia_common::config::ForgettingConfig {
                enabled: true,
                ..corvia_common::config::ForgettingConfig::default()
            });
        }

        let (status, json) = get_json(state, "/api/dashboard/tiers").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json["forgetting_enabled"], true);
    }

    #[tokio::test]
    async fn test_tier_transitions_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/tiers/transitions").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("transitions").is_some(), "missing transitions");
        assert!(json["transitions"].is_array(), "transitions should be an array");
        assert_eq!(json["count"], 0);
    }

    #[tokio::test]
    async fn test_pinned_entries_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/tiers/pinned").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("entries").is_some(), "missing entries");
        assert!(json["entries"].is_array(), "entries should be an array");
        assert_eq!(json["count"], 0);
    }

    #[tokio::test]
    async fn test_pinned_entries_handler_with_data() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        // Insert an entry with a pin
        let scope_id = corvia_common::constants::DEFAULT_SCOPE_ID;
        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "pinned test content".into(),
            scope_id.into(),
            "test.rs".into(),
        );
        entry.embedding = Some(vec![0.1, 0.2, 0.3]);
        entry.pin = Some(corvia_common::types::PinInfo {
            by: "test-agent".into(),
            at: chrono::Utc::now(),
        });
        let entry_id = entry.id.to_string();
        state.store.insert(&entry).await.unwrap();

        // Ensure the knowledge dir has the JSON file so load_scope_entries finds it
        let knowledge_dir = tmp.path().join("knowledge").join(scope_id);
        std::fs::create_dir_all(&knowledge_dir).unwrap();
        let json = serde_json::to_string(&entry).unwrap();
        std::fs::write(knowledge_dir.join(format!("{}.json", entry.id)), &json).unwrap();

        let (status, json_resp) = get_json(state, "/api/dashboard/tiers/pinned").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json_resp["count"], 1);
        let entries = json_resp["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["entry_id"], entry_id);
        assert_eq!(entries[0]["pinned_by"], "test-agent");
        assert_eq!(entries[0]["content_preview"], "pinned test content");
    }

    #[tokio::test]
    async fn test_unpin_bad_uuid_returns_400() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let app = router(state);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/api/dashboard/tiers/unpin/not-a-valid-uuid")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_gc_knowledge_history_handler() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path()).await;

        let (status, json) = get_json(state, "/api/dashboard/tiers/gc-history").await;

        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json.get("cycles").is_some(), "missing cycles");
        assert!(json["cycles"].is_array(), "cycles should be an array");
        assert_eq!(json["count"], 0);
    }

}
