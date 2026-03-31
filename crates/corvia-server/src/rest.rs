use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use chrono::{DateTime, Duration, Utc};
use corvia_common::agent_types::*;
use corvia_common::constants::{CLAUDE_SESSIONS_ADAPTER, RAG_ASK_TIMEOUT_SECS, USER_HISTORY_SCOPE};
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry, SearchResult};
use corvia_kernel::agent_coordinator::AgentCoordinator;
use corvia_kernel::ops::GcHistory;
use corvia_kernel::traits::{GraphStore, InferenceEngine, QueryableStore, TemporalStore};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use rust_embed::Embed;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

/// Embedded dashboard static assets. Built from the workspace's dashboard dist/.
/// The path is resolved at compile time via CORVIA_DASHBOARD_DIR env var,
/// defaulting to a placeholder that produces an empty embed.
#[derive(Embed)]
#[folder = "$CORVIA_DASHBOARD_DIR"]
struct DashboardAssets;

/// Map a CorviaError to the appropriate HTTP status code + message.
fn map_corvia_err(prefix: &str, e: corvia_common::errors::CorviaError) -> (StatusCode, String) {
    let status = match &e {
        corvia_common::errors::CorviaError::NotFound(_) => StatusCode::NOT_FOUND,
        corvia_common::errors::CorviaError::Validation(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, format!("{prefix}: {e}"))
}

pub struct AppState {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub coordinator: Arc<AgentCoordinator>,
    pub graph: Arc<dyn GraphStore>,
    pub temporal: Arc<dyn TemporalStore>,
    pub data_dir: std::path::PathBuf,
    pub rag: Option<Arc<arc_swap::ArcSwap<corvia_kernel::rag_pipeline::RagPipeline>>>,
    pub ready: Arc<AtomicBool>,
    /// Default scope_id from config, used when MCP clients omit scope_id.
    pub default_scope_id: Option<String>,
    /// Live config for hot-reload via MCP control plane.
    pub config: Arc<std::sync::RwLock<corvia_common::config::CorviaConfig>>,
    /// Path to the config file on disk.
    pub config_path: std::path::PathBuf,
    /// Semantic cluster hierarchy for LOD graph rendering.
    pub cluster_store: Arc<crate::dashboard::clustering::ClusterStore>,
    /// In-memory ring buffer of recent GC reports.
    pub gc_history: Arc<GcHistory>,
    /// Serializes session ingest + classify to prevent concurrent state file races.
    pub session_ingest_lock: tokio::sync::Mutex<()>,
    /// Hook-observed Claude sessions (JSONL file watcher state).
    pub hook_sessions: std::sync::Arc<crate::dashboard::session_watcher::SessionWatcherState>,
    /// Cached index coverage metrics (TTL-based, brief lock on cache read).
    pub coverage_cache: Arc<crate::dashboard::coverage::IndexCoverageCache>,
    /// Workspace root directory for server-side ingestion.
    pub workspace_root: std::path::PathBuf,
    /// Server-side ingestion status (for polling via GET /v1/ingest/status).
    pub ingest_status: Arc<std::sync::RwLock<corvia_kernel::ingest::IngestStatus>>,
    /// Cached GPU metrics with 5s TTL and stampede protection.
    pub gpu_cache: Arc<tokio::sync::Mutex<crate::dashboard::gpu::GpuMetricsCache>>,
    /// Counter for Forgotten entry access attempts (read by GC cycle span).
    pub forgotten_access_counter: Arc<corvia_kernel::gc_worker::ForgottenAccessCounter>,
    /// In-memory ring buffer of recent knowledge GC cycle reports.
    pub gc_knowledge_history: Arc<corvia_kernel::ops::GcKnowledgeHistory>,
    /// MCP bearer token for write operations. `Some` when server is bound to
    /// a non-loopback address (e.g., `0.0.0.0`). `None` for loopback-only binding.
    pub mcp_token: Option<String>,
    /// Whether Docker is available for spoke container queries.
    /// The actual Docker client is created on-demand inside `tokio::spawn` to
    /// avoid bollard's axum 0.7 types leaking into handler futures.
    pub docker_available: bool,
    /// Knowledge gap detector: accumulates low-confidence search signals.
    pub gap_detector: Arc<corvia_kernel::gap_detector::GapDetector>,
}

impl AppState {
    /// Load a snapshot of the RAG pipeline for this request.
    ///
    /// Uses `ArcSwap::load()` to obtain a `Guard` that pins the current
    /// pipeline. In-flight requests continue using the pipeline snapshot
    /// they loaded, even if a swap happens mid-request. The `Guard` is
    /// lightweight and does not block concurrent hot-swaps.
    pub fn rag_pipeline(&self) -> Option<arc_swap::Guard<Arc<corvia_kernel::rag_pipeline::RagPipeline>>> {
        self.rag.as_ref().map(|a| a.load())
    }
}

// --- Existing memory types ---

#[derive(Deserialize)]
pub struct WriteRequest {
    pub content: String,
    pub scope_id: String,
    pub source_version: Option<String>,
    pub metadata: Option<corvia_common::types::EntryMetadata>,
}

#[derive(Serialize)]
pub struct WriteResponse {
    pub id: String,
    pub embedded: bool,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub scope_id: String,
    pub limit: Option<usize>,
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
    pub workstream: Option<String>,
    /// Include Cold-tier entries via brute-force cosine scan (default false).
    #[serde(default)]
    pub include_cold: bool,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultDto>,
    pub count: usize,
}

#[derive(Serialize)]
pub struct SearchResultDto {
    pub content: String,
    pub score: f32,
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_score: Option<f32>,
}

impl From<SearchResult> for SearchResultDto {
    fn from(r: SearchResult) -> Self {
        Self {
            content: r.entry.content,
            score: r.score,
            source_file: r.entry.metadata.source_file,
            language: r.entry.metadata.language,
            chunk_type: r.entry.metadata.chunk_type,
            start_line: r.entry.metadata.start_line,
            end_line: r.entry.metadata.end_line,
            content_role: r.entry.metadata.content_role,
            source_origin: r.entry.metadata.source_origin,
            tier: Some(r.tier.to_string()),
            retention_score: r.retention_score,
        }
    }
}

// --- RAG types ---

#[derive(Deserialize)]
pub struct RagRequest {
    pub query: String,
    pub scope_id: String,
    pub limit: Option<usize>,
    pub expand_graph: Option<bool>,
}

#[derive(Serialize)]
pub struct RagResponseDto {
    pub answer: Option<String>,
    pub sources: Vec<SearchResultDto>,
    pub trace: corvia_kernel::rag_types::PipelineTrace,
}

// --- Agent coordination types ---

#[derive(Deserialize)]
pub struct RegisterAgentRequest {
    pub name: String,
    pub scope: String,
    pub display_name: Option<String>,
}

#[derive(Serialize)]
pub struct AgentResponse {
    pub agent_id: String,
    pub display_name: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SessionResponse {
    pub session_id: String,
    pub agent_id: String,
    pub state: String,
    pub entries_written: u64,
    pub entries_merged: u64,
}

#[derive(Deserialize)]
pub struct SessionWriteRequest {
    pub content: String,
    pub scope_id: String,
    pub source_version: Option<String>,
}

#[derive(Serialize)]
pub struct SessionWriteResponse {
    pub entry_id: String,
    pub entry_status: String,
}

#[derive(Serialize)]
pub struct SessionStateResponse {
    pub session_id: String,
    pub state: String,
    pub entries_written: u64,
    pub entries_merged: u64,
}

#[derive(Deserialize)]
pub struct RecoverRequest {
    pub action: String, // "resume", "commit", "rollback"
}

#[derive(Serialize)]
pub struct ConnectResponseDto {
    pub agent_id: String,
    pub active_sessions: Vec<SessionResponse>,
    pub recoverable_sessions: Vec<SessionResponse>,
}

impl From<&SessionRecord> for SessionResponse {
    fn from(s: &SessionRecord) -> Self {
        Self {
            session_id: s.session_id.clone(),
            agent_id: s.agent_id.clone(),
            state: format!("{:?}", s.state),
            entries_written: s.entries_written,
            entries_merged: s.entries_merged,
        }
    }
}

// --- Temporal, graph, and reasoning types ---

#[derive(Serialize)]
pub struct HistoryEntryDto {
    pub id: String,
    pub content: String,
    pub recorded_at: DateTime<Utc>,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub is_current: bool,
}

impl From<&KnowledgeEntry> for HistoryEntryDto {
    fn from(e: &KnowledgeEntry) -> Self {
        // Truncate content to 200 chars for the DTO
        let content: String = e.content.chars().take(200).collect();
        Self {
            id: e.id.to_string(),
            content,
            recorded_at: e.recorded_at,
            valid_from: e.valid_from,
            valid_to: e.valid_to,
            is_current: e.is_current(),
        }
    }
}

#[derive(Deserialize)]
pub struct EdgeQuery {
    pub relation: Option<String>,
}

#[derive(Serialize)]
pub struct EdgeDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub metadata: Option<serde_json::Value>,
}

impl From<&GraphEdge> for EdgeDto {
    fn from(e: &GraphEdge) -> Self {
        Self {
            from: e.from.to_string(),
            to: e.to.to_string(),
            relation: e.relation.clone(),
            metadata: e.metadata.clone(),
        }
    }
}

#[derive(Deserialize)]
pub struct EvolutionQuery {
    pub scope: String,
    pub since: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateEdgeRequest {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct ReasonRequest {
    pub scope_id: String,
    pub check: Option<String>,
}

#[derive(Serialize)]
pub struct FindingDto {
    pub check_type: String,
    pub confidence: f32,
    pub rationale: String,
    pub target_ids: Vec<String>,
}

// --- Session ingest types ---

/// Response from `POST /v1/ingest/sessions`.
#[derive(Debug, Serialize)]
pub struct SessionIngestResponse {
    pub sessions_ingested: usize,
    pub entries_stored: usize,
}

/// Request body for `POST /v1/classify/sessions`.
#[derive(Deserialize, Default)]
pub struct ClassifySessionsRequest {
    /// Max entries to classify per batch (default 10).
    pub batch_size: Option<usize>,
}

/// Response from `POST /v1/classify/sessions`.
#[derive(Debug, Serialize)]
pub struct ClassifySessionsResponse {
    pub processed: usize,
    pub promoted: usize,
    pub rejected: usize,
    /// Entries that failed classification (transient errors). Left in queue for retry.
    pub failed: usize,
    pub remaining: usize,
}

/// Request body for `POST /v1/classify/agent-teams`.
#[derive(Debug, Deserialize, Default)]
pub struct ClassifyAgentTeamsRequest {
    /// Max entries to extract per batch (default 10).
    pub batch_size: Option<usize>,
}

/// Response from `POST /v1/classify/agent-teams`.
#[derive(Debug, Serialize)]
pub struct ClassifyAgentTeamsResponse {
    pub processed: usize,
    pub extracted: usize,
    pub promoted: usize,
    /// Entries not found in store (data integrity issue).
    pub rejected: usize,
    /// Entries where LLM found nothing to extract (legitimate empty result).
    pub skipped: usize,
    /// Entries/items that failed (transient errors). Entries left in queue for retry.
    pub failed: usize,
    pub remaining: usize,
}

// --- Router ---

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
        // Session history endpoints
        .route("/v1/ingest/sessions", post(ingest_sessions))
        .route("/v1/classify/sessions", post(classify_sessions))
        .route("/v1/classify/agent-teams", post(classify_agent_teams))
        // Workspace ingestion endpoints
        .route("/v1/ingest", post(ingest_workspace_handler))
        .route("/v1/ingest/status", get(ingest_status_handler))
        // Tiered knowledge control endpoints
        .route("/v1/entries/{id}/pin", post(pin_entry_handler))
        .route("/v1/entries/{id}/unpin", post(unpin_entry_handler))
        .route("/v1/entries/{id}/inspect", get(inspect_entry_handler))
        .route("/v1/gc/status", get(gc_status_handler))
        .route("/v1/gc/run", post(gc_run_handler))
        .route("/v1/gc/history", get(gc_history_handler))
        .route("/health", get(health))
        // Embedded dashboard: serves static assets at root path.
        // API routes take priority over the fallback.
        .fallback(dashboard_handler)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// --- Helper ---

fn coordinator(state: &AppState) -> &AgentCoordinator {
    &state.coordinator
}

// --- Embedded dashboard handler ---

/// Serve embedded dashboard static assets. Falls back to index.html for SPA
/// routing. Returns 404 if no dashboard is embedded.
async fn dashboard_handler(uri: axum::http::Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    let csp = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'";

    // Try exact file match first
    if let Some(content) = DashboardAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        let cache = if path.starts_with("assets/") {
            // Hashed assets are safe for long-term caching
            "public, max-age=31536000, immutable"
        } else {
            "no-cache"
        };
        return (
            StatusCode::OK,
            [
                (axum::http::header::CONTENT_TYPE, mime.as_ref().to_string()),
                (axum::http::header::CACHE_CONTROL, cache.to_string()),
                (axum::http::header::CONTENT_SECURITY_POLICY, csp.to_string()),
            ],
            content.data.into_owned(),
        )
            .into_response();
    }

    // Don't serve dashboard HTML for API-like paths — return JSON 404 instead.
    if path.starts_with("v1/") || path.starts_with("api/") || path.starts_with("mcp") {
        return (
            StatusCode::NOT_FOUND,
            [
                (axum::http::header::CONTENT_TYPE, "application/json".to_string()),
                (axum::http::header::CACHE_CONTROL, "no-store".to_string()),
            ],
            b"{\"error\":\"Not Found\"}".to_vec(),
        ).into_response();
    }

    // SPA fallback: serve index.html for navigation paths
    if let Some(index) = DashboardAssets::get("index.html") {
        return (
            StatusCode::OK,
            [
                (axum::http::header::CONTENT_TYPE, "text/html".to_string()),
                (axum::http::header::CACHE_CONTROL, "no-cache".to_string()),
                (axum::http::header::CONTENT_SECURITY_POLICY, csp.to_string()),
            ],
            index.data.into_owned(),
        )
            .into_response();
    }

    // No dashboard embedded — return 404 with helpful message
    (
        StatusCode::NOT_FOUND,
        [
            (axum::http::header::CONTENT_TYPE, "text/html".to_string()),
            (axum::http::header::CACHE_CONTROL, "no-store".to_string()),
        ],
        b"<html><body><h1>Corvia</h1>\
          <p>Dashboard not embedded. Build with: \
          <code>cd tools/corvia-dashboard && npm run build</code>, \
          then recompile with: \
          <code>CORVIA_DASHBOARD_DIR=tools/corvia-dashboard/dist cargo build</code></p>\
          <p>API: <a href=\"/health\">/health</a> | \
          <a href=\"/api/dashboard/status\">/api/dashboard/status</a></p>\
          </body></html>"
            .to_vec(),
    )
        .into_response()
}

// --- Workspace ingestion handlers ---

#[derive(Deserialize)]
struct IngestRequest {
    repo: Option<String>,
}

#[derive(Serialize)]
struct IngestResponse {
    status: String,
    message: String,
}

/// POST /v1/ingest — trigger server-side workspace ingestion.
///
/// Returns 202 Accepted and runs ingestion in the background.
/// Returns 409 Conflict if ingestion is already in progress.
async fn ingest_workspace_handler(
    State(state): State<Arc<AppState>>,
    body: Option<Json<IngestRequest>>,
) -> impl IntoResponse {
    use corvia_kernel::ingest::{IngestState, IngestStatus, TracingProgress};

    // Atomically check-and-set: reject if already running (single write lock scope).
    // Use unwrap_or_else to recover from poisoned locks (data is still valid).
    {
        let mut status = state.ingest_status.write().unwrap_or_else(|e| e.into_inner());
        if status.state == IngestState::Running {
            return (
                StatusCode::CONFLICT,
                Json(IngestResponse {
                    status: "already_in_progress".into(),
                    message: "Workspace ingestion is already running. Check GET /v1/ingest/status for progress.".into(),
                }),
            ).into_response();
        }
        // Mark as running while still holding the write lock (atomic transition)
        *status = IngestStatus {
            state: IngestState::Running,
            started_at: Some(chrono::Utc::now()),
            finished_at: None,
            error: None,
            report: None,
        };
    }

    let repo_filter = body.and_then(|b| b.0.repo);

    // Snapshot config to avoid mid-ingest changes
    let config = state.config.read().unwrap().clone();

    // Clone state Arc for the background task (gives access to session_ingest_lock)
    let state_bg = state.clone();

    // Spawn on a blocking thread because ProcessAdapter uses synchronous I/O
    // (stdin/stdout pipes to child processes). Running on the async runtime would
    // stall tokio worker threads for the duration of adapter IPC.
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let progress = TracingProgress;
        let repo_filter_ref = repo_filter.as_deref();

        // Wrap in a timeout (10 minutes)
        let result = rt.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(600),
                corvia_kernel::ingest::run_workspace_ingest(corvia_kernel::ingest::WorkspaceIngestCtx {
                    root: &state_bg.workspace_root,
                    config: &config,
                    store: state_bg.store.clone(),
                    graph: state_bg.graph.clone(),
                    engine: state_bg.engine.clone(),
                    repo_filter: repo_filter_ref,
                    session_lock: Some(&state_bg.session_ingest_lock),
                    progress: &progress,
                }),
            ).await
        });

        let mut status = state_bg.ingest_status.write().unwrap_or_else(|e| e.into_inner());
        match result {
            Ok(Ok(report)) => {
                info!(
                    total_chunks = report.total_chunks,
                    repos = report.repos.len(),
                    "Workspace ingestion completed"
                );
                *status = IngestStatus {
                    state: IngestState::Completed,
                    started_at: status.started_at,
                    finished_at: Some(chrono::Utc::now()),
                    error: None,
                    report: Some(report),
                };
            }
            Ok(Err(e)) => {
                warn!("Workspace ingestion failed: {e}");
                *status = IngestStatus {
                    state: IngestState::Failed,
                    started_at: status.started_at,
                    finished_at: Some(chrono::Utc::now()),
                    error: Some(format!("{e}")),
                    report: None,
                };
            }
            Err(_) => {
                warn!("Workspace ingestion timed out after 600s");
                *status = IngestStatus {
                    state: IngestState::Failed,
                    started_at: status.started_at,
                    finished_at: Some(chrono::Utc::now()),
                    error: Some("Ingestion timed out after 10 minutes".into()),
                    report: None,
                };
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(IngestResponse {
            status: "started".into(),
            message: "Workspace ingestion started. Check GET /v1/ingest/status for progress.".into(),
        }),
    ).into_response()
}

/// GET /v1/ingest/status — poll ingestion status.
async fn ingest_status_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let status = state.ingest_status.read().unwrap_or_else(|e| e.into_inner()).clone();
    (StatusCode::OK, Json(status))
}

// --- Tiered knowledge control handlers ---

#[derive(Deserialize)]
struct PinRequest {
    agent_id: String,
}

async fn pin_entry_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<PinRequest>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;
    let result = corvia_kernel::ops::pin_entry(&state.store, &uuid, &req.agent_id).await
        .map_err(|e| map_corvia_err("Pin failed", e))?;
    let json = serde_json::to_value(result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization failed: {e}")))?;
    Ok(Json(json))
}

async fn unpin_entry_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;
    let result = corvia_kernel::ops::unpin_entry(&state.store, &uuid).await
        .map_err(|e| map_corvia_err("Unpin failed", e))?;
    let json = serde_json::to_value(result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization failed: {e}")))?;
    Ok(Json(json))
}

async fn inspect_entry_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;
    let result = corvia_kernel::ops::inspect_entry(&state.store, &state.graph, &uuid).await
        .map_err(|e| map_corvia_err("Inspect failed", e))?;
    let json = serde_json::to_value(result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization failed: {e}")))?;
    Ok(Json(json))
}

async fn gc_status_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let default_scope = state.default_scope_id.as_deref().unwrap_or("default");
    let scope_id = params.get("scope_id").map(|s| s.as_str()).unwrap_or(default_scope);

    let status = corvia_kernel::ops::system_status(
        state.store.clone(),
        &state.coordinator,
        scope_id,
    ).await.map_err(|e| map_corvia_err("GC status failed", e))?;

    // Check if forgetting is enabled
    let config = state.config.read()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Config lock: {e}")))?;
    let forgetting_enabled = config.forgetting.as_ref().map(|f| f.enabled).unwrap_or(false);

    Ok(Json(serde_json::json!({
        "scope_id": scope_id,
        "forgetting_enabled": forgetting_enabled,
        "tier_distribution": {
            "hot": status.tier_distribution.hot,
            "warm": status.tier_distribution.warm,
            "cold": status.tier_distribution.cold,
            "forgotten": status.tier_distribution.forgotten,
        },
        "total_entries": status.entry_count,
    })))
}

async fn gc_run_handler(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let config = state.config.read()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Config lock: {e}")))?
        .clone();

    let report = corvia_kernel::ops::gc_knowledge_run(
        &state.store,
        &state.graph,
        &state.data_dir,
        &config,
    ).await.map_err(|e| map_corvia_err("GC run failed", e))?;

    // Serialize before push to avoid unnecessary clone
    let json = serde_json::to_value(&report)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization failed: {e}")))?;

    // Store report in knowledge GC history
    state.gc_knowledge_history.push(report);

    Ok(Json(json))
}

async fn gc_history_handler(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let reports = state.gc_knowledge_history.all();

    Ok(Json(serde_json::json!({
        "cycles": serde_json::to_value(&reports).unwrap_or_default(),
        "count": reports.len(),
    })))
}

// --- Existing handlers ---

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"status": "starting", "message": "Rebuilding index, not yet ready"})))
    }
}

async fn write_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WriteRequest>,
) -> std::result::Result<Json<WriteResponse>, (StatusCode, String)> {
    let embedding = state.engine.embed(&req.content).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;

    let mut entry = KnowledgeEntry::new(
        req.content,
        req.scope_id,
        req.source_version.unwrap_or_else(|| "manual".into()),
    ).with_embedding(embedding);

    if let Some(metadata) = req.metadata {
        entry = entry.with_metadata(metadata);
    }

    let id = entry.id.to_string();

    state.store.insert(&entry).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Storage failed: {e}")))?;

    info!("Stored memory {id}");

    Ok(Json(WriteResponse { id, embedded: true }))
}

async fn search_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> std::result::Result<Json<SearchResponse>, (StatusCode, String)> {
    let limit = req.limit.unwrap_or(10);
    let content_role = req.content_role.clone();
    let source_origin = req.source_origin.clone();
    let workstream = req.workstream.clone();

    // Route through RAG pipeline if available (fixes ContextBuilder bypass bug)
    if let Some(rag) = state.rag_pipeline() {
        let opts = corvia_kernel::rag_types::RetrievalOpts {
            limit,
            expand_graph: false, // search endpoint: pure vector (context/ask use graph)
            content_role,
            source_origin,
            workstream,
            include_cold: req.include_cold,
            ..Default::default()
        };
        let response = rag.context(&req.query, &req.scope_id, Some(opts)).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Search failed: {e}")))?;

        let results: Vec<SearchResultDto> = response.context.sources.into_iter().map(Into::into).collect();
        let count = results.len();
        return Ok(Json(SearchResponse { results, count }));
    }

    // Fallback: raw store search (no RAG pipeline configured)
    let query_embedding = state.engine.embed(&req.query).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;

    let search_limit = if content_role.is_some() || source_origin.is_some() || workstream.is_some() {
        limit * 3
    } else {
        limit
    };
    let results = state.store.search(&query_embedding, &req.scope_id, search_limit).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Search failed: {e}")))?;

    let mut results = corvia_kernel::retriever::post_filter_metadata(
        results,
        content_role.as_deref(),
        source_origin.as_deref(),
        workstream.as_deref(),
    );
    results.truncate(limit);

    let count = results.len();
    let results: Vec<SearchResultDto> = results.into_iter().map(Into::into).collect();

    Ok(Json(SearchResponse { results, count }))
}

// --- RAG handlers ---

async fn rag_context(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagRequest>,
) -> std::result::Result<Json<RagResponseDto>, (StatusCode, String)> {
    let rag = state.rag_pipeline()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit: req.limit.unwrap_or(10),
        expand_graph: req.expand_graph.unwrap_or(true),
        ..Default::default()
    };

    let response = rag.context(&req.query, &req.scope_id, Some(opts)).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("RAG failed: {e}")))?;

    Ok(Json(RagResponseDto {
        answer: response.answer,
        sources: response.context.sources.into_iter().map(Into::into).collect(),
        trace: response.trace,
    }))
}

async fn rag_ask(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagRequest>,
) -> std::result::Result<Json<RagResponseDto>, (StatusCode, String)> {
    let rag = state.rag_pipeline()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit: req.limit.unwrap_or(10),
        expand_graph: req.expand_graph.unwrap_or(true),
        ..Default::default()
    };

    // Timeout prevents LLM inference from blocking the server indefinitely.
    // The /v1/ask endpoint can OOM or hang if the model is large.
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(RAG_ASK_TIMEOUT_SECS),
        rag.ask(&req.query, &req.scope_id, Some(opts)),
    ).await
        .map_err(|_| (StatusCode::GATEWAY_TIMEOUT, format!("RAG ask timed out after {RAG_ASK_TIMEOUT_SECS}s")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("RAG failed: {e}")))?;

    Ok(Json(RagResponseDto {
        answer: response.answer,
        sources: response.context.sources.into_iter().map(Into::into).collect(),
        trace: response.trace,
    }))
}

// --- Agent coordination handlers ---

async fn register_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterAgentRequest>,
) -> std::result::Result<(StatusCode, Json<AgentResponse>), (StatusCode, String)> {
    let coord = coordinator(&state);

    let identity = AgentIdentity::Registered {
        agent_id: format!("{}::{}", req.scope, req.name),
        api_key: None,
    };
    let display_name = req.display_name.unwrap_or_else(|| req.name.clone());

    let record = coord.register_agent(
        &identity,
        &display_name,
        AgentPermission::ReadWrite { scopes: vec![req.scope] },
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Registration failed: {e}")))?;

    Ok((StatusCode::CREATED, Json(AgentResponse {
        agent_id: record.agent_id,
        display_name: record.display_name,
        status: format!("{:?}", record.status),
    })))
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
) -> std::result::Result<(StatusCode, Json<ConnectResponseDto>), (StatusCode, String)> {
    let coord = coordinator(&state);

    // Create session (with staging for registered agents)
    let _session = coord.create_session(&agent_id, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Session creation failed: {e}")))?;

    // Return connect info
    let connect = coord.connect(&agent_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Connect failed: {e}")))?;

    Ok((StatusCode::CREATED, Json(ConnectResponseDto {
        agent_id: connect.agent_id,
        active_sessions: connect.active_sessions.iter().map(Into::into).collect(),
        recoverable_sessions: connect.recoverable_sessions.iter().map(Into::into).collect(),
    })))
}

async fn heartbeat(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let coord = coordinator(&state);
    coord.heartbeat(&session_id)
        .map_err(|e| map_corvia_err("Heartbeat failed", e))?;
    Ok(StatusCode::OK)
}

async fn session_write(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<SessionWriteRequest>,
) -> std::result::Result<(StatusCode, Json<SessionWriteResponse>), (StatusCode, String)> {
    let coord = coordinator(&state);

    let entry = coord.write_entry(
        &session_id,
        &req.content,
        &req.scope_id,
        req.source_version.as_deref().unwrap_or("manual"),
        None,
        None,
    ).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Write failed: {e}")))?;

    Ok((StatusCode::CREATED, Json(SessionWriteResponse {
        entry_id: entry.id.to_string(),
        entry_status: format!("{:?}", entry.entry_status),
    })))
}

async fn commit_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let coord = coordinator(&state);
    coord.commit_session(&session_id).await
        .map_err(|e| map_corvia_err("Commit failed", e))?;
    Ok(StatusCode::OK)
}

async fn rollback_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let coord = coordinator(&state);
    coord.rollback_session(&session_id)
        .map_err(|e| map_corvia_err("Rollback failed", e))?;
    Ok(StatusCode::OK)
}

async fn recover_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<RecoverRequest>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let coord = coordinator(&state);
    let action = match req.action.as_str() {
        "resume" => RecoveryAction::Resume,
        "commit" => RecoveryAction::Commit,
        "rollback" => RecoveryAction::Rollback,
        other => return Err((StatusCode::BAD_REQUEST, format!("Unknown action: {other}"))),
    };
    coord.recover(&session_id, action).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Recovery failed: {e}")))?;
    Ok(StatusCode::OK)
}

async fn session_state(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> std::result::Result<Json<SessionStateResponse>, (StatusCode, String)> {
    let coord = coordinator(&state);
    let session = coord.sessions.get(&session_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to get session: {e}")))?
        .ok_or((StatusCode::NOT_FOUND, format!("Session {session_id} not found")))?;

    Ok(Json(SessionStateResponse {
        session_id: session.session_id,
        state: format!("{:?}", session.state),
        entries_written: session.entries_written,
        entries_merged: session.entries_merged,
    }))
}

// --- Temporal, graph, and reasoning handlers ---

async fn entry_history(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<Vec<HistoryEntryDto>>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;

    let chain = state.temporal.history(&uuid).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("History query failed: {e}")))?;

    let entries: Vec<HistoryEntryDto> = chain.iter().map(Into::into).collect();
    Ok(Json(entries))
}

async fn entry_edges(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(query): Query<EdgeQuery>,
) -> std::result::Result<Json<Vec<EdgeDto>>, (StatusCode, String)> {
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {e}")))?;

    let edges = state.graph.edges(&uuid, EdgeDirection::Both).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Graph query failed: {e}")))?;

    let filtered: Vec<EdgeDto> = edges.iter()
        .filter(|e| {
            query.relation.as_ref().is_none_or(|r| &e.relation == r)
        })
        .map(Into::into)
        .collect();

    Ok(Json(filtered))
}

async fn evolution(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EvolutionQuery>,
) -> std::result::Result<Json<Vec<HistoryEntryDto>>, (StatusCode, String)> {
    let now = Utc::now();
    let since = query.since.as_deref().unwrap_or("7d");
    let duration = parse_duration(since)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid duration: {e}")))?;
    let from = now - duration;

    let entries = state.temporal.evolution(&query.scope, from, now).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Evolution query failed: {e}")))?;

    let result: Vec<HistoryEntryDto> = entries.iter().map(Into::into).collect();
    Ok(Json(result))
}

async fn create_edge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateEdgeRequest>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let from = uuid::Uuid::parse_str(&req.from)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid 'from' UUID: {e}")))?;
    let to = uuid::Uuid::parse_str(&req.to)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid 'to' UUID: {e}")))?;

    state.graph.relate(&from, &req.relation, &to, req.metadata).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Relate failed: {e}")))?;

    Ok(StatusCode::CREATED)
}

async fn reason(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReasonRequest>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Load entries on a blocking thread to avoid starving the async runtime.
    let data_dir = state.data_dir.clone();
    let scope_id = req.scope_id.clone();
    let entries = tokio::task::spawn_blocking(move || {
        corvia_kernel::knowledge_files::read_scope(&data_dir, &scope_id)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join failed: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load entries: {e}")))?;

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*state.store, &*state.graph);

    let findings = if let Some(ref check_str) = req.check {
        let check_type = check_str.parse::<corvia_kernel::reasoner::CheckType>()
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        reasoner.run_check(&entries, &req.scope_id, check_type).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Reasoning failed: {e}")))?
    } else {
        reasoner.run_all(&entries, &req.scope_id).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Reasoning failed: {e}")))?
    };

    let items: Vec<FindingDto> = findings.iter().map(|f| FindingDto {
        check_type: f.check_type.as_str().to_string(),
        confidence: f.confidence,
        rationale: f.rationale.clone(),
        target_ids: f.target_ids.iter().map(|id| id.to_string()).collect(),
    }).collect();

    Ok(Json(serde_json::json!({
        "scope_id": req.scope_id,
        "findings": items,
        "count": items.len(),
    })))
}

// --- Session ingest handlers ---

/// Sessions directory path (always ~/.claude/sessions, no user override to prevent path traversal).
fn sessions_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    std::path::PathBuf::from(home).join(".claude").join("sessions")
}

/// Ingest Claude Code session history into the `user-history` scope.
///
/// Discovers and runs the `corvia-adapter-claude-sessions` adapter via IPC,
/// then embeds and stores the resulting entries. Serialized via `session_ingest_lock`
/// to prevent concurrent adapter runs from racing on `.ingested` state.
async fn ingest_sessions(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<Json<SessionIngestResponse>, (StatusCode, String)> {
    // Serialize: only one ingest/classify at a time to protect state files
    let _guard = state.session_ingest_lock.lock().await;

    let sessions_path = sessions_dir();
    if !sessions_path.is_dir() {
        return Ok(Json(SessionIngestResponse {
            sessions_ingested: 0,
            entries_stored: 0,
        }));
    }

    // Read config for adapter search dirs
    let config = state.config.read()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Config lock poisoned: {e}")))?
        .clone();

    // Check user-history scope exists
    let has_scope = config.scope.as_ref()
        .is_some_and(|scopes| scopes.iter().any(|s| s.id == USER_HISTORY_SCOPE));
    if !has_scope {
        return Err((StatusCode::BAD_REQUEST, format!("No '{}' scope in config", USER_HISTORY_SCOPE)));
    }

    // Discover and run the adapter (blocking I/O — must be in spawn_blocking)
    let extra_dirs: Vec<String> = config.adapters
        .as_ref()
        .map(|a| a.search_dirs.clone())
        .unwrap_or_default();
    let sessions_dir_str = sessions_path.to_string_lossy().to_string();

    let source_files = tokio::task::spawn_blocking(move || -> Result<Vec<corvia_kernel::chunking_strategy::SourceFile>, String> {
        let discovered = corvia_kernel::adapter_discovery::discover_adapters(&extra_dirs);
        let adapter_info = discovered.iter()
            .find(|a| a.metadata.domain == CLAUDE_SESSIONS_ADAPTER)
            .ok_or_else(|| format!("{CLAUDE_SESSIONS_ADAPTER} adapter not found on PATH"))?;

        let mut process = corvia_kernel::process_adapter::ProcessAdapter::new(
            adapter_info.binary_path.clone(),
            adapter_info.metadata.clone(),
        );
        process.spawn()?;
        // ProcessAdapter::Drop calls shutdown() if we bail early, but explicit
        // shutdown provides better error reporting.
        let files = match process.ingest(&sessions_dir_str, USER_HISTORY_SCOPE) {
            Ok(f) => f,
            Err(e) => {
                let _ = process.shutdown();
                return Err(e);
            }
        };
        process.shutdown()?;
        Ok(files)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join error: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Adapter error: {e}")))?;

    if source_files.is_empty() {
        return Ok(Json(SessionIngestResponse {
            sessions_ingested: 0,
            entries_stored: 0,
        }));
    }

    // Count unique sessions (distinct file_path values = session IDs)
    let session_count = {
        let mut seen = std::collections::HashSet::new();
        for sf in &source_files {
            seen.insert(&sf.metadata.file_path);
        }
        seen.len()
    };

    info!(
        "Session ingest: {} sessions → {} turn entries",
        session_count, source_files.len()
    );

    // Build entries directly from adapter output. Each SourceFile is one turn,
    // already chunked by the adapter. Skipping the ChunkingPipeline preserves
    // the per-turn source_version (e.g. "ses-abc:turn-1") which the pipeline
    // would collapse to just the file_path.
    let entries: Vec<KnowledgeEntry> = source_files.iter().map(|sf| {
        let mut entry = KnowledgeEntry::new(
            sf.content.clone(),
            USER_HISTORY_SCOPE.to_string(),
            sf.metadata.source_version.clone(),
        );
        entry.workstream = sf.metadata.workstream.clone().unwrap_or_default();
        entry.metadata = corvia_common::types::EntryMetadata {
            source_file: Some(sf.metadata.file_path.clone()),
            language: sf.metadata.language.clone(),
            chunk_type: Some("session-turn".into()),
            start_line: None,
            end_line: None,
            content_role: sf.metadata.content_role.clone(),
            source_origin: sf.metadata.source_origin.clone(),
        };
        entry
    }).collect();

    // Embed and store in batches
    let total = entries.len();
    let mut stored = 0;
    for batch in entries.chunks(corvia_kernel::introspect::EMBED_BATCH_SIZE) {
        let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
        let embeddings = state.engine.embed_batch(&texts).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;
        for (entry, embedding) in batch.iter().zip(embeddings) {
            let mut entry = entry.clone();
            entry.embedding = Some(embedding);
            state.store.insert(&entry).await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Storage failed: {e}")))?;
            stored += 1;
        }
    }

    info!("Session ingest complete: {stored}/{total} entries stored");

    Ok(Json(SessionIngestResponse {
        sessions_ingested: session_count,
        entries_stored: stored,
    }))
}

/// Classify queued session entries and promote product-relevant ones to the `corvia` scope.
///
/// Reads `.classify-queue`, sends each entry to the GenerationEngine for classification,
/// and writes promoted entries to the `corvia` scope. Transient LLM failures leave entries
/// in the queue for retry. Serialized via `session_ingest_lock`.
async fn classify_sessions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClassifySessionsRequest>,
) -> std::result::Result<Json<ClassifySessionsResponse>, (StatusCode, String)> {
    // Serialize: only one ingest/classify at a time
    let _guard = state.session_ingest_lock.lock().await;

    let batch_size = req.batch_size.unwrap_or(10);
    let queue_path = sessions_dir().join(".classify-queue");

    // Read queue (blocking I/O, but classify-queue is tiny — a few KB at most)
    let queue_content = tokio::fs::read_to_string(&queue_path).await.unwrap_or_default();
    let all_entries: Vec<String> = queue_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();

    if all_entries.is_empty() {
        return Ok(Json(ClassifySessionsResponse {
            processed: 0,
            promoted: 0,
            rejected: 0,
            failed: 0,
            remaining: 0,
        }));
    }

    // Get generation engine
    let generator = state.rag_pipeline()
        .and_then(|rag| rag.generator().cloned())
        .ok_or_else(|| (
            StatusCode::SERVICE_UNAVAILABLE,
            "GenerationEngine not configured — cannot classify sessions".into(),
        ))?;

    let to_process: Vec<String> = all_entries.iter().take(batch_size).cloned().collect();
    let mut remaining_entries: Vec<String> = all_entries.iter().skip(batch_size).cloned().collect();

    // Load user-history entries for source_version lookup (blocking I/O in spawn_blocking)
    let data_dir = state.data_dir.clone();
    let uh_entries = tokio::task::spawn_blocking(move || {
        corvia_kernel::knowledge_files::read_scope(&data_dir, USER_HISTORY_SCOPE)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join error: {e}")))?
    .map_err(|e| {
        warn!("Failed to read user-history knowledge files: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Knowledge file read failed: {e}"))
    })?;

    let sv_lookup: std::collections::HashMap<&str, &KnowledgeEntry> = uh_entries.iter()
        .map(|e| (e.source_version.as_str(), e))
        .collect();

    let mut promoted = 0;
    let mut rejected = 0;
    let mut failed = 0;

    for entry_ref in &to_process {
        // Look up entry by source_version (e.g. "ses-abc:turn-1")
        let entry = match sv_lookup.get(entry_ref.as_str()) {
            Some(e) => (*e).clone(),
            None => {
                warn!("Entry not found for {entry_ref}, removing from queue");
                rejected += 1;
                continue;
            }
        };

        // Ask the LLM to classify
        let system_prompt = "You are a knowledge classification assistant. \
            Answer YES or NO followed by one sentence of rationale. \
            If YES, also state the content type: decision, design, finding, or learning.";
        let user_message = format!(
            "Does the following conversation turn contain a product decision, \
            architectural discussion, or research finding relevant to building \
            the corvia software product?\n\n---\n{}\n---",
            entry.content
        );

        let result = match generator.generate(system_prompt, &user_message).await {
            Ok(r) => r,
            Err(e) => {
                // Transient failure: leave in queue for retry
                warn!("Classification LLM call failed for {entry_ref}: {e}");
                remaining_entries.push(entry_ref.clone());
                failed += 1;
                continue;
            }
        };

        let answer = result.text.trim().to_uppercase();
        if answer.starts_with("YES") {
            // Infer content_role from the LLM response first, fall back to keyword heuristic
            let content_role = infer_content_role_from_llm(&result.text)
                .unwrap_or_else(|| infer_product_content_role(&entry.content));

            // Write a copy to the corvia scope
            let mut promoted_entry = KnowledgeEntry::new(
                entry.content.clone(),
                "corvia".to_string(),
                entry.source_version.clone(),
            );
            promoted_entry.workstream = entry.workstream.clone();
            promoted_entry.metadata = corvia_common::types::EntryMetadata {
                source_file: entry.metadata.source_file.clone(),
                language: None,
                chunk_type: Some("session-promoted".into()),
                start_line: None,
                end_line: None,
                content_role: Some(content_role),
                source_origin: Some("claude:promoted".into()),
            };

            match state.engine.embed(&promoted_entry.content).await {
                Ok(emb) => {
                    promoted_entry.embedding = Some(emb);
                    match state.store.insert(&promoted_entry).await {
                        Ok(_) => {
                            info!("Promoted {entry_ref} to corvia scope as {}", promoted_entry.metadata.content_role.as_deref().unwrap_or("?"));
                            promoted += 1;
                        }
                        Err(e) => {
                            warn!("Failed to store promoted entry {entry_ref}: {e}");
                            remaining_entries.push(entry_ref.clone());
                            failed += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to embed promoted entry {entry_ref}: {e}");
                    remaining_entries.push(entry_ref.clone());
                    failed += 1;
                }
            }
        } else {
            rejected += 1;
        }
    }

    // Atomically rewrite the queue (write temp + rename, POSIX atomic on same filesystem)
    let remaining = remaining_entries.len();
    let tmp_path = queue_path.with_extension("tmp");
    let new_content = if remaining_entries.is_empty() {
        String::new()
    } else {
        format!("{}\n", remaining_entries.join("\n"))
    };
    if let Err(e) = tokio::fs::write(&tmp_path, &new_content).await {
        warn!("Failed to write classify queue temp file: {e}");
    } else if let Err(e) = tokio::fs::rename(&tmp_path, &queue_path).await {
        warn!("Failed to rename classify queue temp file: {e}");
    }

    Ok(Json(ClassifySessionsResponse {
        processed: to_process.len(),
        promoted,
        rejected,
        failed,
        remaining,
    }))
}

/// Agent-teams staging directory path.
fn agent_teams_staging_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    std::path::PathBuf::from(home)
        .join(".corvia")
        .join("staging")
        .join("agent-teams")
}

/// Extract decisions, findings, and questions from queued agent-teams message entries,
/// then promote product-relevant extractions to the `corvia` scope.
///
/// Reads `.classify-queue` from the agent-teams staging directory, sends each entry
/// to the GenerationEngine for extraction, creates new entries with `extracted_from`
/// graph edges, and promotes product-relevant extractions. Serialized via
/// `session_ingest_lock`.
async fn classify_agent_teams(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClassifyAgentTeamsRequest>,
) -> std::result::Result<Json<ClassifyAgentTeamsResponse>, (StatusCode, String)> {
    let _guard = state.session_ingest_lock.lock().await;

    let batch_size = req.batch_size.unwrap_or(10);
    let queue_path = agent_teams_staging_dir().join(".classify-queue");

    let queue_content = tokio::fs::read_to_string(&queue_path).await.unwrap_or_default();
    let all_entries: Vec<String> = queue_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();

    if all_entries.is_empty() {
        return Ok(Json(ClassifyAgentTeamsResponse {
            processed: 0,
            extracted: 0,
            promoted: 0,
            rejected: 0,
            skipped: 0,
            failed: 0,
            remaining: 0,
        }));
    }

    let generator = state.rag_pipeline()
        .and_then(|rag| rag.generator().cloned())
        .ok_or_else(|| (
            StatusCode::SERVICE_UNAVAILABLE,
            "GenerationEngine not configured - cannot extract from agent-teams".into(),
        ))?;

    let to_process: Vec<String> = all_entries.iter().take(batch_size).cloned().collect();
    let mut remaining_entries: Vec<String> = all_entries.iter().skip(batch_size).cloned().collect();

    // Load user-history entries for source_version lookup
    let data_dir = state.data_dir.clone();
    let uh_entries = tokio::task::spawn_blocking(move || {
        corvia_kernel::knowledge_files::read_scope(&data_dir, USER_HISTORY_SCOPE)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task join error: {e}")))?
    .map_err(|e| {
        warn!("Failed to read user-history knowledge files: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Knowledge file read failed: {e}"))
    })?;

    let sv_lookup: std::collections::HashMap<&str, &KnowledgeEntry> = uh_entries.iter()
        .map(|e| (e.source_version.as_str(), e))
        .collect();

    let mut extracted = 0usize;
    let mut promoted = 0usize;
    let mut rejected = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for entry_ref in &to_process {
        let entry = match sv_lookup.get(entry_ref.as_str()) {
            Some(e) => *e,
            None => {
                warn!("Entry not found for {entry_ref}, removing from queue");
                rejected += 1;
                continue;
            }
        };

        // Step 1: Extract decisions, findings, and questions via LLM
        let extraction_system = "You are a knowledge extraction assistant. \
            Extract key decisions, findings, and open questions from team discussions. \
            Return a JSON array of objects. Each object must have: \
            \"type\" (one of \"decision\", \"finding\", \"question\"), \
            \"content\" (the conclusion or question), \
            \"rationale\" (supporting reasoning), \
            \"dissent\" (any disagreement, or empty string if none). \
            Return [] if there is nothing worth extracting.";
        let extraction_message = format!(
            "Extract key decisions, findings, and open questions from this team discussion. \
            For each, state the conclusion, the rationale, and any dissent.\n\n---\n{}\n---",
            entry.content
        );

        let extraction_result = match generator.generate(extraction_system, &extraction_message).await {
            Ok(r) => r,
            Err(e) => {
                warn!("Extraction LLM call failed for {entry_ref}: {e}");
                remaining_entries.push(entry_ref.clone());
                failed += 1;
                continue;
            }
        };

        // Parse JSON array from LLM response
        let items = parse_extraction_items(&extraction_result.text);
        if items.is_empty() {
            // Nothing extracted; remove from queue (not a failure)
            skipped += 1;
            continue;
        }

        // Derive source_origin from entry's existing source_origin or from source_version
        let source_origin = entry.metadata.source_origin.clone()
            .unwrap_or_else(|| {
                let team_name = entry_ref.split(':').next().unwrap_or("unknown");
                format!("claude:team:{team_name}")
            });

        for (i, item) in items.iter().enumerate() {
            let content = format!(
                "# {} (extracted from team discussion)\n\n{}\n\n**Rationale:** {}{}",
                capitalize(&item.item_type),
                item.content,
                item.rationale,
                if item.dissent.is_empty() {
                    String::new()
                } else {
                    format!("\n\n**Dissent:** {}", item.dissent)
                }
            );

            // Use source message group's source_version as prefix to avoid cross-batch collisions
            let source_version = format!("{}:extracted:{}", entry_ref, i + 1);

            let mut extracted_entry = KnowledgeEntry::new(
                content.clone(),
                USER_HISTORY_SCOPE.to_string(),
                source_version,
            );
            extracted_entry.metadata = corvia_common::types::EntryMetadata {
                source_file: None,
                language: None,
                chunk_type: Some("team-extracted".into()),
                start_line: None,
                end_line: None,
                content_role: Some(item.item_type.clone()),
                source_origin: Some(source_origin.clone()),
            };

            // Embed and store the extracted entry
            let emb = match state.engine.embed(&content).await {
                Ok(emb) => emb,
                Err(e) => {
                    warn!("Failed to embed extracted entry: {e}");
                    failed += 1;
                    continue;
                }
            };
            extracted_entry.embedding = Some(emb);

            let extracted_id = extracted_entry.id;
            if let Err(e) = state.store.insert(&extracted_entry).await {
                warn!("Failed to store extracted entry: {e}");
                failed += 1;
                continue;
            }

            extracted += 1;

            // Create extracted_from graph edge
            if let Err(e) = state.graph.relate(
                &extracted_id,
                "extracted_from",
                &entry.id,
                None,
            ).await {
                warn!("Failed to create extracted_from edge: {e}");
            }

            // Step 2: Promotion classifier - is this relevant to the corvia product?
            let promo_system = "You are a knowledge classification assistant. \
                Answer YES or NO followed by one sentence of rationale.";
            let promo_message = format!(
                "Is the following extracted insight relevant to building \
                the corvia software product?\n\n---\n{}\n---",
                content
            );

            match generator.generate(promo_system, &promo_message).await {
                Ok(promo_result) => {
                    let answer = promo_result.text.trim().to_uppercase();
                    if answer.starts_with("YES") {
                        let mut promoted_entry = KnowledgeEntry::new(
                            extracted_entry.content.clone(),
                            "corvia".to_string(),
                            extracted_entry.source_version.clone(),
                        );
                        promoted_entry.metadata = corvia_common::types::EntryMetadata {
                            source_file: None,
                            language: None,
                            chunk_type: Some("team-promoted".into()),
                            start_line: None,
                            end_line: None,
                            content_role: extracted_entry.metadata.content_role.clone(),
                            source_origin: Some("claude:promoted".into()),
                        };
                        // Reuse embedding from extracted entry (identical content)
                        promoted_entry.embedding = extracted_entry.embedding.clone();

                        if let Err(e) = state.store.insert(&promoted_entry).await {
                            warn!("Failed to store promoted entry: {e}");
                        } else {
                            info!(
                                "Promoted {entry_ref} extraction {} to corvia scope",
                                i + 1
                            );
                            promoted += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("Promotion LLM call failed: {e}");
                    // Extraction succeeded, promotion failed - not critical
                }
            }
        }
    }

    // Atomically rewrite the queue
    let remaining = remaining_entries.len();
    let tmp_path = queue_path.with_extension("tmp");
    let new_content = if remaining_entries.is_empty() {
        String::new()
    } else {
        format!("{}\n", remaining_entries.join("\n"))
    };
    if let Err(e) = tokio::fs::write(&tmp_path, &new_content).await {
        warn!("Failed to write classify queue temp file: {e}");
    } else if let Err(e) = tokio::fs::rename(&tmp_path, &queue_path).await {
        warn!("Failed to rename classify queue temp file: {e}");
    }

    Ok(Json(ClassifyAgentTeamsResponse {
        processed: to_process.len(),
        extracted,
        promoted,
        rejected,
        skipped,
        failed,
        remaining,
    }))
}

/// Item extracted from a team message group by the LLM.
#[derive(Debug)]
struct ExtractionItem {
    item_type: String,
    content: String,
    rationale: String,
    dissent: String,
}

/// Parse the LLM extraction response into structured items.
/// Expects a JSON array but handles common LLM response quirks.
fn parse_extraction_items(response: &str) -> Vec<ExtractionItem> {
    // Try to find a JSON array in the response (LLMs often wrap in markdown code blocks)
    let text = response.trim();
    let json_str = if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            &text[start..=end]
        } else {
            return Vec::new();
        }
    } else {
        return Vec::new();
    };

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    parsed.iter().filter_map(|item| {
        let item_type = item.get("type")?.as_str()?.to_string();
        if !["decision", "finding", "question"].contains(&item_type.as_str()) {
            return None;
        }
        let content = item.get("content")?.as_str()?.to_string();
        let rationale = item.get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let dissent = item.get("dissent")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Some(ExtractionItem { item_type, content, rationale, dissent })
    }).collect()
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

/// Try to extract content_role from the LLM classification response.
/// The prompt asks the LLM to state "decision", "design", "finding", or "learning".
fn infer_content_role_from_llm(response: &str) -> Option<String> {
    let lower = response.to_lowercase();
    for role in &["decision", "design", "finding", "learning"] {
        if lower.contains(role) {
            return Some((*role).to_string());
        }
    }
    None
}

/// Fallback: infer content_role from turn content via keyword heuristic.
/// Only used when the LLM response doesn't mention a specific role.
fn infer_product_content_role(content: &str) -> String {
    // Only check the first 300 chars (prompt area) to reduce false positives
    // from tool output that happens to mention these words.
    let prefix: String = content.chars().take(300).collect();
    let lower = prefix.to_lowercase();
    if lower.contains("decision") || lower.contains("decided") || lower.contains("chose") {
        "decision".into()
    } else if lower.contains("design") || lower.contains("architecture") || lower.contains("rfc") {
        "design".into()
    } else if lower.contains("research") || lower.contains("finding") || lower.contains("discovered") {
        "finding".into()
    } else if lower.contains("learned") || lower.contains("lesson") || lower.contains("pattern") {
        "learning".into()
    } else {
        "finding".into()
    }
}

// --- Helpers ---

/// Parse a duration string like "7d", "24h", "30m" into a chrono::Duration.
fn parse_duration(s: &str) -> std::result::Result<Duration, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration string".into());
    }

    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: i64 = num_str.parse()
        .map_err(|_| format!("invalid number in duration: {num_str}"))?;

    match unit {
        "d" => Duration::try_days(num).ok_or_else(|| "duration out of range".into()),
        "h" => Duration::try_hours(num).ok_or_else(|| "duration out of range".into()),
        "m" => Duration::try_minutes(num).ok_or_else(|| "duration out of range".into()),
        "s" => Duration::try_seconds(num).ok_or_else(|| "duration out of range".into()),
        _ => Err(format!("unknown duration unit: {unit}. Use d, h, m, or s")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use corvia_kernel::agent_coordinator::AgentCoordinator;
    use corvia_kernel::lite_store::LiteStore;
    use corvia_kernel::traits::{
        GenerationEngine, GenerationResult, GraphStore, InferenceEngine, QueryableStore,
        TemporalStore,
    };

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

    /// Mock generation engine with configurable response text.
    struct MockGenerationEngine {
        response: String,
    }
    impl MockGenerationEngine {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }
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
                text: self.response.clone(),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize {
            4096
        }
    }

    async fn test_state(dir: &std::path::Path) -> Arc<AppState> {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen_engine =
            Arc::new(MockGenerationEngine::new("merged")) as Arc<dyn GenerationEngine>;
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
        Arc::new(AppState {
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
            cluster_store: Arc::new(crate::dashboard::clustering::ClusterStore::new()),
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
            session_ingest_lock: tokio::sync::Mutex::new(()),
            hook_sessions: crate::dashboard::session_watcher::SessionWatcherState::new().0,
            coverage_cache: Arc::new(
                crate::dashboard::coverage::IndexCoverageCache::new(0.9, 60),
            ),
            workspace_root: dir.to_path_buf(),
            ingest_status: Arc::new(std::sync::RwLock::new(corvia_kernel::ingest::IngestStatus::idle())),
            gpu_cache: Arc::new(tokio::sync::Mutex::new(crate::dashboard::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
            mcp_token: None,
            docker_available: false,
            gap_detector: Arc::new(corvia_kernel::gap_detector::GapDetector::new()),
        })
    }

    /// Build a test AppState with a RagPipeline that has a mock generator.
    async fn test_state_with_generator(
        dir: &std::path::Path,
        response: &str,
    ) -> Arc<AppState> {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen_engine =
            Arc::new(MockGenerationEngine::new("merged")) as Arc<dyn GenerationEngine>;
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

        // Build a minimal RagPipeline with a mock generator for classify tests
        let mock_gen =
            Arc::new(MockGenerationEngine::new(response)) as Arc<dyn GenerationEngine>;
        let mock_retriever = Arc::new(MockRetriever) as Arc<dyn corvia_kernel::retriever::Retriever>;
        let mock_augmenter = Arc::new(MockAugmenter) as Arc<dyn corvia_kernel::augmenter::Augmenter>;
        let rag = corvia_kernel::rag_pipeline::RagPipeline::new(
            mock_retriever,
            mock_augmenter,
            Some(mock_gen),
            corvia_kernel::rag_types::RagConfig::default(),
        );

        Arc::new(AppState {
            store: store.clone() as Arc<dyn QueryableStore>,
            engine,
            coordinator,
            graph: store.clone() as Arc<dyn GraphStore>,
            temporal: store as Arc<dyn TemporalStore>,
            data_dir: dir.to_path_buf(),
            rag: Some(Arc::new(arc_swap::ArcSwap::from_pointee(rag))),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            default_scope_id: None,
            config: Arc::new(std::sync::RwLock::new(
                corvia_common::config::CorviaConfig::default(),
            )),
            config_path: dir.join("corvia.toml"),
            cluster_store: Arc::new(crate::dashboard::clustering::ClusterStore::new()),
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
            session_ingest_lock: tokio::sync::Mutex::new(()),
            hook_sessions: crate::dashboard::session_watcher::SessionWatcherState::new().0,
            coverage_cache: Arc::new(
                crate::dashboard::coverage::IndexCoverageCache::new(0.9, 60),
            ),
            workspace_root: dir.to_path_buf(),
            ingest_status: Arc::new(std::sync::RwLock::new(corvia_kernel::ingest::IngestStatus::idle())),
            gpu_cache: Arc::new(tokio::sync::Mutex::new(crate::dashboard::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
            mcp_token: None,
            docker_available: false,
            gap_detector: Arc::new(corvia_kernel::gap_detector::GapDetector::new()),
        })
    }

    struct MockRetriever;
    #[async_trait::async_trait]
    impl corvia_kernel::retriever::Retriever for MockRetriever {
        fn name(&self) -> &str { "mock" }
        async fn retrieve(
            &self,
            _query: &str,
            _scope_id: &str,
            _opts: &corvia_kernel::rag_types::RetrievalOpts,
        ) -> corvia_common::errors::Result<corvia_kernel::rag_types::RetrievalResult> {
            Ok(corvia_kernel::rag_types::RetrievalResult {
                results: vec![],
                metrics: corvia_kernel::rag_types::RetrievalMetrics {
                    latency_ms: 0,
                    embed_latency_ms: 0,
                    search_latency_ms: 0,
                    hnsw_latency_ms: 0,
                    graph_latency_ms: 0,
                    filter_latency_ms: 0,
                    cold_scan_latency_ms: 0,
                    vector_results: 0,
                    cold_results: 0,
                    graph_expanded: 0,
                    graph_reinforced: 0,
                    post_filter_count: 0,
                    retriever_name: "mock".into(),
                    bm25_latency_ms: None,
                    bm25_results: None,
                    fusion_method: None,
                    fusion_latency_ms: None,
                    stages: None,
                },
                query_embedding: None,
            })
        }
    }

    struct MockAugmenter;
    impl corvia_kernel::augmenter::Augmenter for MockAugmenter {
        fn name(&self) -> &str { "mock" }
        fn augment(
            &self,
            _query: &str,
            _results: &[corvia_common::types::SearchResult],
            _budget: &corvia_kernel::rag_types::TokenBudget,
        ) -> corvia_common::errors::Result<corvia_kernel::rag_types::AugmentedContext> {
            Ok(corvia_kernel::rag_types::AugmentedContext {
                system_prompt: String::new(),
                context: String::new(),
                sources: vec![],
                metrics: corvia_kernel::rag_types::AugmentationMetrics {
                    latency_ms: 0,
                    token_estimate: 0,
                    token_budget: 0,
                    sources_included: 0,
                    sources_truncated: 0,
                    augmenter_name: "mock".into(),
                    skills_used: vec![],
                },
            })
        }
    }

    // -----------------------------------------------------------------------
    // Pure function tests: infer_content_role_from_llm
    // -----------------------------------------------------------------------

    #[test]
    fn test_infer_llm_decision() {
        assert_eq!(
            infer_content_role_from_llm("YES. This is a decision about the API."),
            Some("decision".into())
        );
    }

    #[test]
    fn test_infer_llm_design() {
        assert_eq!(
            infer_content_role_from_llm("YES. This discusses the design of the system."),
            Some("design".into())
        );
    }

    #[test]
    fn test_infer_llm_finding() {
        assert_eq!(
            infer_content_role_from_llm("YES. This is a research finding."),
            Some("finding".into())
        );
    }

    #[test]
    fn test_infer_llm_learning() {
        assert_eq!(
            infer_content_role_from_llm("YES. This captures a learning about Rust."),
            Some("learning".into())
        );
    }

    #[test]
    fn test_infer_llm_no_match() {
        assert_eq!(
            infer_content_role_from_llm("YES. This is important."),
            None
        );
    }

    #[test]
    fn test_infer_llm_case_insensitive() {
        assert_eq!(
            infer_content_role_from_llm("YES. Content type: DECISION"),
            Some("decision".into())
        );
    }

    #[test]
    fn test_infer_llm_priority_order() {
        // "decision" checked before "design", so it wins when both present
        assert_eq!(
            infer_content_role_from_llm("A decision about the design."),
            Some("decision".into())
        );
    }

    #[test]
    fn test_infer_llm_no_response() {
        assert_eq!(infer_content_role_from_llm("NO."), None);
    }

    // -----------------------------------------------------------------------
    // Pure function tests: infer_product_content_role
    // -----------------------------------------------------------------------

    #[test]
    fn test_product_role_decision_keyword() {
        assert_eq!(
            infer_product_content_role("We decided to use Rust for the kernel"),
            "decision"
        );
    }

    #[test]
    fn test_product_role_chose() {
        assert_eq!(
            infer_product_content_role("We chose LiteStore as the default"),
            "decision"
        );
    }

    #[test]
    fn test_product_role_design() {
        assert_eq!(
            infer_product_content_role("The architecture of the RAG pipeline"),
            "design"
        );
    }

    #[test]
    fn test_product_role_rfc() {
        assert_eq!(
            infer_product_content_role("RFC: Add PostgreSQL support"),
            "design"
        );
    }

    #[test]
    fn test_product_role_research() {
        assert_eq!(
            infer_product_content_role("Research on vector databases"),
            "finding"
        );
    }

    #[test]
    fn test_product_role_discovered() {
        assert_eq!(
            infer_product_content_role("We discovered that HNSW performs better"),
            "finding"
        );
    }

    #[test]
    fn test_product_role_learned() {
        assert_eq!(
            infer_product_content_role("We learned batch embedding is 10x faster"),
            "learning"
        );
    }

    #[test]
    fn test_product_role_lesson() {
        assert_eq!(
            infer_product_content_role("A lesson from the production incident"),
            "learning"
        );
    }

    #[test]
    fn test_product_role_pattern() {
        assert_eq!(
            infer_product_content_role("A useful pattern for error handling"),
            "learning"
        );
    }

    #[test]
    fn test_product_role_default() {
        assert_eq!(
            infer_product_content_role("Some random text without keywords"),
            "finding"
        );
    }

    #[test]
    fn test_product_role_only_checks_prefix() {
        // Keywords beyond 300 chars should not be detected
        let mut content = "x".repeat(301);
        content.push_str("decision");
        assert_eq!(infer_product_content_role(&content), "finding");
    }

    #[test]
    fn test_product_role_keyword_in_prefix() {
        // Keyword at char 299 should be detected
        let mut content = "x".repeat(290);
        content.push_str("decision");
        assert_eq!(infer_product_content_role(&content), "decision");
    }

    // -----------------------------------------------------------------------
    // Endpoint tests: ingest_sessions
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ingest_sessions_no_sessions_dir() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Point HOME to temp dir (no .claude/sessions exists)
        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = ingest_sessions(State(state)).await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.sessions_ingested, 0);
        assert_eq!(resp.entries_stored, 0);
    }

    #[tokio::test]
    async fn test_ingest_sessions_no_user_history_scope() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Create sessions dir so the "no dir" check passes
        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = ingest_sessions(State(state)).await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        // Config has no user-history scope → 400
        assert!(result.is_err());
        let (status, msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("user-history"));
    }

    // -----------------------------------------------------------------------
    // Endpoint tests: classify_sessions
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_classify_sessions_empty_queue() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // No .classify-queue file → empty queue
        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 0);
        assert_eq!(resp.promoted, 0);
        assert_eq!(resp.rejected, 0);
        assert_eq!(resp.failed, 0);
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_sessions_no_generator() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await; // rag: None

        // Create a non-empty queue so the handler proceeds past the empty check
        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join(".classify-queue"), "ses-abc:turn-1\n").unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_classify_sessions_entry_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_generator(dir.path(), "YES. This is a decision.").await;

        // Create queue referencing an entry that doesn't exist in knowledge files
        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(
            sessions.join(".classify-queue"),
            "ses-nonexistent:turn-1\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.promoted, 0);
        assert_eq!(resp.rejected, 1); // not found → rejected
        assert_eq!(resp.failed, 0);
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_sessions_yes_promotes() {
        let dir = tempfile::tempdir().unwrap();
        let state =
            test_state_with_generator(dir.path(), "YES. This is a decision about storage.").await;

        // Write a user-history entry to knowledge files
        let entry = KnowledgeEntry::new(
            "We decided to use LiteStore as the default backend".into(),
            USER_HISTORY_SCOPE.into(),
            "ses-abc:turn-1".into(),
        );
        corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();

        // Create queue referencing that entry
        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join(".classify-queue"), "ses-abc:turn-1\n").unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.promoted, 1);
        assert_eq!(resp.rejected, 0);
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_sessions_no_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let state =
            test_state_with_generator(dir.path(), "NO. This is just a greeting.").await;

        let entry = KnowledgeEntry::new(
            "Hello, how are you?".into(),
            USER_HISTORY_SCOPE.into(),
            "ses-abc:turn-1".into(),
        );
        corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();

        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join(".classify-queue"), "ses-abc:turn-1\n").unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.promoted, 0);
        assert_eq!(resp.rejected, 1);
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_sessions_batch_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let state =
            test_state_with_generator(dir.path(), "NO. Not relevant.").await;

        // Write 3 entries, but set batch_size to 2
        for i in 1..=3 {
            let entry = KnowledgeEntry::new(
                format!("Turn {i} content"),
                USER_HISTORY_SCOPE.into(),
                format!("ses-abc:turn-{i}"),
            );
            corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();
        }

        let sessions = dir.path().join(".claude").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(
            sessions.join(".classify-queue"),
            "ses-abc:turn-1\nses-abc:turn-2\nses-abc:turn-3\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_sessions(
            State(state),
            Json(ClassifySessionsRequest {
                batch_size: Some(2),
            }),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 2);
        assert_eq!(resp.remaining, 1); // 1 left in queue
    }

    // -----------------------------------------------------------------------
    // Endpoint tests: classify_agent_teams
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_classify_agent_teams_empty_queue() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 0);
        assert_eq!(resp.extracted, 0);
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_agent_teams_no_generator() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await; // rag: None

        // Create a non-empty queue so the handler proceeds past the empty check
        let staging = dir.path().join(".corvia").join("staging").join("agent-teams");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join(".classify-queue"), "my-team:messages:task-1\n").unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        // rag is None, so handler skips gracefully
        assert!(result.is_err(), "should fail when generator is not configured");
    }

    #[tokio::test]
    async fn test_classify_agent_teams_entry_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let extraction_json = r#"[{"type":"decision","content":"Use ArcSwap","rationale":"Lock-free reads","dissent":""}]"#;
        let state = test_state_with_generator(dir.path(), extraction_json).await;

        let staging = dir.path().join(".corvia").join("staging").join("agent-teams");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(
            staging.join(".classify-queue"),
            "nonexistent-team:messages:task-1\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.rejected, 1); // Entry not found
        assert_eq!(resp.extracted, 0);
    }

    #[tokio::test]
    async fn test_classify_agent_teams_extracts_decision() {
        let dir = tempfile::tempdir().unwrap();
        let response = r#"[{"type":"decision","content":"Use ArcSwap for lock-free reads","rationale":"Better perf under contention","dissent":""}]"#;
        let state = test_state_with_generator(dir.path(), response).await;

        // Create an entry in user-history scope
        let entry = KnowledgeEntry::new(
            "Team discussion about using ArcSwap vs RwLock".into(),
            USER_HISTORY_SCOPE.into(),
            "my-team:messages:task-1".into(),
        );
        corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();
        state.store.insert(&entry.with_embedding(vec![1.0, 0.0, 0.0])).await.unwrap();

        let staging = dir.path().join(".corvia").join("staging").join("agent-teams");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(
            staging.join(".classify-queue"),
            "my-team:messages:task-1\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.extracted, 1, "should extract 1 decision");
        // Mock returns extraction JSON for all calls (including promotion check).
        // That JSON does not start with "YES", so promotion does not fire.
        assert_eq!(resp.promoted, 0);
        assert_eq!(resp.remaining, 0, "queue should be drained");
    }

    #[tokio::test]
    async fn test_classify_agent_teams_promotes_to_corvia() {
        let dir = tempfile::tempdir().unwrap();
        // Mock LLM returns "YES" for all calls. The extraction parser will find no
        // JSON array in "YES", so we need a response that works for both extraction
        // AND promotion. We use a response that starts with "YES" and contains JSON.
        let response = r#"YES. [{"type":"finding","content":"HNSW is thread-safe","rationale":"Internal RwLock","dissent":""}]"#;
        let state = test_state_with_generator(dir.path(), response).await;

        let entry = KnowledgeEntry::new(
            "Team discovered HNSW internals".into(),
            USER_HISTORY_SCOPE.into(),
            "promo-team:messages:task-1".into(),
        );
        corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();
        state.store.insert(&entry.with_embedding(vec![1.0, 0.0, 0.0])).await.unwrap();

        let staging = dir.path().join(".corvia").join("staging").join("agent-teams");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(
            staging.join(".classify-queue"),
            "promo-team:messages:task-1\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest::default()),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 1);
        assert_eq!(resp.extracted, 1, "should extract 1 finding");
        assert_eq!(resp.promoted, 1, "YES response should promote to corvia scope");
        assert_eq!(resp.remaining, 0);
    }

    #[tokio::test]
    async fn test_classify_agent_teams_batch_size() {
        let dir = tempfile::tempdir().unwrap();
        let response = "[]"; // Empty extraction
        let state = test_state_with_generator(dir.path(), response).await;

        // Create entries
        for i in 1..=3 {
            let entry = KnowledgeEntry::new(
                format!("Message group {i}"),
                USER_HISTORY_SCOPE.into(),
                format!("batch-team:messages:task-{i}"),
            );
            corvia_kernel::knowledge_files::write_entry(&state.data_dir, &entry).unwrap();
        }

        let staging = dir.path().join(".corvia").join("staging").join("agent-teams");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(
            staging.join(".classify-queue"),
            "batch-team:messages:task-1\nbatch-team:messages:task-2\nbatch-team:messages:task-3\n",
        )
        .unwrap();

        let orig_home = std::env::var("HOME").ok();
        unsafe { std::env::set_var("HOME", dir.path().to_str().unwrap()) };

        let result = classify_agent_teams(
            State(state),
            Json(ClassifyAgentTeamsRequest {
                batch_size: Some(2),
            }),
        )
        .await;

        if let Some(h) = orig_home {
            unsafe { std::env::set_var("HOME", h) };
        }

        let Json(resp) = result.unwrap();
        assert_eq!(resp.processed, 2);
        assert_eq!(resp.skipped, 2, "empty extraction counts as skipped");
        assert_eq!(resp.remaining, 1);
    }

    // -----------------------------------------------------------------------
    // Unit tests: parse_extraction_items
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_extraction_items_valid_json() {
        let response = r#"[
            {"type": "decision", "content": "Use ArcSwap", "rationale": "Lock-free", "dissent": ""},
            {"type": "finding", "content": "HNSW is thread-safe", "rationale": "Internal RwLock", "dissent": "Not documented"}
        ]"#;
        let items = parse_extraction_items(response);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].item_type, "decision");
        assert_eq!(items[1].item_type, "finding");
    }

    #[test]
    fn test_parse_extraction_items_markdown_wrapped() {
        let response = "Here are the extractions:\n```json\n[{\"type\": \"question\", \"content\": \"Should we use pgvector?\", \"rationale\": \"Scalability\", \"dissent\": \"\"}]\n```";
        let items = parse_extraction_items(response);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].item_type, "question");
    }

    #[test]
    fn test_parse_extraction_items_empty_array() {
        let items = parse_extraction_items("[]");
        assert!(items.is_empty());
    }

    #[test]
    fn test_parse_extraction_items_no_json() {
        let items = parse_extraction_items("Nothing worth extracting from this discussion.");
        assert!(items.is_empty());
    }

    #[test]
    fn test_parse_extraction_items_invalid_type() {
        let response = r#"[{"type": "suggestion", "content": "try this", "rationale": "why not", "dissent": ""}]"#;
        let items = parse_extraction_items(response);
        assert!(items.is_empty(), "invalid types should be filtered out");
    }

    // -----------------------------------------------------------------------
    // Helper tests: parse_duration
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_duration_days() {
        assert_eq!(parse_duration("7d").unwrap(), Duration::try_days(7).unwrap());
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("24h").unwrap(), Duration::try_hours(24).unwrap());
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("30m").unwrap(), Duration::try_minutes(30).unwrap());
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("60s").unwrap(), Duration::try_seconds(60).unwrap());
    }

    #[test]
    fn test_parse_duration_invalid_unit() {
        assert!(parse_duration("10x").is_err());
    }

    #[test]
    fn test_parse_duration_empty() {
        assert!(parse_duration("").is_err());
    }
}
