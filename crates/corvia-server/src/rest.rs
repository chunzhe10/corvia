use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use chrono::{DateTime, Duration, Utc};
use corvia_common::agent_types::*;
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry, SearchResult};
use corvia_kernel::agent_coordinator::AgentCoordinator;
use corvia_kernel::ops::GcHistory;
use corvia_kernel::traits::{GraphStore, InferenceEngine, QueryableStore, TemporalStore};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::info;

pub struct AppState {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub coordinator: Arc<AgentCoordinator>,
    pub graph: Arc<dyn GraphStore>,
    pub temporal: Arc<dyn TemporalStore>,
    pub data_dir: std::path::PathBuf,
    pub rag: Option<Arc<corvia_kernel::rag_pipeline::RagPipeline>>,
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
        .route("/health", get(health))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// --- Helper ---

fn coordinator(state: &AppState) -> &AgentCoordinator {
    &state.coordinator
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

    // Route through RAG pipeline if available (fixes ContextBuilder bypass bug)
    if let Some(rag) = &state.rag {
        let opts = corvia_kernel::rag_types::RetrievalOpts {
            limit,
            expand_graph: false, // search endpoint: pure vector (context/ask use graph)
            content_role,
            source_origin,
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

    let search_limit = if content_role.is_some() || source_origin.is_some() {
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
    let rag = state.rag.as_ref()
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
    let rag = state.rag.as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit: req.limit.unwrap_or(10),
        expand_graph: req.expand_graph.unwrap_or(true),
        ..Default::default()
    };

    let response = rag.ask(&req.query, &req.scope_id, Some(opts)).await
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
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Heartbeat failed: {e}")))?;
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
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Commit failed: {e}")))?;
    Ok(StatusCode::OK)
}

async fn rollback_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> std::result::Result<StatusCode, (StatusCode, String)> {
    let coord = coordinator(&state);
    coord.rollback_session(&session_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Rollback failed: {e}")))?;
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
            query.relation.as_ref().map_or(true, |r| &e.relation == r)
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
    // Load entries from knowledge files (direct disk read, not HNSW search)
    let entries = corvia_kernel::knowledge_files::read_scope(&state.data_dir, &req.scope_id)
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

