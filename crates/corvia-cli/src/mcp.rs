//! HTTP MCP server for corvia, built on axum. Exposes:
//!   - `POST /mcp`    — JSON-RPC 2.0 MCP endpoint (corvia_search, corvia_write, corvia_status, corvia_traces)
//!   - `GET /healthz` — deep health check; returns `{"ok":true,"entries":N}` or 503
//!
//! The Embedder is created once at startup and shared across all tool calls.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rmcp::handler::server::ServerHandler;
use tokio::sync::Mutex as TokioMutex;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParam, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{Peer, RequestContext, RoleServer};
use rmcp::Error as McpError;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use corvia_core::config::Config;
use corvia_core::embed::Embedder;
use corvia_core::index::RedbIndex;
use corvia_core::search::{self, SearchParams};
use corvia_core::tantivy_index::TantivyIndex;
use corvia_core::types::Kind;
use corvia_core::write::{self, WriteParams};

// ---------------------------------------------------------------------------
// Tool parameter structs (deserialized from MCP call arguments)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SearchToolParams {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    max_tokens: Option<usize>,
    min_score: Option<f32>,
    kind: Option<String>,
}

fn default_limit() -> usize {
    5
}

#[derive(Deserialize)]
struct WriteToolParams {
    content: String,
    #[serde(default = "default_kind")]
    kind: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    supersedes: Vec<String>,
}

fn default_kind() -> String {
    "learning".to_string()
}

#[derive(Deserialize)]
struct TracesToolParams {
    #[serde(default = "default_traces_limit")]
    limit: usize,
    span_filter: Option<String>,
}

fn default_traces_limit() -> usize {
    10
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

/// MCP server state shared across all tool calls.
///
/// The Embedder and Config are created once at startup. Each tool call opens
/// the index databases as needed (they are lightweight open operations).
#[derive(Clone)]
struct CorviaServer {
    config: Arc<Config>,
    embedder: Arc<Embedder>,
    base_dir: PathBuf,
    peer: Option<Peer<RoleServer>>,
}

impl CorviaServer {
    fn new(config: Config, embedder: Embedder, base_dir: PathBuf) -> Self {
        Self {
            config: Arc::new(config),
            embedder: Arc::new(embedder),
            base_dir,
            peer: None,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP server state (for `corvia serve`)
// ---------------------------------------------------------------------------

/// Persistent index handles held open for the lifetime of the HTTP server.
///
/// `write_lock` serializes write operations (one write at a time); reads are concurrent.
struct IndexHandles {
    redb: RedbIndex,
    tantivy: TantivyIndex,
    write_lock: TokioMutex<()>,
}

/// Axum state shared across all HTTP MCP handler calls.
#[derive(Clone)]
struct ServeState {
    config: Arc<Config>,
    embedder: Arc<Embedder>,
    base_dir: std::path::PathBuf,
    handles: Arc<IndexHandles>,
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema)
// ---------------------------------------------------------------------------

fn search_tool() -> Tool {
    Tool::new(
        "corvia_search",
        "Hybrid semantic + keyword search with reranking across organizational knowledge. Returns raw chunks, not synthesized answers.",
        rmcp::model::object(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 5)",
                    "default": 5
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum total tokens across all results"
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum score threshold; results below this are filtered out"
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by knowledge kind",
                    "enum": ["decision", "learning", "instruction", "reference"]
                }
            },
            "required": ["query"]
        })),
    )
}

fn write_tool() -> Tool {
    Tool::new(
        "corvia_write",
        "Write a knowledge entry. If similar content exists, the old entry is automatically superseded. Classify with 'kind' to aid future retrieval.",
        rmcp::model::object(json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The knowledge content to store (markdown)"
                },
                "kind": {
                    "type": "string",
                    "description": "Knowledge kind (default: learning). decision = choices with rationale. learning = insight, gotcha, pattern (default). instruction = how-to, setup, workflow. reference = code pattern, API doc, config.",
                    "enum": ["decision", "learning", "instruction", "reference"],
                    "default": "learning"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags for the entry"
                },
                "supersedes": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "IDs of entries to explicitly supersede"
                }
            },
            "required": ["content"]
        })),
    )
}

fn status_tool() -> Tool {
    Tool::new(
        "corvia_status",
        "System status: entry counts, index health, storage location.",
        rmcp::model::object(json!({
            "type": "object",
            "properties": {},
            "required": []
        })),
    )
}

fn traces_tool() -> Tool {
    Tool::new(
        "corvia_traces",
        "Recent operation traces with timing and attributes. Use to inspect search latency, reranking performance, dedup behavior, and pipeline health.",
        rmcp::model::object(json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of recent traces to return"
                },
                "span_filter": {
                    "type": "string",
                    "description": "Filter by span name prefix (e.g. 'corvia.search' to see only search traces)"
                }
            }
        })),
    )
}

// ---------------------------------------------------------------------------
// Tool dispatch handlers
// ---------------------------------------------------------------------------

fn handle_search(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: SearchToolParams,
) -> Result<serde_json::Value> {
    let kind = match &params.kind {
        Some(k) => Some(k.parse::<Kind>().map_err(|e| anyhow::anyhow!(e))?),
        None => None,
    };

    let search_params = SearchParams {
        query: params.query,
        limit: params.limit,
        max_tokens: params.max_tokens,
        min_score: params.min_score,
        kind,
    };

    let response = search::search(config, base_dir, embedder, &search_params)
        .context("search failed")?;

    Ok(serde_json::to_value(&response)?)
}

fn handle_write(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: WriteToolParams,
) -> Result<serde_json::Value> {
    let kind = params
        .kind
        .parse::<Kind>()
        .map_err(|e| anyhow::anyhow!(e))?;

    let write_params = WriteParams {
        content: params.content,
        kind,
        tags: params.tags,
        supersedes: params.supersedes,
    };

    let response = write::write(config, base_dir, embedder, write_params)
        .context("write failed")?;

    Ok(serde_json::to_value(&response)?)
}

fn handle_status(config: &Config, base_dir: &Path) -> Result<serde_json::Value> {
    let redb_path = base_dir.join(config.redb_path());
    let tantivy_dir = base_dir.join(config.tantivy_dir());
    let storage_path = base_dir.join(&config.data_dir);

    // Try to open indexes; if they don't exist yet, return zeros.
    let (entry_count, superseded_count, vector_count, last_ingest, stale) =
        match RedbIndex::open(&redb_path) {
            Ok(redb) => {
                let entry_count = redb.entry_count().unwrap_or(0);
                let superseded_count = redb
                    .superseded_ids()
                    .map(|ids| ids.len() as u64)
                    .unwrap_or(0);
                let vector_count = redb.vector_count().unwrap_or(0);
                let last_ingest = redb.get_meta("last_ingest").ok().flatten();

                // Drift detection: compare indexed count with actual files.
                let indexed_count_str = redb.get_meta("entry_count").ok().flatten();
                let indexed_count: u64 = indexed_count_str
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let entries_dir = base_dir.join(config.entries_dir());
                let actual_count = corvia_core::entry::scan_entries(&entries_dir)
                    .map(|v| v.len() as u64)
                    .unwrap_or(0);
                let stale = actual_count != indexed_count;

                (entry_count, superseded_count, vector_count, last_ingest, stale)
            }
            Err(_) => (0, 0, 0, None, false),
        };

    let bm25_docs = match TantivyIndex::open(&tantivy_dir) {
        Ok(tantivy) => tantivy.doc_count(),
        Err(_) => 0,
    };

    // Read recent traces.
    let trace_path = base_dir.join(&config.data_dir).join("traces.jsonl");
    let parsed_traces = corvia_core::trace::read_recent_traces(&trace_path, 10);
    let recent_traces: Vec<corvia_core::types::TraceEntry> = parsed_traces
        .into_iter()
        .map(|t| corvia_core::types::TraceEntry {
            name: t.name,
            elapsed_ms: t.elapsed_ms,
            timestamp_ns: t.timestamp_ns,
            attributes: t.attributes,
        })
        .collect();

    let response = corvia_core::types::StatusResponse {
        entry_count,
        superseded_count,
        index_health: corvia_core::types::IndexHealth {
            bm25_docs,
            vector_count,
            last_ingest,
            stale,
        },
        storage_path: storage_path.display().to_string(),
        recent_traces,
    };

    Ok(serde_json::to_value(&response)?)
}

fn handle_traces(
    config: &Config,
    base_dir: &Path,
    params: TracesToolParams,
) -> Result<serde_json::Value> {
    let trace_path = base_dir.join(&config.data_dir).join("traces.jsonl");
    let mut traces = corvia_core::trace::read_recent_traces(&trace_path, params.limit);

    // Apply optional span name prefix filter.
    if let Some(ref prefix) = params.span_filter {
        traces.retain(|t| t.name.starts_with(prefix.as_str()));
    }

    let entries: Vec<corvia_core::types::TraceEntry> = traces
        .into_iter()
        .map(|t| corvia_core::types::TraceEntry {
            name: t.name,
            elapsed_ms: t.elapsed_ms,
            timestamp_ns: t.timestamp_ns,
            attributes: t.attributes,
        })
        .collect();

    Ok(serde_json::to_value(&entries)?)
}

// ---------------------------------------------------------------------------
// ServerHandler implementation
// ---------------------------------------------------------------------------

impl ServerHandler for CorviaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "corvia".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "corvia is organizational memory for AI agents. \
                 Use corvia_search to find knowledge, corvia_write to store it, \
                 corvia_status to check system health, and corvia_traces to \
                 inspect recent operation traces."
                    .to_string(),
            ),
        }
    }

    fn get_peer(&self) -> Option<Peer<RoleServer>> {
        self.peer.clone()
    }

    fn set_peer(&mut self, peer: Peer<RoleServer>) {
        self.peer = Some(peer);
    }

    async fn list_tools(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: vec![search_tool(), write_tool(), status_tool(), traces_tool()],
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let name = request.name.as_ref();
        let args = request.arguments.unwrap_or_default();
        let args_value = serde_json::Value::Object(args);

        let result = match name {
            "corvia_search" => {
                let params: SearchToolParams =
                    serde_json::from_value(args_value).map_err(|e| {
                        McpError::invalid_params(
                            format!("invalid search params: {e}"),
                            None,
                        )
                    })?;
                handle_search(&self.config, &self.base_dir, &self.embedder, params)
            }
            "corvia_write" => {
                let params: WriteToolParams =
                    serde_json::from_value(args_value).map_err(|e| {
                        McpError::invalid_params(
                            format!("invalid write params: {e}"),
                            None,
                        )
                    })?;
                handle_write(&self.config, &self.base_dir, &self.embedder, params)
            }
            "corvia_status" => handle_status(&self.config, &self.base_dir),
            "corvia_traces" => {
                let params: TracesToolParams =
                    serde_json::from_value(args_value).map_err(|e| {
                        McpError::invalid_params(
                            format!("invalid traces params: {e}"),
                            None,
                        )
                    })?;
                handle_traces(&self.config, &self.base_dir, params)
            }
            other => {
                return Err(McpError::invalid_params(
                    format!("unknown tool: {other}"),
                    None,
                ));
            }
        };

        match result {
            Ok(value) => {
                let json_text = serde_json::to_string(&value).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json_text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Error: {e:#}"
            ))])),
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP MCP helper handlers
// ---------------------------------------------------------------------------

/// MCP `initialize` response — server capabilities handshake.
fn handle_initialize_http() -> serde_json::Value {
    serde_json::json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "corvia",
            "version": env!("CARGO_PKG_VERSION"),
        }
    })
}

/// MCP `tools/list` response — the four corvia tools.
fn handle_tools_list_http() -> serde_json::Value {
    let tools = vec![
        serde_json::to_value(search_tool()).unwrap_or_default(),
        serde_json::to_value(write_tool()).unwrap_or_default(),
        serde_json::to_value(status_tool()).unwrap_or_default(),
        serde_json::to_value(traces_tool()).unwrap_or_default(),
    ];
    serde_json::json!({ "tools": tools })
}

/// `GET /healthz` — deep health check. Queries the live index handles and returns
/// `{"ok":true,"entries":N}` on success or `{"ok":false,"error":"..."}` with a 503
/// if the index is unreadable.
async fn healthz_handler(State(state): State<ServeState>) -> Response {
    match handle_status_with_handles(
        &state.config,
        &state.base_dir,
        &state.handles.redb,
        &state.handles.tantivy,
    ) {
        Ok(status) => {
            let entries = status["entry_count"].as_u64().unwrap_or(0);
            (StatusCode::OK, Json(json!({"ok": true, "entries": entries}))).into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Status handler using pre-opened index handles.
fn handle_status_with_handles(
    config: &Config,
    base_dir: &std::path::Path,
    redb: &RedbIndex,
    tantivy: &TantivyIndex,
) -> Result<serde_json::Value> {
    let storage_path = base_dir.join(&config.data_dir);

    let entry_count = redb.entry_count().unwrap_or(0);
    let superseded_count = redb
        .superseded_ids()
        .map(|ids| ids.len() as u64)
        .unwrap_or(0);
    let vector_count = redb.vector_count().unwrap_or(0);
    let last_ingest = redb.get_meta("last_ingest").ok().flatten();

    let indexed_count_str = redb.get_meta("entry_count").ok().flatten();
    let indexed_count: u64 = indexed_count_str
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let entries_dir = base_dir.join(config.entries_dir());
    let actual_count = corvia_core::entry::scan_entries(&entries_dir)
        .map(|v| v.len() as u64)
        .unwrap_or(0);
    let stale = actual_count != indexed_count;

    let bm25_docs = tantivy.doc_count();

    let trace_path = base_dir.join(&config.data_dir).join("traces.jsonl");
    let parsed_traces = corvia_core::trace::read_recent_traces(&trace_path, 10);
    let recent_traces: Vec<corvia_core::types::TraceEntry> = parsed_traces
        .into_iter()
        .map(|t| corvia_core::types::TraceEntry {
            name: t.name,
            elapsed_ms: t.elapsed_ms,
            timestamp_ns: t.timestamp_ns,
            attributes: t.attributes,
        })
        .collect();

    let response = corvia_core::types::StatusResponse {
        entry_count,
        superseded_count,
        index_health: corvia_core::types::IndexHealth {
            bm25_docs,
            vector_count,
            last_ingest,
            stale,
        },
        storage_path: storage_path.display().to_string(),
        recent_traces,
    };

    Ok(serde_json::to_value(&response)?)
}

/// Route a `tools/call` request to the appropriate handler.
///
/// Returns `Ok(content_value)` on success or `Err(jsonrpc_error_object)` on failure.
async fn handle_tools_call_http(
    state: &ServeState,
    params: serde_json::Value,
) -> Result<serde_json::Value, serde_json::Value> {
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| {
            serde_json::json!({ "code": -32602, "message": "Missing tool name in params" })
        })?;

    // Reject unknown tools before entering the anyhow context.
    match name {
        "corvia_search" | "corvia_write" | "corvia_status" | "corvia_traces" => {}
        other => {
            return Err(serde_json::json!({
                "code": -32602,
                "message": format!("Unknown tool: {other}"),
            }));
        }
    }

    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));

    let tool_result: anyhow::Result<serde_json::Value> = async {
        match name {
            "corvia_search" => {
                let p: SearchToolParams = serde_json::from_value(args)
                    .map_err(|e| anyhow::anyhow!("invalid search params: {e}"))?;
                let kind = match &p.kind {
                    Some(k) => Some(k.parse::<Kind>().map_err(|e| anyhow::anyhow!(e))?),
                    None => None,
                };
                let search_params = SearchParams {
                    query: p.query,
                    limit: p.limit,
                    max_tokens: p.max_tokens,
                    min_score: p.min_score,
                    kind,
                };
                // search_with_handles is CPU-bound (embedding + reranker inference).
                // Use block_in_place so we don't block the tokio worker thread.
                tokio::task::block_in_place(|| {
                    corvia_core::search::search_with_handles(
                        &state.config,
                        &state.base_dir,
                        &state.embedder,
                        &search_params,
                        &state.handles.redb,
                        &state.handles.tantivy,
                    )
                    .and_then(|r| {
                        serde_json::to_value(&r)
                            .map_err(|e| anyhow::anyhow!("serializing search response: {e}"))
                    })
                })
            }
            "corvia_write" => {
                let p: WriteToolParams = serde_json::from_value(args)
                    .map_err(|e| anyhow::anyhow!("invalid write params: {e}"))?;
                let kind = p.kind.parse::<Kind>().map_err(|e| anyhow::anyhow!(e))?;
                let write_params = corvia_core::write::WriteParams {
                    content: p.content,
                    kind,
                    tags: p.tags,
                    supersedes: p.supersedes,
                };
                // Acquire write lock (async) before the blocking work.
                let _lock = state.handles.write_lock.lock().await;
                // write_with_handles is CPU-bound (embedding) + blocking I/O (redb, tantivy).
                // Use block_in_place so we don't block the tokio worker thread.
                tokio::task::block_in_place(|| {
                    corvia_core::write::write_with_handles(
                        &state.config,
                        &state.base_dir,
                        &state.embedder,
                        write_params,
                        &state.handles.redb,
                        &state.handles.tantivy,
                    )
                    .and_then(|r| {
                        serde_json::to_value(&r)
                            .map_err(|e| anyhow::anyhow!("serializing write response: {e}"))
                    })
                })
            }
            "corvia_status" => handle_status_with_handles(
                &state.config,
                &state.base_dir,
                &state.handles.redb,
                &state.handles.tantivy,
            ),
            "corvia_traces" => {
                let p: TracesToolParams =
                    serde_json::from_value(args).unwrap_or(TracesToolParams {
                        limit: 10,
                        span_filter: None,
                    });
                handle_traces(&state.config, &state.base_dir, p)
            }
            _ => unreachable!("tool name validated above"),
        }
    }
    .await;

    match tool_result {
        Ok(value) => Ok(serde_json::json!({
            "content": [{ "type": "text", "text": value.to_string() }]
        })),
        // Use {e} (not {e:#}) to avoid leaking internal filesystem paths in error chains.
        Err(e) => Ok(serde_json::json!({
            "content": [{ "type": "text", "text": format!("Error: {e}") }],
            "isError": true,
        })),
    }
}

/// Returns true if the JSON-RPC request is a notification (no "id") targeting a notifications/ method.
fn is_notification_request(req: &serde_json::Value) -> bool {
    req.get("id").is_none()
        && req
            .get("method")
            .and_then(|m| m.as_str())
            .map(|m| m.starts_with("notifications/"))
            .unwrap_or(false)
}

/// Axum handler for POST /mcp — MCP Streamable HTTP transport.
///
/// Implements the single-endpoint POST pattern from the MCP spec.
/// The `protocolVersion` in the initialize response is `"2024-11-05"` (the version
/// of the MCP protocol, distinct from the Streamable HTTP transport spec revision).
async fn mcp_post_handler(
    State(state): State<ServeState>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));

    // Notifications have no id and expect no response body.
    if is_notification_request(&req) {
        return StatusCode::ACCEPTED.into_response();
    }

    let result = match method {
        "initialize" => Ok(handle_initialize_http()),
        "tools/list" => Ok(handle_tools_list_http()),
        "tools/call" => handle_tools_call_http(&state, params).await,
        "ping" => Ok(serde_json::json!({})),
        _ => Err(serde_json::json!({
            "code": -32601,
            "message": format!("Method not found: {method}"),
        })),
    };

    match result {
        Ok(value) => Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": value,
        }))
        .into_response(),
        Err(err_obj) => Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": err_obj,
        }))
        .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Public entry point for HTTP MCP server
// ---------------------------------------------------------------------------

/// Start the HTTP MCP server. Called from `main.rs` when `corvia serve` is invoked.
///
/// Opens RedbIndex and TantivyIndex once at startup and holds them open for the
/// lifetime of the process. Multiple concurrent clients can connect; reads are
/// fully concurrent, writes are serialized via a tokio Mutex.
pub async fn serve_http(base_dir_arg: Option<&std::path::Path>, host: &str, port: u16) -> Result<()> {
    let base_dir = corvia_core::discover::resolve_base_dir(base_dir_arg)?;
    let config = Config::load_discovered(&base_dir).context("loading config")?;

    // Ensure required directories exist (handles fresh installs before first ingest).
    let index_dir = base_dir.join(config.index_dir());
    let entries_dir = base_dir.join(config.entries_dir());
    std::fs::create_dir_all(&index_dir)
        .with_context(|| format!("creating index dir: {}", index_dir.display()))?;
    std::fs::create_dir_all(&entries_dir)
        .with_context(|| format!("creating entries dir: {}", entries_dir.display()))?;

    // Open index handles once.
    info!("opening index handles");
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))
        .context("opening redb index")?;
    let tantivy_index = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))
        .context("opening tantivy index")?;

    // Create embedder once (model loading is expensive).
    info!("initializing embedder (this may download models on first run)");
    let cache_dir = config.embedding.model_path.clone();
    let embedder = Embedder::new(
        cache_dir.as_deref(),
        &config.embedding.model,
        &config.embedding.reranker_model,
    )
    .context("initializing embedder")?;
    info!("embedder ready");

    let handles = Arc::new(IndexHandles {
        redb,
        tantivy: tantivy_index,
        write_lock: TokioMutex::new(()),
    });

    let state = ServeState {
        config: Arc::new(config),
        embedder: Arc::new(embedder),
        base_dir,
        handles,
    };

    let app = Router::new()
        .route("/mcp", post(mcp_post_handler))
        .route("/healthz", get(healthz_handler))
        .layer(axum::extract::DefaultBodyLimit::max(1024 * 1024))
        .with_state(state);

    // Security note: no authentication is implemented (by design for localhost-only use).
    // If binding to a non-loopback address, warn explicitly.
    let is_loopback = host == "127.0.0.1" || host == "::1" || host == "localhost";
    if !is_loopback {
        eprintln!(
            "WARNING: corvia serve is binding to {host} (not localhost). \
             The MCP server has no authentication — any client on this network \
             can read and write the knowledge store."
        );
    }

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;

    eprintln!("corvia HTTP MCP server ready at http://{addr}/mcp");
    info!("corvia HTTP MCP server listening on http://{addr}/mcp");

    axum::serve(listener, app)
        .await
        .context("HTTP server error")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start the stdio MCP server. Called from `main.rs` when the `mcp` subcommand
/// is invoked.
///
/// Loads config, creates the Embedder (expensive -- model loading), then starts
/// the rmcp server loop on stdin/stdout. Blocks until the client disconnects.
pub async fn run(base_dir_arg: Option<&Path>) -> Result<()> {
    let base_dir = corvia_core::discover::resolve_base_dir(base_dir_arg)?;

    // Load config via project root discovery.
    let config = Config::load_discovered(&base_dir)
        .context("loading config")?;

    // Create embedder once (downloads models on first run).
    info!("initializing embedder (this may download models on first run)");
    let cache_dir = config.embedding.model_path.clone();
    let embedder = Embedder::new(cache_dir.as_deref(), &config.embedding.model, &config.embedding.reranker_model)
        .context("initializing embedder")?;
    info!("embedder ready");

    let server = CorviaServer::new(config, embedder, base_dir);

    // Create stdio transport and start serving.
    let transport = rmcp::transport::io::stdio();
    let service = rmcp::serve_server(server, transport)
        .await
        .map_err(|e: std::io::Error| anyhow::anyhow!("MCP server failed to start: {e}"))?;

    info!("corvia MCP server running on stdio");

    // Block until the client disconnects.
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server task failed: {e}"))?;

    info!("corvia MCP server stopped");
    Ok(())
}

/// Run a self-test: load config, initialize embedder, verify tools, run a
/// test search. Prints a diagnostic report and exits.
pub async fn run_test(base_dir_arg: Option<&Path>) -> Result<()> {
    use std::time::Instant;

    let base_dir = corvia_core::discover::resolve_base_dir(base_dir_arg)?;

    print!("  config:     ");
    let config = Config::load_discovered(&base_dir)
        .context("loading config")?;
    println!(".corvia/corvia.toml (ok)");

    print!("  models:     ");
    let start = Instant::now();
    let cache_dir = config.embedding.model_path.clone();
    let embedder = Embedder::new(
        cache_dir.as_deref(),
        &config.embedding.model,
        &config.embedding.reranker_model,
    )
    .context("initializing embedder")?;
    let elapsed = start.elapsed();
    println!(
        "{} + {} (loaded in {:.1}s)",
        config.embedding.model,
        config.embedding.reranker_model,
        elapsed.as_secs_f64()
    );

    let tools = vec![search_tool(), write_tool(), status_tool(), traces_tool()];
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    println!("  tools:      {} ({})", tools.len(), tool_names.join(", "));

    let status = handle_status(&config, &base_dir)?;
    if let Some(entries) = status.get("entry_count").and_then(|v| v.as_u64()) {
        println!("  entries:    {entries}");
    }

    // Drop embedder to free model memory
    drop(embedder);

    println!("  status:     ready");
    Ok(())
}

#[cfg(test)]
mod http_tests {
    use super::*;

    #[test]
    fn initialize_response_has_required_fields() {
        let resp = handle_initialize_http();
        assert!(resp.get("protocolVersion").is_some(), "missing protocolVersion");
        assert!(resp.get("capabilities").is_some(), "missing capabilities");
        assert!(resp.get("serverInfo").is_some(), "missing serverInfo");
        let caps = resp["capabilities"].as_object().unwrap();
        assert!(caps.contains_key("tools"), "capabilities missing tools");
    }

    #[test]
    fn tools_list_response_has_four_tools() {
        let resp = handle_tools_list_http();
        let tools = resp["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 4, "expected 4 tools");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        assert!(names.contains(&"corvia_search"));
        assert!(names.contains(&"corvia_write"));
        assert!(names.contains(&"corvia_status"));
        assert!(names.contains(&"corvia_traces"));
    }

    #[test]
    fn notification_detection_identifies_notifications_correctly() {
        // Verify the notification detection helper correctly identifies notification requests.
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });
        assert!(
            is_notification_request(&notification),
            "notifications/initialized without id should be identified as notification"
        );

        // Regular requests with an id are NOT notifications.
        let regular_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "notifications/initialized",
            "params": {}
        });
        assert!(
            !is_notification_request(&regular_request),
            "request with id should not be identified as notification"
        );

        // Non-notifications/ methods without id are also NOT notifications.
        let non_notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {}
        });
        assert!(
            !is_notification_request(&non_notification),
            "initialize without id should not be identified as notification"
        );
    }

    #[test]
    fn healthz_core_logic_returns_entry_count_for_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::default();
        let redb = RedbIndex::open(&dir.path().join("store.redb")).unwrap();
        let tantivy = TantivyIndex::open(&dir.path().join("tantivy")).unwrap();

        let result = handle_status_with_handles(&config, dir.path(), &redb, &tantivy);
        assert!(result.is_ok(), "handle_status_with_handles failed: {:?}", result);
        let val = result.unwrap();
        assert_eq!(
            val["entry_count"].as_u64().unwrap_or(999),
            0,
            "expected 0 entries in a fresh index"
        );
    }
}
