use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Router,
};
use corvia_common::agent_types::*;
use corvia_common::types::EdgeDirection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::rest::AppState;

// --- MCP state (stateless — no session tracking) ---

// NOTE: McpSessions removed. The MCP server is fully stateless: no session
// validation, no session persistence. Agent identity comes from _meta.agent_id
// in tool calls, not from transport sessions. This eliminates the "invalid or
// expired session" errors that occurred after server restarts.

/// Shared state for MCP endpoints.
pub struct McpState {
    pub app: Arc<AppState>,
}

// --- JSON-RPC 2.0 types ---

#[derive(Deserialize, Debug)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Serialize, Clone)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Serialize, Clone)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }

    fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: None, error: Some(JsonRpcError { code, message, data: None }) }
    }
}

// JSON-RPC error codes
const METHOD_NOT_FOUND: i32 = -32601;
const INVALID_PARAMS: i32 = -32602;
const INTERNAL_ERROR: i32 = -32603;
/// Service unavailable (optional component not configured).
/// Uses -32000 from the JSON-RPC implementation-defined server error range (-32000..-32099).
const SERVICE_UNAVAILABLE: i32 = -32000;

// --- MCP Streamable HTTP headers ---

const MCP_SESSION_ID: &str = "mcp-session-id";

// --- MCP tool definitions ---

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "corvia_search",
            "description": "Search organizational knowledge entries by semantic similarity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" },
                    "scope_id": { "type": "string", "description": "Scope to search within" },
                    "limit": { "type": "integer", "description": "Maximum results (default 10)" }
                },
                "required": ["query", "scope_id"]
            }
        }),
        json!({
            "name": "corvia_write",
            "description": "Write a knowledge entry to organizational memory (requires agent identity)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The knowledge content to store" },
                    "scope_id": { "type": "string", "description": "Target scope" },
                    "source_version": { "type": "string", "description": "Source version reference" },
                    "agent_id": { "type": "string", "description": "Agent identity for attribution (e.g. 'claude-code')" }
                },
                "required": ["content", "scope_id"]
            }
        }),
        json!({
            "name": "corvia_history",
            "description": "Get the supersession history chain for a knowledge entry",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": { "type": "string", "description": "The entry UUID" }
                },
                "required": ["entry_id"]
            }
        }),
        json!({
            "name": "corvia_graph",
            "description": "Get graph edges for a knowledge entry",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": { "type": "string", "description": "The entry UUID" }
                },
                "required": ["entry_id"]
            }
        }),
        json!({
            "name": "corvia_reason",
            "description": "Run reasoning checks on a scope to find knowledge health issues",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": { "type": "string", "description": "Scope to analyze" },
                    "check": { "type": "string", "description": "Specific check type (stale, broken, orphan, dangling, cycle). Omit for all checks." }
                },
                "required": ["scope_id"]
            }
        }),
        json!({
            "name": "corvia_agent_status",
            "description": "Get the calling agent's contribution summary",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "Agent identity (e.g. 'claude-code')" }
                }
            }
        }),
        json!({
            "name": "corvia_context",
            "description": "Retrieve and assemble organizational knowledge context for a query (no generation)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" },
                    "scope_id": { "type": "string", "description": "Scope to search within" },
                    "limit": { "type": "integer", "description": "Maximum sources (default 10)" },
                    "expand_graph": { "type": "boolean", "description": "Follow graph edges (default true)" }
                },
                "required": ["query", "scope_id"]
            }
        }),
        json!({
            "name": "corvia_ask",
            "description": "Ask a question and get an AI-generated answer from organizational knowledge (full RAG)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The question to answer" },
                    "scope_id": { "type": "string", "description": "Scope to search within" },
                    "limit": { "type": "integer", "description": "Maximum sources (default 10)" },
                    "expand_graph": { "type": "boolean", "description": "Follow graph edges (default true)" }
                },
                "required": ["query", "scope_id"]
            }
        }),
    ]
}

// --- MCP Streamable HTTP router ---

pub fn mcp_router(state: Arc<AppState>) -> Router {
    let mcp_state = Arc::new(McpState {
        app: state,
    });
    Router::new()
        .route("/mcp", post(handle_mcp_post))
        .route("/mcp", get(handle_mcp_get))
        .route("/mcp", delete(handle_mcp_delete))
        .with_state(mcp_state)
}

// --- POST /mcp — main request handler ---

/// A parsed JSON-RPC message: either a single request or a batch.
enum JsonRpcInput {
    Single(JsonRpcRequest),
    Batch(Vec<JsonRpcRequest>),
}

async fn handle_mcp_post(
    State(state): State<Arc<McpState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Parse body as single JSON-RPC request or batch (array)
    let input = match serde_json::from_str::<JsonRpcRequest>(&body) {
        Ok(r) => JsonRpcInput::Single(r),
        Err(_) => match serde_json::from_str::<Vec<JsonRpcRequest>>(&body) {
            Ok(batch) if !batch.is_empty() => JsonRpcInput::Batch(batch),
            Ok(_) => {
                return (StatusCode::BAD_REQUEST, "Empty batch").into_response();
            }
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("Invalid JSON-RPC: {e}")).into_response();
            }
        },
    };

    // Peek at the first request to check for initialize / session validation
    let first = match &input {
        JsonRpcInput::Single(r) => r,
        JsonRpcInput::Batch(v) => &v[0],
    };
    let is_initialize = first.method == "initialize";

    // Stateless: no session validation. All corvia MCP tools are self-contained;
    // agent identity comes from _meta.agent_id in tool calls, not transport sessions.
    // Any Mcp-Session-Id header from the client is accepted but not validated.

    // Process request(s)
    let (responses, is_batch) = match input {
        JsonRpcInput::Single(req) => {
            let resp = process_single_request(&state, &req).await;
            (resp.into_iter().collect::<Vec<_>>(), false)
        }
        JsonRpcInput::Batch(reqs) => {
            let mut responses = Vec::new();
            for req in &reqs {
                if let Some(resp) = process_single_request(&state, req).await {
                    responses.push(resp);
                }
            }
            (responses, true)
        }
    };

    // Stateless: return a session ID for protocol compatibility (clients may expect it),
    // but don't track it. Reuse client-provided ID or generate a fresh one on initialize.
    let session_header: Option<String> = if is_initialize {
        Some(Uuid::now_v7().to_string())
    } else {
        headers.get(MCP_SESSION_ID)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    };

    // All notifications, no responses to send
    if responses.is_empty() {
        return StatusCode::ACCEPTED.into_response();
    }

    // Serialize response body
    let json_body = if is_batch {
        serde_json::to_string(&responses).unwrap()
    } else {
        serde_json::to_string(&responses[0]).unwrap()
    };

    // Per MCP Streamable HTTP spec: for single JSON-RPC responses, prefer
    // application/json when the client accepts it. Only use SSE when the client
    // exclusively accepts text/event-stream or for streaming scenarios.
    let accept = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");
    let accepts_json = accept.contains("application/json") || accept.contains("*/*");
    let accepts_sse = accept.contains("text/event-stream");

    // Prefer JSON for single responses; use SSE only if client doesn't accept JSON
    if accepts_sse && !accepts_json {
        let sse_body = format!("event: message\ndata: {json_body}\n\n");
        let mut builder = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache");
        if let Some(ref sid) = session_header {
            builder = builder.header(MCP_SESSION_ID, sid.as_str());
        }
        builder.body(sse_body.into()).unwrap()
    } else {
        let mut builder = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json");
        if let Some(ref sid) = session_header {
            builder = builder.header(MCP_SESSION_ID, sid.as_str());
        }
        builder.body(json_body.into()).unwrap()
    }
}

/// Process a single JSON-RPC request. Returns None for notifications (no response needed).
async fn process_single_request(
    state: &McpState,
    req: &JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let is_notification = req.id.is_none() && req.method != "initialize";

    if is_notification {
        // Fire-and-forget: dispatch but don't return a response
        let _ = dispatch(&state.app, req).await;
        return None;
    }

    let response = dispatch(&state.app, req).await;
    let rpc_response = match response {
        Ok(result) => JsonRpcResponse::success(req.id.clone(), result),
        Err((code, msg)) => JsonRpcResponse::error(req.id.clone(), code, msg),
    };
    Some(rpc_response)
}

// --- GET /mcp — SSE stream for server-to-client notifications ---

async fn handle_mcp_get(
    State(_state): State<Arc<McpState>>,
    _headers: HeaderMap,
) -> Response {
    // Stateless: open an SSE keepalive stream for any client (no session validation).
    // Corvia doesn't currently push server-initiated messages, but the stream
    // satisfies MCP clients that open a GET connection for notifications.
    let (_tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// --- DELETE /mcp — terminate session ---

async fn handle_mcp_delete(
    State(_state): State<Arc<McpState>>,
    headers: HeaderMap,
) -> Response {
    // Stateless: accept any DELETE as a no-op (nothing to clean up).
    if let Some(session_id) = headers.get(MCP_SESSION_ID).and_then(|v| v.to_str().ok()) {
        info!(session_id = %session_id, "MCP session terminate requested (stateless, no-op)");
    }
    StatusCode::OK.into_response()
}

// --- JSON-RPC dispatch ---

async fn dispatch(
    state: &AppState,
    req: &JsonRpcRequest,
) -> Result<Value, (i32, String)> {
    match req.method.as_str() {
        "initialize" => handle_initialize(&req.params),
        "notifications/initialized" => Ok(json!({})),
        "tools/list" => Ok(handle_tools_list()),
        "tools/call" => handle_tools_call(state, &req.params).await,
        other => Err((METHOD_NOT_FOUND, format!("Method not found: {other}"))),
    }
}

fn handle_initialize(params: &Value) -> Result<Value, (i32, String)> {
    // Extract clientInfo for logging
    if let Some(client_info) = params.get("clientInfo") {
        let name = client_info.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
        let version = client_info.get("version").and_then(|v| v.as_str()).unwrap_or("unknown");
        info!(client_name = name, client_version = version, "MCP client connected");
    }

    // Negotiate protocol version: echo back client's requested version if we support it,
    // otherwise default to latest we support. This ensures compatibility with clients
    // that only support older protocol versions (e.g., Claude Code uses "2024-11-05").
    let supported_versions = ["2025-03-26", "2024-11-05"];
    let client_version = params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("2024-11-05");
    let negotiated = if supported_versions.contains(&client_version) {
        client_version
    } else {
        "2024-11-05"
    };

    Ok(json!({
        "protocolVersion": negotiated,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "corvia",
            "version": env!("CARGO_PKG_VERSION")
        }
    }))
}

fn handle_tools_list() -> Value {
    json!({ "tools": tool_definitions() })
}

async fn handle_tools_call(
    state: &AppState,
    params: &Value,
) -> Result<Value, (i32, String)> {
    let tool_name = params.get("name")
        .and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing tool name".into()))?;

    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    // Resolve agent identity: arguments.agent_id > _meta.agent_id > None
    let meta = params.get("_meta");
    let agent_id = arguments.get("agent_id").and_then(|v| v.as_str())
        .or_else(|| meta.and_then(|m| m.get("agent_id")).and_then(|v| v.as_str()));

    match tool_name {
        "corvia_search" => tool_corvia_search(state, &arguments, agent_id).await,
        "corvia_write" => tool_corvia_write(state, &arguments, agent_id).await,
        "corvia_history" => tool_corvia_history(state, &arguments).await,
        "corvia_graph" => tool_corvia_graph(state, &arguments).await,
        "corvia_reason" => tool_corvia_reason(state, &arguments).await,
        "corvia_agent_status" => tool_corvia_agent_status(state, agent_id),
        "corvia_context" => tool_corvia_context(state, &arguments).await,
        "corvia_ask" => tool_corvia_ask(state, &arguments).await,
        other => Err((METHOD_NOT_FOUND, format!("Unknown tool: {other}"))),
    }
}

// --- Tool implementations ---

async fn tool_corvia_search(
    state: &AppState,
    args: &Value,
    _agent_id: Option<&str>,
) -> Result<Value, (i32, String)> {
    let query = args.get("query").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'query' parameter".into()))?;
    let scope_id = args.get("scope_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'scope_id' parameter".into()))?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    // Route through RAG pipeline if available (fixes ContextBuilder bypass)
    if let Some(rag) = &state.rag {
        let opts = corvia_kernel::rag_types::RetrievalOpts {
            limit,
            expand_graph: false, // search endpoint: pure vector (context/ask use graph)
            ..Default::default()
        };
        let response = rag.context(query, scope_id, Some(opts)).await
            .map_err(|e| (INTERNAL_ERROR, format!("Search failed: {e}")))?;

        let items: Vec<Value> = response.context.sources.iter().map(|r| {
            json!({
                "content": r.entry.content,
                "score": r.score,
                "source_file": r.entry.metadata.source_file,
                "language": r.entry.metadata.language,
            })
        }).collect();

        return Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&json!({
                    "results": items,
                    "count": items.len()
                })).unwrap()
            }]
        }));
    }

    // Fallback: raw store search
    let embedding = state.engine.embed(query).await
        .map_err(|e| (INTERNAL_ERROR, format!("Embedding failed: {e}")))?;

    let results = state.store.search(&embedding, scope_id, limit).await
        .map_err(|e| (INTERNAL_ERROR, format!("Search failed: {e}")))?;

    let items: Vec<Value> = results.iter().map(|r| {
        json!({
            "content": r.entry.content,
            "score": r.score,
            "source_file": r.entry.metadata.source_file,
            "language": r.entry.metadata.language,
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "results": items,
                "count": items.len()
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_write(
    state: &AppState,
    args: &Value,
    agent_id: Option<&str>,
) -> Result<Value, (i32, String)> {
    // Require agent identity for writes (D45 Part 2)
    let agent_id = agent_id
        .ok_or((INVALID_PARAMS, "Write requires _meta.agent_id — anonymous clients are read-only".into()))?;

    let content = args.get("content").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'content' parameter".into()))?;
    let scope_id = args.get("scope_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'scope_id' parameter".into()))?;
    let source_version = args.get("source_version").and_then(|v| v.as_str()).unwrap_or("mcp");

    let coord = &state.coordinator;

    // Ensure agent has a session (create one if needed)
    let mut connect = coord.connect(agent_id)
        .unwrap_or_else(|_| corvia_kernel::agent_coordinator::ConnectResponse {
            agent_id: agent_id.into(),
            active_sessions: vec![],
            recoverable_sessions: vec![],
        });

    if connect.active_sessions.is_empty() {
        // Auto-register MCP agent and create session
        let identity = AgentIdentity::McpClient {
            client_name: agent_id.to_string(),
            client_version: "unknown".into(),
            agent_hint: None,
        };
        let _ = coord.register_agent(
            &identity,
            agent_id,
            AgentPermission::ReadWrite { scopes: vec![scope_id.to_string()] },
        );
        coord.create_session(agent_id, false)
            .map_err(|e| (INTERNAL_ERROR, format!("Session creation failed: {e}")))?;
        connect = coord.connect(agent_id)
            .map_err(|e| (INTERNAL_ERROR, format!("Agent setup failed: {e}")))?;
    }

    let session_id = connect.active_sessions.first()
        .map(|s| s.session_id.as_str())
        .ok_or((INTERNAL_ERROR, "No active session".into()))?;

    let entry = coord.write_entry(session_id, content, scope_id, source_version).await
        .map_err(|e| (INTERNAL_ERROR, format!("Write failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("Entry {} written (status: {:?})", entry.id, entry.entry_status)
        }]
    }))
}

async fn tool_corvia_history(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let entry_id = args.get("entry_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'entry_id' parameter".into()))?;

    let uuid = uuid::Uuid::parse_str(entry_id)
        .map_err(|e| (INVALID_PARAMS, format!("Invalid UUID: {e}")))?;

    let chain = state.temporal.history(&uuid).await
        .map_err(|e| (INTERNAL_ERROR, format!("History query failed: {e}")))?;

    if chain.is_empty() {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": format!("No history found for entry {entry_id}")
            }]
        }));
    }

    let mut lines = Vec::new();
    lines.push(format!("History for entry {} ({} entries):", entry_id, chain.len()));
    for (i, entry) in chain.iter().enumerate() {
        let status = if entry.is_current() { " (current)" } else { "" };
        let valid_to_str = entry.valid_to
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "now".to_string());
        let content_preview: String = entry.content.chars().take(80).collect();
        let ellipsis = if entry.content.chars().nth(80).is_some() { "..." } else { "" };
        lines.push(format!(
            "[{}] {}{}\n  recorded: {}\n  valid: {} -> {}\n  content: {content_preview}{ellipsis}",
            i,
            entry.id,
            status,
            entry.recorded_at.format("%Y-%m-%d %H:%M:%S"),
            entry.valid_from.format("%Y-%m-%d %H:%M:%S"),
            valid_to_str,
        ));
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": lines.join("\n")
        }]
    }))
}

async fn tool_corvia_graph(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let entry_id = args.get("entry_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'entry_id' parameter".into()))?;

    let uuid = uuid::Uuid::parse_str(entry_id)
        .map_err(|e| (INVALID_PARAMS, format!("Invalid UUID: {e}")))?;

    let edges = state.graph.edges(&uuid, EdgeDirection::Both).await
        .map_err(|e| (INTERNAL_ERROR, format!("Graph query failed: {e}")))?;

    if edges.is_empty() {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": format!("No edges found for entry {entry_id}")
            }]
        }));
    }

    let items: Vec<Value> = edges.iter().map(|e| {
        json!({
            "from": e.from.to_string(),
            "to": e.to.to_string(),
            "relation": e.relation,
            "metadata": e.metadata,
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "edges": items,
                "count": items.len()
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_reason(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let scope_id = args.get("scope_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'scope_id' parameter".into()))?;
    let check_str = args.get("check").and_then(|v| v.as_str());

    // Load entries from knowledge files (direct disk read, not HNSW search)
    let entries = corvia_kernel::knowledge_files::read_scope(&state.data_dir, scope_id)
        .map_err(|e| (INTERNAL_ERROR, format!("Failed to load entries: {e}")))?;

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*state.store, &*state.graph);

    let findings = if let Some(check) = check_str {
        let check_type = check.parse::<corvia_kernel::reasoner::CheckType>()
            .map_err(|e| (INVALID_PARAMS, e))?;
        reasoner.run_check(&entries, scope_id, check_type).await
            .map_err(|e| (INTERNAL_ERROR, format!("Reasoning failed: {e}")))?
    } else {
        reasoner.run_all(&entries, scope_id).await
            .map_err(|e| (INTERNAL_ERROR, format!("Reasoning failed: {e}")))?
    };

    if findings.is_empty() {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": format!("No issues found in scope '{scope_id}'.")
            }]
        }));
    }

    let items: Vec<Value> = findings.iter().map(|f| {
        json!({
            "check_type": f.check_type.as_str(),
            "confidence": f.confidence,
            "rationale": f.rationale,
            "target_ids": f.target_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "scope_id": scope_id,
                "findings": items,
                "count": items.len()
            })).unwrap()
        }]
    }))
}


fn tool_corvia_agent_status(
    state: &AppState,
    agent_id: Option<&str>,
) -> Result<Value, (i32, String)> {
    let agent_id = agent_id
        .ok_or((INVALID_PARAMS, "agent_status requires _meta.agent_id".into()))?;

    let coord = &state.coordinator;
    let connect = coord.connect(agent_id)
        .map_err(|e| (INTERNAL_ERROR, format!("Connect failed: {e}")))?;

    let total_written: u64 = connect.active_sessions.iter()
        .chain(connect.recoverable_sessions.iter())
        .map(|s| s.entries_written)
        .sum();
    let total_merged: u64 = connect.active_sessions.iter()
        .chain(connect.recoverable_sessions.iter())
        .map(|s| s.entries_merged)
        .sum();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "agent_id": agent_id,
                "active_sessions": connect.active_sessions.len(),
                "recoverable_sessions": connect.recoverable_sessions.len(),
                "total_entries_written": total_written,
                "total_entries_merged": total_merged,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_context(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let query = args.get("query").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'query' parameter".into()))?;
    let scope_id = args.get("scope_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'scope_id' parameter".into()))?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let expand_graph = args.get("expand_graph").and_then(|v| v.as_bool()).unwrap_or(true);

    let rag = state.rag.as_ref()
        .ok_or((SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit,
        expand_graph,
        ..Default::default()
    };

    let response = rag.context(query, scope_id, Some(opts)).await
        .map_err(|e| (INTERNAL_ERROR, format!("RAG failed: {e}")))?;

    let sources: Vec<Value> = response.context.sources.iter().map(|r| {
        json!({
            "content": r.entry.content,
            "score": r.score,
            "source_file": r.entry.metadata.source_file,
            "language": r.entry.metadata.language,
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "context": response.context.context,
                "sources": sources,
                "trace": response.trace,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_ask(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let query = args.get("query").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'query' parameter".into()))?;
    let scope_id = args.get("scope_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'scope_id' parameter".into()))?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let expand_graph = args.get("expand_graph").and_then(|v| v.as_bool()).unwrap_or(true);

    let rag = state.rag.as_ref()
        .ok_or((SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit,
        expand_graph,
        ..Default::default()
    };

    let response = rag.ask(query, scope_id, Some(opts)).await
        .map_err(|e| (INTERNAL_ERROR, format!("RAG failed: {e}")))?;

    let sources: Vec<Value> = response.context.sources.iter().map(|r| {
        json!({
            "content": r.entry.content,
            "score": r.score,
            "source_file": r.entry.metadata.source_file,
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "answer": response.answer,
                "sources": sources,
                "trace": response.trace,
            })).unwrap()
        }]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_kernel::agent_coordinator::AgentCoordinator;
    use corvia_kernel::lite_store::LiteStore;
    use corvia_kernel::traits::{GenerationEngine, GenerationResult, GraphStore, InferenceEngine, QueryableStore, TemporalStore};

    struct MockEngine;
    #[async_trait::async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize { 3 }
    }

    struct MockGenerationEngine;
    #[async_trait::async_trait]
    impl GenerationEngine for MockGenerationEngine {
        fn name(&self) -> &str { "mock" }
        async fn generate(&self, _system_prompt: &str, user_message: &str) -> corvia_common::errors::Result<GenerationResult> {
            Ok(GenerationResult {
                text: format!("merged: {user_message}"),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize { 4096 }
    }

    async fn test_state(dir: &std::path::Path) -> Arc<AppState> {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen_engine = Arc::new(MockGenerationEngine) as Arc<dyn GenerationEngine>;
        let coordinator = Arc::new(AgentCoordinator::new(
            store.clone() as Arc<dyn QueryableStore>,
            engine.clone(),
            dir,
            corvia_common::config::AgentLifecycleConfig::default(),
            corvia_common::config::MergeConfig { similarity_threshold: 2.0, ..Default::default() },
            gen_engine,
        ).unwrap());
        Arc::new(AppState {
            store: store.clone() as Arc<dyn QueryableStore>,
            engine,
            coordinator,
            graph: store.clone() as Arc<dyn GraphStore>,
            temporal: store as Arc<dyn TemporalStore>,
            data_dir: dir.to_path_buf(),
            rag: None,
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    #[tokio::test]
    async fn test_initialize() {
        // Client sends 2024-11-05 — server should echo it back
        let params = json!({
            "protocolVersion": "2024-11-05",
            "clientInfo": { "name": "test-client", "version": "1.0" }
        });
        let result = handle_initialize(&params).unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["serverInfo"]["name"].as_str().unwrap() == "corvia");

        // Client sends 2025-03-26 — server should echo it back
        let params2 = json!({
            "protocolVersion": "2025-03-26",
            "clientInfo": { "name": "test-client", "version": "1.0" }
        });
        let result2 = handle_initialize(&params2).unwrap();
        assert_eq!(result2["protocolVersion"], "2025-03-26");

        // Client sends unknown version — server should default to 2024-11-05
        let params3 = json!({
            "protocolVersion": "9999-01-01",
            "clientInfo": { "name": "test-client", "version": "1.0" }
        });
        let result3 = handle_initialize(&params3).unwrap();
        assert_eq!(result3["protocolVersion"], "2024-11-05");
    }

    #[tokio::test]
    async fn test_tools_list() {
        let result = handle_tools_list();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 8);

        let names: Vec<&str> = tools.iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"corvia_search"));
        assert!(names.contains(&"corvia_write"));
        assert!(names.contains(&"corvia_history"));
        assert!(names.contains(&"corvia_graph"));
        assert!(names.contains(&"corvia_reason"));
        assert!(names.contains(&"corvia_agent_status"));
        assert!(names.contains(&"corvia_context"));
        assert!(names.contains(&"corvia_ask"));
    }

    #[tokio::test]
    async fn test_corvia_search() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert a test entry
        let entry = corvia_common::types::KnowledgeEntry::new(
            "test knowledge".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        state.store.insert(&entry).await.unwrap();

        let args = json!({ "query": "test", "scope_id": "test-scope" });
        let result = tool_corvia_search(&state, &args, None).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["count"], 1);
    }

    #[tokio::test]
    async fn test_corvia_write_without_meta_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let args = json!({ "content": "some knowledge", "scope_id": "test-scope" });
        let result = tool_corvia_write(&state, &args, None).await;

        assert!(result.is_err());
        let (code, msg) = result.unwrap_err();
        assert_eq!(code, INVALID_PARAMS);
        assert!(msg.contains("read-only"));
    }

    #[tokio::test]
    async fn test_corvia_write_with_agent_id() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Write through coordinator path (auto-registers MCP agent)
        let args = json!({ "content": "agent knowledge", "scope_id": "test-scope" });
        let result = tool_corvia_write(&state, &args, Some("mcp::test-agent")).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("written"));

        // Verify entry was stored
        let count = state.store.count("test-scope").await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_corvia_history_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Querying history for a nonexistent entry returns an error
        let uuid = uuid::Uuid::now_v7();
        let args = json!({ "entry_id": uuid.to_string() });
        let result = tool_corvia_history(&state, &args).await;

        assert!(result.is_err());
        let (code, msg) = result.unwrap_err();
        assert_eq!(code, INTERNAL_ERROR);
        assert!(msg.contains("not found"));
    }

    #[tokio::test]
    async fn test_corvia_history_with_entry() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert an entry so it has a history chain of 1
        let entry = corvia_common::types::KnowledgeEntry::new(
            "historical content".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        let entry_id = entry.id;
        state.store.insert(&entry).await.unwrap();

        let args = json!({ "entry_id": entry_id.to_string() });
        let result = tool_corvia_history(&state, &args).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("History for entry"));
        assert!(text.contains(&entry_id.to_string()));
    }

    #[tokio::test]
    async fn test_corvia_history_invalid_uuid() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let args = json!({ "entry_id": "not-a-uuid" });
        let result = tool_corvia_history(&state, &args).await;

        assert!(result.is_err());
        let (code, _) = result.unwrap_err();
        assert_eq!(code, INVALID_PARAMS);
    }

    #[tokio::test]
    async fn test_corvia_graph_no_edges() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let entry = corvia_common::types::KnowledgeEntry::new(
            "lonely node".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        state.store.insert(&entry).await.unwrap();

        let args = json!({ "entry_id": entry.id.to_string() });
        let result = tool_corvia_graph(&state, &args).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No edges found"));
    }

    #[tokio::test]
    async fn test_corvia_graph_with_edges() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let e1 = corvia_common::types::KnowledgeEntry::new(
            "module A".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = corvia_common::types::KnowledgeEntry::new(
            "module B".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![0.0, 1.0, 0.0]);
        state.store.insert(&e1).await.unwrap();
        state.store.insert(&e2).await.unwrap();
        state.graph.relate(&e1.id, "imports", &e2.id, None).await.unwrap();

        let args = json!({ "entry_id": e1.id.to_string() });
        let result = tool_corvia_graph(&state, &args).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["count"], 1);
        assert_eq!(parsed["edges"][0]["relation"], "imports");
    }

    #[tokio::test]
    async fn test_corvia_reason_no_issues() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Empty scope — no entries means no findings
        let args = json!({ "scope_id": "empty-scope" });
        let result = tool_corvia_reason(&state, &args).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No issues found"));
    }

    #[tokio::test]
    async fn test_corvia_reason_finds_orphans() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert entries with no graph edges — they should be flagged as orphaned
        let e1 = corvia_common::types::KnowledgeEntry::new(
            "orphaned knowledge".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        state.store.insert(&e1).await.unwrap();

        let args = json!({ "scope_id": "test-scope", "check": "orphan" });
        let result = tool_corvia_reason(&state, &args).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert!(parsed["count"].as_u64().unwrap() >= 1);
        assert_eq!(parsed["findings"][0]["check_type"], "orphaned_node");
    }

    #[tokio::test]
    async fn test_corvia_reason_invalid_check() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let args = json!({ "scope_id": "test-scope", "check": "bogus" });
        let result = tool_corvia_reason(&state, &args).await;

        assert!(result.is_err());
        let (code, _) = result.unwrap_err();
        assert_eq!(code, INVALID_PARAMS);
    }

    /// When AppState has rag: Some(...), corvia_search should route through RAG.
    #[tokio::test]
    async fn test_corvia_search_routes_through_rag() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();

        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;

        // Insert entries with embeddings and Merged status for visibility.
        for i in 0..10 {
            let mut entry = corvia_common::types::KnowledgeEntry::new(
                format!("rag-routed item {i}"),
                "rag-scope".into(),
                "v1".into(),
            );
            entry.embedding = Some(vec![1.0, (i as f32) * 0.001, 0.0]);
            entry.entry_status = corvia_common::agent_types::EntryStatus::Merged;
            store.insert(&entry).await.unwrap();
        }

        // Build a RAG pipeline and inject into AppState.
        let config = corvia_common::config::CorviaConfig::default();
        let rag = Arc::new(corvia_kernel::create_rag_pipeline(
            store.clone() as Arc<dyn QueryableStore>,
            engine.clone(),
            None,
            None,
            &config,
        ));

        let gen_engine = Arc::new(MockGenerationEngine) as Arc<dyn GenerationEngine>;
        let coordinator = Arc::new(AgentCoordinator::new(
            store.clone() as Arc<dyn QueryableStore>,
            engine.clone(),
            dir.path(),
            corvia_common::config::AgentLifecycleConfig::default(),
            corvia_common::config::MergeConfig::default(),
            gen_engine,
        ).unwrap());

        let state = Arc::new(AppState {
            store: store.clone() as Arc<dyn QueryableStore>,
            engine,
            coordinator,
            graph: store.clone() as Arc<dyn GraphStore>,
            temporal: store as Arc<dyn TemporalStore>,
            data_dir: dir.path().to_path_buf(),
            rag: Some(rag),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        });

        let args = json!({ "query": "rag-routed", "scope_id": "rag-scope", "limit": 5 });
        let result = tool_corvia_search(&state, &args, None).await.unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        // RAG-routed search should still return results.
        assert!(
            parsed["count"].as_u64().unwrap() >= 1,
            "RAG-routed search should return results, got: {text}"
        );
    }

    #[tokio::test]
    async fn test_dispatch_unknown_method() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(1)),
            method: "unknown/method".into(),
            params: json!({}),
        };
        let result = dispatch(&state, &req).await;
        assert!(result.is_err());
        let (code, _) = result.unwrap_err();
        assert_eq!(code, METHOD_NOT_FOUND);
    }

}
