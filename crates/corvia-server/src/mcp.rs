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
use tower_http::trace::TraceLayer;
use tracing::info;
use uuid::Uuid;

use crate::rest::AppState;

/// MCP tools that modify state and require bearer token auth when the server
/// is bound to a non-loopback address.
const WRITE_TOOLS: &[&str] = &[
    "corvia_write",
    "corvia_gc_run",
    "corvia_config_set",
    "corvia_rebuild_index",
    "corvia_agent_suspend",
    "corvia_merge_retry",
    "corvia_pin",
    "corvia_unpin",
];

/// Constant-time byte comparison to prevent timing side-channel attacks on token validation.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Extract bearer token from Authorization header.
fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

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
/// Unauthorized (bearer token missing or invalid for write operations).
const UNAUTHORIZED: i32 = -32001;

/// Check if a tool call has confirmation via _meta.confirmed.
fn is_confirmed(meta: Option<&Value>) -> bool {
    meta.and_then(|m| m.get("confirmed"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Check if dry_run is requested in arguments.
fn is_dry_run(args: &Value) -> bool {
    args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false)
}

/// Return a confirmation-required response for unconfirmed Tier 2+ tools.
fn confirmation_response(preview: Value, message: &str) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "confirmation_required": true,
                "preview": preview,
                "message": message,
            })).unwrap()
        }]
    })
}

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
                    "scope_id": { "type": "string", "description": "Scope to search within (defaults to workspace scope if omitted)" },
                    "limit": { "type": "integer", "description": "Maximum results (default 10)" },
                    "content_role": { "type": "string", "description": "Filter by content role: design, decision, plan, code, memory, finding, instruction, learning" },
                    "source_origin": { "type": "string", "description": "Filter by source origin: repo:<name>, workspace, memory" },
                    "workstream": { "type": "string", "description": "Filter by workstream (e.g. git branch name)" },
                    "include_cold": { "type": "boolean", "description": "Include Cold-tier entries via brute-force cosine scan (default false)" },
                    "min_score": { "type": "number", "description": "Minimum score threshold — results below this are filtered out" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "corvia_write",
            "description": "Write a knowledge entry to organizational memory (requires agent identity)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The knowledge content to store" },
                    "scope_id": { "type": "string", "description": "Target scope (defaults to workspace scope if omitted)" },
                    "source_version": { "type": "string", "description": "Source version reference" },
                    "agent_id": { "type": "string", "description": "Agent identity for attribution (e.g. 'claude-code')" },
                    "content_role": { "type": "string", "description": "Content role: design, decision, plan, code, memory, finding, instruction, learning" },
                    "source_origin": { "type": "string", "description": "Source origin: repo:<name>, workspace, memory" },
                    "force_write": { "type": "boolean", "description": "Bypass semantic deduplication check (default false)" }
                },
                "required": ["content"]
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
                    "scope_id": { "type": "string", "description": "Scope to analyze (defaults to workspace scope if omitted)" },
                    "check": { "type": "string", "description": "Specific check type (stale, broken, orphan, dangling, cycle). Omit for all checks." }
                }
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
                    "scope_id": { "type": "string", "description": "Scope to search within (defaults to workspace scope if omitted)" },
                    "limit": { "type": "integer", "description": "Maximum sources (default 10)" },
                    "expand_graph": { "type": "boolean", "description": "Follow graph edges (default true)" },
                    "content_role": { "type": "string", "description": "Filter by content role: design, decision, plan, code, memory, finding, instruction, learning" },
                    "source_origin": { "type": "string", "description": "Filter by source origin: repo:<name>, workspace, memory" },
                    "workstream": { "type": "string", "description": "Filter by workstream (e.g. git branch name)" },
                    "include_cold": { "type": "boolean", "description": "Include Cold-tier entries via brute-force cosine scan (default false)" },
                    "max_tokens": { "type": "integer", "description": "Override token budget for context assembly (hard cap: 4000). Recommended: 2000-3000 for subagent injection." },
                    "format": { "type": "string", "description": "Output format: 'default' (full context with citations) or 'compact' (relevance-ranked blocks with entry IDs, no system prompt)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "corvia_ask",
            "description": "Ask a question and get an AI-generated answer from organizational knowledge (full RAG)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The question to answer" },
                    "scope_id": { "type": "string", "description": "Scope to search within (defaults to workspace scope if omitted)" },
                    "limit": { "type": "integer", "description": "Maximum sources (default 10)" },
                    "expand_graph": { "type": "boolean", "description": "Follow graph edges (default true)" },
                    "content_role": { "type": "string", "description": "Filter by content role: design, decision, plan, code, memory, finding, instruction, learning" },
                    "source_origin": { "type": "string", "description": "Filter by source origin: repo:<name>, workspace, memory" },
                    "workstream": { "type": "string", "description": "Filter by workstream (e.g. git branch name)" },
                    "include_cold": { "type": "boolean", "description": "Include Cold-tier entries via brute-force cosine scan (default false)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "corvia_system_status",
            "description": "Get a point-in-time system status snapshot (entry count, active agents, open sessions, merge queue depth, tier distribution)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": { "type": "string", "description": "Scope for entry count (defaults to workspace scope if omitted)" }
                }
            }
        }),
        json!({
            "name": "corvia_config_get",
            "description": "Get the current server configuration, optionally filtered to a single section",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Config section to retrieve. Valid sections: storage, server, embedding, project, telemetry (restart-required), agent_lifecycle, merge, rag, chunking, reasoning, adapters, dashboard (hot-reloadable). Omit for full config." }
                }
            }
        }),
        json!({
            "name": "corvia_adapters_list",
            "description": "List all discovered adapter binaries available on the system",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "corvia_agents_list",
            "description": "List all registered agents in the system",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "corvia_merge_queue",
            "description": "Get current merge queue status (depth and entries)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Maximum queue entries to return (default 50)" }
                }
            }
        }),
        json!({
            "name": "corvia_pin",
            "description": "Pin a knowledge entry to prevent it from being demoted or forgotten by GC. Pinned entries remain in their current tier indefinitely, exempt from all GC scoring and forgetting policies. Requires agent identity via agent_id parameter or _meta.agent_id.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": { "type": "string", "description": "The entry UUID to pin" },
                    "agent_id": { "type": "string", "description": "Agent identity for audit trail (e.g. 'claude-code'). Falls back to _meta.agent_id if omitted." }
                },
                "required": ["entry_id"]
            }
        }),
        json!({
            "name": "corvia_inspect",
            "description": "Inspect a knowledge entry's full lifecycle metadata: tier, retention score, memory type, access history, pin state, and graph connectivity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": { "type": "string", "description": "The entry UUID to inspect" }
                },
                "required": ["entry_id"]
            }
        }),
        // --- Tier 2 (LowRisk) tools — require _meta.confirmed ---
        json!({
            "name": "corvia_unpin",
            "description": "Unpin a knowledge entry, making it eligible for GC demotion again. Requires confirmation via _meta.confirmed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_id": { "type": "string", "description": "The entry UUID to unpin" }
                },
                "required": ["entry_id"]
            },
            "annotations": { "tier": "LowRisk" }
        }),
        json!({
            "name": "corvia_config_set",
            "description": "Update a configuration value. Requires confirmation via _meta.confirmed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Config section (hot-reloadable only: agent_lifecycle, merge, rag, chunking, reasoning, adapters, dashboard). Restart-required sections (storage, server, embedding, project, telemetry) are rejected." },
                    "key": { "type": "string", "description": "Config key within the section" },
                    "value": { "description": "New value to set" }
                },
                "required": ["section", "key", "value"]
            },
            "annotations": { "tier": "LowRisk" }
        }),
        json!({
            "name": "corvia_gc_run",
            "description": "Run garbage collection to clean up orphaned staging branches. Requires confirmation via _meta.confirmed.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            },
            "annotations": { "tier": "LowRisk" }
        }),
        json!({
            "name": "corvia_rebuild_index",
            "description": "Rebuild the HNSW vector index from knowledge files on disk. Requires confirmation via _meta.confirmed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dimensions": { "type": "integer", "description": "Vector dimensions (defaults from config)" }
                }
            },
            "annotations": { "tier": "LowRisk" }
        }),
        // --- Tier 3 (MediumRisk) tools — require _meta.confirmed + support dry_run ---
        json!({
            "name": "corvia_agent_suspend",
            "description": "Suspend an agent, preventing it from writing new entries. Requires confirmation via _meta.confirmed. Supports dry_run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "The agent ID to suspend" },
                    "dry_run": { "type": "boolean", "description": "If true, preview the action without executing" }
                },
                "required": ["agent_id"]
            },
            "annotations": { "tier": "MediumRisk" }
        }),
        json!({
            "name": "corvia_merge_retry",
            "description": "Retry failed merge queue entries. Requires confirmation via _meta.confirmed. Supports dry_run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "UUIDs of merge queue entries to retry"
                    },
                    "dry_run": { "type": "boolean", "description": "If true, preview the action without executing" }
                },
                "required": ["entry_ids"]
            },
            "annotations": { "tier": "MediumRisk" }
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
        .layer(TraceLayer::new_for_http())
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

    // Extract bearer token for write operation auth
    let bearer_token = extract_bearer_token(&headers);

    // Process request(s)
    let (responses, is_batch) = match input {
        JsonRpcInput::Single(req) => {
            let resp = process_single_request(&state, &req, bearer_token.as_deref()).await;
            (resp.into_iter().collect::<Vec<_>>(), false)
        }
        JsonRpcInput::Batch(reqs) => {
            let mut responses = Vec::new();
            for req in &reqs {
                if let Some(resp) = process_single_request(&state, req, bearer_token.as_deref()).await {
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
    bearer_token: Option<&str>,
) -> Option<JsonRpcResponse> {
    let is_notification = req.id.is_none() && req.method != "initialize";

    if is_notification {
        // Fire-and-forget: dispatch but don't return a response
        let _ = dispatch(&state.app, req, bearer_token).await;
        return None;
    }

    let response = dispatch(&state.app, req, bearer_token).await;
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
    bearer_token: Option<&str>,
) -> Result<Value, (i32, String)> {
    match req.method.as_str() {
        "initialize" => handle_initialize(&req.params),
        "notifications/initialized" => Ok(json!({})),
        "tools/list" => Ok(handle_tools_list()),
        "tools/call" => handle_tools_call(state, &req.params, bearer_token).await,
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
    bearer_token: Option<&str>,
) -> Result<Value, (i32, String)> {
    let tool_name = params.get("name")
        .and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing tool name".into()))?;

    // Bearer token auth: when the server has an MCP token configured (non-loopback
    // binding), write operations require a matching Authorization: Bearer <token> header.
    // Uses constant-time comparison to prevent timing side-channel attacks.
    if let Some(expected_token) = &state.mcp_token
        && WRITE_TOOLS.contains(&tool_name)
    {
        match bearer_token {
            Some(provided) if constant_time_eq(provided.as_bytes(), expected_token.as_bytes()) => {}
            Some(_) => {
                return Err((UNAUTHORIZED, format!(
                    "Invalid bearer token for write operation '{}'", tool_name
                )));
            }
            None => {
                return Err((UNAUTHORIZED, format!(
                    "Authorization required for write operation '{}'", tool_name
                )));
            }
        }
    }

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
        "corvia_system_status" => tool_corvia_system_status(state, &arguments).await,
        "corvia_config_get" => tool_corvia_config_get(state, &arguments),
        "corvia_adapters_list" => tool_corvia_adapters_list(state),
        "corvia_agents_list" => tool_corvia_agents_list(state),
        "corvia_merge_queue" => tool_corvia_merge_queue(state, &arguments),
        "corvia_pin" => tool_corvia_pin(state, &arguments, agent_id).await,
        "corvia_inspect" => tool_corvia_inspect(state, &arguments).await,
        // Tier 2 (LowRisk) — require _meta.confirmed
        "corvia_unpin" => tool_corvia_unpin(state, &arguments, meta).await,
        "corvia_config_set" => tool_corvia_config_set(state, &arguments, meta).await,
        "corvia_gc_run" => tool_corvia_gc_run(state, meta).await,
        "corvia_rebuild_index" => tool_corvia_rebuild_index(state, &arguments, meta).await,
        // Tier 3 (MediumRisk) — require _meta.confirmed + support dry_run
        "corvia_agent_suspend" => tool_corvia_agent_suspend(state, &arguments, meta).await,
        "corvia_merge_retry" => tool_corvia_merge_retry(state, &arguments, meta).await,
        other => Err((METHOD_NOT_FOUND, format!("Unknown tool: {other}"))),
    }
}

// --- Tool implementations ---

/// Resolve scope_id from args, falling back to the workspace default.
fn resolve_scope_id<'a>(args: &'a Value, state: &'a AppState) -> Result<&'a str, (i32, String)> {
    if let Some(scope) = args.get("scope_id").and_then(|v| v.as_str()) {
        return Ok(scope);
    }
    if let Some(ref default) = state.default_scope_id {
        // SAFETY: AppState lives for the duration of the request, so this is fine.
        // We need to return a &str with lifetime 'a tied to state.
        return Ok(default.as_str());
    }
    Err((INVALID_PARAMS, "Missing 'scope_id' parameter and no workspace default configured".into()))
}

async fn tool_corvia_search(
    state: &AppState,
    args: &Value,
    _agent_id: Option<&str>,
) -> Result<Value, (i32, String)> {
    let query = args.get("query").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'query' parameter".into()))?;
    let scope_id = resolve_scope_id(args, state)?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let content_role = args.get("content_role").and_then(|v| v.as_str()).map(String::from);
    let source_origin = args.get("source_origin").and_then(|v| v.as_str()).map(String::from);
    let workstream = args.get("workstream").and_then(|v| v.as_str()).map(String::from);
    let include_cold = args.get("include_cold").and_then(|v| v.as_bool()).unwrap_or(false);
    let min_score = args.get("min_score").and_then(|v| v.as_f64()).map(|v| (v as f32).clamp(0.0, 1.0));

    // Route through RAG pipeline if available (fixes ContextBuilder bypass)
    if let Some(rag) = state.rag_pipeline() {
        let opts = corvia_kernel::rag_types::RetrievalOpts {
            limit,
            expand_graph: false, // search endpoint: pure vector (context/ask use graph)
            content_role: content_role.clone(),
            source_origin: source_origin.clone(),
            workstream: workstream.clone(),
            include_cold,
            ..Default::default()
        };
        let response = rag.context(query, scope_id, Some(opts)).await
            .map_err(|e| (INTERNAL_ERROR, format!("Search failed: {e}")))?;

        let all_sources = &response.context.sources;

        // Compute quality signal from ALL results (pre-filter) to reflect true retrieval quality.
        let mut quality_signal = corvia_kernel::rag_types::QualitySignal::from_results(
            all_sources, query, limit,
        );

        // Apply min_score filter if requested.
        let (filtered_sources, below_threshold_count) = if let Some(threshold) = min_score {
            let before = all_sources.len();
            let filtered: Vec<_> = all_sources.iter().filter(|r| r.score >= threshold).collect();
            let below = before - filtered.len();
            (filtered, below)
        } else {
            (all_sources.iter().collect(), 0usize)
        };
        quality_signal.below_threshold_count = below_threshold_count;

        let items: Vec<Value> = filtered_sources.iter().map(|r| {
            let mut item = json!({
                "content": r.entry.content,
                "score": r.score,
                "source_file": r.entry.metadata.source_file,
                "language": r.entry.metadata.language,
                "content_role": r.entry.metadata.content_role,
                "source_origin": r.entry.metadata.source_origin,
            });
            item["tier"] = json!(r.tier.to_string());
            if let Some(rs) = r.retention_score {
                item["retention_score"] = json!(rs);
            }
            item
        }).collect();

        // Record gap signal if confidence is low.
        if quality_signal.confidence == corvia_kernel::rag_types::ConfidenceLevel::Low {
            state.gap_detector.record(corvia_kernel::gap_detector::GapSignal {
                query: query.to_string(),
                top_score: quality_signal.top_score,
                result_count: quality_signal.result_count,
                timestamp: chrono::Utc::now(),
                scope_id: scope_id.to_string(),
            });
        }

        return Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&json!({
                    "results": items,
                    "count": items.len(),
                    "quality_signal": quality_signal
                })).unwrap()
            }]
        }));
    }

    // Fallback: raw store search
    let embedding = state.engine.embed(query).await
        .map_err(|e| (INTERNAL_ERROR, format!("Embedding failed: {e}")))?;

    let search_limit = if content_role.is_some() || source_origin.is_some() || workstream.is_some() {
        limit * 3
    } else {
        limit
    };
    let results = state.store.search(&embedding, scope_id, search_limit).await
        .map_err(|e| (INTERNAL_ERROR, format!("Search failed: {e}")))?;

    let results = corvia_kernel::retriever::post_filter_metadata(
        results,
        content_role.as_deref(),
        source_origin.as_deref(),
        workstream.as_deref(),
    );
    let results: Vec<_> = results.into_iter().take(limit).collect();

    // Compute quality signal from ALL results (pre-filter).
    let mut quality_signal = corvia_kernel::rag_types::QualitySignal::from_results(
        &results, query, limit,
    );

    // Apply min_score filter.
    let (results, below_threshold_count) = if let Some(threshold) = min_score {
        let before = results.len();
        let filtered: Vec<_> = results.into_iter().filter(|r| r.score >= threshold).collect();
        let below = before - filtered.len();
        (filtered, below)
    } else {
        (results, 0usize)
    };
    quality_signal.below_threshold_count = below_threshold_count;

    let items: Vec<Value> = results.iter().map(|r| {
        let mut item = json!({
            "content": r.entry.content,
            "score": r.score,
            "source_file": r.entry.metadata.source_file,
            "language": r.entry.metadata.language,
            "content_role": r.entry.metadata.content_role,
            "source_origin": r.entry.metadata.source_origin,
        });
        item["tier"] = json!(r.tier.to_string());
        if let Some(rs) = r.retention_score {
            item["retention_score"] = json!(rs);
        }
        item
    }).collect();

    // Record gap signal if confidence is low.
    if quality_signal.confidence == corvia_kernel::rag_types::ConfidenceLevel::Low {
        state.gap_detector.record(corvia_kernel::gap_detector::GapSignal {
            query: query.to_string(),
            top_score: quality_signal.top_score,
            result_count: quality_signal.result_count,
            timestamp: chrono::Utc::now(),
            scope_id: scope_id.to_string(),
        });
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "results": items,
                "count": items.len(),
                "quality_signal": quality_signal
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
    let scope_id = resolve_scope_id(args, state)?;
    let source_version = args.get("source_version").and_then(|v| v.as_str()).unwrap_or("mcp");
    let content_role = args.get("content_role").and_then(|v| v.as_str()).map(String::from);
    let source_origin = args.get("source_origin").and_then(|v| v.as_str()).map(String::from)
        .or(Some("workspace".into())); // default per spec
    let force_write = args.get("force_write").and_then(|v| v.as_bool()).unwrap_or(false);

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
        // If agent already existed, register_agent returned the old record
        // without updating permissions. Grant the target scope so agents
        // that write to multiple scopes don't get permission errors.
        let _ = coord.grant_scope(agent_id, scope_id);
        coord.create_session(agent_id, false)
            .map_err(|e| (INTERNAL_ERROR, format!("Session creation failed: {e}")))?;
        connect = coord.connect(agent_id)
            .map_err(|e| (INTERNAL_ERROR, format!("Agent setup failed: {e}")))?;
    }

    let session_id = connect.active_sessions.first()
        .map(|s| s.session_id.as_str())
        .ok_or((INTERNAL_ERROR, "No active session".into()))?;

    let result = coord.write_entry_with_dedup(
        session_id, content, scope_id, source_version,
        content_role, source_origin, force_write,
    ).await
        .map_err(|e| (INTERNAL_ERROR, format!("Write failed: {e}")))?;

    use corvia_kernel::agent_writer::WriteResult;
    match result {
        WriteResult::Written(entry) => {
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!("Entry {} written (status: {:?})", entry.id, entry.entry_status)
                }]
            }))
        }
        WriteResult::WrittenWithWarning { entry, similar_id, similarity } => {
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!(
                        "Entry {} written (status: {:?}). Warning: similar entry {} exists (similarity: {:.2}). Consider checking for overlap.",
                        entry.id, entry.entry_status, similar_id, similarity
                    )
                }]
            }))
        }
        WriteResult::Blocked { existing_id, similarity, existing_preview } => {
            Ok(json!({
                "isError": true,
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&json!({
                        "status": "blocked",
                        "existing_id": existing_id.to_string(),
                        "similarity": similarity,
                        "existing_preview": existing_preview,
                        "message": "Near-duplicate detected. Entry NOT written. Use force_write: true to bypass."
                    })).unwrap()
                }]
            }))
        }
    }
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

    // Check if any entry in the chain is Forgotten — increment counter for observability
    if chain.iter().any(|e| e.tier == corvia_common::types::Tier::Forgotten) {
        state.forgotten_access_counter.increment();
    }

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

    // Check if the entry is Forgotten — increment counter for observability
    if let Ok(Some(entry)) = state.store.get(&uuid).await
        && entry.tier == corvia_common::types::Tier::Forgotten
    {
        state.forgotten_access_counter.increment();
    }

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
    let scope_id = resolve_scope_id(args, state)?;
    let check_str = args.get("check").and_then(|v| v.as_str());

    // Load entries on a blocking thread to avoid starving the async runtime.
    let data_dir = state.data_dir.clone();
    let scope_owned = scope_id.to_string();
    let entries = tokio::task::spawn_blocking(move || {
        corvia_kernel::knowledge_files::read_scope(&data_dir, &scope_owned)
    })
    .await
    .map_err(|e| (INTERNAL_ERROR, format!("Task join failed: {e}")))?
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
    let scope_id = resolve_scope_id(args, state)?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let expand_graph = args.get("expand_graph").and_then(|v| v.as_bool()).unwrap_or(true);
    let content_role = args.get("content_role").and_then(|v| v.as_str()).map(String::from);
    let source_origin = args.get("source_origin").and_then(|v| v.as_str()).map(String::from);
    let workstream = args.get("workstream").and_then(|v| v.as_str()).map(String::from);
    let include_cold = args.get("include_cold").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v.clamp(100, 4000) as usize);
    let format = args.get("format").and_then(|v| v.as_str()).unwrap_or("default");
    if format != "default" && format != "compact" {
        return Err((INVALID_PARAMS, format!("Unknown format: '{format}'. Must be 'default' or 'compact'.")));
    }

    let rag = state.rag_pipeline()
        .ok_or((SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit,
        expand_graph,
        content_role,
        source_origin,
        workstream,
        include_cold,
        ..Default::default()
    };

    // For compact format with max_tokens, we retrieve first then assemble with custom budget.
    if format == "compact" {
        // Retrieve with a large token budget to avoid premature truncation by the default
        // augmenter. The compact augmenter will enforce the actual budget.
        let compact_budget = max_tokens.unwrap_or(4000);
        let response = rag.context_with_token_override(query, scope_id, Some(opts), compact_budget * 2).await
            .map_err(|e| (INTERNAL_ERROR, format!("RAG failed: {e}")))?;

        let budget = corvia_kernel::rag_types::TokenBudget {
            max_context_tokens: max_tokens.or(Some(4000)),
            reserve_for_answer: 0.0, // No answer generation in compact mode.
            ..Default::default()
        };

        let compact = corvia_kernel::augmenter::augment_compact(
            &response.context.sources, &budget,
        ).map_err(|e| (INTERNAL_ERROR, format!("Compact augmentation failed: {e}")))?;

        let sources: Vec<Value> = compact.sources.iter().map(|r| {
            json!({
                "entry_id": r.entry.id.to_string(),
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
                    "context": compact.context,
                    "sources": sources,
                    "token_estimate": compact.metrics.token_estimate,
                    "token_budget": compact.metrics.token_budget,
                    "sources_included": compact.metrics.sources_included,
                    "sources_truncated": compact.metrics.sources_truncated,
                })).unwrap()
            }]
        }));
    }

    // Default format: use standard RAG pipeline context assembly.
    // If max_tokens is specified, override the pipeline's token budget.
    let response = if let Some(tokens) = max_tokens {
        rag.context_with_token_override(query, scope_id, Some(opts), tokens).await
            .map_err(|e| (INTERNAL_ERROR, format!("RAG failed: {e}")))?
    } else {
        rag.context(query, scope_id, Some(opts)).await
            .map_err(|e| (INTERNAL_ERROR, format!("RAG failed: {e}")))?
    };

    let sources: Vec<Value> = response.context.sources.iter().map(|r| {
        json!({
            "entry_id": r.entry.id.to_string(),
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
    let scope_id = resolve_scope_id(args, state)?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let expand_graph = args.get("expand_graph").and_then(|v| v.as_bool()).unwrap_or(true);
    let content_role = args.get("content_role").and_then(|v| v.as_str()).map(String::from);
    let source_origin = args.get("source_origin").and_then(|v| v.as_str()).map(String::from);
    let workstream = args.get("workstream").and_then(|v| v.as_str()).map(String::from);
    let include_cold = args.get("include_cold").and_then(|v| v.as_bool()).unwrap_or(false);

    let rag = state.rag_pipeline()
        .ok_or((SERVICE_UNAVAILABLE, "RAG pipeline not configured".into()))?;

    let opts = corvia_kernel::rag_types::RetrievalOpts {
        limit,
        expand_graph,
        content_role,
        source_origin,
        workstream,
        include_cold,
        ..Default::default()
    };

    // Timeout prevents CPU-bound LLM inference from blocking indefinitely.
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(corvia_common::constants::RAG_ASK_TIMEOUT_SECS),
        rag.ask(query, scope_id, Some(opts)),
    ).await
        .map_err(|_| (INTERNAL_ERROR, format!(
            "RAG ask timed out after {}s — model may be running on CPU. Check inference logs.",
            corvia_common::constants::RAG_ASK_TIMEOUT_SECS
        )))?
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

// --- Tier 1 (ReadOnly) control-plane tool implementations ---

async fn tool_corvia_system_status(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let scope_id = resolve_scope_id(args, state)?;
    let status = corvia_kernel::ops::system_status(
        state.store.clone(),
        &state.coordinator,
        scope_id,
    ).await.map_err(|e| (INTERNAL_ERROR, format!("System status failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "entry_count": status.entry_count,
                "active_agents": status.active_agents,
                "open_sessions": status.open_sessions,
                "merge_queue_depth": status.merge_queue_depth,
                "scope_id": status.scope_id,
                "tier_distribution": {
                    "hot": status.tier_distribution.hot,
                    "warm": status.tier_distribution.warm,
                    "cold": status.tier_distribution.cold,
                    "forgotten": status.tier_distribution.forgotten,
                },
            })).unwrap()
        }]
    }))
}

fn tool_corvia_config_get(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let section = args.get("section").and_then(|v| v.as_str());
    let config = state.config.read()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock poisoned: {e}")))?;
    let result = corvia_kernel::ops::config_get(&config, section)
        .map_err(|e| (INVALID_PARAMS, format!("{e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&result).unwrap()
        }]
    }))
}

fn tool_corvia_adapters_list(
    state: &AppState,
) -> Result<Value, (i32, String)> {
    let search_dirs: Vec<String> = {
        let config = state.config.read()
            .map_err(|e| (INTERNAL_ERROR, format!("Config lock poisoned: {e}")))?;
        config.adapters.as_ref()
            .map(|a| a.search_dirs.clone())
            .unwrap_or_default()
    };
    let adapters = corvia_kernel::ops::adapters_list(&search_dirs);
    let items: Vec<Value> = adapters.iter().map(|a| {
        json!({
            "binary_path": a.binary_path.to_string_lossy(),
            "name": a.metadata.name,
            "version": a.metadata.version,
            "domain": a.metadata.domain,
            "description": a.metadata.description,
            "supported_extensions": a.metadata.supported_extensions,
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "adapters": items,
                "count": items.len()
            })).unwrap()
        }]
    }))
}

fn tool_corvia_agents_list(
    state: &AppState,
) -> Result<Value, (i32, String)> {
    let agents = corvia_kernel::ops::agents_list(&state.coordinator)
        .map_err(|e| (INTERNAL_ERROR, format!("Agents list failed: {e}")))?;
    let items: Vec<Value> = agents.iter().map(|a| {
        json!({
            "agent_id": a.agent_id,
            "status": format!("{:?}", a.status),
            "registered_at": a.registered_at.to_rfc3339(),
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "agents": items,
                "count": items.len()
            })).unwrap()
        }]
    }))
}

fn tool_corvia_merge_queue(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let status = corvia_kernel::ops::merge_queue_status(&state.coordinator, limit)
        .map_err(|e| (INTERNAL_ERROR, format!("Merge queue status failed: {e}")))?;
    let items: Vec<Value> = status.entries.iter().map(|e| {
        json!({
            "entry_id": e.entry_id.to_string(),
            "session_id": e.session_id,
            "enqueued_at": e.enqueued_at.to_rfc3339(),
        })
    }).collect();

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&json!({
                "depth": status.depth,
                "entries": items,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_pin(
    state: &AppState,
    args: &Value,
    agent_id: Option<&str>,
) -> Result<Value, (i32, String)> {
    let entry_id_str = args.get("entry_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'entry_id' parameter".into()))?;
    let entry_id = uuid::Uuid::parse_str(entry_id_str)
        .map_err(|e| (INVALID_PARAMS, format!("Invalid UUID: {e}")))?;

    let agent = agent_id
        .ok_or((INVALID_PARAMS, "Pin requires agent identity (_meta.agent_id or agent_id param)".into()))?;

    let result = corvia_kernel::ops::pin_entry(&state.store, &entry_id, agent).await
        .map_err(|e| (INTERNAL_ERROR, format!("Pin failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "status": "pinned",
                "entry_id": result.entry_id,
                "pinned_by": result.pinned_by,
                "pinned_at": result.pinned_at,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_inspect(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let entry_id_str = args.get("entry_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'entry_id' parameter".into()))?;
    let entry_id = uuid::Uuid::parse_str(entry_id_str)
        .map_err(|e| (INVALID_PARAMS, format!("Invalid UUID: {e}")))?;

    let inspection = corvia_kernel::ops::inspect_entry(&state.store, &state.graph, &entry_id).await
        .map_err(|e| (INTERNAL_ERROR, format!("Inspect failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(
                &serde_json::to_value(&inspection).map_err(|e| (INTERNAL_ERROR, format!("Serialize failed: {e}")))?
            ).unwrap()
        }]
    }))
}

// --- Tier 2 (LowRisk) control-plane tool implementations ---

async fn tool_corvia_unpin(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let entry_id_str = args.get("entry_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'entry_id' parameter".into()))?;
    let entry_id = uuid::Uuid::parse_str(entry_id_str)
        .map_err(|e| (INVALID_PARAMS, format!("Invalid UUID: {e}")))?;

    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "entry_id": entry_id_str }),
            &format!("Unpin entry {entry_id_str}? This makes it eligible for GC demotion."),
        ));
    }

    let result = corvia_kernel::ops::unpin_entry(&state.store, &entry_id).await
        .map_err(|e| (INTERNAL_ERROR, format!("Unpin failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "status": "unpinned",
                "entry_id": result.entry_id,
                "was_pinned": result.was_pinned,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_config_set(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let section = args.get("section").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'section' parameter".into()))?;
    let key = args.get("key").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'key' parameter".into()))?;
    let value = args.get("value")
        .ok_or((INVALID_PARAMS, "Missing 'value' parameter".into()))?;

    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "section": section, "key": key, "new_value": value }),
            &format!("Update config {section}.{key}?"),
        ));
    }

    let mut config = state.config.write()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock poisoned: {e}")))?;
    corvia_kernel::ops::config_set(&state.config_path, &mut config, section, key, value.clone())
        .map_err(|e| (INTERNAL_ERROR, format!("Config set failed: {e}")))?;

    // Hot-swap pipeline when rag.pipeline config changes.
    // Clone config and drop the write lock before rebuilding to avoid blocking
    // concurrent config reads during I/O-heavy pipeline construction.
    let config_snapshot = config.clone();
    drop(config);

    let mut pipeline_swapped = false;
    let mut pipeline_error: Option<String> = None;
    let is_pipeline_key = section == "rag"
        && (key.starts_with("pipeline.") || key == "graph_alpha" || key == "system_prompt");
    if is_pipeline_key && let Some(ref rag_swap) = state.rag {
        // Preserve the current generator so ask() mode continues working.
        let current_generator = rag_swap.load().generator().cloned();
        let graph = Some(state.graph.clone());
        match corvia_kernel::rebuild_pipeline_retriever(
            state.store.clone(),
            state.engine.clone(),
            graph,
            current_generator,
            &config_snapshot,
        ) {
            Ok(new_pipeline) => {
                let old = rag_swap.swap(std::sync::Arc::new(new_pipeline));
                // Defer deallocation of old pipeline off the async runtime.
                std::thread::spawn(move || drop(old));
                pipeline_swapped = true;
                tracing::info!(section, key, "RAG pipeline hot-swapped successfully");
            }
            Err(e) => {
                tracing::warn!(section, key, error = %e, "Pipeline rebuild failed, keeping current pipeline");
                pipeline_error = Some(format!("{e}"));
            }
        }
    }

    let mut resp = json!({
        "status": "updated",
        "section": section,
        "key": key,
    });
    if pipeline_swapped {
        resp["pipeline_swapped"] = json!(true);
    }
    if let Some(ref err) = pipeline_error {
        resp["pipeline_swapped"] = json!(false);
        resp["pipeline_error"] = json!(err);
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&resp).unwrap()
        }]
    }))
}

async fn tool_corvia_gc_run(
    state: &AppState,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "action": "garbage_collection" }),
            "Run garbage collection? This will clean up orphaned staging branches.",
        ));
    }

    let report = corvia_kernel::ops::gc_run(&state.coordinator).await
        .map_err(|e| (INTERNAL_ERROR, format!("GC failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "orphans_rolled_back": report.orphans_rolled_back,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_rebuild_index(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let dimensions = args.get("dimensions").and_then(|v| v.as_u64()).map(|d| d as usize)
        .unwrap_or_else(|| {
            state.config.read()
                .map(|c| c.embedding.dimensions)
                .unwrap_or(768)
        });

    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "data_dir": state.data_dir.to_string_lossy(), "dimensions": dimensions }),
            "Rebuild HNSW index from knowledge files on disk?",
        ));
    }

    let lite_store = state.store.as_any()
        .downcast_ref::<corvia_kernel::lite_store::LiteStore>()
        .ok_or_else(|| (INTERNAL_ERROR, "Rebuild index is only supported for LiteStore".to_string()))?;

    let entries_indexed = corvia_kernel::ops::rebuild_index(lite_store)
        .map_err(|e| (INTERNAL_ERROR, format!("Rebuild index failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "status": "rebuilt",
                "entries_indexed": entries_indexed,
            })).unwrap()
        }]
    }))
}

// --- Tier 3 (MediumRisk) control-plane tool implementations ---

async fn tool_corvia_agent_suspend(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let agent_id = args.get("agent_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'agent_id' parameter".into()))?;

    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "agent_id": agent_id }),
            &format!("Suspend agent '{agent_id}'? This will prevent it from writing new entries."),
        ));
    }

    if is_dry_run(args) {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&json!({
                    "dry_run": true,
                    "would_suspend": agent_id,
                })).unwrap()
            }]
        }));
    }

    corvia_kernel::ops::agent_suspend(&state.coordinator, agent_id)
        .map_err(|e| (INTERNAL_ERROR, format!("Agent suspend failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "status": "suspended",
                "agent_id": agent_id,
            })).unwrap()
        }]
    }))
}

async fn tool_corvia_merge_retry(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let entry_ids_raw = args.get("entry_ids").and_then(|v| v.as_array())
        .ok_or((INVALID_PARAMS, "Missing 'entry_ids' parameter (array of UUID strings)".into()))?;

    let entry_ids: Vec<uuid::Uuid> = entry_ids_raw.iter()
        .map(|v| {
            let s = v.as_str().ok_or((INVALID_PARAMS, "entry_ids must be strings".into()))?;
            uuid::Uuid::parse_str(s).map_err(|e| (INVALID_PARAMS, format!("Invalid UUID '{s}': {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !is_confirmed(meta) {
        return Ok(confirmation_response(
            json!({ "entry_ids": entry_ids_raw, "count": entry_ids.len() }),
            &format!("Retry {} failed merge queue entries?", entry_ids.len()),
        ));
    }

    if is_dry_run(args) {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&json!({
                    "dry_run": true,
                    "would_retry": entry_ids.len(),
                })).unwrap()
            }]
        }));
    }

    let count = corvia_kernel::ops::merge_retry(&state.coordinator, &entry_ids)
        .map_err(|e| (INTERNAL_ERROR, format!("Merge retry failed: {e}")))?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "status": "retried",
                "count": count,
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
            default_scope_id: None,
            config: Arc::new(std::sync::RwLock::new(corvia_common::config::CorviaConfig::default())),
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
            gpu_cache: std::sync::Arc::new(tokio::sync::Mutex::new(crate::dashboard::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: std::sync::Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: std::sync::Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
            mcp_token: None,
            docker_available: false,
            gap_detector: Arc::new(corvia_kernel::gap_detector::GapDetector::new()),
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
        assert_eq!(tools.len(), 21);

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
        assert!(names.contains(&"corvia_system_status"));
        assert!(names.contains(&"corvia_config_get"));
        assert!(names.contains(&"corvia_adapters_list"));
        assert!(names.contains(&"corvia_agents_list"));
        assert!(names.contains(&"corvia_merge_queue"));
        assert!(names.contains(&"corvia_pin"));
        assert!(names.contains(&"corvia_inspect"));
        // Tier 2
        assert!(names.contains(&"corvia_unpin"));
        assert!(names.contains(&"corvia_config_set"));
        assert!(names.contains(&"corvia_gc_run"));
        assert!(names.contains(&"corvia_rebuild_index"));
        // Tier 3
        assert!(names.contains(&"corvia_agent_suspend"));
        assert!(names.contains(&"corvia_merge_retry"));
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

        // Insert entries with no graph edges — they should be flagged as orphaned.
        // Must have a code language to be eligible for the orphan check.
        let mut e1 = corvia_common::types::KnowledgeEntry::new(
            "orphaned knowledge".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        e1.metadata.language = Some("rs".into());
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
            entry.embedding = Some(vec![1.0, (i as f32) * 0.1, 0.0]);
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
        ).await);

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
            rag: Some(Arc::new(arc_swap::ArcSwap::new(rag))),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            default_scope_id: None,
            config: Arc::new(std::sync::RwLock::new(corvia_common::config::CorviaConfig::default())),
            config_path: dir.path().join("corvia.toml"),
            cluster_store: Arc::new(crate::dashboard::clustering::ClusterStore::new()),
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
            session_ingest_lock: tokio::sync::Mutex::new(()),
            hook_sessions: crate::dashboard::session_watcher::SessionWatcherState::new().0,
            coverage_cache: Arc::new(
                crate::dashboard::coverage::IndexCoverageCache::new(0.9, 60),
            ),
            workspace_root: dir.path().to_path_buf(),
            ingest_status: Arc::new(std::sync::RwLock::new(corvia_kernel::ingest::IngestStatus::idle())),
            gpu_cache: std::sync::Arc::new(tokio::sync::Mutex::new(crate::dashboard::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: std::sync::Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: std::sync::Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
            mcp_token: None,
            docker_available: false,
            gap_detector: Arc::new(corvia_kernel::gap_detector::GapDetector::new()),
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
    async fn test_corvia_system_status() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_system_status",
            "arguments": { "scope_id": "test" }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["entry_count"], 0);
    }

    #[tokio::test]
    async fn test_corvia_agents_list() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_agents_list",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(parsed["agents"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_corvia_merge_queue() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_merge_queue",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["depth"], 0);
    }

    #[tokio::test]
    async fn test_corvia_config_get() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_config_get",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(parsed.get("storage").is_some());
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
        let result = dispatch(&state, &req, None).await;
        assert!(result.is_err());
        let (code, _) = result.unwrap_err();
        assert_eq!(code, METHOD_NOT_FOUND);
    }

    // --- Tier 2 confirmation tests ---

    #[tokio::test]
    async fn test_config_set_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Call without _meta.confirmed — should get confirmation_required
        let params = json!({
            "name": "corvia_config_set",
            "arguments": { "section": "rag", "key": "default_limit", "value": 20 }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
        assert_eq!(parsed["preview"]["section"], "rag");
        assert_eq!(parsed["preview"]["key"], "default_limit");
    }

    #[tokio::test]
    async fn test_config_set_with_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Write a full valid TOML config so config_set can read/write/re-parse it
        let default_config = corvia_common::config::CorviaConfig::default();
        let config_toml = toml::to_string_pretty(&default_config).unwrap();
        std::fs::write(state.config_path.clone(), &config_toml).unwrap();

        // Use a hot-reloadable section (rag), not a restart-required one (storage)
        let params = json!({
            "name": "corvia_config_set",
            "_meta": { "confirmed": true },
            "arguments": { "section": "rag", "key": "default_limit", "value": 20 }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["status"], "updated");
        assert_eq!(parsed["section"], "rag");
        assert_eq!(parsed["key"], "default_limit");
    }

    #[tokio::test]
    async fn test_config_set_rejects_restart_required_section() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Write config file
        let default_config = corvia_common::config::CorviaConfig::default();
        let config_toml = toml::to_string_pretty(&default_config).unwrap();
        std::fs::write(state.config_path.clone(), &config_toml).unwrap();

        // Try to set a restart-required section with confirmation
        let params = json!({
            "name": "corvia_config_set",
            "_meta": { "confirmed": true },
            "arguments": { "section": "storage", "key": "data_dir", "value": "/new" }
        });
        let result = handle_tools_call(&state, &params, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gc_run_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_gc_run",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
        assert_eq!(parsed["preview"]["action"], "garbage_collection");
    }

    #[tokio::test]
    async fn test_gc_run_with_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_gc_run",
            "_meta": { "confirmed": true },
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["orphans_rolled_back"], 0);
    }

    #[tokio::test]
    async fn test_rebuild_index_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_rebuild_index",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
        assert!(parsed["preview"]["dimensions"].as_u64().unwrap() > 0);
    }

    // --- Tier 3 confirmation + dry_run tests ---

    #[tokio::test]
    async fn test_agent_suspend_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_agent_suspend",
            "arguments": { "agent_id": "test-agent" }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
        assert_eq!(parsed["preview"]["agent_id"], "test-agent");
    }

    #[tokio::test]
    async fn test_agent_suspend_dry_run() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_agent_suspend",
            "_meta": { "confirmed": true },
            "arguments": { "agent_id": "test-agent", "dry_run": true }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["dry_run"], true);
        assert_eq!(parsed["would_suspend"], "test-agent");
    }

    #[tokio::test]
    async fn test_merge_retry_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let id = uuid::Uuid::now_v7().to_string();
        let params = json!({
            "name": "corvia_merge_retry",
            "arguments": { "entry_ids": [id] }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
        assert_eq!(parsed["preview"]["count"], 1);
    }

    #[tokio::test]
    async fn test_merge_retry_dry_run() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let id = uuid::Uuid::now_v7().to_string();
        let params = json!({
            "name": "corvia_merge_retry",
            "_meta": { "confirmed": true },
            "arguments": { "entry_ids": [id], "dry_run": true }
        });
        let result = handle_tools_call(&state, &params, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["dry_run"], true);
        assert_eq!(parsed["would_retry"], 1);
    }

    #[tokio::test]
    async fn test_corvia_pin() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert a test entry
        let entry = corvia_common::types::KnowledgeEntry::new(
            "pin me".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        let entry_id = entry.id.to_string();
        state.store.insert(&entry).await.unwrap();

        let args = json!({ "entry_id": entry_id, "agent_id": "test-agent" });
        let result = tool_corvia_pin(&state, &args, Some("test-agent")).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["status"], "pinned");
        assert_eq!(parsed["pinned_by"], "test-agent");

        // Verify entry is pinned in store
        let updated = state.store.get(&entry.id).await.unwrap().unwrap();
        assert!(updated.pin.is_some());
    }

    #[tokio::test]
    async fn test_corvia_pin_requires_agent_id() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let args = json!({ "entry_id": uuid::Uuid::now_v7().to_string() });
        let result = tool_corvia_pin(&state, &args, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_corvia_unpin_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        let entry = corvia_common::types::KnowledgeEntry::new(
            "unpin me".into(), "test-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        let entry_id = entry.id.to_string();
        state.store.insert(&entry).await.unwrap();

        // Without confirmation — should return confirmation_required
        let args = json!({ "entry_id": entry_id });
        let result = tool_corvia_unpin(&state, &args, None).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
    }

    #[tokio::test]
    async fn test_corvia_unpin_with_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert a pinned entry
        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "unpin me".into(), "test-scope".into(), "v1".into(),
        );
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        entry.pin = Some(corvia_common::types::PinInfo {
            by: "agent".into(),
            at: chrono::Utc::now(),
        });
        let entry_id = entry.id.to_string();
        state.store.insert(&entry).await.unwrap();

        // With confirmation — should unpin
        let args = json!({ "entry_id": entry_id });
        let meta = json!({ "confirmed": true });
        let result = tool_corvia_unpin(&state, &args, Some(&meta)).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["status"], "unpinned");
        assert_eq!(parsed["was_pinned"], true);

        // Verify unpinned in store
        let updated = state.store.get(&entry.id).await.unwrap().unwrap();
        assert!(updated.pin.is_none());
    }

    #[tokio::test]
    async fn test_system_status_includes_tier_distribution() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;

        // Insert entries with different tiers
        let mut hot_entry = corvia_common::types::KnowledgeEntry::new(
            "hot".into(), "test-scope".into(), "v1".into(),
        );
        hot_entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        hot_entry.tier = corvia_common::types::Tier::Hot;
        state.store.insert(&hot_entry).await.unwrap();

        let mut cold_entry = corvia_common::types::KnowledgeEntry::new(
            "cold".into(), "test-scope".into(), "v1".into(),
        );
        cold_entry.embedding = Some(vec![0.0, 1.0, 0.0]);
        cold_entry.tier = corvia_common::types::Tier::Cold;
        state.store.insert(&cold_entry).await.unwrap();

        let args = json!({ "scope_id": "test-scope" });
        let result = tool_corvia_system_status(&state, &args).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();

        assert!(parsed.get("tier_distribution").is_some());
        let dist = &parsed["tier_distribution"];
        assert!(dist["hot"].as_u64().unwrap() >= 1);
        assert!(dist["cold"].as_u64().unwrap() >= 1);
    }

    // --- MCP bearer token auth tests ---

    /// Build a test AppState with an MCP token configured (simulating non-loopback binding).
    async fn test_state_with_token(dir: &std::path::Path, token: &str) -> Arc<AppState> {
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
            default_scope_id: None,
            config: Arc::new(std::sync::RwLock::new(corvia_common::config::CorviaConfig::default())),
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
            gpu_cache: std::sync::Arc::new(tokio::sync::Mutex::new(crate::dashboard::gpu::GpuMetricsCache::new())),
            forgotten_access_counter: std::sync::Arc::new(corvia_kernel::gc_worker::ForgottenAccessCounter::new()),
            gc_knowledge_history: std::sync::Arc::new(corvia_kernel::ops::GcKnowledgeHistory::new(10)),
            mcp_token: Some(token.to_string()),
            docker_available: false,
            gap_detector: Arc::new(corvia_kernel::gap_detector::GapDetector::new()),
        })
    }

    #[tokio::test]
    async fn test_write_tool_rejected_without_token() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_token(dir.path(), "test-secret-token").await;
        let params = json!({
            "name": "corvia_write",
            "arguments": {
                "scope_id": "test",
                "content": "test content",
                "agent_id": "test-agent"
            }
        });
        // No bearer token provided -> should be rejected
        let result = handle_tools_call(&state, &params, None).await;
        assert!(result.is_err());
        let (code, msg) = result.unwrap_err();
        assert_eq!(code, UNAUTHORIZED);
        assert!(msg.contains("Authorization required"), "Expected auth error, got: {msg}");
    }

    #[tokio::test]
    async fn test_write_tool_rejected_with_wrong_token() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_token(dir.path(), "test-secret-token").await;
        let params = json!({
            "name": "corvia_write",
            "arguments": {
                "scope_id": "test",
                "content": "test content",
                "agent_id": "test-agent"
            }
        });
        // Wrong bearer token -> should be rejected
        let result = handle_tools_call(&state, &params, Some("wrong-token")).await;
        assert!(result.is_err());
        let (code, msg) = result.unwrap_err();
        assert_eq!(code, UNAUTHORIZED);
        assert!(msg.contains("Invalid bearer token"), "Expected invalid token error, got: {msg}");
    }

    #[tokio::test]
    async fn test_write_tool_accepted_with_correct_token() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_token(dir.path(), "test-secret-token").await;
        let params = json!({
            "name": "corvia_write",
            "arguments": {
                "scope_id": "test",
                "content": "test content",
                "agent_id": "test-agent"
            }
        });
        // Correct bearer token -> should succeed
        let result = handle_tools_call(&state, &params, Some("test-secret-token")).await;
        assert!(result.is_ok(), "Expected success with correct token, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_read_tool_allowed_without_token() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_token(dir.path(), "test-secret-token").await;
        let params = json!({
            "name": "corvia_system_status",
            "arguments": { "scope_id": "test" }
        });
        // Read tools should work without any token even when mcp_token is configured
        let result = handle_tools_call(&state, &params, None).await;
        assert!(result.is_ok(), "Read tools should not require auth, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_no_token_required_on_loopback() {
        let dir = tempfile::tempdir().unwrap();
        // Default test_state has mcp_token: None (loopback binding)
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_write",
            "arguments": {
                "scope_id": "test",
                "content": "test content",
                "agent_id": "test-agent"
            }
        });
        // No token configured -> write should succeed without auth
        let result = handle_tools_call(&state, &params, None).await;
        assert!(result.is_ok(), "Loopback binding should not require auth, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_extract_bearer_token() {
        let mut headers = HeaderMap::new();
        assert!(extract_bearer_token(&headers).is_none());

        headers.insert("authorization", "Bearer my-token".parse().unwrap());
        assert_eq!(extract_bearer_token(&headers), Some("my-token".to_string()));

        headers.insert("authorization", "Basic other".parse().unwrap());
        assert!(extract_bearer_token(&headers).is_none());
    }

    #[tokio::test]
    async fn test_all_write_tools_require_auth() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_token(dir.path(), "secret").await;

        for tool_name in WRITE_TOOLS {
            let params = json!({
                "name": tool_name,
                "arguments": {
                    "scope_id": "test",
                    "content": "test",
                    "agent_id": "test-agent"
                }
            });
            let result = handle_tools_call(&state, &params, None).await;
            assert!(result.is_err(), "Write tool '{}' should require auth", tool_name);
        }
    }

}
