//! Stdio MCP server exposing corvia_search, corvia_write, corvia_status, and corvia_traces tools.
//!
//! Uses the rmcp crate (JSON-RPC 2.0 over stdin/stdout) to serve the MCP protocol.
//! The Embedder is created once at startup and shared across all tool calls.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rmcp::handler::server::ServerHandler;
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
