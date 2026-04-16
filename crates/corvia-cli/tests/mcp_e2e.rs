//! End-to-end tests for the MCP stdio server.
//!
//! These tests spawn `corvia mcp` as a child process and communicate using
//! the rmcp stdio transport (newline-delimited JSON-RPC 2.0).

use std::io::{BufRead, BufReader, Write as _};
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// MCP stdio transport helpers (NDJSON -- newline-delimited JSON)
// ---------------------------------------------------------------------------

/// Send a JSON-RPC message as a single line terminated by newline.
fn send_message(child: &mut Child, message: &Value) {
    let body = serde_json::to_string(message).expect("failed to serialize message");

    let stdin = child.stdin.as_mut().expect("stdin not captured");
    stdin.write_all(body.as_bytes()).expect("failed to write body");
    stdin.write_all(b"\n").expect("failed to write newline");
    stdin.flush().expect("failed to flush stdin");
}

/// Read a JSON-RPC response line from stdout.
///
/// Returns `None` if the stream ends before a complete line is read.
fn read_message(reader: &mut BufReader<std::process::ChildStdout>) -> Option<Value> {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => return None, // EOF
        Ok(_) => {}
        Err(_) => return None,
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

/// Spawn the `corvia mcp` process in a temporary directory.
fn spawn_mcp(temp_dir: &std::path::Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_corvia"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(temp_dir)
        .spawn()
        .expect("failed to spawn corvia mcp")
}

/// Send the MCP `initialize` request and consume the response.
///
/// This is required before any other MCP requests can be sent.
/// Returns the initialize response for inspection.
fn do_initialize(child: &mut Child, reader: &mut BufReader<std::process::ChildStdout>) -> Value {
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    send_message(child, &init_request);
    let init_response = read_message(reader).expect("should receive initialize response");

    // Send initialized notification (no id, no response expected).
    let initialized_notification = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });
    send_message(child, &initialized_notification);

    init_response
}

// ---------------------------------------------------------------------------
// 1. mcp_tools_list
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model (server loads embedder on startup)
fn mcp_tools_list() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let mut child = spawn_mcp(temp_dir.path());
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout);

    let _init = do_initialize(&mut child, &mut reader);

    // Send tools/list request.
    let list_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });

    send_message(&mut child, &list_request);
    let response = read_message(&mut reader).expect("should receive tools/list response");

    // Extract tools array from result.
    let result = response.get("result").expect("response should have result");
    let tools = result
        .get("tools")
        .expect("result should have tools")
        .as_array()
        .expect("tools should be an array");

    assert_eq!(tools.len(), 4, "should expose exactly 4 tools");

    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
        .collect();

    assert!(
        tool_names.contains(&"corvia_search"),
        "should have corvia_search tool, got: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"corvia_write"),
        "should have corvia_write tool, got: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"corvia_status"),
        "should have corvia_status tool, got: {tool_names:?}"
    );

    // Cleanup: close stdin to signal EOF, then wait.
    drop(child.stdin.take());
    let _ = child.wait();
}

// ---------------------------------------------------------------------------
// 2. mcp_status_cold_start
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model (server loads embedder on startup)
fn mcp_status_cold_start() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let mut child = spawn_mcp(temp_dir.path());
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout);

    let _init = do_initialize(&mut child, &mut reader);

    // Send status tool call.
    let status_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "corvia_status",
            "arguments": {}
        }
    });

    send_message(&mut child, &status_request);
    let response = read_message(&mut reader).expect("should receive status response");

    let result = response.get("result").expect("response should have result");
    let content = result
        .get("content")
        .expect("result should have content")
        .as_array()
        .expect("content should be an array");

    assert!(!content.is_empty(), "content array should not be empty");

    // The first content item should be text containing JSON with entry_count.
    let text = content[0]
        .get("text")
        .expect("content item should have text")
        .as_str()
        .expect("text should be a string");

    let status: Value = serde_json::from_str(text).expect("status text should be valid JSON");
    let entry_count = status
        .get("entry_count")
        .expect("status should have entry_count")
        .as_u64()
        .expect("entry_count should be a number");

    assert_eq!(entry_count, 0, "cold start should have 0 entries");

    // Cleanup.
    drop(child.stdin.take());
    let _ = child.wait();
}

// ---------------------------------------------------------------------------
// 3. mcp_write_and_search_lifecycle
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model (server loads embedder on startup)
fn mcp_write_and_search_lifecycle() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let mut child = spawn_mcp(temp_dir.path());
    let stdout = child.stdout.take().expect("stdout not captured");
    let mut reader = BufReader::new(stdout);

    let _init = do_initialize(&mut child, &mut reader);

    // Write an entry via MCP.
    let write_request = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "corvia_write",
            "arguments": {
                "content": "Rust ownership model ensures memory safety at compile time without a garbage collector",
                "kind": "learning",
                "tags": ["rust", "memory-safety"]
            }
        }
    });

    send_message(&mut child, &write_request);
    let write_response = read_message(&mut reader).expect("should receive write response");

    let write_result = write_response
        .get("result")
        .expect("write response should have result");
    let write_content = write_result
        .get("content")
        .expect("result should have content")
        .as_array()
        .expect("content should be an array");

    assert!(!write_content.is_empty(), "write content should not be empty");
    let write_text = write_content[0]
        .get("text")
        .expect("content item should have text")
        .as_str()
        .expect("text should be a string");

    let write_data: Value =
        serde_json::from_str(write_text).expect("write text should be valid JSON");
    let entry_id = write_data
        .get("id")
        .expect("write result should have id")
        .as_str()
        .expect("id should be a string");
    assert!(!entry_id.is_empty(), "entry id should not be empty");

    // Search for the written content.
    let search_request = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "corvia_search",
            "arguments": {
                "query": "Rust ownership memory safety",
                "limit": 5
            }
        }
    });

    send_message(&mut child, &search_request);
    let search_response = read_message(&mut reader).expect("should receive search response");

    let search_result = search_response
        .get("result")
        .expect("search response should have result");
    let search_content = search_result
        .get("content")
        .expect("result should have content")
        .as_array()
        .expect("content should be an array");

    assert!(
        !search_content.is_empty(),
        "search content should not be empty"
    );
    let search_text = search_content[0]
        .get("text")
        .expect("content item should have text")
        .as_str()
        .expect("text should be a string");

    let search_data: Value =
        serde_json::from_str(search_text).expect("search text should be valid JSON");
    let results = search_data
        .get("results")
        .expect("search data should have results")
        .as_array()
        .expect("results should be an array");

    assert!(
        !results.is_empty(),
        "search should find the written entry"
    );

    // Verify our written entry appears in results.
    let result_ids: Vec<&str> = results
        .iter()
        .filter_map(|r| r.get("id").and_then(|id| id.as_str()))
        .collect();
    assert!(
        result_ids.contains(&entry_id),
        "written entry {entry_id} should appear in search results, got: {result_ids:?}"
    );

    // Cleanup.
    drop(child.stdin.take());
    let _ = child.wait();
}
