//! Docker-based E2E tests for the MCP server.
//!
//! These tests spin up a real corvia server inside a Docker container with
//! real ONNX embeddings, then exercise the MCP HTTP endpoint via reqwest.
//! Auto-skips when Docker or binaries are unavailable.
//!
//! **Tier 4** — requires Docker + built debug binaries + devcontainer image.
//!
//! Run with:
//!   cargo test --test docker_mcp_e2e_test -- --nocapture
//!   CORVIA_E2E_EXTENDED=1 cargo test --test docker_mcp_e2e_test -- --nocapture
//!
//! Or via Makefile:
//!   make test-e2e-docker
//!   make test-e2e-docker-extended

use serde_json::{json, Value};
use std::io::BufRead;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::OnceCell;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const DOCKER_IMAGE: &str = "corvia-workspace_devcontainer-app:latest";
const API_PORT: u16 = 9020;
const INFERENCE_PORT: u16 = 9030;
const MCP_URL: &str = "http://127.0.0.1:9020/mcp";

/// Shared test environment -- lazily initialized, lives for the process.
static E2E_ENV: OnceCell<E2eEnv> = OnceCell::const_new();

/// Whether Docker E2E prerequisites are met (sync check).
static E2E_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn is_extended() -> bool {
    std::env::var("CORVIA_E2E_EXTENDED").is_ok()
}

struct E2eEnv {
    container_name: String,
    client: reqwest::Client,
    trace_dir: PathBuf,
    _workdir: tempfile::TempDir,
}

impl Drop for E2eEnv {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "-f", &self.container_name])
            .output();
    }
}

/// Check all prerequisites. Returns false (auto-skip) if any fail.
fn check_prerequisites() -> bool {
    *E2E_AVAILABLE.get_or_init(|| {
        if !Command::new("docker")
            .args(["info"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            eprintln!("E2E SKIP: Docker not available");
            return false;
        }

        let image_check = Command::new("docker")
            .args(["images", "--format", "{{.Repository}}:{{.Tag}}", DOCKER_IMAGE])
            .output();
        if !image_check.map(|o| !o.stdout.is_empty()).unwrap_or(false) {
            eprintln!("E2E SKIP: Docker image {DOCKER_IMAGE} not found");
            return false;
        }

        let workspace = workspace_root();
        let corvia_bin = workspace.join("repos/corvia/target/debug/corvia");
        let inference_bin = workspace.join("repos/corvia/target/debug/corvia-inference");
        if !corvia_bin.exists() || !inference_bin.exists() {
            eprintln!(
                "E2E SKIP: Debug binaries not built (run: cargo build -p corvia-cli -p corvia-inference)"
            );
            return false;
        }

        if resolve_host_path().is_none() {
            eprintln!("E2E SKIP: Cannot determine host workspace path (not in a devcontainer?)");
            return false;
        }

        true
    })
}

fn workspace_root() -> PathBuf {
    PathBuf::from(
        std::env::var("CORVIA_WORKSPACE")
            .unwrap_or_else(|_| "/workspaces/corvia-workspace".into()),
    )
}

fn resolve_host_path() -> Option<String> {
    let hostname = String::from_utf8(Command::new("hostname").output().ok()?.stdout)
        .ok()?
        .trim()
        .to_string();

    let inspect = Command::new("docker")
        .args(["inspect", &hostname, "--format", "{{json .Mounts}}"])
        .output()
        .ok()?;

    if !inspect.status.success() {
        return None;
    }

    let mounts: Vec<Value> = serde_json::from_slice(&inspect.stdout).ok()?;
    for mount in mounts {
        if mount["Destination"].as_str() == Some("/workspaces/corvia-workspace") {
            return mount["Source"].as_str().map(String::from);
        }
    }
    None
}

/// Dump the last N lines of container logs for debugging.
fn dump_container_logs(name: &str) -> String {
    Command::new("docker")
        .args(["logs", "--tail", "30", name])
        .output()
        .map(|o| {
            let out = String::from_utf8_lossy(&o.stdout).to_string();
            let err = String::from_utf8_lossy(&o.stderr).to_string();
            format!("{out}\n{err}")
        })
        .unwrap_or_default()
}

async fn get_or_init_env() -> &'static E2eEnv {
    E2E_ENV.get_or_init(|| async {
        let host_workspace =
            resolve_host_path().expect("Host path resolution should succeed (checked in prereqs)");
        let ws = workspace_root();
        let container_name = format!("corvia-e2e-{}", std::process::id());

        // Clean up ALL leftover e2e containers (not just current PID)
        if let Ok(output) = Command::new("docker")
            .args(["ps", "-a", "--format", "{{.Names}}", "--filter", "name=corvia-e2e-"])
            .output()
        {
            for name in String::from_utf8_lossy(&output.stdout).lines() {
                let _ = Command::new("docker").args(["rm", "-f", name]).output();
            }
        }

        // Clean up stale e2e tempdirs from prior runs
        if let Ok(entries) = std::fs::read_dir(ws.join("repos/corvia/target")) {
            for entry in entries.flatten() {
                if entry.file_name().to_str().is_some_and(|n| n.starts_with("e2e-")) {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }

        let workdir = tempfile::Builder::new()
            .prefix("e2e-")
            .tempdir_in(ws.join("repos/corvia/target"))
            .expect("Failed to create tempdir in target/");
        let workdir_path = workdir.path();

        // Patch corvia.toml
        let toml_content = std::fs::read_to_string(ws.join("corvia.toml"))
            .expect("Failed to read corvia.toml");
        let patched = toml_content
            .replace("port = 8020", &format!("port = {API_PORT}"))
            .replace(
                "http://127.0.0.1:8030",
                &format!("http://127.0.0.1:{INFERENCE_PORT}"),
            );
        std::fs::write(workdir_path.join("corvia.toml"), patched)
            .expect("Failed to write patched toml");

        let entrypoint = format!(
            r#"#!/bin/bash
set -e
cd /e2e-workdir

# Init workspace store (creates redb tables)
CORVIA_WORKSPACE=/e2e-workdir corvia workspace init 2>/dev/null || true

# Start inference server
FASTEMBED_CACHE_PATH=/root/.fastembed_cache corvia-inference serve --port {INFERENCE_PORT} &

# Wait for inference gRPC
for i in $(seq 1 60); do
    if bash -c "echo > /dev/tcp/127.0.0.1/{INFERENCE_PORT}" 2>/dev/null; then
        break
    fi
    [ "$i" = "60" ] && {{ echo "ERROR: inference timeout"; exit 1; }}
    sleep 1
done

# Start corvia server
export CORVIA_LOG_DIR=/e2e-traces
export CORVIA_WORKSPACE=/e2e-workdir
exec corvia serve
"#
        );
        std::fs::write(workdir_path.join("entrypoint.sh"), entrypoint)
            .expect("Failed to write entrypoint");

        let traces_dir = workdir_path.join("traces");
        std::fs::create_dir_all(&traces_dir).expect("Failed to create traces dir");

        let host_workdir = workdir_path
            .to_str()
            .unwrap()
            .replace("/workspaces/corvia-workspace", &host_workspace);
        let host_traces = traces_dir
            .to_str()
            .unwrap()
            .replace("/workspaces/corvia-workspace", &host_workspace);

        let hostname = String::from_utf8(Command::new("hostname").output().unwrap().stdout)
            .unwrap()
            .trim()
            .to_string();

        // Override compose labels inherited from the devcontainer image so that
        // VS Code does not mistake E2E containers for the real devcontainer.
        // Uses the unique container_name so multiple E2E instances never collide.
        let label_project = format!("com.docker.compose.project={container_name}");
        let label_service = format!("com.docker.compose.service={container_name}");
        let status = Command::new("docker")
            .args([
                "run", "-d",
                "--name", &container_name,
                "--label", &label_project,
                "--label", &label_service,
                &format!("--network=container:{hostname}"),
                "-v", &format!("{}/repos/corvia/target/debug/corvia:/usr/local/bin/corvia:ro", host_workspace),
                "-v", &format!("{}/repos/corvia/target/debug/corvia-inference:/usr/local/bin/corvia-inference:ro", host_workspace),
                "-v", &format!("{}/.fastembed_cache:/root/.fastembed_cache:ro", host_workspace),
                "-v", &format!("{host_workdir}:/e2e-workdir"),
                "-v", &format!("{host_traces}:/e2e-traces"),
                "-e", "RUST_LOG=info",
                DOCKER_IMAGE,
                "bash", "/e2e-workdir/entrypoint.sh",
            ])
            .status()
            .expect("Failed to run docker");

        assert!(status.success(), "docker run failed");

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        // Wait for server
        let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
        loop {
            if tokio::time::Instant::now() > deadline {
                let logs = dump_container_logs(&container_name);
                // Clean up the container since Drop won't run (we're about to panic)
                let _ = Command::new("docker").args(["rm", "-f", &container_name]).output();
                panic!("E2E server did not start within 90s\n{logs}");
            }
            let probe = client
                .post(MCP_URL)
                .header("Content-Type", "application/json")
                .json(&json!({
                    "jsonrpc": "2.0", "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "clientInfo": {"name": "e2e-test", "version": "0.0.1"}
                    }
                }))
                .send()
                .await;
            if probe.is_ok_and(|r| r.status().is_success()) {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        E2eEnv {
            container_name,
            client,
            trace_dir: traces_dir,
            _workdir: workdir,
        }
    }).await
}

// ---------------------------------------------------------------------------
// MCP client helpers
// ---------------------------------------------------------------------------

async fn mcp_call(env: &E2eEnv, tool: &str, args: Value, agent_id: &str) -> Value {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": args,
            "_meta": { "agent_id": agent_id }
        }
    });

    let resp = env
        .client
        .post(MCP_URL)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .unwrap_or_else(|e| {
            let logs = dump_container_logs(&env.container_name);
            panic!("MCP request to {tool} failed: {e}\nContainer logs:\n{logs}");
        });

    let body: Value = resp.json().await.expect("Failed to parse MCP response");

    if let Some(err) = body.get("error") {
        let logs = dump_container_logs(&env.container_name);
        panic!("MCP error calling {tool}: {err}\nContainer logs:\n{logs}");
    }

    body
}

/// Write an entry via MCP. Asserts the write was actually accepted (not blocked by dedup).
/// Uses force_write to bypass dedup (E2E tests reuse similar content patterns).
/// Returns the response text.
async fn mcp_write(env: &E2eEnv, scope: &str, content: &str, agent_id: &str) -> String {
    let result = mcp_call(
        env,
        "corvia_write",
        json!({
            "content": content,
            "scope_id": scope,
            "content_role": "finding",
            "source_origin": "workspace",
            "force_write": true
        }),
        agent_id,
    )
    .await;

    let text = result["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");

    // Check for dedup block (isError: true in MCP response)
    let is_error = result["result"]["isError"].as_bool().unwrap_or(false);
    assert!(
        !is_error,
        "Write was blocked (likely dedup): {text}"
    );

    text.to_string()
}

async fn mcp_search(env: &E2eEnv, query: &str, scope: &str, agent_id: &str) -> Value {
    let result = mcp_call(
        env,
        "corvia_search",
        json!({ "query": query, "scope_id": scope, "limit": 10 }),
        agent_id,
    )
    .await;

    let text = result["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("{}");
    serde_json::from_str(text).unwrap_or(json!({}))
}

/// Poll search until at least `min_results` are found, with backoff.
async fn search_with_retry(
    env: &E2eEnv,
    query: &str,
    scope: &str,
    agent_id: &str,
    min_results: u64,
    max_wait: Duration,
) -> Value {
    let deadline = tokio::time::Instant::now() + max_wait;
    loop {
        let results = mcp_search(env, query, scope, agent_id).await;
        let count = results["count"].as_u64().unwrap_or(0);
        if count >= min_results {
            return results;
        }
        if tokio::time::Instant::now() > deadline {
            panic!(
                "Search for '{query}' in scope '{scope}' found {count} results (need {min_results}) after {max_wait:?}. Results: {results}"
            );
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

// ---------------------------------------------------------------------------
// Trace reader
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TraceEvent {
    span_name: String,
    level: String,
    elapsed_ms: f64,
}

fn read_traces(env: &E2eEnv) -> Vec<TraceEvent> {
    let trace_file = env.trace_dir.join("corvia-traces.log");
    let file = match std::fs::File::open(&trace_file) {
        Ok(f) => f,
        Err(_) => return vec![],
    };

    std::io::BufReader::new(file)
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let v: Value = serde_json::from_str(&line).ok()?;
            Some(TraceEvent {
                span_name: v["span"]["name"].as_str()?.to_string(),
                level: v["level"].as_str().unwrap_or("unknown").to_string(),
                elapsed_ms: v["elapsed_ms"].as_f64().unwrap_or(0.0),
            })
        })
        .collect()
}

fn assert_span_exists(traces: &[TraceEvent], span_name: &str) {
    assert!(
        traces.iter().any(|t| t.span_name == span_name),
        "Expected span '{span_name}' not found. Found: {:?}",
        traces.iter().map(|t| &t.span_name).collect::<Vec<_>>()
    );
}

fn assert_no_error_spans(traces: &[TraceEvent]) {
    let errors: Vec<_> = traces.iter().filter(|t| t.level == "ERROR").collect();
    assert!(errors.is_empty(), "Found ERROR-level spans: {errors:?}");
}

/// Extract UUID from text using the standard 8-4-4-4-12 hex pattern.
fn extract_uuid(text: &str) -> Option<&str> {
    // UUID v4: 8-4-4-4-12 hex chars
    text.split_whitespace().find(|s| {
        s.len() == 36
            && s.chars()
                .enumerate()
                .all(|(i, c)| match i {
                    8 | 13 | 18 | 23 => c == '-',
                    _ => c.is_ascii_hexdigit(),
                })
    })
}

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

macro_rules! skip_if_unavailable {
    () => {
        if !check_prerequisites() {
            return;
        }
    };
}

macro_rules! skip_if_not_extended {
    () => {
        if !is_extended() {
            eprintln!("SKIPPING (set CORVIA_E2E_EXTENDED=1 to enable)");
            return;
        }
    };
}

// ===========================================================================
// Core tests (Tier 4 -- always run when Docker available)
// ===========================================================================

#[tokio::test]
async fn test_cross_scope_write() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    // Write to "corvia" scope -- agent registered with scopes=["corvia"]
    let r1 = mcp_write(env, "corvia", "Cross-scope test: entry in corvia", "e2e-setup").await;
    assert!(r1.contains("written"), "First write response: {r1}");

    // Write to "devcontainer-telemetry" -- requires grant_scope fix
    let r2 = mcp_write(
        env, "devcontainer-telemetry",
        "Cross-scope test: telemetry entry", "e2e-setup",
    ).await;
    assert!(r2.contains("written"), "Cross-scope write response: {r2}");

    // Write back to "corvia" -- verify no regression
    let r3 = mcp_write(
        env, "corvia",
        "Cross-scope test: back to corvia", "e2e-setup",
    ).await;
    assert!(r3.contains("written"), "Return write response: {r3}");
}

#[tokio::test]
async fn test_write_search_roundtrip() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    mcp_write(
        env, "corvia",
        "Corvia uses HNSW algorithm for approximate nearest neighbor vector search indexing",
        "e2e-search-agent",
    ).await;

    // Poll until search finds results (replaces fixed sleep)
    search_with_retry(
        env, "vector search algorithm", "corvia", "e2e-search-agent",
        1, Duration::from_secs(15),
    ).await;
}

#[tokio::test]
async fn test_agent_lifecycle() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    mcp_write(env, "corvia", "Agent lifecycle test entry", "e2e-lifecycle-agent").await;

    let result = mcp_call(env, "corvia_agents_list", json!({}), "e2e-lifecycle-agent").await;
    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    let agents: Value = serde_json::from_str(text).unwrap_or(json!({}));

    let agent_list = agents["agents"].as_array().expect("agents should be array");
    let found = agent_list
        .iter()
        .any(|a| a["agent_id"].as_str() == Some("e2e-lifecycle-agent"));
    assert!(found, "Agent should appear in agents list: {agents}");
}

#[tokio::test]
async fn test_system_status() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    mcp_write(env, "corvia", "System status test entry", "e2e-status-agent").await;

    let result = mcp_call(
        env, "corvia_system_status",
        json!({"scope_id": "corvia"}), "e2e-status-agent",
    ).await;

    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    let status: Value = serde_json::from_str(text).unwrap_or(json!({}));

    // Verify entry count is positive (not just present)
    let count = status["entry_count"].as_u64()
        .or_else(|| status["entries"].as_u64())
        .unwrap_or(0);
    assert!(count >= 1, "System status entry count should be >= 1, got: {status}");
}

#[tokio::test]
async fn test_config_get() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    let result = mcp_call(
        env, "corvia_config_get",
        json!({"section": "server"}), "e2e-config-agent",
    ).await;

    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    let config: Value = serde_json::from_str(text).unwrap_or(json!({}));

    // Verify the patched port is returned (confirms real config is loaded)
    let port = config["port"].as_u64().unwrap_or(0);
    assert_eq!(port, API_PORT as u64, "Config port should match patched value: {config}");
}

#[tokio::test]
async fn test_reason_health() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    let result = mcp_call(
        env, "corvia_reason",
        json!({"scope_id": "corvia"}), "e2e-reason-agent",
    ).await;

    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    assert!(!text.is_empty(), "Reason should return health check results");
}

#[tokio::test]
async fn test_confirmation_required_rejected() {
    skip_if_unavailable!();
    let env = get_or_init_env().await;

    // Call a Tier 2 tool WITHOUT confirmation -- should be rejected
    let payload = json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {
            "name": "corvia_gc_run",
            "arguments": {"scope_id": "corvia"},
            "_meta": { "agent_id": "e2e-confirm-agent" }
        }
    });

    let resp = env.client
        .post(MCP_URL)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("Request failed");

    let body: Value = resp.json().await.expect("Failed to parse response");
    let text = body["result"]["content"][0]["text"].as_str().unwrap_or("");

    // Should indicate confirmation is required (not execute the GC)
    assert!(
        text.contains("confirm") || text.contains("Confirm"),
        "Tier 2 tool without confirmation should be rejected: {text}"
    );
}

// ===========================================================================
// Extended tests (CORVIA_E2E_EXTENDED=1)
// ===========================================================================

#[tokio::test]
async fn test_context_retrieval() {
    skip_if_unavailable!();
    skip_if_not_extended!();
    let env = get_or_init_env().await;

    mcp_write(
        env, "corvia",
        "The authentication system uses JWT tokens with RSA-256 signatures for session management",
        "e2e-context-agent",
    ).await;
    mcp_write(
        env, "corvia",
        "Database migrations run automatically on startup using an embedded migration runner",
        "e2e-context-agent",
    ).await;
    mcp_write(
        env, "corvia",
        "The CI pipeline uses GitHub Actions with matrix builds across Linux and macOS",
        "e2e-context-agent",
    ).await;

    // Poll until indexing completes
    search_with_retry(
        env, "JWT authentication", "corvia", "e2e-context-agent",
        1, Duration::from_secs(15),
    ).await;

    let result = mcp_call(
        env, "corvia_context",
        json!({
            "query": "how does authentication work",
            "scope_id": "corvia",
            "max_tokens": 2000
        }),
        "e2e-context-agent",
    ).await;

    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    assert!(!text.is_empty(), "Context retrieval should return assembled context");
    // Verify relevance: should mention auth-related content
    let lower = text.to_lowercase();
    assert!(
        lower.contains("jwt") || lower.contains("auth") || lower.contains("session") || lower.contains("rsa"),
        "Context should contain auth-related content, got: {}", &text[..text.len().min(300)]
    );
}

#[tokio::test]
async fn test_multi_agent_writes() {
    skip_if_unavailable!();
    skip_if_not_extended!();
    let env = get_or_init_env().await;

    let (r1, r2) = tokio::join!(
        async {
            let a = mcp_write(env, "corvia", "Alpha agent: distributed tracing design", "e2e-alpha").await;
            let b = mcp_write(env, "corvia", "Alpha agent: metrics collection patterns", "e2e-alpha").await;
            (a, b)
        },
        async {
            let a = mcp_write(env, "corvia", "Beta agent: load testing methodology", "e2e-beta").await;
            let b = mcp_write(env, "corvia", "Beta agent: performance benchmarking results", "e2e-beta").await;
            (a, b)
        }
    );

    // mcp_write already asserts !isError, so if we get here all 4 succeeded
    assert!(r1.0.contains("written"), "Alpha write 1: {}", r1.0);
    assert!(r1.1.contains("written"), "Alpha write 2: {}", r1.1);
    assert!(r2.0.contains("written"), "Beta write 1: {}", r2.0);
    assert!(r2.1.contains("written"), "Beta write 2: {}", r2.1);
}

#[tokio::test]
async fn test_history_chain() {
    skip_if_unavailable!();
    skip_if_not_extended!();
    let env = get_or_init_env().await;

    let write_resp = mcp_write(
        env, "corvia",
        "History chain test: original entry about graph traversal algorithms",
        "e2e-history-agent",
    ).await;

    // Extract UUID using proper hex validation
    let entry_id = extract_uuid(&write_resp)
        .unwrap_or_else(|| panic!("Could not extract UUID from: {write_resp}"));

    let result = mcp_call(
        env, "corvia_history",
        json!({"entry_id": entry_id}), "e2e-history-agent",
    ).await;

    let text = result["result"]["content"][0]["text"].as_str().unwrap_or("");
    assert!(!text.is_empty(), "History should return chain for entry {entry_id}");
}

#[tokio::test]
async fn test_trace_health() {
    skip_if_unavailable!();
    skip_if_not_extended!();
    let env = get_or_init_env().await;

    // Write + search to generate spans
    mcp_write(
        env, "corvia",
        "Trace health test: ensuring spans are emitted correctly",
        "e2e-trace-agent",
    ).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let _ = mcp_search(env, "trace health test", "corvia", "e2e-trace-agent").await;

    // Wait for trace flush across Docker volume mount
    tokio::time::sleep(Duration::from_secs(3)).await;

    let traces = read_traces(env);
    assert!(!traces.is_empty(), "Trace file should contain events");

    assert_span_exists(&traces, "corvia.entry.write_dedup");
    assert_span_exists(&traces, "corvia.store.search");

    assert_no_error_spans(&traces);

    // Timing sanity: no span should take > 30s
    for trace in &traces {
        assert!(
            trace.elapsed_ms < 30_000.0,
            "Span {} took {:.1}ms (> 30s threshold)",
            trace.span_name, trace.elapsed_ms
        );
    }
}
