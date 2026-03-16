# Dashboard REST API (Rust) — Implementation Plan

> **Status:** Shipped (v0.3.7)

**Goal:** Add `/api/dashboard/*` REST endpoints to corvia-server for health, traces, logs, and status — replacing the Python corvia-dev data path.

**Architecture:** New `dashboard` module in corvia-server with health probing (raw TCP/HTTP), structured log parsing (ported from Python `traces.py`), and Axum handlers. Shared response types defined in corvia-common. Dashboard UI (separate plan) polls these endpoints.

**Tech Stack:** Rust, Axum 0.8, tokio, chrono, serde/serde_json

**Design Spec:** `2026-03-11-standalone-dashboard-design.md`

**Parallel Track:** Frontend plan at `2026-03-11-dashboard-ui-standalone.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/corvia-common/src/dashboard.rs` | Shared response types (ServiceStatus, SpanStats, TracesData, etc.) |
| `crates/corvia-server/src/dashboard/mod.rs` | Dashboard router + endpoint handlers |
| `crates/corvia-server/src/dashboard/health.rs` | Service health probing (HTTP, gRPC, TCP) |
| `crates/corvia-server/src/dashboard/traces.rs` | Log file reading, trace line parsing, span aggregation, module classification |

### Modified Files

| File | Change |
|------|--------|
| `crates/corvia-common/src/lib.rs` | Add `pub mod dashboard;` |
| `crates/corvia-server/src/lib.rs` | Add `pub mod dashboard;` |
| `crates/corvia-server/Cargo.toml` | Verify `chrono`, `tempfile` (dev), `serde_json` deps present |
| `crates/corvia-cli/src/main.rs` | Merge dashboard router into app (~line 490) |

All paths relative to `repos/corvia/`.

---

## Chunk 1: Types & Module Skeleton

### Task 1: Define dashboard response types

**Files:**
- Create: `crates/corvia-common/src/dashboard.rs`
- Modify: `crates/corvia-common/src/lib.rs`

-**Step 1: Create `dashboard.rs` with all shared types**

```rust
// crates/corvia-common/src/dashboard.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Service health state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceState {
    Healthy,
    Unhealthy,
    Starting,
    Stopped,
}

/// Individual service status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub state: ServiceState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

/// Span timing statistics (mirrors Python SpanStats)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SpanStats {
    pub count: u64,
    pub count_1h: u64,
    pub avg_ms: f64,
    pub last_ms: f64,
    pub errors: u64,
}

/// A structured trace event (mirrors Python TraceEvent)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEvent {
    pub ts: String,
    pub level: String,
    pub module: String,
    pub msg: String,
}

/// Aggregated trace data (mirrors Python TracesData)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TracesData {
    pub spans: HashMap<String, SpanStats>,
    pub recent_events: Vec<TraceEvent>,
}

/// Dashboard config summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub embedding_provider: String,
    pub merge_provider: String,
    pub storage: String,
    pub workspace: String,
}

/// GET /api/dashboard/status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatusResponse {
    pub services: Vec<ServiceStatus>,
    pub entry_count: u64,
    pub agent_count: usize,
    pub merge_queue_depth: u64,
    pub session_count: usize,
    pub config: DashboardConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traces: Option<TracesData>,
}

/// A single structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub module: String,
    pub message: String,
}

/// GET /api/dashboard/logs response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub entries: Vec<LogEntry>,
    pub total: usize,
}

/// GET /api/dashboard/traces response (same shape as TracesData)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracesResponse {
    pub spans: HashMap<String, SpanStats>,
    pub recent_events: Vec<TraceEvent>,
}
```

-**Step 2: Add `pub mod dashboard;` to lib.rs**

In `crates/corvia-common/src/lib.rs`, add:

```rust
pub mod dashboard;
```

-**Step 3: Write serialization round-trip test**

Append to `crates/corvia-common/src/dashboard.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_state_serializes_lowercase() {
        let status = ServiceStatus {
            name: "corvia-server".to_string(),
            state: ServiceState::Healthy,
            port: Some(8020),
            latency_ms: Some(1.5),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"healthy\""));
        assert!(json.contains("\"corvia-server\""));

        let parsed: ServiceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.state, ServiceState::Healthy);
    }

    #[test]
    fn span_stats_default_is_zeroed() {
        let stats = SpanStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_ms, 0.0);
    }

    #[test]
    fn status_response_omits_none_traces() {
        let resp = DashboardStatusResponse {
            services: vec![],
            entry_count: 0,
            agent_count: 0,
            merge_queue_depth: 0,
            session_count: 0,
            config: DashboardConfig {
                embedding_provider: "corvia".to_string(),
                merge_provider: "corvia".to_string(),
                storage: "lite".to_string(),
                workspace: "test".to_string(),
            },
            traces: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("traces"));
    }
}
```

-**Step 4: Run tests**

Run: `cd repos/corvia && cargo test -p corvia-common dashboard`
Expected: 3 tests PASS

-**Step 5: Commit**

```bash
cd repos/corvia
git add crates/corvia-common/src/dashboard.rs crates/corvia-common/src/lib.rs
git commit -m "feat(dashboard): add shared response types in corvia-common"
```

---

### Task 2: Add dashboard module skeleton to corvia-server

**Files:**
- Create: `crates/corvia-server/src/dashboard/mod.rs`
- Create: `crates/corvia-server/src/dashboard/health.rs`
- Create: `crates/corvia-server/src/dashboard/traces.rs`
- Modify: `crates/corvia-server/src/lib.rs`

-**Step 1: Create skeleton files**

`crates/corvia-server/src/dashboard/mod.rs`:
```rust
pub mod health;
pub mod traces;
```

`crates/corvia-server/src/dashboard/health.rs`:
```rust
// Service health probing — HTTP, gRPC, TCP
```

`crates/corvia-server/src/dashboard/traces.rs`:
```rust
// Structured log parsing and span aggregation
```

-**Step 2: Verify dependencies in `Cargo.toml`**

Check `crates/corvia-server/Cargo.toml` for these dependencies. Add any that are missing:

```toml
# Required (likely already present as workspace deps):
chrono = { workspace = true }
serde_json = { workspace = true }

# Dev-dependencies (for tests):
[dev-dependencies]
tempfile = "3"
```

Run: `grep -E "chrono|tempfile|serde_json" crates/corvia-server/Cargo.toml`

If any are missing, add them. Also verify `tower-http` with `cors` feature is in `crates/corvia-cli/Cargo.toml` (for Task 11).

-**Step 3: Add `pub mod dashboard;` to server lib.rs**

In `crates/corvia-server/src/lib.rs`, add:

```rust
pub mod dashboard;
```

-**Step 5: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles with no errors

-**Step 6: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/ crates/corvia-server/src/lib.rs crates/corvia-server/Cargo.toml
git commit -m "feat(dashboard): add dashboard module skeleton to corvia-server"
```

---

## Chunk 2: Health Probing

### Task 3: Service definitions and health probing

**Files:**
- Modify: `crates/corvia-server/src/dashboard/health.rs`

-**Step 1: Write tests for service definitions and health result mapping**

```rust
// crates/corvia-server/src/dashboard/health.rs

use corvia_common::dashboard::{ServiceState, ServiceStatus};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Health check result
pub struct HealthResult {
    pub healthy: Option<bool>, // None = no port to check
    pub latency_ms: f64,       // -1.0 if unhealthy or indeterminate
}

/// Health check protocol
pub enum HealthProto {
    Http,
    Grpc,
    Tcp,
    None,
}

/// Service definition for health probing
pub struct ServiceDef {
    pub name: &'static str,
    pub port: Option<u16>,
    pub health_proto: HealthProto,
    pub health_path: &'static str,
}

const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_definitions_include_core_services() {
        let defs = service_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"corvia-server"));
        assert!(names.contains(&"corvia-inference"));
    }

    #[test]
    fn corvia_server_uses_http_health() {
        let defs = service_definitions();
        let server = defs.iter().find(|d| d.name == "corvia-server").unwrap();
        assert_eq!(server.port, Some(8020));
        assert!(matches!(server.health_proto, HealthProto::Http));
        assert_eq!(server.health_path, "/health");
    }

    #[test]
    fn corvia_inference_uses_grpc_health() {
        let defs = service_definitions();
        let inference = defs.iter().find(|d| d.name == "corvia-inference").unwrap();
        assert_eq!(inference.port, Some(8030));
        assert!(matches!(inference.health_proto, HealthProto::Grpc));
    }

    #[test]
    fn health_result_to_service_status_healthy() {
        let result = HealthResult { healthy: Some(true), latency_ms: 2.5 };
        let status = result_to_status("corvia-server", Some(8020), &result);
        assert_eq!(status.state, ServiceState::Healthy);
        assert_eq!(status.latency_ms, Some(2.5));
    }

    #[test]
    fn health_result_to_service_status_unhealthy() {
        let result = HealthResult { healthy: Some(false), latency_ms: -1.0 };
        let status = result_to_status("corvia-server", Some(8020), &result);
        assert_eq!(status.state, ServiceState::Unhealthy);
        assert_eq!(status.latency_ms, None);
    }

    #[test]
    fn health_result_to_service_status_no_port() {
        let result = HealthResult { healthy: None, latency_ms: -1.0 };
        let status = result_to_status("vllm", None, &result);
        assert_eq!(status.state, ServiceState::Stopped);
    }
}
```

-**Step 2: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::health`
Expected: FAIL — `service_definitions` and `result_to_status` not found

-**Step 3: Implement service definitions and result mapping**

Add above the `#[cfg(test)]` block:

```rust
/// Known services to probe
pub fn service_definitions() -> Vec<ServiceDef> {
    vec![
        ServiceDef {
            name: "corvia-server",
            port: Some(8020),
            health_proto: HealthProto::Http,
            health_path: "/health",
        },
        ServiceDef {
            name: "corvia-inference",
            port: Some(8030),
            health_proto: HealthProto::Grpc,
            health_path: "",
        },
    ]
}

/// Convert a HealthResult to a ServiceStatus
pub fn result_to_status(name: &str, port: Option<u16>, result: &HealthResult) -> ServiceStatus {
    let state = match result.healthy {
        Some(true) => ServiceState::Healthy,
        Some(false) => ServiceState::Unhealthy,
        None => ServiceState::Stopped,
    };
    let latency_ms = if result.latency_ms > 0.0 {
        Some(result.latency_ms)
    } else {
        None
    };
    ServiceStatus {
        name: name.to_string(),
        state,
        port,
        latency_ms,
    }
}
```

-**Step 4: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::health`
Expected: 6 tests PASS

-**Step 5: Implement health check functions**

Add to `health.rs` (above `result_to_status`):

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// HTTP health check via raw TCP (avoids reqwest dependency)
pub async fn check_http(host: &str, port: u16, path: &str) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    let stream = match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(s)) => s,
        _ => return HealthResult { healthy: Some(false), latency_ms: -1.0 },
    };
    let mut stream = stream;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return HealthResult { healthy: Some(false), latency_ms: -1.0 };
    }
    let mut buf = [0u8; 64];
    match timeout(HEALTH_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 12 => {
            let response = String::from_utf8_lossy(&buf[..n]);
            if response.contains("200") || response.contains("204") {
                HealthResult {
                    healthy: Some(true),
                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                }
            } else {
                HealthResult { healthy: Some(false), latency_ms: -1.0 }
            }
        }
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// gRPC health check — TCP connect + HTTP/2 preface handshake
pub async fn check_grpc(host: &str, port: u16) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    let stream = match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(s)) => s,
        _ => return HealthResult { healthy: Some(false), latency_ms: -1.0 },
    };
    let mut stream = stream;
    let preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    if stream.write_all(preface).await.is_err() {
        return HealthResult { healthy: Some(false), latency_ms: -1.0 };
    }
    let mut buf = [0u8; 9]; // HTTP/2 frame header
    match timeout(HEALTH_TIMEOUT, stream.read_exact(&mut buf)).await {
        Ok(Ok(())) if buf[3] == 0x04 => HealthResult {
            healthy: Some(true),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        },
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// TCP-only health check — just connect, no payload
pub async fn check_tcp(host: &str, port: u16) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => HealthResult {
            healthy: Some(true),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        },
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// Check a service's health based on its definition
pub async fn check_service(svc: &ServiceDef) -> HealthResult {
    let host = "127.0.0.1";
    match (svc.port, &svc.health_proto) {
        (None, _) | (_, HealthProto::None) => HealthResult {
            healthy: None,
            latency_ms: -1.0,
        },
        (Some(port), HealthProto::Http) => check_http(host, port, svc.health_path).await,
        (Some(port), HealthProto::Grpc) => check_grpc(host, port).await,
        (Some(port), HealthProto::Tcp) => check_tcp(host, port).await,
    }
}

/// Check all known services and return their statuses
pub async fn check_all_services() -> Vec<ServiceStatus> {
    let defs = service_definitions();
    let mut statuses = Vec::with_capacity(defs.len());
    for def in &defs {
        let result = check_service(def).await;
        statuses.push(result_to_status(def.name, def.port, &result));
    }
    statuses
}
```

-**Step 6: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles with no errors

-**Step 7: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/health.rs
git commit -m "feat(dashboard): implement service health probing (HTTP, gRPC, TCP)"
```

---

## Chunk 3: Trace Parsing

### Task 4: Module classification functions

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs`

-**Step 1: Write tests for `span_to_module`**

```rust
// crates/corvia-server/src/dashboard/traces.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_to_module_exact_override() {
        assert_eq!(span_to_module("corvia.entry.embed"), "inference");
    }

    #[test]
    fn span_to_module_prefix_match() {
        assert_eq!(span_to_module("corvia.agent.register"), "agent");
        assert_eq!(span_to_module("corvia.session.create"), "agent");
        assert_eq!(span_to_module("corvia.entry.write"), "entry");
        assert_eq!(span_to_module("corvia.merge.resolve"), "merge");
        assert_eq!(span_to_module("corvia.store.insert"), "storage");
        assert_eq!(span_to_module("corvia.rag.retrieve"), "rag");
        assert_eq!(span_to_module("corvia.gc.sweep"), "gc");
    }

    #[test]
    fn span_to_module_unknown() {
        assert_eq!(span_to_module("something.else"), "unknown");
    }
}
```

-**Step 2: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: FAIL — `span_to_module` not found

-**Step 3: Implement `span_to_module`**

Add above the `#[cfg(test)]` block:

```rust
/// Classify a span name into a module.
/// Exact matches checked first, then prefix matches (first match wins).
pub fn span_to_module(span: &str) -> &'static str {
    // Exact overrides
    if span == "corvia.entry.embed" {
        return "inference";
    }

    // Prefix matches (order matters — first match wins)
    const PREFIX_MAP: &[(&str, &str)] = &[
        ("corvia.agent.", "agent"),
        ("corvia.session.", "agent"),
        ("corvia.entry.", "entry"),
        ("corvia.merge.", "merge"),
        ("corvia.store.", "storage"),
        ("corvia.rag.", "rag"),
        ("corvia.gc.", "gc"),
    ];

    for (prefix, module) in PREFIX_MAP {
        if span.starts_with(prefix) {
            return module;
        }
    }

    "unknown"
}
```

-**Step 4: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: 3 tests PASS

-**Step 5: Write tests for `target_to_module`**

Add to the `tests` module:

```rust
    #[test]
    fn target_to_module_matches_rust_paths() {
        assert_eq!(target_to_module("corvia_kernel::agent_coordinator"), "agent");
        assert_eq!(target_to_module("corvia_kernel::merge_worker"), "merge");
        assert_eq!(target_to_module("corvia_kernel::lite_store::write"), "storage");
        assert_eq!(target_to_module("corvia_kernel::knowledge_store"), "storage");
        assert_eq!(target_to_module("corvia_kernel::rag_pipeline"), "rag");
        assert_eq!(target_to_module("corvia_kernel::graph_store"), "storage");
        assert_eq!(target_to_module("corvia_inference::embedding_service"), "inference");
        assert_eq!(target_to_module("corvia_inference::chat_service"), "inference");
        assert_eq!(target_to_module("corvia_inference::model_manager"), "inference");
        assert_eq!(target_to_module("corvia_kernel::chunking"), "entry");
    }

    #[test]
    fn target_to_module_unknown() {
        assert_eq!(target_to_module("some::other::module"), "unknown");
    }
```

-**Step 6: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: FAIL — `target_to_module` not found

-**Step 7: Implement `target_to_module`**

Add below `span_to_module`:

```rust
/// Classify a Rust module target path into a dashboard module.
pub fn target_to_module(target: &str) -> &'static str {
    const TARGET_MAP: &[(&str, &str)] = &[
        ("agent_coordinator", "agent"),
        ("merge_worker", "merge"),
        ("lite_store", "storage"),
        ("knowledge_store", "storage"),
        ("postgres_store", "storage"),
        ("rag_pipeline", "rag"),
        ("graph_store", "storage"),
        ("chunking", "entry"),
        ("embedding_service", "inference"),
        ("chat_service", "inference"),
        ("model_manager", "inference"),
    ];

    for (pattern, module) in TARGET_MAP {
        if target.contains(pattern) {
            return module;
        }
    }

    "unknown"
}
```

-**Step 8: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: 5 tests PASS

-**Step 9: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs
git commit -m "feat(dashboard): implement span/target module classification"
```

---

### Task 5: Trace line parsing

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs`

-**Step 1: Write tests for `parse_trace_line`**

Add to the `tests` module:

```rust
    #[test]
    fn parse_span_with_timing() {
        let line = r#"{"timestamp":"2026-03-10T14:31:52Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{"session_id":"s1"},"elapsed_ms":12.5}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Span { level, span_name, elapsed_ms, .. } => {
                assert_eq!(level, "INFO");
                assert_eq!(span_name, "corvia.entry.write");
                assert!((elapsed_ms - 12.5).abs() < 0.01);
            }
            _ => panic!("expected Span variant"),
        }
    }

    #[test]
    fn parse_structured_event() {
        let line = r#"{"timestamp":"2026-03-10T14:31:52Z","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Event { level, msg, target, .. } => {
                assert_eq!(level, "WARN");
                assert_eq!(msg, "Slow embed: 210ms");
                assert_eq!(target, "corvia_kernel::agent_coordinator");
            }
            _ => panic!("expected Event variant"),
        }
    }

    #[test]
    fn parse_invalid_line_returns_none() {
        assert!(parse_trace_line("not json at all").is_none());
        assert!(parse_trace_line("").is_none());
        assert!(parse_trace_line("{}").is_none()); // missing required fields
    }
```

-**Step 2: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: FAIL — `parse_trace_line` and `ParsedTrace` not found

-**Step 3: Implement `parse_trace_line`**

Add above the `#[cfg(test)]` block:

```rust
use serde_json::Value;

/// Parsed trace line — either a span with timing or a structured event
pub enum ParsedTrace {
    Span {
        level: String,
        timestamp: String,
        span_name: String,
        elapsed_ms: f64,
    },
    Event {
        level: String,
        timestamp: String,
        msg: String,
        target: String,
    },
}

/// Parse a single JSON-structured trace line.
/// Returns None for invalid or non-JSON lines.
pub fn parse_trace_line(line: &str) -> Option<ParsedTrace> {
    let v: Value = serde_json::from_str(line).ok()?;

    let level = v.get("level")?.as_str()?.to_string();
    let timestamp = v.get("timestamp")?.as_str()?.to_string();

    // Check if it's a span with timing
    if let Some(span_name) = v
        .get("span")
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
    {
        if let Some(elapsed_ms) = v.get("elapsed_ms").and_then(|e| e.as_f64()) {
            return Some(ParsedTrace::Span {
                level,
                timestamp,
                span_name: span_name.to_string(),
                elapsed_ms,
            });
        }
    }

    // Structured event
    let msg = v
        .get("fields")
        .and_then(|f| f.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let target = v
        .get("target")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    Some(ParsedTrace::Event {
        level,
        timestamp,
        msg,
        target,
    })
}
```

-**Step 4: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: 8 tests PASS (5 prior + 3 new)

-**Step 5: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs
git commit -m "feat(dashboard): implement trace line parsing"
```

---

### Task 6: Trace aggregation and log file reading

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs`

-**Step 1: Write test for `collect_traces_from_lines`**

Add to the `tests` module:

```rust
    #[test]
    fn collect_traces_aggregates_spans() {
        let lines = vec![
            r#"{"timestamp":"2026-03-10T14:31:50Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":10.0}"#,
            r#"{"timestamp":"2026-03-10T14:31:51Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":20.0}"#,
            r#"{"timestamp":"2026-03-10T14:31:52Z","level":"ERROR","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":30.0}"#,
        ];
        let data = collect_traces_from_lines(&lines);

        let span = data.spans.get("corvia.entry.write").unwrap();
        assert_eq!(span.count, 3);
        assert!((span.avg_ms - 20.0).abs() < 0.01);
        assert!((span.last_ms - 30.0).abs() < 0.01);
        assert_eq!(span.errors, 1);
    }

    #[test]
    fn collect_traces_captures_events() {
        let lines = vec![
            r#"{"timestamp":"2026-03-10T14:31:52Z","level":"WARN","fields":{"message":"Slow embed"},"target":"corvia_kernel::agent_coordinator"}"#,
        ];
        let data = collect_traces_from_lines(&lines);

        assert_eq!(data.recent_events.len(), 1);
        assert_eq!(data.recent_events[0].level, "warn");
        assert_eq!(data.recent_events[0].module, "agent");
        assert_eq!(data.recent_events[0].msg, "Slow embed");
        assert_eq!(data.recent_events[0].ts, "14:31:52");
    }

    #[test]
    fn collect_traces_limits_to_50_events() {
        let lines: Vec<String> = (0..60)
            .map(|i| {
                format!(
                    r#"{{"timestamp":"2026-03-10T14:{:02}:00Z","level":"INFO","fields":{{"message":"event {i}"}},"target":"corvia_kernel::agent_coordinator"}}"#,
                    i
                )
            })
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let data = collect_traces_from_lines(&line_refs);

        assert_eq!(data.recent_events.len(), 50);
    }

    #[test]
    fn normalize_level_variants() {
        assert_eq!(normalize_level("WARN"), "warn");
        assert_eq!(normalize_level("WARNING"), "warn");
        assert_eq!(normalize_level("ERROR"), "error");
        assert_eq!(normalize_level("ERR"), "error");
        assert_eq!(normalize_level("DEBUG"), "debug");
        assert_eq!(normalize_level("TRACE"), "debug");
        assert_eq!(normalize_level("INFO"), "info");
        assert_eq!(normalize_level("anything"), "info");
    }

    #[test]
    fn short_timestamp_extracts_time() {
        assert_eq!(short_timestamp("2026-03-10T14:31:52Z"), "14:31:52");
        assert_eq!(short_timestamp("2026-03-10T14:31:52.123Z"), "14:31:52");
        assert_eq!(short_timestamp("no-t-here"), "no-t-here");
    }
```

-**Step 2: Run tests to verify they fail**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: FAIL — `collect_traces_from_lines`, `normalize_level`, `short_timestamp` not found

-**Step 3: Implement helper functions and aggregation**

Add to `traces.rs` (above `#[cfg(test)]`):

```rust
use corvia_common::dashboard::{SpanStats, TraceEvent, TracesData};
use std::collections::HashMap;
use std::path::Path;

/// Normalize log level string to lowercase standard form
pub fn normalize_level(level: &str) -> &'static str {
    match level.to_lowercase().as_str() {
        "warn" | "warning" => "warn",
        "error" | "err" => "error",
        "debug" | "trace" => "debug",
        _ => "info",
    }
}

/// Extract HH:MM:SS from an ISO timestamp string
pub fn short_timestamp(ts: &str) -> String {
    if let Some(t_pos) = ts.find('T') {
        let rest = &ts[t_pos + 1..];
        if rest.len() >= 8 {
            return rest[..8].to_string();
        }
    }
    ts.to_string()
}

/// Parse ISO timestamp to epoch seconds (for 1-hour window filtering)
fn timestamp_to_epoch(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|dt| dt.timestamp())
        .or_else(|| {
            // Try without timezone suffix
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S")
                .ok()
                .map(|ndt| ndt.and_utc().timestamp())
        })
}

/// Aggregate trace data from parsed log lines.
/// Computes span statistics (all-time + 1-hour window) and collects recent events.
pub fn collect_traces_from_lines(lines: &[&str]) -> TracesData {
    let now = chrono::Utc::now().timestamp();
    let one_hour_ago = now - 3600;

    let mut span_all: HashMap<String, Vec<f64>> = HashMap::new();
    let mut span_1h: HashMap<String, Vec<f64>> = HashMap::new();
    let mut span_errors: HashMap<String, u64> = HashMap::new();
    let mut events: Vec<TraceEvent> = Vec::new();

    for line in lines {
        let parsed = match parse_trace_line(line) {
            Some(p) => p,
            None => continue,
        };

        match parsed {
            ParsedTrace::Span {
                level,
                timestamp,
                span_name,
                elapsed_ms,
            } => {
                span_all
                    .entry(span_name.clone())
                    .or_default()
                    .push(elapsed_ms);

                if let Some(epoch) = timestamp_to_epoch(&timestamp) {
                    if epoch >= one_hour_ago {
                        span_1h
                            .entry(span_name.clone())
                            .or_default()
                            .push(elapsed_ms);
                    }
                }

                let level_lower = level.to_lowercase();
                if level_lower == "error" || level_lower == "err" {
                    *span_errors.entry(span_name).or_default() += 1;
                }
            }
            ParsedTrace::Event {
                level,
                timestamp,
                msg,
                target,
            } => {
                let module = target_to_module(&target);
                events.push(TraceEvent {
                    ts: short_timestamp(&timestamp),
                    level: normalize_level(&level).to_string(),
                    module: module.to_string(),
                    msg,
                });
            }
        }
    }

    // Build SpanStats
    let mut spans = HashMap::new();
    for (name, timings) in &span_all {
        let count = timings.len() as u64;
        let avg_ms = timings.iter().sum::<f64>() / count as f64;
        let last_ms = *timings.last().unwrap_or(&0.0);
        let count_1h = span_1h.get(name).map(|v| v.len() as u64).unwrap_or(0);
        let errors = span_errors.get(name).copied().unwrap_or(0);

        spans.insert(
            name.clone(),
            SpanStats {
                count,
                count_1h,
                avg_ms,
                last_ms,
                errors,
            },
        );
    }

    // Keep last 50 events
    let recent_events = if events.len() > 50 {
        events[events.len() - 50..].to_vec()
    } else {
        events
    };

    TracesData {
        spans,
        recent_events,
    }
}

/// Read the last `n` lines from a file
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
    let start = all_lines.len().saturating_sub(n);
    all_lines[start..].to_vec()
}

/// Resolve the log directory — checks env var, then default path
pub fn log_dir() -> std::path::PathBuf {
    std::env::var("CORVIA_LOG_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/corvia-dev-logs"))
}

/// Collect traces from all .log files in a directory.
/// Reads last 500 lines per file to bound memory.
pub fn collect_traces(log_dir: &Path) -> TracesData {
    let mut all_lines = Vec::new();

    if let Ok(entries) = std::fs::read_dir(log_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                let lines = tail_lines(&path, 500);
                all_lines.extend(lines);
            }
        }
    }

    let line_refs: Vec<&str> = all_lines.iter().map(|s| s.as_str()).collect();
    collect_traces_from_lines(&line_refs)
}
```

-**Step 4: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: 13 tests PASS

-**Step 5: Write test for `tail_lines` with a temp file**

Add to the `tests` module:

```rust
    #[test]
    fn tail_lines_reads_last_n() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.log");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 0..10 {
            writeln!(f, "line {i}").unwrap();
        }
        drop(f);

        let lines = tail_lines(&path, 3);
        assert_eq!(lines, vec!["line 7", "line 8", "line 9"]);
    }

    #[test]
    fn tail_lines_missing_file_returns_empty() {
        let lines = tail_lines(Path::new("/nonexistent/file.log"), 10);
        assert!(lines.is_empty());
    }
```

-**Step 6: Run tests to verify they pass**

Run: `cd repos/corvia && cargo test -p corvia-server dashboard::traces`
Expected: 15 tests PASS

-**Step 7: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs
git commit -m "feat(dashboard): implement trace aggregation and log file reading"
```

---

## Chunk 4: API Endpoints & Router

### Task 7: Status endpoint handler

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Implement the status endpoint handler**

Replace `crates/corvia-server/src/dashboard/mod.rs`:

```rust
pub mod health;
pub mod traces;

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};

use corvia_common::dashboard::{
    DashboardConfig, DashboardStatusResponse, LogEntry, LogsResponse,
    TracesResponse,
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
        .with_state(state)
}

/// GET /api/dashboard/status
/// Returns service health, store metrics, config summary, and optional traces.
async fn status_handler(
    State(state): State<Arc<AppState>>,
) -> Json<DashboardStatusResponse> {
    // Health check all services
    let services = health::check_all_services().await;

    // Store metrics via kernel ops (same as corvia_system_status MCP tool)
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

    // Config summary
    let cfg = state.config.read().unwrap();
    let config = DashboardConfig {
        embedding_provider: cfg.embedding.provider.clone(),
        merge_provider: cfg
            .merge
            .as_ref()
            .map(|m| m.provider.clone())
            .unwrap_or_else(|| "none".to_string()),
        storage: cfg.storage.backend.clone(),
        workspace: cfg
            .project
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "unknown".to_string()),
    };
    drop(cfg);

    // Traces from log files
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
```

-**Step 2: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles (adjust field names if config struct differs — check `corvia-common/src/config.rs`)

> **Note:** If config field names don't match exactly (e.g., `cfg.embedding.provider` vs a different accessor), adjust based on actual `CorviaConfig` fields. The pattern is the same.

-**Step 3: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): implement /api/dashboard/status endpoint"
```

---

### Task 8: Traces endpoint handler

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Add the traces handler**

Add below `status_handler` in `mod.rs`:

```rust
/// GET /api/dashboard/traces
/// Returns span statistics and recent events from structured logs.
async fn traces_handler(
    State(_state): State<Arc<AppState>>,
) -> Json<TracesResponse> {
    let log_dir = traces::log_dir();
    let data = traces::collect_traces(&log_dir);

    Json(TracesResponse {
        spans: data.spans,
        recent_events: data.recent_events,
    })
}
```

-**Step 2: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles with no errors

-**Step 3: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): implement /api/dashboard/traces endpoint"
```

---

### Task 9: Logs endpoint handler

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Add query params and logs handler**

Add the query params struct and handler in `mod.rs`:

```rust
/// Query params for /api/dashboard/logs
#[derive(Debug, serde::Deserialize)]
pub struct LogsQuery {
    /// Filter by service name (log file)
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
    let limit = params.limit.unwrap_or(100);

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
```

-**Step 2: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles with no errors

-**Step 3: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): implement /api/dashboard/logs endpoint with filtering"
```

---

### Task 10: Config endpoint (thin wrapper)

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Add config handler**

```rust
/// GET /api/dashboard/config
/// Returns current server config summary (read-only).
async fn config_handler(
    State(state): State<Arc<AppState>>,
) -> Json<DashboardConfig> {
    let cfg = state.config.read().unwrap();
    let config = DashboardConfig {
        embedding_provider: cfg.embedding.provider.clone(),
        merge_provider: cfg
            .merge
            .as_ref()
            .map(|m| m.provider.clone())
            .unwrap_or_else(|| "none".to_string()),
        storage: cfg.storage.backend.clone(),
        workspace: cfg
            .project
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "unknown".to_string()),
    };
    Json(config)
}
```

-**Step 2: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles with no errors

-**Step 3: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): implement /api/dashboard/config endpoint"
```

---

### Task 11: Graph endpoint (thin wrapper)

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

The design spec includes `GET /api/dashboard/graph?scope=X` which forwards to the existing kernel graph store. This is a thin REST wrapper.

-**Step 1: Add graph query params and handler**

Add to `mod.rs`:

```rust
/// Query params for /api/dashboard/graph
#[derive(Debug, serde::Deserialize)]
pub struct GraphQuery {
    pub scope: Option<String>,
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

    let direction = corvia_common::EdgeDirection::Both;
    let edges = state
        .graph
        .edges(entry_id, direction, None)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let edge_dtos: Vec<serde_json::Value> = edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "from": e.from,
                "to": e.to,
                "relation": e.relation,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "edges": edge_dtos })))
}
```

> **Note:** Adjust `GraphStore::edges()` method signature to match the actual trait. The pattern follows the existing `corvia_graph` MCP tool handler in `mcp.rs`.

-**Step 2: Add route to router**

Update the `router()` function to include:

```rust
.route("/api/dashboard/graph", get(graph_handler))
```

-**Step 3: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-server`
Expected: Compiles (adjust `GraphStore` method call if needed)

-**Step 4: Commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): implement /api/dashboard/graph endpoint"
```

---

### Task 12: Wire dashboard router into server + CORS

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

-**Step 1: Merge dashboard router in server startup**

In `crates/corvia-cli/src/main.rs`, find the router merge section (~line 490):

```rust
// BEFORE:
let mut app = corvia_server::rest::router(state.clone());
app = app.merge(corvia_server::mcp::mcp_router(state));

// AFTER:
let mut app = corvia_server::rest::router(state.clone());
app = app.merge(corvia_server::mcp::mcp_router(state.clone()));
app = app.merge(corvia_server::dashboard::router(state));
```

-**Step 2: Add permissive CORS for dashboard origin**

If CORS is not already configured, add to the app builder in `main.rs`:

```rust
use tower_http::cors::CorsLayer;

// After merging all routers:
let app = app.layer(CorsLayer::permissive());
```

> **Note:** If CORS is already configured (check existing code), extend it to include `http://localhost:8021`. For development, `CorsLayer::permissive()` is simplest. For production, restrict to specific origins.

-**Step 3: Verify compilation**

Run: `cd repos/corvia && cargo check -p corvia-cli`
Expected: Compiles with no errors

-**Step 4: Commit**

```bash
cd repos/corvia
git add crates/corvia-cli/src/main.rs
git commit -m "feat(dashboard): wire dashboard router into server with CORS"
```

---

### Task 13: Manual integration test

-**Step 1: Start the server**

```bash
cd /workspaces/corvia-workspace && corvia serve &
```

Wait for "Server listening on 127.0.0.1:8020"

-**Step 2: Test status endpoint**

```bash
curl -s http://localhost:8020/api/dashboard/status | python3 -m json.tool
```

Expected: JSON with `services`, `entry_count`, `agent_count`, `merge_queue_depth`, `session_count`, `config` fields.

-**Step 3: Test traces endpoint**

```bash
curl -s http://localhost:8020/api/dashboard/traces | python3 -m json.tool
```

Expected: JSON with `spans` (possibly empty) and `recent_events` (possibly empty).

-**Step 4: Test logs endpoint with filters**

```bash
curl -s "http://localhost:8020/api/dashboard/logs?level=error&limit=10" | python3 -m json.tool
```

Expected: JSON with `entries` array and `total` count.

-**Step 5: Test config endpoint**

```bash
curl -s http://localhost:8020/api/dashboard/config | python3 -m json.tool
```

Expected: JSON with `embedding_provider`, `merge_provider`, `storage`, `workspace`.

-**Step 6: Test graph endpoint**

```bash
curl -s "http://localhost:8020/api/dashboard/graph" | python3 -m json.tool
```

Expected: JSON with `edges` array (possibly empty).

-**Step 7: Run full test suite**

```bash
cd repos/corvia && cargo test -p corvia-server dashboard && cargo test -p corvia-common dashboard
```

Expected: All tests PASS

-**Step 8: Final commit**

```bash
cd repos/corvia
git add crates/corvia-server/src/dashboard/ crates/corvia-common/src/dashboard.rs crates/corvia-common/src/lib.rs crates/corvia-server/src/lib.rs crates/corvia-cli/src/main.rs
git commit -m "feat(dashboard): complete REST API — status, traces, logs, config, graph endpoints"
```
