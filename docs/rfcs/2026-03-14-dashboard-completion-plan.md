# Dashboard Completion Implementation Plan

> **Status:** Shipped (v0.4.3)

**Goal:** Complete the remaining 3 dashboard features (#7 Live Session State, #9 OTEL Span Drill-Down, #10 GC Operations Dashboard) with proper OTEL trace context, live session monitoring, and GC operations visibility.

**Architecture:** Shared infrastructure first (OtelContextLayer + enhanced GcReport), then feature endpoints and frontend components. Backend in Rust (Axum handlers, tracing layers), frontend in Preact/TypeScript. All new state is in-memory (no database schema changes).

**Tech Stack:** Rust (tracing, tracing-subscriber, tracing-opentelemetry, opentelemetry, axum, serde), Preact (TypeScript, Vite), SVG for sparkline/waterfall rendering.

**Spec:** `docs/rfcs/2026-03-13-dashboard-completion-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `crates/corvia-telemetry/src/otel_context_layer.rs` | Custom tracing Layer: bridge OTEL SpanContext into fmt JSON output |
| `tools/corvia-dashboard/src/components/GcPanel.tsx` | GC operations panel (last run, history sparkline, trigger button) |
| `tools/corvia-dashboard/src/components/WaterfallView.tsx` | Span drill-down waterfall (trace list + horizontal bar chart) |
| `tools/corvia-dashboard/src/components/LiveSessionsBar.tsx` | Active session pills bar for Agents tab |

### Modified files
| File | Changes |
|------|---------|
| `crates/corvia-telemetry/src/lib.rs` | Add `mod otel_context_layer`, refactor `init_telemetry()` layer composition |
| `crates/corvia-common/src/dashboard.rs` | Add `p50_ms/p95_ms/p99_ms` to SpanStats, new GC + session + trace response types |
| `crates/corvia-kernel/src/agent_coordinator.rs` | Expand GcReport with 6 new fields, populate from gc() |
| `crates/corvia-kernel/src/ops.rs` | Add timing wrapper in gc_run(), GcHistory type |
| `crates/corvia-server/src/rest.rs` | Add `gc_history` field to AppState |
| `crates/corvia-server/src/dashboard/mod.rs` | 4 new routes + handlers |
| `crates/corvia-server/src/dashboard/traces.rs` | Add trace_id/span_id/parent_span_id to ParsedTrace, trace tree builder, percentile computation |
| `tools/corvia-dashboard/src/types.ts` | New TypeScript interfaces for GC, live sessions, trace trees, enhanced SpanStats |
| `tools/corvia-dashboard/src/api.ts` | 4 new fetch functions |
| `tools/corvia-dashboard/src/components/TracesView.tsx` | Import GcPanel + WaterfallView, percentile display in span rows |
| `tools/corvia-dashboard/src/components/AgentsView.tsx` | Import LiveSessionsBar at top of view |

---

## Chunk 1: Shared Infrastructure — Enhanced GcReport

### Task 1: Expand GcReport struct

**Files:**
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs:28-31`

- **Step 1: Write the failing test**

In the existing `#[cfg(test)] mod tests` at the bottom of `agent_coordinator.rs`, add:

```rust
#[test]
fn test_gc_report_has_enhanced_fields() {
    let report = GcReport {
        orphans_rolled_back: 1,
        duration_ms: 150,
        stale_transitioned: 2,
        closed_sessions_cleaned: 3,
        agents_suspended: 0,
        entries_deduplicated: 0,
        started_at: "2026-03-14T00:00:00Z".to_string(),
    };
    assert_eq!(report.orphans_rolled_back, 1);
    assert_eq!(report.duration_ms, 150);
    assert_eq!(report.stale_transitioned, 2);
    assert_eq!(report.closed_sessions_cleaned, 3);
    assert_eq!(report.started_at, "2026-03-14T00:00:00Z");
}
```

- **Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_gc_report_has_enhanced_fields 2>&1 | tail -20`
Expected: FAIL — `GcReport` has no field named `duration_ms`, etc.

- **Step 3: Expand GcReport struct**

In `crates/corvia-kernel/src/agent_coordinator.rs`, replace the existing `GcReport` (lines 27-31).
Note: the existing struct only has `#[derive(Debug, Clone, Default)]` — we add `serde::Serialize, serde::Deserialize`
so the struct can be serialized for the dashboard API. No `use serde` import is needed — the `serde::` prefix
works because `serde` is a workspace dependency.

```rust
/// Report returned after a GC sweep.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GcReport {
    pub orphans_rolled_back: usize,
    pub duration_ms: u64,
    pub stale_transitioned: usize,
    pub closed_sessions_cleaned: usize,
    pub agents_suspended: usize,
    pub entries_deduplicated: usize,
    pub started_at: String,
}
```

- **Step 4: Update gc() method to populate new fields**

In the same file, update the `gc()` method. The method currently starts with `let mut report = GcReport::default();`. Change it to capture stale count, and set `started_at`:

```rust
pub async fn gc(&self) -> Result<GcReport> {
    let mut report = GcReport {
        started_at: chrono::Utc::now().to_rfc3339(),
        ..Default::default()
    };

    // Step 1: Find Active sessions past heartbeat timeout → mark Stale
    let stale_timeout = std::time::Duration::from_secs(self.config.stale_timeout_secs);
    let stale = self.sessions.find_stale(stale_timeout)?;
    for session in &stale {
        if let Err(e) = self.sessions.transition(&session.session_id, SessionState::Stale) {
            warn!(session_id = %session.session_id, error = %e, "gc_stale_transition_failed");
        } else {
            report.stale_transitioned += 1;
        }
    }
    // ... rest of method unchanged ...
```

After the orphan rollback loop, add the `Ok(report)` return as before. The `duration_ms` and `closed_sessions_cleaned` will be populated by the `ops::gc_run` wrapper (Task 2).

- **Step 5: Fix any compile errors from callers**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build --workspace 2>&1 | head -30`

Check that `GcReport::default()` still works (it does, since all new fields are `u64`/`usize`/`String` with Default). Fix any issues.

- **Step 6: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_gc_report_has_enhanced_fields -- --nocapture 2>&1 | tail -10`
Expected: PASS

- **Step 7: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-kernel/src/agent_coordinator.rs
git commit -m "feat(kernel): expand GcReport with duration, stale, cleanup fields"
```

### Task 2: Add timing wrapper and GcHistory in ops.rs

**Files:**
- Modify: `crates/corvia-kernel/src/ops.rs:248-254`

- **Step 1: Write the failing test**

In `ops.rs` `mod tests`, add:

```rust
#[tokio::test]
async fn test_gc_run_populates_timing() {
    let dir = tempfile::tempdir().unwrap();
    let (_store, coord) = setup_coordinator(dir.path()).await;

    let report = gc_run(&coord).await.unwrap();
    // started_at should be populated (already set by gc() from Task 1)
    assert!(!report.started_at.is_empty());
    // duration_ms should be > 0 — this is what gc_run adds
    assert!(report.duration_ms > 0, "gc_run should populate duration_ms");
    assert!(report.duration_ms < 10_000); // less than 10s for an empty GC
}
```

- **Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_gc_run_populates_timing -- --nocapture 2>&1 | tail -10`
Expected: FAIL — `duration_ms` is 0 because the current `gc_run` just delegates to `coordinator.gc()` without timing

- **Step 3: Add timing wrapper in gc_run()**

Replace the `gc_run` function in `ops.rs`:

```rust
/// Run garbage collection sweep with timing.
pub async fn gc_run(coordinator: &AgentCoordinator) -> Result<GcReport> {
    let start = std::time::Instant::now();
    let mut report = coordinator.gc().await?;
    report.duration_ms = start.elapsed().as_millis() as u64;
    if report.started_at.is_empty() {
        report.started_at = chrono::Utc::now().to_rfc3339();
    }
    Ok(report)
}
```

Add `chrono` import if not already present. Check the existing imports — `ops.rs` doesn't currently import chrono. Add: `use chrono::Utc;` (chrono is already a workspace dependency).

- **Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_gc_run_populates_timing -- --nocapture 2>&1 | tail -10`
Expected: PASS

- **Step 5: Add GcHistory type**

Add to `ops.rs` after the gc_run function:

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

/// In-memory ring buffer of recent GC reports.
pub struct GcHistory {
    reports: Mutex<VecDeque<GcReport>>,
    max_size: usize,
}

impl GcHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            reports: Mutex::new(VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn push(&self, report: GcReport) {
        let mut reports = self.reports.lock().unwrap();
        if reports.len() >= self.max_size {
            reports.pop_front();
        }
        reports.push_back(report);
    }

    pub fn last(&self) -> Option<GcReport> {
        self.reports.lock().unwrap().back().cloned()
    }

    pub fn all(&self) -> Vec<GcReport> {
        self.reports.lock().unwrap().iter().cloned().collect()
    }
}
```

- **Step 6: Write test for GcHistory**

```rust
#[test]
fn test_gc_history_ring_buffer() {
    let history = GcHistory::new(3);
    assert!(history.last().is_none());
    assert!(history.all().is_empty());

    for i in 0..5 {
        history.push(GcReport {
            orphans_rolled_back: i,
            duration_ms: i as u64 * 10,
            ..Default::default()
        });
    }

    // Only last 3 should remain
    let all = history.all();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].orphans_rolled_back, 2);
    assert_eq!(all[2].orphans_rolled_back, 4);

    // last() returns most recent
    assert_eq!(history.last().unwrap().orphans_rolled_back, 4);
}
```

- **Step 7: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_gc_history_ring_buffer test_gc_run_populates_timing -- --nocapture 2>&1 | tail -15`
Expected: both PASS

- **Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-kernel/src/ops.rs
git commit -m "feat(kernel): add GcHistory ring buffer and timing wrapper for gc_run"
```

### Task 3: Add GcHistory to AppState

**Files:**
- Modify: `crates/corvia-server/src/rest.rs:19-36`

- **Step 1: Add gc_history field to AppState**

In `rest.rs`, add the import and field:

```rust
// Add to imports at the top
use corvia_kernel::ops::GcHistory;
```

Add to the `AppState` struct after `cluster_store`:

```rust
    /// In-memory GC run history for dashboard.
    pub gc_history: Arc<GcHistory>,
```

- **Step 2: Update AppState construction sites**

`AppState` is constructed in two places in `crates/corvia-server/src/mcp.rs`:

**Site 1 — line ~1391** (production `make_test_state`): Add after `cluster_store`:
```rust
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
```

**Site 2 — line ~1693** (test `setup_mcp_test`): Add after `cluster_store`:
```rust
            gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
```

Also check if there is a third construction site in `main.rs` or `rest.rs`:
```bash
cd /workspaces/corvia-workspace/repos/corvia && grep -rn "AppState {" crates/corvia-server/src/ | head -10
```
Update any additional sites found.

- **Step 3: Build to verify compilation**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build -p corvia-server 2>&1 | tail -20`
Expected: compiles successfully

- **Step 4: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/rest.rs crates/corvia-server/src/*.rs
git commit -m "feat(server): add GcHistory to AppState"
```

### Task 4: Add dashboard response types in corvia-common

**Files:**
- Modify: `crates/corvia-common/src/dashboard.rs`

- **Step 1: Add p50/p95/p99 to SpanStats**

Add three fields to the existing `SpanStats` struct:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SpanStats {
    pub count: u64,
    pub count_1h: u64,
    pub avg_ms: f64,
    pub last_ms: f64,
    pub errors: u64,
    #[serde(default)]
    pub p50_ms: f64,
    #[serde(default)]
    pub p95_ms: f64,
    #[serde(default)]
    pub p99_ms: f64,
}
```

The `#[serde(default)]` ensures backward compatibility with existing JSON that lacks these fields.

**Important:** After adding these fields, the existing `SpanStats` construction in
`crates/corvia-server/src/dashboard/traces.rs` (line ~240) will fail to compile. Add
`..Default::default()` to the existing struct literal so Tasks 5-7 can compile:

```rust
// In traces.rs collect_traces_from_lines(), existing SpanStats construction:
SpanStats {
    count,
    count_1h,
    avg_ms,
    last_ms,
    errors,
    ..Default::default()  // ← add this line for p50/p95/p99
}
```

This is a temporary fix — Task 8 will replace it with actual percentile values.

- **Step 2: Add GC response types**

Add after `TracesResponse`:

```rust
/// GC report (mirrors corvia-kernel GcReport)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcReportDto {
    pub orphans_rolled_back: usize,
    pub duration_ms: u64,
    pub stale_transitioned: usize,
    pub closed_sessions_cleaned: usize,
    pub agents_suspended: usize,
    pub entries_deduplicated: usize,
    pub started_at: String,
}

/// GET /api/dashboard/gc response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcStatusResponse {
    pub last_run: Option<GcReportDto>,
    pub history: Vec<GcReportDto>,
    pub scheduled: bool,
}

/// Live session entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSession {
    pub session_id: String,
    pub agent_id: String,
    pub agent_name: String,
    pub state: String,
    pub started_at: String,
    pub duration_secs: u64,
    pub entries_written: u64,
    pub entries_merged: u64,
    pub pending_entries: u64,
    pub git_branch: Option<String>,
    pub has_staging_dir: bool,
}

/// Live sessions summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionsSummary {
    pub total_active: usize,
    pub total_stale: usize,
    pub total_entries_pending: u64,
}

/// GET /api/dashboard/sessions/live response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionsResponse {
    pub sessions: Vec<LiveSession>,
    pub summary: LiveSessionsSummary,
}

/// A node in a span trace tree.
/// Note: `module` is an intentional enhancement not in the original spec — derived via
/// `span_to_module()` (already exists in traces.rs) so the waterfall UI can color-code
/// bars by module without client-side parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanNode {
    pub span_id: String,
    pub parent_span_id: String,
    pub trace_id: String,
    pub span_name: String,
    pub elapsed_ms: f64,
    pub start_offset_ms: f64,
    pub depth: usize,
    pub module: String,
    pub fields: serde_json::Value,
    pub children: Vec<SpanNode>,
}

/// A complete trace tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTree {
    pub trace_id: String,
    pub root_span: String,
    pub total_ms: f64,
    pub span_count: usize,
    pub started_at: String,
    pub spans: Vec<SpanNode>,
}

/// GET /api/dashboard/traces/recent response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentTracesResponse {
    pub traces: Vec<TraceTree>,
}
```

- **Step 3: Add a test for new SpanStats fields**

```rust
#[test]
fn span_stats_default_includes_percentiles() {
    let stats = SpanStats::default();
    assert_eq!(stats.p50_ms, 0.0);
    assert_eq!(stats.p95_ms, 0.0);
    assert_eq!(stats.p99_ms, 0.0);
}

#[test]
fn gc_status_response_serializes() {
    let resp = GcStatusResponse {
        last_run: None,
        history: vec![],
        scheduled: false,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"scheduled\":false"));
    assert!(json.contains("\"last_run\":null"));
}
```

- **Step 4: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-common 2>&1 | tail -15`
Expected: all PASS

- **Step 5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-common/src/dashboard.rs
git commit -m "feat(common): add GC, live session, trace tree, and percentile response types"
```

---

## Chunk 2: Shared Infrastructure — OtelContextLayer

### Task 5: Create OtelContextLayer

**Files:**
- Create: `crates/corvia-telemetry/src/otel_context_layer.rs`

- **Step 1: Create the OtelContextLayer module**

```rust
//! Custom tracing Layer that bridges OpenTelemetry span context into
//! the fmt JSON output by storing trace_id/span_id in span extensions.

use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Extracted OTEL trace context fields, stored in span extensions.
#[derive(Clone, Debug)]
pub struct OtelFields {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
}

/// A tracing Layer that reads `tracing_opentelemetry::OtelData` from span
/// extensions (populated by the OTEL layer composed earlier) and stores
/// extracted trace/span IDs as `OtelFields` for the fmt layer to read.
///
/// Layer composition order matters: the OTEL layer must be `.with()`-ed
/// before this layer so `OtelData` is in extensions when `on_new_span` fires.
/// `OtelData` is public in tracing-opentelemetry 0.29.
pub struct OtelContextLayer;

impl<S> Layer<S> for OtelContextLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };

        let extensions = span.extensions();

        // tracing-opentelemetry stores OtelData in extensions during on_new_span.
        // OtelData.builder has the span_id and trace_id for this span.
        // OtelData.parent_cx has the parent's span context.
        if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
            let trace_id = otel_data.builder.trace_id
                .map(|t| format!("{t}"))
                .or_else(|| {
                    let parent_sc = otel_data.parent_cx.span().span_context();
                    if parent_sc.is_valid() { Some(format!("{}", parent_sc.trace_id())) } else { None }
                });

            let span_id = otel_data.builder.span_id.map(|s| format!("{s}"));

            let parent_span_id = {
                let parent_sc = otel_data.parent_cx.span().span_context();
                if parent_sc.is_valid() { format!("{}", parent_sc.span_id()) } else { String::new() }
            };

            if let (Some(tid), Some(sid)) = (trace_id, span_id) {
                let fields = OtelFields {
                    trace_id: tid,
                    span_id: sid,
                    parent_span_id,
                };
                drop(extensions);
                span.extensions_mut().insert(fields);
            }
        }
    }
}
```

- **Step 2: Add module declaration in lib.rs**

In `crates/corvia-telemetry/src/lib.rs`, add after `pub mod propagation;`:

```rust
pub mod otel_context_layer;
```

- **Step 3: Build to verify compilation**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build -p corvia-telemetry 2>&1 | tail -20`

- **Step 4: Add unit tests**

Add to the bottom of `otel_context_layer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otel_fields_stores_ids() {
        let fields = OtelFields {
            trace_id: "abcdef1234567890abcdef1234567890".to_string(),
            span_id: "1234567890abcdef".to_string(),
            parent_span_id: String::new(),
        };
        assert_eq!(fields.trace_id.len(), 32);
        assert_eq!(fields.span_id.len(), 16);
        assert!(fields.parent_span_id.is_empty());
    }

    #[test]
    fn otel_context_layer_is_constructible() {
        // Verify the layer can be created (no runtime OTEL needed)
        let _layer = OtelContextLayer;
    }
}
```

Note: Full integration tests (verifying trace_id appears in JSON output) require a running
OTEL collector or mock subscriber, which is beyond unit test scope. The above tests verify
the data structures. Integration testing happens in Task 19.

- **Step 5: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry 2>&1 | tail -15`
Expected: all PASS

- **Step 6: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-telemetry/src/otel_context_layer.rs crates/corvia-telemetry/src/lib.rs
git commit -m "feat(telemetry): add OtelContextLayer bridging OTEL trace context to fmt"
```

### Task 6: Refactor init_telemetry() layer composition

**Files:**
- Modify: `crates/corvia-telemetry/src/lib.rs:95-135`

- **Step 1: Refactor layer composition order**

The current code has `registry.with(env_filter).with(local_layer).with(otel_layer)`. We need to change the order so OTEL is composed before the context layer and fmt layer:

```rust
// In the match on config.exporter:
match config.exporter.as_str() {
    "file" => {
        let file_appender = tracing_appender::rolling::daily("logs", "corvia.log");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        let local_layer = if config.log_format == "json" {
            fmt::layer()
                .json()
                .with_writer(non_blocking)
                .boxed()
        } else {
            fmt::layer()
                .with_writer(non_blocking)
                .boxed()
        };

        let context_layer = if otel_layer.is_some() {
            Some(otel_context_layer::OtelContextLayer)
        } else {
            None
        };

        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .with(context_layer)
            .with(local_layer)
            .init();
    }
    _ => {
        let local_layer = if config.log_format == "json" {
            fmt::layer().json().boxed()
        } else {
            fmt::layer().boxed()
        };

        let context_layer = if otel_layer.is_some() {
            Some(otel_context_layer::OtelContextLayer)
        } else {
            None
        };

        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .with(context_layer)
            .with(local_layer)
            .init();
    }
}
```

- **Step 2: Build and verify**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build -p corvia-telemetry 2>&1 | tail -20`
Expected: compiles. The `Option<OtelContextLayer>` impl works because `Option<L: Layer<S>>` implements `Layer<S>`.

- **Step 3: Run existing telemetry tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry 2>&1 | tail -15`
Expected: all existing tests pass

- **Step 4: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-telemetry/src/lib.rs
git commit -m "refactor(telemetry): reorder layer composition for OTEL context injection"
```

---

## Chunk 3: Backend — Dashboard Endpoints

### Task 7: Add trace context fields to ParsedTrace

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs:63-76`

- **Step 1: Write the failing test**

Add to `traces.rs` tests:

```rust
#[test]
fn parse_span_with_otel_context() {
    let line = r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":12.5,"otel.trace_id":"abcdef1234567890abcdef1234567890","otel.span_id":"1234567890abcdef","otel.parent_span_id":"fedcba0987654321"}"#;
    let result = parse_trace_line(line).unwrap();
    match result {
        ParsedTrace::Span { trace_id, span_id, parent_span_id, .. } => {
            assert_eq!(trace_id.as_deref(), Some("abcdef1234567890abcdef1234567890"));
            assert_eq!(span_id.as_deref(), Some("1234567890abcdef"));
            assert_eq!(parent_span_id.as_deref(), Some("fedcba0987654321"));
        }
        _ => panic!("expected Span variant"),
    }
}

#[test]
fn parse_span_without_otel_context() {
    let line = r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":12.5}"#;
    let result = parse_trace_line(line).unwrap();
    match result {
        ParsedTrace::Span { trace_id, span_id, parent_span_id, .. } => {
            assert!(trace_id.is_none());
            assert!(span_id.is_none());
            assert!(parent_span_id.is_none());
        }
        _ => panic!("expected Span variant"),
    }
}
```

- **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server parse_span_with_otel_context parse_span_without_otel_context 2>&1 | tail -20`
Expected: FAIL — `ParsedTrace::Span` has no field `trace_id`

- **Step 3: Add Optional fields to ParsedTrace::Span**

Update the `ParsedTrace` enum:

```rust
pub enum ParsedTrace {
    Span {
        level: String,
        timestamp: String,
        span_name: String,
        elapsed_ms: f64,
        trace_id: Option<String>,
        span_id: Option<String>,
        parent_span_id: Option<String>,
    },
    Event {
        level: String,
        timestamp: String,
        msg: String,
        target: String,
    },
}
```

- **Step 4: Update parse_trace_line() to extract OTEL fields**

In the `parse_trace_line` function, update the Span return:

```rust
if let Some(elapsed_ms) = v.get("elapsed_ms").and_then(|e| e.as_f64()) {
    let trace_id = v.get("otel.trace_id").and_then(|t| t.as_str()).map(String::from);
    let span_id = v.get("otel.span_id").and_then(|t| t.as_str()).map(String::from);
    let parent_span_id = v.get("otel.parent_span_id").and_then(|t| t.as_str()).map(String::from);
    return Some(ParsedTrace::Span {
        level,
        timestamp,
        span_name: span_name.to_string(),
        elapsed_ms,
        trace_id,
        span_id,
        parent_span_id,
    });
}
```

- **Step 5: Fix all match arms that destructure ParsedTrace::Span**

In `collect_traces_from_lines()` (same file, line ~184), update the match:

```rust
ParsedTrace::Span {
    level,
    timestamp,
    span_name,
    elapsed_ms,
    ..  // ignore trace context fields for aggregation
} => {
```

In `mod.rs` `logs_handler()` (line ~212), update the match:

```rust
traces::ParsedTrace::Span {
    timestamp, level, span_name, elapsed_ms, ..
} => {
```

- **Step 6: Run all tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server 2>&1 | tail -20`
Expected: all PASS

- **Step 7: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): add OTEL trace context fields to ParsedTrace"
```

### Task 8: Add percentile computation to traces.rs

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs:167-261`

- **Step 1: Write the failing test**

```rust
#[test]
fn percentile_computation() {
    let mut durations: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let p50 = compute_percentile(&mut durations, 50.0);
    let p95 = compute_percentile(&mut durations, 95.0);
    let p99 = compute_percentile(&mut durations, 99.0);
    assert!((p50 - 50.0).abs() < 1.5);
    assert!((p95 - 95.0).abs() < 1.5);
    assert!((p99 - 99.0).abs() < 1.5);
}
```

- **Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server percentile_computation 2>&1 | tail -10`
Expected: FAIL — `compute_percentile` not found

- **Step 3: Add compute_percentile function**

Add before `collect_traces_from_lines`:

```rust
/// Compute the p-th percentile from a mutable slice of durations.
/// Sorts the slice in place. Returns 0.0 for empty slices.
pub fn compute_percentile(durations: &mut [f64], percentile: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((percentile / 100.0) * (durations.len() - 1) as f64).round() as usize;
    durations[idx.min(durations.len() - 1)]
}
```

- **Step 4: Update collect_traces_from_lines to compute percentiles**

In the `collect_traces_from_lines` function, update the SpanStats construction to include percentiles. After the `for (name, timings) in &span_all` loop header, clone the timings for percentile computation:

```rust
for (name, timings) in &span_all {
    let count = timings.len() as u64;
    let avg_ms = timings.iter().sum::<f64>() / count as f64;
    let last_ms = *timings.last().unwrap_or(&0.0);
    let count_1h = span_1h.get(name).map(|v| v.len() as u64).unwrap_or(0);
    let errors = span_errors.get(name).copied().unwrap_or(0);

    let mut sorted = timings.clone();
    let p50_ms = compute_percentile(&mut sorted, 50.0);
    let p95_ms = compute_percentile(&mut sorted, 95.0);
    let p99_ms = compute_percentile(&mut sorted, 99.0);

    spans.insert(
        name.clone(),
        SpanStats {
            count,
            count_1h,
            avg_ms,
            last_ms,
            errors,
            p50_ms,
            p95_ms,
            p99_ms,
        },
    );
}
```

- **Step 5: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server 2>&1 | tail -15`
Expected: all PASS

- **Step 6: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs
git commit -m "feat(dashboard): add percentile computation (p50/p95/p99) for span stats"
```

### Task 9: Add trace tree builder to traces.rs

**Files:**
- Modify: `crates/corvia-server/src/dashboard/traces.rs`

- **Step 1: Write the failing test**

```rust
#[test]
fn collect_trace_trees_builds_hierarchy() {
    let lines = vec![
        r#"{"timestamp":"2026-03-14T10:00:00.000Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":50.0,"otel.trace_id":"aaaa","otel.span_id":"1111","otel.parent_span_id":""}"#,
        r#"{"timestamp":"2026-03-14T10:00:00.010Z","level":"INFO","span":{"name":"corvia.entry.embed"},"fields":{},"elapsed_ms":30.0,"otel.trace_id":"aaaa","otel.span_id":"2222","otel.parent_span_id":"1111"}"#,
        r#"{"timestamp":"2026-03-14T10:00:00.020Z","level":"INFO","span":{"name":"corvia.store.insert"},"fields":{},"elapsed_ms":10.0,"otel.trace_id":"aaaa","otel.span_id":"3333","otel.parent_span_id":"1111"}"#,
    ];
    let trees = collect_trace_trees(&lines, 10);
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].trace_id, "aaaa");
    assert_eq!(trees[0].root_span, "corvia.entry.write");
    assert_eq!(trees[0].span_count, 3);
    assert!(trees[0].total_ms >= 50.0);

    // Verify parent-child tree structure
    assert_eq!(trees[0].spans.len(), 1, "should have 1 root span");
    let root = &trees[0].spans[0];
    assert_eq!(root.span_name, "corvia.entry.write");
    assert_eq!(root.depth, 0);
    assert_eq!(root.children.len(), 2, "root should have 2 children");
    assert_eq!(root.children[0].depth, 1);
    assert_eq!(root.children[1].depth, 1);
}

#[test]
fn collect_trace_trees_skips_lines_without_trace_id() {
    let lines = vec![
        r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":10.0}"#,
    ];
    let trees = collect_trace_trees(&lines, 10);
    assert!(trees.is_empty());
}
```

- **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server collect_trace_trees 2>&1 | tail -10`
Expected: FAIL — `collect_trace_trees` not found

- **Step 3: Implement collect_trace_trees**

Add the function and its necessary import for `TraceTree` and `SpanNode` from `corvia_common::dashboard`:

```rust
use corvia_common::dashboard::{SpanStats, TraceEvent, TracesData, SpanNode, TraceTree};
```

Then add the function:

```rust
/// Build trace trees from log lines that have OTEL trace context.
/// Groups spans by trace_id, sorts by timestamp, builds parent-child trees.
/// Returns the most recent `limit` traces.
pub fn collect_trace_trees(lines: &[&str], limit: usize) -> Vec<TraceTree> {
    use std::collections::HashMap;

    // Collect spans with trace context
    struct RawSpan {
        trace_id: String,
        span_id: String,
        parent_span_id: String,
        span_name: String,
        elapsed_ms: f64,
        timestamp: String,
        fields: serde_json::Value,
    }

    let mut spans_by_trace: HashMap<String, Vec<RawSpan>> = HashMap::new();

    for line in lines {
        if let Some(ParsedTrace::Span {
            timestamp, span_name, elapsed_ms,
            trace_id: Some(tid), span_id: Some(sid), parent_span_id, ..
        }) = parse_trace_line(line) {
            spans_by_trace.entry(tid.clone()).or_default().push(RawSpan {
                trace_id: tid,
                span_id: sid,
                parent_span_id: parent_span_id.unwrap_or_default(),
                span_name,
                elapsed_ms,
                timestamp,
                fields: serde_json::Value::Null,
            });
        }
    }

    let mut trees: Vec<TraceTree> = Vec::new();

    for (trace_id, mut trace_spans) in spans_by_trace {
        // Sort by timestamp
        trace_spans.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let root_ts = trace_spans.first().map(|s| &s.timestamp).cloned().unwrap_or_default();

        // Find root span (empty parent_span_id)
        let root_name = trace_spans.iter()
            .find(|s| s.parent_span_id.is_empty())
            .map(|s| s.span_name.clone())
            .unwrap_or_else(|| trace_spans.first().map(|s| s.span_name.clone()).unwrap_or_default());

        let total_ms = trace_spans.iter()
            .find(|s| s.parent_span_id.is_empty())
            .map(|s| s.elapsed_ms)
            .unwrap_or_else(|| trace_spans.iter().map(|s| s.elapsed_ms).fold(0.0f64, f64::max));

        let span_count = trace_spans.len();

        // Build parent-child tree from flat spans.
        // 1. Create all SpanNodes indexed by span_id.
        // 2. Attach children to parents. Orphaned spans become roots.
        let mut node_map: std::collections::HashMap<String, SpanNode> = trace_spans.iter().map(|s| {
            let offset_ms = compute_offset_ms(&root_ts, &s.timestamp);
            (s.span_id.clone(), SpanNode {
                span_id: s.span_id.clone(),
                parent_span_id: s.parent_span_id.clone(),
                trace_id: s.trace_id.clone(),
                span_name: s.span_name.clone(),
                elapsed_ms: s.elapsed_ms,
                start_offset_ms: offset_ms,
                depth: 0,
                module: span_to_module(&s.span_name).to_string(),
                fields: s.fields.clone(),
                children: vec![],
            })
        }).collect();

        // Collect parent→children relationships
        let mut children_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        let mut root_ids: Vec<String> = Vec::new();

        for s in &trace_spans {
            if s.parent_span_id.is_empty() || !node_map.contains_key(&s.parent_span_id) {
                root_ids.push(s.span_id.clone());
            } else {
                children_map.entry(s.parent_span_id.clone()).or_default().push(s.span_id.clone());
            }
        }

        // Recursive function to build tree with depth
        fn build_tree(
            id: &str,
            depth: usize,
            node_map: &mut std::collections::HashMap<String, SpanNode>,
            children_map: &std::collections::HashMap<String, Vec<String>>,
        ) -> Option<SpanNode> {
            let mut node = node_map.remove(id)?;
            node.depth = depth;
            if let Some(child_ids) = children_map.get(id) {
                for cid in child_ids {
                    if let Some(child) = build_tree(cid, depth + 1, node_map, children_map) {
                        node.children.push(child);
                    }
                }
            }
            Some(node)
        }

        let tree_roots: Vec<SpanNode> = root_ids.iter()
            .filter_map(|id| build_tree(id, 0, &mut node_map, &children_map))
            .collect();

        trees.push(TraceTree {
            trace_id,
            root_span: root_name,
            total_ms,
            span_count,
            started_at: root_ts,
            spans: tree_roots,
        });
    }

    // Sort by started_at descending, take limit
    trees.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    trees.truncate(limit);
    trees
}

/// Compute millisecond offset between two RFC 3339 timestamps.
fn compute_offset_ms(base: &str, ts: &str) -> f64 {
    let base_dt = chrono::DateTime::parse_from_rfc3339(base).ok();
    let ts_dt = chrono::DateTime::parse_from_rfc3339(ts).ok();
    match (base_dt, ts_dt) {
        (Some(b), Some(t)) => {
            let diff = t.signed_duration_since(b);
            diff.num_milliseconds() as f64
        }
        _ => 0.0,
    }
}
```

- **Step 4: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server collect_trace_trees 2>&1 | tail -15`
Expected: PASS

- **Step 5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/traces.rs
git commit -m "feat(dashboard): add trace tree builder for OTEL span drill-down"
```

### Task 10: Add `From<GcReport>` impl and 4 new dashboard endpoints

**Files:**
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

- **Step 0: Add conversion helper to eliminate GcReport→GcReportDto duplication**

At the top of `mod.rs` (after imports), add a plain conversion function. We use a function
instead of `From` because Rust's orphan rule forbids `impl From<ForeignType> for OtherForeignType`
in a third crate.

```rust
use corvia_kernel::agent_coordinator::GcReport;

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
```

- **Step 1: Add routes to the router**

In the `router()` function, add before `.with_state(state)`:

```rust
.route("/api/dashboard/gc", get(gc_status_handler))
.route("/api/dashboard/gc/run", post(gc_run_handler))
.route("/api/dashboard/sessions/live", get(live_sessions_handler))
.route("/api/dashboard/traces/recent", get(recent_traces_handler))
```

- **Step 2: Add the GC status handler**

```rust
/// GET /api/dashboard/gc
async fn gc_status_handler(
    State(state): State<Arc<AppState>>,
) -> Json<corvia_common::dashboard::GcStatusResponse> {
    let last_run = state.gc_history.last().map(gc_report_to_dto);
    let history: Vec<corvia_common::dashboard::GcReportDto> = state.gc_history.all()
        .into_iter()
        .map(gc_report_to_dto)
        .collect();

    Json(corvia_common::dashboard::GcStatusResponse {
        last_run,
        history,
        scheduled: false,
    })
}
```

- **Step 3: Add the GC run handler**

```rust
/// POST /api/dashboard/gc/run
async fn gc_run_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::GcReportDto>, (StatusCode, String)> {
    let report = corvia_kernel::ops::gc_run(&state.coordinator).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("GC failed: {e}")))?;

    state.gc_history.push(report.clone());

    Ok(Json(gc_report_to_dto(report)))
}
```

- **Step 4: Add the live sessions handler**

```rust
/// GET /api/dashboard/sessions/live
async fn live_sessions_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<corvia_common::dashboard::LiveSessionsResponse>, (StatusCode, String)> {
    let open_sessions = state.coordinator.sessions.list_open()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list sessions: {e}")))?;

    let agents = corvia_kernel::ops::agents_list(&state.coordinator)
        .unwrap_or_default();
    let agent_names: std::collections::HashMap<String, String> = agents.into_iter()
        .map(|a| (a.agent_id.clone(), a.display_name))
        .collect();

    let now = chrono::Utc::now();
    let mut total_active = 0usize;
    let mut total_stale = 0usize;
    let mut total_entries_pending = 0u64;

    let sessions: Vec<corvia_common::dashboard::LiveSession> = open_sessions.iter().map(|s| {
        let state_str = format!("{:?}", s.state);
        let pending = s.entries_written.saturating_sub(s.entries_merged);
        // s.created_at is DateTime<Utc>, not String — use direct arithmetic
        let duration = (now - s.created_at).num_seconds().max(0) as u64;
        let has_staging = s.staging_dir.as_ref()
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
            agent_name: agent_names.get(&s.agent_id).cloned().unwrap_or_else(|| s.agent_id.clone()),
            state: state_str,
            started_at: s.created_at.to_rfc3339(),
            duration_secs: duration,
            entries_written: s.entries_written,
            entries_merged: s.entries_merged,
            pending_entries: pending,
            git_branch: s.git_branch.clone(),
            has_staging_dir: has_staging,
        }
    }).collect();

    Ok(Json(corvia_common::dashboard::LiveSessionsResponse {
        sessions,
        summary: corvia_common::dashboard::LiveSessionsSummary {
            total_active,
            total_stale,
            total_entries_pending,
        },
    }))
}
```

- **Step 5: Add the recent traces handler**

```rust
/// Query params for /api/dashboard/traces/recent
#[derive(Debug, Deserialize)]
pub struct RecentTracesQuery {
    pub limit: Option<usize>,
}

/// GET /api/dashboard/traces/recent
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
```

- **Step 6: Build to verify**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo build -p corvia-server 2>&1 | tail -20`
Expected: compiles. Fix any import issues (add `use chrono;` if needed, fix `SessionState` path, etc.).

- **Step 7: Run all dashboard tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server 2>&1 | tail -20`
Expected: all PASS

- **Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): add GC, live sessions, and recent traces endpoints"
```

---

## Chunk 4: Frontend — TypeScript Types and API

### Task 11: Add TypeScript types

**Files:**
- Modify: `tools/corvia-dashboard/src/types.ts`

- **Step 1: Add p50/p95/p99 to SpanStats interface**

Update `SpanStats`:

```typescript
export interface SpanStats {
  count: number;
  count_1h: number;
  avg_ms: number;
  last_ms: number;
  errors: number;
  p50_ms?: number;
  p95_ms?: number;
  p99_ms?: number;
}
```

- **Step 2: Add GC types**

Add after `ActivityFeedResponse`:

```typescript
// --- GC types ---

export interface GcReportDto {
  orphans_rolled_back: number;
  duration_ms: number;
  stale_transitioned: number;
  closed_sessions_cleaned: number;
  agents_suspended: number;
  entries_deduplicated: number;
  started_at: string;
}

export interface GcStatusResponse {
  last_run: GcReportDto | null;
  history: GcReportDto[];
  scheduled: boolean;
}

// --- Live session types ---

export interface LiveSession {
  session_id: string;
  agent_id: string;
  agent_name: string;
  state: string;
  started_at: string;
  duration_secs: number;
  entries_written: number;
  entries_merged: number;
  pending_entries: number;
  git_branch: string | null;
  has_staging_dir: boolean;
}

export interface LiveSessionsSummary {
  total_active: number;
  total_stale: number;
  total_entries_pending: number;
}

export interface LiveSessionsResponse {
  sessions: LiveSession[];
  summary: LiveSessionsSummary;
}

// --- Trace tree types ---

export interface SpanNode {
  span_id: string;
  parent_span_id: string;
  trace_id: string;
  span_name: string;
  elapsed_ms: number;
  start_offset_ms: number;
  depth: number;
  module: string;
  fields: Record<string, unknown>;
  children: SpanNode[];
}

export interface TraceTree {
  trace_id: string;
  root_span: string;
  total_ms: number;
  span_count: number;
  started_at: string;
  spans: SpanNode[];
}

export interface RecentTracesResponse {
  traces: TraceTree[];
}
```

- **Step 3: Commit**

Note: `tools/corvia-dashboard/` lives in the workspace root repo, not `repos/corvia/`.

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/types.ts
git commit -m "feat(dashboard-ui): add GC, live session, and trace tree TypeScript types"
```

### Task 12: Add API functions

**Files:**
- Modify: `tools/corvia-dashboard/src/api.ts`

- **Step 1: Add imports**

Update the import block at the top to include new types:

```typescript
import type {
  // ... existing imports ...
  GcStatusResponse,
  GcReportDto,
  LiveSessionsResponse,
  RecentTracesResponse,
} from "./types";
```

- **Step 2: Add 4 new API functions**

Add at the bottom of the file:

```typescript
// --- GC ---

export function fetchGcStatus(): Promise<GcStatusResponse> {
  return get("/gc");
}

export function triggerGcRun(): Promise<GcReportDto> {
  return post("/gc/run", {});
}

// --- Live sessions ---

export function fetchLiveSessions(): Promise<LiveSessionsResponse> {
  return get("/sessions/live");
}

// --- Recent traces ---

export function fetchRecentTraces(limit?: number): Promise<RecentTracesResponse> {
  const qs = limit ? `?limit=${limit}` : "";
  return get(`/traces/recent${qs}`);
}
```

- **Step 3: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/api.ts
git commit -m "feat(dashboard-ui): add GC, live sessions, and recent traces API functions"
```

---

## Chunk 5: Frontend — Components

### Task 13: Create LiveSessionsBar component

**Files:**
- Create: `tools/corvia-dashboard/src/components/LiveSessionsBar.tsx`

- **Step 1: Create the component**

```tsx
import { useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchLiveSessions } from "../api";
import type { LiveSession } from "../types";

const STATE_DOT_COLORS: Record<string, string> = {
  Active: "var(--mint)",
  Stale: "var(--amber)",
  Created: "var(--text-dim)",
  Committing: "var(--gold)",
  Merging: "var(--peach)",
  Orphaned: "var(--coral)",
};

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function SessionPill({ session, onClick }: { session: LiveSession; onClick?: () => void }) {
  const dotColor = STATE_DOT_COLORS[session.state] ?? "var(--text-dim)";
  return (
    <button class="session-pill" onClick={onClick} title={`${session.agent_name} — ${session.state}`}>
      <span class="session-pill-dot" style={{ background: dotColor }} />
      <span class="session-pill-name">{session.agent_name}</span>
      <span class="session-pill-stat">{session.pending_entries} pending</span>
      <span class="session-pill-time">{formatDuration(session.duration_secs)}</span>
    </button>
  );
}

export function LiveSessionsBar({ onSessionClick }: { onSessionClick?: (agentId: string) => void }) {
  const fetcher = useCallback(() => fetchLiveSessions(), []);
  const { data } = usePoll(fetcher, 5000);

  if (!data || data.sessions.length === 0) return null;

  return (
    <div class="live-sessions-bar">
      <div class="live-sessions-header">
        <span class="live-sessions-label">Live Sessions</span>
        <span class="live-sessions-count">
          {data.summary.total_active} active
          {data.summary.total_stale > 0 && ` · ${data.summary.total_stale} stale`}
        </span>
      </div>
      <div class="live-sessions-pills">
        {data.sessions.map((s) => (
          <SessionPill
            key={s.session_id}
            session={s}
            onClick={() => onSessionClick?.(s.agent_id)}
          />
        ))}
      </div>
    </div>
  );
}
```

- **Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/LiveSessionsBar.tsx
git commit -m "feat(dashboard-ui): add LiveSessionsBar component"
```

### Task 14: Create GcPanel component

**Files:**
- Create: `tools/corvia-dashboard/src/components/GcPanel.tsx`

- **Step 1: Create the component**

```tsx
import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchGcStatus, triggerGcRun } from "../api";
import type { GcReportDto, GcStatusResponse } from "../types";

function Sparkline({ history }: { history: GcReportDto[] }) {
  if (history.length === 0) return null;

  const maxDuration = Math.max(...history.map((r) => r.duration_ms), 1);
  const barW = Math.max(4, Math.floor(280 / history.length));
  const h = 60;

  return (
    <svg width={barW * history.length + 4} height={h} class="gc-sparkline">
      {history.map((r, i) => {
        const barH = Math.max(2, (r.duration_ms / maxDuration) * (h - 4));
        const color =
          r.orphans_rolled_back === 0 ? "var(--mint)" :
          r.orphans_rolled_back > 10 ? "var(--coral)" : "var(--amber)";
        return (
          <rect
            key={i}
            x={i * barW + 2}
            y={h - barH - 2}
            width={barW - 2}
            height={barH}
            fill={color}
            rx={1}
          >
            <title>
              {r.started_at}: {r.duration_ms}ms, {r.orphans_rolled_back} orphans
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

function LastRunCard({ report }: { report: GcReportDto }) {
  return (
    <div class="gc-last-run">
      <div class="mini-stats">
        <div class="mini-stat">
          <div class="mini-stat-val">{report.duration_ms}<span style={{ fontSize: "11px" }}>ms</span></div>
          <div class="mini-stat-lbl">Duration</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.orphans_rolled_back}</div>
          <div class="mini-stat-lbl">Orphans</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.stale_transitioned}</div>
          <div class="mini-stat-lbl">Stale</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.closed_sessions_cleaned}</div>
          <div class="mini-stat-lbl">Cleaned</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.agents_suspended}</div>
          <div class="mini-stat-lbl">Suspended</div>
        </div>
      </div>
      <div style={{ fontSize: "11px", color: "var(--text-dim)", marginTop: "6px" }}>
        Last run: {new Date(report.started_at).toLocaleString()}
      </div>
    </div>
  );
}

export function GcPanel() {
  const [running, setRunning] = useState(false);
  const [lastTriggerResult, setLastTriggerResult] = useState<GcReportDto | null>(null);
  const fetcher = useCallback(() => fetchGcStatus(), []);
  const { data } = usePoll(fetcher, 10000);

  const handleRun = useCallback(async () => {
    setRunning(true);
    try {
      const result = await triggerGcRun();
      setLastTriggerResult(result);
    } catch { /* ignore */ }
    setRunning(false);
  }, []);

  if (!data) return null;

  return (
    <>
      <div class="trace-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", margin: 0 }}>
            Garbage Collection
          </h2>
          <span style={{ fontSize: "11px", color: "var(--text-dim)" }}>Manual trigger only</span>
        </div>

        {data.last_run ? (
          <LastRunCard report={data.last_run} />
        ) : (
          <div style={{ fontSize: "12px", color: "var(--text-dim)", padding: "16px 0", textAlign: "center" }}>
            No GC runs yet
          </div>
        )}

        <button
          class="gc-trigger-btn"
          onClick={handleRun}
          disabled={running}
          style={{ marginTop: "12px" }}
        >
          {running ? "Running..." : "Run GC Now"}
        </button>
      </div>

      {data.history.length > 0 && (
        <div class="trace-card">
          <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
            History ({data.history.length} runs)
          </h2>
          <Sparkline history={data.history} />
        </div>
      )}
    </>
  );
}
```

- **Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/GcPanel.tsx
git commit -m "feat(dashboard-ui): add GcPanel component with sparkline and trigger"
```

### Task 15: Create WaterfallView component

**Files:**
- Create: `tools/corvia-dashboard/src/components/WaterfallView.tsx`

- **Step 1: Create the component**

```tsx
import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchRecentTraces } from "../api";
import type { TraceTree, SpanNode } from "../types";

const MODULE_COLORS: Record<string, string> = {
  agent: "var(--peach)",
  entry: "var(--gold)",
  merge: "var(--mint)",
  storage: "var(--lavender)",
  rag: "var(--sky)",
  inference: "var(--coral)",
  gc: "var(--amber)",
  unknown: "var(--text-dim)",
};

function WaterfallBar({
  span,
  totalMs,
  depth,
}: {
  span: SpanNode;
  totalMs: number;
  depth: number;
}) {
  const leftPct = totalMs > 0 ? (span.start_offset_ms / totalMs) * 100 : 0;
  const widthPct = totalMs > 0 ? Math.max(0.5, (span.elapsed_ms / totalMs) * 100) : 1;
  const color = MODULE_COLORS[span.module] ?? MODULE_COLORS.unknown;
  const shortName = span.span_name.replace("corvia.", "");

  return (
    <div class="waterfall-row" style={{ paddingLeft: `${depth * 16}px` }}>
      <div class="waterfall-label" title={span.span_name}>
        {shortName}
      </div>
      <div class="waterfall-track">
        <div
          class="waterfall-bar"
          style={{
            left: `${leftPct}%`,
            width: `${widthPct}%`,
            background: color,
          }}
          title={`${span.span_name}: ${span.elapsed_ms.toFixed(1)}ms`}
        >
          <span class="waterfall-bar-label">
            {span.elapsed_ms.toFixed(1)}ms
          </span>
        </div>
      </div>
    </div>
  );
}

function TraceDetail({ trace }: { trace: TraceTree }) {
  // Render spans as flat list with depth-based indentation
  const renderSpans = (spans: SpanNode[], depth: number) => {
    const elements: any[] = [];
    for (const span of spans) {
      elements.push(
        <WaterfallBar key={span.span_id} span={span} totalMs={trace.total_ms} depth={depth} />
      );
      if (span.children.length > 0) {
        elements.push(...renderSpans(span.children, depth + 1));
      }
    }
    return elements;
  };

  return (
    <div class="waterfall-detail">
      <div class="waterfall-header">
        <span class="waterfall-root">{trace.root_span.replace("corvia.", "")}</span>
        <span class="waterfall-total">{trace.total_ms.toFixed(1)}ms</span>
        <span class="waterfall-count">{trace.span_count} spans</span>
      </div>
      <div class="waterfall-chart">
        {renderSpans(trace.spans, 0)}
      </div>
    </div>
  );
}

export function WaterfallView() {
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const fetcher = useCallback(() => fetchRecentTraces(20), []);
  const { data } = usePoll(fetcher, 10000);

  if (!data || data.traces.length === 0) {
    return (
      <div class="trace-card">
        <div style={{ fontSize: "12px", color: "var(--text-dim)", textAlign: "center", padding: "16px 0" }}>
          Enable OTEL exporter to see trace trees.
        </div>
      </div>
    );
  }

  const active = data.traces.find((t) => t.trace_id === selectedTrace) ?? data.traces[0];

  return (
    <div class="waterfall-view">
      <div class="trace-card">
        <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Recent Traces ({data.traces.length})
        </h2>
        <div class="trace-list">
          {data.traces.map((t) => (
            <button
              key={t.trace_id}
              class={`trace-list-item${t.trace_id === active.trace_id ? " active" : ""}`}
              onClick={() => setSelectedTrace(t.trace_id)}
            >
              <span class="trace-list-name">{t.root_span.replace("corvia.", "")}</span>
              <span class="trace-list-ms">{t.total_ms.toFixed(0)}ms</span>
              <span class="trace-list-count">{t.span_count} spans</span>
            </button>
          ))}
        </div>
      </div>

      <TraceDetail trace={active} />
    </div>
  );
}
```

- **Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/WaterfallView.tsx
git commit -m "feat(dashboard-ui): add WaterfallView component for OTEL span drill-down"
```

### Task 16: Integrate components into TracesView and AgentsView

**Files:**
- Modify: `tools/corvia-dashboard/src/components/TracesView.tsx`
- Modify: `tools/corvia-dashboard/src/components/AgentsView.tsx`

- **Step 1: Integrate GcPanel into TracesView**

In `TracesView.tsx`, add import:

```typescript
import { GcPanel } from "./GcPanel";
import { WaterfallView } from "./WaterfallView";
```

In the `DetailPanel` usage (around line 395-408), replace the `selectedModule` conditional with:

```tsx
{!selectedModule ? (
  <div class="trace-card">
    <div class="trace-empty">Select a module to inspect its telemetry</div>
  </div>
) : selectedModule === "gc" ? (
  <GcPanel />
) : (
  <DetailPanel
    moduleId={selectedModule}
    modDef={MODULES[selectedModule]}
    modStats={ms(selectedModule)}
    spans={data.spans}
    events={data.recent_events}
    onNavigate={onNavigate}
  />
)}
```

Inside the `traces-workspace` div, after the `trace-detail` div closes (line 409), add the waterfall view as a sibling:

```tsx
{/* Waterfall drill-down — inside traces-workspace, after trace-detail */}
<div class="trace-waterfall-section">
  <WaterfallView />
</div>
```

- **Step 2: Update span rows with percentiles**

In `DetailPanel`, add a percentile line inside each span row. Find the existing block (around line 303) that renders `{fields && <div class="span-fields">{fields}</div>}` and add after it:

```tsx
{(stats.p50_ms ?? 0) > 0 && (
  <div class="span-percentiles">
    p50: {stats.p50_ms?.toFixed(0)}ms · p95: {stats.p95_ms?.toFixed(0)}ms · p99: {stats.p99_ms?.toFixed(0)}ms
  </div>
)}
```

This adds a new `<div>` inside the existing `<div>` wrapper, after the fields line. Do NOT replace the surrounding code — only insert this block.

- **Step 3: Integrate LiveSessionsBar into AgentsView**

In `AgentsView.tsx`, add import:

```typescript
import { LiveSessionsBar } from "./LiveSessionsBar";
```

Add `LiveSessionsBar` at the top of the `AgentsView` return, just inside the `agents-view` div:

```tsx
return (
  <div class="agents-view">
    <LiveSessionsBar onSessionClick={(agentId) => setExpandedId(agentId)} />

    <div class="agents-header">
      {/* ... existing ... */}
```

- **Step 4: Build the dashboard to verify**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dashboard && npm run build 2>&1 | tail -20`
Expected: builds successfully (or note TypeScript errors and fix them)

- **Step 5: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/TracesView.tsx tools/corvia-dashboard/src/components/AgentsView.tsx
git commit -m "feat(dashboard-ui): integrate GcPanel, WaterfallView, LiveSessionsBar into dashboard"
```

---

## Chunk 6: Verification and Finalization

### Task 17: Run all backend tests

- **Step 1: Run full Rust test suite**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test --workspace 2>&1 | tail -30`
Expected: all tests pass. Fix any compilation errors or test failures.

- **Step 2: Run the dashboard build**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dashboard && npm run build 2>&1 | tail -20`
Expected: clean build

### Task 18: Update documentation

**Files:**
- Modify: `repos/corvia/AGENTS.md`
- Modify: `repos/corvia/CHANGELOG.md`

- **Step 1: Update AGENTS.md key files**

Add to the Key Files section in `repos/corvia/AGENTS.md`:

```
- `crates/corvia-telemetry/src/otel_context_layer.rs` — Custom OTEL→fmt bridge layer
```

- **Step 2: Add CHANGELOG entry**

Add a new section at the top of `CHANGELOG.md` for the new features.

- **Step 3: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add AGENTS.md CHANGELOG.md
git commit -m "docs: update AGENTS.md and CHANGELOG for dashboard completion features"
```

### Task 19: Final integration test

- **Step 1: Start the server and verify endpoints**

```bash
cd /workspaces/corvia-workspace && corvia serve &
sleep 3
curl -s http://localhost:8020/api/dashboard/gc | python3 -m json.tool
curl -s http://localhost:8020/api/dashboard/sessions/live | python3 -m json.tool
curl -s http://localhost:8020/api/dashboard/traces/recent | python3 -m json.tool
```

Expected: all 3 endpoints return valid JSON with the expected structure.

- **Step 2: Trigger GC and verify**

```bash
curl -s -X POST http://localhost:8020/api/dashboard/gc/run | python3 -m json.tool
```

Expected: returns a GcReportDto with `duration_ms`, `started_at`, etc.

- **Step 3: Verify GC history updates**

```bash
curl -s http://localhost:8020/api/dashboard/gc | python3 -m json.tool
```

Expected: `last_run` is now populated, `history` has 1 entry.
