# Dashboard Completion — Features 7, 9, 10

> **Status:** Shipped (v0.4.3)

> Design spec for the three remaining dashboard roadmap features:
> Live Session State (#7), OTEL Span Drill-Down (#9), GC Operations Dashboard (#10).

**Goal:** Complete the dashboard roadmap (10/10 features) with proper OTEL trace context,
live session monitoring, and GC operations visibility.

**Scope:** Roadmap-faithful — full feature depth, not minimal stubs.

**Build order:** Shared infrastructure first (trace context layer + enhanced GcReport),
then Live Sessions, GC Dashboard, Span Drill-Down.

**Decisions locked in:**
- Real OTEL trace context via custom tracing Layer (not shortcut/approximation)
- Poll-based live sessions (consistent with dashboard architecture, no WebSocket)
- Enhanced GcReport with 6 new fields + in-memory history ring
- Shared infrastructure first, then features

---

## Section 1: Trace Context in Logs — OtelContextLayer

### Problem

The JSON fmt layer and OpenTelemetry layer in `corvia-telemetry` are composed
independently on the tracing registry. Neither injects the other's context. JSON
log lines have no `trace_id`, `span_id`, or `parent_span_id` — making it impossible
to correlate log lines into OTEL trace trees.

### Solution

A custom `tracing::Layer` — `OtelContextLayer` — that bridges OpenTelemetry span
context into the fmt JSON output.

**Layer composition order:**
```
registry()
  .with(env_filter)
  .with(otel_layer)           // populates OtelContext in extensions
  .with(otel_context_layer)   // reads OtelContext, injects trace_id/span_id
  .with(fmt_json_layer)       // emits JSON with trace context fields
```

**Fields injected into each JSON log line (only when OTEL is active):**
- `otel.trace_id` — 32-hex OTEL trace ID
- `otel.span_id` — 16-hex OTEL span ID
- `otel.parent_span_id` — 16-hex parent span ID (empty string at root)

### Design Points

1. **Conditional** — when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, the layer is a
   no-op. No trace context fields appear. Zero overhead.

2. **Separate from RAG trace_id** — the RAG pipeline's `trace_id: Uuid::now_v7()` is
   a business-logic correlation ID. Both coexist without conflict.

3. **New file** — `crates/corvia-telemetry/src/otel_context_layer.rs` (~120 lines).
   Existing `lib.rs` requires a refactor of `init_telemetry()`: the current code
   uses `.boxed()` layers with a match on `config.exporter` for stdout vs file.
   The refactor restructures composition so the OTEL layer is composed before the
   fmt layer, with `OtelContextLayer` inserted between them. All code paths
   (stdout, file, no-exporter) must be updated. This is more than "one extra `.with()`"
   — expect ~40 lines changed in `lib.rs`.

4. **Implementation mechanism** — reads `tracing_opentelemetry::OtelData` from span
   extensions to extract `SpanContext`. Stores extracted IDs as a custom `OtelFields`
   struct in extensions. A custom `FormatEvent` wrapper reads these when formatting.

5. **No new dependencies** — uses `tracing`, `tracing-subscriber`, `tracing-opentelemetry`,
   and `opentelemetry` already in the dependency tree.

6. **Tests** — unit tests with a mock subscriber verifying trace_id/span_id appear in
   JSON output when OTEL is configured, and are absent when it's not.

### Dashboard Consumption

`traces.rs` parser gains 3 new fields on `ParsedTrace::Span`: `trace_id: Option<String>`,
`span_id: Option<String>`, `parent_span_id: Option<String>`. These are `Option` because
log lines predating the `OtelContextLayer` (or emitted without OTEL) will not have them.
The parser handles both formats gracefully. The Span Drill-Down UI uses these to build
parent-child trees when present.

---

## Section 2: Enhanced GcReport

### Problem

`GcReport` only has `orphans_rolled_back: usize`. The GC Operations Dashboard needs
richer data to show what happened during a GC run.

### Solution

Expand `GcReport` in `corvia-common` and populate from `coordinator.gc()`.

```rust
pub struct GcReport {
    pub orphans_rolled_back: usize,       // existing
    pub duration_ms: u64,                 // wall-clock time of GC run
    pub stale_transitioned: usize,        // sessions moved Active→Stale
    pub closed_sessions_cleaned: usize,   // Closed sessions with staging dirs removed
    pub agents_suspended: usize,          // agents auto-suspended (no activity)
    pub entries_deduplicated: usize,      // duplicate entries merged
    pub started_at: String,               // ISO 8601 timestamp
}
```

**Field population:**
- `duration_ms` — `Instant::now()` before/after in `ops::gc_run()`
- `stale_transitioned` — count from `SessionManager` timeout sweep
- `closed_sessions_cleaned` — staging dirs removed for Closed sessions
- `agents_suspended` — count from agent coordinator inactivity check
- `entries_deduplicated` — placeholder (0) for future dedup pass
- `started_at` — `Utc::now().to_rfc3339()` at GC start

### GC History

`GcHistory` in `ops.rs` — `Arc<Mutex<VecDeque<GcReport>>>` (max 50 entries, in-memory).
Added as a field on `AppState` in `crates/corvia-server/src/rest.rs`.

### Endpoints

**New:** `GET /api/dashboard/gc`
```json
{
  "last_run": null,
  "history": [],
  "scheduled": false
}
```
`last_run` is `null` when no GC has ever run. `scheduled` is `false` (GC is manual-only;
no background timer exists). If a periodic GC timer is added later, this field becomes
`true` with an additional `next_scheduled_secs` field.

**New:** `POST /api/dashboard/gc/run` — triggers GC and returns the full `GcReport`.
(This is a new REST endpoint; GC was previously only exposed via the `corvia_gc_run` MCP tool.)

---

## Section 3: Live Session State

### Problem

The Agents tab shows agent cards but no real-time view of active sessions.

### Solution

Poll `SessionManager::list_open()` + staging dir reads via new endpoint.

**New endpoint:** `GET /api/dashboard/sessions/live`
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "agent_id": "uuid",
      "agent_name": "claude-code-abc",
      "state": "Active",
      "started_at": "2026-03-13T...",
      "duration_secs": 342,
      "entries_written": 5,
      "entries_merged": 2,
      "pending_entries": 3,
      "git_branch": "staging/agent-abc/session-xyz",
      "has_staging_dir": true
    }
  ],
  "summary": {
    "total_active": 3,
    "total_stale": 1,
    "total_entries_pending": 12
  }
}
```

**Data sources:**
- `SessionManager::list_open()` — non-Closed sessions with `SessionRecord` fields
- `pending_entries` — `entries_written - entries_merged` (derived)
- `duration_secs` — computed from `started_at` to now
- `agent_name` — joined from `ops::agents_list()` (calls `coordinator.registry.list_all()`)
- `has_staging_dir` — `Path::exists()` check on staging_dir

### Frontend — LiveSessionsBar

Horizontal bar at top of Agents tab with active session pills. Each pill shows:
agent name, state dot (green=Active, yellow=Stale), entry count, duration.
Clicking a pill scrolls to that agent's card and expands its session timeline.

**Polling:** `usePoll` at 5-second interval (consistent with dashboard architecture).

---

## Section 4: GC Operations Dashboard

### Problem

GC is a black box — users can trigger it but can't see results or trends.

### Solution

`GcPanel` in TracesView replaces the generic DetailPanel when the "gc" module is selected.

### GcPanel Contents

1. **Last Run Summary** — card with duration_ms, orphans_rolled_back,
   stale_transitioned, closed_sessions_cleaned, agents_suspended, started_at

2. **History Sparkline** — SVG bar chart of last 50 runs. Bar height = orphans_rolled_back,
   x-axis = time. Hover shows full report. Color-coded: green (0 orphans),
   yellow (some cleanup), red (>10 orphans).

3. **Trigger Button** — "Run GC Now" POSTs to `/api/dashboard/gc/run`,
   refreshes panel with new report.

4. **Status indicator** — shows "Manual trigger only" (no periodic GC configured).
   If a periodic GC timer is added later, this becomes a countdown.

**Empty state:** When no GC has ever run (`last_run: null`, `history: []`), the panel
shows a "No GC runs yet" message with the trigger button prominently displayed.

### Implementation

New file: `tools/corvia-dashboard/src/components/GcPanel.tsx` (~150 lines).
Imported by `TracesView.tsx`, rendered conditionally for "gc" module.
Sparkline uses SVG `<rect>` (no charting library — consistent with edge animations).

---

## Section 5: OTEL Span Drill-Down

### Problem

Traces tab shows flat span statistics but no way to explore individual traces
as parent-child trees.

### Solution

Trace tree reconstruction in `traces.rs` + waterfall UI.

### Backend — Trace Tree Parsing

```rust
pub struct SpanNode {
    pub span_id: String,
    pub parent_span_id: String,
    pub trace_id: String,
    pub span_name: String,
    pub elapsed_ms: f64,
    pub start_offset_ms: f64,  // relative to trace root
    pub depth: usize,
    pub fields: serde_json::Value,
    pub children: Vec<SpanNode>,
}
```

`collect_trace_trees(lines) -> Vec<TraceTree>` groups spans by `trace_id`, sorts by
timestamp, builds parent-child trees via `parent_span_id` linkage. Returns most
recent N traces (default 20).

**`start_offset_ms` computation:** Each log line has an RFC 3339 `timestamp` field.
For each trace group, find the earliest timestamp (root span) and compute
`start_offset_ms` as the delta from root to each span's timestamp in milliseconds.
Log lines may arrive out of order; the algorithm sorts by timestamp before building
the tree.

**New endpoint:** `GET /api/dashboard/traces/recent?limit=20`
```json
{
  "traces": [
    {
      "trace_id": "abc123...",
      "root_span": "kernel.write",
      "total_ms": 145.2,
      "span_count": 8,
      "started_at": "2026-03-13T...",
      "spans": [ ...SpanNode tree... ]
    }
  ]
}
```

### Frontend — Waterfall View

Triggered by clicking a span row or "Recent Traces" button:

1. **Trace list** — left column: root span name, total duration, span count, timestamp
2. **Waterfall chart** — nested horizontal bars:
   - X-axis = time (0ms to total_ms)
   - Each bar = one span at `start_offset_ms`, width = `elapsed_ms`
   - Indented by `depth` for parent-child nesting
   - Color-coded by module (same palette as topology nodes)
   - Hover shows span name, duration, fields
3. **Percentile indicators** — p50/p95/p99 vertical lines overlaid

### SpanStats Enhancement

```rust
pub struct SpanStats {
    pub count: usize,        // existing
    pub count_1h: usize,     // existing
    pub avg_ms: f64,         // existing
    pub last_ms: f64,        // existing
    pub errors: usize,       // existing
    pub p50_ms: f64,         // new
    pub p95_ms: f64,         // new
    pub p99_ms: f64,         // new
}
```

Percentiles from rolling window of last 1000 span durations per span name,
stored in `TracesState` (in-memory `VecDeque<f64>` per span). Capped at 50 tracked
span names (matching the ~19 defined constants in `corvia-telemetry` plus headroom).
Spans beyond the cap are not tracked for percentiles.

### Graceful Degradation

When OTEL is not configured (no trace_id in logs), the waterfall section shows
"Enable OTEL exporter to see trace trees." Existing flat span stats remain fully
functional. Percentiles work regardless — they only need `elapsed_ms`.

New file: `tools/corvia-dashboard/src/components/WaterfallView.tsx` (~200 lines).

---

## New Files Summary

| File | Purpose | ~Lines |
|------|---------|--------|
| `crates/corvia-telemetry/src/otel_context_layer.rs` | Custom Layer bridging OTEL→fmt | 120 |
| `tools/corvia-dashboard/src/components/GcPanel.tsx` | GC operations panel | 150 |
| `tools/corvia-dashboard/src/components/WaterfallView.tsx` | Span drill-down waterfall | 200 |
| `tools/corvia-dashboard/src/components/LiveSessionsBar.tsx` | Active session pills bar | 80 |

## Modified Files Summary

| File | Changes |
|------|---------|
| `crates/corvia-telemetry/src/lib.rs` | Add `mod otel_context_layer`, compose new layer |
| `crates/corvia-common/src/dashboard.rs` | Enhanced `SpanStats` (p50/p95/p99), `GcReport` (6 new fields), new response types |
| `crates/corvia-kernel/src/ops.rs` | `GcHistory`, populate enhanced `GcReport`, timing |
| `crates/corvia-server/src/dashboard/mod.rs` | 4 new routes + handlers (gc GET, gc/run POST, sessions/live, traces/recent) |
| `crates/corvia-server/src/dashboard/traces.rs` | Trace tree parsing, `SpanNode`, `collect_trace_trees()`, percentile computation |
| `tools/corvia-dashboard/src/api.ts` | 3 new fetch functions |
| `tools/corvia-dashboard/src/types.ts` | New TypeScript interfaces |
| `tools/corvia-dashboard/src/components/TracesView.tsx` | Import GcPanel + WaterfallView, percentile display |
| `tools/corvia-dashboard/src/components/AgentsView.tsx` | Import LiveSessionsBar |
| `crates/corvia-server/src/rest.rs` | Add `GcHistory` to `AppState` |

## New Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/dashboard/gc` | GC history + last run + scheduled status |
| POST | `/api/dashboard/gc/run` | Trigger GC, return full GcReport |
| GET | `/api/dashboard/sessions/live` | Active sessions with staging state |
| GET | `/api/dashboard/traces/recent?limit=N` | Recent traces as parent-child trees |
