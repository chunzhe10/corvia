# Telemetry Traces Page Implementation Plan

> **Status:** Shipped (v0.3.7)

**Goal:** Add an interactive telemetry module map as the "Traces" view tab in the Corvia VS Code dashboard, with Map, Data Flow, and Heat rendering modes.

**Architecture:** A new `traces.py` module in corvia-dev parses JSON tracing output from service logs and aggregates span timings into the status JSON. The extension.js Traces tab renders a 7-node topology graph (left) with a reactive detail panel (right), using absolute-positioned divs and inline SVG.

**Tech Stack:** Python (corvia-dev CLI), JavaScript/HTML/CSS (VS Code webview), SVG (edges + animations)

**Spec:** `2026-03-10-telemetry-traces-page-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `tools/corvia-dev/corvia_dev/traces.py` | Parse JSON tracing logs, aggregate span timings, collect recent events |
| `tools/corvia-dev/tests/test_traces.py` | Tests for trace parsing and aggregation |

### Modified files

| File | Changes |
|------|---------|
| `tools/corvia-dev/corvia_dev/models.py` | Add `SpanStats`, `TraceEvent`, `TracesData` models + `traces` field on `StatusResponse` |
| `tools/corvia-dev/corvia_dev/manager.py` | Call `collect_traces()` in `write_state()` to populate traces field |
| `tools/corvia-dev/corvia_dev/cli.py` | Call `collect_traces()` in fallback status path (no manager) |
| `.devcontainer/extensions/corvia-services/extension.js` | Add Traces tab rendering: CSS, graph canvas, detail panel, 3 modes, node interactions |

---

## Chunk 1: Backend — Trace Data Collection

### Task 1: Add trace models to `models.py`

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/models.py`

-**Step 1: Add SpanStats, TraceEvent, TracesData models**

Add after the `StatusResponse` class:

```python
class SpanStats(BaseModel):
    """Aggregated stats for a single tracing span."""
    count: int = 0
    count_1h: int = 0
    avg_ms: float = 0.0
    last_ms: float = 0.0
    errors: int = 0


class TraceEvent(BaseModel):
    """A single structured log event."""
    ts: str
    level: str
    module: str
    msg: str


class TracesData(BaseModel):
    """Aggregated tracing data for the dashboard."""
    spans: dict[str, SpanStats] = {}
    recent_events: list[TraceEvent] = []
```

-**Step 2: Add `traces` field to `StatusResponse`**

Add to the `StatusResponse` class:

```python
class StatusResponse(BaseModel):
    """Full status response -- the JSON contract for the VS Code extension."""
    manager: ManagerStatus | None = None
    services: list[ServiceStatus] = []
    config: ConfigSummary
    enabled_services: list[str] = []
    logs: list[str] = []
    service_logs: dict[str, list[str]] = {}
    stale_binaries: list[str] = []
    traces: TracesData | None = None
```

-**Step 3: Verify models load**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -c "from corvia_dev.models import TracesData, SpanStats, TraceEvent; print('OK')"`
Expected: `OK`

-**Step 4: Commit**

```bash
git add tools/corvia-dev/corvia_dev/models.py
git commit -m "feat(corvia-dev): add trace data models (SpanStats, TracesData)"
```

---

### Task 2: Create `traces.py` — log parser and aggregator

**Files:**
- Create: `tools/corvia-dev/corvia_dev/traces.py`
- Create: `tools/corvia-dev/tests/test_traces.py`

-**Step 1: Write tests for trace parsing**

Create `tools/corvia-dev/tests/test_traces.py`:

```python
"""Tests for trace log parsing and aggregation."""

from corvia_dev.traces import parse_trace_line, collect_traces_from_lines, SPAN_TO_MODULE


def test_parse_json_span_line():
    """Parse a JSON tracing line with span timing."""
    line = '{"timestamp":"2026-03-10T14:31:52","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{"session_id":"s1"},"elapsed_ms":12}'
    result = parse_trace_line(line)
    assert result is not None
    assert result["span"] == "corvia.entry.write"
    assert result["elapsed_ms"] == 12
    assert result["level"] == "INFO"


def test_parse_json_event_line():
    """Parse a JSON tracing event (no span timing)."""
    line = '{"timestamp":"2026-03-10T14:31:52","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}'
    result = parse_trace_line(line)
    assert result is not None
    assert result["level"] == "WARN"
    assert "Slow embed" in result["msg"]


def test_parse_non_json_line_returns_none():
    """Non-JSON lines return None."""
    result = parse_trace_line("INFO some plain text log")
    assert result is None


def test_parse_empty_line():
    result = parse_trace_line("")
    assert result is None


def test_span_to_module_mapping():
    """Verify specific-first matching: entry.embed -> inference."""
    assert SPAN_TO_MODULE["corvia.entry.embed"] == "inference"
    # entry.write should NOT be in the specific map, falls to prefix
    assert "corvia.entry.write" not in SPAN_TO_MODULE


def test_collect_traces_from_lines():
    """Full aggregation from a set of log lines."""
    lines = [
        '{"timestamp":"2026-03-10T14:00:00","level":"INFO","span":{"name":"corvia.entry.write"},"elapsed_ms":10}',
        '{"timestamp":"2026-03-10T14:00:01","level":"INFO","span":{"name":"corvia.entry.write"},"elapsed_ms":14}',
        '{"timestamp":"2026-03-10T14:00:02","level":"INFO","span":{"name":"corvia.entry.embed"},"elapsed_ms":80}',
        '{"timestamp":"2026-03-10T14:00:03","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}',
        'not a json line',
    ]
    traces = collect_traces_from_lines(lines)
    assert "corvia.entry.write" in traces.spans
    assert traces.spans["corvia.entry.write"].count == 2
    assert traces.spans["corvia.entry.write"].avg_ms == 12.0
    assert traces.spans["corvia.entry.write"].last_ms == 14.0
    assert traces.spans["corvia.entry.embed"].count == 1
    assert len(traces.recent_events) == 1  # only the WARN event (non-span)
    assert traces.recent_events[0].level == "warn"


def test_collect_traces_empty():
    traces = collect_traces_from_lines([])
    assert traces.spans == {}
    assert traces.recent_events == []
```

-**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_traces.py -v`
Expected: FAIL (module not found)

-**Step 3: Implement `traces.py`**

Create `tools/corvia-dev/corvia_dev/traces.py`:

```python
"""Parse JSON tracing output and aggregate span timings."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from corvia_dev.models import SpanStats, TraceEvent, TracesData


# Specific span → module overrides (checked before prefix matching)
SPAN_TO_MODULE: dict[str, str] = {
    "corvia.entry.embed": "inference",
}

# Prefix → module mapping (evaluated in order)
_PREFIX_MAP: list[tuple[str, str]] = [
    ("corvia.agent.", "agent"),
    ("corvia.session.", "agent"),
    ("corvia.entry.", "entry"),
    ("corvia.merge.", "merge"),
    ("corvia.store.", "storage"),
    ("corvia.rag.", "rag"),
    ("corvia.gc.", "gc"),
]

# Target (Rust module path) → module mapping for events without span names
_TARGET_MAP: list[tuple[str, str]] = [
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
]


def span_to_module(span_name: str) -> str:
    """Map a span name to its module."""
    if span_name in SPAN_TO_MODULE:
        return SPAN_TO_MODULE[span_name]
    for prefix, module in _PREFIX_MAP:
        if span_name.startswith(prefix):
            return module
    return "unknown"


def target_to_module(target: str) -> str:
    """Map a Rust target path to a module name."""
    for pattern, module in _TARGET_MAP:
        if pattern in target:
            return module
    return "unknown"


def parse_trace_line(line: str) -> dict | None:
    """Parse a single JSON tracing line. Returns dict or None."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    result: dict = {}
    result["level"] = obj.get("level", "INFO")
    result["timestamp"] = obj.get("timestamp", "")

    # Span with timing
    span = obj.get("span", {})
    span_name = span.get("name") if isinstance(span, dict) else None
    elapsed = obj.get("elapsed_ms")

    if span_name and elapsed is not None:
        result["span"] = span_name
        result["elapsed_ms"] = float(elapsed)
        result["fields"] = obj.get("fields", {})
        return result

    # Structured event (no span timing)
    fields = obj.get("fields", {})
    msg = fields.get("message", "")
    target = obj.get("target", "")
    if msg or target:
        result["msg"] = msg
        result["target"] = target
        result["module"] = target_to_module(target)
        return result

    return None


def collect_traces_from_lines(lines: list[str]) -> TracesData:
    """Aggregate span stats and collect recent events from log lines."""
    span_totals: dict[str, list[float]] = {}
    events: list[TraceEvent] = []
    now = time.time()
    one_hour_ago = now - 3600

    for line in lines:
        parsed = parse_trace_line(line)
        if parsed is None:
            continue

        ts_str = parsed.get("timestamp", "")
        ts_short = ""
        ts_epoch = now  # default to now if unparseable
        if ts_str:
            ts_short = ts_str.split("T")[1][:8] if "T" in ts_str else ts_str[:8]
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_epoch = dt.timestamp()
            except (ValueError, OSError):
                pass

        if "span" in parsed:
            span_name = parsed["span"]
            elapsed = parsed["elapsed_ms"]
            if span_name not in span_totals:
                span_totals[span_name] = []
            span_totals[span_name].append(elapsed)
        elif "msg" in parsed and parsed["msg"]:
            level = parsed["level"].lower()
            if level in ("warn", "warning"):
                level = "warn"
            elif level in ("error", "err"):
                level = "error"
            elif level == "debug":
                level = "debug"
            else:
                level = "info"
            module = parsed.get("module", "unknown")
            events.append(TraceEvent(
                ts=ts_short,
                level=level,
                module=module,
                msg=parsed["msg"],
            ))

    # Build SpanStats
    spans: dict[str, SpanStats] = {}
    for name, timings in span_totals.items():
        count = len(timings)
        avg = sum(timings) / count if count else 0
        spans[name] = SpanStats(
            count=count,
            count_1h=count,  # all lines are within the rolling window
            avg_ms=round(avg, 1),
            last_ms=round(timings[-1], 1) if timings else 0,
            errors=0,
        )

    # Keep last 50 events
    recent = events[-50:]

    return TracesData(spans=spans, recent_events=recent)


def collect_traces(log_dir: Path) -> TracesData:
    """Collect traces from all service log files in log_dir."""
    all_lines: list[str] = []
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            try:
                text = log_file.read_text(errors="replace")
                lines = text.strip().split("\n")
                # Take last 500 lines per file to bound memory
                all_lines.extend(lines[-500:])
            except OSError:
                continue
    return collect_traces_from_lines(all_lines)
```

-**Step 4: Run tests to verify they pass**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_traces.py -v`
Expected: All 7 tests PASS

-**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/traces.py tools/corvia-dev/tests/test_traces.py
git commit -m "feat(corvia-dev): add trace log parser and aggregator"
```

---

### Task 3: Wire traces into status output

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/manager.py:237-251`
- Modify: `tools/corvia-dev/corvia_dev/cli.py:111-123`

-**Step 1: Add traces to manager's `write_state()`**

In `manager.py`, add import at top:

```python
from corvia_dev.traces import collect_traces
```

In `write_state()` method (around line 237), add traces collection before writing:

```python
    def write_state(self) -> None:
        """Write current state to the state file as JSON."""
        resp = StatusResponse(
            manager=ManagerStatus(
                pid=os.getpid(),
                uptime_s=round(time.time() - self._started_at, 1),
                state="running" if self._running else "stopped",
            ),
            services=[mp.to_status() for mp in self.processes.values()],
            config=self.config_summary,
            enabled_services=self.enabled_services,
            logs=self._log_lines[-20:],
            service_logs={name: _tail_log(name) for name in self.processes},
            traces=collect_traces(LOG_DIR),
        )
        self.state_path.write_text(resp.model_dump_json(indent=2))
```

-**Step 2: Add traces to CLI fallback status path**

In `cli.py`, add import at top:

```python
from corvia_dev.traces import collect_traces
from corvia_dev.manager import LOG_DIR
```

Note: `LOG_DIR` is already imported on line 22. Just add the traces import.

In the fallback status construction (around line 111), add traces:

```python
    resp = StatusResponse(
        manager=None,
        services=service_statuses,
        config=ConfigSummary(
            embedding_provider=cfg.embedding_provider,
            merge_provider=cfg.merge_provider,
            storage=cfg.storage,
            workspace=cfg.workspace_name,
        ),
        enabled_services=enabled,
        logs=[],
        service_logs=svc_logs,
        traces=collect_traces(LOG_DIR),
    )
```

-**Step 3: Verify JSON output includes traces**

Run: `cd /workspaces/corvia-workspace && corvia-dev status --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('traces' in d, type(d.get('traces')))"`
Expected: `True <class 'dict'>` (or `True <class 'NoneType'>` if no logs yet — both are valid)

-**Step 4: Commit**

```bash
git add tools/corvia-dev/corvia_dev/manager.py tools/corvia-dev/corvia_dev/cli.py
git commit -m "feat(corvia-dev): wire trace collection into status JSON output"
```

---

## Chunk 2: Frontend — CSS and Graph Structure

### Task 4: Add Traces CSS to extension.js

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js`

-**Step 1: Add `--sky` color tokens**

In the `:root` CSS block, after the `--amber-medium` line, add:

```css
  --sky: #7dd3fc;
  --sky-soft: rgba(125, 211, 252, 0.10);
  --sky-medium: rgba(125, 211, 252, 0.16);
```

-**Step 2: Add graph panel CSS**

After the `/* ===== Responsive ===== */` media query closing brace, add the Traces CSS block:

```css
/* ===== Traces ===== */
.traces-workspace {
  display: grid; grid-template-columns: 1fr 280px;
  gap: 16px; padding: 0 28px 28px;
  height: calc(100vh - 260px); min-height: 400px;
  margin-top: 16px;
}
.graph-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-xl); box-shadow: var(--shadow-card);
  display: flex; flex-direction: column; overflow: hidden;
}
.graph-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px; border-bottom: 1px solid var(--border-subtle);
}
.mode-switcher {
  display: flex; gap: 3px; background: var(--bg-input);
  border-radius: var(--radius-xs); padding: 3px;
}
.mode-btn {
  padding: 6px 16px; font-size: 11px; font-weight: 500;
  color: var(--text-dim); background: transparent; border: none;
  border-radius: var(--radius-xs); cursor: pointer;
  font-family: var(--font-ui); transition: all var(--transition);
}
.mode-btn:hover { color: var(--text-muted); }
.mode-btn.active { color: var(--gold); background: var(--gold-soft); font-weight: 600; }
.graph-hint { font-size: 10px; color: var(--text-dim); }

.graph-canvas {
  flex: 1; position: relative; overflow: hidden; padding: 24px;
}

/* Nodes */
.tnode {
  position: absolute; background: var(--bg-card);
  border: 1.5px solid var(--border); border-radius: var(--radius-md);
  padding: 16px 20px; cursor: pointer; transition: all var(--transition);
  min-width: 120px; text-align: center; z-index: 2;
}
.tnode:hover { border-color: var(--border-bright); background: var(--bg-card-hover); }
.tnode.selected { box-shadow: 0 0 0 3px var(--gold-soft); }

.tnode-icon {
  width: 32px; height: 32px; border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 10px; font-size: 14px;
}
.tnode-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; margin-bottom: 4px;
}
.tnode-stat { font-size: 10px; color: var(--text-dim); }
.tnode-bar {
  margin-top: 8px; height: 3px; border-radius: 2px;
  background: var(--bg-input); overflow: hidden;
}
.tnode-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }

/* SVG edges */
.edge-layer { position: absolute; inset: 0; pointer-events: none; z-index: 1; }
.edge-path { stroke: var(--border); stroke-width: 1.5; fill: none; }

/* Heat mode glow */
@keyframes heat-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
.tnode.heat-cool { box-shadow: 0 0 12px rgba(94,234,212,0.4); animation: heat-pulse 2s ease-in-out infinite; }
.tnode.heat-warm { box-shadow: 0 0 16px rgba(240,201,76,0.5); animation: heat-pulse 2s ease-in-out infinite; }
.tnode.heat-hot { box-shadow: 0 0 20px rgba(255,138,128,0.6); animation: heat-pulse 2s ease-in-out infinite; }

/* Detail panel */
.trace-detail { display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
.trace-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); box-shadow: var(--shadow-card); overflow: hidden;
}
.trace-card-hdr { padding: 18px 20px 0; }
.trace-card-body { padding: 0 20px 18px; }
.trace-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--text-dim); font-weight: 700; margin-bottom: 14px;
}

.module-hdr {
  display: flex; align-items: center; gap: 12px;
  padding: 16px 20px; border-bottom: 1px solid var(--border-subtle);
}
.module-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.module-name { font-size: 13px; font-weight: 700; color: var(--text-bright); }
.module-desc { font-size: 10px; color: var(--text-dim); margin-top: 2px; }

.mini-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 16px 20px; }
.mini-stat {
  background: var(--bg-input); border-radius: var(--radius-sm);
  padding: 12px; text-align: center;
}
.mini-stat-val { font-size: 18px; font-weight: 800; color: var(--text-bright); }
.mini-stat-lbl {
  font-size: 9px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.06em; margin-top: 4px; font-weight: 600;
}

.span-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0; border-bottom: 1px solid var(--border-subtle); font-size: 12px;
}
.span-row:last-child { border-bottom: none; }
.span-name { font-family: var(--font-mono); font-size: 11px; color: var(--text-primary); }
.span-fields { font-size: 10px; color: var(--text-dim); margin-top: 2px; }
.span-pill {
  font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 99px; flex-shrink: 0;
}
.span-fast { color: var(--mint); background: var(--mint-soft); }
.span-medium { color: var(--peach); background: var(--peach-soft); }
.span-slow { color: var(--coral); background: var(--coral-soft); }

.evt-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; font-size: 11px; }
.evt-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.evt-dot.info { background: var(--mint); }
.evt-dot.warn { background: var(--amber); }
.evt-dot.error { background: var(--coral); }
.evt-msg { color: var(--text-muted); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.evt-time { color: var(--text-dim); font-family: var(--font-mono); font-size: 10px; flex-shrink: 0; }

.trace-empty {
  display: flex; align-items: center; justify-content: center;
  height: 100%; font-size: 12px; color: var(--text-dim); padding: 40px;
  text-align: center;
}

@media (max-width: 700px) {
  .traces-workspace { grid-template-columns: 1fr; height: auto; }
  .graph-panel { min-height: 50vh; }
}
```

-**Step 3: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): add Traces page CSS (graph, nodes, detail panel)"
```

---

### Task 5: Add Traces JS data structures and module map

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js`

-**Step 1: Add module topology constants**

In the JavaScript `<script>` section, after the `PROVIDERS` object, add:

```javascript
// --- Traces: module topology ---
const MODULES = {
  agent:     { label: 'Agent',     color: 'peach',    desc: 'Agent registration & session lifecycle',
               icon: '\u{1F916}', pos: [8, 10] },
  entry:     { label: 'Entry',     color: 'gold',     desc: 'Write, embed, insert pipeline',
               icon: '\u{1F4DD}', pos: [40, 8] },
  merge:     { label: 'Merge',     color: 'mint',     desc: 'Conflict detection & resolution',
               icon: '\u{1F500}', pos: [40, 48] },
  storage:   { label: 'Storage',   color: 'lavender', desc: 'LiteStore / Postgres persistence',
               icon: '\u{1F4BE}', pos: [72, 8] },
  rag:       { label: 'RAG',       color: 'sky',      desc: 'Retrieval-augmented generation',
               icon: '\u{1F50E}', pos: [72, 48] },
  inference: { label: 'Inference', color: 'coral',    desc: 'ONNX embedding via gRPC',
               icon: '\u26A1',    pos: [8, 50] },
  gc:        { label: 'GC',        color: 'amber',    desc: 'Garbage collection sweeps',
               icon: '\u{1F9F9}', pos: [8, 82] },
};

// Edges: [from, to]
const EDGES = [
  ['agent', 'entry'], ['agent', 'gc'],
  ['entry', 'storage'], ['entry', 'merge'], ['entry', 'inference'],
  ['merge', 'storage'],
  ['storage', 'rag'],
  ['gc', 'storage'],
];

// Span name -> module (specific-first matching)
const SPAN_MODULE_SPECIFIC = { 'corvia.entry.embed': 'inference' };
const SPAN_MODULE_PREFIX = [
  ['corvia.agent.', 'agent'], ['corvia.session.', 'agent'],
  ['corvia.entry.', 'entry'], ['corvia.merge.', 'merge'],
  ['corvia.store.', 'storage'], ['corvia.rag.', 'rag'],
  ['corvia.gc.', 'gc'],
];

// Span fields reference
const SPAN_FIELDS = {
  'corvia.agent.register': 'display_name',
  'corvia.session.create': 'agent_id, with_staging',
  'corvia.session.commit': 'session_id',
  'corvia.entry.write': 'session_id',
  'corvia.entry.embed': 'gRPC / Ollama',
  'corvia.entry.insert': 'entry_id, scope_id',
  'corvia.merge.process': '',
  'corvia.merge.process_entry': 'entry_id',
  'corvia.merge.conflict': 'entry_id, scope_id',
  'corvia.merge.llm_resolve': 'new_id, existing_id',
  'corvia.store.insert': 'entry_id, scope_id',
  'corvia.store.search': 'scope_id',
  'corvia.store.get': '',
  'corvia.rag.context': 'scope_id',
  'corvia.rag.ask': 'scope_id',
  'corvia.gc.run': '',
};

function spanToModule(name) {
  if (SPAN_MODULE_SPECIFIC[name]) return SPAN_MODULE_SPECIFIC[name];
  for (var i = 0; i < SPAN_MODULE_PREFIX.length; i++) {
    if (name.startsWith(SPAN_MODULE_PREFIX[i][0])) return SPAN_MODULE_PREFIX[i][1];
  }
  return 'unknown';
}

let traceMode = 'map';
let selectedModule = null;
```

-**Step 2: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): add Traces module topology constants and helpers"
```

---

## Chunk 3: Frontend — Rendering Functions

### Task 6: Implement Traces rendering in the `render()` function

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js`

-**Step 1: Add `renderTraces()` function**

After the existing `applyLogFilters()` function, add:

```javascript
// --- Traces rendering ---
function renderTraces(data) {
  var traces = (data && data.traces) || { spans: {}, recent_events: [] };
  var spans = traces.spans || {};

  // Aggregate per-module stats
  var modStats = {};
  for (var mod in MODULES) { modStats[mod] = { count: 0, count_1h: 0, avg_ms: 0, errors: 0, spanCount: 0 }; }
  var maxCount = 1;
  for (var sname in spans) {
    var mod = spanToModule(sname);
    if (!modStats[mod]) continue;
    var s = spans[sname];
    modStats[mod].count += s.count;
    modStats[mod].count_1h += (s.count_1h || 0);
    modStats[mod].avg_ms += s.avg_ms * s.count;
    modStats[mod].errors += s.errors;
    modStats[mod].spanCount++;
  }
  for (var mod in modStats) {
    var ms = modStats[mod];
    ms.avg_ms = ms.count > 0 ? Math.round(ms.avg_ms / ms.count) : 0;
    if (ms.count > maxCount) maxCount = ms.count;
  }

  var html = '<div class="traces-workspace">';

  // Graph panel
  html += '<div class="graph-panel">';
  html += '<div class="graph-toolbar">';
  html += '<div class="mode-switcher">';
  for (var m of ['map', 'dataflow', 'heat']) {
    var label = m === 'dataflow' ? 'Data Flow' : m.charAt(0).toUpperCase() + m.slice(1);
    html += '<button class="mode-btn' + (traceMode === m ? ' active' : '') + '" data-trace-mode="' + m + '">' + label + '</button>';
  }
  html += '</div>';
  html += '<span class="graph-hint">Click a module to inspect</span>';
  html += '</div>';

  // Canvas
  html += '<div class="graph-canvas" id="graphCanvas">';

  // SVG edges
  html += '<svg class="edge-layer" id="edgeLayer" style="width:100%;height:100%;position:absolute;top:0;left:0;"></svg>';

  // Nodes
  for (var id in MODULES) {
    var mod = MODULES[id];
    var st = modStats[id] || {};
    var barW = maxCount > 0 ? Math.max(5, Math.round((st.count / maxCount) * 100)) : 5;
    var sel = selectedModule === id ? ' selected' : '';

    // Heat mode classes
    var heatCls = '';
    if (traceMode === 'heat') {
      var heatScore = (st.count / maxCount) * 0.6 + (st.errors > 0 ? 0.4 : 0);
      if (heatScore > 0.7) heatCls = ' heat-hot';
      else if (heatScore > 0.3) heatCls = ' heat-warm';
      else heatCls = ' heat-cool';
    }

    // Selected border color
    var selStyle = selectedModule === id ? 'border-color:var(--' + mod.color + ');' : '';

    html += '<div class="tnode' + sel + heatCls + '" style="left:' + mod.pos[0] + '%;top:' + mod.pos[1] + '%;' + selStyle + '" data-tnode="' + id + '">';
    html += '<div class="tnode-icon" style="background:var(--' + mod.color + '-soft);color:var(--' + mod.color + ');">' + mod.icon + '</div>';
    html += '<div class="tnode-label" style="color:var(--' + mod.color + ');">' + esc(mod.label) + '</div>';
    html += '<div class="tnode-stat">' + formatNum(st.count) + ' ops &middot; ' + st.spanCount + ' spans</div>';
    html += '<div class="tnode-bar"><div class="tnode-bar-fill" style="width:' + barW + '%;background:var(--' + mod.color + ');"></div></div>';
    html += '</div>';
  }

  html += '</div>'; // close graph-canvas
  html += '</div>'; // close graph-panel

  // Detail panel
  html += '<div class="trace-detail">';
  if (!selectedModule) {
    html += '<div class="trace-card"><div class="trace-empty">Select a module to inspect its telemetry</div></div>';
  } else {
    var sm = MODULES[selectedModule];
    var ss = modStats[selectedModule] || {};
    var modColor = sm.color;

    // Module summary card
    html += '<div class="trace-card">';
    html += '<div class="module-hdr">';
    html += '<div class="module-dot" style="background:var(--' + modColor + ');box-shadow:0 0 6px var(--' + modColor + '-soft);"></div>';
    html += '<div><div class="module-name">' + esc(sm.label) + '</div>';
    html += '<div class="module-desc">' + esc(sm.desc) + '</div></div>';
    html += '</div>';
    html += '<div class="mini-stats">';
    html += miniStat(formatNum(ss.count), 'Total');
    html += miniStat(formatNum(ss.count_1h), 'Last hour');
    var avgColor = ss.avg_ms < 50 ? 'var(--mint)' : ss.avg_ms < 150 ? 'var(--peach)' : 'var(--coral)';
    html += miniStat('<span style="color:' + avgColor + '">' + ss.avg_ms + '<span style="font-size:11px;font-weight:500">ms</span></span>', 'Avg latency');
    var errColor = ss.errors === 0 ? 'var(--mint)' : 'var(--coral)';
    html += miniStat('<span style="color:' + errColor + '">' + ss.errors + '</span>', 'Errors');
    html += '</div></div>';

    // Spans card
    var moduleSpans = [];
    for (var sn in spans) {
      if (spanToModule(sn) === selectedModule) {
        moduleSpans.push({ name: sn, stats: spans[sn] });
      }
    }

    html += '<div class="trace-card"><div class="trace-card-hdr"><div class="trace-label">Instrumented Spans</div></div>';
    html += '<div class="trace-card-body">';
    if (moduleSpans.length === 0) {
      html += '<div style="font-size:12px;color:var(--text-dim);padding:8px 0;">No span data available</div>';
    } else {
      for (var sp of moduleSpans) {
        var shortName = sp.name.replace('corvia.', '');
        var fields = SPAN_FIELDS[sp.name] || '';
        var ms = sp.stats.avg_ms;
        var pillCls = ms < 50 ? 'span-fast' : ms < 150 ? 'span-medium' : 'span-slow';
        html += '<div class="span-row"><div>';
        html += '<div class="span-name">' + esc(shortName) + '</div>';
        if (fields) html += '<div class="span-fields">' + esc(fields) + '</div>';
        html += '</div><span class="span-pill ' + pillCls + '">' + Math.round(ms) + 'ms</span></div>';
      }
    }
    html += '</div></div>';

    // Events card
    var modEvents = (traces.recent_events || []).filter(function(ev) { return ev.module === selectedModule; }).slice(0, 10);
    html += '<div class="trace-card"><div class="trace-card-hdr"><div class="trace-label">Recent Events</div></div>';
    html += '<div class="trace-card-body">';
    if (modEvents.length === 0) {
      html += '<div style="font-size:12px;color:var(--text-dim);padding:8px 0;">No recent events</div>';
    } else {
      for (var ev of modEvents) {
        html += '<div class="evt-row">';
        html += '<div class="evt-dot ' + ev.level + '"></div>';
        html += '<span class="evt-msg">' + esc(ev.msg) + '</span>';
        html += '<span class="evt-time">' + esc(ev.ts) + '</span>';
        html += '</div>';
      }
    }
    html += '</div></div>';
  }
  html += '</div>'; // close trace-detail
  html += '</div>'; // close traces-workspace

  return html;
}

function miniStat(valueHtml, label) {
  return '<div class="mini-stat"><div class="mini-stat-val">' + valueHtml + '</div>' +
    '<div class="mini-stat-lbl">' + esc(label) + '</div></div>';
}
```

-**Step 2: Update `render()` to call `renderTraces()` for the traces view**

In the `render()` function, find the block that handles `activeView !== 'logs'` (the placeholder for Graph/Traces). Replace the placeholder block:

```javascript
  } else {
    // Placeholder for Graph / Traces
    html += '<div class="view-placeholder">' +
      '<svg width="48" height="48" viewBox="0 0 16 16" fill="currentColor"><path d="M1 1v14h14V1H1zm13 13H2V2h12v12zM3 13V8h2v5H3zm3 0V5h2v8H6zm3 0V9h2v4H9zm3 0V3h1v10h-1z"/></svg>' +
      '<p>' + esc(activeView.charAt(0).toUpperCase() + activeView.slice(1)) + ' view coming soon</p>' +
    '</div>';
  }
```

With:

```javascript
  } else if (activeView === 'traces') {
    // Traces view — close log-panel early, render outside it
    html += '</div>'; // close log-panel
    html += renderTraces(data);
    html += '</div>'; // close workspace
    // Set flags to skip normal closing tags
    el.innerHTML = html;
    bindAll();
    bindTraces();
    drawEdges();
    return;
  } else {
    // Placeholder for Graph
    html += '<div class="view-placeholder">' +
      '<svg width="48" height="48" viewBox="0 0 16 16" fill="currentColor"><path d="M1 1v14h14V1H1zm13 13H2V2h12v12zM3 13V8h2v5H3zm3 0V5h2v8H6zm3 0V9h2v4H9zm3 0V3h1v10h-1z"/></svg>' +
      '<p>Graph view coming soon</p>' +
    '</div>';
  }
```

Note: the traces view needs a different layout (no sidebar), so it breaks out of the normal log-panel + sidebar structure. The early return avoids appending the sidebar HTML.

-**Step 3: Add `bindTraces()` and `drawEdges()` functions**

After `bindAll()`, add:

```javascript
function bindTraces() {
  document.querySelectorAll('[data-tnode]').forEach(function(n) {
    n.onclick = function() {
      selectedModule = n.dataset.tnode;
      vscode.postMessage({ type: 'refresh' });
    };
  });

  document.querySelectorAll('[data-trace-mode]').forEach(function(b) {
    b.onclick = function() {
      traceMode = b.dataset.traceMode;
      vscode.postMessage({ type: 'refresh' });
    };
  });
}

function drawEdges() {
  var svg = document.getElementById('edgeLayer');
  var canvas = document.getElementById('graphCanvas');
  if (!svg || !canvas) return;

  var cw = canvas.offsetWidth;
  var ch = canvas.offsetHeight;
  svg.setAttribute('viewBox', '0 0 ' + cw + ' ' + ch);

  var paths = '';
  var animations = '';
  for (var i = 0; i < EDGES.length; i++) {
    var e = EDGES[i];
    var fromMod = MODULES[e[0]];
    var toMod = MODULES[e[1]];
    if (!fromMod || !toMod) continue;

    var x1 = (fromMod.pos[0] / 100) * cw + 60;
    var y1 = (fromMod.pos[1] / 100) * ch + 40;
    var x2 = (toMod.pos[0] / 100) * cw + 60;
    var y2 = (toMod.pos[1] / 100) * ch + 40;
    var mx = (x1 + x2) / 2;
    var my = (y1 + y2) / 2;

    var pathId = 'edge-' + e[0] + '-' + e[1];
    var d = 'M' + x1 + ',' + y1 + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2;
    paths += '<path id="' + pathId + '" class="edge-path" d="' + d + '"/>';

    if (traceMode === 'dataflow') {
      var color = 'var(--' + fromMod.color + ')';
      animations += '<circle r="3" fill="' + color + '" style="filter:drop-shadow(0 0 3px ' + color + ')">' +
        '<animateMotion dur="3s" repeatCount="indefinite"><mpath href="#' + pathId + '"/></animateMotion>' +
        '</circle>';
    }
  }

  svg.innerHTML = paths + animations;
}
```

-**Step 4: Verify the extension loads without syntax errors**

Run: `cd /workspaces/corvia-workspace/.devcontainer/extensions/corvia-services && node -e "try { require('./extension.js'); } catch(e) { if (e.code === 'MODULE_NOT_FOUND' && e.message.includes('vscode')) { console.log('Syntax OK'); } else { throw e; } }"`
Expected: `Syntax OK`

-**Step 5: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): add Traces page rendering (graph, detail panel, 3 modes)"
```

---

## Chunk 4: Integration, VSIX, and Verification

### Task 7: Build and install VSIX

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js` (if fixes needed)

-**Step 1: Rebuild VSIX**

Run: `cd /workspaces/corvia-workspace/.devcontainer/extensions/corvia-services && vsce package --no-dependencies`
Expected: `DONE  Packaged: corvia-services-0.3.0.vsix`

-**Step 2: Install extension**

Run: `code --install-extension /workspaces/corvia-workspace/.devcontainer/extensions/corvia-services/corvia-services-0.3.0.vsix --force`
Expected: `Extension 'corvia-services-0.3.0.vsix' was successfully installed.`

-**Step 3: Commit any fixes**

```bash
git add .devcontainer/extensions/corvia-services/
git commit -m "chore(extension): rebuild VSIX with Traces page"
```

---

### Task 8: End-to-end verification

Run through the 12-point checklist from the spec:

-**Step 1: Verify Traces tab renders**

Open Corvia Dashboard → click Traces tab. Should show 7 module nodes in a graph layout with a detail panel on the right.

-**Step 2: Verify all 7 modules render with correct colors**

Check: Agent (peach), Entry (gold), Merge (mint), Storage (lavender), RAG (sky), Inference (coral), GC (amber).

-**Step 3: Verify SVG edges connect correctly**

Visual check: 8 curved edges connecting modules per the topology table.

-**Step 4: Verify mode switcher**

Click Map → Data Flow → Heat. Map shows static nodes. Data Flow shows animated dots. Heat shows pulsing glows.

-**Step 5: Verify node click → detail panel**

Click Entry node. Detail panel should show: module name, description, 4 mini-stats, span list with timing pills, recent events.

-**Step 6: Verify empty state**

On initial load with no node selected, detail panel should say "Select a module to inspect its telemetry".

-**Step 7: Verify graceful degradation**

If `traces` field is missing from status JSON, nodes should show "- ops" and detail panel should show "No span data available".

-**Step 8: Verify timing pill colors**

Fast spans (<50ms) = mint, Medium (50-150ms) = peach, Slow (>150ms) = coral.

-**Step 9: Verify responsive layout**

Resize window below 700px. Graph and detail should stack vertically.

-**Step 10: Run Python tests**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_traces.py -v`
Expected: All tests pass.

-**Step 11: Verify status JSON includes traces**

Run: `corvia-dev status --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('traces:', 'present' if d.get('traces') else 'absent')"`

-**Step 12: Commit final state**

```bash
git add -A
git commit -m "feat: telemetry traces page — interactive module map with 3 rendering modes"
```
