# Telemetry Traces Page — Design Spec

**Date:** 2026-03-10
**Status:** Shipped (v0.3.7)
**Depends on:** M4 observability (spans), dashboard redesign (view tabs)
**Affects:** `.devcontainer/extensions/corvia-services/extension.js`, `tools/corvia-dev/`

---

## Overview

Add an interactive telemetry map as the "Traces" view tab in the Corvia VS Code dashboard. Visualizes the 7 instrumented kernel modules as a node-graph topology with three rendering modes: Map (static topology), Data Flow (animated data movement), and Heat (load-derived intensity). A persistent detail panel shows span timings, mini-stats, and recent events for the selected module.

---

## Architecture

### Page Structure
- Replaces the current "Traces" placeholder in the view tab system
- Uses the existing 70/30 workspace layout but replaces log panel + sidebar with graph panel + detail panel
- Graph panel (left, flex: 1) contains mode switcher + canvas
- Detail panel (right, 280px) contains 3 stacked cards

### Rendering Modes
1. **Map** — static node-graph topology. Nodes display live stats. Click to inspect.
2. **Data Flow** — same topology with animated dots flowing along edges showing data movement direction and volume.
3. **Heat** — same topology with nodes pulsing/glowing based on derived load (span frequency, queue depth, error rate).

### Data Source
- Primary: `corvia-dev status --json` extended with a `traces` field containing span timing aggregates and recent events
- Module mapping done client-side using span name prefixes (`corvia.entry.*` → Entry, etc.)
- Span-derived heat metrics now; real resource metrics (CPU/RAM/GPU/disk) as a follow-up

### Code Structure
- All rendering in `extension.js` (existing single-file pattern)
- Graph uses absolute-positioned divs for nodes, inline SVG for edges
- Fixed layout grid for node positions (deterministic, no force-directed physics)
- Node positions use percentage-based coordinates for responsive scaling
- No external libraries

### Node Position Grid

Nodes are placed on a 3-column layout within the graph canvas (percentage-based):

```
         Col 1 (8%)       Col 2 (40%)      Col 3 (72%)
Row 1    Agent (8%, 10%)   Entry (40%, 8%)   Storage (72%, 8%)
         (top-left)        (top-center)      (top-right)

Row 2    Inference (8%,50%) Merge (40%, 48%)  RAG (72%, 48%)
         (mid-left)        (mid-center)      (mid-right)

Row 3    GC (8%, 82%)
         (bottom-left)
```

Format: `(left%, top%)` relative to canvas. Node width: 120px fixed. Canvas has 24px padding.

---

## Module Topology

7 modules with fixed connections:

| Module | Color | Spans | Connects To |
|--------|-------|-------|-------------|
| Agent | peach | `agent.register`, `session.create`, `session.commit` | Entry, GC |
| Entry | gold | `entry.write`, `entry.embed`, `entry.insert` | Storage, Merge, Inference |
| Merge | mint | `merge.process`, `merge.process_entry`, `merge.conflict`, `merge.llm_resolve` | Storage |
| Storage | lavender | `store.insert`, `store.search`, `store.get` | RAG |
| RAG | sky (#7dd3fc) | `rag.context`, `rag.ask` | (terminal) |
| Inference | coral | `entry.embed` (called by Entry) | (called by Entry) |
| GC | amber | `gc.run` | Storage |

Module descriptions (hardcoded client-side):
| Module | Description |
|--------|-------------|
| Agent | Agent registration & session lifecycle |
| Entry | Write, embed, insert pipeline |
| Merge | Conflict detection & resolution |
| Storage | LiteStore / Postgres persistence |
| RAG | Retrieval-augmented generation |
| Inference | ONNX embedding via gRPC |
| GC | Garbage collection sweeps |

Span name → module mapping prefix table:
```
corvia.agent.* → Agent
corvia.session.* → Agent
corvia.entry.write → Entry
corvia.entry.insert → Entry
corvia.entry.embed → Inference  (special case: embed runs on inference service)
corvia.merge.* → Merge
corvia.store.* → Storage
corvia.rag.* → RAG
corvia.gc.* → GC
```
The `corvia.entry.embed` span is mapped to **Inference** (not Entry) because the work is performed by the inference service. The prefix table is evaluated most-specific-first, so `corvia.entry.embed` matches before `corvia.entry.*`.

---

## Components

### Graph Panel

**Mode Switcher:**
- Segmented control (same style as log filter bar): Map / Data Flow / Heat
- Active mode: gold text + gold-soft background
- Positioned in graph toolbar alongside "Click a module to inspect" hint

**Canvas:**
- Relative-positioned container with two layers:
  - SVG edge layer (`pointer-events: none`) — curved connection lines between modules
  - Node layer — absolutely-positioned module nodes on fixed layout grid

**Module Nodes:**
Each node contains:
- Color-coded icon (32px, radius-sm, accent-soft background)
- Uppercase label in module accent color
- Stat line: key metric + span count (text-dim)
- Mini activity bar (3px, width proportional to span frequency relative to most active module)

Node states:
- Default: `bg-card` + `border`
- Hover: `bg-card-hover` + `border-bright`
- Selected: border becomes module color + `0 0 0 3px {color}-soft` outer ring

**SVG Edges:**
- Curved paths (`C` bezier) connecting module centers
- Stroke: `border` color, 1.5px
- Data Flow mode: animated dots travel along edges using `<circle>` elements with `<animateMotion>` referencing the edge `<path>` via `<mpath>`. This avoids CSS `offset-path` compatibility issues.
- Each dot: r=3, fill = source node accent color, filter = `drop-shadow(0 0 3px {color} 0.5)`
- Dot color matches source node's accent
- Animation duration: `max(1s, 6s - (count / maxCount) * 5s)` — more activity = faster dots (bounds: 1s–6s)

**Heat Mode Overlay:**
- Nodes get pulsing border-glow whose intensity maps to derived load
- Heat score per module = `(spanCount / maxSpanCount) * 0.6 + (errorCount > 0 ? 0.4 : 0)`
- Cool (score 0–0.3): mint glow — `box-shadow: 0 0 12px rgba(94,234,212,0.4)`
- Warm (score 0.3–0.7): gold glow — `box-shadow: 0 0 16px rgba(240,201,76,0.5)`
- Hot (score 0.7–1.0 or errors > 0): coral glow — `box-shadow: 0 0 20px rgba(255,138,128,0.6)`
- Glow pulses via CSS animation: `2s ease-in-out infinite` alternating shadow intensity
- Mini activity bar color also reflects heat level (same thresholds)

### Detail Panel (right 280px)

Three stacked cards, updating reactively on node selection:

**1. Module Summary:**
- Header: color dot (10px) + module name (13px/700) + description
- 2x2 mini-stat grid:
  - Total count
  - Last hour count
  - Avg latency (ms)
  - Error count
- Mini-stat style: bg-input rounded container, 18px/800 value, 9px uppercase label

**2. Instrumented Spans:**
- List of span rows with bottom borders
- Each row: mono span name (11px), field names below (10px, text-dim)
- Right-aligned timing pill (rounded, 11px mono):
  - Fast (< 50ms): mint text, mint-soft background
  - Medium (50–150ms): peach text, peach-soft background
  - Slow (> 150ms): coral text, coral-soft background

**3. Recent Events:**
- Last 10 structured log events for the selected module
- Each row: level dot (6px: mint=info, amber=warn, coral=error) + message + timestamp
- Scrollable if overflow

**Empty State:**
When no node is selected: centered text "Select a module to inspect its telemetry" (text-dim)

---

## Data Contract

### New `traces` field in status JSON

```json
{
  "traces": {
    "spans": {
      "corvia.agent.register": { "count": 3, "count_1h": 1, "avg_ms": 5, "last_ms": 4, "errors": 0 },
      "corvia.session.create": { "count": 12, "count_1h": 4, "avg_ms": 8, "last_ms": 6, "errors": 0 },
      "corvia.session.commit": { "count": 10, "count_1h": 3, "avg_ms": 45, "last_ms": 38, "errors": 0 },
      "corvia.entry.write": { "count": 24, "count_1h": 12, "avg_ms": 12, "last_ms": 8, "errors": 0 },
      "corvia.entry.embed": { "count": 24, "count_1h": 12, "avg_ms": 84, "last_ms": 92, "errors": 0 },
      "corvia.entry.insert": { "count": 24, "count_1h": 12, "avg_ms": 8, "last_ms": 6, "errors": 0 },
      "corvia.merge.process": { "count": 5, "count_1h": 2, "avg_ms": 340, "last_ms": 290, "errors": 0 },
      "corvia.merge.conflict": { "count": 2, "count_1h": 1, "avg_ms": 120, "last_ms": 110, "errors": 0 },
      "corvia.store.insert": { "count": 48, "count_1h": 18, "avg_ms": 15, "last_ms": 12, "errors": 0 },
      "corvia.store.search": { "count": 30, "count_1h": 10, "avg_ms": 22, "last_ms": 18, "errors": 0 },
      "corvia.store.get": { "count": 15, "count_1h": 5, "avg_ms": 3, "last_ms": 2, "errors": 0 },
      "corvia.rag.context": { "count": 12, "count_1h": 4, "avg_ms": 450, "last_ms": 380, "errors": 0 },
      "corvia.rag.ask": { "count": 8, "count_1h": 3, "avg_ms": 1200, "last_ms": 980, "errors": 0 },
      "corvia.gc.run": { "count": 2, "count_1h": 1, "avg_ms": 180, "last_ms": 160, "errors": 0 }
    },
    "recent_events": [
      { "ts": "14:31:52", "level": "info", "module": "entry", "msg": "Entry inserted scope:corvia" },
      { "ts": "14:31:51", "level": "info", "module": "inference", "msg": "Embedded 768d vector" },
      { "ts": "14:30:18", "level": "warn", "module": "inference", "msg": "Slow embed: 210ms" },
      { "ts": "14:30:12", "level": "info", "module": "agent", "msg": "Session write from claude" },
      { "ts": "14:28:44", "level": "info", "module": "merge", "msg": "auto_merged entry" },
      { "ts": "14:25:10", "level": "info", "module": "storage", "msg": "Rebuilt graph with 84 edges" }
    ]
  }
}
```

Each span object includes `count_1h` (count in the last hour) for the detail panel's "Last hour" mini-stat. The `recent_events` array contains the last 50 events globally, with the `module` field for client-side per-module filtering. Client filters by module and displays the first 10 matches.

### Parsing Strategy
- corvia-dev CLI parses JSON-formatted tracing output from service logs
- Aggregates span timings over a rolling window
- Collects last N structured events per module
- Returns in status JSON under `traces` key

### Graceful Degradation
- If `traces` field is absent (older corvia-dev): nodes show "—" for stats, detail panel shows "Upgrade corvia-dev for span data"
- Topology map always renders regardless of data availability

---

## Design Tokens

One new color added for RAG to avoid duplication with Merge (both were mint). Module-to-color mapping:
- Agent → peach (`#ffb07c`)
- Entry → gold (`#f0c94c`)
- Merge → mint (`#5eead4`)
- Storage → lavender (`#c4b5fd`)
- RAG → sky (`#7dd3fc`) — **new token**: `--sky: #7dd3fc; --sky-soft: rgba(125,211,252,0.10); --sky-medium: rgba(125,211,252,0.16);`
- Inference → coral (`#ff8a80`)
- GC → amber (`#fcd34d`)

All accent colors already have `-soft` and `-medium` variants defined. The new `--sky` tokens follow the same pattern. `--amber-soft` and `--amber-medium` are already in the dashboard CSS.

### Timing Pill Thresholds
| Range | Color | Background |
|-------|-------|------------|
| < 50ms | mint | mint-soft |
| 50–150ms | peach | peach-soft |
| > 150ms | coral | coral-soft |

### Transitions
- All node hover/select: `0.25s cubic-bezier(0.4, 0, 0.2, 1)`
- Data Flow dots: CSS `offset-distance` animation, duration inversely proportional to activity
- Heat glow: CSS `box-shadow` animation, 2s ease-in-out infinite

---

## Responsive Behavior

Below 700px:
- Graph panel and detail panel stack vertically (single column)
- Graph panel gets min-height: 50vh
- Node positions scale proportionally via percentage-based layout

---

## What Does NOT Change
- Extension activation logic, polling interval (3s), status bar item
- Header, metrics row, view tab structure
- Logs and Graph tab content
- `onDidReceiveMessage` handler patterns
- `package.json` manifest

---

## Verification Criteria
1. Traces tab activates and renders the module topology graph
2. All 7 modules render as clickable nodes with correct colors
3. SVG edges connect modules according to the topology table
4. Mode switcher toggles between Map, Data Flow, and Heat
5. Data Flow mode shows animated dots along edges
6. Heat mode shows intensity-based glow on nodes
7. Clicking a node populates the detail panel with module summary, spans, and events
8. Span timing pills use correct color thresholds
9. Empty state shows when no node is selected
10. Graceful degradation when `traces` field is missing from status JSON
11. Layout collapses to single column below 700px
12. All transitions and hover states follow existing design token conventions

---

*Mockup reference: `/tmp/brainstorm/telemetry-map-layout-v2.html`*
*Created: 2026-03-10*
