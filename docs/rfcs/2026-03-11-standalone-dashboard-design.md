# Standalone Dashboard & corvia-dev Rust Migration — Design

**Date**: 2026-03-11
**Status**: Shipped (v0.3.7)

## Problem

The corvia dashboard is embedded as a ~1200-line HTML blob inside the VS Code extension
(`extension.js`). This means:
- Only accessible inside VS Code — developers using Neovim, JetBrains, or terminal-only workflows can't see it
- Not testable with browser automation (Playwright)
- Dashboard changes require extension rebuild + reload
- Three languages in the data path: Python (trace parsing) → Rust (API) → JS (rendering)
- Maintenance burden from duplicated/embedded UI code

## Goals

1. Standalone browser-accessible dashboard with full parity to current VS Code UI
2. Eliminate Python dependency — migrate `corvia-dev` functionality to Rust
3. Single source of truth — VS Code extension becomes a thin wrapper embedding the standalone UI
4. Playwright-testable dashboard
5. Reduce maintenance to 2-language data path: Rust (API) → TypeScript (UI)

## Audience (priority order)

1. **Developers outside VS Code** — primary
2. **Demo/showcase** — secondary
3. **Ops/team leads** — tertiary

## Architecture

### Approach: Parallel tracks with shared API contract

Define the REST API contract first. Two parallel tracks:
- **Track A (Rust)**: Migrate `corvia-dev` trace/log/health logic into `corvia-server` dashboard endpoints
- **Track B (UI)**: Build Vite + Preact dashboard against the contract (mock data initially, real endpoints as Track A delivers)
- **Final step**: Retrofit VS Code extension to embed the standalone dashboard

### Component ownership

| Component | Location | Owner |
|---|---|---|
| Dashboard REST API endpoints | `repos/corvia/crates/corvia-server/src/dashboard/` | corvia core |
| Dashboard response types | `repos/corvia/crates/corvia-common/src/dashboard.rs` | corvia core |
| Vite + Preact dashboard app | `tools/corvia-dashboard/` | corvia workspace |
| VS Code extension (retrofitted) | `.devcontainer/extensions/corvia-services/` | corvia workspace |

## Section 1: API Contract

New REST endpoints on `corvia-server` (`:8020`) under `/api/dashboard/`:

| Endpoint | Returns | Source (currently in corvia-dev) |
|---|---|---|
| `GET /api/dashboard/status` | Services health, entry counts, agents, sessions, merge queue | `corvia_dev/cli.py` → `status --json` |
| `GET /api/dashboard/traces` | Span statistics, module topology, recent events | `corvia_dev/traces.py` → `collect_traces()` |
| `GET /api/dashboard/logs?module=X&level=Y` | Filtered structured log entries | `corvia_dev/traces.py` → log parsing |
| `GET /api/dashboard/graph?scope=X` | Knowledge graph edges for visualization | Already exists in kernel (`corvia_graph`) |
| `GET /api/dashboard/config` | Current server config (read-only) | `corvia_config_get` |

All endpoints return JSON. Response types defined in `corvia-common::dashboard` for shared use.

**Why REST, not gRPC**: Browser `fetch()` works natively, payloads are small JSON, polling pattern
doesn't need streaming, Axum already serves REST. gRPC-web would add complexity for no benefit here.

## Section 2: Rust Migration — corvia-dev → corvia-server

### What moves into Rust

1. **Service health checking** (`cli.py` — process detection, port probing) → `corvia-server::dashboard::health`
2. **Trace/log collection** (`traces.py` — JSON structured log parsing, span aggregation, module classification) → `corvia-server::dashboard::traces`
3. **Status aggregation** (combines health + traces + store metrics) → `/api/dashboard/status` handler

### Disposition of corvia-dev features

| Feature | Action |
|---|---|
| `status --json` | Port to Rust (dashboard endpoints) |
| `collect_traces()` / log parsing | Port to Rust |
| `corvia-dev serve` (starts server) | Keep as thin shell script or remove |
| `corvia-dev workspace ingest` | Already calls Rust CLI — becomes `corvia workspace ingest` |
| MCP server startup | Already Rust |

### New module structure

```
crates/corvia-server/src/
├── dashboard/
│   ├── mod.rs          # Router: /api/dashboard/*
│   ├── health.rs       # Service health probing
│   ├── traces.rs       # Log parsing, span aggregation
│   └── types.rs        # Re-exports from corvia-common::dashboard
```

### Log access strategy

Start with log file reading (parity with Python). Follow-up: in-process `tracing` subscriber
for zero-file-IO trace capture.

## Section 3: Standalone Dashboard (Vite + Preact)

### Structure

```
tools/corvia-dashboard/
├── package.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx              # Entry point, router
│   ├── api.ts                # Typed fetch wrapper for /api/dashboard/*
│   ├── types.ts              # Mirrors corvia-common::dashboard types
│   ├── hooks/
│   │   └── use-poll.ts       # Generic polling hook
│   ├── components/
│   │   ├── Layout.tsx        # Header, sidebar, tab navigation
│   │   ├── StatusBar.tsx     # Service health pills, metrics grid
│   │   ├── LogsView.tsx      # Filterable structured log viewer
│   │   ├── GraphView.tsx     # Knowledge graph visualization
│   │   ├── TracesView.tsx    # Module topology, span stats, recent events
│   │   └── ConfigPanel.tsx   # Read-only config sidebar
│   └── styles/
│       └── theme.css         # Ported CSS variables (gold/navy theme)
```

### Key decisions

- **Polling**: 3s default (matches current extension), configurable
- **API base URL**: Env var, defaults to `http://localhost:8020`
- **Routing**: Hash router (`/#/logs`, `/#/traces`, `/#/graph`) — works in browser and webview
- **Theme**: Direct port of existing CSS variables from `extension.js`
- **Build output**: `dist/` — static assets

### VS Code extension retrofit

Extension becomes ~50 lines:
1. Create webview panel pointing at `http://localhost:8021`
2. Status bar polls `/api/dashboard/status` directly (no more `corvia-dev` shell exec)

## Section 4: Development & Maintenance

### Startup sequence

1. `corvia-server` starts on `:8020` (API + dashboard endpoints)
2. `corvia-dashboard` dev server on `:8021` (Vite HMR in dev, static server in prod)
3. VS Code extension webview loads `:8021`

### Maintenance improvements

| Before | After |
|---|---|
| ~1200-line embedded HTML blob | Component-based Preact app |
| Python + Rust + JS (3 languages) | Rust + TypeScript (2 languages) |
| VS Code-only | Browser + VS Code |
| Can't test with Playwright | Playwright-testable |
| Extension rebuild for UI changes | Vite HMR — instant |

### Product positioning

- **corvia core** = the product. Ships the API including dashboard endpoints.
- **corvia-dashboard** = dev/ops companion. Workspace tooling. Not part of core distribution.
- **VS Code extension** = thin shell embedding the dashboard. IDE-specific glue only.

No product overlap — the extension no longer has its own UI, it's a viewport.

### Testing strategy

- **Dashboard**: Playwright tests against Vite dev server
- **API endpoints**: Rust integration tests in core repo
- **Extension**: Minimal — verifies webview loads dashboard URL

## Scope — v1

Full parity with current VS Code dashboard:
- Logs tab
- Graph tab
- Traces tab (module topology, span stats, recent events)
- Metrics grid (entries, agents, merge queue, sessions)
- Service health status
- Config sidebar (read-only)
