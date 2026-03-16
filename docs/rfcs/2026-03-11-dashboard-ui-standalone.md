# Standalone Dashboard UI — Implementation Plan

> **Status:** Shipped (v0.3.7)

**Goal:** Build a browser-accessible Vite + Preact dashboard with full parity to the current VS Code extension, then retrofit the extension to embed it.

**Architecture:** Preact SPA with hash routing, typed fetch API client, 3-second polling, and component-based UI. CSS design tokens ported from the extension's embedded HTML. Runs on `:8021` (dev) and calls corvia-server API at `:8020`.

**Tech Stack:** Vite, Preact, TypeScript, CSS custom properties

**Design Spec:** `2026-03-11-standalone-dashboard-design.md`

**UI Reference:** `.devcontainer/extensions/corvia-services/extension.js` (lines 119-688 for CSS, 690-1470 for HTML/JS)

**Parallel Track:** Rust API plan at `2026-03-11-dashboard-api-rust.md`

**Depends on:** Dashboard API endpoints must be available (or use mock data during development)

---

## File Structure

### New Files

```
tools/corvia-dashboard/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/
│   ├── main.tsx                    # Entry point, router setup
│   ├── types.ts                    # TypeScript types (mirrors Rust dashboard types)
│   ├── api.ts                      # Typed fetch wrapper for /api/dashboard/*
│   ├── hooks/
│   │   └── use-poll.ts             # Generic polling hook (3s default)
│   ├── styles/
│   │   └── theme.css               # Design tokens (colors, typography, shadows)
│   ├── components/
│   │   ├── Layout.tsx              # Shell: header + metrics + main area
│   │   ├── Header.tsx              # Brand, status pills, scope badge
│   │   ├── MetricsGrid.tsx         # 4 metric cards with trends
│   │   ├── LogsView.tsx            # Log terminal with level filters and search
│   │   ├── TracesView.tsx          # Module topology graph + detail panel
│   │   ├── GraphView.tsx           # Knowledge graph (placeholder, extend later)
│   │   ├── ConfigPanel.tsx         # Config sidebar (read-only)
│   │   └── OfflineState.tsx        # Error/offline fallback
│   └── test/
│       └── dashboard.spec.ts       # Playwright test suite
```

### Modified Files

| File | Change |
|------|--------|
| `.devcontainer/extensions/corvia-services/extension.js` | Retrofit: replace HTML blob with webview pointing at dashboard URL |

---

## Chunk 1: Project Setup & Foundation

### Task 1: Scaffold Vite + Preact project

**Files:**
- Create: `tools/corvia-dashboard/package.json`
- Create: `tools/corvia-dashboard/vite.config.ts`
- Create: `tools/corvia-dashboard/tsconfig.json`
- Create: `tools/corvia-dashboard/index.html`

-**Step 1: Create `package.json`**

```json
{
  "name": "corvia-dashboard",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --port 8021",
    "build": "vite build",
    "preview": "vite preview --port 8021",
    "test": "npx playwright test"
  },
  "dependencies": {
    "preact": "^10.25.0"
  },
  "devDependencies": {
    "@preact/preset-vite": "^2.9.0",
    "typescript": "^5.7.0",
    "vite": "^6.0.0",
    "@playwright/test": "^1.50.0"
  }
}
```

-**Step 2: Create `vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  plugins: [preact()],
  server: {
    port: 8021,
    proxy: {
      "/api": {
        target: "http://localhost:8020",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
```

-**Step 3: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "jsxImportSource": "preact",
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "dist"
  },
  "include": ["src"]
}
```

-**Step 4: Create `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Corvia Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

-**Step 5: Create `.gitignore`**

```
node_modules/
dist/
```

Save to `tools/corvia-dashboard/.gitignore`.

-**Step 6: Install dependencies**

```bash
cd tools/corvia-dashboard && npm install
```

Expected: `node_modules/` created, no errors

-**Step 7: Verify Vite starts**

```bash
cd tools/corvia-dashboard && npx vite --port 8021 &
sleep 2 && curl -s http://localhost:8021/ | head -5
kill %1
```

Expected: HTML response containing `<div id="app">`

-**Step 8: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/package.json tools/corvia-dashboard/vite.config.ts tools/corvia-dashboard/tsconfig.json tools/corvia-dashboard/index.html tools/corvia-dashboard/.gitignore
git commit -m "feat(dashboard-ui): scaffold Vite + Preact project"
```

---

### Task 2: Design tokens (theme CSS)

**Files:**
- Create: `tools/corvia-dashboard/src/styles/theme.css`

-**Step 1: Create `theme.css` with all design tokens**

Port all CSS custom properties from `extension.js` (lines 119-184):

```css
/* tools/corvia-dashboard/src/styles/theme.css */

:root {
  /* Backgrounds */
  --bg-primary: #12141a;
  --bg-elevated: #1a1d26;
  --bg-card: #1e2230;
  --bg-card-hover: #252a3a;
  --bg-input: #282d3e;
  --bg-surface: #2e3447;

  /* Accent colors */
  --gold: #f0c94c;
  --gold-bright: #ffe066;
  --gold-soft: rgba(240, 201, 76, 0.10);
  --gold-medium: rgba(240, 201, 76, 0.18);
  --mint: #5eead4;
  --mint-soft: rgba(94, 234, 212, 0.10);
  --mint-medium: rgba(94, 234, 212, 0.16);
  --coral: #ff8a80;
  --coral-soft: rgba(255, 138, 128, 0.10);
  --coral-medium: rgba(255, 138, 128, 0.16);
  --peach: #ffb07c;
  --peach-soft: rgba(255, 176, 124, 0.10);
  --peach-medium: rgba(255, 176, 124, 0.16);
  --lavender: #c4b5fd;
  --lavender-soft: rgba(196, 181, 253, 0.10);
  --lavender-medium: rgba(196, 181, 253, 0.16);
  --amber: #fcd34d;
  --amber-soft: rgba(252, 211, 77, 0.10);
  --sky: #7dd3fc;
  --sky-soft: rgba(125, 211, 252, 0.10);

  /* Text */
  --text-bright: #ffffff;
  --text-primary: #e0ddd8;
  --text-muted: #b0a99f;
  --text-dim: #8a8279;

  /* Borders */
  --border: rgba(80, 75, 68, 0.4);
  --border-bright: rgba(100, 94, 86, 0.45);
  --border-subtle: rgba(65, 60, 54, 0.35);

  /* Radius */
  --radius-xs: 6px;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;

  /* Shadows */
  --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.18);
  --shadow-hover: 0 4px 12px rgba(0, 0, 0, 0.25);
  --shadow-gold: 0 2px 8px rgba(240, 201, 76, 0.08);

  /* Typography */
  --font-ui: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "Cascadia Code", "JetBrains Mono", "Fira Code", monospace;

  /* Transitions */
  --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-ui);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--bg-surface);
  border-radius: 3px;
}
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/styles/theme.css
git commit -m "feat(dashboard-ui): add design token theme CSS"
```

---

### Task 3: TypeScript types and API client

**Files:**
- Create: `tools/corvia-dashboard/src/types.ts`
- Create: `tools/corvia-dashboard/src/api.ts`
- Create: `tools/corvia-dashboard/src/hooks/use-poll.ts`

-**Step 1: Create `types.ts`**

Mirror the Rust `corvia-common::dashboard` types:

```typescript
// tools/corvia-dashboard/src/types.ts

export type ServiceState = "healthy" | "unhealthy" | "starting" | "stopped";

export interface ServiceStatus {
  name: string;
  state: ServiceState;
  port?: number;
  latency_ms?: number;
}

export interface SpanStats {
  count: number;
  count_1h: number;
  avg_ms: number;
  last_ms: number;
  errors: number;
}

export interface TraceEvent {
  ts: string;
  level: string;
  module: string;
  msg: string;
}

export interface TracesData {
  spans: Record<string, SpanStats>;
  recent_events: TraceEvent[];
}

export interface DashboardConfig {
  embedding_provider: string;
  merge_provider: string;
  storage: string;
  workspace: string;
}

export interface DashboardStatus {
  services: ServiceStatus[];
  entry_count: number;
  agent_count: number;
  merge_queue_depth: number;
  session_count: number;
  config: DashboardConfig;
  traces?: TracesData;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

export interface LogsResponse {
  entries: LogEntry[];
  total: number;
}

export interface TracesResponse {
  spans: Record<string, SpanStats>;
  recent_events: TraceEvent[];
}

/** Trend calculation result */
export interface Trend {
  label: string;
  cls: "up" | "down" | "neutral" | "clear";
}
```

-**Step 2: Create `api.ts`**

```typescript
// tools/corvia-dashboard/src/api.ts

import type {
  DashboardStatus,
  LogsResponse,
  TracesResponse,
  DashboardConfig,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) {
    throw new Error(`API ${path}: ${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

export const api = {
  status: () => get<DashboardStatus>("/api/dashboard/status"),
  traces: () => get<TracesResponse>("/api/dashboard/traces"),
  logs: (params?: { service?: string; module?: string; level?: string; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.service) query.set("service", params.service);
    if (params?.module) query.set("module", params.module);
    if (params?.level) query.set("level", params.level);
    if (params?.limit) query.set("limit", String(params.limit));
    const qs = query.toString();
    return get<LogsResponse>(`/api/dashboard/logs${qs ? `?${qs}` : ""}`);
  },
  config: () => get<DashboardConfig>("/api/dashboard/config"),
};
```

-**Step 3: Create `use-poll.ts`**

```typescript
// tools/corvia-dashboard/src/hooks/use-poll.ts

import { useState, useEffect, useCallback, useRef } from "preact/hooks";

interface UsePollOptions<T> {
  fetcher: () => Promise<T>;
  interval?: number; // ms, default 3000
  enabled?: boolean;
}

interface UsePollResult<T> {
  data: T | null;
  prev: T | null;
  error: Error | null;
  loading: boolean;
  refresh: () => void;
}

export function usePoll<T>({
  fetcher,
  interval = 3000,
  enabled = true,
}: UsePollOptions<T>): UsePollResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [prev, setPrev] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const prevRef = useRef<T | null>(null);

  const doFetch = useCallback(async () => {
    try {
      const result = await fetcher();
      setPrev(prevRef.current);
      prevRef.current = result;
      setData(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    if (!enabled) return;
    doFetch();
    const id = setInterval(doFetch, interval);
    return () => clearInterval(id);
  }, [doFetch, interval, enabled]);

  return { data, prev, error, loading, refresh: doFetch };
}
```

-**Step 4: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/types.ts tools/corvia-dashboard/src/api.ts tools/corvia-dashboard/src/hooks/use-poll.ts
git commit -m "feat(dashboard-ui): add TypeScript types, API client, and polling hook"
```

---

## Chunk 2: Layout & Core Components

### Task 4: Main entry point and hash routing

**Files:**
- Create: `tools/corvia-dashboard/src/main.tsx`

-**Step 1: Create `main.tsx` with hash routing**

```tsx
// tools/corvia-dashboard/src/main.tsx

import { render } from "preact";
import { useState, useEffect } from "preact/hooks";
import "./styles/theme.css";
import { Layout } from "./components/Layout";

type View = "logs" | "graph" | "traces";

function App() {
  const [view, setView] = useState<View>(() => {
    const hash = window.location.hash.slice(2); // remove #/
    return (["logs", "graph", "traces"].includes(hash) ? hash : "logs") as View;
  });

  useEffect(() => {
    const onHash = () => {
      const hash = window.location.hash.slice(2);
      if (["logs", "graph", "traces"].includes(hash)) {
        setView(hash as View);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = (v: View) => {
    window.location.hash = `/${v}`;
    setView(v);
  };

  return <Layout view={view} onNavigate={navigate} />;
}

render(<App />, document.getElementById("app")!);
```

-**Step 2: Verify Vite dev server starts with the entry point**

```bash
cd tools/corvia-dashboard && npx vite --port 8021 &
sleep 2 && curl -s http://localhost:8021/ | grep -c "module"
kill %1
```

Expected: At least 1 match (the script module tag)

-**Step 3: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/main.tsx
git commit -m "feat(dashboard-ui): add entry point with hash routing"
```

---

### Task 5: Layout component shell

**Files:**
- Create: `tools/corvia-dashboard/src/components/Layout.tsx`

-**Step 1: Create `Layout.tsx`**

```tsx
// tools/corvia-dashboard/src/components/Layout.tsx

import type { FunctionComponent } from "preact";
import { api } from "../api";
import { usePoll } from "../hooks/use-poll";
import type { DashboardStatus, Trend } from "../types";
import { Header } from "./Header";
import { MetricsGrid } from "./MetricsGrid";
import { LogsView } from "./LogsView";
import { TracesView } from "./TracesView";
import { GraphView } from "./GraphView";
import { ConfigPanel } from "./ConfigPanel";
import { OfflineState } from "./OfflineState";

type View = "logs" | "graph" | "traces";

interface LayoutProps {
  view: View;
  onNavigate: (v: View) => void;
}

/** Compute trend between current and previous values */
export function trend(current: number | undefined, prev: number | undefined): Trend {
  if (prev == null || current == null) return { label: "-", cls: "neutral" };
  const delta = current - prev;
  if (delta > 0) return { label: `↑ ${delta}`, cls: "up" };
  if (delta === 0) return { label: "stable", cls: "neutral" };
  return { label: `↓ ${Math.abs(delta)}`, cls: "neutral" };
}

export const Layout: FunctionComponent<LayoutProps> = ({ view, onNavigate }) => {
  const { data, prev, error } = usePoll({ fetcher: api.status });

  if (error && !data) {
    return <OfflineState />;
  }

  if (!data) {
    return <div class="loading">Loading...</div>;
  }

  const metrics = [
    {
      label: "Entries",
      value: data.entry_count,
      trend: trend(data.entry_count, prev?.entry_count),
      color: "gold",
    },
    {
      label: "Active Agents",
      value: data.agent_count,
      trend: trend(data.agent_count, prev?.agent_count),
      color: "peach",
    },
    {
      label: "Merge Queue",
      value: data.merge_queue_depth,
      trend:
        data.merge_queue_depth === 0
          ? { label: "clear", cls: "clear" as const }
          : trend(data.merge_queue_depth, prev?.merge_queue_depth),
      color: "mint",
    },
    {
      label: "Sessions",
      value: data.session_count,
      trend: trend(data.session_count, prev?.session_count),
      color: "lavender",
    },
  ];

  return (
    <div class="dashboard">
      <Header services={data.services} scope={data.config.workspace} />
      <MetricsGrid metrics={metrics} />
      <div class="workspace">
        <div class="main-panel">
          <nav class="view-tabs">
            {(["logs", "graph", "traces"] as View[]).map((v) => (
              <button
                key={v}
                class={`view-tab ${view === v ? "active" : ""}`}
                onClick={() => onNavigate(v)}
              >
                {v.charAt(0).toUpperCase() + v.slice(1)}
              </button>
            ))}
          </nav>
          {view === "logs" && <LogsView />}
          {view === "graph" && <GraphView />}
          {view === "traces" && <TracesView traces={data.traces} />}
        </div>
        <aside class="sidebar">
          <ConfigPanel config={data.config} />
        </aside>
      </div>
    </div>
  );
};
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/Layout.tsx
git commit -m "feat(dashboard-ui): add Layout shell with routing and trend calculation"
```

---

### Task 6: Header component

**Files:**
- Create: `tools/corvia-dashboard/src/components/Header.tsx`

-**Step 1: Create `Header.tsx`**

```tsx
// tools/corvia-dashboard/src/components/Header.tsx

import type { FunctionComponent } from "preact";
import type { ServiceStatus } from "../types";

interface HeaderProps {
  services: ServiceStatus[];
  scope: string;
}

const stateColor = (state: string): string => {
  switch (state) {
    case "healthy": return "var(--mint)";
    case "unhealthy": return "var(--coral)";
    case "starting": return "var(--amber)";
    default: return "var(--text-dim)";
  }
};

export const Header: FunctionComponent<HeaderProps> = ({ services, scope }) => {
  const now = new Date().toLocaleTimeString("en-US", { hour12: false });

  return (
    <header class="header">
      <div class="header-left">
        <div class="brand">
          <div class="brand-icon">C</div>
          <span class="brand-text">Corvia</span>
        </div>
        <div class="status-pills">
          {services.map((svc) => (
            <div key={svc.name} class="status-pill">
              <span
                class="status-dot"
                style={{ background: stateColor(svc.state) }}
              />
              <span class="pill-label">{svc.name.replace("corvia-", "")}</span>
            </div>
          ))}
        </div>
      </div>
      <div class="header-right">
        <span class="timestamp">{now}</span>
        <span class="scope-badge">{scope}</span>
      </div>
    </header>
  );
};
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/Header.tsx
git commit -m "feat(dashboard-ui): add Header with status pills and scope badge"
```

---

### Task 7: MetricsGrid component

**Files:**
- Create: `tools/corvia-dashboard/src/components/MetricsGrid.tsx`

-**Step 1: Create `MetricsGrid.tsx`**

```tsx
// tools/corvia-dashboard/src/components/MetricsGrid.tsx

import type { FunctionComponent } from "preact";
import type { Trend } from "../types";

interface MetricCard {
  label: string;
  value: number;
  trend: Trend;
  color: string; // "gold" | "peach" | "mint" | "lavender"
}

interface MetricsGridProps {
  metrics: MetricCard[];
}

const icons: Record<string, string> = {
  gold: "📋",
  peach: "👤",
  mint: "📦",
  lavender: "🔐",
};

const trendColor = (cls: string): string => {
  switch (cls) {
    case "up": return "var(--mint)";
    case "clear": return "var(--mint)";
    case "down": return "var(--coral)";
    default: return "var(--gold)";
  }
};

export const MetricsGrid: FunctionComponent<MetricsGridProps> = ({ metrics }) => (
  <div class="metrics-grid">
    {metrics.map((m) => (
      <div key={m.label} class="metric-card" style={{ borderTopColor: `var(--${m.color})` }}>
        <div class="metric-icon" style={{ background: `var(--${m.color}-soft)` }}>
          {icons[m.color] ?? "•"}
        </div>
        <div class="metric-label">{m.label}</div>
        <div class="metric-value">{m.value.toLocaleString()}</div>
        <div class="metric-trend" style={{ color: trendColor(m.trend.cls) }}>
          {m.trend.label}
        </div>
      </div>
    ))}
  </div>
);
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/MetricsGrid.tsx
git commit -m "feat(dashboard-ui): add MetricsGrid with trend pills"
```

---

## Chunk 3: Log & Trace Views

### Task 8: LogsView component

**Files:**
- Create: `tools/corvia-dashboard/src/components/LogsView.tsx`

-**Step 1: Create `LogsView.tsx`**

```tsx
// tools/corvia-dashboard/src/components/LogsView.tsx

import type { FunctionComponent } from "preact";
import { useState, useRef, useEffect } from "preact/hooks";
import { api } from "../api";
import { usePoll } from "../hooks/use-poll";
import type { LogEntry } from "../types";

type Level = "all" | "info" | "warn" | "error";

const levelColor = (level: string): string => {
  switch (level) {
    case "warn": return "var(--amber)";
    case "error": return "var(--coral)";
    case "debug": return "var(--text-dim)";
    default: return "var(--mint)";
  }
};

export const LogsView: FunctionComponent = () => {
  const [filter, setFilter] = useState<Level>("all");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const { data } = usePoll({
    fetcher: () => api.logs({ limit: 200 }),
  });

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [data, autoScroll]);

  const entries = (data?.entries ?? []).filter((e: LogEntry) => {
    if (filter !== "all" && e.level !== filter) return false;
    if (search && !e.message.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  return (
    <div class="logs-view">
      <div class="log-toolbar">
        <div class="level-filters">
          {(["all", "info", "warn", "error"] as Level[]).map((l) => (
            <button
              key={l}
              class={`filter-btn ${filter === l ? "active" : ""}`}
              onClick={() => setFilter(l)}
            >
              {l.charAt(0).toUpperCase() + l.slice(1)}
            </button>
          ))}
        </div>
        <input
          class="search-input"
          type="text"
          placeholder="Search logs..."
          value={search}
          onInput={(e) => setSearch((e.target as HTMLInputElement).value)}
        />
        <button
          class={`auto-scroll-btn ${autoScroll ? "active" : ""}`}
          onClick={() => setAutoScroll(!autoScroll)}
          title="Auto-scroll"
        >
          ⬇
        </button>
      </div>
      <div class="log-output" ref={logRef}>
        {entries.map((e: LogEntry, i: number) => (
          <div
            key={i}
            class={`log-line ${e.level === "error" || e.level === "warn" ? `log-${e.level}` : ""}`}
          >
            <span class="log-ts">[{e.timestamp}]</span>
            <span class="log-level" style={{ color: levelColor(e.level) }}>
              [{e.level.toUpperCase()}]
            </span>
            <span class="log-msg">{e.message}</span>
          </div>
        ))}
        {entries.length === 0 && (
          <div class="log-empty">No log entries{filter !== "all" ? ` at ${filter} level` : ""}</div>
        )}
      </div>
    </div>
  );
};
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/LogsView.tsx
git commit -m "feat(dashboard-ui): add LogsView with filtering and auto-scroll"
```

---

### Task 9: TracesView component

**Files:**
- Create: `tools/corvia-dashboard/src/components/TracesView.tsx`

-**Step 1: Create `TracesView.tsx` with module topology and detail panel**

```tsx
// tools/corvia-dashboard/src/components/TracesView.tsx

import type { FunctionComponent } from "preact";
import { useState } from "preact/hooks";
import type { TracesData, SpanStats, TraceEvent } from "../types";

interface TracesViewProps {
  traces?: TracesData;
}

interface ModuleDef {
  name: string;
  label: string;
  emoji: string;
  color: string;
  desc: string;
  x: string;
  y: string;
}

const MODULES: ModuleDef[] = [
  { name: "agent", label: "Agent", emoji: "🤖", color: "peach", desc: "Agent registration & session lifecycle", x: "18%", y: "6%" },
  { name: "entry", label: "Entry", emoji: "📝", color: "gold", desc: "Write, embed, insert pipeline", x: "50%", y: "4%" },
  { name: "merge", label: "Merge", emoji: "🔄", color: "mint", desc: "Conflict detection & resolution", x: "50%", y: "50%" },
  { name: "storage", label: "Storage", emoji: "💾", color: "lavender", desc: "LiteStore / Postgres persistence", x: "82%", y: "4%" },
  { name: "rag", label: "RAG", emoji: "🔍", color: "sky", desc: "Retrieval-augmented generation", x: "82%", y: "50%" },
  { name: "inference", label: "Inference", emoji: "⚡", color: "coral", desc: "ONNX embedding via gRPC", x: "18%", y: "50%" },
  { name: "gc", label: "GC", emoji: "🧹", color: "amber", desc: "Garbage collection sweeps", x: "50%", y: "72%" },
];

/** Get span stats for a module by matching span names */
function moduleSpans(
  module: string,
  spans: Record<string, SpanStats>
): [string, SpanStats][] {
  return Object.entries(spans).filter(([name]) => {
    // Check if this span belongs to the module
    const spanPrefixes: Record<string, string[]> = {
      agent: ["corvia.agent.", "corvia.session."],
      entry: ["corvia.entry."],
      merge: ["corvia.merge."],
      storage: ["corvia.store."],
      rag: ["corvia.rag."],
      gc: ["corvia.gc."],
      inference: ["corvia.entry.embed"],
    };
    const prefixes = spanPrefixes[module] ?? [];
    return prefixes.some((p) => name.startsWith(p));
  });
}

const latencyColor = (ms: number): string => {
  if (ms < 50) return "var(--mint)";
  if (ms < 150) return "var(--peach)";
  return "var(--coral)";
};

export const TracesView: FunctionComponent<TracesViewProps> = ({ traces }) => {
  const [selected, setSelected] = useState<string | null>(null);

  const spans = traces?.spans ?? {};
  const events = traces?.recent_events ?? [];

  const selectedModule = MODULES.find((m) => m.name === selected);
  const selectedSpans = selected ? moduleSpans(selected, spans) : [];
  const selectedEvents = selected
    ? events.filter((e) => e.module === selected)
    : [];

  return (
    <div class="traces-view">
      <div class="topology-graph">
        {MODULES.map((mod) => {
          const modSpans = moduleSpans(mod.name, spans);
          const totalCount = modSpans.reduce((s, [, st]) => s + st.count, 0);

          return (
            <div
              key={mod.name}
              class={`tnode ${selected === mod.name ? "selected" : ""}`}
              style={{ left: mod.x, top: mod.y }}
              onClick={() => setSelected(selected === mod.name ? null : mod.name)}
            >
              <div class="tnode-header">
                <span class="tnode-emoji">{mod.emoji}</span>
                <span class="tnode-label">{mod.label}</span>
              </div>
              <div class="tnode-count" style={{ color: `var(--${mod.color})` }}>
                {totalCount > 0 ? totalCount.toLocaleString() : "—"}
              </div>
            </div>
          );
        })}
      </div>

      {selectedModule && (
        <div class="traces-detail">
          <div class="detail-header">
            <span
              class="detail-dot"
              style={{ background: `var(--${selectedModule.color})` }}
            />
            <h3>{selectedModule.label}</h3>
            <p class="detail-desc">{selectedModule.desc}</p>
          </div>

          <div class="detail-stats">
            <div class="stat">
              <span class="stat-label">Total</span>
              <span class="stat-value">
                {selectedSpans.reduce((s, [, st]) => s + st.count, 0)}
              </span>
            </div>
            <div class="stat">
              <span class="stat-label">Last hour</span>
              <span class="stat-value">
                {selectedSpans.reduce((s, [, st]) => s + st.count_1h, 0)}
              </span>
            </div>
            <div class="stat">
              <span class="stat-label">Avg latency</span>
              <span
                class="stat-value"
                style={{
                  color: latencyColor(
                    selectedSpans.length > 0
                      ? selectedSpans.reduce((s, [, st]) => s + st.avg_ms, 0) /
                          selectedSpans.length
                      : 0
                  ),
                }}
              >
                {selectedSpans.length > 0
                  ? (
                      selectedSpans.reduce((s, [, st]) => s + st.avg_ms, 0) /
                      selectedSpans.length
                    ).toFixed(1)
                  : "—"}
                ms
              </span>
            </div>
            <div class="stat">
              <span class="stat-label">Errors</span>
              <span class="stat-value" style={{ color: "var(--coral)" }}>
                {selectedSpans.reduce((s, [, st]) => s + st.errors, 0)}
              </span>
            </div>
          </div>

          {selectedSpans.length > 0 && (
            <div class="detail-card">
              <h4>Instrumented Spans</h4>
              {selectedSpans.map(([name, st]) => (
                <div key={name} class="span-row">
                  <span class="span-name">{name}</span>
                  <span
                    class="span-latency"
                    style={{ color: latencyColor(st.avg_ms) }}
                  >
                    {st.avg_ms.toFixed(1)}ms
                  </span>
                </div>
              ))}
            </div>
          )}

          {selectedEvents.length > 0 && (
            <div class="detail-card">
              <h4>Recent Events</h4>
              {selectedEvents.slice(-10).map((ev, i) => (
                <div key={i} class="event-row">
                  <span
                    class="event-dot"
                    style={{
                      background:
                        ev.level === "error"
                          ? "var(--coral)"
                          : ev.level === "warn"
                          ? "var(--amber)"
                          : "var(--mint)",
                    }}
                  />
                  <span class="event-msg">{ev.msg}</span>
                  <span class="event-ts">{ev.ts}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/TracesView.tsx
git commit -m "feat(dashboard-ui): add TracesView with module topology and detail panel"
```

---

## Chunk 4: Remaining Components & Polish

### Task 10: GraphView placeholder

**Files:**
- Create: `tools/corvia-dashboard/src/components/GraphView.tsx`

-**Step 1: Create `GraphView.tsx`**

```tsx
// tools/corvia-dashboard/src/components/GraphView.tsx

import type { FunctionComponent } from "preact";

export const GraphView: FunctionComponent = () => (
  <div class="graph-view">
    <div class="graph-placeholder">
      <div class="placeholder-icon">🔗</div>
      <h3>Knowledge Graph</h3>
      <p class="placeholder-text">
        Graph visualization coming soon. Use{" "}
        <code>corvia_graph</code> MCP tool for graph queries.
      </p>
    </div>
  </div>
);
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/GraphView.tsx
git commit -m "feat(dashboard-ui): add GraphView placeholder"
```

---

### Task 11: ConfigPanel component

**Files:**
- Create: `tools/corvia-dashboard/src/components/ConfigPanel.tsx`

-**Step 1: Create `ConfigPanel.tsx`**

```tsx
// tools/corvia-dashboard/src/components/ConfigPanel.tsx

import type { FunctionComponent } from "preact";
import type { DashboardConfig } from "../types";

interface ConfigPanelProps {
  config: DashboardConfig;
}

export const ConfigPanel: FunctionComponent<ConfigPanelProps> = ({ config }) => (
  <div class="config-panel">
    <h3 class="config-title">Configuration</h3>
    <div class="config-rows">
      <div class="config-row">
        <span class="config-key">Embedding</span>
        <span class="config-value">{config.embedding_provider}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Merge LLM</span>
        <span class="config-value">{config.merge_provider}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Storage</span>
        <span class="config-value">{config.storage}</span>
      </div>
      <div class="config-row">
        <span class="config-key">Workspace</span>
        <span class="config-value">
          {config.workspace}
          <span class="synced-badge">✓ Synced</span>
        </span>
      </div>
    </div>
  </div>
);
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/ConfigPanel.tsx
git commit -m "feat(dashboard-ui): add ConfigPanel sidebar"
```

---

### Task 12: OfflineState component

**Files:**
- Create: `tools/corvia-dashboard/src/components/OfflineState.tsx`

-**Step 1: Create `OfflineState.tsx`**

```tsx
// tools/corvia-dashboard/src/components/OfflineState.tsx

import type { FunctionComponent } from "preact";

export const OfflineState: FunctionComponent = () => (
  <div class="offline-state">
    <div class="offline-icon">⚠</div>
    <h2>corvia-server not responding</h2>
    <p>
      Run <code>corvia serve</code> to start the API server,
      or check that it's running on port 8020.
    </p>
  </div>
);
```

-**Step 2: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/OfflineState.tsx
git commit -m "feat(dashboard-ui): add OfflineState fallback"
```

---

### Task 13: Component CSS

**Files:**
- Modify: `tools/corvia-dashboard/src/styles/theme.css`

-**Step 1: Add component styles to `theme.css`**

Append layout and component styles ported from `extension.js` (lines 192-688). Key sections:

```css
/* Append to tools/corvia-dashboard/src/styles/theme.css */

/* --- Layout --- */
.dashboard {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  color: var(--text-muted);
}

/* --- Header --- */
.header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border);
}

.header-left, .header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-icon {
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, var(--gold), var(--gold-bright));
  color: #12141a;
  font-weight: 800;
  font-size: 16px;
}

.brand-text {
  font-weight: 700;
  font-size: 15px;
  color: var(--text-bright);
}

.status-pills {
  display: flex;
  gap: 8px;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 999px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
  color: var(--text-muted);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.timestamp {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-dim);
}

.scope-badge {
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-muted);
  font-weight: 600;
}

/* --- Metrics Grid --- */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding: 22px 24px;
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 3px solid;
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-card);
  transition: box-shadow var(--transition);
}

.metric-card:hover {
  box-shadow: var(--shadow-hover);
}

.metric-icon {
  width: 38px;
  height: 38px;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  margin-bottom: 12px;
}

.metric-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.metric-value {
  font-size: 30px;
  font-weight: 800;
  color: var(--text-bright);
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
}

.metric-trend {
  font-size: 11px;
  font-weight: 600;
  margin-top: 4px;
}

/* --- Workspace (70/30 split) --- */
.workspace {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 16px;
  padding: 0 24px 24px;
  flex: 1;
}

.main-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* --- View Tabs --- */
.view-tabs {
  display: flex;
  gap: 0;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
}

.view-tab {
  background: none;
  border: none;
  padding: 10px 20px;
  font-family: var(--font-ui);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  border-bottom: 2.5px solid transparent;
  transition: color var(--transition), border-color var(--transition);
}

.view-tab:hover {
  color: var(--text-bright);
}

.view-tab.active {
  color: var(--gold);
  border-bottom-color: var(--gold);
}

/* --- Logs View --- */
.logs-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.log-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
}

.level-filters {
  display: flex;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  overflow: hidden;
}

.filter-btn {
  background: var(--bg-input);
  border: none;
  padding: 5px 12px;
  font-family: var(--font-ui);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  transition: all var(--transition);
}

.filter-btn.active {
  background: var(--gold-medium);
  color: var(--gold-bright);
}

.search-input {
  flex: 1;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 5px 10px;
  font-family: var(--font-ui);
  font-size: 12px;
  color: var(--text-primary);
  outline: none;
}

.search-input:focus {
  border-color: var(--gold);
}

.auto-scroll-btn {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 4px 8px;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 12px;
}

.auto-scroll-btn.active {
  background: var(--gold-medium);
  color: var(--gold-bright);
}

.log-output {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  padding: 12px 16px;
  overflow-y: auto;
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.9;
  min-height: 300px;
  max-height: 60vh;
}

.log-line {
  padding: 1px 8px;
  border-radius: 3px;
  transition: background var(--transition);
}

.log-line:hover {
  background: var(--bg-card-hover);
}

.log-warn {
  border-left: 3px solid var(--amber);
  background: var(--amber-soft);
}

.log-error {
  border-left: 3px solid var(--coral);
  background: var(--coral-soft);
}

.log-ts {
  color: var(--text-dim);
  opacity: 0.6;
  margin-right: 6px;
}

.log-level {
  font-weight: 600;
  margin-right: 6px;
}

.log-msg {
  color: var(--text-primary);
}

.log-empty {
  color: var(--text-dim);
  text-align: center;
  padding: 40px;
  font-family: var(--font-ui);
}

/* --- Traces View --- */
.traces-view {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 16px;
  flex: 1;
}

.topology-graph {
  position: relative;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  min-height: 400px;
  padding: 20px;
}

.tnode {
  position: absolute;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  cursor: pointer;
  transition: all var(--transition);
  min-width: 100px;
  text-align: center;
}

.tnode:hover {
  border-color: var(--border-bright);
  box-shadow: var(--shadow-hover);
}

.tnode.selected {
  box-shadow: 0 0 0 3px var(--gold-soft);
}

.tnode-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-bottom: 4px;
}

.tnode-emoji {
  font-size: 16px;
}

.tnode-label {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-bright);
}

.tnode-count {
  font-size: 18px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

/* --- Traces Detail Panel --- */
.traces-detail {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-header h3 {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-bright);
  display: flex;
  align-items: center;
  gap: 8px;
}

.detail-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.detail-desc {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}

.detail-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.stat {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px;
}

.stat-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
  color: var(--text-muted);
  display: block;
}

.stat-value {
  font-size: 18px;
  font-weight: 800;
  color: var(--text-bright);
  font-variant-numeric: tabular-nums;
}

.detail-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 14px;
}

.detail-card h4 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 10px;
}

.span-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  border-bottom: 1px solid var(--border-subtle);
}

.span-name {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-primary);
}

.span-latency {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
}

.event-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 11px;
}

.event-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.event-msg {
  flex: 1;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.event-ts {
  font-family: var(--font-mono);
  color: var(--text-dim);
  font-size: 10px;
}

/* --- Graph View (placeholder) --- */
.graph-view {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.graph-placeholder {
  text-align: center;
  color: var(--text-muted);
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.graph-placeholder h3 {
  font-size: 18px;
  color: var(--text-bright);
  margin-bottom: 8px;
}

.placeholder-text {
  font-size: 13px;
}

.placeholder-text code {
  font-family: var(--font-mono);
  background: var(--bg-input);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

/* --- Config Panel --- */
.config-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
}

.config-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 16px;
}

.config-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-subtle);
}

.config-key {
  font-size: 12px;
  color: var(--text-muted);
}

.config-value {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-bright);
  display: flex;
  align-items: center;
  gap: 8px;
}

.synced-badge {
  font-size: 10px;
  color: var(--mint);
  background: var(--mint-medium);
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 600;
}

/* --- Offline State --- */
.offline-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  text-align: center;
  gap: 12px;
}

.offline-icon {
  font-size: 48px;
  color: var(--amber);
}

.offline-state h2 {
  font-size: 18px;
  color: var(--text-bright);
}

.offline-state p {
  font-size: 13px;
  color: var(--text-muted);
  max-width: 400px;
}

.offline-state code {
  font-family: var(--font-mono);
  background: var(--bg-input);
  padding: 2px 6px;
  border-radius: 4px;
}

/* --- Responsive --- */
@media (max-width: 700px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .traces-view {
    grid-template-columns: 1fr;
  }
}
```

-**Step 2: Verify Vite compiles without errors**

```bash
cd tools/corvia-dashboard && npx vite build
```

Expected: Build succeeds, output in `dist/`

-**Step 3: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/styles/theme.css
git commit -m "feat(dashboard-ui): add full component CSS (ported from extension)"
```

---

### Task 14: VS Code extension retrofit

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js`

The current `extension.js` is ~1,477 lines. The retrofit keeps ~lines 1-111 (activation, panel lifecycle, status bar, command registration) and replaces the rest (the embedded HTML dashboard) with a thin iframe wrapper.

-**Step 1: Read `extension.js` and identify boundaries**

Read `.devcontainer/extensions/corvia-services/extension.js`.

**Keep** (lines 1-111):
- `activate()` function — panel creation, command registration, status bar setup
- `deactivate()` function
- Panel lifecycle (`createOrShow`, `onDidDispose`, `retainContextWhenHidden`)
- Status bar item creation

**Remove/replace**:
- The `getDashboardHtml()` function (or equivalent content-generation function — everything after panel lifecycle)
- The `pollStatus()` internals that shell-exec `corvia-dev status --json`

-**Step 2: Replace `getDashboardHtml()` with iframe wrapper**

Delete the entire HTML blob function body and replace with:

```javascript
function getWebviewContent() {
  return `<!DOCTYPE html>
<html>
<head>
  <style>
    body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
    iframe { width: 100%; height: 100%; border: none; }
  </style>
</head>
<body>
  <iframe src="http://localhost:8021" />
</body>
</html>`;
}
```

-**Step 3: Replace status bar polling to use fetch instead of shell exec**

Find the polling function that calls `exec('corvia-dev status --json')` and replace its body with:

```javascript
async function pollStatus() {
  try {
    const resp = await fetch('http://localhost:8020/api/dashboard/status');
    const data = await resp.json();
    const allHealthy = data.services.every(s => s.state === 'healthy');
    statusBarItem.text = allHealthy ? '$(check) Corvia' : '$(warning) Corvia';
    statusBarItem.backgroundColor = allHealthy
      ? undefined
      : new vscode.ThemeColor('statusBarItem.errorBackground');
  } catch {
    statusBarItem.text = '$(error) Corvia';
    statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
  }
}
```

-**Step 4: Remove the `webview.onDidReceiveMessage` handler**

The current extension handles messages from the dashboard webview (commands, refresh). Since the standalone dashboard handles its own state, remove the message handler entirely. The extension no longer needs to proxy commands.

-**Step 5: Verify the extension activates**

Open VS Code, check that:
- Status bar shows "Corvia" with health indicator
- Clicking opens a webview panel loading `http://localhost:8021`
- No console errors

-**Step 6: Commit**

```bash
cd /workspaces/corvia-workspace
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): retrofit to embed standalone dashboard via iframe"
```

---

### Task 15: Playwright smoke test

**Files:**
- Create: `tools/corvia-dashboard/src/test/dashboard.spec.ts`
- Create: `tools/corvia-dashboard/playwright.config.ts`

-**Step 1: Create `playwright.config.ts`**

```typescript
// tools/corvia-dashboard/playwright.config.ts

import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./src/test",
  webServer: {
    command: "npm run dev",
    port: 8021,
    reuseExistingServer: true,
  },
  use: {
    baseURL: "http://localhost:8021",
  },
});
```

-**Step 2: Create smoke test**

```typescript
// tools/corvia-dashboard/src/test/dashboard.spec.ts

import { test, expect } from "@playwright/test";

test.describe("Dashboard", () => {
  test("shows header with brand", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".brand-text")).toHaveText("Corvia");
  });

  test("shows metrics grid with 4 cards", async ({ page }) => {
    await page.goto("/");
    const cards = page.locator(".metric-card");
    await expect(cards).toHaveCount(4);
  });

  test("navigates between views via tabs", async ({ page }) => {
    await page.goto("/");

    // Default is logs
    await expect(page.locator(".logs-view")).toBeVisible();

    // Switch to traces
    await page.click('button:text("Traces")');
    await expect(page.locator(".traces-view")).toBeVisible();

    // Hash should update
    expect(page.url()).toContain("#/traces");
  });

  test("shows offline state when API is down", async ({ page }) => {
    // Block API requests
    await page.route("**/api/dashboard/**", (route) => route.abort());
    await page.goto("/");

    await expect(page.locator(".offline-state")).toBeVisible({ timeout: 10000 });
  });

  test("logs view shows filter buttons", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".filter-btn")).toHaveCount(4); // all, info, warn, error
  });

  test("traces view shows module topology nodes", async ({ page }) => {
    await page.goto("/#/traces");
    const nodes = page.locator(".tnode");
    await expect(nodes).toHaveCount(7); // 7 modules
  });
});
```

-**Step 3: Install Playwright browsers**

```bash
cd tools/corvia-dashboard && npx playwright install chromium
```

-**Step 4: Run tests**

```bash
cd tools/corvia-dashboard && npx playwright test
```

Expected: Tests that interact with the API may fail without a running backend (expected). Tests for static structure (offline state, button counts, navigation) should pass.

-**Step 5: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/playwright.config.ts tools/corvia-dashboard/src/test/
git commit -m "feat(dashboard-ui): add Playwright smoke tests"
```

---

### Task 16: Final verification

-**Step 1: Start corvia-server (for API)**

```bash
cd /workspaces/corvia-workspace && corvia serve &
```

-**Step 2: Start dashboard dev server**

```bash
cd tools/corvia-dashboard && npm run dev &
```

-**Step 3: Open dashboard in browser**

Navigate to `http://localhost:8021`

Verify:
- Header shows with brand and status pills
- Metrics grid shows 4 cards with data from API
- Logs tab shows structured log entries
- Traces tab shows 7 module topology nodes
- Config sidebar shows current settings
- 3-second polling updates data

-**Step 4: Run full Playwright suite**

```bash
cd tools/corvia-dashboard && npx playwright test
```

Expected: All tests PASS

-**Step 5: Final commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/ tools/corvia-dashboard/playwright.config.ts
git commit -m "feat(dashboard-ui): standalone dashboard with full parity — logs, traces, config"
```
