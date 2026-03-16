# Dashboard Redesign Implementation Plan

> **Status:** Shipped (v0.3.7)

**Goal:** Rewrite the Corvia VS Code extension dashboard with a warm dark Figma-inspired design, M4 metrics, view tabs, and interactive log terminal.

**Architecture:** Single-file rewrite of `extension.js` — the extension host code (activate, refresh, openDashboard, deactivate) stays nearly identical. The `getDashboardHtml()` function is fully replaced with new CSS, HTML skeleton, and JavaScript. The existing `corvia-dev status --json` polling contract is unchanged.

**Tech Stack:** VS Code Webview API, vanilla HTML/CSS/JS (embedded in extension.js template literal), Inter font from Google Fonts.

**Spec:** `2026-03-10-dashboard-redesign-design.md`
**Mockup:** `.superpowers/brainstorm/layout-v5.html`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `.devcontainer/extensions/corvia-services/extension.js` | Rewrite | Extension host + embedded dashboard UI |
| `.devcontainer/extensions/corvia-services/package.json` | Modify | Version bump 0.2.0 → 0.3.0 |

The extension host code (lines 1-107 and 700-706 of current file) is preserved with minor edits. The `getDashboardHtml()` function (lines 108-699) is fully replaced.

---

## Chunk 1: Extension Host + Package.json

### Task 1: Version bump

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/package.json`

-**Step 1: Update version**

Change `"version": "0.2.0"` to `"version": "0.3.0"` in package.json.

-**Step 2: Commit**

```bash
git add .devcontainer/extensions/corvia-services/package.json
git commit -m "chore(extension): bump version to 0.3.0 for dashboard redesign"
```

### Task 2: Update extension host code

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js` (lines 1-107)

The host code is almost identical. Two changes:

1. Track previous poll data for metric trends
2. Add `entry_count`, `agent_count`, `session_count`, `merge_queue_depth` to the data posted to webview (these fields may already be in the JSON from `corvia-dev status --json` — if not, the webview handles missing fields gracefully with `?? '-'`)

-**Step 1: Add `prevData` tracking**

At the top of the file (after line 6), add:

```javascript
let prevData = null;
```

-**Step 2: Update refresh() to track previous data and post it**

In the `refresh()` function, after `data = JSON.parse(raw)` and before the status bar logic, add:

```javascript
    data._prev = prevData;
    prevData = { ...data, _prev: undefined };
```

This passes the previous poll snapshot to the webview so it can compute metric trends (e.g., entry count delta).

-**Step 3: Verify host code compiles**

Open the extension in VS Code (it auto-loads from `.devcontainer/extensions/`). Confirm no syntax errors in the developer console (`Help > Toggle Developer Tools`).

-**Step 4: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): track previous poll data for metric trends"
```

---

## Chunk 2: CSS — Design Tokens & Component Styles

### Task 3: Replace the entire CSS block in getDashboardHtml()

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js` — the `<style>` block inside `getDashboardHtml()`

Replace everything between `<style>` and `</style>` (current lines 114-390) with the new design token system and all component styles. The CSS is adapted directly from the approved mockup (`.superpowers/brainstorm/layout-v5.html`) with these adjustments for the webview context:

1. Inter font loaded via `@import` inside `<style>` (VS Code webviews support external CSS imports)
2. All VS Code theme variable fallbacks removed — we own the full palette now
3. Additional styles for offline state, skeleton loading, and dynamic elements not in the static mockup

-**Step 1: Write the complete CSS**

Replace the `<style>...</style>` block with the following. This is the full CSS — copy it exactly:

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-primary: #12141a;
  --bg-elevated: #1a1d26;
  --bg-card: #1e2230;
  --bg-card-hover: #252a3a;
  --bg-input: #282d3e;
  --bg-surface: #2e3447;

  --gold: #f0c94c;
  --gold-bright: #ffe066;
  --gold-soft: rgba(240, 201, 76, 0.10);
  --gold-medium: rgba(240, 201, 76, 0.18);
  --gold-glow: rgba(240, 201, 76, 0.25);

  --mint: #5eead4;
  --mint-soft: rgba(94, 234, 212, 0.10);
  --mint-medium: rgba(94, 234, 212, 0.16);

  --coral: #ff8a80;
  --coral-soft: rgba(255, 138, 128, 0.08);
  --coral-medium: rgba(255, 138, 128, 0.14);

  --peach: #ffb07c;
  --peach-soft: rgba(255, 176, 124, 0.10);
  --peach-medium: rgba(255, 176, 124, 0.16);

  --lavender: #c4b5fd;
  --lavender-soft: rgba(196, 181, 253, 0.10);
  --lavender-medium: rgba(196, 181, 253, 0.16);

  --amber: #fcd34d;
  --amber-soft: rgba(252, 211, 77, 0.10);
  --amber-medium: rgba(252, 211, 77, 0.16);

  --text-bright: #f2f0ed;
  --text-primary: #c5c0b8;
  --text-muted: #918b82;
  --text-dim: #615c55;

  --border: rgba(80, 75, 68, 0.4);
  --border-bright: rgba(100, 94, 86, 0.45);
  --border-subtle: rgba(65, 60, 54, 0.35);

  --radius-xl: 20px;
  --radius-lg: 16px;
  --radius-md: 12px;
  --radius-sm: 8px;
  --radius-xs: 6px;

  --shadow-card: 0 4px 20px rgba(0,0,0,0.2), 0 0 1px rgba(255,255,255,0.03) inset;
  --shadow-hover: 0 8px 32px rgba(0,0,0,0.3);
  --shadow-gold: 0 4px 20px rgba(240,201,76,0.10);

  --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', monospace;
  --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

body {
  font-family: var(--font-ui); background: var(--bg-primary);
  color: var(--text-primary); font-size: 13px; line-height: 1.5;
  -webkit-font-smoothing: antialiased; min-height: 100vh;
  background-image:
    radial-gradient(ellipse 600px 400px at 8% 0%, rgba(240,201,76,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 500px 400px at 92% 0%, rgba(94,234,212,0.025) 0%, transparent 70%);
}

/* ===== Header ===== */
.header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 28px; background: rgba(18,20,26,0.8);
  backdrop-filter: blur(16px); border-bottom: 1px solid var(--border-subtle);
  position: sticky; top: 0; z-index: 10;
}
.header-left { display: flex; align-items: center; gap: 20px; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon {
  width: 30px; height: 30px; border-radius: var(--radius-sm);
  background: linear-gradient(135deg, var(--gold), #d4a820);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 14px; color: var(--bg-primary);
  box-shadow: 0 2px 12px rgba(240,201,76,0.35);
}
.brand-name { font-size: 17px; font-weight: 700; color: var(--text-bright); letter-spacing: -0.02em; }

.status-pills { display: flex; gap: 8px; }
.pill {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 16px; background: var(--bg-card);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  font-size: 11px; font-weight: 500; color: var(--text-muted);
  cursor: default; transition: all var(--transition); box-shadow: var(--shadow-card);
}
.pill:hover { background: var(--bg-card-hover); border-color: var(--mint-medium); }
.pill:hover .pill-restart { opacity: 1; }
.pill-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.pill-dot.ok {
  background: var(--mint);
  box-shadow: 0 0 10px rgba(94,234,212,0.6), 0 0 3px rgba(94,234,212,0.8);
  animation: glow-pulse 2.5s ease-in-out infinite;
}
.pill-dot.down { background: var(--coral); box-shadow: 0 0 10px rgba(255,138,128,0.6); }
.pill-dot.warn { background: var(--amber); box-shadow: 0 0 8px rgba(252,211,77,0.5); }

@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 10px rgba(94,234,212,0.5), 0 0 3px rgba(94,234,212,0.7); }
  50% { box-shadow: 0 0 16px rgba(94,234,212,0.7), 0 0 6px rgba(94,234,212,1); }
}

.pill-label { text-transform: uppercase; letter-spacing: 0.05em; font-size: 10px; font-weight: 600; }
.pill-restart {
  opacity: 0; color: var(--text-dim); cursor: pointer;
  transition: all var(--transition); display: flex;
}
.pill-restart:hover { color: var(--gold); }

.header-right { display: flex; align-items: center; gap: 14px; }
.header-time { font-family: var(--font-mono); font-size: 11px; color: var(--text-dim); }
.scope-badge {
  font-size: 11px; color: var(--text-muted); background: var(--bg-input);
  padding: 4px 16px; border-radius: 99px; font-weight: 500; border: 1px solid var(--border-subtle);
}

/* ===== Metrics ===== */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 22px 28px; }
.metric-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 20px 22px;
  transition: all var(--transition); box-shadow: var(--shadow-card);
  position: relative; overflow: hidden;
}
.metric-card::after {
  content: ''; position: absolute; top: 0; left: 16px; right: 16px;
  height: 3px; border-radius: 0 0 3px 3px;
}
.metric-card.gold::after { background: linear-gradient(90deg, var(--gold), var(--gold-bright)); box-shadow: 0 2px 12px rgba(240,201,76,0.25); }
.metric-card.mint::after { background: linear-gradient(90deg, var(--mint), #7df4e1); box-shadow: 0 2px 12px rgba(94,234,212,0.25); }
.metric-card.peach::after { background: linear-gradient(90deg, var(--peach), #ffc99e); box-shadow: 0 2px 12px rgba(255,176,124,0.25); }
.metric-card.lavender::after { background: linear-gradient(90deg, var(--lavender), #d4c5fe); box-shadow: 0 2px 12px rgba(196,181,253,0.25); }
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); border-color: var(--border-bright); }

.metric-icon {
  width: 38px; height: 38px; border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center; margin-bottom: 14px;
}
.metric-icon.gold { background: var(--gold-soft); color: var(--gold); }
.metric-icon.mint { background: var(--mint-soft); color: var(--mint); }
.metric-icon.peach { background: var(--peach-soft); color: var(--peach); }
.metric-icon.lavender { background: var(--lavender-soft); color: var(--lavender); }

.metric-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-dim); font-weight: 600; margin-bottom: 4px;
}
.metric-row { display: flex; align-items: baseline; justify-content: space-between; }
.metric-value {
  font-size: 30px; font-weight: 800; color: var(--text-bright);
  letter-spacing: -0.03em; font-variant-numeric: tabular-nums;
}
.metric-trend {
  font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 99px;
}
.metric-trend.up { background: var(--mint-soft); color: var(--mint); }
.metric-trend.neutral { background: var(--gold-soft); color: var(--gold); }
.metric-trend.clear { background: var(--mint-medium); color: var(--mint); }

/* ===== Workspace ===== */
.workspace {
  display: grid; grid-template-columns: 7fr 3fr; gap: 16px;
  padding: 0 28px 28px; height: calc(100vh - 195px); min-height: 400px;
}

/* ===== Log Panel ===== */
.log-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-xl); box-shadow: var(--shadow-card);
  display: flex; flex-direction: column; overflow: hidden;
}

.view-tabs {
  display: flex; gap: 0; padding: 0 24px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-elevated); border-radius: var(--radius-xl) var(--radius-xl) 0 0;
}
.view-tab {
  padding: 14px 18px; font-size: 12px; font-weight: 500;
  color: var(--text-dim); background: none; border: none;
  border-bottom: 2.5px solid transparent; cursor: pointer;
  font-family: var(--font-ui); transition: all var(--transition);
  display: flex; align-items: center; gap: 8px;
}
.view-tab:hover { color: var(--text-muted); }
.view-tab.active { color: var(--gold); border-bottom-color: var(--gold); font-weight: 600; }
.view-tab-icon { width: 15px; height: 15px; opacity: 0.4; }
.view-tab.active .view-tab-icon { opacity: 1; }

.log-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 24px; border-bottom: 1px solid var(--border-subtle); gap: 12px;
}
.log-filters {
  display: flex; gap: 3px; background: var(--bg-input);
  border-radius: var(--radius-xs); padding: 3px;
}
.log-filter {
  padding: 5px 14px; border: none; border-radius: var(--radius-xs);
  background: transparent; color: var(--text-dim);
  font-family: var(--font-ui); font-size: 11px; font-weight: 500;
  cursor: pointer; transition: all var(--transition);
}
.log-filter:hover { color: var(--text-muted); }
.log-filter.active { color: var(--gold); background: var(--gold-soft); font-weight: 600; }

.log-actions { display: flex; align-items: center; gap: 8px; }
.log-search {
  background: var(--bg-input); border: 1.5px solid transparent;
  border-radius: var(--radius-xs); padding: 6px 14px;
  color: var(--text-primary); font-family: var(--font-mono);
  font-size: 11px; width: 200px; outline: none; transition: all var(--transition);
}
.log-search::placeholder { color: var(--text-dim); }
.log-search:focus { border-color: var(--gold); box-shadow: 0 0 0 3px var(--gold-soft); }

.log-btn {
  background: var(--bg-input); border: none; color: var(--text-dim);
  cursor: pointer; padding: 6px 12px; border-radius: var(--radius-xs);
  display: flex; align-items: center; gap: 5px;
  transition: all var(--transition); font-size: 11px;
  font-family: var(--font-ui); font-weight: 500;
}
.log-btn:hover { color: var(--gold); background: var(--gold-soft); }
.log-btn.active { color: var(--gold); background: var(--gold-soft); }

.log-source-tabs {
  display: flex; gap: 0; padding: 0 24px; border-bottom: 1px solid var(--border-subtle);
}
.source-tab {
  padding: 8px 16px; font-size: 11px; font-weight: 500;
  color: var(--text-dim); background: none; border: none;
  border-bottom: 2px solid transparent; cursor: pointer;
  font-family: var(--font-ui); transition: all var(--transition);
}
.source-tab:hover { color: var(--text-muted); }
.source-tab.active { color: var(--text-bright); border-bottom-color: var(--gold); }
.source-tab-count {
  font-size: 9px; background: var(--bg-surface); color: var(--text-dim);
  padding: 1px 7px; border-radius: 99px; margin-left: 5px; font-weight: 600;
}

.log-output {
  flex: 1; overflow-y: auto; padding: 10px 0;
  font-family: var(--font-mono); font-size: 11.5px; line-height: 1.9;
}
.log-line {
  display: flex; gap: 12px; padding: 2px 24px;
  transition: background var(--transition); border-left: 3px solid transparent;
}
.log-line:hover { background: rgba(46,52,71,0.4); }
.log-line.hidden { display: none; }
.log-ts {
  color: var(--text-dim); opacity: 0.6; flex-shrink: 0;
  user-select: none; font-variant-numeric: tabular-nums; min-width: 58px;
}
.log-level {
  flex-shrink: 0; font-weight: 700; font-size: 10px;
  text-transform: uppercase; min-width: 34px; padding-top: 1px;
}
.log-level.info { color: var(--mint); }
.log-level.error { color: var(--coral); }
.log-level.warn { color: var(--amber); }
.log-level.debug { color: var(--text-dim); }
.log-msg { color: var(--text-primary); word-break: break-word; }

.log-line.is-error { border-left-color: var(--coral); background: var(--coral-soft); padding-left: 21px; }
.log-line.is-error:hover { background: var(--coral-medium); }
.log-line.is-warn { border-left-color: var(--amber); background: var(--amber-soft); padding-left: 21px; }
.log-line.is-warn:hover { background: var(--amber-medium); }

.log-empty {
  display: flex; align-items: center; justify-content: center;
  height: 100%; font-size: 12px; color: var(--text-dim);
}

/* ===== Sidebar ===== */
.sidebar { display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
.sidebar-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); box-shadow: var(--shadow-card); overflow: hidden;
}
.sidebar-card-header { padding: 18px 20px 0; }
.sidebar-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--text-dim); font-weight: 700; margin-bottom: 14px;
}
.sidebar-card-body { padding: 0 20px 18px; }

.toggle-group {
  display: flex; background: var(--bg-input); border-radius: var(--radius-xs);
  padding: 3px; gap: 3px;
}
.toggle-opt {
  flex: 1; padding: 8px 16px; font-size: 11px; font-weight: 500;
  color: var(--text-dim); background: transparent; border: none;
  border-radius: var(--radius-xs); cursor: pointer; text-align: center;
  font-family: var(--font-ui); transition: all var(--transition);
}
.toggle-opt:hover { color: var(--text-muted); }
.toggle-opt.active {
  background: var(--gold-medium); color: var(--gold-bright); font-weight: 700;
  box-shadow: 0 2px 8px rgba(240,201,76,0.12);
}

.cfg-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 0; font-size: 12px; border-bottom: 1px solid var(--border-subtle);
}
.cfg-row:last-child { border-bottom: none; }
.cfg-key { color: var(--text-muted); }
.cfg-val {
  color: var(--text-bright); font-weight: 600;
  display: flex; align-items: center; gap: 8px;
}
.cfg-synced {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 10px; color: var(--mint); font-weight: 700;
  background: var(--mint-medium); padding: 2px 10px; border-radius: 99px;
}

.empty-state {
  border: 1.5px dashed var(--border-bright); border-radius: var(--radius-sm);
  padding: 28px 16px; text-align: center;
  transition: all var(--transition); background: rgba(46,52,71,0.15);
}
.empty-state:hover { border-color: var(--gold); background: var(--gold-soft); }
.empty-state p { font-size: 12px; color: var(--text-dim); margin-bottom: 12px; }
.ghost-btn {
  background: none; border: 1px solid var(--border-bright);
  border-radius: var(--radius-xs); color: var(--text-muted);
  padding: 7px 18px; font-size: 11px; font-family: var(--font-ui); font-weight: 600;
  cursor: pointer; transition: all var(--transition);
}
.ghost-btn:hover {
  border-color: var(--gold); color: var(--gold);
  background: var(--gold-soft); box-shadow: var(--shadow-gold);
}

.svc-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 0; border-bottom: 1px solid var(--border-subtle);
}
.svc-item:last-child { border-bottom: none; }
.svc-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.svc-dot.ok { background: var(--mint); box-shadow: 0 0 6px rgba(94,234,212,0.3); }
.svc-dot.down { background: var(--coral); }
.svc-dot.stopped { background: var(--text-dim); }
.svc-name { flex: 1; font-size: 12px; color: var(--text-primary); font-weight: 500; }
.svc-state { font-size: 10px; color: var(--text-dim); font-family: var(--font-mono); }

.switch {
  width: 32px; height: 16px; border-radius: 99px;
  background: var(--bg-input); border: 1px solid var(--border);
  position: relative; cursor: pointer; flex-shrink: 0; transition: all var(--transition);
}
.switch::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--text-dim); transition: all var(--transition);
}
.switch.on { background: var(--mint-soft); border-color: var(--mint); }
.switch.on::after { transform: translateX(16px); background: var(--mint); }

/* ===== View placeholders ===== */
.view-placeholder {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 12px; color: var(--text-dim);
}
.view-placeholder svg { opacity: 0.3; }
.view-placeholder p { font-size: 13px; }

/* ===== Offline ===== */
.offline {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 80px 20px; text-align: center;
}
.offline-icon {
  width: 56px; height: 56px; border-radius: 50%; margin-bottom: 20px;
  background: var(--coral-soft); display: flex; align-items: center;
  justify-content: center; color: var(--coral);
}
.offline h2 { font-size: 18px; font-weight: 700; color: var(--text-bright); margin-bottom: 8px; }
.offline p { font-size: 13px; color: var(--text-muted); margin-bottom: 20px; }
.offline code {
  background: var(--bg-input); padding: 2px 8px;
  border-radius: var(--radius-xs); font-size: 12px;
  font-family: var(--font-mono); color: var(--text-primary);
}
.offline-btn {
  background: var(--gold-soft); border: 1px solid var(--gold);
  border-radius: var(--radius-sm); color: var(--gold);
  padding: 8px 24px; font-size: 12px; font-weight: 600;
  font-family: var(--font-ui); cursor: pointer; transition: all var(--transition);
}
.offline-btn:hover { background: var(--gold-medium); box-shadow: var(--shadow-gold); }

/* ===== Skeleton ===== */
.skeleton {
  background: linear-gradient(90deg, var(--bg-input) 25%, var(--bg-surface) 50%, var(--bg-input) 75%);
  background-size: 200% 100%; animation: shimmer 1.5s infinite;
  border-radius: var(--radius-xs); height: 14px;
}
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-surface); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ===== Responsive ===== */
@media (max-width: 700px) {
  .workspace { grid-template-columns: 1fr; height: auto; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .log-panel { min-height: 50vh; }
}
```

-**Step 2: Verify the CSS compiles**

Check that the template literal has no broken backticks or escape issues. The CSS should be pasted inside the existing `<style>...</style>` tags.

-**Step 3: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): replace dashboard CSS with warm dark Figma-inspired theme"
```

---

## Chunk 3: HTML Skeleton

### Task 4: Replace the HTML body structure

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js` — the `<body>` content inside `getDashboardHtml()`

Replace everything between `<body>` and the `<script>` tag with the new HTML skeleton. This is the static structure — `#content` div gets populated dynamically by `render()`. The log panel structure is persistent (outside `#content`).

-**Step 1: Write the HTML skeleton**

Replace the HTML body (between `</style></head><body>` and `<script>`) with:

```html
<body>
<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="brand">
      <div class="brand-icon">C</div>
      <span class="brand-name">Corvia</span>
    </div>
    <div class="status-pills" id="statusPills">
      <div class="pill"><div class="pill-dot"></div><span class="pill-label">API Server</span></div>
      <div class="pill"><div class="pill-dot"></div><span class="pill-label">Inference</span></div>
    </div>
  </div>
  <div class="header-right">
    <span class="header-time" id="headerTime"></span>
    <span class="scope-badge" id="scopeBadge">loading</span>
  </div>
</div>

<!-- Dynamic content (metrics + workspace) -->
<div id="content">
  <div class="metrics">
    <div class="metric-card gold" style="padding:22px"><div class="skeleton" style="width:60%"></div></div>
    <div class="metric-card peach" style="padding:22px"><div class="skeleton" style="width:60%"></div></div>
    <div class="metric-card mint" style="padding:22px"><div class="skeleton" style="width:60%"></div></div>
    <div class="metric-card lavender" style="padding:22px"><div class="skeleton" style="width:60%"></div></div>
  </div>
</div>
```

Note: the log panel and sidebar are rendered inside `#content` by the `render()` function (Task 5), since they need the status data to populate source tabs and config values.

-**Step 2: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): replace dashboard HTML skeleton with new layout"
```

---

## Chunk 4: JavaScript — Render Logic & Interactivity

### Task 5: Rewrite the JavaScript section

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js` — the `<script>` block inside `getDashboardHtml()`

This is the core logic rewrite. Replace the entire `<script>...</script>` block with new code that:

1. Renders metrics, workspace (log panel + sidebar) from status JSON
2. Implements log filtering (All/Info/Warn/Error)
3. Implements log search
4. Implements view tabs (Logs active, Graph/Traces placeholder)
5. Implements auto-scroll toggle
6. Computes metric trends from previous poll data
7. Updates header pills with correct state dots and hover restart icons
8. Handles offline state

-**Step 1: Write the complete JavaScript**

Replace the `<script>...</script>` block with:

```javascript
const vscode = acquireVsCodeApi();

// --- State ---
let activeView = 'logs';
let activeLogTab = null;
let activeFilter = 'all';
let autoScroll = true;
let searchQuery = '';
let lastLogData = {};

// --- SVG constants ---
const SVG_RESTART = '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M13.5 2.5a.5.5 0 0 0-1 0v2.05A6.48 6.48 0 0 0 8 2.5a6.5 6.5 0 1 0 6.5 6.5.5.5 0 0 0-1 0A5.5 5.5 0 1 1 8 3.5a5.48 5.48 0 0 1 3.94 1.66h-1.94a.5.5 0 0 0 0 1h3a.5.5 0 0 0 .5-.5v-3.16z"/></svg>';
const SVG_CHECK = '<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 1 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0z"/></svg>';
const SVG_LOGS = '<svg class="view-tab-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M2 2a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V2zm1 0v12h10V2H3zm1.5 2h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1zm0 2.5h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1zm0 2.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 1 0-1z"/></svg>';
const SVG_GRAPH = '<svg class="view-tab-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M1 1v14h14V1H1zm13 13H2V2h12v12zM3 13V8h2v5H3zm3 0V5h2v8H6zm3 0V9h2v4H9zm3 0V3h1v10h-1z"/></svg>';
const SVG_TRACES = '<svg class="view-tab-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M0 3.5A.5.5 0 0 1 .5 3h15a.5.5 0 0 1 0 1H.5a.5.5 0 0 1-.5-.5zM3 7.5A.5.5 0 0 1 3.5 7h9a.5.5 0 0 1 0 1h-9A.5.5 0 0 1 3 7.5zM6 11.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5z"/></svg>';
const SVG_OFFLINE = '<svg width="28" height="28" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 13A6 6 0 1 1 8 2a6 6 0 0 1 0 12zm-.5-3h1v1h-1v-1zm0-7h1v5.5h-1V4z"/></svg>';

const METRIC_ICONS = {
  entries: '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M4 0h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm0 1a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H4zm1.5 3a.5.5 0 0 1 0-1h5a.5.5 0 0 1 0 1h-5zm0 2a.5.5 0 0 1 0-1h5a.5.5 0 0 1 0 1h-5zm0 2a.5.5 0 0 1 0-1h3a.5.5 0 0 1 0 1h-3z"/></svg>',
  agents: '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10z"/></svg>',
  queue: '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm15 0a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V2zM4 4h8v1H4V4zm0 2.5h8v1H4v-1zm0 2.5h5v1H4V9z"/></svg>',
  sessions: '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M1.5 1a.5.5 0 0 0-.5.5v3a.5.5 0 0 1-1 0v-3A1.5 1.5 0 0 1 1.5 0h3a.5.5 0 0 1 0 1h-3zM11 .5a.5.5 0 0 1 .5-.5h3A1.5 1.5 0 0 1 16 1.5v3a.5.5 0 0 1-1 0v-3a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 1-.5-.5zM.5 11a.5.5 0 0 1 .5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 1 0 1h-3A1.5 1.5 0 0 1 0 14.5v-3a.5.5 0 0 1 .5-.5zm15 0a.5.5 0 0 1 .5.5v3a1.5 1.5 0 0 1-1.5 1.5h-3a.5.5 0 0 1 0-1h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 1 .5-.5z"/><path d="M3 14s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H3zm8-9a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/></svg>',
};

const SVC_META = {
  'corvia-server':    { desc: 'REST + MCP API server' },
  'corvia-inference': { desc: 'ONNX embeddings + chat (gRPC)' },
  'ollama':           { desc: 'Local LLM runtime' },
  'vllm':             { desc: 'High-perf LLM serving' },
  'surrealdb':        { desc: 'SurrealDB storage backend' },
  'postgres':         { desc: 'PostgreSQL storage backend' },
  'coding-llm':       { desc: 'Code assistant via Continue' },
};

const PROVIDERS = {
  embedding: [
    { value: 'corvia', label: 'Corvia', cmd: 'corvia-dev use corvia-inference' },
    { value: 'ollama', label: 'Ollama', cmd: 'corvia-dev use ollama' },
  ],
};

// --- Helpers ---
function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function dotClass(state) {
  if (state === 'healthy') return 'ok';
  if (state === 'starting') return 'warn';
  return 'down';
}

function formatNum(n) {
  if (n == null) return '-';
  return Number(n).toLocaleString();
}

function trend(current, prev) {
  if (prev == null || current == null) return { label: '-', cls: 'neutral' };
  const delta = current - prev;
  if (delta > 0) return { label: '\\u2191 ' + delta, cls: 'up' };
  if (delta === 0) return { label: 'stable', cls: 'neutral' };
  return { label: '\\u2193 ' + Math.abs(delta), cls: 'neutral' };
}

// --- Message handler ---
window.addEventListener('message', (e) => {
  if (e.data.type === 'status') render(e.data.data);
});

// --- Main render ---
function render(data) {
  const el = document.getElementById('content');

  // Update header time
  document.getElementById('headerTime').textContent = new Date().toLocaleTimeString();

  if (!data) {
    document.getElementById('scopeBadge').textContent = 'offline';
    updatePills(null, null);
    el.innerHTML =
      '<div class="offline">' +
        '<div class="offline-icon">' + SVG_OFFLINE + '</div>' +
        '<h2>corvia-dev not responding</h2>' +
        '<p>Run <code>corvia-dev up</code> to start services</p>' +
        '<button class="offline-btn" data-cmd="corvia-dev up">Start Services</button>' +
      '</div>';
    bindAll();
    return;
  }

  const svcs = data.services || [];
  const cfg = data.config || {};
  const enabled = new Set(data.enabled_services || []);
  const prev = data._prev || {};

  const server = svcs.find(s => s.name === 'corvia-server');
  const inference = svcs.find(s => s.name === 'corvia-inference');
  const optional = svcs.filter(s => !['corvia-server', 'corvia-inference'].includes(s.name));

  document.getElementById('scopeBadge').textContent = cfg.workspace || 'corvia';
  updatePills(server, inference);

  // --- Metrics ---
  const entries = data.entry_count ?? data.entries ?? null;
  const agents = data.agent_count ?? data.agents ?? null;
  const queue = data.merge_queue_depth ?? data.queue_depth ?? null;
  const sessions = data.session_count ?? data.sessions ?? null;
  const pEntries = prev.entry_count ?? prev.entries ?? null;
  const pAgents = prev.agent_count ?? prev.agents ?? null;
  const pQueue = prev.merge_queue_depth ?? prev.queue_depth ?? null;
  const pSessions = prev.session_count ?? prev.sessions ?? null;

  const tEntries = trend(entries, pEntries);
  const tAgents = trend(agents, pAgents);
  const tQueue = queue === 0 ? { label: 'clear', cls: 'clear' } : trend(queue, pQueue);
  const tSessions = sessions != null ? { label: 'active', cls: 'neutral' } : { label: '-', cls: 'neutral' };

  let html = '';

  // Metrics row
  html += '<div class="metrics">';
  html += metricCard('gold', METRIC_ICONS.entries, 'Entries', formatNum(entries), tEntries, '');
  html += metricCard('peach', METRIC_ICONS.agents, 'Active Agents', formatNum(agents), tAgents, '');
  html += metricCard('mint', METRIC_ICONS.queue, 'Merge Queue', formatNum(queue), tQueue,
    queue === 0 ? ' style="color:var(--mint)"' : '');
  html += metricCard('lavender', METRIC_ICONS.sessions, 'Sessions', formatNum(sessions), tSessions, '');
  html += '</div>';

  // Workspace
  html += '<div class="workspace">';

  // --- Log panel ---
  html += '<div class="log-panel">';

  // View tabs
  html += '<div class="view-tabs">';
  html += viewTab('logs', SVG_LOGS, 'Logs');
  html += viewTab('graph', SVG_GRAPH, 'Graph');
  html += viewTab('traces', SVG_TRACES, 'Traces');
  html += '</div>';

  if (activeView === 'logs') {
    // Toolbar
    html += '<div class="log-toolbar">';
    html += '<div class="log-filters">';
    for (const f of ['all', 'info', 'warn', 'error']) {
      html += '<button class="log-filter' + (activeFilter === f ? ' active' : '') + '" data-filter="' + f + '">' +
        f.charAt(0).toUpperCase() + f.slice(1) + '</button>';
    }
    html += '</div>';
    html += '<div class="log-actions">' +
      '<input class="log-search" type="text" placeholder="Search logs..." value="' + esc(searchQuery) + '" id="logSearchInput">' +
      '<button class="log-btn' + (autoScroll ? ' active' : '') + '" id="autoScrollBtn">\\u2193 Auto</button>' +
      '<button class="log-btn" data-action="clear-logs">Clear</button>' +
    '</div></div>';

    // Source tabs
    const tabs = buildLogTabs(data.logs || [], data.service_logs || {});
    const tabNames = Object.keys(tabs);
    if (tabNames.length > 0 && (!activeLogTab || !tabs[activeLogTab])) {
      activeLogTab = tabNames[0];
    }

    html += '<div class="log-source-tabs">';
    for (const name of tabNames) {
      html += '<button class="source-tab' + (name === activeLogTab ? ' active' : '') + '" data-log-tab="' + esc(name) + '">' +
        esc(name) + '<span class="source-tab-count">' + tabs[name].length + '</span></button>';
    }
    html += '</div>';

    // Log output
    html += '<div class="log-output" id="logOutput">';
    const lines = tabs[activeLogTab] || [];
    if (lines.length === 0) {
      html += '<div class="log-empty">No log output yet</div>';
    } else {
      for (const line of lines) {
        const parsed = parseLine(line);
        const levelMatch = activeFilter === 'all' || parsed.level === activeFilter;
        const searchMatch = !searchQuery || line.toLowerCase().includes(searchQuery.toLowerCase());
        const hidden = (!levelMatch || !searchMatch) ? ' hidden' : '';
        const errClass = parsed.level === 'error' ? ' is-error' : parsed.level === 'warn' ? ' is-warn' : '';
        html += '<div class="log-line' + errClass + hidden + '">' +
          '<span class="log-ts">' + esc(parsed.ts) + '</span>' +
          '<span class="log-level ' + parsed.level + '">' + esc(parsed.levelLabel) + '</span>' +
          '<span class="log-msg">' + esc(parsed.msg) + '</span>' +
        '</div>';
      }
    }
    html += '</div>';
  } else {
    // Placeholder for Graph / Traces
    html += '<div class="view-placeholder">' +
      '<svg width="48" height="48" viewBox="0 0 16 16" fill="currentColor"><path d="M1 1v14h14V1H1zm13 13H2V2h12v12zM3 13V8h2v5H3zm3 0V5h2v8H6zm3 0V9h2v4H9zm3 0V3h1v10h-1z"/></svg>' +
      '<p>' + esc(activeView.charAt(0).toUpperCase() + activeView.slice(1)) + ' view coming soon</p>' +
    '</div>';
  }

  html += '</div>'; // close log-panel

  // --- Sidebar ---
  html += '<div class="sidebar">';

  // Embedding provider
  html += '<div class="sidebar-card"><div class="sidebar-card-header"><div class="sidebar-label">Embedding Provider</div></div>' +
    '<div class="sidebar-card-body"><div class="toggle-group">';
  for (const p of PROVIDERS.embedding) {
    const isActive = cfg.embedding_provider === p.value;
    html += '<button class="toggle-opt' + (isActive ? ' active' : '') + '"' +
      (isActive ? '' : ' data-cmd="' + esc(p.cmd) + '"') + '>' + esc(p.label) + '</button>';
  }
  html += '</div></div></div>';

  // Configuration
  html += '<div class="sidebar-card"><div class="sidebar-card-header"><div class="sidebar-label">Configuration</div></div>' +
    '<div class="sidebar-card-body">' +
    cfgRow('Storage', esc(cfg.storage || '-')) +
    cfgRow('Merge LLM', cfg.merge_provider ? esc(cfg.merge_provider) : '<span style="color:var(--text-dim)">&mdash;</span>') +
    cfgRow('Workspace', esc(cfg.workspace || '-') +
      ' <span class="cfg-synced">' + SVG_CHECK + ' Synced</span>') +
    cfgRow('Telemetry', esc(cfg.telemetry_exporter || cfg.telemetry || 'stdout')) +
  '</div></div>';

  // Optional services
  html += '<div class="sidebar-card"><div class="sidebar-card-header"><div class="sidebar-label">Optional Services</div></div>' +
    '<div class="sidebar-card-body">';
  if (optional.length === 0) {
    html += '<div class="empty-state"><p>No optional services configured</p>' +
      '<button class="ghost-btn" data-cmd="corvia-dev enable ollama">Configure Services</button></div>';
  } else {
    for (const svc of optional) {
      const meta = SVC_META[svc.name] || { desc: '' };
      const isEnabled = enabled.has(svc.name);
      const dc = svc.state === 'healthy' ? 'ok' : svc.state === 'stopped' ? 'stopped' : 'down';
      html += '<div class="svc-item">' +
        '<div class="svc-dot ' + dc + '"></div>' +
        '<span class="svc-name">' + esc(svc.name) + '</span>' +
        '<span class="svc-state">' + esc(svc.state) + '</span>' +
        '<div class="switch' + (isEnabled ? ' on' : '') + '" data-toggle="' + esc(svc.name) + '" data-enabled="' + (isEnabled ? '1' : '0') + '"></div>' +
      '</div>';
    }
  }
  html += '</div></div>';

  html += '</div>'; // close sidebar
  html += '</div>'; // close workspace

  el.innerHTML = html;

  // Post-render: bind events, auto-scroll
  bindAll();
  if (autoScroll) {
    const output = document.getElementById('logOutput');
    if (output) output.scrollTop = output.scrollHeight;
  }
}

// --- Component builders ---
function metricCard(color, icon, label, value, trendObj, valueAttr) {
  return '<div class="metric-card ' + color + '">' +
    '<div class="metric-icon ' + color + '">' + icon + '</div>' +
    '<div class="metric-label">' + esc(label) + '</div>' +
    '<div class="metric-row">' +
      '<span class="metric-value"' + valueAttr + '>' + value + '</span>' +
      '<span class="metric-trend ' + trendObj.cls + '">' + esc(trendObj.label) + '</span>' +
    '</div></div>';
}

function viewTab(id, icon, label) {
  return '<button class="view-tab' + (activeView === id ? ' active' : '') + '" data-view="' + id + '">' +
    icon + ' ' + esc(label) + '</button>';
}

function cfgRow(key, valueHtml) {
  return '<div class="cfg-row"><span class="cfg-key">' + esc(key) + '</span>' +
    '<span class="cfg-val">' + valueHtml + '</span></div>';
}

// --- Update header pills ---
function updatePills(server, inference) {
  const pills = document.getElementById('statusPills');
  if (!pills) return;
  const sState = server ? server.state : 'stopped';
  const iState = inference ? inference.state : 'stopped';

  pills.innerHTML =
    pill('API Server', sState, 'corvia-server') +
    pill('Inference', iState, 'corvia-inference');

  // Bind restart clicks on pills
  pills.querySelectorAll('.pill-restart').forEach(btn => {
    btn.onclick = (e) => {
      e.stopPropagation();
      vscode.postMessage({ type: 'command', command: btn.dataset.cmd });
    };
  });
}

function pill(label, state, svcName) {
  const dc = dotClass(state);
  const action = (state === 'healthy' || state === 'starting' || state === 'unhealthy')
    ? 'corvia-dev restart ' + svcName
    : 'corvia-dev up';
  return '<div class="pill">' +
    '<div class="pill-dot ' + dc + '"></div>' +
    '<span class="pill-label">' + esc(label) + '</span>' +
    '<span class="pill-restart" data-cmd="' + esc(action) + '">' + SVG_RESTART + '</span>' +
  '</div>';
}

// --- Log helpers ---
function buildLogTabs(managerLogs, serviceLogs) {
  const tabs = {};
  if (managerLogs && managerLogs.length > 0) tabs['manager'] = managerLogs;
  if (serviceLogs) {
    for (const [name, lines] of Object.entries(serviceLogs)) {
      if (lines && lines.length > 0) tabs[name] = lines;
    }
  }
  lastLogData = tabs;
  return tabs;
}

function parseLine(line) {
  // Try to extract timestamp and level from common log formats
  // Format: "2026-03-10T12:34:05 INFO message" or "12:34:05 INFO message" or just "message"
  let ts = '', level = 'info', levelLabel = 'INFO', msg = line;

  const m = line.match(/^(\\d{2}:\\d{2}:\\d{2}|\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2})\\s+(\\w+)\\s+(.*)/);
  if (m) {
    ts = m[1].includes('T') ? m[1].split('T')[1] : m[1];
    const lvl = m[2].toUpperCase();
    if (lvl === 'ERROR' || lvl === 'ERR') { level = 'error'; levelLabel = 'ERR'; }
    else if (lvl === 'WARN' || lvl === 'WARNING') { level = 'warn'; levelLabel = 'WARN'; }
    else if (lvl === 'DEBUG' || lvl === 'DBUG' || lvl === 'TRACE') { level = 'debug'; levelLabel = 'DBUG'; }
    else { level = 'info'; levelLabel = 'INFO'; }
    msg = m[3];
  } else {
    // No timestamp — try level only
    const m2 = line.match(/^(\\w+):\\s+(.*)/);
    if (m2) {
      const lvl = m2[1].toUpperCase();
      if (['ERROR', 'ERR', 'WARN', 'WARNING', 'DEBUG', 'INFO'].includes(lvl)) {
        if (lvl === 'ERROR' || lvl === 'ERR') { level = 'error'; levelLabel = 'ERR'; }
        else if (lvl === 'WARN' || lvl === 'WARNING') { level = 'warn'; levelLabel = 'WARN'; }
        else if (lvl === 'DEBUG') { level = 'debug'; levelLabel = 'DBUG'; }
        msg = m2[2];
      }
    }
  }

  return { ts, level, levelLabel, msg };
}

// --- Bind all interactive elements ---
function bindAll() {
  // Command buttons
  document.querySelectorAll('[data-cmd]').forEach(b => {
    b.onclick = () => vscode.postMessage({ type: 'command', command: b.dataset.cmd });
  });

  // Refresh action
  document.querySelectorAll('[data-action="refresh"]').forEach(b => {
    b.onclick = () => vscode.postMessage({ type: 'refresh' });
  });

  // View tabs
  document.querySelectorAll('[data-view]').forEach(b => {
    b.onclick = () => {
      activeView = b.dataset.view;
      // Re-render with last data by requesting refresh
      vscode.postMessage({ type: 'refresh' });
    };
  });

  // Log filter buttons
  document.querySelectorAll('[data-filter]').forEach(b => {
    b.onclick = () => {
      activeFilter = b.dataset.filter;
      applyLogFilters();
      document.querySelectorAll('.log-filter').forEach(f => f.classList.toggle('active', f.dataset.filter === activeFilter));
    };
  });

  // Log source tabs
  document.querySelectorAll('[data-log-tab]').forEach(b => {
    b.onclick = () => {
      activeLogTab = b.dataset.logTab;
      vscode.postMessage({ type: 'refresh' });
    };
  });

  // Search input
  const searchInput = document.getElementById('logSearchInput');
  if (searchInput) {
    searchInput.oninput = () => {
      searchQuery = searchInput.value;
      applyLogFilters();
    };
  }

  // Auto-scroll toggle
  const autoBtn = document.getElementById('autoScrollBtn');
  if (autoBtn) {
    autoBtn.onclick = () => {
      autoScroll = !autoScroll;
      autoBtn.classList.toggle('active', autoScroll);
      if (autoScroll) {
        const output = document.getElementById('logOutput');
        if (output) output.scrollTop = output.scrollHeight;
      }
    };
  }

  // Clear logs
  document.querySelectorAll('[data-action="clear-logs"]').forEach(b => {
    b.onclick = () => {
      const output = document.getElementById('logOutput');
      if (output) output.innerHTML = '<div class="log-empty">Logs cleared</div>';
    };
  });

  // Service toggles
  document.querySelectorAll('[data-toggle]').forEach(t => {
    t.onclick = () => {
      const svc = t.dataset.toggle;
      const isEnabled = t.dataset.enabled === '1';
      const cmd = isEnabled ? 'corvia-dev disable ' + svc : 'corvia-dev enable ' + svc;
      vscode.postMessage({ type: 'command', command: cmd });
    };
  });
}

// --- Filter log lines in-place (no re-render) ---
function applyLogFilters() {
  const lines = document.querySelectorAll('.log-line');
  const q = searchQuery.toLowerCase();
  lines.forEach(line => {
    const level = line.querySelector('.log-level');
    const msg = line.querySelector('.log-msg');
    if (!level || !msg) return;
    const lvl = level.textContent.trim().toLowerCase();
    const levelMap = { info: 'info', err: 'error', warn: 'warn', dbug: 'debug' };
    const normalizedLevel = levelMap[lvl] || 'info';
    const levelMatch = activeFilter === 'all' || normalizedLevel === activeFilter;
    const searchMatch = !q || line.textContent.toLowerCase().includes(q);
    line.classList.toggle('hidden', !levelMatch || !searchMatch);
  });
}

// Initial binding for header
bindAll();
```

-**Step 2: Test the full render cycle**

1. Reload the VS Code window (`Developer: Reload Window`)
2. Run `Corvia: Open Dashboard` from the command palette
3. Verify: header pills show service status with pulsing dots
4. Verify: metric cards show values (or `-` if fields not yet in status JSON)
5. Verify: log panel renders with source tabs and log lines
6. Verify: clicking filters hides/shows log lines without full re-render
7. Verify: search input filters logs in real-time
8. Verify: auto-scroll button toggles
9. Verify: view tabs switch to placeholder for Graph/Traces
10. Verify: sidebar shows config and embedding provider toggle
11. Verify: offline state shows when services are down
12. Verify: responsive layout at narrow widths

-**Step 3: Commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "feat(extension): rewrite dashboard JS with metrics, log filtering, view tabs"
```

---

## Chunk 5: Final Verification

### Task 6: End-to-end verification

-**Step 1: Full verification checklist**

Run through every item from the spec's verification criteria:

| # | Criteria | How to test |
|---|----------|-------------|
| 1 | Dashboard opens | Command palette → "Corvia: Open Dashboard" |
| 2 | Status pills | Check dots pulse green when healthy, red when down |
| 3 | Metric cards | Verify numbers update on each poll cycle |
| 4 | Log filters | Click Info/Warn/Error — lines filter without re-render |
| 5 | Source tabs | Click corvia-server tab — shows different logs |
| 6 | Search | Type "merge" — only matching lines visible |
| 7 | Embedding toggle | Click "Ollama" — terminal opens with `corvia-dev use ollama` |
| 8 | Service toggles | If optional services exist, toggle sends enable/disable |
| 9 | Offline state | Stop services → dashboard shows offline banner |
| 10 | Responsive | Drag panel narrow → single column layout |
| 11 | View tabs | Click "Graph" → shows placeholder |
| 12 | Hover effects | Gold glows, pill restart icon appears, card lifts |

-**Step 2: Fix any issues found**

Address any visual or functional issues. Common things to check:
- Template literal backtick escaping (any `\`` in the embedded HTML/JS)
- SVG paths not broken by string concatenation
- CSS `@import` loading correctly in webview (requires network access)

-**Step 3: Final commit**

```bash
git add .devcontainer/extensions/corvia-services/extension.js
git commit -m "fix(extension): address dashboard verification issues"
```

Only create this commit if there were fixes needed. Skip if everything passed.

### Task 7: Rebuild VSIX (optional)

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/corvia-services-0.2.0.vsix` → rename to `corvia-services-0.3.0.vsix`

-**Step 1: Package the extension**

```bash
cd .devcontainer/extensions/corvia-services
npx @vscode/vsce package --no-dependencies
```

This creates `corvia-services-0.3.0.vsix`. Remove the old 0.2.0 vsix.

-**Step 2: Commit**

```bash
git add .devcontainer/extensions/corvia-services/
git commit -m "chore(extension): package v0.3.0 VSIX"
```
