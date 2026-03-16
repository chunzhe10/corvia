# Corvia Dashboard Redesign — Design Spec

**Date:** 2026-03-10
**Status:** Shipped (v0.3.7)
**Depends on:** M4 observability + control plane (for metrics data)
**Affects:** `.devcontainer/extensions/corvia-services/extension.js`

---

## Overview

Redesign the Corvia VS Code extension dashboard from the current card-based layout to a high-density, Figma-inspired warm dark theme with Navy & Gold aesthetic. Surfaces M4 observability data (entry counts, agents, merge queue, sessions) and introduces a view-tab system for future Graph and Traces pages.

## Design Direction

Warm charcoal dark mode (not navy) with gold as the dominant accent. Figma-inspired: large border radius, elevated cards with soft shadows, generous spacing, bold metric numbers, vivid multi-color accents. The mockup is at `.superpowers/brainstorm/layout-v5.html`.

---

## Architecture & Layout

### Structure
- **Header ribbon** (sticky) — brand, status pills, scope badge, timestamp
- **Metrics row** — 4 elevated cards from `corvia_system_status` data
- **70/30 workspace** — main content (left) + config sidebar (right)
- **Main content uses view tabs** — Logs now; Graph + Traces added later
- **Sidebar** — 3 stacked cards: embedding toggle, config, optional services

### Data Flow
- Extension polls `corvia-dev status --json` every 3 seconds (unchanged)
- Metrics data comes from the same status JSON
- View tabs switch content in the left panel; sidebar stays fixed
- All commands execute via VS Code terminal (unchanged)

### File Structure
- Single `extension.js` with embedded HTML/CSS/JS (same pattern as current)
- Version bump to 0.3.0

---

## Design Tokens

### Colors (warm charcoal base)
| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#12141a` | Page background |
| `--bg-elevated` | `#1a1d26` | Elevated surfaces (view tabs bg) |
| `--bg-card` | `#1e2230` | Card backgrounds |
| `--bg-card-hover` | `#252a3a` | Card hover state |
| `--bg-input` | `#282d3e` | Input fields, recessed controls |
| `--bg-surface` | `#2e3447` | Scrollbar thumb, badges |
| `--gold` | `#f0c94c` | Primary accent |
| `--gold-bright` | `#ffe066` | Active toggle text |
| `--mint` | `#5eead4` | Success/healthy state |
| `--coral` | `#ff8a80` | Error state |
| `--peach` | `#ffb07c` | Agents accent |
| `--lavender` | `#c4b5fd` | Sessions accent |
| `--amber` | `#fcd34d` | Warning state |
| `--text-bright` | `#f2f0ed` | Headings, metric values |
| `--text-primary` | `#c5c0b8` | Body text, log messages |
| `--text-muted` | `#918b82` | Labels, secondary text |
| `--text-dim` | `#615c55` | Timestamps, disabled text |

Each accent color has `-soft` (10% opacity) and `-medium` (16-18% opacity) variants for backgrounds.

### Typography
- **UI font:** Inter (Google Fonts in webview), with system fallbacks
- **Mono font:** Cascadia Code → JetBrains Mono → Fira Code → monospace
- **Metric values:** 30px / weight 800 / -0.03em tracking
- **Section labels:** 10-11px / uppercase / 0.06em tracking / weight 600-700
- **Body:** 13px / weight 400-500

### Border Radius Scale
| Token | Value |
|-------|-------|
| `--radius-xs` | 6px |
| `--radius-sm` | 8px |
| `--radius-md` | 12px |
| `--radius-lg` | 16px |
| `--radius-xl` | 20px |

### Shadows
- **Card:** `0 4px 20px rgba(0,0,0,0.2)` + `0 0 1px rgba(255,255,255,0.03) inset`
- **Hover:** `0 8px 32px rgba(0,0,0,0.3)`
- **Gold glow:** `0 4px 20px rgba(240,201,76,0.10)`

### Transitions
- All hover: `0.25s cubic-bezier(0.4, 0, 0.2, 1)`

---

## Components

### Header Ribbon
- Frosted glass: `background: rgba(18,20,26,0.8)` + `backdrop-filter: blur(16px)`
- Sticky top, z-index 10
- **Brand:** 30px gold gradient icon + "Corvia" (17px/700)
- **Status pills:** card bg + border, containing:
  - 8px dot with glow shadow + pulse animation (2.5s) when healthy
  - Uppercase label (10px/600)
  - Restart SVG icon: opacity 0, fades to 1 on pill hover, turns gold on icon hover
- **Right side:** mono timestamp + scope badge (pill with border)

### Metric Cards (4-up grid)
Each card contains:
- **Colored icon badge** (38px, radius-sm) with accent-soft background
- **Label** (uppercase, dim)
- **Value** (30px/800, text-bright)
- **Trend pill** (rounded, accent-soft bg): e.g., "↑ 12", "stable", "clear", "active"
- **Top accent bar** (3px gradient, positioned absolute): gold, peach, mint, lavender
- **Hover:** translateY(-2px) + shadow-hover + border-bright

Card color assignments:
| Metric | Color | Data Source |
|--------|-------|-------------|
| Entries | gold | Store entry count |
| Active Agents | peach | Active agent count |
| Merge Queue | mint | Queue depth (value colored mint when 0) |
| Sessions | lavender | Open session count |

Trend values computed by diffing current vs previous poll.

### Log Terminal (left 70%)

**View Tabs** (top of panel):
- Horizontal tabs: Logs (active), Graph, Traces
- Each tab has an SVG icon + label
- Active: gold text + 2.5px gold bottom border
- Background: bg-elevated, rounded top corners match panel radius

**Filter Bar:**
- Segmented control group (recessed bg-input container with 3px padding)
- Buttons: All / Info / Warn / Error
- Active filter: gold text + gold-soft background

**Toolbar:**
- Search input: mono font, gold focus ring (1.5px border + 3px gold-soft shadow)
- Auto-scroll button + Clear button: bg-input, gold on hover

**Source Tabs:**
- Per-service tabs: manager, corvia-server, corvia-inference
- Count badges (9px, rounded pill, bg-surface)
- Active: text-bright + gold bottom border

**Log Output:**
- Monospace, 11.5px, line-height 1.9
- Each line: timestamp (dim, 60% opacity) + level badge + message
- Level colors: INFO=mint, ERROR=coral, WARN=amber, DEBUG=dim
- Error lines: 3px coral left-border + coral-soft bg
- Warn lines: 3px amber left-border + amber-soft bg
- Hover: subtle bg highlight

### Sidebar (right 30%)

Three stacked cards with 16px gap:

**1. Embedding Provider:**
- Segmented toggle: Corvia / Ollama
- Active option: gold-medium bg + gold-bright text + weight 700
- Recessed container: bg-input with 3px padding

**2. Configuration:**
- Key/value rows with subtle bottom borders
- Keys: text-muted, Values: text-bright/600
- Workspace row includes "Synced" badge: mint text, mint-medium bg, checkmark SVG
- Telemetry row shows current exporter (new M4 config)

**3. Optional Services:**
- **Empty state:** 1.5px dashed border, centered text + ghost "Configure Services" button
- Ghost button: border-bright, turns gold on hover with gold-soft bg + glow
- Empty state container: hover turns border gold + gold-soft bg
- **With services:** list items with 7px status dot (mint=healthy, dim=stopped) + name + state text + toggle switch
- Toggle switch: 32x16px, mint-soft bg + mint dot when on

### Offline State
- Replaces `#content` div entirely
- Centered: icon + "corvia-dev not responding" title + hint with `<code>` + "Start Services" primary button

---

## Responsive Behavior

Below 700px width:
- Workspace collapses to single column (log panel above sidebar)
- Metrics grid becomes 2x2
- Log panel gets min-height: 50vh

---

## What Does NOT Change
- Extension activation logic, polling interval (3s), status bar item
- `corvia-dev` CLI JSON contract
- Command execution via VS Code terminal
- `package.json` manifest structure (just version bump to 0.3.0)
- `onDidReceiveMessage` handler (command, refresh message types)

---

## Verification Criteria
1. Dashboard opens via `Corvia: Open Dashboard` command
2. Status pills reflect real service health with animated dots
3. Metric cards show live data from status JSON
4. Log filters (All/Info/Warn/Error) filter displayed lines
5. Log source tabs switch between manager and per-service logs
6. Search input filters log lines by text match
7. Embedding provider toggle sends correct `corvia-dev use` command
8. Optional service toggles send enable/disable commands
9. Offline state shows when `corvia-dev status --json` fails
10. Layout collapses to single column below 700px
11. View tabs render (Graph/Traces show placeholder, only Logs is functional)
12. All hover/focus interactions work (gold accents, glows, transitions)

---

*Mockup reference: `.superpowers/brainstorm/layout-v5.html`*
*Created: 2026-03-10*
