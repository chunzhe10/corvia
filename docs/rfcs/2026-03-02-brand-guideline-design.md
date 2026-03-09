# corvia Brand Guideline — Design

**Date:** 2026-03-02
**Status:** Design
**Format:** Single-page HTML + PDF export
**Location:** `docs/brand/corvia-brand-guide.html` + `docs/brand/corvia-brand-guide.pdf`

---

## Purpose

A public-facing, professional brand guideline for corvia. Serves as reference for contributors, partners, and press. Establishes visual identity with enough detail to ensure consistency across GitHub, social media, documentation, and future website.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Brand casing | All lowercase `corvia` | Approachable, modern (stripe, airbnb pattern) |
| Primary font | Poppins 700, letter-spacing -2px | Geometric rounded terminals, distinct wordmark |
| Body font | Inter 400/500/700 | Clean readability, wide Unicode support |
| Primary color | Navy `#0f1b33` | Deep, professional, high contrast |
| Accent color (light) | Gold `#D4A832` | Warm, distinctive, not overused in dev tools |
| Accent color (dark) | Bright Gold `#E8C44E` | Lightened for legibility on dark backgrounds |
| Dark mode logo | Gold `#E8C44E` on transparent | Replaces current white variant |
| Logo format | Poppins wordmark (no icon) | Clean, reproducible, no raster dependency |

## Page Structure (single continuous page, multi-page PDF)

### Page 1 — Cover
- corvia logo centered (large)
- "Brand Guidelines" subtitle
- Version + date

### Page 2 — Logo
- Primary logo: navy on light
- Dark mode variant: gold `#E8C44E` on dark
- Pill variant: white on navy, gold on navy
- Clear space: minimum padding = height of the "i" dot
- Minimum size: 80px width
- Don'ts: no stretching, no rotating, no drop shadows, no recoloring outside palette

### Page 3 — Color Palette
- Navy `#0f1b33` — swatch + hex + RGB + usage
- Gold `#D4A832` — swatch + hex + RGB + usage (light mode accent)
- Bright Gold `#E8C44E` — swatch + hex + RGB + usage (dark mode accent)
- White `#ffffff` — text on dark backgrounds
- Light Gray `#f5f5f7` — page/card backgrounds
- Mid Gray `#86868b` — secondary text, captions
- Dark Surface `#0d1117` — GitHub dark mode reference

### Page 4 — Typography
- Poppins 700 — brand name, headlines (Google Fonts)
- Inter 400/500/700 — body text, UI, documentation (Google Fonts)
- System monospace — code blocks
- Sizing scale: 48px (hero) / 32px (h1) / 24px (h2) / 16px (body) / 14px (caption)
- Letter spacing: -2px for brand, -0.5px for headings, normal for body

### Page 5 — Applications
- GitHub README: light + dark mode side-by-side mockup
- LinkedIn social card: navy background, gold accent, Poppins brand name
- Terminal/CLI output style: how `corvia` commands look
- Slide deck: reference palette for presentations

## Logo Asset Generation

Regenerate the dark-mode logo PNG (`corvia-logo-light.png`) using gold `#E8C44E` instead of white. Update all three repos:
- `/root/corvia/docs/assets/corvia-logo-light.png`
- `/root/corvia-adapter-git/docs/assets/corvia-logo-light.png`
- `/root/corvia-workspace/docs/assets/corvia-logo-light.png`

## README Copy Fix

Replace the current "What is corvia?" opener with exploratory framing:

**Before (dismissive):**
> AI agents are getting memory — Claude has CLAUDE.md, Cursor has rules files, Copilot has memory. But these are *personal notes*: flat files, one repo, no relationships, no history.

**After (exploratory):**
> AI agents are starting to develop memory — CLAUDE.md, Cursor rules, Copilot memory are all early steps in this direction. These work well for individual developers in individual sessions. corvia explores what happens when that memory becomes *organizational* — shared across agents, tracked over time, connected as a graph.

## Implementation Tasks

1. Generate gold dark-mode logo PNG (`#E8C44E` on transparent)
2. Build `docs/brand/corvia-brand-guide.html` — 5-section single-page layout
3. Export to PDF via Playwright
4. Update dark-mode logo across all 3 repos
5. Update README "What is corvia?" copy
6. Commit + push all repos
