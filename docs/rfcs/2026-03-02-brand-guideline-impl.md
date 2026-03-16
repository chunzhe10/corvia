# corvia Brand Guideline — Implementation Plan

> **Status:** Shipped (v0.2.0)

**Goal:** Build a polished brand guideline page (HTML + PDF) and update all repos with gold dark-mode logo and refined README copy.

**Architecture:** Single-page HTML brand guide with CSS `@page` rules for 5-page PDF export via Playwright. Gold logo generated via Playwright screenshot. README copy updated across 3 repos.

**Tech Stack:** HTML/CSS, Playwright (HTML→PDF, HTML→PNG), Google Fonts (Poppins, Inter)

---

### Task 1: Generate gold dark-mode logo

**Files:**
- Create: `/root/corvia/docs/assets/corvia-logo-light.png` (overwrite existing white version)

**Step 1: Generate gold wordmark PNG**

Run Playwright to render "corvia" in Poppins 700 with color `#E8C44E` on transparent background at 4x resolution:

```python
# Inline Playwright script
from playwright.sync_api import sync_playwright

HTML = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:transparent; font-family:'Poppins',sans-serif;
         display:flex; align-items:center; justify-content:center; padding:20px 40px; }
  .wordmark { font-weight:700; letter-spacing:-2px; line-height:1;
              white-space:nowrap; color:#E8C44E; font-size:120px; }
</style></head>
<body><div class="wordmark">corvia</div></body></html>'''

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width':800,'height':300}, device_scale_factor=4)
    page.set_content(HTML)
    page.wait_for_timeout(800)
    el = page.query_selector('.wordmark')
    el.screenshot(path='/root/corvia/docs/assets/corvia-logo-light.png', omit_background=True)
    browser.close()
```

**Step 2: Copy to all repos**

```bash
cp /root/corvia/docs/assets/corvia-logo-light.png /root/corvia-adapter-git/docs/assets/corvia-logo-light.png
cp /root/corvia/docs/assets/corvia-logo-light.png /root/corvia-workspace/docs/assets/corvia-logo-light.png
```

**Step 3: Verify visually**

Read the generated PNG to confirm gold text on transparent background.

---

### Task 2: Build brand guideline HTML

**Files:**
- Create: `/root/corvia/docs/brand/corvia-brand-guide.html`

Build a single-page HTML document with 5 sections. Each section is a full `@page` for PDF. Design: clean white background, navy text, Inter body font, Poppins for brand specimens.

**Sections:**

**Page 1 — Cover:**
- Large corvia wordmark (navy, Poppins 700, 96px)
- "Brand Guidelines" subtitle (Inter 500, 24px, mid-gray)
- "v1.0 — March 2026" footer

**Page 2 — Logo:**
- 4 logo variants shown on appropriate backgrounds:
  - Navy on white (light mode primary)
  - Gold #E8C44E on navy (dark mode primary)
  - White on navy pill
  - Gold on navy pill
- Clear space rule: padding = "i" dot height
- Minimum size: 80px width
- Don'ts grid (4 items): no stretching, no rotating, no drop shadows, no recoloring

**Page 3 — Color Palette:**
- 7 color swatches, each showing: swatch rectangle + hex + RGB + role description
  - Navy #0f1b33 (primary)
  - Gold #D4A832 (accent, light)
  - Bright Gold #E8C44E (accent, dark)
  - White #ffffff
  - Light Gray #f5f5f7
  - Mid Gray #86868b
  - Dark Surface #0d1117 (GitHub dark reference)

**Page 4 — Typography:**
- Poppins 700 specimen: "corvia" at 48/32/24px with letter-spacing notes
- Inter specimen: 700/500/400 weights at 32/24/16/14px
- Monospace specimen: `corvia search "query"` at 14px
- Scale table: Hero 48 / H1 32 / H2 24 / Body 16 / Caption 14

**Page 5 — Applications:**
- GitHub README mockup: two side-by-side cards simulating light and dark mode with the logo
- LinkedIn card: navy bg, gold accent bar, "corvia" in Poppins, tagline in Inter
- CLI output: simulated terminal with corvia commands and colored output

**CSS approach:**
- `@page { size: 1080px 1350px; margin: 0; }` for LinkedIn-sized PDF pages
- `page-break-after: always` between sections
- Embedded Google Fonts via `@import`
- All colors as CSS custom properties for easy reference

---

### Task 3: Export to PDF

**Files:**
- Create: `/root/corvia/docs/brand/corvia-brand-guide.pdf`

```python
from playwright.sync_api import sync_playwright
import pathlib

HTML = pathlib.Path('/root/corvia/docs/brand/corvia-brand-guide.html')
PDF = pathlib.Path('/root/corvia/docs/brand/corvia-brand-guide.pdf')

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1080, 'height': 1350})
    page.goto(f'file://{HTML.resolve()}')
    page.wait_for_timeout(1000)
    page.pdf(path=str(PDF), width='1080px', height='1350px',
        print_background=True, margin={'top':'0','right':'0','bottom':'0','left':'0'})
    browser.close()
```

Verify: PDF should be 5 pages, < 500 KB.

---

### Task 4: Update README copy

**Files:**
- Modify: `/root/corvia/README.md` (lines 27-29)

Replace:
```
AI agents are getting memory — Claude has CLAUDE.md, Cursor has rules files, Copilot has
memory. But these are *personal notes*: flat files, one repo, no relationships, no history.
They help one developer in one session. They don't help your *organization* learn.
```

With:
```
AI agents are starting to develop memory — CLAUDE.md, Cursor rules, Copilot memory are all
early steps in this direction. These work well for individual developers in individual
sessions. corvia explores what happens when that memory becomes *organizational* — shared
across agents, tracked over time, connected as a graph.
```

---

### Task 5: Commit + push all repos

**corvia repo:**
```bash
cd /root/corvia
git add docs/assets/corvia-logo-light.png docs/brand/ README.md
git commit -m "brand: add brand guideline, gold dark-mode logo, refine README copy"
git push
```

**adapter-git repo:**
```bash
cd /root/corvia-adapter-git
git add docs/assets/corvia-logo-light.png
git commit -m "brand: gold dark-mode logo"
git push
```

**workspace repo:**
```bash
cd /root/corvia-workspace
git add docs/assets/corvia-logo-light.png
git commit -m "brand: gold dark-mode logo"
git push
```
