# Corvia — Design Document

**Status:** Shipped (v0.1.0)
**Date:** 2026-02-25
**Author:** Lim Chun Zhe + Claude (brainstorming session)
**Brainstorm log:** `docs/rfcs/2026-02-25-corvia-brainstorm.md`
**License:** MIT

---

## 1. Project Identity

- **Name:** Corvia
- **Tagline:** Knowledge-gathering agents for living documentation
- **Origin:** Derived from Corvus (the raven constellation). Ravens in mythology are knowledge-gatherers — Odin's ravens Huginn and Muninn flew the world daily, gathering knowledge and reporting back. "Corvia" softens the sound while preserving the lineage.
- **What it is:** A VS Code extension + Rust backend that turns any codebase into a living, self-documenting knowledge system. It indexes code via tree-sitter AST, generates documentation through AI agents, captures development conversations, and maintains a temporal knowledge graph — all while documenting itself.
- **Target audience:** Portfolio showcase (primary), open source community (secondary)
- **Built from scratch** — original design, not a fork or refactor of prior work.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     VS Code Extension (TypeScript)              │
│  Activity Bar │ Sidebar │ Code Lens │ Dashboard │ Status Bar    │
└────────┬──────────────────────────┬─────────────────────────────┘
         │ NAPI-RS (in-process)     │ MCP (sidecar process)
         │                          │
┌────────▼────────────┐  ┌─────────▼──────────────────────┐
│   corvia-core       │  │   corvia-mcp                   │
│   (Rust)            │  │   (Rust MCP Server)             │
│                     │  │                                 │
│ • tree-sitter AST   │  │ • Agent orchestration           │
│ • Chunking engine   │  │ • LLM API calls (provider-agn.) │
│ • Vector search     │  │ • Doc generation agent           │
│ • File watcher      │  │ • Distillation agent             │
│ • Event logger      │  │ • Review agent                   │
│ • Embedding runner   │  │ • Eval agent                     │
└─────────────────────┘  └─────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        .corvia/ (in target repo)                │
│  corvia.toml │ events.jsonl │ docs/ │ conversations/ │ index/   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Git (text) + DVC    │
│  (binary artifacts)  │
│  + Git tags          │
│  (snapshots)         │
└──────────────────────┘
```

### Three Data Flows

1. **Code → Docs:** File watch → AST parse → chunk → embed → index → agent generates doc
2. **Conversations → Decisions:** Chat captured → raw log → LLM distills → decisions extracted → living doc updated
3. **Everything → History:** Every action appends to `events.jsonl` → git tags mark snapshots → temporal queries over history

### Three Operating Modes

- **Self-mode:** Corvia indexes its own monorepo (dogfooding)
- **Target-mode:** End user opens their project, Corvia creates `.corvia/` and indexes it
- **Dual-mode:** Corvia dev uses VS Code Extension Development Host (two windows)

---

## 3. Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Extension / UI | TypeScript | Native VS Code extension language |
| Core backend | Rust | Speed, safety, tree-sitter has native Rust bindings |
| Agent layer | Rust (MCP server) | 4,700 QPS vs Python's ~300 QPS. "Entire stack in Rust" portfolio story. |
| Fine-tuning only | Python | PyTorch/HuggingFace have no Rust equivalent (M5c only) |
| LLM provider | Provider-agnostic | Trait-based abstraction. Anthropic, OpenAI, Ollama supported. |
| Embedding model | Configurable | Local default (Nomic ONNX), cloud option (OpenAI). |
| Vector store | Configurable | USearch default, pluggable backends. |
| Extension ↔ Core | NAPI-RS | In-process, zero IPC overhead for hot path (indexing, search) |
| Extension ↔ Agents | MCP over stdio | Sidecar process. LLM calls take seconds anyway — MCP overhead negligible. Showcases MCP. |

---

## 4. Monorepo Structure

```
corvia/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── corvia-common/                  # Shared types, config, provider abstractions
│   │   └── src/
│   │       ├── config.rs              # corvia.toml parsing, defaults
│   │       ├── events.rs             # Event enum + JSONL serialization
│   │       ├── providers.rs          # LLM provider trait
│   │       ├── embedders.rs          # Embedding provider trait
│   │       └── stores.rs             # Vector store trait
│   │
│   ├── corvia-core/                    # Hot path — indexing engine
│   │   └── src/
│   │       ├── watcher.rs            # File system watcher
│   │       ├── parser.rs             # tree-sitter AST parsing
│   │       ├── chunker.rs            # AST-aware chunking
│   │       ├── indexer.rs            # Parse → chunk → embed → store
│   │       ├── search.rs             # Vector similarity search
│   │       └── events.rs             # Append to events.jsonl
│   │
│   ├── corvia-embed/                   # Embedding model runner
│   │   └── src/
│   │       ├── onnx.rs               # Local ONNX runtime
│   │       ├── cloud.rs              # Cloud embedding API
│   │       └── lib.rs                # Dispatches to configured provider
│   │
│   ├── corvia-napi/                    # NAPI-RS bridge to TypeScript
│   │   └── src/
│   │       └── lib.rs                # Exposes core functions to Node.js
│   │
│   └── corvia-mcp/                     # MCP server — agent layer
│       └── src/
│           ├── server.rs             # MCP server setup (rmcp crate)
│           ├── agents/
│           │   ├── doc_gen.rs        # Documentation generation
│           │   ├── distiller.rs      # Conversation → decision distillation
│           │   ├── reviewer.rs       # Doc quality review
│           │   ├── eval.rs           # Accuracy, freshness, completeness
│           │   ├── narrator.rs       # Evolution narrative generator
│           │   └── drift.rs          # Semantic drift detector
│           ├── tools.rs              # MCP tool definitions
│           ├── prompts.rs            # Structured prompts per agent
│           └── history.rs            # Temporal queries over events + git
│
├── extension/                          # VS Code extension
│   └── src/
│       ├── extension.ts              # Activation, lifecycle
│       ├── sidebar/                  # Tree views (docs, agents, decisions)
│       ├── webview/                  # Dashboard, chat panel (React)
│       ├── codelens/                 # Per-function doc status
│       ├── statusbar/                # Live metrics
│       └── mcp-client.ts            # MCP client connecting to corvia-mcp
│
├── scripts/fine-tune/                 # Python — M5c only
├── test-projects/                      # Git-ignored, dual-mode testing
├── docs/                               # Corvia's own living docs (dogfood)
├── .corvia/                            # Corvia documenting itself
└── corvia.toml                         # Default config template
```

### Crate Dependency Graph

```
corvia-common (shared types, traits)
     ▲           ▲           ▲
     │           │           │
corvia-embed  corvia-core  corvia-mcp
                 ▲
                 │
            corvia-napi
```

Core and MCP are decoupled — they share data through `.corvia/` (index files, events.jsonl), not direct function calls. Core can be tested without agents. MCP can restart without re-indexing.

---

## 5. Data Flow Detail

### Flow 1: Code → Index (Hot Path — NAPI-RS)

```
File saved → extension/watcher → corvia-napi → corvia-core
  → parser.rs (tree-sitter) → chunker.rs (AST-aware) → indexer.rs (embed + store)
  → .corvia/index/ updated → events.jsonl appended → status bar updates
```

| Trigger | Scope | Speed target |
|---------|-------|--------------|
| File save | Changed file only | <500ms |
| Git commit | Full consistency check | 1-10s |

### Flow 2: Code + Index → Living Docs (MCP)

```
Hook fires → extension → MCP client → corvia-mcp
  → doc_gen agent (search + read + LLM) → reviewer agent (quality check)
  → pass: write .corvia/docs/ + append events.jsonl → Code Lens + sidebar update
  → fail: feedback → doc_gen retries
```

Agent coordination is sequential, not orchestrated: `doc_gen → reviewer → (pass | retry)`.

### Flow 3: Conversations → Decisions (MCP)

```
Chat message → Tier 1 (raw log) → events.jsonl
  → distiller agent (async) → extracts decisions, rationale, alternatives
  → Tier 2 (distilled YAML) → events.jsonl
  → Tier 3: doc_gen weaves decisions into living docs on next run
```

"Save as Decision" button is a shortcut — user promotes a message directly to Tier 2.

### Flow 4: Temporal Queries (On-Demand)

```
User asks "How did auth evolve?"
  → grep events.jsonl for auth-related events
  → git log --tags for snapshot points
  → narrator agent: read_doc_at_version (git show tag:path) + semantic_diff (LLM)
  → evolution narrative displayed in chat or dashboard
```

Git is the time machine. `git show <tag>:.corvia/docs/auth.md` retrieves any historical version.

---

## 6. Historical Semantic Understanding

### Architecture: Simplified Hybrid (Event Log + Git Tags + On-Demand LLM)

```
Action happens (file save, doc generated, decision made)
  │
  ├──→ Update live state (docs, index) ← needed regardless
  │
  └──→ Append one line to .corvia/events.jsonl ← one extra line of code
           │
           └──→ On git commit: auto-tag + DVC push indexes ← git hook
```

Not classic event sourcing (no CQRS, no projections, no state derivation from events). Current state is always live. Events are a secondary recording for history.

### Event Schema

```jsonl
{"ts":"2026-02-25T10:30:00Z","type":"FileChanged","path":"src/auth.rs","diff_hash":"abc123"}
{"ts":"2026-02-25T10:30:01Z","type":"ChunksIndexed","file":"src/auth.rs","chunks":3}
{"ts":"2026-02-25T10:31:00Z","type":"DocGenerated","doc":"docs/auth.md","source_chunks":["auth.rs:1","auth.rs:2"],"caused_by":["D7"]}
{"ts":"2026-02-25T10:35:00Z","type":"DecisionCaptured","id":"D8","summary":"Use JWT over sessions","conversation":"C12"}
```

### Five Capabilities (Priority Order)

| Priority | Capability | How it works |
|----------|-----------|-------------|
| P1 | **Semantic diff** | LLM compares two doc versions on-demand, produces meaning-level summary |
| P2 | **Decision provenance** | Events carry `caused_by` refs. Query: "why does this doc exist?" → trace event chain |
| P3 | **Evolution narratives** | Filter events by topic, walk chronologically, LLM narrates the story |
| P4 | **Temporal queries** | `git show <tag>:path` retrieves historical docs. RAG over past state. |
| P5 | **Drift detection** | Compare current doc embeddings vs current code embeddings. Flag semantic divergence. |

### Versioning Strategy

| Content | Version with | Rationale |
|---------|-------------|-----------|
| `docs/`, `conversations/distilled/`, `conversations/living/`, `events.jsonl`, `corvia.toml` | Git | Text, diffable, meaningful history |
| `conversations/raw/` | DVC (optional) | Large, frequent |
| `index/` | DVC (optional) | Binary, 10-100MB+ |
| `cache/` | Neither (.gitignore) | Regeneratable |

DVC is optional. Without it, Corvia re-indexes from scratch. With it, `dvc pull` gives full state for team onboarding.

---

## 7. VS Code Extension UI

### Seven Surfaces

| Surface | Purpose |
|---------|---------|
| **Activity Bar** | Raven icon, badge for stale docs / agent errors |
| **Sidebar: Doc Tree** | Living doc tree with freshness icons (✅ ⚠️ 🔴 🔄). Click → preview. Right-click → regenerate, history, provenance. |
| **Sidebar: Agents** | Agent status tree (idle/running/waiting/error). Click → logs. Right-click → run now, view cost. |
| **Sidebar: Chat** | Documentation-focused chat. RAG-grounded answers. Auto-captured to Tier 1. "Save as Decision" button. |
| **Sidebar: Timeline** | Event timeline from events.jsonl, grouped by day. Click → event details. |
| **Code Lens** | Per-function/class: ✅ Documented / ⚠️ Stale / 🔴 Undocumented. Clickable: View, Generate, Regenerate, History. |
| **Dashboard (Editor Tab)** | Full-page webview: agent pipeline visualization, coverage heatmap, cost tracker, eval scores, evolution timeline. |
| **Status Bar** | `🐦 Corvia │ ✅ 1,248 chunks │ 🤖 2 active │ 📖 72% │ 💰 $0.12` — each segment clickable. |
| **Notifications** | Event-driven toasts. Configurable: all / important / none. |

### Code Lens Detail

```python
# ✅ Documented (2h ago) | View ↗ | History ↗
def authenticate(token: str) -> User:

# ⚠️ Stale — code changed 3x since doc | Regenerate ↗
class TokenValidator:

# 🔴 Undocumented | Generate ↗
def _parse_claims(raw: bytes) -> Claims:
```

Staleness: compare `last_indexed_at` of code chunk vs `generated_at` of linked doc. Pure metadata, no LLM.

---

## 8. Configuration

### corvia.toml

```toml
[project]
name = "my-app"
languages = ["rust", "typescript"]

[indexing]
on_save = true
on_commit = true
exclude = ["node_modules/**", "target/**", ".git/**", "*.lock", "*.min.js"]
chunk_max_tokens = 512

[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"

[llm.local]
enabled = false
provider = "ollama"
model = "llama3"
url = "http://localhost:11434"

[embedding]
provider = "local"
model = "nomic-embed-text-v1.5"

[vector_store]
backend = "usearch"
dimensions = 768

[docs]
output_dir = "docs"
style = "developer"
generate_on = "commit"

[conversations]
capture = true
auto_distill = true
distill_on = "session_end"

[versioning]
events = true
dvc = false

[ui]
notifications = "important"
codelens = true

[history]
semantic_diff = "on_demand"
drift_detection = "daily"
event_max_size = "10MB"
```

### .corvia/ Directory

```
.corvia/
├── corvia.toml                         # Config (git-tracked)
├── events.jsonl                        # Event log (git-tracked)
├── docs/                               # Generated living docs (git-tracked)
│   └── _index.md                       # Auto-generated TOC
├── conversations/
│   ├── raw/                            # Tier 1 (DVC or gitignored)
│   ├── distilled/                      # Tier 2 decisions (git-tracked)
│   └── living/                         # Tier 3 summaries (git-tracked)
├── index/                              # Vector indexes (DVC or gitignored)
└── cache/                              # Always gitignored
```

### First-Run Flow

1. User opens project with Corvia installed
2. No `.corvia/` found → notification: "Initialize?"
3. [Initialize] → auto-detect language/framework → generate `corvia.toml` with smart defaults → create `.corvia/` → append `.gitignore` → begin initial index
4. Status bar shows progress → "Indexed 1,248 chunks in 3.2s. Ready."

### Progressive Feature Unlocking

UI only shows sections that have data. Features appear naturally as the user engages:
- **Install:** Indexing, Code Lens, status bar, search
- **First doc generated:** Doc tree, coverage
- **First chat:** Conversation capture, "Save as Decision"
- **After 1 week:** Drift detection, evolution timeline, evals

---

## 9. Milestone Roadmap

### Dependency Map

```
M1 ──→ M2a ──→ M2b ──┐
  │                    ├──→ M4 ──→ M5a ──→ M5b ──→ M5c ──→ M6 ──→ M7
  └──→ M3 ────────────┘
```

10 milestones = 10 LinkedIn posts.

### Milestone Summary

| # | Name | Key deliverable | Techniques | LinkedIn angle |
|---|------|----------------|------------|----------------|
| M1 | Index & Understand | Rust AST indexer + extension skeleton | RAG, streaming, tool use | "Rust-powered indexer understands any repo in seconds" |
| M2a | Watch & React | Hook/event system | Hooks, streaming | "Event-driven AI reacts to every code change" |
| M2b | Agents Write Docs | MCP server + doc generation | Agent orchestration, MCP, tool use, prompts | "AI agents coordinate to write documentation" |
| M3 | Conversations → Docs | Chat + distillation pipeline | Memory, hooks, prompts | "Every design decision automatically preserved" |
| M4 | Self-Documenting | Corvia documents itself | Full recursion | "AI tool that writes its own documentation" |
| M5a | Evals | Quality grading framework | Evals | "Automated quality gates for AI output" |
| M5b | Optimization | Cost tracking + prompt tuning | Agent optimization | "Reduced agent costs X% while improving quality" |
| M5c | Custom Models | Fine-tuned embeddings | Fine-tuning | "Fine-tuned a model on my own codebase" |
| M6 | Multi-Modal | Diagrams + images | Multi-modal, MCP | "Reads architecture diagrams, keeps docs in sync" |
| M7 | OSS Launch | Published extension + packages | All 12 | "Introducing Corvia — try it today" |

### Parallelism

- M2a/M2b and M3 run in parallel after M1
- M4 is the convergence point (needs M2b + M3)
- M5a → M5b → M5c → M6 → M7 are strictly sequential

### Minimum Viable Corvia

If scope must be cut: M1 + M2a + M2b + M3 + M4 + M5a + M7 (drop M5b, M5c, M6).

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NAPI-RS + tree-sitter integration | Medium | High | Build standalone Rust CLI first, add NAPI bindings second. Fallback: sidecar binary. |
| Rust MCP SDK maturity | Medium | Medium | Start MCP at M2b. Fallback: raw JSON-RPC over stdio. |
| Provider-agnostic abstraction | Medium | Low | Start with one provider, abstract behind trait, add providers incrementally. |
| LLM cost for heavy users | High | Medium | Cost tracker, configurable triggers, local model fallback, caching. |
| VS Code webview performance | Medium | Medium | Lightweight charting, lazy-load, render only visible sections. |
| 12 techniques is ambitious | Medium | Medium | Each delivered at specific milestone. M5c and M6 are droppable. |

---

## 11. Success Criteria

### Portfolio

- A technical evaluator understands Corvia's architecture by reading its self-generated docs
- Decision trail from brainstorm → implementation is visible (D1-D19 in Corvia's own decision tree)
- All 12 LLM techniques are demonstrable via UI surfaces or commands
- Indexing <5s for 10K-file repo, search <100ms, doc generation <30s
- Clean Rust, idiomatic TypeScript, comprehensive tests, CI passing

### OSS Adoption

- Zero to indexed in <2 minutes (install → open → indexed)
- Works on any language without config
- Generated docs answer real questions correctly
- <$1/day for active use with cloud LLM
- Local-only mode fully functional

---

## 12. LLM Techniques Showcased

| # | Technique | Where in Corvia | Milestone |
|---|-----------|----------------|-----------|
| 1 | RAG pipeline | Dual-index code + docs, AST chunking, vector search | M1 |
| 2 | Agent orchestration | doc_gen → reviewer pipeline, multi-agent coordination | M2b |
| 3 | Agent optimization | Benchmarking, prompt tuning, cost tracking | M5b |
| 4 | Hooks / middleware | on-save, on-commit event system | M2a |
| 5 | MCP servers | Rust MCP server exposing tools to agents | M2b |
| 6 | Prompt engineering | Structured prompts per agent, context routing | M2b |
| 7 | Evaluation / evals | Accuracy, completeness, freshness scoring | M5a |
| 8 | Streaming / real-time | Live status bar, dashboard updates during indexing | M1 |
| 9 | Memory / state | Persistent session memory, conversation history | M3 |
| 10 | Fine-tuning | Custom embedding model trained on project data | M5c |
| 11 | Tool use / function calling | MCP tools: search, read, write_doc, semantic_diff | M2b |
| 12 | Multi-modal | Image/diagram understanding and generation | M6 |

---

## 13. Decisions Log

All 19 decisions from the brainstorming session are documented with alternatives considered and rationale in `docs/rfcs/2026-02-25-corvia-brainstorm.md`. Key decisions:

| # | Decision | Choice |
|---|----------|--------|
| D2 | Codebase invariance | Drop-in, tree-sitter AST |
| D3 | Code understanding | AST chunking → path to LLM summarization |
| D4 | Form factor | VS Code extension + optional devcontainer |
| D5 | Multi-source | Code (hybrid) + Docs (input/output) + Prompts (tiered) |
| D6 | Dogfooding | Fully recursive from day one |
| D7 | Target audience | Portfolio first, OSS second |
| D8 | LLM techniques | All 12 |
| D9 | Tech stack | TypeScript + all-Rust backend + Python (fine-tuning only) |
| D10 | Dev approach | Portfolio-milestone driven |
| D11 | Name | Corvia |
| D12 | Extension ↔ Backend | NAPI-RS (hot path) + Rust MCP (agent layer) |
| D13 | Drop Python | All-Rust production, Python fine-tuning only |
| D14 | Monorepo | Cargo workspace |
| D15 | Client repo integration | `.corvia/` directory + workspace detection |
| D16 | License | MIT |
| D17 | VS Code UI | All 7 surfaces |
| D18 | Versioning | Git (text) + DVC (binary) + temporal knowledge graph |
| D19 | History architecture | Simplified hybrid: event log + git tags + on-demand LLM |

---

*Design approved: 2026-02-25*
*Next step: Implementation plan (writing-plans skill)*
