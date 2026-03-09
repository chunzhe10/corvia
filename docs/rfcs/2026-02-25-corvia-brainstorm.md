# Corvia — Living Documentation Framework

## Brainstorming Session: 2026-02-25

Status: **Complete** — Design approved 2026-02-25. Full design doc: `docs/rfcs/2026-02-25-corvia-design.md`

---

## Project Identity

- **Name**: Corvia
- **Tagline**: Knowledge-gathering agents for living documentation
- **Name origin**: Derived from Corvus (the raven constellation). Ravens in mythology are knowledge-gatherers — Odin's ravens Huginn and Muninn flew the world daily gathering knowledge and reporting back. "Corvia" softens the sound while preserving the lineage.
- **Alternatives considered**: Lyra (harmony metaphor), Circinus (compass/loop metaphor), Vela (navigation metaphor), Aquila (eagle-vision metaphor). Corvus was the concept pick but sounded too dark; Corvia retains the meaning with a warmer tone.

---

## Decisions Made

### D2: Codebase Invariance Strategy
- **Decision**: Drop-in for any project (Option A), powered by tree-sitter AST
- **Alternatives considered**:
  - (B) Adapter-based — ship core engine + per-ecosystem adapters (Python, React, Go, etc.)
  - (C) Language-agnostic by design — treat all code as text, let LLM handle everything
- **Why not B**: Per-language adapters create maintenance burden and slow down language support. Tree-sitter already provides 100+ grammars through one dependency — adapters are unnecessary overhead.
- **Why not C**: Research shows pure-LLM chunking is 10-100x more expensive and produces inconsistent boundaries. AST gives deterministic, structurally-correct chunks that LLMs can then interpret.
- **Evidence**: cAST paper (CMU, June 2025) showed +4.3 points Recall@5 on RepoEval, +2.67 Pass@1 on SWE-bench vs fixed-size chunking. Production experience on 300K LOC monorepo showed ~40% reduction in irrelevant retrieval with AST chunking.

### D3: Code Understanding Approach
- **Decision**: Tree-sitter AST chunking with path to LLM summarization
- **Alternatives considered**:
  - (A) Structure only — AST for boundaries, LLM interprets at query time (chosen as starting point)
  - (B) Structure + Semantics — AST chunking + LLM pre-computes summaries per chunk (future path)
  - (C) Semantics only — skip AST, chunk by token windows/heuristics
- **Why not C**: Pure heuristic chunking (blank lines, indentation) loses structural integrity. Research shows it underperforms AST by 4-5 points on retrieval benchmarks.
- **Why A→B phasing**: Start simple (AST only), add LLM summarization pass later when the pipeline is stable. Architecture supports both without changes.
- **Key libraries identified**: supermemory/code-chunk (TS, tree-sitter), astchunk (Python), code-splitter

### D4: Development Environment Form Factor
- **Decision**: VS Code extension (primary) + devcontainer (optional)
- **Alternatives considered**:
  - (A) Devcontainer-first — framework IS the devcontainer definition
  - (C) Standalone CLI + Devcontainer — CLI scaffolds setup, devcontainer optional
- **Why not A**: Requires Docker, heavy for onboarding, limits audience. Many developers don't use devcontainers.
- **Why not C (as primary)**: CLI-only misses the UI showcase opportunity critical for portfolio. VS Code extension allows visual demonstration of streaming, real-time indexing, agent coordination.
- **Why extension + optional devcontainer**: Extension is lightweight (no Docker), showcases UI skills. Devcontainer available for users who want guaranteed environment consistency.

### D5: Multi-Source Architecture
- **Decision**: Three sources with tiered processing
- **No alternatives presented** — this was a refinement question. User specified the three sources and tiering approach directly.

#### Source 1: Codebase
- Hybrid indexing: lightweight re-index on file save (real-time), full re-index on git commit
- AST chunking via tree-sitter
- **Alternatives considered**: (a) save-only, (b) commit-only, (c) both/hybrid
- **Why hybrid**: Real-time gives immediate feedback during development; commit-triggered gives thorough re-index with full context. Tiered approach balances responsiveness with completeness.

#### Source 2: Documentation
- Docs are BOTH input AND output (living loop)
- Two documentation layers:
  - **Framework docs**: Documents Corvia itself (portfolio layer)
  - **Project docs**: Documents the client repo Corvia is pointed at
- Separate pipeline from code — markdown-optimized chunking, different embedding strategy for prose vs code
- **Alternatives considered**: (a) treat same as code, (b) separate pipeline, (c) docs as output only
- **Why separate pipeline + input/output**: Prose and code have different optimal chunking strategies. Docs must be both readable inputs (existing knowledge) and generated outputs (new knowledge). Dual-layer separates framework portfolio content from per-project content.

#### Source 3: Development Prompts/Conversations
- Tiered retention (Option D — all tiers):
  - **Tier 1 (Raw)**: Full conversation logs — time-limited or size-capped
  - **Tier 2 (Distilled)**: Extracted decisions, rationale, alternatives — permanent, searchable
  - **Tier 3 (Living doc)**: Polished summaries integrated into project documentation — permanent, markdown
- Pipeline: Raw → (LLM distillation agent) → Distilled → (doc generation agent) → Living doc
- **Alternatives considered**: (a) full logs only, (b) decisions only, (c) auto-distilled summaries only
- **Why all tiers**: Raw logs preserve full context for debugging/auditing. Distilled decisions are the searchable knowledge layer. Living doc is the polished output. Each tier serves a different use case; tiering avoids losing information while keeping the doc layer clean.

### D6: Self-Documenting (Dogfooding)
- **Decision**: Yes, fully recursive from day one (Option A)
- **Alternatives considered**:
  - (B) Yes, but later — build first with manual docs, point at itself once stable
  - (C) No, keep separate — framework only generates docs for target projects
- **Why not B**: Delays the most compelling portfolio story. Early conversations (like this brainstorm) are the most valuable to capture — they show the genesis of architectural decisions.
- **Why not C**: Misses the "ouroboros" showcase moment entirely. Self-documentation is the headline feature for portfolio impact.
- **Rationale**: The portfolio IS the output of the tool. Conversations building Corvia get captured, distilled, and become Corvia's own living docs.

### D7: Target Audience
- **Decision**: Portfolio showcase primary (C), open source community second (D)
- **Alternatives considered**:
  - (A) Solo developers / indie hackers — zero-config emphasis
  - (B) Small teams (2-10 devs) — collaboration features, shared docs
- **Why not A as primary**: Solo dev focus would de-prioritize the "impressive demo" quality needed for portfolio. Usability is important but secondary to demonstrating technical depth.
- **Why not B**: Team features (collaboration, permissions, shared state) add significant complexity without portfolio value. Can be added post-launch if OSS adoption demands it.
- **Priority order**: Showcase first → OSS adoption second → team features later if needed.

### D8: LLM Techniques to Showcase
- **Decision**: All 12 techniques — must-have for all
- **No alternatives** — user requested all techniques to demonstrate comprehensive LLM mastery.

| # | Technique | Status |
|---|-----------|--------|
| 1 | RAG pipeline (dual-index, AST chunking, embeddings, vector search) | Must have |
| 2 | Agent orchestration (indexer, distiller, doc-generator, reviewer) | Must have |
| 3 | Agent optimization (benchmarking, prompt tuning, cost tracking) | Must have |
| 4 | Hooks / middleware (event-driven pipeline) | Must have |
| 5 | MCP servers (custom Model Context Protocol tools) | Must have |
| 6 | Prompt engineering (skills, context routing, chain-of-thought) | Must have |
| 7 | Evaluation / evals (automated doc quality checks) | Must have |
| 8 | Streaming / real-time (live VS Code updates) | Must have |
| 9 | Memory / state (persistent cross-session memory) | Must have |
| 10 | Fine-tuning / custom models (embeddings, classifiers, summarizers) | Must have |
| 11 | Tool use / function calling (LLM tool invocation, structured output) | Must have |
| 12 | Multi-modal (diagrams, screenshots, architecture images) | Must have |

### D9: Tech Stack
- **Extension / UI**: TypeScript (native VS Code extension language)
- **Core backend**: Rust (speed, safety, tree-sitter has native Rust bindings)
- **ML / agent glue**: ~~Python~~ → Rust (all agent orchestration, LLM calls, evals in Rust)
- **Python**: Fine-tuning only (M5c milestone) — PyTorch/HuggingFace remain Python-only
- **LLM provider**: Provider-agnostic with local model support. Abstract LLM layer.
- **Embedding model**: Configurable — local default (e.g., Nomic ONNX), cloud option (e.g., OpenAI)
- **Vector store**: Configurable — USearch default, pluggable backends
- **Alternatives considered for language**: TypeScript-only (simpler but slower for core), Python+TS+Rust tri-language (Python for agent glue). Chose all-Rust because benchmarks show Rust MCP servers at 4,700 QPS vs Python's ~300 QPS, Rust agent frameworks (AutoAgents) 25-84% faster than LangChain/LangGraph, and <1.1GB memory vs >4.7GB for Python. Portfolio story: "entire agent stack in Rust" > "used LangChain like everyone else."
- **Alternatives considered for LLM provider**: Claude-only (simpler), OpenAI-only, provider-agnostic without local. Chose full agnostic + local for maximum OSS adoption and offline capability.
- **Alternatives considered for embeddings**: Nomic-only local, cloud-only. Chose configurable to support both offline and quality-optimized use cases.
- **Alternatives considered for vector store**: USearch-only, ChromaDB/LanceDB, Qdrant/Weaviate. Chose configurable with lightweight default to keep onboarding simple while allowing production scaling.

### D12: Extension ↔ Backend Communication
- **Decision**: Hybrid — Rust via NAPI-RS (in-process, hot path) + Rust MCP server (sidecar, agent layer)
- **Alternatives considered**:
  - (A) Rust sidecar binary + JSON-RPC over stdio — battle-tested (rust-analyzer pattern), but IPC overhead for indexing hot path
  - (B) Everything through MCP — clean abstraction but 300-800ms handshake overhead, too slow for on-save indexing
  - (C) Rust compiled to WASM — no sidecar process, but limited filesystem access, limited tree-sitter support
  - (D) Rust as NAPI-RS native addon only — zero overhead but can't showcase MCP
- **Why hybrid**: NAPI-RS for hot path (AST parsing, indexing, vector search — happens on every file save, needs <10ms). MCP for agent layer (LLM calls take seconds anyway, MCP overhead negligible, showcases MCP technique for portfolio).
- **Evidence**: ast-grep already validates NAPI-RS + tree-sitter pattern in production. Official Rust MCP SDK (`rmcp`) exists. Rust MCP servers benchmark at 4,700+ QPS.
- **Research**: rust-analyzer uses sidecar + LSP, Sourcegraph Cody uses agent + JSON-RPC, Oso used Rust→WASM. The hybrid approach combines the best patterns from each.

### D13: Drop Python from Agent Layer
- **Decision**: All-Rust for production code. Python only for fine-tuning experiments (M5c).
- **Alternatives considered**:
  - Python for all agent orchestration (LangChain/LangGraph ecosystem)
  - Go for MCP servers (official SDK, fast startup, goroutines)
  - Rust+Python hybrid (Rust core, Python agents)
- **Why not Python**: 5-26x slower MCP throughput, 4x more memory, +5,788ms latency overhead at 5K RPS. "Used LangChain" is not a differentiator.
- **Why not Go**: Rust is 10-20% faster in CPU-bound tasks, lower memory, and we're already using Rust for the core — one fewer language to maintain.
- **Why Rust all the way**: Portfolio impact ("entire stack in Rust"), consistent toolchain, tree-sitter is Rust-native, official Rust MCP SDK exists, AutoAgents framework proves Rust agent orchestration is viable.
- **Python exception**: Fine-tuning (M5c) still needs PyTorch/HuggingFace — no Rust equivalent yet.

### D10: Development Approach
- **Decision**: Portfolio-milestone driven (Option D)
- **Alternatives considered**:
  - (A) Ship incrementally — MVP fast, layer features
  - (B) Build full architecture first — all abstractions before features
  - (C) Spike-driven — isolated POCs then integrate
- **Why not A**: Pure incremental risks an incoherent portfolio story — each piece works but doesn't tell a narrative.
- **Why not B**: Too long before first demo moment. Portfolio needs visible progress.
- **Why not C**: Integration risk at the end. Spikes don't produce demo-worthy artifacts.
- **Rationale**: Each milestone is a LinkedIn post. Narrative builds from "watch it index" → "watch it document" → "watch it document itself" → "watch it grade itself" → "use it yourself."

### D11: Project Name
- **Decision**: Corvia
- **Alternatives considered**: Lyra, Corvus, Circinus, Vela, Aquila, Munin, Branwen, Merla
- **Why Corvia over Corvus**: Same raven meaning but warmer, more modern sound. Corvus felt too dark/Gothic for an approachable developer tool.
- **Why Corvia over Lyra**: Lyra (harmony) is beautiful but generic — could be any music/data tool. Corvia's raven lineage directly maps to the "agents gathering knowledge" concept.
- **Why Corvia over Munin**: Munin (Odin's raven of memory) was runner-up. Strong mythology but less intuitive to spell/pronounce for international audience.

---

## Milestones (Draft — Under Discussion)

### Dependency Map

```
M1 (Index & Understand) ──→ M2 (Generate Docs) ──→ M4 (Self-Document)
       │                          │                        │
       │                          ↓                        ↓
       │                    M3 (Capture Prompts) ──→ M4 (Self-Document)
       │                                                   │
       ↓                                                   ↓
M1 ──────────────────────────────────────────────→ M5 (Evals & Optimization)
                                                           │
                                                           ↓
                                                   M6 (Multi-modal & MCP)
                                                           │
                                                           ↓
                                                   M7 (OSS Launch)
```

### M1: "Watch Corvia index and understand any repo in seconds"
- **Depends on**: Nothing (foundation)
- **Blocks**: M2, M3, M5
- **Techniques**: RAG pipeline, AST chunking (Rust), streaming/real-time, VS Code extension skeleton
- **Deliverables**: Rust AST indexer, vector search, basic extension UI showing indexing progress
- **LinkedIn angle**: "I built a Rust-powered code indexer that understands any repo in seconds"

### M2a: "Corvia watches your code and reacts"
- **Depends on**: M1
- **Blocks**: M2b
- **Techniques**: Hooks/middleware, streaming/real-time
- **Deliverables**: Event-driven hook pipeline — on-save and on-commit triggers, event system architecture
- **LinkedIn angle**: "Event-driven AI — my framework reacts to every code change in real-time"

### M2b: "Agents that write your docs"
- **Depends on**: M2a (agents need the hook pipeline to trigger them)
- **Blocks**: M4
- **Techniques**: Agent orchestration, tool use/function calling, prompt engineering
- **Deliverables**: Doc generation agent, indexer agent coordination, generated markdown output
- **LinkedIn angle**: "I built AI agents that coordinate to write documentation from your code"

### M3: "Every conversation becomes documentation"
- **Depends on**: M1 (needs index for context), can parallel with M2
- **Blocks**: M4
- **Techniques**: Memory/state, hooks/middleware, prompt engineering
- **Deliverables**: Conversation capture, tiered distillation pipeline, decision extraction agent
- **LinkedIn angle**: "What if every design decision you discussed with AI was automatically preserved?"

### M4: "Corvia documents itself — here's the proof"
- **Depends on**: M2 + M3 (needs both doc generation and conversation capture)
- **Blocks**: M5
- **Techniques**: Full pipeline recursion — all prior techniques applied to self
- **Deliverables**: Corvia's own living docs generated by Corvia, visible decision trail from brainstorm → implementation
- **LinkedIn angle**: "I built an AI tool that writes its own documentation. Here's the proof."

### M5a: "How good are the docs? Corvia grades itself"
- **Depends on**: M4 (needs generated docs to evaluate)
- **Blocks**: M5b
- **Techniques**: Evaluation/evals
- **Deliverables**: Eval framework — accuracy, freshness, completeness checks on generated docs
- **LinkedIn angle**: "Building AI isn't enough — here's how I built automated quality gates for AI output"

### M5b: "Making agents faster, cheaper, smarter"
- **Depends on**: M5a (need eval baselines before optimizing)
- **Blocks**: M5c
- **Techniques**: Agent optimization, cost tracking
- **Deliverables**: Benchmarking framework, prompt tuning pipeline, cost tracking dashboards in VS Code
- **LinkedIn angle**: "I reduced my AI agent costs by X% while improving quality — here's the framework"

### M5c: "Custom models for custom problems"
- **Depends on**: M5b (need optimization metrics to justify fine-tuning)
- **Blocks**: M6
- **Techniques**: Fine-tuning/custom models
- **Deliverables**: Custom-trained embedding model or classifier on code+doc pairs, before/after comparison
- **LinkedIn angle**: "I fine-tuned a model on my own codebase — here's what I learned and the results"

### M6: "Diagrams, screenshots, and architecture — Corvia sees everything"
- **Depends on**: M5 (evals ensure quality of multi-modal output)
- **Blocks**: M7
- **Techniques**: Multi-modal, MCP servers, fine-tuning/custom models
- **Deliverables**: Image/diagram understanding, architecture diagram generation, custom MCP servers
- **LinkedIn angle**: "Corvia now reads your architecture diagrams and keeps docs in sync with what it sees"

### M7: "Open source launch — plug Corvia into your project today"
- **Depends on**: M6 (all features complete)
- **Techniques**: All 12 — polished and packaged
- **Deliverables**: Published VS Code extension, npm/crate packages, devcontainer option, docs site, contributing guide
- **LinkedIn angle**: "Introducing Corvia — open source living documentation powered by AI. Try it today."

### Dependency Map (Final)

```
M1 ──→ M2a ──→ M2b ──┐
  │                    ├──→ M4 ──→ M5a ──→ M5b ──→ M5c ──→ M6 ──→ M7
  └──→ M3 ────────────┘
```

- **10 milestones = 10 LinkedIn posts**
- **M2a/M2b and M3 can run in parallel** after M1 (hooks+agents and conversation capture are independent tracks)
- **M4 is the convergence point** — needs both doc generation (M2b) and conversation capture (M3)
- **M5a → M5b → M5c are strictly sequential** (eval → optimize → fine-tune builds on prior results)
- **Alternatives considered**: Keeping M2 and M5 as single milestones. Split because M2 combines event system + agent orchestration (different concerns), and M5 combines evals + optimization + fine-tuning (three distinct disciplines).

---

### D14: Monorepo Layout
- **Decision**: Cargo workspace monorepo (Option A)
- **Structure**:
  ```
  corvia/
  ├── Cargo.toml                    # Workspace root
  ├── crates/
  │   ├── corvia-core/              # Tree-sitter AST, chunking, vector search, file watcher
  │   ├── corvia-napi/              # NAPI-RS bindings (exposes core to TS extension)
  │   ├── corvia-mcp/               # MCP server (agents, LLM calls, doc gen, evals)
  │   ├── corvia-common/            # Shared types, config, provider abstractions
  │   └── corvia-embed/             # Embedding model runner (ONNX runtime, configurable)
  ├── extension/                    # TypeScript — VS Code extension UI
  ├── scripts/fine-tune/            # Python — fine-tuning only (M5c)
  ├── docs/                         # Corvia's own living docs (dogfood output)
  └── corvia.toml                   # User-facing config for target projects
  ```
- **Alternatives considered**:
  - (B) Polyrepo — separate repos per language. Rejected: cross-repo coordination overhead, harder to dogfood.
  - (C) Monorepo with independent publishing — extra CI complexity without clear benefit at this stage.

### D15: Client Repo Integration & Recursive Architecture
- **Decision**: `.corvia/` directory in target repo + VS Code workspace detection (Option C — both)
- **Three operating modes**:
  - **Self-mode**: Corvia indexes its own monorepo (dogfooding)
  - **Target-mode**: End user opens their project, Corvia creates `.corvia/` and indexes it
  - **Dual-mode**: Corvia dev uses VS Code Extension Development Host (two windows)
- **`.corvia/` directory structure** (created in any target repo):
  ```
  .corvia/
  ├── corvia.toml       # Project-specific config (auto-generated, customizable)
  ├── index/            # Vector indexes
  ├── docs/             # Generated living docs
  ├── conversations/    # Prompt tiers (raw → distilled → living)
  └── cache/            # AST cache, embeddings cache
  ```
- **Key insight**: `.corvia/` is target-agnostic — the same pipeline runs whether the target is Corvia itself or an external project. Recursion requires no special wiring.
- **Gitignore**: `index/`, `cache/` always ignored. `docs/` and `conversations/` user's choice to commit.
- **Development workflow**: `test-projects/` in monorepo (git-ignored) for sample repos. VS Code Extension Development Host handles two-instance testing natively.
- **Alternatives considered**:
  - (A) Workspace detection only — no explicit config. Rejected: doesn't support CI pipelines or multi-repo setups.
  - (B) Explicit path config only — user must configure. Rejected: friction for the common case (just open a project).
- **Research**: MyCoder uses same-repo self-referential pattern. VS Code Extension Development Host is the standard solution for extension-develops-itself workflows.

---

### D16: Licensing
- **Decision**: MIT
- **Alternatives considered**: Apache 2.0 (patent protection), AGPL (copyleft, prevents cloud provider exploitation)
- **Why MIT**: Maximum adoption for OSS. Most dev tools (Cursor, Continue.dev, ast-grep) use MIT. Simplest for portfolio — no friction for anyone evaluating.

### D17: VS Code Extension UI
- **Decision**: Full 7-surface UI approach
- **Surfaces used**:
  1. **Activity Bar** — Raven icon, badge for stale docs / agent attention needed
  2. **Sidebar Tree View** — Living doc tree (with freshness icons), agent status tree, decision log
  3. **Sidebar Webview** — Chat panel (documentation-focused, conversations auto-captured to tier pipeline, "Save as decision" button)
  4. **Code Lens** — Per-function/class doc status inline (✅ Documented, ⚠️ Stale, 🔴 Undocumented) with clickable actions (View/Generate/Regenerate)
  5. **Editor Tab (Webview)** — Full dashboard: agent pipeline visualization, cost tracker, eval scores, coverage heatmap, conversation timeline
  6. **Status Bar** — Always-visible: chunks indexed, active agents, doc coverage %, daily cost
  7. **Notifications** — Event-driven toasts for indexing, staleness, distillation events
- **Differentiation from Cursor/Cody**: Code Lens doc status (no other tool does this), living doc tree with freshness, agent pipeline visualization, cost tracking, self-documentation visible in same UI
- **Alternatives considered**: Minimal UI (sidebar only), CLI-only interface. Rejected: misses portfolio showcase opportunity, VS Code surfaces are the demo.

### D18: Doc Versioning & Historical Semantic Understanding
- **Decision**: Tiered versioning (git for text, DVC for binaries) + temporal knowledge graph
- **Priority order**: Semantic diff (P1) → Decision provenance (P2) → Evolution narratives (P3) → Temporal queries (P4) → Drift detection (P5)

#### Versioning Strategy by Content Type

| Content | Version with | Why |
|---------|-------------|-----|
| `docs/` (generated markdown) | **Git** | Text, diffable, meaningful history |
| `conversations/tier-3` (living doc) | **Git** | Small text, infrequent changes |
| `conversations/tier-2` (distilled decisions) | **Git** | Structured YAML/JSON, permanent record |
| `conversations/tier-1` (raw logs) | **DVC** (optional) | Large, frequent, optional reproducibility |
| `index/` (vector embeddings) | **DVC** | Binary blobs, 10-100MB+, regeneratable but expensive |
| `cache/` (AST cache) | **Neither** (always .gitignore) | Regeneratable, ephemeral |
| `corvia.toml` (config) | **Git** | Tiny text config |

#### Historical Semantic Capabilities (Priority Order)

**P1: Semantic Diff** — Not just "line 42 changed" but "the auth module migrated from session-based to JWT-based." Corvia compares doc versions semantically using LLM, producing human-readable summaries of what changed in meaning, not just in text.

**P2: Decision Provenance** — Every doc section and code region links back to the decisions and conversations that created/modified it. A provenance chain: Code function → Doc section → Decision D7 → Conversation C12 → Original brainstorm. Stored as metadata in the temporal knowledge graph.

**P3: Evolution Narratives** — Corvia auto-generates timeline stories: "The auth system evolved through 4 phases: basic sessions (Sprint 1) → OAuth integration (Sprint 3) → JWT migration (Sprint 5) → MFA addition (Sprint 7). Each driven by [linked decisions]." Generated from the provenance graph.

**P4: Temporal Queries** — RAG over time. "How did emissions calculation work 3 months ago?" retrieves historical doc versions, not current. Requires versioned embeddings or re-embedding historical snapshots at query time.

**P5: Drift Detection** — Corvia notices semantic drift between code and docs. Not just "file modified since last doc update" (timestamp) but "the code now uses a different algorithm than what the docs describe" (semantic comparison). Flags meaningful divergence.

#### Implementation Architecture

**Selected approach: Simplified Hybrid (Event Log + Git Tags + On-Demand LLM)**

Three alternatives were evaluated:
1. **Event-Sourced** — all state derived from immutable events (CQRS pattern). Strong history but complex: requires projections, replay, eventual consistency.
2. **Snapshot-Based** — periodic full-state snapshots, diffed semantically. Simple but coarse: can't see between snapshots, provenance is metadata-only.
3. **Hybrid** — events for history, snapshots for fast access. Best of both but described with unnecessary CQRS complexity.

**Key simplification insight**: Classic event sourcing derives current state FROM events. Corvia doesn't need that — current state already exists (live docs, live index). Events are a **secondary recording for history**, not the primary source of truth. This eliminates CQRS, projections, and event replay as requirements.

**The actual implementation is three simple things:**

```
Action happens (file save, doc generated, decision made)
  │
  ├──→ Update live state (docs, index) ← needed regardless of versioning
  │
  └──→ Append one line to .corvia/events.jsonl ← one extra line of code
           │
           └──→ On git commit: auto-tag + DVC push indexes ← git hook
```

**Component breakdown:**

| Component | What it actually is | Complexity |
|-----------|-------------------|------------|
| Event log | Append JSON line to `.corvia/events.jsonl` | Trivial — `serde_json::to_string() >> file` |
| Event types | Rust enum: `FileChanged`, `DocGenerated`, `DecisionCaptured`, etc. | Low — Rust enums |
| Provenance | `caused_by: [decision_id, conversation_id]` field on events | Low — just metadata |
| Snapshots | Git tags on commits (text tracked by git, binary by DVC) | Trivial — git tags, not custom directories |
| Semantic diff | LLM prompt comparing two doc versions, run on-demand | Medium — but only runs when asked |
| Temporal queries | Load git-tagged state, or filter events.jsonl by timestamp | Low — git checkout tag or grep JSONL |
| Evolution narratives | Filter events by topic, walk chronologically, LLM narrates | Low — query pattern over event log |
| Drift detection | Compare current doc embedding vs current code embedding | Medium — cosine similarity + LLM verification |
| Event rotation | Cap at 10MB or 10K lines, rotate old file | Trivial — log rotation |

**No graph database. No complex projections. No event replay for state derivation.** Just a JSONL file, git tags, and smart LLM queries.

**Event schema example:**
```jsonl
{"ts":"2026-02-25T10:30:00Z","type":"FileChanged","path":"src/auth.rs","diff_hash":"abc123"}
{"ts":"2026-02-25T10:30:01Z","type":"ChunksIndexed","file":"src/auth.rs","chunks":3}
{"ts":"2026-02-25T10:31:00Z","type":"DocGenerated","doc":"docs/auth.md","source_chunks":["auth.rs:1","auth.rs:2"],"caused_by":["D7"]}
{"ts":"2026-02-25T10:35:00Z","type":"DecisionCaptured","id":"D8","summary":"Use JWT over sessions","conversation":"C12"}
```

#### DVC Integration
- DVC tracks binary artifacts (`index/`, optionally `tier-1` logs) with pointers in git
- `dvc push` / `dvc pull` for team onboarding (clone + `dvc pull` = full knowledge state without re-indexing)
- Remote storage configurable: S3, GCS, local, or none (re-index from scratch)
- **DVC is optional** — Corvia works without it (just re-indexes), but with DVC you get full reproducibility

#### Alternatives Considered
- **Full CQRS event sourcing** — derives all state from events. Rejected: unnecessary complexity, Corvia already has live state.
- **Snapshot-only** — simple but coarse. Rejected: loses fine-grained history, provenance is metadata-only not structural.
- **Git-only for everything** — rejected: binary indexes bloat git history, slow clones
- **DVC-only for everything** — rejected: overkill for markdown docs, adds friction for simple text versioning
- **No versioning (always regenerate)** — rejected: loses historical semantic understanding entirely
- **Git LFS instead of DVC** — rejected: DVC is purpose-built for ML artifacts, better remote storage options, signals ML/AI competency in portfolio
- **Custom snapshot directories** — rejected: git tags achieve the same thing with zero custom infrastructure

---

## Open Questions (Still in brainstorming)

- CI/CD and release strategy
- Specific agent architecture (which agents, how they coordinate)
- Conversation capture mechanism (how prompts are intercepted)
- Eval framework specifics
- Fine-tuning dataset and approach

---

## Research References

- [cAST Paper — CMU (arxiv)](https://arxiv.org/html/2506.15655v1)
- [Better Retrieval Beats Better Models](https://sderosiaux.substack.com/p/better-retrieval-beats-better-models)
- [How Cursor Indexes Codebases](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast)
- [How Cursor Actually Indexes Your Codebase](https://towardsdatascience.com/how-cursor-actually-indexes-your-codebase/)
- [supermemory/code-chunk (GitHub)](https://github.com/supermemoryai/code-chunk)
- [Building RAG on Codebases — LanceDB](https://lancedb.com/blog/building-rag-on-codebases-part-1/)
- [Building code-chunk: AST Aware Code Chunking](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/)
- [RAG for 10k Repos — Qodo](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/)
- [Enhancing LLM Code Gen with RAG and AST Chunking](https://vxrl.medium.com/enhancing-llm-code-generation-with-rag-and-ast-based-chunking-5b81902ae9fc)
- [Official Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk)
- [Building High-Performance MCP Server with Rust](https://medium.com/@bohachu/building-a-high-performance-mcp-server-with-rust-a-complete-implementation-guide-8a18ab16b538)
- [UltraFast MCP (Rust)](https://lib.rs/crates/ultrafast-mcp)
- [Benchmarking AI Agent Frameworks 2026: AutoAgents (Rust)](https://dev.to/saivishwak/benchmarking-ai-agent-frameworks-in-2026-autoagents-rust-vs-langchain-langgraph-llamaindex-338f)
- [Go vs Python AI Infrastructure Throughput 2026](https://dasroot.net/posts/2026/02/go-vs-python-ai-infrastructure-throughput-benchmarks-2026/)
- [ast-grep (Rust + NAPI)](https://github.com/ast-grep/ast-grep)
- [tree-sitter Rust crate](https://docs.rs/tree-sitter)
- [MCP Server Performance Benchmarks](https://www.tmdevlab.com/mcp-server-performance-benchmark.html)
- [NAPI-RS Framework](https://napi-rs.github.io/napi-rs/)
- [How MyCoder Builds Itself (Recursive Dev)](https://docs.mycoder.ai/blog/how-we-use-mycoder-to-build-mycoder)
- [VS Code Extension Development Host](https://code.visualstudio.com/api/advanced-topics/extension-host)
- [VS Code Multi-Root Workspaces](https://code.visualstudio.com/docs/editing/workspaces/multi-root-workspaces)
