# Corvia

**Organizational reasoning memory for AI agents.**

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Built with Rust](https://img.shields.io/badge/built%20with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Version 0.3.0](https://img.shields.io/badge/version-0.3.0-green.svg)](Cargo.toml)

> **Pre-release (v0.3.0).** Core kernel, two-tier storage, multi-agent coordination, temporal
> queries, knowledge graph, and automated reasoning are implemented and tested (203 tests).
> API surface may change before 1.0.

## What is Corvia?

AI agents are getting memory. Claude has CLAUDE.md, Cursor has rules files, Copilot has
memory — and that's good. But these are *personal notes*: flat files, single-repo, no
relationships, no history. They help one developer in one session. They don't help your
*organization* learn.

Corvia is an open-source knowledge layer (written in Rust) that gives AI agents
**organizational memory** — knowledge that spans repositories, tracks relationships as a
graph, evolves over time, and stays consistent when multiple agents write concurrently.
All stored as Git-trackable JSON files and rebuildable from source.

```
Personal memory:     Agent remembers "use tabs not spaces" for this project
Organizational memory:  Agent knows why the auth system was redesigned last month,
                        which decisions led there, what changed across three repos,
                        and what a different agent discovered about the side effects
```

Corvia gives agents **temporal awareness** ("what did we know last week?"),
**graph-structured relationships** between pieces of knowledge, **multi-agent coordination**
so concurrent agents don't corrupt each other, and **automated reasoning** that catches
knowledge-health problems before they propagate.

Your team's AI agents share a knowledge base. Decisions made in one session inform the next.
Temporal queries let you ask "what changed between the v2.0 release and now?" Agent writes are
staged in isolation and merged on commit — no silent last-write-wins.

## Features

All features below are implemented, tested, and working in the current release.

- **Semantic search** — Vector similarity search over ingested knowledge using local embeddings
  (Ollama/nomic-embed-text). No API keys, no cloud dependency.

- **Knowledge graph** — Directed, labeled edges between knowledge entries. BFS traversal,
  shortest path, cycle detection. Built on petgraph (LiteStore) or SurrealDB graph queries
  (FullStore).

- **Temporal queries** — Bi-temporal model tracking both when knowledge was recorded and when
  it was valid. Query point-in-time snapshots (`as_of`), follow supersession chains (`history`),
  and see what changed over a time range (`evolution`).

- **Automated reasoning** — Five deterministic health checks (stale knowledge, broken
  supersession chains, orphaned entries, dangling graph edges, dependency cycles) that run as
  pure graph algorithms — same input, same findings, every time. Two additional LLM-powered
  checks (semantic gap detection, contradiction finding) layer on top as opt-in complements.

- **Multi-agent coordination** — Session-based write isolation with staging areas, crash
  recovery (resume/commit/rollback), LLM-assisted merge for conflicting writes, and garbage
  collection for abandoned sessions.

- **Two-tier storage** — LiteStore (zero Docker, embedded: hnsw_rs + petgraph + Redb) is the
  full product. FullStore (SurrealDB + vLLM) is an opt-in power-user upgrade. Both implement
  the same kernel traits.

- **Git-trackable knowledge files** — All knowledge is stored as JSON files in
  `.corvia/knowledge/`, designed to be committed to Git. `corvia rebuild` reconstructs all
  indexes from these files alone — the database is a cache, the JSON files are truth.
  Knowledge files are human-readable, diffable, and auditable.

- **Three integration paths** — `corvia-kernel` as a Rust crate for in-process embedding,
  `corvia serve` for a REST API at `:8020`, or `corvia serve --mcp` for an MCP server that
  works natively with Claude and other agent frameworks.

## Philosophy

Corvia is built on a few strong opinions about how agent memory should work:

**Files are truth, databases are caches.** All knowledge lives as JSON files in
`.corvia/knowledge/`. Delete the database, run `corvia rebuild`, and everything comes back.
You can `git diff` your knowledge, review it in a PR, and understand exactly what your agents
know. The indexes (vector, graph, temporal) are derived state — they accelerate queries, but
the flat files are the source of truth.

**Structural reasoning should be deterministic.** Corvia's five core health checks — stale
knowledge, broken supersession chains, orphaned entries, dangling edges, dependency cycles —
are graph algorithms and set operations, not LLM calls. Same input, same findings, every time.
You can run `corvia reason` in CI and get reproducible results. LLM-powered checks
(semantic gap detection, contradiction finding) layer on top as an opt-in complement.

**Agent memory is infrastructure, not a service.** Corvia runs in your process or on your
machine. `cargo build && corvia init` and you're running. The embedding engine (Ollama) runs
locally — no API keys, no cloud account required for core operations.

**Concurrent agents shouldn't corrupt each other.** Each agent writes to its own staging area.
Commits go through a merge pipeline with crash recovery. If two agents write conflicting
knowledge, an LLM-assisted merge resolves the conflict — or the human does. No silent
last-write-wins.

Corvia doesn't compete with AI agents or their built-in memory. It fills the gap between
personal session notes and organizational knowledge. The pitch isn't "use Corvia instead of
CLAUDE.md." It's "use Corvia *with* any agent and your team's knowledge compounds instead of
scattering across flat files."

## Quick Start

### Prerequisites

- **Rust 1.85+** (2024 edition)
- **Ollama** with `nomic-embed-text` model:
  ```bash
  # Install Ollama: https://ollama.com/download
  ollama pull nomic-embed-text
  ```

### Build from source

```bash
git clone https://github.com/corvia/corvia.git
cd corvia
cargo build --release
```

### Initialize and ingest

```bash
# Initialize Corvia in the current directory (creates .corvia/)
corvia init

# Ingest a repository — tree-sitter parses code into semantic chunks,
# Ollama generates embeddings, and everything lands in .corvia/knowledge/
corvia ingest /path/to/your/repo

# Semantic search across all ingested knowledge
corvia search "how does authentication work?"

# Run all five reasoning checks against the knowledge store
corvia reason

# Follow the supersession history of a specific entry
corvia history <entry-id>

# See what knowledge changed in the last 7 days
corvia evolution --scope default --since 7d

# Explore graph relationships for an entry
corvia graph <entry-id>

# Create a relationship between two entries
corvia relate <from-id> depends_on <to-id>

# Start the REST API server on :8020
corvia serve

# Start with MCP endpoint enabled (POST /mcp for agent frameworks)
corvia serve --mcp
```

### REST API

Once the server is running, any language can integrate:

```bash
# Semantic search
curl -X POST http://localhost:8020/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "error handling", "scope_id": "default", "limit": 5}'

# Write a knowledge entry
curl -X POST http://localhost:8020/v1/memories/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Auth uses JWT tokens with 24h expiry", "scope_id": "default"}'

# Run reasoning checks
curl -X POST http://localhost:8020/v1/reason \
  -H "Content-Type: application/json" \
  -d '{"scope_id": "default"}'
```

### MCP integration

For AI agents that support the Model Context Protocol (Claude, etc.), Corvia exposes six tools
through a JSON-RPC 2.0 endpoint at `POST /mcp`:

- `corvia_search` — Semantic similarity search
- `corvia_write` — Write a knowledge entry (agent-authenticated)
- `corvia_history` — Supersession chain for an entry
- `corvia_graph` — Graph edges for an entry
- `corvia_reason` — Run health checks on a scope
- `corvia_agent_status` — Agent session and contribution summary

### Self-demo

Corvia can ingest and reason about its own source code:

```bash
corvia demo
```

This runs the full pipeline — ingest, embed, search, and reason — against Corvia's own
codebase, so you can see results immediately without pointing it at another repository.

## Architecture

```
Integration Surface:  REST API  ·  MCP Server  ·  Rust crate
                            │
Frontends:           CLI  ·  VS Code Extension (planned)
                            │
Adapters:            Git/Code (tree-sitter)  ·  Community adapters
                            │
Kernel:              Knowledge Store  ·  Agent Coordinator
                     Embedding Pipeline  ·  Context Builder
                     Merge Worker  ·  Reasoner  ·  Graph Store
                            │
Storage:             LiteStore ─── hnsw_rs · petgraph · Redb · Git
                     FullStore ─── SurrealDB · vLLM · Redb · Git
```

### Workspace crates

| Crate | Path | Description |
|-------|------|-------------|
| `corvia-common` | `crates/corvia-common` | Shared types, config, errors, namespace, events |
| `corvia-kernel` | `crates/corvia-kernel` | All kernel subsystems — storage, coordination, reasoning |
| `corvia-server` | `crates/corvia-server` | Axum HTTP server — REST endpoints + MCP protocol |
| `corvia` (CLI) | `crates/corvia-cli` | CLI binary and workspace management |
| `corvia-adapter-git` | External crate | Git + tree-sitter code ingestion adapter |

**Design principles:**

- **Local-first** — works with zero cost and no API keys. Ollama provides local embeddings.
- **Provider-agnostic** — any LLM, any embedding model. All inference goes through traits.
- **Domain-agnostic kernel** — adapters bring domain knowledge. The Git adapter (tree-sitter
  parsing for Rust, JavaScript, TypeScript, Python) ships first; the kernel doesn't assume code.
- **Trait-bounded everything** — `QueryableStore`, `InferenceEngine`, `TemporalStore`,
  `GraphStore`, and `IngestionAdapter` are all swappable at compile time.

For a detailed walkthrough of kernel subsystems, storage tiers, core traits, and API endpoints,
see [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap

- [x] **M1** — Core kernel + two-tier storage (LiteStore + SurrealStore) + embedding pipeline
- [x] **M2** — Multi-agent coordination, staging, merge worker, crash recovery
- [x] **M3** — Temporal queries, knowledge graph, automated reasoning *(you are here)*
- [ ] **M3.1** — gRPC inference server (ONNX Runtime + candle, replaces Ollama as default)
- [ ] **M3.2** — RAG pipeline (Retriever → Augmenter → GenerationEngine)
- [ ] **M3.3** — Embedding and chunking strategies (AST, Markdown, Config, PDF)
- [ ] **M4** — Observability (OpenTelemetry spans, pipeline tracing)
- [ ] **M5** — VS Code extension + Python SDK (`pip install corvia`)
- [ ] **M6** — Eval framework (precision@k, MRR, NDCG, merge quality)
- [ ] **M7** — OSS launch + PyPI publish + 1.0

## Contributing

Corvia is pre-release software built by a solo developer. The architecture is intentionally
designed for community contribution — the adapter system, trait-based storage, and modular
kernel subsystems all provide clean extension points.

**Right now, the most valuable contributions are:**

- **Try it** — Ingest your codebase, run `corvia reason`, and tell me what breaks
- **Use cases** — Share how organizational memory could improve your AI workflow
- **Issues** — Bug reports and feature requests help prioritize the roadmap
- **Adapters** — The `IngestionAdapter` trait lets you bring new domains (docs, wikis, APIs)

The project is pre-1.0 and APIs may change. If you're interested in contributing code, open an
issue first so we can discuss the approach.

See the [issue tracker](https://github.com/corvia/corvia/issues) for open items.

## License

Corvia is licensed under [AGPL-3.0-only](LICENSE). This ensures the project stays open source
while providing SaaS protection. A dual-licensing path is planned for commercial use.

The name *Corvia* comes from *Corvus* — the genus of ravens, known for their intelligence,
tool use, and ability to share knowledge across their social groups.
