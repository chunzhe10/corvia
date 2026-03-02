<p align="center">
  <img src="docs/assets/corvia-logo.png" alt="corvia" width="280">
</p>

<p align="center">
  <strong>Organizational memory for AI agents.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/built%20with-Rust-dea584.svg" alt="Built with Rust"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/version-0.3.0-green.svg" alt="Version 0.3.0"></a>
</p>

> **Pre-release (v0.3.0).** Core kernel, two-tier storage, multi-agent coordination, temporal
> queries, knowledge graph, and automated reasoning are implemented and tested (203 tests).
> API surface may change before 1.0.

---

## What is corvia?

AI agents are getting memory — Claude has CLAUDE.md, Cursor has rules files, Copilot has
memory. But these are *personal notes*: flat files, one repo, no relationships, no history.
They help one developer in one session. They don't help your *organization* learn.

corvia is an open-source knowledge layer that gives AI agents **organizational memory** —
knowledge that spans repositories, tracks relationships as a graph, evolves over time, and
stays consistent when multiple agents write concurrently.

```
Personal memory:        "use tabs not spaces" in this project
Organizational memory:  why the auth system was redesigned, which decisions led there,
                        what changed across three repos, and what a different agent
                        discovered about the side effects
```

## How it works

1. **Ingest** — tree-sitter parses code into semantic chunks, Ollama generates embeddings locally
2. **Store** — knowledge files land in `.corvia/knowledge/` as Git-trackable JSON (the database is a cache, these files are truth)
3. **Connect** — a knowledge graph links related entries with directed, labeled edges
4. **Query** — semantic search, temporal queries ("what did we know last week?"), graph traversal
5. **Reason** — five deterministic health checks catch stale knowledge, broken chains, orphans, dangling edges, and dependency cycles
6. **Coordinate** — each agent writes to its own staging area, conflicts go through LLM-assisted merge

## Features

| Feature | What it does |
|---------|-------------|
| **Semantic search** | Vector similarity over ingested knowledge. Local embeddings via Ollama — no API keys |
| **Knowledge graph** | Directed edges between entries. BFS traversal, shortest path, cycle detection |
| **Temporal queries** | Bi-temporal model. Point-in-time snapshots, supersession chains, time-range evolution |
| **Automated reasoning** | 5 deterministic checks + 2 opt-in LLM checks. Same input, same findings, every time |
| **Multi-agent coordination** | Session isolation, staging, crash recovery, LLM-assisted merge. No last-write-wins |
| **Two-tier storage** | LiteStore (zero Docker, embedded) is the full product. SurrealDB is an opt-in upgrade |
| **Git as truth** | All knowledge stored as JSON in `.corvia/knowledge/`. `corvia rebuild` reconstructs everything from files alone |
| **Three integration paths** | Rust crate, REST API (`:8020`), or MCP server for Claude and other agent frameworks |

## Quick start

### Prerequisites

- **Rust 1.85+** (2024 edition)
- **Ollama** with `nomic-embed-text`:
  ```bash
  # https://ollama.com/download
  ollama pull nomic-embed-text
  ```

### Build and run

```bash
git clone https://github.com/corvia/corvia.git
cd corvia
cargo build --release

corvia init                              # creates .corvia/
corvia ingest /path/to/your/repo         # parse + embed + store
corvia search "how does authentication work?"
corvia reason                            # run all health checks
```

### More commands

```bash
corvia history <entry-id>                # supersession chain
corvia evolution --scope default --since 7d  # what changed recently
corvia graph <entry-id>                  # graph relationships
corvia relate <from-id> depends_on <to-id>   # create an edge
corvia serve                             # REST API on :8020
corvia serve --mcp                       # + MCP endpoint for agents
corvia demo                              # ingest corvia's own source code
```

### REST API

```bash
# search
curl -X POST http://localhost:8020/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "error handling", "scope_id": "default", "limit": 5}'

# write
curl -X POST http://localhost:8020/v1/memories/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Auth uses JWT tokens with 24h expiry", "scope_id": "default"}'

# reason
curl -X POST http://localhost:8020/v1/reason \
  -H "Content-Type: application/json" \
  -d '{"scope_id": "default"}'
```

### MCP integration

corvia exposes six tools through JSON-RPC 2.0 at `POST /mcp` for AI agents that support
the Model Context Protocol:

| Tool | Description |
|------|-------------|
| `corvia_search` | Semantic similarity search |
| `corvia_write` | Write a knowledge entry (agent-authenticated) |
| `corvia_history` | Supersession chain for an entry |
| `corvia_graph` | Graph edges for an entry |
| `corvia_reason` | Run health checks on a scope |
| `corvia_agent_status` | Agent session and contribution summary |

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

| Crate | Path | What it does |
|-------|------|-------------|
| `corvia-common` | `crates/corvia-common` | Shared types, config, errors, namespace, events |
| `corvia-kernel` | `crates/corvia-kernel` | Storage, coordination, reasoning, graph, temporal |
| `corvia-server` | `crates/corvia-server` | Axum HTTP server — REST + MCP protocol |
| `corvia` (CLI) | `crates/corvia-cli` | CLI binary and workspace management |
| `corvia-adapter-git` | External crate | Git + tree-sitter code ingestion adapter |

### Design principles

- **Local-first** — zero cost, no API keys. Ollama provides local embeddings
- **Provider-agnostic** — any LLM, any embedding model. All inference goes through traits
- **Domain-agnostic kernel** — adapters bring domain knowledge. The kernel doesn't assume code
- **Trait-bounded** — `QueryableStore`, `InferenceEngine`, `TemporalStore`, `GraphStore`, `IngestionAdapter` are all swappable

For detailed internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap

- [x] **M1** — Core kernel + two-tier storage + embedding pipeline
- [x] **M2** — Multi-agent coordination, staging, merge worker, crash recovery
- [x] **M3** — Temporal queries, knowledge graph, automated reasoning *(current)*
- [ ] **M3.1** — gRPC inference server (ONNX Runtime + candle)
- [ ] **M3.2** — RAG pipeline (Retriever → Augmenter → GenerationEngine)
- [ ] **M3.3** — Chunking strategies (AST, Markdown, Config, PDF)
- [ ] **M4** — Observability (OpenTelemetry)
- [ ] **M5** — VS Code extension + Python SDK
- [ ] **M6** — Eval framework (precision@k, MRR, NDCG)
- [ ] **M7** — 1.0 + PyPI publish

## Contributing

corvia is pre-release software built by a solo developer. The adapter system, trait-based
storage, and modular kernel all provide clean extension points.

**Most valuable right now:**

- **Try it** — ingest your codebase, run `corvia reason`, tell me what breaks
- **Use cases** — how could organizational memory improve your AI workflow?
- **Issues** — bug reports and feature requests help prioritize the roadmap
- **Adapters** — the `IngestionAdapter` trait lets you bring new domains (docs, wikis, APIs)

Pre-1.0 APIs may change. Open an issue before contributing code so we can discuss the approach.

## License

[AGPL-3.0-only](LICENSE). SaaS protection with a dual-licensing path planned for commercial use.

The name *corvia* comes from *Corvus* — the genus of ravens, known for their intelligence,
tool use, and ability to share knowledge across their social groups.
