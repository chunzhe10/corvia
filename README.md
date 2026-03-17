<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/corvia-logo-light.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/corvia-logo.png">
    <img src="docs/assets/corvia-logo.png" alt="corvia" width="280">
  </picture>
</p>

<p align="center">
  <strong>Organizational memory for AI agents.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/built%20with-Rust-dea584.svg" alt="Built with Rust"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/version-0.4.3-green.svg" alt="Version 0.4.3"></a>
</p>

> **Pre-release (v0.4.3).** Core kernel, two-tier storage (LiteStore + PostgresStore),
> multi-agent coordination, temporal queries, knowledge graph, automated reasoning, RAG
> pipeline, chunking pipeline, adapter plugin system, observability, control plane, CLI
> metrics, standalone dashboard (10 features), GPU-accelerated inference (OpenVINO/CUDA),
> and docs workflow are implemented and tested (433+ tests). API surface may change before 1.0.

---

## What is corvia?

AI agents are starting to develop memory — CLAUDE.md, Cursor rules, Copilot memory are all
early steps in this direction. These work well for individual developers in individual
sessions. corvia explores what happens when that memory becomes *organizational* — shared
across agents, tracked over time, connected as a graph.

corvia is an open-source knowledge layer (written in Rust) that gives AI agents
**organizational memory** — knowledge that spans repositories, evolves over time, and stays
consistent when multiple agents write concurrently.

```
Personal memory:        "use tabs not spaces" in this project
Organizational memory:  why the auth system was redesigned, which decisions led there,
                        what changed across three repos, and what a different agent
                        discovered about the side effects
```

## How it works

1. **Ingest** — tree-sitter parses code into semantic chunks, corvia-inference generates embeddings locally (ONNX Runtime; Ollama also supported)
2. **Store** — knowledge files land in `.corvia/knowledge/` as Git-trackable JSON (the database is a cache, these files are truth)
3. **Connect** — a knowledge graph links related entries with directed, labeled edges
4. **Query** — semantic search, temporal queries ("what did we know last week?"), graph traversal
5. **Reason** — five deterministic health checks catch stale knowledge, broken chains, orphans, dangling edges, and dependency cycles
6. **Coordinate** — each agent writes to its own staging area, conflicts go through LLM-assisted merge

## Features

| Feature | What it does |
|---------|-------------|
| **Semantic search** | Vector similarity over ingested knowledge. Local embeddings via corvia-inference (ONNX) or Ollama — no API keys |
| **Knowledge graph** | Directed edges between entries. BFS traversal, shortest path, cycle detection |
| **Temporal queries** | Bi-temporal model. Point-in-time snapshots, supersession chains, time-range evolution |
| **Automated reasoning** | 5 deterministic checks + 2 opt-in LLM checks. Same input, same findings, every time |
| **Multi-agent coordination** | Session isolation, staging, crash recovery, LLM-assisted merge. No last-write-wins |
| **Two-tier storage** | LiteStore (zero Docker, embedded) is the full product. PostgreSQL is an opt-in upgrade |
| **Git as truth** | All knowledge stored as JSON in `.corvia/knowledge/`. `corvia rebuild` reconstructs everything from files alone |
| **Observability** | Structured tracing across all kernel subsystems. Configurable exporters (stdout, file, OTLP). CLI metrics via `corvia status --metrics` |
| **MCP control plane** | 18 MCP tools across 3 safety tiers (read-only, low-risk, medium-risk). Config hot-reload, GC, index rebuild |
| **Three integration paths** | Rust crate, REST API (`:8020`), or MCP server for Claude and other agent frameworks |
| **Dashboard** | Standalone web dashboard (`:8021`) with knowledge browser, semantic clustering, activity feed, agent status, and system health |
| **GPU inference** | OpenVINO for Intel iGPU, CUDA for NVIDIA GPU. Runtime backend switching via config. CPU fallback with automatic detection |
| **Docs workflow** | Doc placement enforcement (hooks), docs health checks (Aggregator), multi-repo ownership tracking |

## Quick start

### Prerequisites

- **Rust 1.88+** (2024 edition)
- **corvia-inference** (default, recommended):
  ```bash
  cargo build -p corvia-inference --release
  corvia-inference &  # starts gRPC server on :8030
  ```
- **Ollama** (alternative):
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
corvia status --metrics                   # extended metrics (entries, agents, latency)
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

corvia exposes eighteen tools across three safety tiers through JSON-RPC 2.0 at `POST /mcp`
for AI agents that support the Model Context Protocol:

**Tier 1 — Read-only (auto-approved):**

| Tool | Description |
|------|-------------|
| `corvia_search` | Semantic similarity search |
| `corvia_write` | Write a knowledge entry (agent-authenticated) |
| `corvia_history` | Supersession chain for an entry |
| `corvia_graph` | Graph edges for an entry |
| `corvia_reason` | Run health checks on a scope |
| `corvia_agent_status` | Agent session and contribution summary |
| `corvia_context` | Retrieve assembled context (RAG retrieval only) |
| `corvia_ask` | Full RAG: question to AI-generated answer from knowledge |
| `corvia_system_status` | Entry counts, agents, sessions, merge queue depth |
| `corvia_config_get` | Read any config section as JSON |
| `corvia_adapters_list` | Discovered adapter binaries |
| `corvia_agents_list` | All registered agents with status |

**Tier 2 — Low-risk mutation (single confirmation):**

| Tool | Description |
|------|-------------|
| `corvia_config_set` | Update a hot-reloadable config value |
| `corvia_gc_run` | Trigger garbage collection |
| `corvia_rebuild_index` | Rebuild HNSW vector index from files |

**Tier 3 — Medium-risk (confirmation + dry-run):**

| Tool | Description |
|------|-------------|
| `corvia_agent_suspend` | Suspend an agent and close its sessions |
| `corvia_merge_retry` | Retry failed merge queue entries |
| `corvia_merge_queue` | Inspect merge queue status |

## Architecture

```
Integration Surface:  REST API  ·  MCP Server  ·  Rust crate
                            │
Frontends:           CLI  ·  VS Code Extension (planned)
                            │
Adapters:            Git/Code (tree-sitter)  ·  Basic (filesystem)  ·  Community adapters
                            │
Inference:           gRPC inference server (ONNX/candle)  ·  Ollama
                            │
Kernel:              Knowledge Store  ·  Agent Coordinator  ·  RAG Pipeline
                     Embedding Pipeline  ·  Context Builder  ·  Chunking Pipeline
                     Merge Worker  ·  Reasoner  ·  Graph Store
                            │
Telemetry:           corvia-telemetry (structured tracing, span contracts)
                            │
Storage:             LiteStore ─── hnsw_rs · petgraph · Redb · Git
                     PostgresStore ─── pgvector · PostgreSQL
```

### Workspace crates

| Crate | Path | What it does |
|-------|------|-------------|
| `corvia-common` | `crates/corvia-common` | Shared types, config, errors, namespace, events |
| `corvia-kernel` | `crates/corvia-kernel` | Storage, coordination, reasoning, graph, temporal, RAG, chunking |
| `corvia-server` | `crates/corvia-server` | Axum HTTP server — REST + MCP protocol |
| `corvia` (CLI) | `crates/corvia-cli` | CLI binary and workspace management |
| `corvia-inference` | `crates/corvia-inference` | gRPC inference server (ONNX Runtime) |
| `corvia-proto` | `crates/corvia-proto` | Protocol Buffers for gRPC inference |
| `corvia-adapter-git` | `adapters/corvia-adapter-git/rust` | Git + tree-sitter code ingestion adapter |
| `corvia-telemetry` | `crates/corvia-telemetry` | Structured tracing initialization, span name contracts |
| `corvia-adapter-basic` | `adapters/corvia-adapter-basic/rust` | Basic filesystem ingestion adapter |

### Design principles

- **Local-first** — zero cost, no API keys. corvia-inference (ONNX) or Ollama provides local embeddings
- **Provider-agnostic** — any LLM, any embedding model. All inference goes through traits
- **Domain-agnostic kernel** — adapters bring domain knowledge. The kernel doesn't assume code
- **Trait-bounded** — `QueryableStore`, `InferenceEngine`, `TemporalStore`, `GraphStore`, `IngestionAdapter` are all swappable

For detailed internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap

- [x] **M1** — Core kernel + three-tier storage + embedding pipeline
- [x] **M2** — Multi-agent coordination, staging, merge worker, crash recovery
- [x] **M3** — Temporal queries, knowledge graph, automated reasoning
- [x] **M3.1** — gRPC inference server (ONNX Runtime + candle)
- [x] **M3.2** — RAG pipeline (Retriever → Augmenter → GenerationEngine)
- [x] **M3.3** — Chunking strategies (AST, Markdown, Config, PDF)
- [x] **M3.4** — Graph edge improvements + cross-file relation discovery
- [x] **M4** — Observability + control plane (structured tracing, 18 MCP tools, CLI metrics)
- [x] **M4.2** — Standalone dashboard (semantic clustering, activity feed, agent identity, knowledge browser)
- [x] **M4.3** — GPU-accelerated inference (OpenVINO/CUDA), docs workflow enforcement
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
