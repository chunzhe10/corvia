# Architecture

This document describes the internal architecture of corvia — its layered design, crate
structure, kernel subsystems, storage implementations, core traits, API surface, and key
design decisions. It is intended for contributors and anyone who wants to understand how
the system is built.

For a user-facing overview and quick start guide, see [README.md](README.md).

## Layered Architecture

corvia follows a strict layered dependency model. Each layer may only depend on the layer
directly below it. This prevents circular dependencies and ensures the kernel remains
independent of any specific frontend, adapter, or integration surface.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Integration Surface                                                │
│  REST API (:8020)  ·  MCP Server (POST /mcp)  ·  Rust crate (lib) │
├─────────────────────────────────────────────────────────────────────┤
│  Frontends                                                          │
│  CLI (corvia binary)  ·  VS Code Extension (planned, M5)           │
├─────────────────────────────────────────────────────────────────────┤
│  Adapters                                                           │
│  Git/Code (tree-sitter) · Basic (filesystem) · Community adapters  │
├─────────────────────────────────────────────────────────────────────┤
│  Kernel                                                             │
│  Knowledge Store  ·  Agent Coordinator  ·  RAG Pipeline            │
│  Embedding Pipeline · Chunking Pipeline · Context Builder          │
│  Merge Worker  ·  Reasoner  ·  Graph Store                         │
├─────────────────────────────────────────────────────────────────────┤
│  Telemetry                                                          │
│  corvia-telemetry (tracing init, span contracts, exporters)        │
├─────────────────────────────────────────────────────────────────────┤
│  Storage                                                            │
│  LiteStore ─── hnsw_rs · petgraph · Redb · Git (JSON files)       │
│  PostgresStore ─── pgvector · PostgreSQL                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Dependency rules:**

- The kernel never imports from frontends, adapters, or the integration surface.
- Adapters depend on `corvia-kernel` for the `IngestionAdapter` trait and `corvia-common` for shared types.
- Storage implementations live inside the kernel crate and implement kernel traits.
- `corvia-common` is the shared foundation — types, config, errors — imported by all layers.

## Crate Map

| Crate | Path | Role |
|-------|------|------|
| `corvia-common` | `crates/corvia-common` | Shared types (`KnowledgeEntry`, `GraphEdge`, `EntryMetadata`), configuration (`CorviaConfig`), error types, namespace parsing, event types, agent identity types |
| `corvia-kernel` | `crates/corvia-kernel` | All kernel subsystems: both storage implementations (LiteStore, PostgresStore), embedding engines, agent coordination, graph store, temporal queries, reasoning, merge worker, context builder |
| `corvia-server` | `crates/corvia-server` | Axum-based HTTP server exposing REST endpoints under `/v1/` and the MCP JSON-RPC 2.0 endpoint at `/mcp`. Thin orchestration layer that wires kernel types together and handles request/response translation |
| `corvia` (CLI) | `crates/corvia-cli` | Binary crate. CLI commands (`init`, `ingest`, `search`, `serve`, `reason`, etc.) and workspace management. Orchestrates kernel + server |
| `corvia-inference` | `crates/corvia-inference` | gRPC inference server with ONNX Runtime for local embeddings |
| `corvia-proto` | `crates/corvia-proto` | Protocol Buffers for gRPC inference |
| `corvia-adapter-git` | `adapters/corvia-adapter-git/rust` | Git repository ingestion using tree-sitter for code-aware chunking. Supports Rust, JavaScript, TypeScript, and Python. Implements the `IngestionAdapter` trait |
| `corvia-adapter-basic` | `adapters/corvia-adapter-basic/rust` | Basic filesystem ingestion adapter for non-Git sources |
| `corvia-telemetry` | `crates/corvia-telemetry` | Telemetry initialization (`init_telemetry()`), D45 span name constants (`spans::*`), configurable exporters (stdout, file, OTLP), `TelemetryGuard` for log flushing |

All workspace crates share `version.workspace = true` from the root `Cargo.toml`.

## Kernel Subsystems

The kernel contains eleven subsystems, each in its own module within `corvia-kernel`:

| Subsystem | Module | Description |
|-----------|--------|-------------|
| **Knowledge Store** | `lite_store.rs`, `postgres_store.rs` | Read/write knowledge entries with vector search. Two implementations: `LiteStore` (hnsw_rs + Redb, embedded) and `PostgresStore` (pgvector). Both implement `QueryableStore` |
| **Agent Coordinator** | `agent_coordinator.rs`, `agent_registry.rs`, `session_manager.rs`, `staging.rs`, `agent_writer.rs` | Multi-agent lifecycle management. Agent registration with multi-layer identity (three tiers: Registered, MCP Client, Anonymous — chosen over single-tier to support both long-lived agents and one-shot queries without forcing registration). Session state machine (Created → Active → Committing → Merging → Closed), staging isolation, crash recovery (resume/commit/rollback), garbage collection |
| **Embedding Pipeline** | `embedding_pipeline.rs`, `ollama_engine.rs`, `grpc_engine.rs`, `ollama_provisioner.rs` | Provider-agnostic embedding generation. `OllamaEngine` for Ollama (HTTP), `GrpcEngine` for the corvia-inference gRPC server. Both implement `InferenceEngine` |
| **RAG Pipeline** | `rag_pipeline.rs`, `retriever.rs`, `augmenter.rs`, `rag_types.rs` | Retrieval-augmented generation: Retriever → Augmenter → GenerationEngine. Graph-expanded retrieval with configurable oversample and scoring |
| **Chunking Pipeline** | `chunking_pipeline.rs`, `chunking_strategy.rs`, `chunking_markdown.rs`, `chunking_config_fmt.rs`, `chunking_pdf.rs`, `chunking_fallback.rs` | Format-aware chunking with FormatRegistry routing. Merge small chunks, split oversized ones, add overlap context |
| **Context Builder** | `context_builder.rs` | Scope-aware knowledge retrieval. Enforces agent read isolation — agents only see entries matching their authorized scopes and entry visibility (Pending/Committed/Merged) |
| **Merge Worker** | `merge_worker.rs`, `merge_queue.rs`, `commit_pipeline.rs` | LLM-assisted conflict resolution — chosen over last-write-wins or manual merge because knowledge conflicts are semantic, not textual |
| **Reasoner** | `reasoner.rs` | Algorithmic knowledge health checks. Seven check types: `StaleEntry`, `BrokenChain`, `OrphanedNode`, `DanglingImport`, `DependencyCycle` (deterministic, no LLM), plus `SemanticGap` and `Contradiction` (LLM-powered, opt-in) |
| **Graph Store** | `graph_store.rs` | Directed knowledge graph. `LiteGraphStore` uses petgraph (in-memory DiGraph) backed by Redb persistence. Supports `relate`, `edges`, `traverse` (BFS), and `shortest_path`. Cross-file relation discovery with graph-expanded scoring |
| **Adapter System** | `adapter_discovery.rs`, `adapter_protocol.rs`, `process_adapter.rs` | Runtime adapter discovery, JSONL IPC protocol for adapter processes, and ProcessChunkingStrategy for adapter-provided chunking |
| **Shared Operations** | `ops.rs` | Shared kernel operations callable from CLI and MCP. System status, agent/session listing, merge queue inspection, config get/set with hot-reload, GC, index rebuild |

### Session State Machine

Agent sessions follow a defined lifecycle with recovery paths for crash resilience:

```
Created ──► Active ──► Committing ──► Merging ──► Closed
               │            │             │
               ▼            ▼             ▼
             Stale      Orphaned      Orphaned
               │            │             │
               └─── recover (resume | commit | rollback) ───┘
```

- **Created → Active**: First heartbeat or write transitions the session.
- **Active → Stale**: No heartbeat within the timeout window.
- **Committing/Merging → Orphaned**: Process crashed mid-operation.
- **Recovery**: Stale and orphaned sessions can be resumed, force-committed, or rolled back.

## Storage Tiers

corvia has two storage backends that implement the same kernel traits. The **two-tier storage**
design exists because requiring Docker for a dev tool kills adoption — but PostgreSQL is the
standard in production environments. Both tiers implement the same traits: LiteStore is the full
product with zero external dependencies, while PostgresStore is an opt-in upgrade for users
who need production-grade PostgreSQL capabilities.

### LiteStore (default, zero Docker)

| Component | Library | Purpose |
|-----------|---------|---------|
| Vector search | `hnsw_rs` (Rust, in-memory) | Approximate nearest neighbor search. Cosine distance. Tuned for up to ~100K entries. Rebuilt from knowledge JSON files on startup |
| Knowledge graph | `petgraph` (in-memory DiGraph) | BFS/DFS traversal, shortest path, cycle detection. Rebuilt from Redb on startup |
| Metadata + temporal | `Redb` (embedded B-tree) | Entry storage, scope indexes, HNSW ID mapping, temporal compound-key range scans, graph edge persistence, agent coordination (sessions, merge queue) |
| Persistence | Git-tracked JSON files | `.corvia/knowledge/{scope}/{uuid}.json` — source of truth. `corvia rebuild` reconstructs all indexes from these files |
| Embeddings | Ollama (HTTP, local) | `nomic-embed-text` model (768 dimensions). Auto-provisioned on first use |

**Both tiers share:** Git-trackable JSON knowledge files as the rebuildable source of truth, Redb for coordination, and the same kernel traits
(`QueryableStore`, `TemporalStore`, `GraphStore`). Switching tiers is a configuration change
(`corvia init --store lite|postgres`), not a code change.

## Core Traits

All kernel subsystems are built around five async traits defined in
`crates/corvia-kernel/src/traits.rs`. Each trait uses `async_trait` and requires `Send + Sync`.

### `QueryableStore`

The primary storage interface. Handles entry insertion (with embeddings), semantic search,
retrieval by ID, scope counting, schema initialization, and scope deletion.

```rust
pub trait QueryableStore: Send + Sync {
    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()>;
    async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<Vec<SearchResult>>;
    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>>;
    async fn count(&self, scope_id: &str) -> Result<u64>;
    async fn init_schema(&self) -> Result<()>;
    async fn delete_scope(&self, scope_id: &str) -> Result<()>;
    fn as_any(&self) -> &dyn Any;
}
```

The `as_any()` method enables downcasting for store-specific operations (e.g., `LiteStore::rebuild_from_files`).

**Implementations:** `LiteStore`, `PostgresStore`.

### `InferenceEngine`

Provider-agnostic embedding generation. Supports single and batch embedding, and reports
the model's output dimensionality.

```rust
pub trait InferenceEngine: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}
```

**Implementations:** `OllamaEngine` (Ollama HTTP API), `VllmEngine` (vLLM HTTP API).

### `IngestionAdapter`

Domain-specific source ingestion. An adapter knows how to read a particular source type
(Git repositories, documentation, wikis) and produce `KnowledgeEntry` values. The kernel
adds embeddings after ingestion.

```rust
pub trait IngestionAdapter: Send + Sync {
    fn domain(&self) -> &str;
    fn register_chunking(&self, registry: &mut FormatRegistry);
    async fn ingest_sources(&self, source_path: &str) -> Result<Vec<SourceFile>>;
}
```

**Implementations:** `GitAdapter` (tree-sitter parsing for Rust, JS, TS, Python), `ProcessAdapter` (IPC to external adapter binaries).

### `TemporalStore`

Bi-temporal query interface. Redb compound keys `(scope, valid_from, entry_id)` give O(log n)
temporal lookups via B-tree range scans — chosen over a dedicated time-series database because
Redb is already embedded for coordination and temporal queries don't need a separate service.
Supports point-in-time snapshots, supersession chain traversal, and time-range evolution queries.

```rust
pub trait TemporalStore: Send + Sync {
    async fn as_of(&self, scope_id: &str, timestamp: DateTime<Utc>, limit: usize) -> Result<Vec<KnowledgeEntry>>;
    async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>>;
    async fn evolution(&self, scope_id: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<KnowledgeEntry>>;
}
```

**Implementations:** `LiteStore` (Redb compound-key range scans, O(log n)).

### `GraphStore`

Directed knowledge graph interface. petgraph was chosen so that LiteStore gets full traversal
capabilities (BFS, shortest path, cycle detection) without any external service. Supports edge
creation, edge queries by direction, BFS traversal with optional relation filtering, and
shortest path computation.

```rust
pub trait GraphStore: Send + Sync {
    async fn relate(&self, from: &Uuid, relation: &str, to: &Uuid, metadata: Option<Value>) -> Result<()>;
    async fn edges(&self, entry_id: &Uuid, direction: EdgeDirection) -> Result<Vec<GraphEdge>>;
    async fn traverse(&self, start: &Uuid, relation: Option<&str>, direction: EdgeDirection, max_depth: usize) -> Result<Vec<KnowledgeEntry>>;
    async fn shortest_path(&self, from: &Uuid, to: &Uuid) -> Result<Option<Vec<KnowledgeEntry>>>;
    async fn remove_edges(&self, entry_id: &Uuid) -> Result<()>;
}
```

**Implementations:** `LiteGraphStore` (petgraph DiGraph + Redb persistence).

## API Surface

### REST Endpoints

All endpoints are served by `corvia-server` via Axum at `http://localhost:8020`.

**Memory**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/memories/write` | Embed and store a knowledge entry |
| `POST` | `/v1/memories/search` | Semantic vector search within a scope |

**Agent Coordination**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/agents` | Register a new agent identity |
| `POST` | `/v1/agents/{agent_id}/sessions` | Create a new agent session |
| `POST` | `/v1/sessions/{session_id}/heartbeat` | Session keepalive heartbeat |
| `POST` | `/v1/sessions/{session_id}/write` | Write a knowledge entry via agent session (staged) |
| `POST` | `/v1/sessions/{session_id}/commit` | Commit all staged entries to main store |
| `POST` | `/v1/sessions/{session_id}/rollback` | Rollback all staged entries |
| `POST` | `/v1/sessions/{session_id}/recover` | Recover a stale/orphaned session (resume, commit, or rollback) |
| `GET`  | `/v1/sessions/{session_id}/state` | Get session state and entry counters |

**Temporal, Graph, and Reasoning**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/entries/{id}/history` | Bi-temporal supersession chain for an entry |
| `GET`  | `/v1/entries/{id}/edges` | Graph edges for an entry (optional `?relation=` filter) |
| `GET`  | `/v1/evolution` | Entries that changed in a scope within a time window (`?scope=...&since=7d`) |
| `POST` | `/v1/edges` | Create a directed graph edge between two entries |
| `POST` | `/v1/reason` | Run reasoning checks on a scope (all checks or a specific check) |

**Utility**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check — returns `{"status": "ok"}` |

### MCP Tools

Corvia implements the Model Context Protocol (version `2024-11-05`) via JSON-RPC 2.0 at
`POST /mcp`. Enabled with `corvia serve --mcp`. Supports `initialize`, `tools/list`, and
`tools/call` methods.

Tools use a three-tier safety model. Tier 1 tools execute immediately. Tier 2 tools require
`_meta.confirmed: true`. Tier 3 tools require confirmation and support `dry_run` mode.

**Tier 1 — Read-only (auto-approved):**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `corvia_search` | `query`, `scope_id`, `limit?` | Semantic similarity search |
| `corvia_write` | `content`, `scope_id`, `agent_id?`, `source_version?` | Write a knowledge entry |
| `corvia_history` | `entry_id` | Supersession history chain |
| `corvia_graph` | `entry_id` | Graph edges (both directions) |
| `corvia_reason` | `scope_id`, `check?` | Reasoning health checks |
| `corvia_agent_status` | `agent_id?` | Agent contribution summary |
| `corvia_context` | `query`, `scope_id`, `limit?` | RAG context retrieval (no generation) |
| `corvia_ask` | `query`, `scope_id`, `limit?` | Full RAG with AI-generated answer |
| `corvia_system_status` | `scope_id` | Entry counts, agents, sessions, merge queue |
| `corvia_config_get` | `section?` | Read config section as JSON |
| `corvia_adapters_list` | *(none)* | Discovered adapter binaries |
| `corvia_agents_list` | *(none)* | All registered agents |

**Tier 2 — Low-risk mutation (single confirmation):**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `corvia_config_set` | `section`, `key`, `value` | Update hot-reloadable config value |
| `corvia_gc_run` | *(none)* | Trigger garbage collection |
| `corvia_rebuild_index` | *(none)* | Rebuild HNSW vector index |

**Tier 3 — Medium-risk (confirmation + dry-run):**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `corvia_agent_suspend` | `agent_id` | Suspend agent, close sessions |
| `corvia_merge_retry` | `entry_ids` | Retry failed merge entries |
| `corvia_merge_queue` | *(none)* | Inspect merge queue |

## Key Design Decisions

corvia's architecture is shaped by documented design decisions made during development. The
most architecturally significant ones are listed here with the alternative that was considered
and the reasoning behind the choice. The full decision log with alternatives and rationale is
available at [`docs/rfcs/`](docs/rfcs/).

| Decision | Alternative rejected | Why this choice wins |
|----------|---------------------|---------------------|
| **AGPL-3.0-only license** | MIT/Apache-2.0 | SaaS protection — prevents cloud providers from offering corvia-as-a-service without contributing back. Dual-license path preserves commercial option (Grafana/MinIO playbook) |
| **Trait-bounded storage** | Hardcode storage directly | Trait boundary enabled building LiteStore (zero-Docker) and PostgresStore without touching kernel code. Storage became a compile-time choice, not a runtime dependency |
| **LLM-assisted merge** | Last-write-wins / manual merge | Knowledge conflicts are semantic, not textual — an LLM can reason about which version of "how auth works" is more accurate. Last-write-wins silently loses knowledge |
| **Two-tier storage (LiteStore + PostgresStore)** | Single backend | Requiring Docker for a dev tool kills adoption. LiteStore is the full product at zero cost; PostgresStore is an opt-in upgrade for production environments |
| **petgraph + Redb for graph** | External graph database | LiteStore needed full traversal (BFS, shortest path, cycles) without any external service. petgraph handles computation in-memory, Redb handles persistence |
| **Redb compound-key range scans for temporal** | Dedicated time-series DB | Redb is already embedded for coordination. Compound keys `(scope, valid_from, entry_id)` give O(log n) temporal lookups without adding another dependency |
| **Git-trackable JSON knowledge files** | Database-only storage | Storing knowledge as JSON files designed for Git gives auditability and diffability. `corvia rebuild` reconstructs all indexes from JSON files alone — the database is a cache, the files are truth |
| **Staging hybrid (branches + shared HNSW)** | Full git branch isolation | Pure branch isolation prevents cross-agent search during staging. Shared HNSW lets agents see each other's work-in-progress while writes stay isolated until commit |
| **Multi-layer agent identity** | Single registration model | Forcing registration blocks MCP clients and anonymous queries. Three tiers (Registered, MCP Client, Anonymous) let any agent framework integrate with appropriate write permissions |
| **Python SDK via PyO3 (planned)** | HTTP-only Python client | Embedding the Rust kernel in Python (like Polars) gives zero-latency access to all kernel operations — not just what the REST API exposes |
| **Three-tier MCP safety model** | Single confirmation for all mutations | Graduated risk — read-only tools execute freely, low-risk mutations need one confirmation, medium-risk mutations support dry-run preview. Prevents accidental destructive operations without adding friction to safe queries |

## Data Model

### KnowledgeEntry

The fundamental unit of storage. Every piece of knowledge in corvia is a `KnowledgeEntry`.

```
KnowledgeEntry
├── id: Uuid (v7, time-ordered)
├── content: String (the actual knowledge text)
├── source_version: String (git commit hash or version ref)
├── scope_id: String (organizational scope, e.g., "myproject")
├── workstream: String (default: "main", reserved for future branch isolation)
├── recorded_at: DateTime<Utc> (transaction time — when it was stored)
├── valid_from: DateTime<Utc> (valid time — when the knowledge became true)
├── valid_to: Option<DateTime<Utc>> (None = currently valid)
├── superseded_by: Option<Uuid> (points to the replacement entry)
├── embedding: Option<Vec<f32>> (768-dim for nomic-embed-text)
├── metadata: EntryMetadata
│   ├── source_file: Option<String>
│   ├── language: Option<String> (e.g., "rust", "typescript")
│   ├── chunk_type: Option<String> (e.g., "function", "struct", "module")
│   ├── start_line: Option<u32>
│   └── end_line: Option<u32>
├── agent_id: Option<String> (which agent wrote this)
├── session_id: Option<String> (which session wrote this)
└── entry_status: EntryStatus (Pending | Committed | Merged)
```

**Bi-temporal model:** Each entry has two time axes. `recorded_at` tracks when the entry was
stored in corvia (transaction time). `valid_from`/`valid_to` track when the knowledge was
true in the real world (valid time). This enables queries like "what did we know last Tuesday?"
(transaction time) and "what was the API like before the v3 migration?" (valid time).

**Supersession:** When knowledge is updated, the old entry gets `valid_to` set and
`superseded_by` pointing to the new entry. This creates a linked chain that `corvia history`
traverses. No entry is ever deleted — the full history is always available.

### GraphEdge

Relationships between knowledge entries.

```
GraphEdge
├── from: Uuid (source entry)
├── to: Uuid (target entry)
├── relation: String (e.g., "imports", "depends_on", "tests", "documents")
└── metadata: Option<Value> (arbitrary JSON, e.g., {"weight": 0.8})
```

Edges are directed and labeled. The `relation` field enables typed traversal — for example,
`corvia graph --relation imports <entry-id>` shows only import relationships. The reasoner
uses graph structure for its `DanglingImport` and `DependencyCycle` checks.

### Agent Identity

corvia supports three layers of agent identity:

| Layer | Type | Write access | Staging |
|-------|------|-------------|---------|
| **Registered** | `{scope}::{agent-name}` | Yes | Full (git branch + staging dir) |
| **MCP Client** | Client info + `_meta.agent_id` | Yes (with hint) | Lightweight |
| **Anonymous** | No identity | Read-only | None |

This multi-layer model allows corvia to work with any agent framework — from fully registered
long-lived agents to anonymous one-shot queries — while maintaining write safety through
identity-based access control.
