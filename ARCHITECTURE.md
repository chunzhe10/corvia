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
│  Git/Code (tree-sitter)  ·  Community adapters (IngestionAdapter)  │
├─────────────────────────────────────────────────────────────────────┤
│  Kernel                                                             │
│  Knowledge Store  ·  Agent Coordinator  ·  Embedding Pipeline      │
│  Context Builder  ·  Merge Worker  ·  Reasoner  ·  Graph Store     │
├─────────────────────────────────────────────────────────────────────┤
│  Storage                                                            │
│  LiteStore ─── hnsw_rs · petgraph · Redb · Git (JSON files)       │
│  FullStore ─── SurrealDB · vLLM · Redb · Git (JSON files)         │
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
| `corvia-kernel` | `crates/corvia-kernel` | All kernel subsystems: both storage implementations (LiteStore, SurrealStore), embedding engines, agent coordination, graph store, temporal queries, reasoning, merge worker, context builder |
| `corvia-server` | `crates/corvia-server` | Axum-based HTTP server exposing REST endpoints under `/v1/` and the MCP JSON-RPC 2.0 endpoint at `/mcp`. Thin orchestration layer that wires kernel types together and handles request/response translation |
| `corvia` (CLI) | `crates/corvia-cli` | Binary crate. CLI commands (`init`, `ingest`, `search`, `serve`, `reason`, etc.) and workspace management. Orchestrates kernel + server |
| `corvia-adapter-git` | External (sibling repo) | Git repository ingestion using tree-sitter for code-aware chunking. Supports Rust, JavaScript, TypeScript, and Python. Implements the `IngestionAdapter` trait |

All workspace crates share `version.workspace = true` from the root `Cargo.toml`. The adapter
is a separate repository to demonstrate that adapters are independently publishable.

## Kernel Subsystems

The kernel contains seven subsystems, each in its own module within `corvia-kernel`:

| Subsystem | Module | Description |
|-----------|--------|-------------|
| **Knowledge Store** | `knowledge_store.rs`, `lite_store.rs` | Read/write knowledge entries with vector search. Two implementations: `SurrealStore` (SurrealDB, HTTP) and `LiteStore` (hnsw_rs + Redb, embedded). Both implement `QueryableStore` |
| **Agent Coordinator** | `agent_coordinator.rs`, `agent_registry.rs`, `session_manager.rs`, `staging.rs`, `agent_writer.rs` | Multi-agent lifecycle management. Agent registration with multi-layer identity (three tiers: Registered, MCP Client, Anonymous — chosen over single-tier to support both long-lived agents and one-shot queries without forcing registration). Session state machine (Created → Active → Committing → Merging → Closed), staging isolation, crash recovery (resume/commit/rollback), garbage collection |
| **Embedding Pipeline** | `embedding_pipeline.rs`, `ollama_engine.rs`, `ollama_provisioner.rs` | Provider-agnostic embedding generation. `VllmEngine` for vLLM (HTTP), `OllamaEngine` for Ollama (HTTP). Both implement `InferenceEngine`. `OllamaProvisioner` auto-downloads models on first use |
| **Context Builder** | `context_builder.rs` | Scope-aware knowledge retrieval. Enforces agent read isolation — agents only see entries matching their authorized scopes and entry visibility (Pending/Committed/Merged). Uses application-level RBAC rather than database-level ACLs, so LiteStore gets the same isolation guarantees as SurrealDB without requiring a database with built-in permissions |
| **Merge Worker** | `merge_worker.rs`, `merge_queue.rs`, `commit_pipeline.rs` | LLM-assisted conflict resolution — chosen over last-write-wins or manual merge because knowledge conflicts are semantic, not textual, and an LLM can reason about which version is more accurate. When two agents write to overlapping scopes, the merge worker detects conflicts via embedding similarity and uses an LLM to produce a merged entry. Commit pipeline moves entries from staging to main store |
| **Reasoner** | `reasoner.rs` | Algorithmic knowledge health checks. Seven check types: `StaleEntry`, `BrokenChain`, `OrphanedNode`, `DanglingImport`, `DependencyCycle` (deterministic, no LLM), plus `SemanticGap` and `Contradiction` (LLM-powered, opt-in). Returns structured `Finding` results with confidence scores, affected entry IDs, and human-readable rationale |
| **Graph Store** | `graph_store.rs` | Directed knowledge graph. `LiteGraphStore` uses petgraph (in-memory DiGraph) backed by Redb persistence. Supports `relate`, `edges`, `traverse` (BFS), and `shortest_path`. Rebuilds in-memory graph from Redb on startup |

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
design exists because requiring Docker for a dev tool kills adoption — but SurrealDB's unified
vector+graph+temporal queries are genuinely valuable at scale. Rather than choose one, both
tiers implement the same traits: LiteStore is the full product with zero external dependencies,
while FullStore (SurrealStore) is an opt-in upgrade for users who need SurrealDB's query
capabilities or vLLM's GPU-accelerated inference.

### LiteStore (default, zero Docker)

| Component | Library | Purpose |
|-----------|---------|---------|
| Vector search | `hnsw_rs` (Rust, in-memory) | Approximate nearest neighbor search. Cosine distance. Tuned for up to ~100K entries. Rebuilt from knowledge JSON files on startup |
| Knowledge graph | `petgraph` (in-memory DiGraph) | BFS/DFS traversal, shortest path, cycle detection. Rebuilt from Redb on startup |
| Metadata + temporal | `Redb` (embedded B-tree) | Entry storage, scope indexes, HNSW ID mapping, temporal compound-key range scans, graph edge persistence, agent coordination (sessions, merge queue) |
| Persistence | Git-tracked JSON files | `.corvia/knowledge/{scope}/{uuid}.json` — source of truth. `corvia rebuild` reconstructs all indexes from these files |
| Embeddings | Ollama (HTTP, local) | `nomic-embed-text` model (768 dimensions). Auto-provisioned on first use |

### FullStore (SurrealDB + vLLM)

| Component | Service | Purpose |
|-----------|---------|---------|
| Vector search + graph + temporal | SurrealDB | Unified queryable store. Vectors via native vector search, graphs via edge records in an `edges` table, temporal via SurrealQL time-range queries |
| Coordination | `Redb` (embedded) | Same as LiteStore — agent sessions and merge queue are always local |
| Persistence | Git-tracked JSON files | Same as LiteStore — Git is always the source of truth |
| Embeddings | vLLM (HTTP, GPU) | GPU-accelerated embedding generation for higher throughput |

**Both tiers share:** Git-trackable JSON knowledge files as the rebuildable source of truth, Redb for coordination, and the same kernel traits
(`QueryableStore`, `TemporalStore`, `GraphStore`). Switching tiers is a configuration change
(`corvia init` vs `corvia init --full`), not a code change.

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
}
```

**Implementations:** `LiteStore` (hnsw_rs + Redb), `SurrealStore` (SurrealDB HTTP client).

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
    async fn ingest(&self, source_path: &str) -> Result<Vec<KnowledgeEntry>>;
}
```

**Implementations:** `GitAdapter` (tree-sitter parsing for Rust, JS, TS, Python).

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

**Implementations:** `LiteStore` (Redb compound-key range scans, O(log n)),
`SurrealStore` (SurrealQL time-range queries).

### `GraphStore`

Directed knowledge graph interface. petgraph was chosen over requiring SurrealDB for graph
queries so that LiteStore gets full traversal capabilities (BFS, shortest path, cycle detection)
without any external service. Supports edge creation, edge queries by direction, BFS traversal
with optional relation filtering, and shortest path computation.

```rust
pub trait GraphStore: Send + Sync {
    async fn relate(&self, from: &Uuid, relation: &str, to: &Uuid, metadata: Option<Value>) -> Result<()>;
    async fn edges(&self, entry_id: &Uuid, direction: EdgeDirection) -> Result<Vec<GraphEdge>>;
    async fn traverse(&self, start: &Uuid, relation: Option<&str>, direction: EdgeDirection, max_depth: usize) -> Result<Vec<KnowledgeEntry>>;
    async fn shortest_path(&self, from: &Uuid, to: &Uuid) -> Result<Option<Vec<KnowledgeEntry>>>;
    async fn remove_edges(&self, entry_id: &Uuid) -> Result<()>;
}
```

**Implementations:** `LiteGraphStore` (petgraph DiGraph + Redb persistence),
`SurrealStore` (SurrealDB edge table + graph queries).

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

| Tool | Parameters | Description |
|------|-----------|-------------|
| `corvia_search` | `query`, `scope_id`, `limit?` | Semantic similarity search over knowledge entries |
| `corvia_write` | `content`, `scope_id`, `source_version?` | Write a knowledge entry (requires `_meta.agent_id` — anonymous clients are read-only) |
| `corvia_history` | `entry_id` | Get the supersession history chain for a knowledge entry |
| `corvia_graph` | `entry_id` | Get all graph edges (both directions) for a knowledge entry |
| `corvia_reason` | `scope_id`, `check?` | Run reasoning health checks on a scope (omit `check` to run all) |
| `corvia_agent_status` | *(needs `_meta.agent_id`)* | Get the calling agent's session count and contribution summary |

## Key Design Decisions

corvia's architecture is shaped by documented design decisions made during development. The
most architecturally significant ones are listed here with the alternative that was considered
and the reasoning behind the choice. The full decision log with alternatives and rationale is
available at [`docs/plans/`](docs/plans/).

| Decision | Alternative rejected | Why this choice wins |
|----------|---------------------|---------------------|
| **AGPL-3.0-only license** | MIT/Apache-2.0 | SaaS protection — prevents cloud providers from offering corvia-as-a-service without contributing back. Dual-license path preserves commercial option (Grafana/MinIO playbook) |
| **SurrealDB behind `QueryableStore` trait** | Hardcode SurrealDB directly | Trait boundary enabled building LiteStore (zero-Docker) without touching kernel code. Storage became a compile-time choice, not a runtime dependency |
| **LLM-assisted merge** | Last-write-wins / manual merge | Knowledge conflicts are semantic, not textual — an LLM can reason about which version of "how auth works" is more accurate. Last-write-wins silently loses knowledge |
| **Two-tier storage (LiteStore + FullStore)** | SurrealDB-only | Requiring Docker for a dev tool kills adoption. LiteStore is the full product at zero cost; FullStore is a power-user upgrade, not a requirement |
| **petgraph + Redb for graph** | SurrealDB graph queries only | LiteStore needed full traversal (BFS, shortest path, cycles) without any external service. petgraph handles computation in-memory, Redb handles persistence |
| **Redb compound-key range scans for temporal** | Dedicated time-series DB | Redb is already embedded for coordination. Compound keys `(scope, valid_from, entry_id)` give O(log n) temporal lookups without adding another dependency |
| **Git-trackable JSON knowledge files** | Database-only storage | Storing knowledge as JSON files designed for Git gives auditability and diffability. `corvia rebuild` reconstructs all indexes from JSON files alone — the database is a cache, the files are truth |
| **Staging hybrid (branches + shared HNSW)** | Full git branch isolation | Pure branch isolation prevents cross-agent search during staging. Shared HNSW lets agents see each other's work-in-progress while writes stay isolated until commit |
| **Multi-layer agent identity** | Single registration model | Forcing registration blocks MCP clients and anonymous queries. Three tiers (Registered, MCP Client, Anonymous) let any agent framework integrate with appropriate write permissions |
| **Python SDK via PyO3 (planned)** | HTTP-only Python client | Embedding the Rust kernel in Python (like Polars) gives zero-latency access to all kernel operations — not just what the REST API exposes |

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
