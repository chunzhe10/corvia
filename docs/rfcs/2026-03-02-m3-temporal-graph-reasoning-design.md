# M3 Design: Temporal + Graph + Reasoning

*"Knowledge that remembers — and reasons"*

**Milestone:** M3 (single milestone, D56)
**Prerequisites:** M1 ✓, M2 ✓, M2.1 ✓
**Decisions:** D37, D38, D44, D56-D59

---

## Overview

M3 adds three subsystems to the kernel: bi-temporal queries, graph traversal, and algorithmic reasoning. Together they transform Corvia from a search engine into a knowledge system that tracks *when* things changed, *how* they relate, and *what patterns* emerge.

**Design principles:**
- Temporal + graph are pure storage traits — no LLM needed (D57)
- Algorithmic reasoning runs without external dependencies
- LLM-powered findings are opt-in, disabled by default
- Both LiteStore and SurrealStore implement all new traits
- `corvia upgrade` provides the LiteStore → FullStore migration path (D59)

---

## Architecture

```
                    CLI / REST / MCP
                         │
              ┌──────────┼──────────┐
              │     Context Builder  │  (existing, enhanced)
              │          │           │
    ┌─────────┼──────────┼───────────┼─────────┐
    │         ▼          ▼           ▼         │
    │   TemporalStore  GraphStore  Reasoner    │  ← NEW
    │         │          │           │         │
    │         └──────────┼───────────┘         │
    │                    ▼                     │
    │              QueryableStore              │  (existing)
    │            LiteStore / SurrealStore       │
    └──────────────────────────────────────────┘
```

**New traits:** `TemporalStore`, `GraphStore` — implemented by both LiteStore and SurrealStore.
**New module:** `Reasoner` — a compute layer consuming all three store traits. Not a storage trait itself.
**New CLI commands:** `history`, `graph`, `relate`, `reason`, `upgrade`.

---

## 1. Temporal Store (D38)

### Storage: Redb Compound-Key Index (LiteStore)

New Redb table:

```
TEMPORAL_INDEX: (scope_id, valid_from_millis, entry_id) → (valid_to_millis, recorded_at_millis)
```

The compound key enables O(log n) range scans — Redb's B-tree gives ordered iteration over entries within a scope, sorted by valid time. This is the bi-temporal foundation: `valid_from`/`valid_to` = when the knowledge was true, `recorded_at` = when it was stored.

### Storage: SurrealDB (FullStore)

SurrealDB entries already store temporal fields. Queries use SurrealQL:
- `as_of`: `SELECT * FROM knowledge WHERE scope_id = $scope AND valid_from <= $ts AND (valid_to > $ts OR valid_to IS NONE)`
- `history`: Follow `superseded_by` chain via recursive SELECT
- `evolution`: `SELECT * FROM knowledge WHERE scope_id = $scope AND (valid_from BETWEEN $from AND $to OR valid_to BETWEEN $from AND $to)`

### Trait

```rust
#[async_trait]
pub trait TemporalStore: Send + Sync {
    /// Return entries valid at a point in time within a scope.
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Follow the supersession chain for an entry (newest → oldest).
    async fn history(&self, entry_id: &Uuid) -> Result<Vec<KnowledgeEntry>>;

    /// Return entries that changed within a time range.
    async fn evolution(
        &self,
        scope_id: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<KnowledgeEntry>>;
}
```

### Insert Path Integration

`LiteStore::insert()` gains one additional Redb write to `TEMPORAL_INDEX` during each insert. When an entry supersedes another (via `superseded_by`), the old entry's `valid_to` is updated in both `ENTRIES` and `TEMPORAL_INDEX`.

### CLI

```
corvia history <entry-id>                  # supersession chain
corvia evolution --since 7d                # what changed in last 7 days
corvia evolution --scope kernel            # scoped to namespace
corvia evolution --scope kernel --since 1d # combined
```

---

## 2. Graph Store (D37, D58)

### Storage: Redb Edges + petgraph In-Memory (LiteStore)

New Redb table:

```
GRAPH_EDGES: (from_id, relation, to_id) → edge_metadata_json
```

On startup, LiteStore reads all edges from Redb and populates a `petgraph::DiGraph<Uuid, GraphEdge>` in memory. This gives BFS, DFS, shortest path, and cycle detection at graph-traversal speed. Redb is the persistence layer; petgraph is the compute layer.

### Storage: SurrealDB (FullStore)

SurrealDB's native graph features via `RELATE`:
- `RELATE entry_a->imports->entry_b SET metadata = {...}`
- Traversal: `SELECT ->imports->knowledge FROM $start`
- Advanced (FullStore-only): centrality, community detection, PageRank

### Types

```rust
pub struct GraphEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub relation: String,
    pub metadata: Option<serde_json::Value>,
}

pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}
```

### Trait

```rust
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Create a directed edge between two entries.
    async fn relate(
        &self,
        from: &Uuid,
        relation: &str,
        to: &Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()>;

    /// Get all edges from/to an entry.
    async fn edges(
        &self,
        entry_id: &Uuid,
        direction: EdgeDirection,
    ) -> Result<Vec<GraphEdge>>;

    /// BFS/DFS traversal from a starting node, optionally filtering by relation type.
    async fn traverse(
        &self,
        start: &Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Shortest path between two entries.
    async fn shortest_path(
        &self,
        from: &Uuid,
        to: &Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>>;

    /// Delete edges involving an entry (called during delete_scope).
    async fn remove_edges(&self, entry_id: &Uuid) -> Result<()>;
}
```

### Auto-Extraction During Ingestion (D58)

The `corvia-adapter-git` crate already uses tree-sitter to parse code into chunks. M3 extends the adapter to also extract structural relations alongside chunks:

| Language Pattern | Relation | Example |
|---|---|---|
| `use foo::bar` / `import x` | `imports` | file A imports file B |
| `fn_call(...)` expression | `calls` | function A calls function B |
| `impl Trait for Type` | `implements` | type implements trait |
| Module/function nesting | `contains` | module contains function |

Relations are emitted from the adapter as `Vec<(chunk_index, relation, chunk_index)>` alongside the `Vec<KnowledgeEntry>` chunks. The kernel resolves chunk indices to UUIDs after insert and calls `GraphStore::relate()`.

Relation extraction is **best-effort** — tree-sitter gives AST structure, but cross-file resolution (which `bar` does `use foo::bar` refer to?) requires matching against the scope's entry set by source file path. Unresolvable references are skipped silently.

**Relation types are strings, not enums.** The above are conventions. Agents can create custom relation types via the API.

### CLI

```
corvia graph <entry-id>                        # show edges
corvia relate <from-id> depends_on <to-id>     # manual edge creation
corvia graph --scope kernel --relation imports  # all import edges in scope
```

---

## 3. Reasoner (D44 Levels 2-3, D57)

The Reasoner is a compute module — not a storage trait. It reads from `TemporalStore` + `GraphStore` + `QueryableStore` and produces findings as `KnowledgeEntry` objects.

### Level 2: Algorithmic Analysis (no LLM needed)

| Check | Description | Detection Method |
|---|---|---|
| Stale entries | `valid_to` set but no replacement via `superseded_by` | Temporal query + get |
| Broken chains | `superseded_by` points to nonexistent entry | History traversal |
| Orphaned nodes | Entries with zero graph edges (isolated knowledge) | Graph degree check |
| Dangling imports | `imports` edge target doesn't resolve to any entry | Graph edge validation |
| Dependency cycles | Circular `depends_on` chains | petgraph cycle detection |
| Missing coverage | Scopes with low entry count relative to repo size | Count heuristic |

### Finding Format

Findings are stored as `KnowledgeEntry` with metadata identifying them:

```rust
EntryMetadata {
    chunk_type: Some("finding".into()),
    // Additional fields encoded in a JSON metadata extension:
    // {
    //   "finding_type": "stale_entry",
    //   "target_ids": ["uuid1", "uuid2"],
    //   "confidence": 0.9,
    //   "rationale": "Entry superseded 3 days ago with no replacement"
    // }
}
```

Findings have `scope_id` matching the analyzed scope, so they appear in search results for that scope. They can be queried by `chunk_type = "finding"` to list all findings.

### Level 3: LLM-Powered Findings (opt-in, D57)

Enabled when `[reasoning]` section exists in `corvia.toml`:

```toml
[reasoning]
provider = "ollama"        # or "anthropic", "openai"
model = "llama3.2:3b"      # any completion model
```

Additional checks:
- **Semantic gap analysis** — embed a set of probing queries, find areas with sparse coverage
- **Contradiction detection** — compare entries with high embedding similarity but divergent content
- **Natural language findings** — LLM summarizes detected patterns in human-readable form

When `[reasoning]` is absent, `corvia reason --llm` prints: *"Configure [reasoning] in corvia.toml to enable LLM-powered analysis."*

### CLI

```
corvia reason                          # run all algorithmic checks
corvia reason --scope kernel           # scope-filtered
corvia reason --check stale            # single check type
corvia reason --llm                    # include LLM checks (if configured)
```

---

## 4. `corvia upgrade` (D59)

Migration path: LiteStore → FullStore.

```
corvia upgrade --target surrealdb --url ws://localhost:8000
```

**Steps:**
1. Read all `.corvia/knowledge/**/*.json` files
2. Connect to SurrealDB, create schema
3. Bulk-insert entries with embeddings (batched, 100 per transaction)
4. Migrate graph edges: Redb `GRAPH_EDGES` → SurrealDB `RELATE`
5. Migrate temporal index entries
6. Verify: compare counts (LiteStore vs SurrealStore)
7. Update `corvia.toml`: `store_type = "surrealdb"`, add connection config
8. Print summary and next steps

**Safety:** Original LiteStore files are not deleted. Revert by changing `store_type` back to `"lite"` in `corvia.toml`.

---

## 5. `corvia rebuild` Enhancement

Currently rebuilds HNSW index from knowledge JSON files. M3 extends to also reconstruct:
- **Temporal index** — re-scan all entries, rebuild `TEMPORAL_INDEX` table from `valid_from`/`valid_to` fields
- **Graph edges** — re-extract from code chunks via tree-sitter, rebuild `GRAPH_EDGES` table
- **petgraph** — rebuild in-memory graph from restored Redb edges

```
corvia rebuild                # full rebuild: HNSW + temporal + graph
corvia rebuild --hnsw-only    # just the vector index (existing behavior)
```

---

## 6. MCP + REST Enhancements

### MCP Tools (flesh out existing placeholders)

- `corvia_history` — already stubbed in `mcp.rs`, now backed by `TemporalStore::history()`
- `corvia_graph` — new tool, returns edges for an entry
- `corvia_reason` — new tool, runs algorithmic checks and returns findings

### REST Endpoints

- `GET /api/v1/entries/{id}/history` — supersession chain
- `GET /api/v1/entries/{id}/edges` — graph edges
- `POST /api/v1/edges` — create edge (body: `{from, relation, to, metadata}`)
- `GET /api/v1/evolution?scope={scope}&since={duration}` — temporal evolution
- `POST /api/v1/reason?scope={scope}` — run reasoning, return findings

---

## 7. Self-Dogfooding

Corvia's own design decisions (D1-D59) ingested as searchable knowledge entries with provenance chains. Validates the full M3 pipeline: temporal tracking of decisions, graph relations between decisions (D37 depends on D35), and reasoner detecting any gaps.

---

## Dependencies

| Crate | Version | Purpose | Status |
|---|---|---|---|
| `hnsw_rs` | 0.3 | Vector search | Already in Cargo.toml |
| `redb` | 2 | Metadata/temporal/graph persistence | Already in Cargo.toml |
| `petgraph` | latest | In-memory graph traversal | **NEW — add to Cargo.toml** |
| `surrealdb` | 3 | FullStore backend | Already in Cargo.toml |
| `chrono` | latest | Temporal types | Already in Cargo.toml |

---

## Test Strategy

- **Unit tests** per trait method (temporal, graph, reasoner) — both LiteStore and SurrealStore
- **Integration tests** — ingest code → extract edges → run reasoner → verify findings
- **Rebuild test** — insert entries, drop indexes, rebuild, verify search+temporal+graph all work
- **Upgrade test** — create LiteStore data, upgrade to SurrealStore, verify parity
- **Self-dogfood test** — ingest Corvia's own codebase, run reasoner, verify non-empty findings

---

## Estimated Scope

| Component | New Files | Estimated LOC |
|---|---|---|
| `TemporalStore` trait + LiteStore impl | 1-2 | ~300 |
| `TemporalStore` SurrealStore impl | 1 | ~150 |
| `GraphStore` trait + types | 1 | ~100 |
| `GraphStore` LiteStore impl (petgraph) | 1 | ~400 |
| `GraphStore` SurrealStore impl | 1 | ~200 |
| Reasoner module | 1 | ~500 |
| Adapter relation extraction | 1 (extend existing) | ~300 |
| CLI commands (history, graph, relate, reason, upgrade) | extend main.rs | ~400 |
| REST/MCP endpoints | extend server | ~200 |
| Tests | 2-3 | ~500 |
| **Total** | **~10 files** | **~3,000 LOC** |
