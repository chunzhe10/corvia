# Benchmark Script Design: Vector vs Graph-Expanded Retrieval

**Date:** 2026-03-03
**Purpose:** Bash script to run the Post 2 benchmark — comparing vector-only vs graph-expanded retrieval on Corvia's own codebase using LiteStore + corvia-inference.
**Output:** Raw JSON files for later PDF carousel generation.

## Decisions

- **Script:** Single `benchmark.sh` with `--skip-setup` flag for re-runs
- **Storage:** LiteStore (embedded redb, zero Docker)
- **Inference:** corvia-inference (built-in fastembed/ONNX, gRPC :8030)
- **Server:** corvia serve (REST :8020)
- **Data:** Self-ingest Corvia's own codebase via `corvia demo --keep`
- **Output:** Raw JSON only — PDF carousel generated separately by Claude

## Script Lifecycle

```
benchmark.sh [--skip-setup] [--results-dir DIR]

Phases:
  1. PREFLIGHT   — check corvia binary, check ports 8020/8030 free
  2. INFERENCE    — start corvia-inference serve (gRPC :8030), wait for health
  3. INIT         — corvia init --store lite (provider=corvia in config)
  4. INGEST       — corvia demo --keep (self-ingest Corvia codebase)
  5. SERVE        — corvia serve (REST :8020), wait for GET /health
  6. BENCHMARK    — run 4 query pairs (vector-only vs graph-expanded)
  7. TEARDOWN     — stop server, stop inference, save summary

--skip-setup skips phases 2-5, assumes server already running with data.
```

Each phase prints a status line. If any phase fails, the script traps EXIT to
kill background processes and exit cleanly.

## The 4 Query Pairs

Each query runs twice against `POST /v1/context`:
- `expand_graph: false` (vector-only)
- `expand_graph: true` (graph-expanded, α=0.3, depth=2)

### Query 1: Vector wins (or ties) — direct config lookup

**Query:** `"How do I configure the embedding model and what dimensions does it use?"`

**Why realistic:** Onboarding question. A developer joining the project asks how to set up inference.

**Expected behavior:**
- **Vector:** Finds config.rs and embedding_pipeline.rs directly — high semantic similarity.
- **Graph:** Finds the same files plus graph neighbors. Extra context doesn't hurt but doesn't help for a direct lookup.
- **Lesson:** For "where is this thing?" questions, vector search is already great.

### Query 2: Graph wins — structural relationships

**Query:** `"I'm adding a new storage backend. Walk me through what traits I need to implement and how the existing backends do it."`

**Why realistic:** New contributor question. Needs to connect traits.rs → LiteStore → SurrealStore → PostgresStore.

**Expected behavior:**
- **Vector:** Finds traits.rs (high similarity) and maybe one backend. Can't see the full picture.
- **Graph:** Follows `implements` edges from traits → all three backend implementations. Surfaces the complete picture.
- **Lesson:** For "how do these connect?" questions, graph edges surface the chain.

### Query 3: Graph wins clearly — debugging trace

**Query:** `"A search request is returning unexpected results. Trace the full path from the REST API endpoint through the RAG pipeline to the storage layer."`

**Why realistic:** Debugging scenario. Developer needs the full call chain across modules.

**Expected behavior:**
- **Vector:** Finds retriever.rs or server routes — whichever matches the query terms best. Misses the connections.
- **Graph:** Follows the chain: server routes → RAG pipeline → retriever → graph expansion → LiteStore search. Multiple hops surfaced.
- **Lesson:** For debugging across module boundaries, graph traversal finds what vector can't.

### Query 4: Graph wins clearly — cross-subsystem architecture

**Query:** `"How does the agent session lifecycle interact with the staging system and conflict resolution during merge?"`

**Why realistic:** Architecture investigation. Three interconnected subsystems that span multiple files.

**Expected behavior:**
- **Vector:** Finds agent_coordinator.rs or staging.rs — whichever has highest term overlap. Can't connect all three.
- **Graph:** Follows edges from agent_coordinator → staging → merge/conflict resolution. Surfaces the full interaction across three subsystems.
- **Lesson:** For architecture questions spanning subsystems, graph is essential.

## API Calls

Each query pair produces two curl calls:

```bash
# Vector-only
curl -s -X POST http://localhost:8020/v1/context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "<QUERY_TEXT>",
    "scope_id": "corvia-introspect",
    "limit": 10,
    "expand_graph": false
  }'

# Graph-expanded
curl -s -X POST http://localhost:8020/v1/context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "<QUERY_TEXT>",
    "scope_id": "corvia-introspect",
    "limit": 10,
    "expand_graph": true
  }'
```

## Output Structure

```
benchmark-results/2026-03-03T14-30-00/
  metadata.json          # run info: timestamp, corvia version, git SHA, config
  query-1-vector.json    # raw /v1/context response (expand_graph=false)
  query-1-graph.json     # raw /v1/context response (expand_graph=true)
  query-2-vector.json
  query-2-graph.json
  query-3-vector.json
  query-3-graph.json
  query-4-vector.json
  query-4-graph.json
  summary.json           # condensed comparison of key metrics
```

### metadata.json

```json
{
  "run_at": "2026-03-03T14:30:00Z",
  "corvia_version": "0.3.0",
  "git_sha": "abc1234",
  "storage_backend": "lite",
  "inference_provider": "corvia",
  "inference_model": "nomic-embed-text-v1.5",
  "graph_alpha": 0.3,
  "graph_depth": 2,
  "scope_id": "corvia-introspect"
}
```

### summary.json

```json
{
  "run_at": "2026-03-03T14:30:00Z",
  "queries": [
    {
      "id": 1,
      "query": "How do I configure the embedding model and what dimensions does it use?",
      "expected_winner": "vector",
      "vector": {
        "sources_count": 5,
        "top_score": 0.89,
        "top_source_file": "config.rs",
        "source_files": ["config.rs", "embedding_pipeline.rs", "..."],
        "latency_ms": 12,
        "vector_results": 10,
        "graph_expanded": 0
      },
      "graph": {
        "sources_count": 5,
        "top_score": 0.91,
        "top_source_file": "config.rs",
        "source_files": ["config.rs", "embedding_pipeline.rs", "..."],
        "latency_ms": 28,
        "vector_results": 10,
        "graph_expanded": 3
      }
    }
  ]
}
```

## Implementation Notes

- **Trap cleanup:** `trap cleanup EXIT` to kill background pids on failure
- **Health polling:** Poll `GET /health` and gRPC health with timeout (30s max)
- **jq required:** For JSON extraction in summary generation
- **Scope:** Uses `corvia-introspect` scope from `corvia demo`
- **Config override:** After `corvia init`, patch corvia.toml to set `provider = "corvia"` and `url = "http://127.0.0.1:8030"` before running demo/serve
- **Idempotency:** If `--skip-setup` is passed, skip phases 2-5 entirely

## Script Location

```
scripts/benchmark.sh
```

Relative to repo root. Executable (`chmod +x`).
