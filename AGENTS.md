# corvia

> Organizational memory for AI agents. Rust workspace, AGPL-3.0.

This file follows the [AGENTS.md standard](https://agents.md/) — the cross-platform
convention for AI agent project instructions, governed by the Agentic AI Foundation
under the Linux Foundation. Read natively by Codex CLI, Cursor, GitHub Copilot,
Windsurf, and Goose. Supported via import/config by Claude Code (`@AGENTS.md`
in CLAUDE.md), Gemini CLI (settings.json), and Aider (`--read AGENTS.md`).

## Quick Reference

```bash
cargo build --workspace          # Build everything
cargo test --workspace           # Run all tests (SurrealDB/PG tests auto-skip)
make test-full                   # Start SurrealDB + run ALL tests fully
make test-postgres               # Start PostgreSQL + run tests with --features postgres
make test-all                    # Start both DBs + run all tests
```

## Test Tiers

**Tier 1 — No external services (default):**
```bash
cargo test --workspace
```
Runs 433+ tests. LiteStore tests exercise full functionality. SurrealDB and PostgreSQL
tests auto-skip gracefully when the server is unreachable. No Docker, no Ollama required.
This is the primary test suite and must always pass.

**Tier 2 — With SurrealDB (FullStore tests fully exercised):**
```bash
make test-full                   # Start SurrealDB, run tests, stop SurrealDB
```
Or manually: `make surrealdb-up && cargo test --workspace && make surrealdb-down`

All tests run with SurrealDB tests fully exercised instead of skipping.

**Tier 2.5 — With PostgreSQL (PostgresStore tests fully exercised):**
```bash
make test-postgres               # Start PostgreSQL, run tests, stop PostgreSQL
```
Or manually: `make postgres-up && cargo test --workspace --features postgres && make postgres-down`

Requires the `postgres` feature flag. PostgresStore tests use pgvector/pgvector:pg17.

**Tier 3 — With Ollama (real embeddings):**
Requires Ollama running on port 11434 with `nomic-embed-text` model.
The e2e integration tests (`test_ollama_*`) auto-skip when Ollama is unreachable.

**All tiers combined:**
```bash
make test-all                    # Start SurrealDB + PostgreSQL, run all tests
```

## Workspace Crates

| Crate | Path | What it does |
|-------|------|-------------|
| `corvia-common` | `crates/corvia-common` | Types, config, errors, namespace, events |
| `corvia-kernel` | `crates/corvia-kernel` | Storage, coordination, reasoning, graph, temporal, RAG, chunking |
| `corvia-server` | `crates/corvia-server` | Axum REST + MCP protocol server |
| `corvia-cli` | `crates/corvia-cli` | CLI binary, workspace management |
| `corvia-inference` | `crates/corvia-inference` | gRPC inference server (ONNX Runtime) |
| `corvia-proto` | `crates/corvia-proto` | Protocol Buffers for gRPC inference |
| `corvia-adapter-git` | `adapters/corvia-adapter-git/rust` | Git + tree-sitter code ingestion adapter |
| `corvia-adapter-basic` | `adapters/corvia-adapter-basic/rust` | Basic filesystem ingestion adapter |
| `corvia-telemetry` | `crates/corvia-telemetry` | Structured tracing, span contracts, telemetry init |

## Key Traits (extend these, don't modify)

- `QueryableStore` — init_schema, insert, search, get, count, delete_scope, as_any
- `TemporalStore` — as_of, history, evolution
- `GraphStore` — relate, edges, traverse, shortest_path, remove_edges
- `InferenceEngine` — embed (text → vector)
- `IngestionAdapter` — ingest (path → chunks)

## Architecture Decisions to Respect

- **Three-tier storage**: LiteStore is the full product (zero Docker). SurrealStore and PostgresStore are opt-in upgrades.
  PostgresStore requires `--features postgres` at compile time.
- **Git as truth**: All knowledge stored as JSON in `.corvia/knowledge/`, tracked by Git.
- **Local-first**: No API keys required. Ollama provides embeddings.
- **Trait-bounded**: All storage/inference/ingestion is behind traits. Don't add concrete dependencies.

## Known SurrealDB v3 Quirks

When writing SurrealDB queries or tests, be aware of these SDK issues:

1. **Record type deserialization**: `SELECT *` fails — use explicit field lists with
   `record::id(id) AS id` to convert Record types to strings.

2. **NONE vs NULL**: Rust `Option::None` → JSON `null` → SurrealDB `NULL` (not `NONE`).
   Use `IS NULL` in queries, not `IS NONE`.

3. **Native datetimes**: `time::now()` creates native datetimes that can't deserialize as strings.
   Use: `IF type::is_datetime(f) THEN type::string(f) ELSE f END`

4. **HNSW dimension caching (server-level bug)**: Once an HNSW index is created with dimension N
   on a SurrealDB server, ALL subsequent indexes with the same name enforce dimension N — even
   in different namespaces/databases, even after `REMOVE TABLE`. Workaround: all test suites
   sharing a SurrealDB instance MUST use the same embedding dimension (768).

5. **`DEFINE INDEX IF NOT EXISTS`** won't update an existing index. To change dimensions,
   you must `REMOVE INDEX` first.

6. **Namespace/database creation**: Must `DEFINE NAMESPACE IF NOT EXISTS` and
   `DEFINE DATABASE IF NOT EXISTS` before `use_ns()`/`use_db()`.

## PostgreSQL (pgvector) Notes

PostgresStore (`crates/corvia-kernel/src/postgres_store.rs`) is behind the `postgres` feature flag.

- **Dependencies**: `sqlx` (async PostgreSQL driver) + `pgvector` (vector type support)
- **Docker image**: `pgvector/pgvector:pg17` — PostgreSQL 17 with pgvector pre-installed
- **Default URL**: `postgres://corvia:corvia@127.0.0.1:5432/corvia`
- **Env override**: `CORVIA_POSTGRES_URL`
- **Schema**: Uses `vector(768)` column type, HNSW index with cosine ops, JSONB for metadata
- **Graph**: Relational `edges` table with composite primary key `(from_id, relation, to_id)`
- **Temporal**: `valid_from`/`valid_to` TIMESTAMPTZ columns, `superseded_by` UUID chain
- **Tests**: Each test gets its own database (`corvia_test_{PID}_{suffix}`), dropped on teardown

## Testing Conventions

- Unit tests live in `#[cfg(test)] mod tests` inside each module
- Integration tests live in `tests/integration/`
- SurrealDB tests use `connect_test_store("unique_suffix")` for per-test database isolation
- SurrealDB database names include a per-process run ID (`TEST_RUN_ID` = PID hex) so
  concurrent test runs against the same SurrealDB server don't collide
- SurrealDB tests gracefully skip when server is unreachable (check the pattern in knowledge_store.rs)
- HNSW approximate recall is unreliable at <10 entries — use `>=` assertions, not `==`
- Use `tempfile::tempdir()` for test directories (auto-cleanup)
- Env var config tests (`test_env_override_*`) can be flaky under high parallelism
  due to process-global env var mutations — known pre-existing issue

## Key Files

- `traits.rs` — All kernel trait definitions (QueryableStore, InferenceEngine, TemporalStore, GraphStore, IngestionAdapter)
- `lite_store.rs` — LiteStore implementation (default, zero-Docker)
- `knowledge_store.rs` — SurrealStore implementation (FullStore)
- `postgres_store.rs` — PostgresStore implementation (feature-gated)
- `rag_pipeline.rs` — RAG orchestrator (context + ask modes)
- `retriever.rs` — Vector + graph-expanded retrieval with visibility filtering
- `chunking_pipeline.rs` — Format-aware chunking orchestrator with FormatRegistry
- `reasoner.rs` — 5 deterministic health checks + 2 LLM-assisted checks
- `graph_store.rs` — petgraph-based graph for LiteStore
- `adapter_discovery.rs` — Runtime adapter discovery via PATH scan
- `process_adapter.rs` — IPC wrapper for adapter binaries (JSONL protocol)
- `staging.rs` — Agent write isolation (branch-per-session)
- `agent_coordinator.rs` — Multi-agent lifecycle orchestration
- `ops.rs` — Shared kernel operations (system status, config get/set, GC, rebuild)
- `grpc_engine.rs` — gRPC client for corvia-inference server
- `crates/corvia-telemetry/src/lib.rs` — Telemetry init, span name constants, exporters
- `crates/corvia-server/src/dashboard/mod.rs` — Dashboard routes and handlers
- `crates/corvia-server/src/dashboard/clustering.rs` — K-means ClusterStore, shared embedding vocabulary
- `crates/corvia-server/src/dashboard/activity.rs` — Activity feed with semantic grouping
- `crates/corvia-inference/src/backend.rs` — GPU backend resolution (CUDA/OpenVINO/CPU)
- `crates/corvia-cli/src/hooks.rs` — Doc-placement hook generation from DocsConfig
- `crates/corvia-common/src/config.rs` — DocsConfig, InferenceConfig, content_role/source_origin filters
