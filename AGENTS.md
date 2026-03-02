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
cargo test --workspace           # Run all tests (SurrealDB tests auto-skip)
make test-full                   # Start SurrealDB + run ALL tests fully
```

## Test Tiers

**Tier 1 — No external services (default):**
```bash
cargo test --workspace
```
Runs 203 tests. LiteStore tests exercise full functionality. SurrealDB-dependent tests (10)
auto-skip gracefully when the server is unreachable. No Docker, no Ollama required.
This is the primary test suite and must always pass.

**Tier 2 — With SurrealDB (FullStore tests fully exercised):**
```bash
make test-full                   # Start SurrealDB, run tests, stop SurrealDB
```
Or manually: `make surrealdb-up && cargo test --workspace && make surrealdb-down`

All 203 tests run with SurrealDB tests fully exercised instead of skipping.

**Tier 3 — With Ollama (real embeddings):**
Requires Ollama running on port 11434 with `nomic-embed-text` model.
The e2e integration tests (`test_ollama_*`) auto-skip when Ollama is unreachable.

## Workspace Crates

| Crate | Path | What it does |
|-------|------|-------------|
| `corvia-common` | `crates/corvia-common` | Types, config, errors, namespace, events |
| `corvia-kernel` | `crates/corvia-kernel` | Storage, coordination, reasoning, graph, temporal |
| `corvia-server` | `crates/corvia-server` | Axum REST + MCP protocol server |
| `corvia-cli` | `crates/corvia-cli` | CLI binary, workspace management |
| `corvia-adapter-git` | External crate | Git + tree-sitter ingestion adapter |

## Key Traits (extend these, don't modify)

- `QueryableStore` — init_schema, insert, search, get, count, delete_scope
- `TemporalStore` — as_of, history, evolution
- `GraphStore` — relate, edges, traverse, shortest_path, remove_edges
- `InferenceEngine` — embed (text → vector)
- `IngestionAdapter` — ingest (path → chunks)

## Architecture Decisions to Respect

- **Two-tier storage**: LiteStore is the full product (zero Docker). SurrealStore is opt-in.
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

- `knowledge_store.rs` — SurrealStore implementation (FullStore)
- `lite_store.rs` — LiteStore implementation (default)
- `traits.rs` — All kernel trait definitions
- `reasoner.rs` — 5 deterministic health checks + 2 LLM-assisted checks
- `graph_store.rs` — petgraph-based graph for LiteStore
- `staging.rs` — Agent write isolation (branch-per-session)
- `agent_coordinator.rs` — Multi-agent lifecycle orchestration
