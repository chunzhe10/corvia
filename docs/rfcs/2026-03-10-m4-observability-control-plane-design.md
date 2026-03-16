# M4: Observability + Control Plane — Design Spec

**Date:** 2026-03-10
**Status:** Shipped (v0.4.0)
**Depends on:** M3 (complete)
**Supersedes:** `docs/rfcs/2026-03-04-m4-revised-plan.md` (git commit `55f0a3b`)

---

## Overview

M4 adds two capabilities to corvia:

1. **Observability** — structured tracing spans across kernel subsystems, telemetry configuration, and `corvia status --metrics` for CLI users.
2. **Control plane** — 10 new MCP admin tools that let LLMs (and humans via MCP clients) inspect and manage corvia: view system status, read/write config, trigger GC, manage agents, retry failed merges.

Both capabilities share a foundation: extracted kernel operations (`ops.rs`) callable from both CLI and MCP, eliminating code duplication.

## Key Design Decisions

### Informed by OSS research

| Lesson | Source | Application |
|--------|--------|-------------|
| Centralize initialization, not instrumentation | Databend, InfluxDB IOx | `corvia-telemetry` owns `init_telemetry()` + constants; leaf crates use `tracing` directly |
| Don't wrap what doesn't need wrapping | InfluxDB IOx removed `observability_deps` | No re-export crate; crates depend on `tracing` directly |
| Constants and helpers earn their crate | Databend `common-tracing` pattern | Span name constants prevent drift across 8 workspace crates |
| Separate product telemetry from operational observability | Quickwit, Databend | `corvia-telemetry` is operational only; usage analytics is a future concern |

### Informed by codebase investigation

| Finding | Impact on M4 |
|---------|-------------|
| MCP server is transport-stateless (no session tracking) | Control plane tools fit cleanly — just add handlers that take `&AppState` |
| `CorviaConfig` not in `AppState` | Must add `Arc<RwLock<CorviaConfig>>` + `config_path` for config read/write tools |
| `MergeQueue::dequeue_batch()` is read-only despite name | No new method needed — use existing `dequeue_batch()` for queue inspection, rename to `list()` for clarity |
| `_meta` already parsed in `handle_tools_call` | Safety tier `_meta.confirmed` needs zero protocol changes |
| CLI server-aware routing is orthogonal to ops.rs | `ops.rs` is local-only in-process functions; routing is a separate concern |
| Protocol version negotiates `2025-03-26` / `2024-11-05` | Both support `_meta`; no upgrade needed for safety tiers |
| `AgentCoordinator` exposes `registry`, `sessions`, `merge_queue` | ops.rs functions access these via `coordinator.registry`, `coordinator.sessions`, `coordinator.merge_queue` |

---

## Deliverables (10 items, 4 phases)

### Phase 1: Foundation

#### D80: `corvia-telemetry` crate

New crate at `crates/corvia-telemetry/`.

**Contains:**
- `init_telemetry(config: &TelemetryConfig) -> Result<()>` — configures `tracing-subscriber` pipeline. Stdout and file exporters wired in M4; OTLP exporter wired in M5.
- Span name constants matching D45 observability contract:
  ```rust
  pub mod spans {
      pub const AGENT_REGISTER: &str = "corvia.agent.register";
      pub const SESSION_CREATE: &str = "corvia.session.create";
      pub const ENTRY_WRITE: &str = "corvia.entry.write";
      pub const ENTRY_EMBED: &str = "corvia.entry.embed";
      pub const ENTRY_INSERT: &str = "corvia.entry.insert";
      pub const SESSION_COMMIT: &str = "corvia.session.commit";
      pub const MERGE_PROCESS: &str = "corvia.merge.process";
      pub const MERGE_CONFLICT: &str = "corvia.merge.conflict";
      pub const MERGE_LLM_RESOLVE: &str = "corvia.merge.llm_resolve";
      pub const GC_RUN: &str = "corvia.gc.run";
      pub const SEARCH: &str = "corvia.search";
      pub const STORE_INSERT: &str = "corvia.store.insert";
      pub const STORE_SEARCH: &str = "corvia.store.search";
      pub const STORE_GET: &str = "corvia.store.get";
      pub const RAG_CONTEXT: &str = "corvia.rag.context";
      pub const RAG_ASK: &str = "corvia.rag.ask";
  }
  ```

**Does NOT contain:**
- No re-exports of `tracing` macros (InfluxDB IOx lesson)
- No metric types or custom layers (YAGNI until M5/Grafana)
- No panic hooks (existing `tracing` integration sufficient)
- No business logic

**Config type placement:** `TelemetryConfig` lives in `corvia-common/src/config.rs` alongside all other config structs (consistent with existing pattern). `corvia-telemetry` depends on `corvia-common` for the type — NOT the reverse. This avoids circular dependencies and keeps config deserialization unified.

**Dependencies added to workspace `Cargo.toml`:**
- `tracing` (already present)
- `tracing-subscriber` with `env-filter`, `json`, `fmt` features
- `tracing-appender` (for file exporter / `RollingFileAppender`)
- `opentelemetry` + `opentelemetry-otlp` (declared but OTLP init deferred to M5)

**Dependency direction constraint:** `corvia-telemetry` depends on `corvia-common` (for `TelemetryConfig`). `corvia-common` does NOT depend on `corvia-telemetry`. Leaf crates (`corvia-kernel`, `corvia-server`, `corvia-cli`) depend on both.

**Estimated size:** ~200-400 lines across 2-3 files.

#### D82: Shared kernel operations — `ops.rs`

New file at `crates/corvia-kernel/src/ops.rs`.

Extracted functions callable from both CLI and MCP server (in-process only, no remote path):

| Function | Extracted From | Purpose |
|----------|---------------|---------|
| `system_status(store, coordinator, scope_id)` | `cmd_status` | Entry counts, agents, sessions, queue depth |
| `rebuild_index(data_dir, dimensions)` | `cmd_rebuild` | Rebuild HNSW from knowledge files |
| `agents_list(coordinator)` | `cmd_agent list` | All registered agents (via `coordinator.registry`) |
| `sessions_list(coordinator, agent_id)` | `cmd_agent sessions` | Sessions for an agent (via `coordinator.sessions`) |
| `merge_queue_status(coordinator, limit)` | New | Queue depth + peek (via `coordinator.merge_queue.dequeue_batch()`) |
| `adapters_list(search_dirs)` | `cmd_adapters list` | Wraps `discover_adapters()` |
| `config_get(config, section)` | New | Read config section as JSON |
| `config_set(config_path, config, section, key, value)` | New | Validate, write TOML, return updated |
| `agent_suspend(coordinator, agent_id)` | New | Set agent status to Suspended (via `coordinator.registry.set_status()`) |
| `gc_run(coordinator)` | New (wraps `coordinator.gc()`) | Run GC sweep |
| `merge_retry(coordinator, entry_ids)` | New | Reset `retry_count` to 0 and re-enqueue failed entries for the merge worker to pick up on its next cycle |

**Kernel note:** `MergeQueue::dequeue_batch()` is already non-destructive (read-only Redb transaction despite the name). Rename to `list()` for clarity as part of this deliverable. No new method needed.

**Access pattern:** ops.rs functions take `&AgentCoordinator` and access sub-components via `coordinator.registry`, `coordinator.sessions`, `coordinator.merge_queue`. These are already public fields.

**CLI refactor:** `cmd_status`, `cmd_rebuild`, `cmd_agent list/sessions` refactored to call `ops::*`. Behavior stays identical.

**Gate:** `cargo test --workspace` passes, CLI output unchanged.

---

### Phase 2: Instrumentation + Config Infrastructure

#### D81: Kernel instrumentation

Add `#[tracing::instrument]` to key public methods. Each crate depends on `tracing` directly and imports span name constants from `corvia-telemetry`.

| Module | Methods | Span Constants Used |
|--------|---------|-------------------|
| `agent_coordinator.rs` | `register_agent`, `create_session`, `write_entry`, `commit_session`, `gc`, `process_merge_queue` | `AGENT_REGISTER`, `SESSION_CREATE`, `ENTRY_WRITE`, `SESSION_COMMIT`, `GC_RUN`, `MERGE_PROCESS` |
| `merge_worker.rs` | `detect_conflict`, `merge_entries`, `process_entry` | `MERGE_CONFLICT`, `MERGE_LLM_RESOLVE`, `MERGE_PROCESS` |
| `lite_store.rs` | `insert`, `search`, `get`, `delete_scope` | `STORE_INSERT`, `STORE_SEARCH`, `STORE_GET` |
| `ollama_engine.rs` / `grpc_engine.rs` | `embed` | `ENTRY_EMBED` |
| `rag_pipeline.rs` | `context`, `ask` | `RAG_CONTEXT`, `RAG_ASK` |

**Key fields on spans:** `entry_id`, `agent_id`, `session_id`, `scope_id` where applicable.

**Scope:** LiteStore path only. SurrealDB/PostgresStore instrumentation deferred — same trait methods, same span names, added when those stores are tested with telemetry.

#### D87: Config hot-reload

Add to `AppState` in `rest.rs`:
```rust
pub config: Arc<RwLock<CorviaConfig>>,
pub config_path: PathBuf,
```

Hot-reload matrix:

| Section | Hot-reloadable? | Reason |
|---------|----------------|--------|
| `agent_lifecycle` | Yes | Thresholds read on each GC sweep |
| `merge` | Yes | Read per merge operation |
| `rag` | Yes | Read per RAG query |
| `chunking` | Yes | Read per ingest |
| `reasoning` | Yes | Read per reason invocation |
| `adapters` | Yes | Re-discover on next call |
| `storage` | **No** | Store connection at startup |
| `server` | **No** | Listener bound at startup |
| `embedding` | **No** | Engine constructed at startup |
| `project` | **No** | Scope used throughout |
| `telemetry` | **No** | Subscriber configured at startup |

`ops::config_set()` on non-hot-reloadable sections returns error: `"requires server restart"`.

#### D89: Telemetry config wiring

Add `telemetry: TelemetryConfig` to `CorviaConfig` with `#[serde(default)]` for backward compat:

```toml
[telemetry]
exporter = "stdout"        # stdout | file | otlp
otlp_endpoint = ""         # only when exporter = "otlp"
log_format = "text"        # text | json
metrics_enabled = true
```

Wire `init_telemetry()` into `cmd_serve` startup path. Existing `corvia.toml` files without `[telemetry]` get defaults automatically.

**Gate:** Spans visible in logs at `RUST_LOG=debug`, config read/write works programmatically.

---

### Phase 3: MCP Control Plane

#### D83: Tiered safety model

| Tier | Level | Behavior |
|------|-------|----------|
| `ReadOnly` | Auto-approved | Executes immediately |
| `LowRisk` | Single confirmation | Without `_meta.confirmed`: returns preview + `"confirmation_required": true`. With `_meta.confirmed: true`: executes |
| `MediumRisk` | Confirmation + dry-run | Same as LowRisk + accepts `dry_run: true` argument for preview without mutation |

Implementation:
- `ToolTier` enum in `mcp.rs` (`ReadOnly`, `LowRisk`, `MediumRisk`)
- Each tool definition includes tier in `annotations` metadata
- Dispatch checks `_meta.confirmed` before executing Tier 2+ operations
- Works within both `2024-11-05` and `2025-03-26` protocol versions
- Upgradable to MCP Elicitation when protocol bumps to `2025-06-18`

Existing 8 tools: `corvia_search`, `corvia_context`, `corvia_ask`, `corvia_history`, `corvia_graph`, `corvia_reason`, `corvia_agent_status` are all classified as `ReadOnly`. `corvia_write` keeps its existing auth guard (`_meta.agent_id` required) and is classified as `ReadOnly` tier — the agent_id check is orthogonal to the safety tier system (it's identity-based access control, not mutation confirmation).

**Confirmation response format:** When a Tier 2+ tool is called without `_meta.confirmed`, it returns a **successful MCP response** (`isError: false`) with a content block containing:
```json
{
  "content": [{ "type": "text", "text": "{\"confirmation_required\": true, \"preview\": {...}, \"message\": \"...\"}" }]
}
```
This is a normal response, not an error. The client (LLM or human) reads the preview and re-calls with `_meta.confirmed: true` to execute.

#### D84: Tier 1 MCP tools — read-only (5 tools)

| Tool | Calls | Returns |
|------|-------|---------|
| `corvia_system_status` | `ops::system_status()` | Entry counts, agents, sessions, queue depth |
| `corvia_config_get` | `ops::config_get()` | Config section as JSON (or full config) |
| `corvia_adapters_list` | `ops::adapters_list()` | Discovered adapters with metadata |
| `corvia_agents_list` | `ops::agents_list()` | All registered agents with status |
| `corvia_merge_queue` | `ops::merge_queue_status()` | Queue depth + top entries |

All auto-approved, no confirmation needed. These tools do NOT require `scope_id` — they are cross-scope system operations (separate from the existing `resolve_scope_id` pattern).

#### D85: Tier 2 MCP tools — low-risk mutation (3 tools)

| Tool | Calls | Confirmation |
|------|-------|-------------|
| `corvia_config_set` | `ops::config_set()` | Shows diff, requires `_meta.confirmed` |
| `corvia_gc_run` | `ops::gc_run()` | Shows what would be cleaned, requires `_meta.confirmed` |
| `corvia_rebuild_index` | `ops::rebuild_index()` | Shows current index state, requires `_meta.confirmed` |

Without `_meta.confirmed`: returns a preview of what would change.
With `_meta.confirmed: true`: executes the mutation.

#### D86: Tier 3 MCP tools — medium-risk mutation (2 tools)

| Tool | Calls | Confirmation |
|------|-------|-------------|
| `corvia_agent_suspend` | `ops::agent_suspend()` | Supports `dry_run`, requires `_meta.confirmed` |
| `corvia_merge_retry` | `ops::merge_retry()` | Supports `dry_run`, requires `_meta.confirmed` |

These additionally support `dry_run: true` in arguments for preview without mutation.

**Gate:** `tools/list` returns 18 tools (8 existing + 10 new), all existing tools unchanged.

---

### Phase 4: CLI Observability

#### D88: `corvia status --metrics`

Add `--metrics` flag to `status` command. Calls `ops::system_status()` (same as MCP) plus:
- Config summary (store type, inference provider, telemetry exporter)
- Agent lifecycle stats (from coordinator: active agents, open sessions, merge queue depth)
- Adapter discovery results
- Plain text output (no TUI)

Note: cumulative counters (entries committed/merged/rejected over time) require adding persistent Redb counter tables to the kernel. This is deferred — M4 `--metrics` reports current snapshot state only (what's in the store now, what's in the queue now). Cumulative counters are a future enhancement.

Without `--metrics`: behavior identical to current `corvia status`.

**Gate:** Full test suite passes.

---

## Shared Kernel Pattern

```
CLI cmd_status ────────────┐
                           ├──→ ops::system_status() ──→ store/coordinator
MCP corvia_system_status ──┘

CLI cmd_rebuild ───────────┐
                           ├──→ ops::rebuild_index() ──→ LiteStore
MCP corvia_rebuild_index ──┘

CLI cmd_agent list ────────┐
                           ├──→ ops::agents_list() ──→ AgentRegistry
MCP corvia_agents_list ────┘
```

This is orthogonal to CLI server-aware routing. ops.rs functions execute in-process. The CLI's HTTP delegation to the server (for search, history, etc.) is a separate layer that does not affect ops.rs.

## Files to Create

| File | Purpose |
|------|---------|
| `crates/corvia-telemetry/Cargo.toml` | New crate manifest |
| `crates/corvia-telemetry/src/lib.rs` | `init_telemetry()`, span constants |
| `crates/corvia-kernel/src/ops.rs` | Shared kernel operations |

## Files to Modify

| File | Changes |
|------|---------|
| `Cargo.toml` (workspace) | Add `corvia-telemetry` member |
| `crates/corvia-common/src/config.rs` | Add `TelemetryConfig` struct + `telemetry` field to `CorviaConfig` |
| `crates/corvia-kernel/src/lib.rs` | Add `pub mod ops;` |
| `crates/corvia-kernel/src/merge_queue.rs` | Rename `dequeue_batch()` to `list()` for clarity |
| `crates/corvia-kernel/Cargo.toml` | Add `corvia-telemetry` dependency |
| `crates/corvia-kernel/src/agent_coordinator.rs` | Add `#[tracing::instrument]` spans |
| `crates/corvia-kernel/src/merge_worker.rs` | Add spans |
| `crates/corvia-kernel/src/lite_store.rs` | Add spans |
| `crates/corvia-kernel/src/ollama_engine.rs` | Add spans |
| `crates/corvia-kernel/src/grpc_engine.rs` | Add spans |
| `crates/corvia-kernel/src/rag_pipeline.rs` | Add spans |
| `crates/corvia-server/src/rest.rs` | Add `config` + `config_path` to `AppState` |
| `crates/corvia-server/src/mcp.rs` | Add 10 tool definitions + dispatch + handlers + safety tier. Update test helpers (`test_state`) for new `AppState` fields |
| `crates/corvia-server/Cargo.toml` | Add `corvia-telemetry` dependency |
| `crates/corvia-cli/src/main.rs` | Refactor to use `ops::*`, add `--metrics`, wire `init_telemetry()` |
| `crates/corvia-cli/Cargo.toml` | Add `corvia-telemetry` dependency |

## Dependency Graph

```
D80 (telemetry crate) ──────→ D81 (kernel instrumentation) ──→ D88 (CLI --metrics)
                                                                    ↑
D82 (ops.rs extraction) ──→ D84 (Tier 1 read tools, needs D82 only) │
                         │                                           │
                         ├──→ D85 (Tier 2 tools, needs D82 + D83 + D87)
                         │        ↑          ↑
D83 (safety tier model) ─┘        │          │
D87+D89 (config + telemetry) ─────┘          │
                                             │
                         └──→ D86 (Tier 3 tools, needs D82 + D83)
```

Note: D84 (Tier 1 read-only tools) does NOT depend on D83 (safety tier) — Tier 1 tools are auto-approved. D84 can run in parallel with D83 + D87/D89.

Critical path: D80 + D82 (parallel) → D84 + D83 + D87/D89 (parallel) → D85 → D86 → D81 → D88

## Verification Criteria

1. `cargo test --workspace` — all existing tests pass (385+)
2. `cargo test -p corvia-telemetry` — new crate tests pass
3. `cargo test -p corvia-kernel -- ops` — ops module tests pass
4. `cargo test -p corvia-server -- mcp` — all 18 MCP tools tested
5. `corvia status` — unchanged output (backward compat)
6. `corvia status --metrics` — enhanced output with metrics
7. `corvia serve` → MCP `tools/list` returns 18 tools
8. MCP `corvia_system_status` returns same data shape as CLI `corvia status`
9. MCP `corvia_config_get` → `corvia_config_set` round-trip works
10. Tier 2 tool without `_meta.confirmed` returns confirmation prompt, not mutation
11. Tier 3 tool with `dry_run: true` returns preview without side effects
12. Spans visible in structured logs at `RUST_LOG=debug`

---

*Created: 2026-03-10*
*Supersedes: docs/rfcs/2026-03-04-m4-revised-plan.md (git commit 55f0a3b)*
*References: docs/rfcs/2026-03-04-m4-control-plane-research.md (git commit 55f0a3b)*
*OSS research: Databend, InfluxDB IOx, Linkerd2-proxy, Vector, TiKV patterns*
