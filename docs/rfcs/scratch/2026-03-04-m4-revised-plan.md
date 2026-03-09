# Revised M4: "See and steer what's happening inside"

**Date:** 2026-03-04
**Status:** Plan approved, ready for implementation
**Depends on:** M3
**LinkedIn pitch:** *"Giving AI agents a dashboard with controls — observability + operational MCP for a multi-agent knowledge system"*
**Research:** See `2026-03-04-m4-control-plane-research.md` for protocol comparison, industry precedents, and A2A notes.

---

## Context

M4 was originally 4 deliverables focused on pure observability (telemetry crate, CLI metrics, Grafana, OTLP exporter). Research showed that expanding M4 to include LLM-controllable operations via MCP fills an open gap — no AI memory system ships a first-party admin/control MCP server yet.

**Key decisions:**
- **MCP is the control plane protocol** — extend existing MCP server in `mcp.rs`, no new server
- **No duplicated code** — extract shared kernel functions (`ops.rs`) callable from both CLI and MCP
- **A2A deferred** to M7+ (Agent Card publishing as first step)
- **Config stays in TOML** — MCP wraps read/write/validate/hot-reload
- **Grafana dashboards deferred to M5** — pairs naturally with VS Code extension
- **MCP protocol stays at `2024-11-05`** — safety model uses `_meta.confirmed` pattern, upgradable to MCP Elicitation when protocol bumps to `2025-06-18`

---

## Deliverables (10 items, 4 phases)

### Phase 1: Foundation

**D80: `corvia-telemetry` crate**
- New crate at `crates/corvia-telemetry/`
- `init_telemetry(config) -> Result<()>` — configures tracing subscriber pipeline (stdout, file, or OTLP)
- Span name constants matching D45 observability contract (`corvia.entry.write`, `corvia.session.create`, `corvia.merge.process`, etc.)
- Metrics primitives (counters, gauges, histograms) via `opentelemetry` metrics API
- No business logic — purely wiring and constants

**D82: Shared kernel operations — `corvia-kernel/src/ops.rs`**
- Extract CLI-embedded logic into reusable kernel functions callable from both CLI and MCP:

| Function | Extracted from | Purpose |
|----------|---------------|---------|
| `ops::system_status(store, scope_id, coordinator, adapter_dirs)` | `cmd_status` | Entry counts, agents, sessions, queue depth, adapters |
| `ops::rebuild_index(data_dir, dimensions)` | `cmd_rebuild` | Rebuild HNSW from knowledge files |
| `ops::agents_list(registry)` | `cmd_agent list` | All registered agents |
| `ops::sessions_list(session_mgr, agent_id)` | `cmd_agent sessions` | Sessions for an agent |
| `ops::merge_queue_status(merge_queue)` | New | Queue depth + peek at top entries |
| `ops::adapters_list(search_dirs)` | `cmd_adapters list` | Wraps `discover_adapters()` |
| `ops::config_get(config, section)` | New | Read config section as JSON |
| `ops::config_set(config_path, section, key, value)` | New | Validate, write TOML, return updated config |
| `ops::agent_suspend(registry, agent_id)` | New | Set agent status to Suspended |
| `ops::agent_deregister(registry, agent_id)` | New | Deregister agent |
| `ops::gc_run(coordinator)` | Wraps `coordinator.gc()` | Run GC sweep |
| `ops::merge_retry(coordinator, entry_ids)` | New | Retry failed merge entries |

- Refactor CLI `cmd_status`, `cmd_rebuild`, `cmd_agent` to call `ops::*`

**Gate:** `cargo test --workspace` passes, CLI behavior unchanged.

### Phase 2: Instrumentation + Config Infrastructure

**D81: Kernel instrumentation**
- Add `#[tracing::instrument]` to key public methods across:
  - `agent_coordinator.rs` — register, create_session, write_entry, commit, gc, process_merge_queue
  - `ollama_engine.rs` — embed duration, model name
  - `lite_store.rs` / `knowledge_store.rs` / `postgres_store.rs` — insert/search/get latency
  - `merge_worker.rs` — conflict detection, LLM merge duration
  - `rag_pipeline.rs` — retrieval + augmentation spans

**D87: Config hot-reload + AppState changes**

Add to `AppState` in `rest.rs`:
```rust
pub config: Arc<RwLock<CorviaConfig>>,  // hot-reloadable
pub config_path: PathBuf,               // for write-back
```

`ops::config_set()` writes TOML + swaps in-memory config via `RwLock`.

| Config Section | Hot-reloadable? | Rationale |
|----------------|----------------|-----------|
| `agent_lifecycle` | Yes | Thresholds, read on each GC sweep |
| `merge` | Yes | Read per merge operation |
| `rag` | Yes | Read per RAG query |
| `chunking` | Yes | Read per ingest |
| `reasoning` | Yes | Read per reason invocation |
| `adapters` | Yes | Re-discover on next call |
| `storage` | **No** | Store connection at startup |
| `server` | **No** | Listener bound at startup |
| `embedding` | **No** | Engine constructed at startup |
| `project` | **No** | Scope used throughout |

`config_set` on structural (non-hot-reloadable) sections returns error with "requires restart" message.

**D89: Telemetry config section**

Add `TelemetryConfig` to `CorviaConfig` (`#[serde(default)]` for backward compat):
```toml
[telemetry]
exporter = "stdout"        # stdout | file | otlp
otlp_endpoint = ""         # only when exporter = "otlp"
log_format = "text"        # text | json
metrics_enabled = true
```

Wire `init_telemetry()` into `cmd_serve`.

**Gate:** Spans visible in logs at `RUST_LOG=debug`, config read/write works programmatically.

### Phase 3: MCP Control Plane

**D83: Tiered safety model**

| Tier | Level | Behavior |
|------|-------|----------|
| `ReadOnly` | Auto-approved | Executes immediately, no confirmation |
| `LowRisk` | Single confirmation | Without `_meta.confirmed`: returns preview + `"confirmation_required": true`. With `_meta.confirmed: true`: executes |
| `MediumRisk` | Confirmation + dry-run | Same as LowRisk, plus accepts `dry_run: true` for preview without mutation |
| `HighRisk` | Double confirmation + audit | Same as MediumRisk with additional safeguards (future Tier 4 tools) |

Implementation:
- `ToolTier` enum in `mcp.rs`
- Each tool definition gets tier in `annotations` metadata
- Dispatch checks `_meta.confirmed` before executing Tier 2+ operations
- No MCP protocol upgrade needed — works within `2024-11-05` spec
- Upgradable to MCP Elicitation when protocol bumps to `2025-06-18`

**D84: Tier 1 MCP tools (read-only, auto-approved) — 5 tools**

| Tool | Calls | Returns |
|------|-------|---------|
| `corvia_system_status` | `ops::system_status()` | Entry counts, agents, sessions, queue depth, adapters |
| `corvia_config_get` | `ops::config_get()` | Config section as JSON (or full config if no section) |
| `corvia_adapters_list` | `ops::adapters_list()` | Discovered adapters with metadata |
| `corvia_agents_list` | `ops::agents_list()` | All registered agents with status |
| `corvia_merge_queue` | `ops::merge_queue_status()` | Queue depth + top entries |

**D85: Tier 2 MCP tools (low-risk mutation) — 3 tools**

| Tool | Calls | Confirmation |
|------|-------|-------------|
| `corvia_config_set` | `ops::config_set()` | Shows diff of what changes, requires `_meta.confirmed` |
| `corvia_gc_run` | `ops::gc_run()` | Shows what would be cleaned, requires `_meta.confirmed` |
| `corvia_rebuild_index` | `ops::rebuild_index()` | Shows current index state, requires `_meta.confirmed` |

**D86: Tier 3 MCP tools (medium-risk mutation) — 2 tools**

| Tool | Calls | Confirmation |
|------|-------|-------------|
| `corvia_agent_suspend` | `ops::agent_suspend()` | Supports `dry_run`, requires `_meta.confirmed` |
| `corvia_merge_retry` | `ops::merge_retry()` | Supports `dry_run`, requires `_meta.confirmed` |

**Gate:** `tools/list` returns 18 tools (8 existing + 10 new), all existing tools unchanged.

### Phase 4: CLI Observability

**D88: `corvia status --metrics`**
- Add `--metrics` flag to `status` command
- Calls `ops::system_status()` (same function as MCP `corvia_system_status`) plus:
  - Cumulative metrics from Redb counters (entries committed/merged/rejected)
  - Adapter status
  - Config summary (store type, inference provider, telemetry exporter)
- Plain text output (no TUI for v1)

**Grafana/Docker Compose dashboards deferred to M5** — keeps M4 scope manageable, M5 already has VS Code extension which pairs naturally with dashboards.

**Gate:** Full test suite passes (364+ existing + ~42 new).

---

## Shared Kernel Pattern (No Duplicated Code)

```
CLI cmd_status ────────┐
                       ├──→ ops::system_status() ──→ store/coordinator/registry
MCP corvia_system_status ─┘

CLI cmd_rebuild ───────┐
                       ├──→ ops::rebuild_index() ──→ LiteStore
MCP corvia_rebuild_index ─┘

CLI cmd_agent list ────┐
                       ├──→ ops::agents_list() ──→ AgentRegistry
MCP corvia_agents_list ─┘
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `crates/corvia-telemetry/Cargo.toml` | New crate manifest |
| `crates/corvia-telemetry/src/lib.rs` | Telemetry init, span constants, metrics |
| `crates/corvia-kernel/src/ops.rs` | Shared kernel operations |

## Files to Modify

| File | Changes |
|------|---------|
| `Cargo.toml` (workspace) | Add `corvia-telemetry` member + `opentelemetry` deps |
| `crates/corvia-common/src/config.rs` | Add `TelemetryConfig`, `telemetry` field |
| `crates/corvia-kernel/src/lib.rs` | Add `pub mod ops;` |
| `crates/corvia-kernel/Cargo.toml` | Add `corvia-telemetry` dep |
| `crates/corvia-server/src/rest.rs` | Add `config` + `config_path` to `AppState` |
| `crates/corvia-server/src/mcp.rs` | Add 10 tool definitions + dispatch + handlers + safety tier |
| `crates/corvia-server/Cargo.toml` | Add `corvia-telemetry` dep |
| `crates/corvia-cli/src/main.rs` | Refactor to use `ops::*`, add `--metrics`, wire telemetry init |
| `crates/corvia-cli/Cargo.toml` | Add `corvia-telemetry` dep |
| `crates/corvia-kernel/src/agent_coordinator.rs` | Add `#[tracing::instrument]` |
| `crates/corvia-kernel/src/merge_worker.rs` | Add spans |
| `crates/corvia-kernel/src/lite_store.rs` | Add spans |
| `crates/corvia-kernel/src/ollama_engine.rs` | Add spans |
| `crates/corvia-kernel/src/rag_pipeline.rs` | Add spans |

---

## Dependency Graph

```
D80 (telemetry crate)  ──┐
                          ├──→ D81 (kernel instrumentation) ──→ D89 (OTLP config)
D82 (ops.rs extraction) ─┤
                          ├──→ D84 (Tier 1 tools)
D87 (config hot-reload) ──┤
                          ├──→ D85 (Tier 2 tools, needs D82 + D87)
D83 (safety model) ───────┤
                          ├──→ D86 (Tier 3 tools, needs D82 + D83)
                          └──→ D88 (CLI --metrics, needs D81 + D82)
```

Critical path: D80 + D82 (parallel) → D83 + D87 (parallel) → D84 → D85 → D86 → D88

---

## Verification

1. `cargo test --workspace` — all 364+ existing tests pass
2. `cargo test -p corvia-telemetry` — new telemetry crate tests pass
3. `cargo test -p corvia-kernel -- ops` — new ops module tests pass
4. `cargo test -p corvia-server -- mcp` — all MCP tests pass (existing 8 + 10 new tools)
5. `corvia status` — unchanged output (backward compat)
6. `corvia status --metrics` — enhanced output with metrics
7. `corvia serve --mcp` → MCP `tools/list` returns 18 tools
8. MCP `corvia_system_status` returns same data as `corvia status`
9. MCP `corvia_config_get` → `corvia_config_set` round-trip works
10. Tier 2 tool without `_meta.confirmed` returns confirmation prompt, not mutation

---

## Impact on Other Milestones

### M5 (revised)
- **Gains:** Grafana/Docker Compose dashboards (moved from M4)
- **Gains:** Pre-built dashboards feed from telemetry data wired in M4
- **VS Code extension** can now use both existing 8 knowledge tools AND 10 new control tools

### M6 (unchanged)
- Evals can now measure control plane latency and safety model effectiveness

### M7 (unchanged)
- OSS launch includes the control plane — differentiator vs competition
- A2A Agent Card publishing becomes the first step toward agent-to-agent control

---

## What Works After M4

- **LLMs can observe:** `corvia_system_status`, `corvia_merge_queue`, `corvia_agents_list` give full visibility
- **LLMs can control:** `corvia_config_set`, `corvia_gc_run`, `corvia_rebuild_index` let agents tune and maintain the system
- **LLMs can manage:** `corvia_agent_suspend`, `corvia_merge_retry` handle operational issues
- **Safety is built in:** Tiered confirmation model prevents accidental mutations
- **CLI stays powerful:** `corvia status --metrics` gives humans the same data
- **No code duplication:** Both interfaces call the same `ops::*` kernel functions

---

*Created: 2026-03-04*
*Supersedes: M4 section in milestone-revision-notes.md (lines 95-123)*
*References: 2026-03-04-m4-control-plane-research.md*
