# M4 Control Plane Research: Observability + Operational Control via MCP

**Date:** 2026-03-04
**Status:** Research complete, pending formalization into M4 revision
**Decision:** MCP is the control plane protocol. No duplicated code — extend existing MCP server with admin tools.
**Deferred:** A2A protocol review (interesting for inter-agent delegation, revisit at M7+)

---

## Motivation

M4 as originally scoped is pure observability — `corvia status --metrics` and Grafana dashboards. But since Corvia now interacts heavily with LLMs (RAG, merge conflict resolution, reasoning), the LLM needs to **control** operations, not just observe them.

Current surface area has a gap:

| Interface | What it can do | What it can't do |
|-----------|---------------|-----------------|
| **MCP** (8 tools) | Knowledge CRUD, RAG, reasoning | Config mgmt, process control, runtime tuning |
| **REST** (18 endpoints) | Full agent lifecycle | Config mgmt, process control, runtime tuning |
| **CLI** (13+ commands) | Everything + workspace mgmt | Not LLM-accessible without shell |

None of the interfaces expose: config management, process control (trigger GC, flush caches, reload adapters), runtime tuning (change thresholds, switch models, adjust chunking), or merge queue inspection.

---

## Protocol Comparison

| Factor | CLI (shell) | MCP | REST |
|--------|------------|-----|------|
| LLM familiarity | Highest — massive pretraining | Growing — tool discovery is native | Medium — requires schema |
| Discoverability | None — must know commands | Built-in — `tools/list` at runtime | Requires docs/OpenAPI |
| Session state | None — each invocation isolated | Yes — context across calls | Stateless |
| Safety | Low — shell blast radius | High — tiered tools, elicitation | Medium — HTTP auth |
| Async tasks | Poor — blocks on long ops | Native — MCP Tasks primitive | Requires polling/webhooks |
| Composability | Unix pipes, 50 years | Emerging | Well-understood |

**Decision: MCP.** It has built-in discovery, session state, async tasks (Nov 2025 Tasks primitive), and safety primitives (Elicitation for confirmation prompts). The existing MCP server in `corvia-server/src/mcp.rs` is extended — no new server, no duplicated dispatch logic.

---

## Industry Precedents

### AWS Cloud Control API MCP Server
The most mature operational MCP. CRUD on 1,200+ AWS resource types. Key innovation: **token-based workflow chains** that enforce step ordering:
1. `check_environment_variables()` → generates `environment_token`
2. `get_aws_session_info(environment_token)` → generates `credentials_token`
3. `generate_infrastructure_code(credentials_token)` → generates `generated_code_token`
4. `explain(generated_code_token)` → generates `explained_token`
5. Resource CREATE/UPDATE/DELETE requires all preceding tokens

Server-side token validation. LLM cannot skip steps. `--readonly` mode. Double-confirm deletes. Auto-tag resources for audit.

### Kubernetes MCP Servers
Multiple production servers (Go-based, kubectl-based) giving agents full cluster management — pods, deployments, services via natural language.

### Letta MCP Server (Community)
Exposes agent management, memory block CRUD, and tool integration as MCP tools. Operational control of the memory framework itself — closest precedent to what we're building.

### CrewAI MCP Server (Community)
MCP server for running, managing, and creating CrewAI workflows — operational control of the framework.

### MCPX Control Plane (Lunar.dev)
Meta-control: ACL-based tool access per agent, live budgets, request throttling, Prometheus metrics per tool call, immutable audit trail.

**Key finding:** Nobody in the AI memory space ships a first-party admin/control MCP server yet. This is an open gap Corvia can fill.

---

## MCP Spec Features Supporting Control Plane

### Tasks Primitive (Nov 2025 Spec)
- Lifecycle: `working` → `input_required` → `completed` / `failed` / `cancelled`
- Methods: `tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel`
- Two-phase: server returns task handle immediately, result retrieved later
- Tool-level negotiation: `execution.taskSupport` as `"required"`, `"optional"`, `"forbidden"`
- Perfect for: ingestion, index rebuild, migration, bulk operations

### Elicitation (June 2025 Spec)
- Servers pause execution and request structured user input via JSON Schema
- Three actions: Accept, Decline, Cancel
- Official mechanism for confirmation prompts before destructive operations

---

## Proposed Control Tools — Tiered Safety Model

### Tier 1: Read-only (auto-approved)

| Tool | Purpose | Notes |
|------|---------|-------|
| `corvia_config_get` | Read any config section | Returns current `corvia.toml` values as structured JSON |
| `corvia_status` | System health + metrics | Entry counts, index size, active sessions, merge queue depth |
| `corvia_adapters_list` | Discovered adapters | Already exists as CLI `adapters list` |
| `corvia_agents_list` | Registered agents + sessions | Already exists as CLI `agent list` |
| `corvia_merge_queue` | Inspect pending merges | New — exposes merge queue state |

### Tier 2: Low-risk mutation (single confirmation via Elicitation)

| Tool | Purpose | Notes |
|------|---------|-------|
| `corvia_config_set` | Update a config value | Validates schema, writes TOML, triggers hot-reload |
| `corvia_gc_run` | Trigger garbage collection | Runs GC sweep on demand, returns cleanup report |
| `corvia_rebuild_index` | Rebuild HNSW from knowledge files | Async via Tasks primitive |
| `corvia_adapter_reload` | Re-discover adapters from PATH | Refresh adapter registry |

### Tier 3: Medium-risk (confirmation + dry-run preview)

| Tool | Purpose | Notes |
|------|---------|-------|
| `corvia_ingest_trigger` | Start ingestion pipeline | Async Task — returns handle, poll for completion |
| `corvia_migrate` | Backend migration | Supports `--dry-run` preview before execution |
| `corvia_agent_suspend` | Suspend an agent | Prevents new sessions, existing sessions unaffected |
| `corvia_merge_retry` | Retry failed merges | Re-queue entries with `merge_failed` status |

### Tier 4: High-risk (double confirmation + audit log)

| Tool | Purpose | Notes |
|------|---------|-------|
| `corvia_scope_delete` | Delete all knowledge in a scope | Irreversible — requires explicit scope name + confirmation |
| `corvia_agent_deregister` | Permanently remove an agent | Cleans up sessions, staging, branches |
| `corvia_config_reset` | Reset config to defaults | Overwrites current config |

---

## Config Management Design

**Hybrid approach — config file stays canonical, MCP wraps operations:**

1. **`corvia.toml` is the source of truth** — LLMs know TOML well from pretraining
2. **`corvia_config_get`** reads and returns structured sections
3. **`corvia_config_set`** validates against schema → writes TOML → triggers hot-reload
4. **Hot-reload** — config changes take effect without restart (file watch or signal-based)
5. **Git-tracked** — config changes appear in `git diff`, rollback is `git revert`
6. **Validation** — server-side schema validation catches invalid values before they persist

### Config Sections Controllable at Runtime

```toml
# These can be changed without restart:
[agent_lifecycle]        # GC thresholds, heartbeat intervals
[merge]                  # Similarity threshold, max retries
[rag]                    # Limits, graph expansion, alpha weighting
[chunking]               # Token sizes, overlap, strategy
[reasoning]              # Provider, model for LLM-assisted checks

# These require restart (structural changes):
[storage]                # Store type, data dir
[server]                 # Host, port
[embedding]              # Provider, model, dimensions (affects all vectors)
```

---

## Implementation Approach: No Duplicated Code

The existing MCP server (`mcp.rs`) already has:
- JSON-RPC dispatch with method routing
- `AppState` holding store, engine, coordinator, graph, temporal, RAG
- Agent identity extraction from `_meta`
- Error handling with JSON-RPC error codes

**Extend, don't duplicate:**
- Add new tool handlers to the existing `handle_tool_call` match arms
- Add a `ToolSafety` tier enum that gates execution (auto-approve, elicit, double-confirm)
- Config operations share the same `CorviaConfig` from `corvia-common`
- Status/metrics operations read from the same `AgentCoordinator` and store instances
- CLI commands that overlap (e.g., `agent list`, `adapters list`, `status`) should extract shared logic into kernel functions callable from both CLI and MCP

### Shared Kernel Pattern

```
CLI command ──────┐
                  ├──→ kernel function (shared logic) ──→ store/coordinator/config
MCP tool handler ─┘
```

Example: `corvia agent list` (CLI) and `corvia_agents_list` (MCP) both call the same `AgentCoordinator::list_agents()`. No duplicated query logic.

---

## Revised M4 Scope

### M4: "See and steer what's happening inside"

**Depends on:** M3
**LinkedIn pitch:** *"Giving AI agents a dashboard with controls — observability + operational MCP for a multi-agent knowledge system"*

**Deliverables (original — unchanged):**
1. `corvia-telemetry` crate — OpenTelemetry spans across kernel subsystems
2. LiteStore tier: `corvia status --metrics` (CLI)
3. FullStore tier: `corvia serve --grafana` (Docker Compose + dashboards)
4. OTLP exporter with configurable target

**Deliverables (new — control plane):**
5. Tier 1-4 MCP admin tools (see above) added to existing MCP server
6. Config read/write/validate/hot-reload via MCP
7. Async Tasks support for long-running operations (ingest, rebuild, migrate)
8. Safety tier enforcement with Elicitation-based confirmation
9. Audit logging for all control operations (feeds into telemetry)
10. Extract shared kernel functions for CLI/MCP code reuse

---

## Deferred: A2A Protocol

A2A (Google, 150+ orgs, v0.3 July 2025) is the inter-agent complement to MCP:
- **MCP** = how an agent talks to tools
- **A2A** = how agents talk to each other

Relevant for scenarios like: a planning agent delegating "reorganize the knowledge base" to Corvia. Uses Agent Cards at `/.well-known/agent-card.json` for discovery, task lifecycle (`submitted → working → completed`) for coordination, and webhook push notifications for long-running ops.

**Revisit at M7+** — publishing an Agent Card would be the first step. Low effort, high signaling value for the OSS launch.

---

## Research Sources

- AWS Cloud Control API MCP: https://awslabs.github.io/mcp/servers/ccapi-mcp-server
- MCP Tasks Primitive (Nov 2025): https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks
- MCP Elicitation (June 2025): https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation
- Letta MCP Server: https://github.com/oculairmedia/Letta-MCP-server
- CrewAI MCP: https://docs.crewai.com/en/mcp/overview
- MCPX Control Plane (Lunar.dev): https://www.lunar.dev/product/mcp
- Microsoft MCP Gateway: https://github.com/microsoft/mcp-gateway
- Arize Agent Interfaces 2026: https://arize.com/blog/agent-interfaces-in-2026-filesystem-vs-api-vs-database-what-actually-works/
- MCP Security Checklist: https://github.com/slowmist/MCP-Security-Checklist
- IETF CHEQ (human-in-the-loop confirmation): https://www.ietf.org/archive/id/draft-rosenberg-cheq-00.html
- A2A Protocol: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/

*Created: 2026-03-04*
