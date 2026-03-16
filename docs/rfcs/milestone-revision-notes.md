# Milestone Revision Notes (Post-Hybrid D35-D44)

> **Status:** Reference

Status: **Revised** — M2-M7 updated for hybrid architecture, pending formalization into brainstorm doc

## Impact Summary

| Milestone | Revision | Key Changes |
|-----------|----------|-------------|
| **M2** | **Major** | Staging hybrid (D43), app-level RBAC (D39), visibility modes |
| **M3** | **Major** | petgraph (D37) + Redb temporal (D38), Level 2-3 reasoning (D44), `corvia upgrade` |
| **M4** | **Major** | Two-tier observability: CLI metrics (LiteStore) + Grafana (FullStore) |
| **M5** | Minor | Auto-detect backend, graph visualization panel |
| **M6** | Minor | Extend Introspect pipeline, add Level 2-3 evals |
| **M7** | Medium | Two-tier onboarding (LiteStore default, FullStore upgrade) |

---

## Revised Milestones

### M2: "Multiple agents, no collisions"

**Depends on**: M1.2
**LinkedIn**: "What happens when 10 AI agents write to the same knowledge base simultaneously? Here's how I solved it"

**What changed from original**:
- Isolation: application-level RBAC via Redb key prefixes + file dirs (D39), NOT SurrealDB namespace auth
- Write path: staging hybrid with git branches + shared HNSW (D43), NOT Redb-only logical isolation
- Merge worker: uses configured InferenceEngine (Ollama or vLLM), NOT hardcoded vLLM
- Agent scope: knowledge workers only, Levels 0-1 (D44)

**Deliverables**:
1. `AgentCoordinator` — multi-layer identity (D45 Part 1), agent registration, session lifecycle
2. Atomic write path (D45 Part 4): JSON → embed → atomic Redb transaction (metadata + HNSW). No ghost vectors.
3. Staging hybrid flow (D43): staging dirs + git branches + shared HNSW + idempotent recovery scan
4. Commit + merge pipeline: idempotent commit steps → merge queue → merge worker (conflict detection + LLM merge)
5. Merge worker — semantic similarity conflict detection + LLM-assisted merge + exponential retry on failure
6. Context builder — RBAC filtering via Redb metadata (D39)
7. MCP server (`corvia serve --mcp`) — `corvia_search`, `corvia_write`, `corvia_history`, `corvia_agent_status`
   - Multi-layer identity: `_meta.agent_id` for write access, `clientInfo` fallback for read-only
8. REST endpoints for agent session CRUD + recovery
9. Visibility modes: `own` (main + own pending), `all` (main + all agents'), `explicit` (main + named agents')
10. Crash recovery — reconnection protocol, recovery scan, resume/commit/rollback orphaned sessions (D45 Part 5)
11. Garbage collection — configurable thresholds for orphaned sessions, closed records, inactive agents (D45 Part 6)

**What works after M2**: Multiple agents open sessions, write to staging dirs, merge into main knowledge base. Cross-agent search works instantly via shared HNSW. `git log --all` shows every agent's contributions. MCP server exposes knowledge to any orchestrator (CrewAI, LangGraph, etc.). Agents survive crashes and reconnect without data loss.

---

### M3: "Knowledge that remembers — and reasons"

**Depends on**: M2
**LinkedIn**: "My AI tool remembers *when*, reasons *why*, and I proved it by pointing it at itself"

**What changed from original**:
- Graph: petgraph in-memory overlay + Redb edge persistence (D37), NOT SurrealDB graph
- Temporal: Redb compound-key range scans (D38), NOT SurrealDB temporal queries
- New traits: `TemporalStore`, `GraphStore` — Redb impl (LiteStore), SurrealDB impl (FullStore)
- New capabilities: Level 2 reasoning + Level 3 structured findings (D44)
- New command: `corvia upgrade` (LiteStore → FullStore migration)
- Updated command: `corvia rebuild` now also reconstructs petgraph + temporal indexes

**Deliverables**:
1. `TemporalStore` trait + Redb implementation
   - Compound keys: `{scope}:{valid_from}:{entry_id}`
   - Methods: `as_of(timestamp)`, `history(entry_id)`, `evolution(scope, time_range)`
   - Bi-temporal: `recorded_at` (when stored) + `valid_from`/`valid_to` (when true)
2. `GraphStore` trait + petgraph implementation
   - Redb edge persistence: key `{from_id}:{relation}:{to_id}`, value: edge metadata
   - In-memory petgraph rebuilt on startup from Redb edges
   - Operations: BFS, DFS, shortest path, cycle detection, subgraph extraction
   - Relations: `depends_on`, `implements`, `supersedes`, `relates_to`, `contradicts`
3. `corvia upgrade` — migrate LiteStore knowledge files → SurrealDB
   - Reads all `.corvia/knowledge/**/*.json`
   - Bulk-inserts into SurrealDB with embeddings
   - Updates config from `store = "lite"` to `store = "surrealdb"`
   - Verifies count matches post-migration
4. `corvia rebuild` — reconstruct HNSW + petgraph + temporal indexes from JSON files
5. Level 2: Knowledge Reasoner
   - Cross-reference knowledge entries across scopes
   - Detect knowledge gaps (referenced concepts with no entries)
   - Surface contradictions (entries with conflicting claims about same topic)
   - Cross-source dependency tracking (D18)
6. Level 3: Structured Findings (revised — no diff generation)
   - Pattern detection: inconsistencies, stale knowledge, missing coverage
   - Findings stored as knowledge entries (type: `finding`)
   - Fields: `finding_type`, `target`, `rationale`, `confidence`, `source_knowledge[]`
   - Coding agents query findings via MCP, generate their own diffs
   - Corvia tracks which findings were acted on (acceptance rate)
7. Self-dogfooding: Corvia's own decisions (D1-D44) as searchable knowledge entries with provenance chains

**What works after M3**: "What did we know about auth 3 months ago?" works. Graph traversal shows how concepts connect. Pattern detection finds inconsistencies. `corvia upgrade` migrates to FullStore when ready. Corvia documents its own development.

---

### M4: "See what's happening inside"

**Depends on**: M3
**LinkedIn**: "OpenTelemetry for AI agents — instrumenting a multi-agent knowledge system end to end"

**What changed from original**:
- Two-tier observability: LiteStore (zero-Docker) needs CLI-based metrics, not just Grafana
- OTLP export targets: stdout/file (LiteStore) + OTLP endpoint (FullStore/Grafana)

**Deliverables**:
1. `corvia-telemetry` crate — OpenTelemetry spans across all kernel subsystems
   - Embedding pipeline: embed duration, batch size, model, token count
   - Knowledge store: insert/search/get latency, result count
   - Agent coordinator: session count, merge queue depth, conflict rate
   - Context builder: query build time, filter selectivity
2. **LiteStore tier**: `corvia status --metrics`
   - Ingest rate (entries/min), search latency (P50/P95/P99)
   - Active agent sessions, merge queue depth, conflict rate
   - HNSW index size, knowledge entry count by scope
   - GC activity: orphaned sessions cleaned, entries discarded, agents suspended
   - Crash recovery events: sessions resumed, committed, rolled back
   - Structured JSON logs to file
3. **FullStore tier**: `corvia serve --grafana`
   - Docker Compose stack: Grafana + Tempo + Prometheus
   - Pre-built dashboards: pipeline overview, agent activity, merge worker, search performance, GC/recovery
4. OTLP exporter with configurable target (stdout, file, OTLP endpoint)

**What works after M4**: LiteStore users get `corvia status --metrics` in terminal. FullStore users get Grafana dashboards. Both have structured logs and OpenTelemetry traces.

---

### M5: "A window into the mind"

**Depends on**: M4
**LinkedIn**: "Corvia now has a VS Code extension — a visual window into your code's memory"

**What changed from original**: Minimal — MCP is backend-agnostic. Two additions.

**Deliverables** (mostly unchanged):
1. `corvia-extension` repo — VS Code extension connecting via MCP
2. Dashboard panels: pipeline flow, namespace map, merge activity, temporal health, cost tracker
3. Search panel with relevance scores
4. Timeline view (temporal history)
5. Agent status panel
6. **New**: Auto-detect LiteStore vs FullStore from config (adjust available panels)
7. **New**: Graph visualization panel — petgraph-based concept map (if M3 graph available)

**What works after M5**: VS Code users get a visual window into Corvia. Works with both LiteStore and FullStore.

---

### M6: "How good is it?"

**Depends on**: M5
**LinkedIn**: "Building evals for AI agent memory — what I measured and what surprised me"

**What changed from original**: Slimmed down — extend Introspect pipeline (D30-D31) rather than building separate framework.

**Deliverables**:
1. **Extend Introspect** — canonical queries with expected results + automated scoring
   - Retrieval precision/recall at K
   - Temporal decay tests on self-ingested knowledge
   - Graph connectivity tests (expected edges present)
2. **Level 2-3 evals** (new)
   - Precision/recall on pattern detection (planted inconsistencies)
   - Finding quality scoring (confidence calibration)
   - Acceptance rate tracking for acted-on findings
3. **With-vs-without comparison** — agent output quality with and without Corvia context
4. Cost tracking — embedding costs, LLM merge costs, storage costs per scope
5. Benchmarks — publish numbers against Letta, Mem0 baselines
   - Search latency, memory usage, ingest throughput
   - Quality: retrieval relevance, temporal accuracy, graph completeness

**What works after M6**: Measurable quality metrics with numbers, not claims. Automated regression testing via extended Introspect.

---

### M7: "Try it yourself"

**Depends on**: M6
**LinkedIn**: "Introducing Corvia — open source code memory. AI agents that understand your codebase and get smarter over time. Try it today."

**What changed from original**: Two-tier onboarding story (LiteStore default, FullStore upgrade).

**Deliverables**:
1. Published crates: `cargo install corvia`
2. **Two installation paths**:
   - "5-minute quickstart" — `cargo install corvia && corvia init && corvia demo` (LiteStore, zero Docker)
   - "Production setup" — `corvia init --full` (SurrealDB + vLLM Docker stack)
3. **Migration docs**: `corvia upgrade` path from LiteStore → FullStore
4. VS Code Marketplace listing
5. Docs site with:
   - Architecture decision records (D1-D44) as navigable docs
   - API reference (REST + MCP)
   - Adapter authoring guide
   - Contributing guide
6. GitHub org polished (README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT)
7. One cloud deployment (free tier, FullStore)

**What works after M7**: Anyone can install and use Corvia. Zero-friction local start, upgrade path to production.

---

## Revised Dependency Map

```
M1.1 ✓ (Kernel + SurrealDB + Introspect)
  │
M1.2 (LiteStore + Ollama + zero-Docker demo)
  │
M2 (Agent Coordination + Staging Hybrid + MCP + REST)
  │       Level 0-1: Passive Index + Knowledge Writer
  │
M3 (Temporal + Graph + Reasoning + corvia upgrade)
  │       Level 2-3: Knowledge Reasoner + Structured Findings
  │
M4 (Observability — two-tier: CLI metrics + optional Grafana)
  │
M5 (VS Code Extension via MCP)
  │
M6 (Evals — extend Introspect + benchmarks)
  │
M7 (OSS Launch — two-tier onboarding)
```

---

## Preliminary Decisions

### D43: Agent Write Isolation — Staging Hybrid (Git Branches + Shared HNSW)

**Status**: Preliminary — to be formalized when M2 implementation begins

**Decision**: Agent write isolation uses a lightweight staging directory + git branches, NOT full git worktrees or Redb-only logical isolation. The HNSW index and Redb metadata remain shared (single instance). Git branches provide audit trail.

**Write path**:
1. Agent starts session → gets staging dir `.corvia/staging/{agent-id}/` + git branch `{agent-id}/session-{uuid}`
2. Agent writes knowledge → embeds via Ollama → writes JSON to staging dir → inserts vector into shared HNSW immediately (tagged as `{agent-id}:pending` in Redb metadata)
3. Agent commits → `git commit` on agent branch → entries enter merge queue
4. Merge worker processes → no conflict: auto-merge to main. Conflict: LLM-assisted merge (re-embed merged content)
5. Post-merge → moves JSON from staging to `.corvia/knowledge/`, retags Redb metadata as `main`, HNSW vectors stay in place (or replaced if merged). Git merge agent branch into main. Cleanup staging dir + delete branch.

**Read path**:
- Agent searches → shared HNSW index → Redb metadata filter (scope, workstream, permissions) → results include main knowledge + agent's own pending
- Visibility configurable: `own` (main + own pending), `all` (main + all agents' pending), `explicit` (main + named agents' pending)

**Embedding flow for conflicts**:
- Non-conflicting writes: embedding stays as-is in HNSW. No re-embedding.
- Conflicting writes: merge worker produces new merged content → re-embeds → replaces conflicting vectors in HNSW → writes new merged JSON.

**Why not full git worktrees**: HNSW binary duplication (~50-100MB per agent), knowledge file duplication, cross-agent search requires shared index.
**Why not Redb-only**: No git history per agent, rollback requires scanning key prefixes vs simple `git branch -D`.

---

### D44: Agent Capability Scope — Levels 0-3 (Memory + Reasoning + Findings)

**Status**: Preliminary — to be formalized when M2/M3 implementation begins

**Decision**: Corvia targets capability Levels 0-3. Code writing (Level 4) is explicitly out of scope.

| Level | Name | What It Does | Milestone |
|-------|------|-------------|-----------|
| **0** | Passive Index | Ingest, embed, search. Read-only over source code. | M1 (done) |
| **1** | Knowledge Writer | Agents create/update knowledge entries, merge conflicts. | M2 |
| **2** | Knowledge Reasoner | Cross-reference knowledge, detect gaps, surface contradictions. | M3 |
| **3** | Structured Findings | Detect patterns, post findings as knowledge entries. No diff generation. | M3 |
| **4** | Code Writer | Directly modify source code. **OUT OF SCOPE.** | Separate project |

**Level 3 revised scope**: Detect patterns and post structured findings — no diff generation. Corvia is a "suggestion board": it posts findings, external coding agents pick them up via MCP, generate their own diffs. Corvia tracks acceptance rate. No orchestrator overlap.

**Expanded git adapter vision**:
- Current `corvia-adapter-git`: reads source files, tree-sitter chunks, extracts HEAD SHA
- Expanded: also ingest git commits, diffs, commit messages as knowledge entries
- Future `corvia-adapter-github`: PRs, issues, CI results, review comments

---

### D45: Agent Identity & Session Recovery Model (Multi-Layer)

**Status**: Preliminary — to be formalized when M2 implementation begins

**Decision**: Corvia accepts agent identity at multiple layers for ecosystem compatibility. Internally, agents have persistent human-readable IDs. Externally, agents from any framework can connect via MCP with optional identity metadata. Sessions are ephemeral work units that survive crashes.

#### Part 1: Multi-Layer Identity

**Industry context (2026-03-01)**: No standard exists for agent-level identity. MCP identifies the *application* (e.g., "crewai-mcp-adapter"), not the agent within it. IETF `actor_token` draft and W3C `did:wba` are 1-2 years from standardization. A2A Agent Cards are the closest but only for A2A-to-A2A communication. Corvia must work today and adopt standards as they land.

**Identity types**:
```rust
enum AgentIdentity {
    /// Internal: registered Corvia agent with persistent ID
    /// Format: "{scope}::{agent-name}", e.g., "myproject::code-indexer"
    Registered { agent_id: String, api_key: Option<String> },

    /// External via MCP: identified by client info + optional _meta
    /// Used by CrewAI, LangGraph, OpenAI Agents SDK, Claude Code, etc.
    McpClient {
        client_name: String,        // from MCP clientInfo.name
        client_version: String,     // from MCP clientInfo.version
        agent_hint: Option<String>, // from _meta.agent_id in tool calls
    },

    /// External via OAuth actor token (future IETF standard)
    ActorToken { actor_sub: String, principal: String },

    /// Anonymous: no identity, read-only access
    Anonymous,
}
```

**Permission model by identity type**:

| Identity Type | Access Level | Session Type | Write Path |
|---------------|-------------|-------------|------------|
| `Registered` | Full D45 lifecycle | Staging hybrid (D43): staging dir, git branch, HNSW pending, crash recovery | Staging → commit → merge queue |
| `McpClient` + `_meta.agent_id` | Write access, simplified | Lightweight session: no staging dir, entries go direct to merge queue | Direct to merge queue |
| `McpClient` (no `_meta`) | Read-only | No session | N/A |
| `ActorToken` | Write access (future) | Mapped to Registered or McpClient depending on registration | TBD when IETF standardizes |
| `Anonymous` | Read-only, rate-limited | No session | N/A |

**MCP `_meta` convention** (works today, zero spec changes needed):
```json
{
  "method": "tools/call",
  "params": {
    "name": "corvia_write",
    "arguments": { "content": "...", "scope": "myproject" },
    "_meta": {
      "agent_id": "code-indexer",
      "agent_name": "Code Indexer",
      "agent_role": "Source code analysis"
    }
  }
}
```
MCP spec allows `_meta` on every tool call. Any framework can pass this with a one-line config. If absent, Corvia falls back to `clientInfo.name`.

**How each framework integrates**:

| Framework | Connects via | Identity Corvia sees | Access |
|-----------|-------------|---------------------|--------|
| Corvia's own agents | REST API + registered ID | `Registered("myproject::code-indexer")` | Full (D43 staging hybrid) |
| CrewAI agent | MCP + `_meta` | `McpClient("crewai", "0.3", Some("financial-advisor"))` | Write (simplified) |
| LangGraph node | MCP (no `_meta`) | `McpClient("langgraph", "0.2", None)` | Read-only |
| Claude Code | MCP + optional `_meta` | `McpClient("claude-code", "1.0", ...)` | Read or write |
| Custom integration | REST API + API key | `Registered("acme::custom-bot")` | Full |
| Anonymous browse | MCP (no auth) | `Anonymous` | Read-only, rate-limited |

**Future standard adoption path**:
- A2A Agent Cards: Corvia publishes its own card at `/.well-known/agent-card.json`, accepts remote agent cards for identity
- IETF `actor_token`: extract `sub` claim as agent identity, `principal` as authorizing user
- W3C `did:wba`: accept DID as agent identifier, verify via DID document

#### Part 2: Internal Identity Scheme (Registered Agents)

**ID format**:
```
Agent ID:    "{scope}::{agent-name}"     e.g., "myproject::code-indexer"
Session ID:  "{agent-id}/sess-{uuid}"    e.g., "myproject::code-indexer/sess-550e8400"
```
- `::` separator distinguishes from 5-segment namespace (which uses `:`)
- Agent names are human-readable — visible in `git log`, `corvia status`, MCP queries
- Session ID embeds agent ID for traceability

**Redb agent registry** (`agents` table):
- Key: `agent_id` (string)
- Value: `AgentRecord` — `agent_id`, `display_name`, `identity_type`, `registered_at`, `permissions`, `last_seen`, `status`
- Permissions: `ReadOnly` (Level 0), `ReadWrite { scopes }` (Level 0-1), `Admin` (all scopes)
- Status: `Active`, `Suspended`, `Deregistered`

#### Part 3: Knowledge Ownership

- `KnowledgeEntry` extended with: `agent_id: Option<String>`, `session_id: Option<String>`, `status: EntryStatus`
- `EntryStatus`: `Pending` (in staging) → `Committed` (in merge queue) → `Merged` (in main) or `Rejected`
- For `McpClient` agents with `_meta`: `agent_id` = `"{client_name}::{agent_hint}"` (e.g., `"crewai::financial-advisor"`)
- For `McpClient` without `_meta`: `agent_id` = `"{client_name}"` (e.g., `"langgraph"`)
- Query "my memory": `agent_id == self AND status == Merged` for contributions, `status == Pending` for in-flight

#### Part 4: Atomic Write Path (Revised from D43)

The original D43 write path (JSON → embed → HNSW → Redb) has a crash-safety gap: a vector inserted into HNSW without Redb metadata creates a ghost entry. Revised path eliminates this.

**Revised write path** (per entry):
```
Step 1: Write JSON file to staging dir (filesystem)
        → Cheap, recoverable. Orphaned file is harmless.
Step 2: Embed content via Ollama/vLLM (network call)
        → Idempotent: same content always produces same vector.
        → If crash here: orphaned staging file, re-embed on recovery.
Step 3: Redb transaction (ATOMIC) {
          - Write entry metadata (agent_id, session_id, status: Pending)
          - Write HNSW mapping (uuid ↔ hnsw_id)
          - Insert vector into HNSW index
        }
        → All-or-nothing. No partial state possible.
```

**Why this is crash-safe**: Step 1 alone = orphaned file (recovery: re-embed + insert). Steps 1+2 without 3 = orphaned file + lost embedding (recovery: same — re-embed is idempotent). Step 3 is atomic via Redb transaction. No inconsistent state is possible.

**Recovery scan** (runs on session resume):
```
For each JSON file in staging dir:
  Check Redb: does entry metadata exist for this file's entry_id?
  YES → already inserted, skip
  NO  → re-embed content → atomic Redb transaction (insert metadata + HNSW vector)
```

#### Part 5: Session Lifecycle & Crash Recovery (Registered Agents Only)

**State machine**:
```
Created → Active (heartbeat keeps alive)
  Active → Committing → Merging → Closed (happy path)
  Active → Stale (heartbeat timeout, default 5min)
    Stale → Orphaned (grace period, default 20min)
      Orphaned → Recoverable (has pending work on disk)
```

**Redb session table**: `session_id`, `agent_id`, `created_at`, `last_heartbeat`, `status`, `git_branch`, `staging_dir`, `entries_written`, `entries_merged`

**Heartbeat**: Agent sends every 30s → updates `last_heartbeat` in Redb. Network blip → session goes Stale after 5min → next successful heartbeat resets to Active. Self-healing.

**Reconnection protocol** (the full dance):
```
Agent process starts (fresh or after crash)
  │
  ├─ Agent knows its agent_id (from config, env var, or hardcoded)
  │   └─ POST /agents/{agent_id}/sessions { action: "connect" }
  │
  └─ Agent is brand new
      └─ POST /agents { name: "code-indexer", scope: "myproject" }
         → Returns agent_id, then POST /sessions as above
  │
  ▼
Corvia responds:
{
  "agent_id": "myproject::code-indexer",
  "recoverable_sessions": [
    {
      "session_id": "myproject::code-indexer/sess-550e8400",
      "status": "orphaned",
      "last_heartbeat": "2026-03-01T10:05:30Z",
      "entries_pending": 3,
      "staging_files": 3,
      "git_branch": "code-indexer/sess-550e8400",
      "recovery_options": ["resume", "commit", "rollback"]
    }
  ],
  "active_sessions": []
}
  │
  ▼
Agent chooses:
  RESUME  → recovery scan (re-embed orphaned files) → status: Active → continue writing
  COMMIT  → recovery scan → commit flow → entries to merge queue → new session available
  ROLLBACK → delete staging dir + git branch + pending HNSW vectors + Redb metadata → Closed
  IGNORE  → create new session, old one stays Orphaned (GC'd later)
```

**Key design point**: Agent remembers nothing except its `agent_id`. All state is on Corvia's side (Redb, staging dir, git branch, HNSW). Even a completely stateless LLM agent can reconnect and Corvia tells it everything.

**Commit flow with failure points**:
```
POST /sessions/{session_id}/commit
  Step 1: Session status → Committing (Redb)
  Step 2: git add staging files + git commit on agent branch
  Step 3: All entry statuses → Committed (Redb)
  Step 4: Entries enter merge queue (Redb)
  Step 5: Session status → Merging (Redb)
```
Every step is idempotent. If crash during commit, recovery re-runs from step 1 — already-completed steps are no-ops (`git commit` on committed branch = no-op, Redb status update on already-updated entry = no-op).

**Merge worker flow**:
```
Merge worker picks up entries from queue:
  Step 6: Check conflicts (semantic similarity vs existing entries)
  Step 7a: No conflict → move JSON staging → .corvia/knowledge/,
           retag Redb (status: Merged), git merge branch → main
  Step 7b: Conflict → LLM merge → re-embed → replace vector → write merged JSON
  Step 8: Session status → Closed (Redb)
  Step 9: Cleanup staging dir, delete git branch
```
If merge worker crashes: queue entries persist in Redb, worker restarts and replays. Each merge operation is idempotent (moving an already-moved file = no-op, merging an already-merged branch = no-op).

If LLM merge fails (Ollama down): entry marked `merge_failed`, retry with exponential backoff (max 3). After max retries: stays in queue, `corvia status` shows failed merges, human intervention required. No data lost.

**Corvia process crash**: Redb is ACID, HNSW uses mmap (persisted), staging files on filesystem — all survive. On restart: resume monitoring heartbeats, active sessions whose agents died will timeout → Stale → Orphaned. Self-healing.

**McpClient agents**: No crash recovery. Writes go direct to merge queue (no staging dir or git branch). Completed writes are safe in queue. In-flight MCP tool calls are lost per MCP spec. This is acceptable — external agents retry at the orchestrator level.

#### Part 6: Garbage Collection

```
Corvia background GC thread (runs every hour):
  │
  ├─ Orphaned sessions with last_heartbeat > 24h:
  │   → Auto-rollback (delete staging, branch, pending vectors)
  │   → Session status → Closed
  │   → Log: "GC: cleaned orphaned session {id}, discarded {n} entries"
  │
  ├─ Closed sessions with closed_at > 7d:
  │   → Delete Redb session record (metadata cleanup)
  │
  └─ Active agents with last_seen > 30d:
      → Status → Suspended (reactivated on next connect)
```

**Configuration** (`corvia.toml`):
```toml
[agent_lifecycle]
heartbeat_interval = "30s"
stale_timeout = "5m"
orphan_grace_period = "20m"
gc_orphan_after = "24h"
gc_closed_session_after = "7d"
gc_inactive_agent_after = "30d"
```

All GC thresholds configurable. GC events are OpenTelemetry-traced (feeds into M4 observability).

#### Part 7: API Surface

**REST API** (registered agents):
- `POST /agents` — register new agent
- `POST /agents/{agent_id}/sessions` — create or reconnect (returns recoverable sessions)
- `POST /sessions/{session_id}/heartbeat` — keep alive
- `POST /sessions/{session_id}/commit` — commit and merge
- `POST /sessions/{session_id}/rollback` — discard
- `POST /sessions/{session_id}/recover` — resume/commit/rollback orphaned session
- `GET /agents/{agent_id}/knowledge` — agent's contribution summary
- `GET /sessions/{session_id}/state` — session's current entries and status

**MCP tools** (all identity types):
- `corvia_search` — search knowledge (all identity types)
- `corvia_write` — write knowledge entry (Registered + McpClient with `_meta`)
- `corvia_history` — temporal history query (all identity types)
- `corvia_agent_status` — agent's own contribution summary (Registered + McpClient with `_meta`)

#### Part 8: Observability Contract

Observability is designed into D45 at lifecycle definition time, not bolted on in M4. Every lifecycle phase emits telemetry. M4 configures where it goes (stdout, file, OTLP, Grafana). M2/M3 code emits it from day one.

**Layer 1: OpenTelemetry Spans** (request-scoped, for tracing individual operations)
```
corvia.agent.register           — agent registration
corvia.session.create           — session creation + recovery check
corvia.session.recover          — crash recovery (scan + action)
corvia.entry.write              — single entry write (parent span)
  └─ corvia.entry.embed         — embedding sub-span (latency, model, tokens)
  └─ corvia.entry.insert        — Redb atomic transaction sub-span
corvia.session.commit           — commit flow (git + queue insertion)
corvia.merge.process            — single entry merge (parent span)
  └─ corvia.merge.conflict      — conflict detection sub-span
  └─ corvia.merge.llm_resolve   — LLM merge sub-span (only if conflict)
corvia.gc.run                   — GC sweep
corvia.search                   — search query (latency, result count, scope)
```

**Layer 2: OpenTelemetry Metrics** (aggregates, for dashboards and alerting)
```
# Gauges (current state)
corvia.agents.active                — currently active agents
corvia.agents.by_identity_type      — count per type (Registered, McpClient, Anonymous)
corvia.sessions.active              — currently active sessions
corvia.sessions.stale               — stale sessions awaiting recovery
corvia.sessions.orphaned            — orphaned sessions awaiting GC
corvia.entries.pending              — entries in staging (not yet committed)
corvia.merge.queue_depth            — current merge queue depth

# Counters (cumulative)
corvia.entries.committed            — total entries committed
corvia.entries.merged               — total entries merged to main
corvia.entries.rejected             — total entries rejected by merge
corvia.gc.sessions_cleaned          — total sessions GC'd
corvia.gc.entries_discarded         — total entries discarded by GC
corvia.merge.conflicts              — total conflicts detected
corvia.merge.llm_invocations        — total LLM merge calls

# Histograms (distributions)
corvia.embed.duration_ms            — embedding latency per entry
corvia.search.duration_ms           — search latency
corvia.search.results_count         — results per search
corvia.merge.duration_ms            — merge latency per entry
corvia.merge.llm_duration_ms        — LLM merge latency (conflict resolution)
corvia.heartbeat.delta_ms           — time between heartbeats (detects slow agents)
corvia.merge.conflict_rate          — conflicts per merge attempt
```

**Layer 3: Structured Log Events** (audit trail and debugging)
```
INFO  agent_registered        { agent_id, identity_type, permissions }
INFO  session_created         { session_id, agent_id, recoverable_count }
INFO  entry_written           { entry_id, agent_id, session_id, embed_ms, insert_ms }
WARN  session_stale           { session_id, agent_id, last_heartbeat, entries_at_risk }
WARN  session_orphaned        { session_id, agent_id, entries_pending }
INFO  session_recovered       { session_id, action: resume|commit|rollback, entries_affected }
INFO  session_committed       { session_id, entries_committed, merge_queue_depth }
INFO  entry_merged            { entry_id, agent_id, had_conflict: bool, merge_ms }
WARN  merge_failed            { entry_id, agent_id, error, retry_count }
ERROR merge_exhausted         { entry_id, agent_id, max_retries_reached }
INFO  gc_sweep                { sessions_cleaned, entries_discarded, agents_suspended, duration_ms }
INFO  mcp_client_connected    { client_name, client_version, agent_hint, identity_type }
WARN  mcp_write_denied        { client_name, reason: "no _meta.agent_id" }
```

**Implementation pattern** (M2 code emits, M4 configures export):
```rust
// Every D45 function emits telemetry at construction time
async fn write_entry(&self, entry: &KnowledgeEntry, session: &Session) -> Result<()> {
    let span = tracing::info_span!("corvia.entry.write",
        entry_id = %entry.id,
        agent_id = %session.agent_id,
        session_id = %session.session_id,
    );
    let _guard = span.enter();

    // Step 1: staging file
    self.write_staging_file(entry, session).await?;

    // Step 2: embed (sub-span with timing)
    let embedding = {
        let _s = tracing::info_span!("corvia.entry.embed").entered();
        let start = std::time::Instant::now();
        let result = self.engine.embed(&entry.content).await?;
        metrics::histogram!("corvia.embed.duration_ms").record(start.elapsed().as_millis() as f64);
        result
    };

    // Step 3: atomic insert (sub-span with timing)
    {
        let _s = tracing::info_span!("corvia.entry.insert").entered();
        self.store.atomic_insert(entry, &embedding).await?;
    }

    metrics::counter!("corvia.entries.pending").increment(1);
    tracing::info!(entry_id = %entry.id, agent_id = %session.agent_id, "entry_written");
    Ok(())
}
```

**What M4 adds on top**: M4 does NOT add instrumentation — it configures the export pipeline:
- LiteStore: `tracing-subscriber` → stdout/file, `corvia status --metrics` reads Redb counters
- FullStore: OTLP exporter → Grafana Tempo (traces) + Prometheus (metrics), pre-built dashboards

**Alerting thresholds** (configurable, FullStore only):
- `corvia.sessions.orphaned > 0` for > 1hr → alert
- `corvia.merge.queue_depth > 100` → alert
- `corvia.merge.conflict_rate > 0.3` → alert (30%+ conflicts suggests stale knowledge)
- `corvia.embed.duration_ms P99 > 5000` → alert (embedding service degraded)

---

## Open Questions (Per Milestone)

### M2
- ~~MCP server: does it need different behavior for LiteStore vs FullStore?~~ → No, but different behavior per **identity type** (D45)
- Merge worker conflict detection: semantic similarity threshold? Metadata overlap? File path overlap?
- ~~Agent heartbeat/timeout: how long before abandoned session is garbage collected?~~ → Answered by D45 (5min timeout, 15min grace, configurable)
- Should McpClient agents with `_meta` be auto-registered in the agents table? Or kept separate?

### M3
- petgraph overlay: when is the in-memory graph rebuilt? (On startup? On-demand? Background thread?)
- Redb temporal index: exact compound key layout and range scan patterns
- `corvia upgrade` migration: bulk insert strategy for large knowledge bases (batched? streamed?)
- `corvia rebuild`: should it also rebuild petgraph + temporal indexes? (Yes — confirmed)

### M4
- `corvia status --metrics`: TUI dashboard or plain text? (Start with plain text, TUI later)
- OpenTelemetry stdout/file export: is this sufficient for LiteStore, or do users expect more?

### M5-M7
- VS Code extension: different panels available based on LiteStore vs FullStore capabilities?
- Evals: Introspect extension vs separate framework? (Decision: extend Introspect)
- OSS launch: two installation paths documented, but single `cargo install corvia` binary

---

## Research Notes

### Competitive Landscape (2026-03-01)
- **Letta**: Full git worktrees for Context Repos. No graph layer. No HNSW index.
- **Mem0**: $24M funded. Conversational memory, not organizational. No git.
- **Cognee**: $7.5M. Vector+graph (pluggable backends). No git integration.
- **Copilot Memory**: Per-repo, expires 28 days, simple text. "Good enough" threat.
- **Corvia differentiators**: organizational namespaces, git-truth, vector+graph, LLM-merge, zero-Docker

### Inference Serving Landscape (2026-03-01)
- **Ollama**: Dev/local tier. 41 TPS peak (A100). Single-binary, zero-Docker.
- **vLLM**: Production tier. 793 TPS peak (A100). PagedAttention.
- **SGLang**: High-performance production. 16K TPS. RadixAttention.
- **D40 validated**: Ollama default, vLLM opt-in. SGLang compatible via OpenAI API.

### Git Worktree Analysis (2026-03-01)
- Full worktrees rejected: HNSW binary duplication, knowledge file duplication, cross-agent search
- Staging hybrid adopted: lightweight staging dirs + git branches + shared HNSW

### Agent Orchestrator Landscape (2026-03-01)

**Key finding**: No Rust-native orchestrator mature enough to adopt. MCP is the universal integration surface.

**Tier 1 (>20k stars)**: OpenHands (65k), AutoGen/MS Agent FW (50k), MetaGPT (46k), CrewAI (44k), LangGraph (25k)
**Tier 2 (5k-20k)**: OpenAI Agents SDK (19k), PydanticAI (15k), SWE-agent (10.5k)
**Rust emerging**: Rig, OpenFang (4k stars in 4 days), AutoAgents

**Memory gap**: Most orchestrators lack persistent memory. Only Letta and CrewAI have serious memory. Nobody does organizational memory.
**Suggestion pattern gap**: No system produces structured findings as first-class artifacts flowing between agents. Open niche.

**Corvia strategy**: Own agent coordination (M1-M2), MCP as universal adapter, monitor A2A protocol for M5+.

### Agent Identity Standards Landscape (2026-03-01)

**Key finding**: No standard exists for agent-level identity. MCP identifies the application, not the agent. The industry is in a "pre-standard" moment.

**Current state**:
| Protocol | Agent Identity | Session Model | Auth |
|----------|---------------|---------------|------|
| MCP (2025-11-25) | `clientInfo.name` (application-level) | `MCP-Session-Id` header | OAuth 2.1 + CIMD |
| A2A (v0.3) | Agent Card at `/.well-known/agent-card.json` | `contextId` + `taskId` | SecuritySchemes in card |
| OpenAI Agents SDK | `name` field (internal only, not transmitted) | `session_id` (pluggable backends) | Via MCP client |
| CrewAI | `role` + `goal` (internal only) | Flow-level UUID | Via MCP client |
| LangGraph | Graph node name (internal only) | `thread_id` + checkpoints | Via MCP client |

**Emerging standards (not yet ready)**:
- IETF `draft-oauth-ai-agents-on-behalf-of-user`: adds `actor_token` with JWT `sub` claim for agent identity
- IETF `draft-yl-agent-id-requirements`: calls for globally unique Agent Identifiers (AID), format TBD
- W3C `did:wba`: Decentralized Identifiers for agents, 1-2 years from standardization
- OASF/AGNTCY: OCI-based agent registry under Linux Foundation, early stage
- AAIF (Anthropic, Google, MS, OpenAI): governance home, houses MCP + AGENTS.md + goose, no identity spec yet

**Practical reality**: When any framework connects to an MCP server, all agents share one connection. The server cannot distinguish individual agents. MCP's `_meta` extensibility field is the pragmatic bridge — any framework can pass agent identity today with zero spec changes.

**Corvia's approach (D45)**: Accept identity at multiple layers. Registered agents get full D43 staging hybrid. MCP clients with `_meta.agent_id` get simplified write access. MCP clients without `_meta` get read-only. Future-proof for IETF `actor_token`, A2A Agent Cards, and W3C DIDs.

### Demo Alignment Check (2026-03-01)
- Demo design (self-ingest + search REPL) aligns with Levels 0-3 vision
- M1.2 improves demo by removing Docker provisioning entirely
- Future demo (M3+): show Corvia detecting patterns in its own code

---

*Created: 2026-03-01*
*Updated: 2026-03-01 (full M2-M7 revision, D43, D44, D45 8-part design with observability contract, research notes)*
*Next: Formalize into brainstorm doc when ready to implement each milestone*
