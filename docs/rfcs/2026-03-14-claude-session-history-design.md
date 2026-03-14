# Claude Session History — Design Spec

**Date:** 2026-03-14
**Status:** Draft
**Scope:** corvia-adapter-claude-sessions · corvia-kernel (workstream filter) · workspace hooks

---

## 1. Overview

This design introduces a personal conversation recording system: every Claude Code
session is captured as a structured history, ingested into a dedicated corvia scope
(`user-history`), and made searchable across sessions. The scope is independent of
product knowledge — it can be deleted, queried, and managed on its own lifecycle.

A secondary pass promotes product-relevant entries (design decisions, research
findings) from `user-history` into the `corvia` product scope via automated
classification. A kernel enhancement (workstream filter) is included as a prerequisite
to enable feature-branch scoped queries.

### Goals

- Record user prompts, tool calls (name + inputs + full outputs), agent responses,
  and failures for every Claude Code session
- Store in a personal scope (`user-history`) isolated from product knowledge
- Support multi-agent conversations: main agents, parallel subagents, and their
  research findings, all traceable via graph edges
- Enable cross-session queries: "what did I research on feat/auth last week?"
- Auto-promote product decisions/findings to the `corvia` scope
- Zero ongoing maintenance: hooks + adapter runs automatically at session end

### Non-Goals

- Real-time indexing within a session (intra-session context is available natively)
- Recording sessions from non-Claude-Code tools
- Centralised multi-user history (this is per-user, local-first)

---

## 2. Architecture

```
Claude Code session
  ├── UserPromptSubmit hook  ─┐
  ├── PreToolUse hook         ├──▶  ~/.claude/sessions/<session-id>.jsonl
  ├── PostToolUse hook        │         (append-only, one line per event)
  └── SessionEnd hook        ─┘
                                      │
                              gzip + trigger ingest
                                      │
                                      ▼
                      corvia-adapter-claude-sessions
                        (reads ~/.claude/sessions/*.jsonl.gz)
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                    path heuristics          LLM classifier
                    (fast, free)             (ambiguous cases)
                          │                       │
                          ▼                       ▼
              scope: "user-history"     scope: "user-history"
              (personal, deletable)     (queued in .classify-queue)
                                                  │
                                        auto-discovery pass
                                                  │
                                                  ▼
                                        scope: "corvia"
                                      (promoted product entries)
```

---

## 3. Session Log Format

### File Location

```
~/.claude/sessions/
  <session-id>.jsonl        # active session (append-only)
  <session-id>.jsonl.gz     # closed session (gzipped by SessionEnd hook)
  .current-session-id       # written by SessionStart hook; read by all hooks
  .ingested                 # newline-delimited text; one session ID per line
  .classify-queue           # newline-delimited entry IDs awaiting LLM classification
  archive/                  # processed .jsonl.gz files moved here after ingestion
```

Files are user-global — not per-workspace. The `workspace` field on
`session_start` records which project each session belongs to.

### Event Schema (JSONL, one JSON object per line)

**session_start**
```json
{
  "type": "session_start",
  "session_id": "ses-abc123",
  "timestamp": "2026-03-14T05:00:00Z",
  "workspace": "/workspaces/corvia-workspace",
  "git_branch": "feat/auth",
  "agent_type": "main",
  "parent_session_id": null,
  "corvia_agent_id": "chunzhe"
}
```

**user_prompt**
```json
{
  "type": "user_prompt",
  "session_id": "ses-abc123",
  "turn": 1,
  "timestamp": "2026-03-14T05:01:00Z",
  "content": "can we record what tools was used?"
}
```

**tool_start**
```json
{
  "type": "tool_start",
  "session_id": "ses-abc123",
  "turn": 1,
  "timestamp": "2026-03-14T05:01:01.123456789Z",
  "tool": "Grep",
  "input": { "pattern": "scope_id", "path": "repos/corvia/crates" }
}
```

**tool_end**
```json
{
  "type": "tool_end",
  "session_id": "ses-abc123",
  "turn": 1,
  "timestamp": "2026-03-14T05:01:01.987654321Z",
  "tool": "Grep",
  "input": { "pattern": "scope_id", "path": "repos/corvia/crates" },
  "output": "...first 500 chars of output...",
  "truncated": false,
  "success": true,
  "duration_ms": 45
}
```

Failures: `"success": false`, `"output": "<error message>"` — same format.
Parallel tool calls within one turn are distinguished by nanosecond timestamps
and assembled in order by the adapter.

**agent_response**
```json
{
  "type": "agent_response",
  "session_id": "ses-abc123",
  "turn": 1,
  "timestamp": "2026-03-14T05:01:10Z",
  "content": "...Claude's reply text..."
}
```

Captured via the `Stop` hook if it provides response text; omitted gracefully
if the hook does not expose it.

**session_end**
```json
{
  "type": "session_end",
  "session_id": "ses-abc123",
  "timestamp": "2026-03-14T05:30:00Z",
  "total_turns": 12,
  "duration_ms": 1800000
}
```

### Compression

`SessionEnd` hook gzips the file immediately after writing `session_end`.
Estimated volume: 60–200 KB per heavy session after compression (70–85%
reduction on JSON). A month of daily use ≈ 2–6 MB total.

---

## 4. Hook Scripts

Five scripts added to `.claude/hooks/` and wired into `.claude/settings.json`.

### 4.1 `record-session-start.sh`

Triggered by: `SessionStart`

Generates a session ID and writes it to `~/.claude/sessions/.current-session-id`
(overwriting any prior value — one active session per user at a time for hooks).
Session ID source priority: `CLAUDE_CODE_SESSION_ID` env var if set by Claude Code;
otherwise `uuidgen` (or `cat /proc/sys/kernel/random/uuid` as fallback).
`agent_type` detection: if `CLAUDE_CODE_IS_SUBAGENT=1` or similar env var is set,
mark as `"subagent"`; otherwise `"main"`. This requires a research spike (Deliverable 5)
to confirm exact env var names.
`parent_session_id` from `CLAUDE_CODE_PARENT_SESSION_ID` env var if set.
Creates `~/.claude/sessions/<session-id>.jsonl`, writes `session_start` event
including `$PWD` for workspace, `git rev-parse --abbrev-ref HEAD` for branch,
and `CORVIA_AGENT_ID` for user identity.

### 4.2 `record-prompt.sh`

Triggered by: `UserPromptSubmit`

Reads stdin JSON, extracts `.prompt`, appends `user_prompt` event.
Increments turn counter (stored as `~/.claude/sessions/<session-id>.turn`).
Uses `O_APPEND` write mode (atomic for writes under PIPE_BUF on Linux).

### 4.3 `record-tool-start.sh`

Triggered by: `PreToolUse`

Reads stdin JSON (tool name + input), appends `tool_start` with nanosecond
timestamp. Does NOT block — exits 0 immediately after append.

### 4.4 `record-tool-end.sh`

Triggered by: `PostToolUse`

Reads stdin JSON (tool name + input + output), truncates output to first 500
characters if over 500 chars and sets `"truncated": true`, appends `tool_end`.

**Skipped tools** (low signal, not recorded):
- `SessionStart`-related internal calls
- `health` endpoint calls
- MCP tool calls that are pure metadata lookups

### 4.5 `record-session-end.sh`

Triggered by: `SessionEnd`

Writes `session_end` event, gzips the file
(`gzip -f ~/.claude/sessions/<session-id>.jsonl`), then triggers
`corvia workspace ingest` so the session is indexed before the next session starts.
Runs after the existing orphan-cleanup hook.

---

## 5. Adapter: `corvia-adapter-claude-sessions`

### Location

`adapters/corvia-adapter-claude-sessions/rust/`

Follows the established adapter structure (see `corvia-adapter-git` and
`corvia-adapter-basic`).

### Interface

Implements `IngestionAdapter`:
- `domain()` → `"claude-sessions"`
- `ingest_sources(source_path)` → reads `~/.claude/sessions/` regardless of
  `source_path`. The adapter ignores the argument by convention; the workspace
  ingest pipeline invokes it with a fixed sentinel value (`"~/.claude/sessions"`)
  configured in `corvia.toml` alongside the adapter domain declaration. This
  matches the pattern used for adapters that own their own source path (e.g.
  a database adapter that always reads from a fixed connection string).
- `register_chunking()` → registers `SessionChunkingStrategy`

### Processing Pipeline

**1. Discovery**

Scans `~/.claude/sessions/*.jsonl.gz`. Reads `.ingested` state file to skip
already-processed sessions. New sessions only.

**2. Parsing**

Gunzips and parses JSONL. Groups events by `turn` counter. Sorts events within
a turn by nanosecond timestamp (handles parallel tool calls).

**3. Turn chunking (one `KnowledgeEntry` per turn)**

Each turn becomes a single structured text chunk:

```
[Turn 3 | feat/auth | ses-abc123 | 2026-03-14T05:01:00Z]
USER: can we record what tools was used?
TOOLS:
  - Grep("scope_id" in repos/corvia/crates) → 45ms ✓
  - Read(repos/corvia/crates/corvia-kernel/src/lite_store.rs) → 12ms ✓
  - Grep("delete_scope") → 23ms ✓
RESPONSE: Good news — it's straightforward. Here's the full picture...
```

Failures are included as `→ ERROR: <message>`.

**4. `SourceMetadata` extension (C1)**

The `IngestionAdapter` returns `Vec<SourceFile>` where each `SourceFile` contains
a `SourceMetadata`. The current `SourceMetadata` struct carries only `file_path`,
`extension`, `language`, `scope_id`, `source_version`. It cannot carry `workstream`,
`content_role`, or `source_origin` without a schema change.

**Resolution — Option A:** extend `SourceMetadata` with three optional fields:

```rust
// crates/corvia-kernel/src/chunking_strategy.rs
pub struct SourceMetadata {
    // existing fields unchanged
    pub file_path: String,
    pub extension: String,
    pub language: Option<String>,
    pub scope_id: String,
    pub source_version: String,
    // new optional fields (serde default = None; no impact on existing adapters)
    #[serde(default)]
    pub workstream: Option<String>,
    #[serde(default)]
    pub content_role: Option<String>,
    #[serde(default)]
    pub source_origin: Option<String>,
}
```

The `ChunkingPipeline` must be updated to propagate these fields to the resulting
`KnowledgeEntry` when present:
- `metadata.workstream` → `entry.workstream`
- `metadata.content_role` → `entry.metadata.content_role`
- `metadata.source_origin` → `entry.metadata.source_origin`

The Claude session ID is already encoded in `source_version`
(`"<ses-abc123>:turn-3"`) and is recoverable without a dedicated field.
`KnowledgeEntry.session_id` is reserved for corvia staging sessions and is
**not set** by this adapter.

**Entry metadata per turn:**

```
scope_id:       "user-history"
workstream:     git branch from session_start (e.g. "feat/auth")
source_version: "<claude-session-id>:turn-<N>"
content_role:   "session-turn" | "research"
                ("research" if agent_type=subagent and turn contains
                 semantic research patterns: search + read + synthesise)
source_origin:  "claude:main" | "claude:subagent"
file_path:      "<claude-session-id>"   (reused as session identifier)
```

`agent_id` on `KnowledgeEntry` is not set by this adapter (it carries the
corvia write-agent identity, not the recording agent). The Claude Code user
identity is recoverable from `source_origin` and `file_path`.

**5. Graph edges**

After ingesting all turns of a session, if `parent_session_id` is set:
```
relate(
  from: first_turn_entry_id_of_this_subagent_session,
  relation: "spawned_by",
  to:   first_turn_entry_id_of_parent_session
)
```

Edge direction: **subagent → `spawned_by` → parent**. To find all subagents
spawned from a main-agent session, call:
```rust
store.traverse(
    start: &main_session_first_turn_entry_id,
    relation: Some("spawned_by"),
    direction: EdgeDirection::Incoming,   // entries that point TO this node
    max_depth: 3,
)
```
`Incoming` returns all entries whose `spawned_by` edge points to the start node,
i.e., the subagent sessions. `Outgoing` from a subagent entry reaches its parent.

**6. State update**

Appends the session ID to `.ingested` after successful processing.

### Classification Pass (C2)

Runs after all sessions are ingested. Classification state is tracked in
`~/.claude/sessions/.classify-queue` (newline-delimited corvia entry IDs),
not in `source_origin` or any other entry field. This keeps `source_origin`
clean for its intended exact-match filter semantics.

**Heuristic (fast, free):**
- During turn chunking, if any `tool_end` input references paths under
  `repos/*/` or the workspace root, append the resulting entry ID to
  `.classify-queue` immediately after writing the entry to corvia.

**LLM pass (reads `.classify-queue`):**
- For each queued entry ID, fetch the entry content and send to generation engine:
  > "Does this conversation turn contain a product decision, architectural
  > discussion, or research finding relevant to building the corvia software
  > product? Answer YES or NO with one sentence of rationale."
- YES → writes a copy to `"corvia"` scope with `content_role` inferred from
  content (design, decision, research, learning); remove ID from queue
- NO → remove ID from queue; entry stays in `"user-history"` only
- Queue is rewritten atomically (write temp file, rename) after each batch

---

## 6. Kernel Change: Workstream Filter

### Motivation

`workstream` is stored on every `KnowledgeEntry` and persisted in all backends
(LiteStore Redb, SurrealDB, PostgreSQL) but is not currently filterable via
search. Adding this filter unlocks feature-branch scoped queries:

```
corvia_search "OAuth research" scope:user-history workstream:feat/auth
```

### Touchpoints (~35 lines across 6 files)

**`crates/corvia-kernel/src/rag_types.rs`**
Add `workstream: Option<String>` to `RetrievalOpts` and `Default` impl.

**`crates/corvia-kernel/src/retriever.rs`**
1. Extend `post_filter_metadata` signature with `workstream: Option<&str>` as
   third parameter (after `source_origin`)
2. Add workstream check inside the filter closure:
   `if r.entry.workstream.as_str() != ws { return false; }`
3. Extend the oversample condition in **both** `VectorRetriever::retrieve` and
   `GraphExpandRetriever::retrieve`:
   `|| opts.content_role.is_some() || opts.source_origin.is_some() || opts.workstream.is_some()`
4. Thread `opts.workstream.as_deref()` through all **three** `post_filter_metadata`
   call sites:
   - `VectorRetriever::retrieve` (line ~131 in retriever.rs)
   - `GraphExpandRetriever::retrieve` (line ~410 in retriever.rs)
   - Any direct call in `mcp.rs` if it calls `post_filter_metadata` independently

**`crates/corvia-server/src/rest.rs`**
Add `workstream: Option<String>` to `SearchRequest`, wire to `RetrievalOpts`.

**`crates/corvia-server/src/mcp.rs`**
Add `workstream` optional parameter to `corvia_search` tool schema.

**Tests**
Mirror existing `test_post_filter_by_content_role` pattern with workstream cases.

### No Breaking Changes

`workstream` defaults to `None` (no filter). All existing callers unchanged.

---

## 7. Scope Configuration

### `corvia.toml` Addition

```toml
[[scope]]
id          = "user-history"
description = "Personal Claude Code session history — independent of product knowledge"
ttl_days    = 180   # entries expire after 6 months (TTL enforcement: future work)
```

### Isolation Properties

| Operation | Behaviour |
|---|---|
| Search in `corvia` scope | Never returns `user-history` entries |
| Search in `user-history` | Never returns product entries |
| `delete_scope("user-history")` | Removes all personal entries from files + Redb; HNSW vectors become orphaned |
| `corvia rebuild` | Compacts HNSW, removing orphaned vectors |
| Full deletion workflow | `delete_scope user-history` → `corvia rebuild` |

---

## 8. Multi-Agent Structure

### Session Identity Fields

Every session log and every ingested entry carries:

| Field | Source | Meaning |
|---|---|---|
| `session_id` | `CORVIA_SESSION_ID` env | Unique per Claude Code process |
| `agent_type` | Detected at hook time | `"main"` or `"subagent"` |
| `parent_session_id` | `CORVIA_PARENT_SESSION_ID` env | Links subagent to spawning session |
| `corvia_agent_id` | `CORVIA_AGENT_ID` env | User identity (shared across agents) |
| `workstream` | `git rev-parse --abbrev-ref HEAD` | Feature branch |

### Parallel Main Agents

Two main agents on different feature branches write to separate files
(`~/.claude/sessions/<ses-A>.jsonl`, `~/.claude/sessions/<ses-B>.jsonl`).
No concurrency issues — session ID provides natural file isolation.

### Subagent Research Preservation

Subagent turns with `content_role: "research"` are retained in `user-history`
independently of the parent session. A future query "what did subagents find
about Stripe API?" searches `content_role:research source_origin:claude:subagent`
and returns findings across all sessions and dates.

### Example Graph Structure

```
user-history scope
├── workstream: feat/auth
│   ├── ses-A turn-1 ──spawned_by──▶ (none, is root)
│   ├── ses-A turn-2
│   ├── ses-A1 turn-1 ──spawned_by──▶ ses-A turn-2  [research: OAuth libraries]
│   └── ses-A2 turn-1 ──spawned_by──▶ ses-A turn-2  [research: JWT design]
└── workstream: feat/billing
    ├── ses-B turn-1
    └── ses-B1 turn-1 ──spawned_by──▶ ses-B turn-1  [research: Stripe API]
```

---

## 9. Performance Considerations

### HNSW Index Impact

LiteStore uses a **single global HNSW index** across all scopes. Scope filtering
is a post-filter. A large `user-history` scope degrades search in `corvia` scope
because the HNSW returns personal history candidates that are then discarded.

**Mitigations built into this design:**

| Mitigation | Effect |
|---|---|
| Turn-level chunking (not per-event) | ~7× fewer entries than naive per-event recording |
| Skip low-signal events | Omit health pings, internal hooks, duplicate reads |
| TTL via `valid_to` (future) | Expired entries excluded from search results |
| `corvia rebuild` after `delete_scope` | Compacts HNSW, restores search performance |

**Volume estimates (turn-level, heavy daily use):**

| Period | Entries |
|---|---|
| 1 week | ~140 |
| 1 month | ~600 |
| 6 months (TTL window) | ~3,600 |
| Steady-state (TTL enforced) | ≤ 3,600 |

For comparison, the corvia codebase itself ≈ 1,500–2,000 entries.
At steady state with TTL, `user-history` is ~2× the product scope — manageable.

### Per-Scope HNSW (Future Work)

A per-scope HNSW would eliminate cross-scope pollution entirely. Not in scope
for this feature but noted as the permanent fix.

---

## 10. Deletion and Lifecycle

```
Normal expiry:    entries pass ttl_days → valid_to set → excluded from search
                  (TTL enforcement is future kernel work; ttl_days is metadata only today)
Full scope wipe:  POST /v1/scopes/user-history/delete  (REST API)
                  followed by: corvia rebuild            (compact HNSW)
Raw log cleanup:  adapter moves ~/.claude/sessions/*.jsonl.gz → archive/ after ingestion
                  archived files kept for re-ingestion (e.g. after corvia rebuild --fresh)
```

`corvia gc --scope` does not yet exist as a CLI command. Full wipe is via the
REST API (`delete_scope`) or the MCP `corvia_gc_run` tool. A scoped GC CLI
command is follow-on work (see Section 11).

---

## 11. Open Questions / Future Work

| Item | Notes |
|---|---|
| `Stop` hook response capture | Depends on whether Claude Code exposes response text in `Stop` hook stdin; design degrades gracefully if not available |
| Claude Code hook env var names | Exact names of `CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_PARENT_SESSION_ID`, `CLAUDE_CODE_IS_SUBAGENT` need verification (Deliverable 5 spike); hooks fall back to `uuidgen` if unset |
| TTL enforcement in kernel | `ttl_days` in `corvia.toml` is metadata today; GC enforcement is follow-on kernel work |
| Scoped GC CLI command | `corvia gc --scope <id>` does not exist; deletion today is via REST API + `corvia rebuild` |
| Per-scope HNSW indices | Permanent fix for HNSW pollution; separate kernel design |
| Workstream filter in `corvia_ask` / `corvia_context` MCP tools | Extend beyond `corvia_search` in a follow-on |
| `ChunkingPipeline` propagation of new `SourceMetadata` fields | Requires identifying the exact insertion point in `chunking_pipeline.rs` where `KnowledgeEntry` is constructed from a `RawChunk` + `SourceMetadata` |

---

## 12. Deliverables

| # | Deliverable | Type |
|---|---|---|
| 1 | `SourceMetadata` extension + `ChunkingPipeline` propagation PR | `corvia-kernel` |
| 2 | Workstream filter PR | `corvia-kernel` + `corvia-server` |
| 3 | 5 hook scripts + settings.json update | Workspace |
| 4 | `corvia-adapter-claude-sessions` crate | New adapter |
| 5 | `corvia.toml` `user-history` scope entry | Config |
| 6 | Claude Code hook env var names spike | Research |
