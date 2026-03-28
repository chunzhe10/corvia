# Agent Teams Adapter -- Design Spec

**Date:** 2026-03-28
**Status:** Draft
**Scope:** corvia-adapter-claude-sessions (extension) . workspace hooks . graph edges
**Prerequisite:** claude-session-history design (2026-03-14)

---

## 1. Overview

This design extends the `corvia-adapter-claude-sessions` adapter to capture Claude
Code Agent Teams coordination artifacts -- team structure, task lifecycle, and
inter-agent messages -- before they are deleted on team cleanup. The adapter writes
these artifacts into the `user-history` scope as searchable knowledge entries with
graph edges linking teams to their members, tasks, and message threads.

Agent Teams (experimental, Claude Code v2.1.32+) coordinate multiple Claude Code
sessions working in parallel. All team state is ephemeral: deleted when the lead
runs cleanup. This adapter makes that state persistent and queryable.

### Goals

- Capture team structure, task lifecycle, and messages before cleanup deletes them
- Create graph edges: team to members, tasks to owners, task dependencies
- Correlate team members to their session history entries (from the session adapter)
- Support dynamic team membership (teammates spawned at any time)
- Zero configuration: hooks install alongside the session history hooks
- Extend the existing claude-sessions adapter (not a new binary)

### Non-Goals

- Real-time indexing during team lifetime (hooks capture incrementally, ingestion is batch)
- Capturing teammate session content (that is the session history adapter's job)
- Multi-user team coordination (local-first, same as session history)
- Replacing Agent Teams' native task list or messaging

---

## 2. Architecture

```
Claude Code Agent Teams session
  |
  |-- TeamCreate            --> ~/.claude/teams/{team-name}/config.json
  |-- Agent(teammate)       --> config.json members[] updated
  |-- TaskCreate            --> ~/.claude/tasks/{team-name}/{id}.json
  |-- TaskComplete          --> task status updated
  |-- SendMessage           --> inboxes/{name}.json updated
  |-- TeammateIdle          --> hook fires (lead process)
  |-- Teammate SessionEnd   --> hook fires (teammate process)
  |-- TeamDelete            --> both directories deleted
  |
  |----- Hooks capture to staging BEFORE deletion -----
  |
  v
~/.corvia/staging/agent-teams/{team-name}/
  config.json              # copied from teams dir
  tasks.jsonl              # one line per task event (created, completed)
  messages.jsonl           # one line per inbox snapshot
  .capture-log             # which hooks fired, when
  |
  v
corvia-adapter-claude-sessions --domain agent-teams
  (reads staging dir, emits D75 JSONL)
  |
  v
corvia kernel
  embed --> store --> wire graph edges
  scope: "user-history"
```

---

## 3. Hook Scripts

Three hooks capture team state incrementally. All hooks write to the staging
directory at `~/.corvia/staging/agent-teams/{team-name}/`. Staging survives
team cleanup because it is outside `~/.claude/`.

### 3.0 Security baseline (all hooks)

All hook scripts must apply these defenses before any file operations:

**Input validation.** Team names and teammate names from stdin JSON must match
`^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`. If validation fails, the hook exits 0
(allow the Claude Code operation) and logs the rejection to stderr. This
prevents path traversal via names like `../../etc/cron.d/malicious`.

**Directory permissions.** Staging directories are created with restrictive
permissions. Agent Teams messages may contain secrets, credentials, or
proprietary code.

```bash
umask 077
mkdir -p -m 0700 "$STAGING_DIR/$team_name"
```

**Atomic writes.** All file writes use the temp-then-rename pattern to prevent
partial files from corrupting staging data:

```bash
tmp=$(mktemp "$target.XXXXXX")
echo "$json_line" > "$tmp"
mv "$tmp" "$target"          # atomic rename
```

For append-only files (`tasks.jsonl`), use `flock` to serialize concurrent
writes from multiple hooks:

```bash
(
  flock -x 9
  echo "$json_line" >> "$STAGING_DIR/$team_name/tasks.jsonl"
) 9>"$STAGING_DIR/$team_name/tasks.jsonl.lock"
```

This prevents interleaved partial lines when `TaskCreated` and `TaskCompleted`
hooks fire concurrently (the PIPE_BUF limit of 4096 bytes is not guaranteed
sufficient for task-completed events with embedded `full_task` objects).

**Truncation.** Task descriptions exceeding 64 KB are truncated. This bounds
the maximum staging file growth per hook invocation.

### 3.1 `capture-task-created.sh`

Triggered by: `TaskCreated`

Reads stdin JSON:
```json
{
  "session_id": "lead-session-id",
  "hook_event_name": "TaskCreated",
  "task_id": "1",
  "task_subject": "Review auth module",
  "task_description": "..."
}
```

Action: Appends a task-created event to `staging/{team-name}/tasks.jsonl`:
```json
{"event":"created","task_id":"1","subject":"...","description":"...","timestamp":"..."}
```

Note: `TaskCreated` does NOT include `team_name` or `teammate_name` on stdin.
The hook infers team name by scanning `~/.claude/tasks/` for the directory
containing task `{task_id}`. If multiple teams exist (rare -- one team per
session limitation), the most recently modified directory wins.

Target latency: < 50ms.

### 3.2 `capture-task-completed.sh`

Triggered by: `TaskCompleted`

Reads stdin JSON (same schema as TaskCreated plus completion fields).

Action:
1. Read the full task file from `~/.claude/tasks/{team-name}/{task-id}.json`
   (includes `owner`, `status`, `blocks`, `blockedBy` -- fields not in stdin).
2. Append a task-completed event to `staging/{team-name}/tasks.jsonl`:
   ```json
   {"event":"completed","task_id":"1","subject":"...","owner":"researcher","timestamp":"...","full_task":{...}}
   ```

Target latency: < 100ms.

### 3.3 `capture-teammate-idle.sh`

Triggered by: `TeammateIdle`

Reads stdin JSON:
```json
{
  "session_id": "lead-session-id",
  "hook_event_name": "TeammateIdle",
  "teammate_name": "researcher",
  "team_name": "my-team"
}
```

Action:
1. Copy `~/.claude/teams/{team-name}/config.json` to staging via atomic rename
   (`cp source tmp && mv tmp target`). Two concurrent TeammateIdle hooks may
   both copy config.json; atomic rename ensures neither produces a partial file.
2. Snapshot the teammate's inbox: copy
   `~/.claude/teams/{team-name}/inboxes/{teammate-name}.json` to staging
   (same atomic pattern). Each teammate writes to a different inbox file, so
   there is no cross-teammate write contention on inbox snapshots.
3. Snapshot the lead's inbox (to capture messages FROM the teammate):
   copy `inboxes/team-lead.json` to staging. This file IS subject to concurrent
   writes from multiple TeammateIdle hooks; atomic rename resolves this (last
   writer wins, which is correct since the latest snapshot is the most complete).
4. Append to `.capture-log` via `flock` (shared with task hooks).

TeammateIdle fires every time a teammate's LLM turn ends. To avoid redundant
copies, the hook checks file mtimes: only copy if the source is newer than the
staging copy.

Target latency: < 200ms (may be slower on WSL2/NFS; hooks are non-blocking
to the Claude Code session via the async hook mechanism).

### 3.4 Final sweep (teammate SessionEnd)

The session history adapter's existing `record-session-end.sh` hook fires when
each teammate shuts down. At this point, team files still exist (cleanup requires
all teammates to be shut down first, but the lead hasn't run cleanup yet).

Enhancement to the existing SessionEnd hook: if `~/.claude/teams/` contains any
team directories, do a final copy of config.json, all task files, and all inbox
files to staging. The sweep is scoped to teams matching the ending session's
`leadSessionId` (checked via config.json). This prevents a teammate's SessionEnd
from capturing unrelated teams' data.

The team sweep is wrapped in a conditional (`if [ -d ~/.claude/teams ]`) and
any error in the sweep is caught and logged but does not abort the session
history gzip/ingest. The session history adapter is the established system;
the team sweep must not break it.

### 3.5 Hook installation

Hooks are added to `settings.json` alongside the existing session history hooks
(same `hooks` block). The `corvia hooks init` command is extended to generate
these three additional hook entries.

**Conditional installation.** `corvia hooks init` checks whether Agent Teams
is enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json env
block). If not, team hooks are omitted. Users who enable Agent Teams later
can re-run `corvia hooks init` to add them. Unused hooks are harmless no-ops
(the hook events simply never fire if Agent Teams is not used), but omitting
them keeps settings.json clean.

---

## 4. Adapter Extension

### Domain mode and D75 dispatch

The adapter gains a second domain mode. Dispatch uses the `source_path` field
in the D75 `Ingest` request, which the current adapter ignores (it always reads
`~/.claude/sessions/`). The extension gives it meaning:

- `source_path` containing `"agent-teams"` -> teams ingestion path
- Any other `source_path` (or empty) -> existing sessions ingestion path

**Adapter metadata** (unchanged `--corvia-metadata` response):
```json
{
  "name": "claude-sessions",
  "version": "0.2.0",
  "domain": "claude-sessions",
  "protocol_version": 1,
  "supported_extensions": ["jsonl", "json"],
  "chunking_extensions": []
}
```

The adapter name and domain stay `"claude-sessions"`. Multi-domain is an
internal dispatch detail, not a protocol change. The kernel treats it as one
adapter invoked twice with different `source_path` values.

**corvia.toml configuration** (two source entries for one adapter):
```toml
[[sources]]
path = "~/.claude/sessions"
adapter = "claude-sessions"

[[sources]]
path = "agent-teams"
adapter = "claude-sessions"
```

The workspace ingest pipeline calls the adapter once per source entry.
When `source_path = "agent-teams"`, the adapter reads from
`~/.corvia/staging/agent-teams/` (the path is a sentinel, not a literal
directory). This matches the existing pattern where the sessions adapter
ignores `source_path` and always reads from a fixed location.

### Ingestion flow

```
1. Scan ~/.corvia/staging/agent-teams/ for team directories
2. Read .ingested state file, skip already-processed teams
3. For each unprocessed team:
   a. Parse config.json -> team structure entry
   b. Parse tasks.jsonl -> task entries (grouped by task_id)
   c. Parse inbox snapshots -> message entries (grouped by task)
   d. Emit SourceFile objects via D75 JSONL
4. Append team name to .ingested
5. Optionally: scan ~/.claude/teams/ for live teams not yet in staging
   (fallback for hooks that didn't fire)
```

### Entry types produced

**Team structure entry** (one per team):

```
[Team: security-review | Created: 2026-03-28T10:00:00Z | Ended: 10:45:00Z | Status: completed]
LEAD: team-lead (claude-opus-4-6)
MEMBERS:
  - security-reviewer (haiku, joined +0s)
  - perf-reviewer (haiku, joined +2s)
  - test-validator (sonnet, joined +15min)
TASKS: 5 total (4 completed, 1 pending)
PURPOSE: Review PR #142 from security, performance, and test coverage angles
```

`Ended` is the timestamp of the last TeammateIdle or SessionEnd capture for
this team. `Status` is inferred: `completed` if all tasks are completed or
deleted, `abandoned` if incomplete tasks remain and no cleanup event was
captured, `unknown` otherwise.

Metadata:
```
scope_id:       "user-history"
content_role:   "memory"
source_origin:  "claude:team:security-review"
source_version: "security-review:config"
file_path:      "security-review"
```

**Task entry** (one per task):

```
[Task #1: review-auth-module | Team: security-review | Status: completed]
DESCRIPTION: Review authentication module for security vulnerabilities.
  Focus on token handling, session management, input validation.
ASSIGNED TO: security-reviewer
DEPENDS ON: (none)
CREATED: 2026-03-28T10:01:00Z -> COMPLETED: 2026-03-28T10:15:00Z (14min)
```

Metadata:
```
scope_id:       "user-history"
content_role:   "task"
source_origin:  "claude:team:security-review"
source_version: "security-review:task:1"
file_path:      "security-review"
```

Note: `content_role` uses `"task"` (not `"plan"`) to distinguish Agent Teams
work items from implementation plans. This requires extending the content_role
enum with a `"task"` variant (one-line change in `corvia-common/src/config.rs`).

**Message entry** (one per task-grouped thread):

Messages are grouped by the task they relate to. Messages referencing a
`task_assignment` are grouped with that task. System messages (idle_notification,
shutdown_request/response) go into a "coordination" group. Ungrouped messages
go into a "general" group.

```
[Messages: Task #1 review-auth-module | Team: security-review]
[10:05] security-reviewer -> perf-reviewer:
  "Found that JWT validation skips expiry check on refresh tokens.
   Does the refresh endpoint have rate limiting?"
[10:07] perf-reviewer -> security-reviewer:
  "No rate limiting on /api/refresh. Endpoint doesn't log failed
   attempts either. Flagging both as high severity."
[10:08] security-reviewer -> broadcast:
  "Consensus: refresh token handling needs a rewrite."
```

Metadata:
```
scope_id:       "user-history"
content_role:   "finding"
source_origin:  "claude:team:security-review"
source_version: "security-review:messages:task-1"
file_path:      "security-review"
```

**Broadcast deduplication.** A broadcast message appears in every teammate's
inbox. Before grouping, the adapter deduplicates messages by
`hash(sender + content + timestamp)`. This prevents a broadcast to 5 teammates
from producing 5 copies of the same message in the entry. The sender field is
included in the hash to prevent collisions between identical messages from
different senders.

Messages exceeding 512 tokens are split by the kernel's ChunkingPipeline with
64-token overlap (existing mechanism).

### Entry count estimate

For a typical 5-agent, 30-minute team session:
- 1 team structure entry
- 3-8 task entries
- 3-8 message group entries (one per task + coordination)
- Total: 7-17 entries per team session

6-month steady state with daily use: ~1,200-3,000 entries.

---

## 5. Graph Edges

After ingesting all entries for a team, the adapter wires these edges.

### Deferred edge wiring

Graph edges that reference session history entries (from the session adapter)
may fail if the session adapter has not yet run. The adapter handles this
gracefully:

1. Attempt to resolve the target entry by querying the store.
2. If found, create the edge immediately.
3. If not found, append the edge spec to `.pending-edges` in the staging
   directory: `{from_source_version, relation, to_source_version}`.
4. On the next ingest run, retry pending edges. Edges that resolve are
   created and removed from the file. Edges that still fail remain pending.

This eliminates the ordering dependency between session history and agent-teams
adapter runs.

### Team to session (lifecycle)

```
relate(
  from: team_structure_entry_id,
  relation: "ran_during",
  to: lead_session_first_turn_entry_id
)
```

This `ran_during` edge links the team to its lead session, enabling temporal
queries ("what teams ran on March 15?") via the session's timestamp metadata.
Distinct from `has_member` which captures membership, not lifecycle.

### Team to members

```
relate(
  from: team_structure_entry_id,
  relation: "has_member",
  to: teammate_subagent_first_turn_entry_id   // one edge per teammate
)
```

The lead session's first turn entry is found by querying:
`scope:"user-history" source_version:"<leadSessionId>:turn-1"`

For teammate sessions, the adapter uses the description-matching correlation
from Section 6 to find the subagent transcript entry. The query:
`scope:"user-history" source_origin:"claude:subagent"` filtered by
`source_version` prefix matching the lead's session ID.

The existing `spawned_by` edges (from the session history adapter) link
subagent entries to the lead's session. The `has_member` edges from the
teams adapter add the team-level grouping on top.

### Task to owner (teammate)

```
relate(
  from: task_entry_id,
  relation: "assigned_to",
  to: teammate_subagent_first_turn_entry_id   // the specific teammate
)
```

The task's `owner` field (e.g., `"security-reviewer"`) is matched to the
teammate's subagent entry via the same correlation used for `has_member`.
If the teammate's session entry is not yet available (deferred wiring),
the edge is queued in `.pending-edges`.

This enables "what tasks did security-reviewer work on?" via a single
incoming-edge traversal from the teammate's session entry.

### Task dependencies

```
relate(
  from: blocked_task_entry_id,
  relation: "depends_on",
  to: blocking_task_entry_id
)
```

### Message to task

```
relate(
  from: message_group_entry_id,
  relation: "discusses",
  to: task_entry_id
)
```

### Edge summary

| Edge | From | To | Direction |
|------|------|----|-----------|
| `ran_during` | team entry | lead session entry | outgoing |
| `has_member` | team entry | teammate session entry | outgoing |
| `assigned_to` | task entry | teammate session entry | outgoing |
| `depends_on` | task entry | task entry | outgoing |
| `discusses` | message entry | task entry | outgoing |
| `spawned_by` | subagent entry | lead session entry | outgoing (existing) |

Full traversal for "everything about a team":
```
team_entry
  --ran_during--> lead_session_entry
  --has_member--> teammate_session_entries
    <--spawned_by-- (already linked by session adapter)
  <--assigned_to-- task_entries
    <--discusses-- message_entries
```

Full traversal for "what did security-reviewer work on?":
```
teammate_session_entry
  <--assigned_to-- task_entries
  <--has_member-- team_entry
```

---

## 6. Teammate Identity Correlation

### Verified mechanism (spike 2026-03-28)

Teammate identity is resolved through three complementary signals. No env vars
or mapping files are needed.

**Signal 1: Hook payloads (primary)**

`TeammateIdle` provides `teammate_name` + `team_name` on stdin. This identifies
which teammate went idle and which team they belong to. The hook copies the
teammate's inbox and config to staging, tagged with the teammate name.

**Signal 2: config.json**

`leadSessionId` bridges the team to the lead's session history.
`agentId` = `{name}@{team-name}` (deterministic, not UUID).
`joinedAt` provides spawn timing per member.

**Signal 3: Lead transcript (for subagent correlation)**

Teammate transcripts live at:
`{leadSessionId}/subagents/agent-{internalId}.jsonl`

To map `internalId` to teammate name:
1. Lead's transcript contains Agent tool calls with `name` (teammate name) and
   `description` (spawn description).
2. Each subagent transcript's first message has
   `<teammate-message summary="{description}">`.
3. Match `description` from Agent tool call to `summary` in transcript.

This is a structural match (same string), not content-dependent.

**Fallback: Temporal correlation**

`joinedAt` in config.json (ms precision) correlated with the first message
timestamp in each subagent transcript. Reliable for late-joiners (large time
gaps). For batch spawns (~1-2s apart), combined with description matching.

### Dynamic membership

Teammates can be spawned at any time during the team's lifetime. The adapter
handles this naturally: each `TeammateIdle` hook captures the current config.json,
which always reflects the latest membership. Late-joiners appear in the config
with a later `joinedAt` and trigger their own `TeammateIdle` events.

---

## 7. Optional: LLM Extraction Pass

After structural ingestion, an async classification pass extracts high-signal
entries from message groups. This follows the same `.classify-queue` pattern as
the session history adapter.

**Extraction targets:**
- Decisions made (with rationale and dissenting opinions)
- Key findings (research results, discoveries)
- Open questions (unresolved disagreements)

**Produced entries:**
```
scope_id:       "user-history"
content_role:   "decision" | "finding" | "question"
source_origin:  "claude:team:{team-name}"
source_version: "security-review:extracted:{n}"
```

Graph edge: `extracted_from` linking the extracted entry to its source message
group entry.

**Promotion:** Extracted entries with product relevance are promoted to the
`corvia` scope via the same LLM classifier used by the session history adapter.

This pass is optional. The adapter produces useful entries without it.

---

## 8. Staging Directory Layout

```
~/.corvia/staging/agent-teams/
  {team-name}/
    config.json            # Latest team config (overwritten on each TeammateIdle)
    tasks.jsonl            # Append-only: one JSON line per task event
    tasks.jsonl.lock       # flock file for concurrent append serialization
    inboxes/
      {agent-name}.json    # Latest inbox snapshot per teammate
      team-lead.json       # Lead's inbox
    .capture-log           # Hook event log (debugging only, not consumed by adapter)
    .pending-edges         # Deferred graph edges awaiting session history entries
  .ingested                # Global: newline-delimited team names already processed
```

**`.capture-log` format** (for debugging, not consumed by the adapter):
```jsonl
{"event":"TeammateIdle","teammate":"researcher","team":"my-team","timestamp":"..."}
{"event":"TaskCompleted","task_id":"1","team":"my-team","timestamp":"..."}
```

**Cleanup policy.** After a team is marked in `.ingested` and all `.pending-edges`
are resolved (or the team is older than 7 days), the adapter deletes the staging
directory during the next ingest run. This is controlled by a config option
(default: delete after ingest, option: retain for N days). Users who need to
explicitly purge staging data can run `rm -rf ~/.corvia/staging/agent-teams/{team}/`.

Files in staging are small (KB range). A month of daily team use produces
~100-500 KB of staging data.

---

## 9. Implementation Plan

### Phase 1: Hooks (3 scripts)

| Deliverable | Description | Effort |
|-------------|-------------|--------|
| H1 | `capture-task-created.sh` | Small |
| H2 | `capture-task-completed.sh` | Small |
| H3 | `capture-teammate-idle.sh` | Small |
| H4 | Extend `record-session-end.sh` with team file sweep | Small |
| H5 | Extend `corvia hooks init` to generate team hooks | Small |

### Phase 2: Adapter extension

| Deliverable | Description | Effort |
|-------------|-------------|--------|
| A1 | Add `agent-teams` domain mode to claude-sessions adapter | Medium |
| A2 | Staging directory scanner + parser | Medium |
| A3 | Team structure entry emitter | Small |
| A4 | Task entry emitter (with task-grouping for messages) | Medium |
| A5 | Message entry emitter (task-grouped chunking) | Medium |
| A6 | State tracking (`.ingested` files) | Small |

### Phase 3: Graph edges

| Deliverable | Description | Effort |
|-------------|-------------|--------|
| G1 | `ran_during` edge (team to lead session) | Small |
| G2 | `has_member` edges (team to teammate sessions) | Small |
| G3 | `assigned_to` edges (task to teammate session) | Medium |
| G4 | `depends_on` edges (task to task) | Small |
| G5 | `discusses` edges (message to task) | Small |
| G6 | Subagent correlation (description matching) | Medium |
| G7 | Deferred edge wiring (`.pending-edges` retry) | Medium |

### Phase 4: Optional extraction

| Deliverable | Description | Effort |
|-------------|-------------|--------|
| E1 | LLM extraction pass for decisions/findings | Medium |
| E2 | Promotion to product scope | Small |

### Tests

**Fixtures.** Test staging directories live in `tests/fixtures/agent-teams/` with
three scenarios: minimal (1 member, 0 tasks), typical (3 members, 5 tasks),
and large (10 members, 20 tasks). Fixtures are static JSON, no real Claude Code
sessions needed.

| Test | What it verifies |
|------|-----------------|
| T1 | Hook scripts write correct staging files from mock stdin |
| T2 | Hook rejects invalid team names (path traversal, special chars) |
| T3 | Concurrent hook writes to `tasks.jsonl` via `flock` produce valid JSONL |
| T4 | Adapter parses staging files into correct SourceFile entries |
| T5 | Task-grouped message chunking produces expected entry count |
| T6 | Broadcast dedup removes duplicate messages (5 inboxes, 1 unique broadcast) |
| T7 | Graph edges wired correctly (ran_during, has_member, assigned_to, depends_on, discusses) |
| T8 | Description-based subagent correlation finds correct matches |
| T9 | Late-joiner teammate appears in entries and graph |
| T10 | Idempotency: re-running adapter on same staging produces no duplicates |
| T11 | Fallback: adapter reads live team files when staging is missing |
| T12 | Deferred edges: missing session entries produce `.pending-edges`, resolved on retry |
| T13 | Partial staging: truncated last line in `tasks.jsonl` is skipped, rest processed |
| T14 | Empty team: config.json with 1 member, no tasks, produces valid team entry |
| T15 | Large team: 10 members, 20 tasks, correct entry count and edge count |
| T16 | SessionEnd sweep does not break session history gzip/ingest on error |

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Agent Teams API changes (experimental feature) | Medium | High | Hook payloads and file formats are the contract. Monitor Claude Code changelogs. Adapter fails gracefully on schema changes. |
| Cleanup runs before hooks capture | Low | Medium | Hooks fire before cleanup becomes possible (shutdown precedes cleanup). SessionEnd sweep is the safety net. |
| Description matching fails for subagent correlation | Low | Low | Falls back to temporal correlation. Worst case: entries exist but are unlinked. |
| Large teams produce too many message entries | Low | Low | Task-grouped chunking keeps count at O(tasks), not O(messages). |
| Staging directory grows unbounded | Low | Low | `.ingested` tracking + periodic cleanup of processed teams. |

---

## 11. Query Patterns Enabled

Once ingested, users and agents can answer these questions:

| Query | Mechanism |
|-------|-----------|
| "What did the security review team find?" | `corvia_search "security review findings" source_origin:claude:team:security-review` |
| "What tasks did security-reviewer work on?" | Graph: traverse incoming `assigned_to` edges from teammate session entry |
| "What teams were active on March 15?" | Search team entries by timestamp, `Ended` field confirms overlap |
| "Show me the messages where teammates disagreed" | `corvia_search "disagreed" content_role:finding source_origin:claude:team:*` |
| "Which teammates worked on the auth refactor?" | Graph: team entry -> `has_member` edges -> teammate entries |
| "What was decided during the parallel review?" | LLM extraction entries with `content_role:decision` |
| "How long did the investigation take?" | Team entry `Created` and `Ended` timestamps |

Graph traversal example for "everything about a team":
```
corvia_graph(entry: team_entry_id, max_depth: 2)
  -> ran_during -> lead session
  -> has_member -> teammate sessions
  <- assigned_to <- tasks
    <- discusses <- messages
```

---

## 12. Open for Discussion

1. **Should team entries promote to the product scope by default?** Team structure
   entries are metadata. Task and message entries may contain product-relevant
   decisions. Current design: all start in `user-history`, optional LLM extraction
   promotes relevant ones.

2. **Lead transcript tagging.** The lead's transcript entries have a `teamName`
   field during team activity. A future Phase 3 deliverable could tag these
   entries to enable "what did the lead do while the team was active?" queries.
   Not blocking for v1.

3. **Dashboard integration.** Team data is inherently visual (task dependency
   graphs, message flow, member contribution timelines). A future dashboard
   enhancement could render team timelines. Not in scope for this RFC.

4. **Multi-tool generalization.** The capture and parsing layers are Claude Code
   specific. If other tools (Codex, Cursor) develop team features, consider
   factoring the entry emitters into a shared module that takes a normalized
   intermediate representation. The current design does not preclude this.
