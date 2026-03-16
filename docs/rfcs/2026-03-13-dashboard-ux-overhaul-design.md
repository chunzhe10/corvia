# Dashboard UX Overhaul — Design Spec

> **Date:** 2026-03-13
> **Status:** Shipped (v0.4.3)
> **Scope:** 5 subsystems — graph simplification, agent registration, history UX, collapsible panel, agent descriptions

---

## Problem Statement

The corvia dashboard has several UX and data integrity issues that make it unusable for human consumption at its current scale (~3,583 entries):

1. **Graph visualization** is too fractured — single-level clustering can't simplify 3,583 entries into a coherent view. Falls back to a flat table at >500 edges.
2. **Agent/session counts** are inaccurate — Claude Code sessions don't register with meaningful identities, leading to anonymous or duplicate agent records.
3. **History tab** requires knowing a raw UUID — no browsing, no discovery, no activity feed.
4. **Right sidebar** is always visible, consuming screen space even when irrelevant.
5. **Agent descriptions** are machine-generated and meaningless — users can't identify agents for reconnection.

---

## Section 1: Graph Simplification — Multi-Tiered Semantic LOD

### 1.1 Backend: Cluster Hierarchy

A new `ClusterStore` module in `corvia-server/src/dashboard/clustering.rs` computes and caches a 3-level hierarchy. This is a **server-side view optimization**, not a kernel trait — it takes `Arc<dyn QueryableStore>` and stores results in its own Redb table.

**Embedding access:** Uses `knowledge_files::read_scope()` (already used by the graph endpoint) which returns `KnowledgeEntry` with `embedding: Option<Vec<f32>>` populated. At 3,583 entries × 768 dimensions, this is ~11 MB of float data per clustering run — acceptable for a debounced background task.

**Level 0 — Super-clusters (5–12):** K-means on all entry embeddings. K chosen by sampled silhouette score: compute silhouette on a random 10% sample (max 500 entries) for K=3..12, pick best K. Each super-cluster gets an auto-generated label from its centroid's nearest entry topic. Examples: "Agent Lifecycle", "Merge Pipeline", "Graph Store".

**Level 1 — Sub-clusters (3–8 per super-cluster):** Same algorithm applied within each super-cluster. Entries about "graph edges" vs "graph traversal" separate at this level.

**Level 2 — File groups:** Current path-based grouping (repo → crate → file). Existing behavior preserved as the deepest semantic level before individual entries.

**Level 3 — Individual entries:** Current entry-mode view.

**Recomputation trigger:** A background tokio task checks entry count every 60 seconds. If count has changed since last clustering run, recompute. Stored in Redb as `cluster_id → {level, parent_cluster_id, label, centroid, entry_ids}`.

**Degraded mode:** If clustering has not yet completed (server just started) or fails (too few entries, all embeddings None), the graph falls back to the current path-based grouping at all zoom levels. The frontend shows a "Computing clusters..." indicator during initial computation. Minimum 3 entries with embeddings required to run clustering.

**New endpoint:** `GET /api/dashboard/graph/scope?level=N` where:
- `level=0` returns super-clusters as nodes with inter-cluster edges aggregated
- `level=1&parent=<cluster_id>` returns sub-clusters within a given parent
- `level=2&parent=<cluster_id>` returns file groups
- `level=3` returns individual entries (existing behavior)

### 1.2 Frontend: Zoom-Driven LOD

The force-directed canvas adapts based on zoom level:

| Zoom range | Renders | Typical node count |
|------------|---------|-------------------|
| 0.2x–0.8x (exclusive) | Super-clusters (L0) | 5–12 |
| 0.8x–1.5x (exclusive) | Sub-clusters (L1) | 20–60 |
| 1.5x–3.0x (exclusive) | File groups (L2) | 50–200 |
| 3.0x–5.0x | Individual entries (L3) | Viewport-culled |

Boundary rule: each threshold is exclusive on the upper bound (e.g., zoom=0.8 renders L1, not L0). Hysteresis of 0.05x prevents flickering at boundaries.

Transitions animate: nodes split apart or coalesce as zoom crosses thresholds. The force simulation re-runs at each level with parameters tuned to node count (existing per-level tuning in GraphView already supports this pattern).

**Double-click** on any cluster node drills into the next level (extends current behavior to multiple tiers).

**Breadcrumbs** at top of graph: `All → Agent Lifecycle → Session Management → file.rs` — clickable to zoom back out to any level.

**Viewport culling at L3:** Only render entries within the visible canvas region + 20% margin. Nodes outside are excluded from the force simulation. This removes the current 500-edge hard limit.

### 1.3 Shared Vocabulary

The cluster labels generated here (super-cluster and sub-cluster names) are reused as the topic vocabulary for the History activity feed (Section 3) and agent activity summaries (Section 5). One clustering computation, three consumers.

---

## Section 2: Agent/Session Registration — Hook-Based Identity with Reconnect

### 2.1 Backend Changes

**AgentRecord extended with two new fields:**
- `description: Option<String>` — user-provided purpose at registration (e.g., "working on graph simplification")
- `activity_summary: Option<ActivitySummary>` — auto-generated, updated on session close or GC sweep

**ActivitySummary struct:**
```
ActivitySummary {
    entry_count: u64,
    topic_tags: Vec<String>,      // Top 3–5 topics from embedding clustering
    last_topics: Vec<String>,     // Topics from last session only (drift detection)
    last_active: DateTime<Utc>,
    session_count: u64,
}
```

**Activity summary generation:** On session close (or GC transition), compute topic tags from the agent's recent entries using embedding similarity clustering (same centroid-matching algorithm as graph clustering). No LLM calls — pure embedding math.

**New endpoint:** `GET /api/agents/reconnectable` returns agents that have sessions in `Stale` or `Orphaned` state. Note: `AgentStatus` only has `Active` and `Suspended` variants — the stale/orphaned states live on `SessionState`, not `AgentStatus`. This endpoint filters by session state, not agent state. Returns agent records with `description` and `activity_summary`, sorted by `last_seen` descending.

**Connect flow enhanced:** `POST /api/agents/{agent_id}/connect` now also accepts an optional `description` update, so reconnecting agents can revise their purpose.

### 2.2 Agent Selection via CLI Command

Claude Code `SessionStart` hooks are non-interactive (they execute and produce output but cannot accept user input). Instead, agent selection uses a **pre-session CLI command**:

**Option A (recommended): User runs `corvia agent connect` before starting Claude Code:**

```bash
$ corvia agent connect
Reconnectable agents:
  [1] graph-refactor (chunzhe)
      Purpose: working on graph simplification
      Activity: 12 entries across [graph store, edge handling]
      ⚠ Last session drifted to: [merge pipeline, conflict resolution]
      Last active: 2h ago
  [2] docs-update (chunzhe)
      Purpose: documentation sweep
      Activity: 5 entries across [API docs, README]
      Last active: 1d ago
  [N] Register new agent
Pick one: 1
✓ Connected as graph-refactor. CORVIA_AGENT_ID exported.
```

The command sets `CORVIA_AGENT_ID` in the shell environment. The `.mcp.json` config references this as `_meta.agent_id`.

**Option B: SessionStart hook with display-only reminder:**

A `SessionStart` hook in `.claude/settings.json` runs a non-interactive check:
1. Calls `GET /api/agents/reconnectable`
2. If reconnectable agents exist, prints a reminder: "Run `corvia agent connect` to select an identity (3 agents available)"
3. If `CORVIA_AGENT_ID` is already set, prints: "Connected as: graph-refactor"

Both options can coexist — B reminds the user if they forgot to run A.

### 2.3 GC Adjustment

Agent records persist indefinitely (one Redb row each, tiny). GC only rolls back orphaned **sessions**, not agents. Stale agents remain reconnectable until explicitly suspended via `corvia agent suspend`.

---

## Section 3: History UX — Activity Feed with Semantic Grouping

### 3.1 Activity Feed (New Default View)

The History tab landing page becomes a reverse-chronological activity feed. The UUID lookup moves to a search bar within the feed.

**Feed item anatomy:**
- Agent color dot + name
- Action verb: "wrote", "superseded", "merged"
- Entry title (source_file or content preview, 80 chars)
- Topic tags as pills (e.g., `[graph store]` `[edge handling]`)
- Content delta indicator: `+427` / `-203` bytes (green/red, inspired by Wikipedia)
- Relative timestamp ("3m ago", "2h ago")

**Semantic grouping:** Related items auto-cluster using embedding similarity of the involved entries. Starting threshold: cosine similarity > 0.8 (to be calibrated against actual embedding distribution — nomic-embed-text-v1.5 may need adjustment; use 90th percentile of pairwise similarity as calibration target). Grouped items collapse into a single row: "Agent claude-chunzhe updated graph store documentation (5 entries)" with an expand chevron. Rapid same-agent edits within 5 minutes always collapse regardless of similarity.

**Topic tags as filter facets:** A filter bar at the top shows discovered topics as toggleable pills. Topics derived from the shared cluster vocabulary (Section 1.3). Clicking a topic filters the feed. Multiple topics combine as OR. An agent dropdown filters by contributor.

### 3.2 Entry Detail (Existing, Enhanced)

Clicking any feed item opens the existing timeline + diff view in the right panel (using the collapsible sidebar from Section 4). UUID lookup bar available here as secondary entry point.

Enhancements to existing view:
- Agent color dot next to each version in the timeline
- Content delta indicator per version
- Breadcrumb back to feed

### 3.3 Cross-Tab Deeplinks

Any entry reference elsewhere in the dashboard becomes clickable:

| Source tab | Clickable element | Navigates to |
|-----------|-------------------|--------------|
| Graph | Entry node | Sidebar: entry detail with "View history →" link |
| Agents | Session entry list | History tab with entry pre-loaded |
| Health | Finding target IDs | History tab with entry pre-loaded |
| RAG | Source entries | History tab with entry pre-loaded |
| Traces | Entry IDs in span events | History tab with entry pre-loaded |
| Logs | Entry IDs in log messages | History tab with entry pre-loaded |

### 3.4 Backend

**New endpoint:** `GET /api/dashboard/activity?limit=50&offset=0&agent=X&topic=Y`

Returns the activity feed. Data access: uses `knowledge_files::read_scope()` to load entries, sorted by `recorded_at` descending, paginated by limit/offset. This is the same access pattern as the graph endpoint.

**Semantic grouping is precomputed server-side**, not per-request. The ClusterStore (Section 1) already assigns each entry to a cluster. The activity endpoint reads the cluster assignment for each entry and groups adjacent feed items that share the same cluster and agent within a 5-minute window. Grouping results are cached and invalidated when ClusterStore recomputes.

Topic tags computed by finding the nearest super-cluster centroid for each entry's embedding (reuses graph clustering from Section 1).

**Content delta:** Computed as UTF-8 byte difference between current entry content and its superseded predecessor (if any). New entries with no predecessor show `+N` only. Stored as part of the activity response, not precomputed.

---

## Section 4: Collapsible Right Panel — Context-Aware Auto-Show

### 4.1 Three Sidebar States

| State | Width | Trigger |
|-------|-------|---------|
| **Collapsed** | 0px (hidden) | Default on page load, click close button, click empty canvas |
| **Narrow** | 320px | Config/Health mode, agent card, feed item click |
| **Wide** | 480px | Graph entry detail with DocumentReader, History timeline+diff |

A subtle collapse/expand toggle button (chevron) at the right edge of the content area, always visible. CSS transition animates width changes (200ms ease).

### 4.2 Auto-Show Rules

| User action | Sidebar behavior |
|-------------|-----------------|
| Click graph cluster node | Open narrow: cluster stats + connected clusters |
| Click graph entry node | Open wide: DocumentReader + neighbors |
| Click agent card | Open narrow: session timeline + contribution |
| Click health finding | Open narrow: finding detail + target entries |
| Click activity feed item | Open wide: History timeline + diff |
| Click health pulse dots | Open narrow: Health panel |
| Click empty canvas / close button | Collapse |
| Switch tabs | Collapse (fresh context) |

### 4.3 Content Area Responsive

When sidebar collapses, content area smoothly expands to full width. The graph canvas resizes its coordinate space on sidebar transitions (existing `ResizeObserver` pattern in GraphView handles this). No content reflow — just width change.

### 4.4 Config/Health Access

Config and Health are no longer the sidebar default. They move to:
- **Config:** Gear icon in the header (opens sidebar in narrow mode)
- **Health:** Health pulse dots (existing behavior, opens sidebar in narrow mode)

The sidebar serves the content you're interacting with, not a persistent settings panel.

---

## Section 5: Agent Descriptions — Hybrid Identity with Topic Drift Detection

### 5.1 Data Model

Uses the `ActivitySummary` struct defined in Section 2.1. The key addition is **topic drift detection**.

### 5.2 Topic Drift Detection

On session close, compare `last_topics` (current session) against the agent's historical `topic_tags` (all sessions). If overlap < 50%, flag as drifted.

The reconnect list surfaces drift with a warning indicator (see Section 2.2 example). This helps users decide: reconnect as this agent (drift was intentional) or register a new one (old purpose no longer applies).

### 5.3 Display in Dashboard

The Agents tab cards show hybrid info:
- User-provided `description` as the subtitle
- `topic_tags` as pills (same visual language as History feed and Graph clusters)
- Drift indicator when `last_topics` diverged from historical `topic_tags`
- Activity sparkline (existing) annotated with topic color bands

### 5.4 Summary Update Triggers

Activity summary recomputes on:
1. **Session close** (commit or rollback)
2. **GC sweep** (stale → orphaned transition)
3. **Manual refresh** via `POST /api/agents/{id}/refresh-summary`

All pure embedding math — no LLM calls. Reuses the same centroid-matching logic as graph clustering (Section 1) and activity feed topics (Section 3).

---

## Cross-Cutting: Shared Embedding Vocabulary

A key architectural decision across this design: **one clustering computation serves three features.**

```
Entry embeddings (existing HNSW index)
        │
        ▼
  K-means clustering (ClusterStore)
        │
        ├──→ Graph LOD levels (Section 1)
        ├──→ Activity feed topic tags + grouping (Section 3)
        └──→ Agent activity summaries + drift detection (Section 5)
```

This ensures consistent topic vocabulary across the dashboard. "Graph Store" means the same thing in the graph view, the activity feed, and an agent's description.

---

## Subsystem Independence

These five sections can be implemented in parallel by separate agents:

| Section | Backend changes | Frontend changes | Dependencies |
|---------|----------------|-----------------|-------------|
| 1. Graph LOD | ClusterStore, new endpoint | GraphView rewrite | None (new module) |
| 2. Agent Registration | AgentRecord fields, new endpoint, CLI command | None (CLI) | Section 5 shares ActivitySummary; Section 5 consumes for frontend display |
| 3. History Feed | Activity endpoint | HistoryView rewrite | Section 1 for shared topic vocabulary |
| 4. Collapsible Panel | None | Layout.tsx refactor | None |
| 5. Agent Descriptions | ActivitySummary (shared with Section 2) | AgentsView enhancement | Section 1 for shared topic vocabulary |

**Recommended build order:**
1. Section 1 (Graph LOD) — establishes ClusterStore and shared vocabulary
2. Sections 2 + 4 + 5 in parallel — independent frontend/backend work
3. Section 3 (History Feed) — depends on shared vocabulary from Section 1

---

## Out of Scope

- LLM-generated semantic change summaries (future enhancement)
- Character-level diff highlighting (future enhancement)
- GitHub-style blame view for entry content (future enhancement)
- Grafana-style time picker for temporal queries (future enhancement)
- Event overlays on trace/metric graphs (future enhancement)
