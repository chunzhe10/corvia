# Agentic Retrieval Protocol -- Design

**Issue:** chunzhe10/corvia#47
**Date:** 2026-03-31
**Status:** Approved (design reviewed by Systems Architect + AI Agent Researcher)

## Summary

Define how AI agents interact with corvia as an organizational memory system.
Push complex logic server-side (deterministic, portable). Keep agent instructions
minimal (~20 lines in AGENTS.md). Five design areas, two implementation tracks.

## Architecture Principle

```
Server-side (Track B): robust, deterministic, cross-platform
Agent-side (Track A): minimal unconditional rules (~15-20 lines)
```

LLM self-assessment is ~70% reliable. More AGENTS.md instructions past a threshold
means lower compliance. Simple rules get followed; complex reasoning does not.

## Design Area 1: Adaptive Retrieval (Quality Signals)

### Server-side changes

Add `quality_signal` object to `corvia_search` MCP response:

```json
{
  "results": [...],
  "count": 3,
  "quality_signal": {
    "confidence": "low",
    "top_score": 0.32,
    "result_count": 2,
    "suggestion": "Try broader terms: HNSW configuration parameters",
    "gap_detected": true
  }
}
```

**Fields:**
- `confidence`: "high" | "medium" | "low" based on top_score and result_count
- `top_score`: highest score in results (already computed)
- `result_count`: number of results returned (already computed as `count`)
- `suggestion`: server-generated query hint when confidence is low/medium
- `gap_detected`: true when top_score < low threshold and count < limit/2

**Confidence thresholds** (empirically grounded from benchmark data and codebase):
- `high`: top_score >= 0.65 AND result_count >= 3
- `medium`: top_score >= 0.45 OR result_count >= 3
- `low`: everything else

Rationale: corvia_search results from the codebase exploration averaged 0.73-0.75
for relevant results. The merge similarity threshold is 0.85 (near-duplicate).
Setting high at 0.65 gives comfortable margin below known-good scores.

**Suggestion generation:** Template-based, not LLM-generated (zero latency):
- Low + few results: "Try broader terms: {extract key nouns from query}"
- Low + many results: "Results may not match intent. Try more specific terms."
- Medium: "Consider adding context: scope, component name, or time period."

**Implementation location:** `crates/corvia-server/src/mcp.rs` in `tool_corvia_search()`

### Optional: `min_score` parameter

Add optional `min_score: f32` to `corvia_search` input schema. Server filters
results below threshold and returns `below_threshold_count` in quality_signal.

### Agent-side rule (AGENTS.md)

> Check `quality_signal.confidence`. If `low`, follow the `suggestion` and retry once.

## Design Area 2: Context Injection for Subagents

### Server-side changes

Add `max_tokens` parameter to `corvia_context` MCP tool:

```json
{
  "query": "...",
  "scope_id": "corvia",
  "max_tokens": 2000,
  "format": "compact"
}
```

**Parameters:**
- `max_tokens` (integer, optional): Override token budget. Hard cap at 4000.
  Recommended: 5-10% of subagent context window.
- `format` (string, optional): "default" (existing) or "compact"
  - "compact": Strips system prompt, citation headers, footer. Returns just
    relevance-ranked content blocks with entry IDs for provenance.

**Existing infrastructure:** `TokenBudget.max_context_tokens` already exists in
`rag_types.rs`. The MCP handler just needs to read the parameter and pass it through.

**Implementation locations:**
- Tool definition: `mcp.rs` tool_definitions() (add max_tokens, format params)
- Handler: `mcp.rs` tool_corvia_context() (read params, pass to RAG pipeline)
- Augmenter: `augmenter.rs` (add compact format mode)

### Agent-side rule (AGENTS.md)

> Before spawning subagents for non-trivial work, call `corvia_context` with
> `max_tokens` appropriate for the task and include the result in the prompt.

## Design Area 3: Knowledge Gap Signaling

### Server-side changes (automatic, no agent action)

Every `corvia_search` call already goes through the RAG pipeline which produces
`RetrievalMetrics`. Capture gap signals by logging low-quality searches.

**New module:** `crates/corvia-kernel/src/gap_detector.rs`

**GapSignal struct:**
```rust
pub struct GapSignal {
    pub query: String,
    pub top_score: f32,
    pub result_count: usize,
    pub timestamp: DateTime<Utc>,
    pub scope_id: String,
}
```

**GapDetector:**
- In-memory ring buffer (bounded, e.g. 1000 entries)
- Records every search with top_score < 0.45 (low confidence threshold)
- Aggregation method: group by query similarity (cosine > 0.8 = same topic),
  count occurrences, rank by frequency
- Exposed via dashboard API endpoint: `GET /api/dashboard/gaps`

**Dashboard panel:** "Knowledge Gaps" showing:
- Top unanswered topics (aggregated queries)
- Frequency count
- Most recent occurrence
- Average top_score for that topic cluster

**Optional:** `corvia_report_gap` MCP tool for explicit gap reporting. Additive.
Not required by protocol.

**Important:** Gaps are operational telemetry, not knowledge entries.

### Implementation locations:
- New: `crates/corvia-kernel/src/gap_detector.rs`
- Dashboard: `crates/corvia-server/src/dashboard/mod.rs` (new endpoint)
- Integration: `crates/corvia-server/src/mcp.rs` (record gap signals after search)

## Design Area 4: Retrieval Feedback Loop

### Server-side changes (extends issue #12)

Issue #12 (buffered access tracking) is implemented in `access_buffer.rs`.
Extend with usage-based implicit feedback:

**Tracked signals:**
1. Results returned to agent (already tracked via access_buffer.record())
2. Re-search patterns: same topic searched again within a session = first results
   insufficient. Detect via embedding similarity of consecutive queries.
3. Session-level write correlation: if agent calls `corvia_write` with content
   similar to recent search results (cosine > 0.7), infer the search was useful.

**Implementation:** Add a `SessionSearchLog` to the gap detector that tracks
per-session query history. On each search, compare against previous queries
in the same session. If similarity > 0.8 = re-search (gap signal). If a write
follows with similar content = useful search (positive signal).

**No explicit rating protocol.** Implicit signals only.

### Implementation locations:
- Extend: `crates/corvia-kernel/src/gap_detector.rs` (SessionSearchLog)
- Integration: `crates/corvia-server/src/mcp.rs` (record session-level signals)

## Design Area 5: Write Deduplication

### Server-side changes

On `corvia_write`, check embedding similarity against existing entries in scope:

1. Embed the new content (already done in `agent_writer.rs`)
2. Search HNSW index for similar entries (top-1, same scope)
3. If cosine similarity > 0.90: return "similar entry exists" with existing entry ID
4. If 0.80 < similarity <= 0.90: warn but still write (informational)
5. If similarity <= 0.80: write normally

**Response format for near-duplicate:**
```json
{
  "content": [{
    "type": "text",
    "text": "Near-duplicate detected (similarity: 0.93). Existing entry: <id>. Content: <preview>. Entry NOT written. To force write, use force_write: true."
  }]
}
```

**Optional:** `force_write: bool` parameter to bypass dedup check.

### Implementation locations:
- Writer: `crates/corvia-kernel/src/agent_writer.rs` (add dedup check)
- MCP handler: `crates/corvia-server/src/mcp.rs` tool_corvia_write() (force_write param)
- Store trait: may need a `search_by_embedding` method on KnowledgeStore

## Protocol Layering (Backward Compatible)

| Layer | What | Who Changes | Non-adopters |
|-------|------|-------------|--------------|
| 0 | MCP tool interface (unchanged) | Nobody | Work as before |
| 1 | Quality signals in responses | Server (additive) | Ignore new fields |
| 2 | Adaptive retrieval behavior | AGENTS.md (opt-in) | Blind-first-call |
| 3 | Context injection | AGENTS.md (opt-in) | No subagent context |
| 4 | Gap signals + write dedup | Server (automatic) | Benefit automatically |

## AGENTS.md Update (Track A)

Add ~15-20 lines to the "Hybrid Tool Usage" section:

```markdown
### Agentic Retrieval Protocol

1. **Check quality signals**: After `corvia_search`, check `quality_signal.confidence`.
   If `low`, follow the `suggestion` field and retry once (max 1 retry).

2. **Inject context into subagents**: Before spawning subagents for non-trivial work,
   call `corvia_context` with `max_tokens` appropriate for the task (recommended:
   2000-3000 tokens) and include the result in the subagent prompt.

3. **Write discipline**: After discovering non-obvious insights, call `corvia_write`
   immediately. The server handles deduplication automatically.
```

## Dependencies

- Issue #12 (buffered access tracking): IMPLEMENTED. Area 4 extends it.
- Issue #37 (BM25 hybrid search): improves retrieval quality baseline.
  Not blocking. Thresholds may need re-tuning after BM25 lands.

## Deliverables

1. [x] Protocol specification (this document)
2. [ ] Quality signals in `corvia_search` (Area 1)
3. [ ] `corvia_context` max_tokens + compact format (Area 2)
4. [ ] Gap detector module + dashboard endpoint (Area 3)
5. [ ] Session search log + feedback signals (Area 4)
6. [ ] Write deduplication (Area 5)
7. [ ] AGENTS.md update (~15-20 lines)
8. [ ] Tests for all new functionality

## Files to Modify

| File | Changes |
|------|---------|
| `crates/corvia-server/src/mcp.rs` | Search quality signals, context params, write dedup, gap recording |
| `crates/corvia-kernel/src/gap_detector.rs` | NEW: gap signal accumulator + session search log |
| `crates/corvia-kernel/src/agent_writer.rs` | Dedup check before insert |
| `crates/corvia-kernel/src/augmenter.rs` | Compact format mode |
| `crates/corvia-kernel/src/rag_types.rs` | QualitySignal struct |
| `crates/corvia-kernel/src/lib.rs` | Export gap_detector module |
| `crates/corvia-server/src/dashboard/mod.rs` | Gaps endpoint |
| `AGENTS.md` | Agentic retrieval protocol section |
| `repos/corvia/AGENTS.md` | Agentic retrieval protocol section |
