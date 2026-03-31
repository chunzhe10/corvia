# Agentic Retrieval Protocol -- Implementation Plan

**Design:** `2026-03-31-agentic-retrieval-protocol-design.md`
**Issue:** chunzhe10/corvia#47

## Task Breakdown

### Task 1: QualitySignal type + computation (kernel)
**File:** `crates/corvia-kernel/src/rag_types.rs`
- Add `QualitySignal` struct: confidence (enum), top_score, result_count, suggestion, gap_detected
- Add `ConfidenceLevel` enum: High, Medium, Low
- Add `QualitySignal::from_results()` constructor that computes signals from search results
- Suggestion generation: template-based from query text
- Thresholds: high >= 0.65 AND count >= 3, medium >= 0.45 OR count >= 3

### Task 2: GapDetector module (kernel)
**New file:** `crates/corvia-kernel/src/gap_detector.rs`
**Modify:** `crates/corvia-kernel/src/lib.rs` (add module export)
- `GapSignal` struct: query, top_score, result_count, timestamp, scope_id
- `GapDetector`: bounded ring buffer (VecDeque, max 1000)
- `record()`: add a gap signal when confidence is low
- `top_gaps()`: aggregate by query similarity (simplified: exact prefix match), return ranked
- Thread-safe via Mutex

### Task 3: Enhance corvia_search with quality signals
**File:** `crates/corvia-server/src/mcp.rs`
- Modify `tool_corvia_search()`: compute QualitySignal from results, include in response JSON
- Add optional `min_score` parameter to tool definition and handler
- Filter results below min_score, report below_threshold_count in quality_signal
- Record gap signals to GapDetector (via AppState) when confidence is low

### Task 4: Enhance corvia_context with max_tokens + compact format
**Files:**
- `crates/corvia-server/src/mcp.rs`: Add `max_tokens` (int, cap 4000) and `format` (string) params
- `crates/corvia-kernel/src/augmenter.rs`: Add `augment_compact()` method to StructuredAugmenter
- `crates/corvia-kernel/src/rag_pipeline.rs`: Accept optional max_tokens override
- `crates/corvia-kernel/src/rag_types.rs`: Add `compact` field to AugmentedContext

Compact format: returns only relevance-ranked content blocks with entry IDs. No system
prompt, no citation headers, no footer.

### Task 5: Write deduplication
**Files:**
- `crates/corvia-kernel/src/agent_writer.rs`: Add `write_with_dedup()` that checks HNSW
  for similar entries before insert
- `crates/corvia-server/src/mcp.rs`: Add `force_write` param, call dedup-aware write
- Threshold: > 0.90 = block (return existing entry info), 0.80-0.90 = warn but write

### Task 6: Wire GapDetector into server + dashboard
**Files:**
- `crates/corvia-server/src/rest.rs`: Add `gap_detector` field to AppState
- `crates/corvia-server/src/dashboard/mod.rs`: Add `GET /api/dashboard/gaps` endpoint
- Server startup: create GapDetector, pass to AppState

### Task 7: AGENTS.md update
**Files:**
- `AGENTS.md` (workspace root): Add "Agentic Retrieval Protocol" section
- `repos/corvia/AGENTS.md`: Add matching section

### Task 8: Tests
- `rag_types.rs`: QualitySignal computation tests
- `gap_detector.rs`: ring buffer, aggregation tests
- `agent_writer.rs`: dedup check tests
- `augmenter.rs`: compact format test
- `mcp.rs` or integration: verify response format includes quality_signal

## Execution Order

Tasks 1 and 2 are independent (kernel-level, no server changes). Task 3 depends on
1 and 2. Task 4 is independent. Task 5 is independent. Task 6 depends on 2. Task 7
is independent. Task 8 is woven into each task.

```
[1: QualitySignal] ─┐
                     ├─→ [3: Search signals] ─┐
[2: GapDetector]  ───┤                         │
                     └─→ [6: Dashboard]        │
[4: Context inject] ──────────────────────────────→ [8: Final tests]
[5: Write dedup]   ──────────────────────────────→
[7: AGENTS.md]     ──────────────────────────────→
```
