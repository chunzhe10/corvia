# Search Telemetry Extension — Design

**Issue:** [#123](https://github.com/chunzhe10/corvia/issues/123) [eval 1/7]
**Parent:** #122 (RAG eval harness umbrella)
**Author:** chunzhe10 (brainstormed with Claude Code agent)
**Status:** Draft
**Date:** 2026-04-18

## 1. Problem

The `corvia.search` span pipeline currently emits counts (`result_count`, `candidate_count`) and coarse metadata (`query_len`, `kind_filter`, `confidence`), but **loses three pieces of information that every downstream eval ticket needs**:

1. Which **chunks** (not just entries) were returned — `SearchResult` drops `chunk_id` when constructed in `search.rs:418`.
2. The **raw query text** — only `query_len` reaches the root span.
3. The **per-stage scores** (BM25 raw, vector cosine, RRF, reranker) — computed internally and discarded.

Without these, the downstream eval harness (#124–#129) cannot:
- Join traces to an eval golden set by query.
- Compute recall@k at chunk granularity (an entry can have many chunks; matching the right one is the recall signal).
- Diagnose which pipeline stage caused a regression.

## 2. Goals

1. Add `chunk_id` to `SearchResult` as a first-class required field.
2. Record raw `query` and final `result_chunk_ids` as attributes on the `corvia.search` root span.
3. Record each stage's `chunk_ids` and `scores` as attributes on its sub-span (`corvia.search.bm25`, `.vector`, `.fusion`, `.rerank`).
4. Extend the unit test suite to prove the new attrs appear in emitted traces.

## 3. Non-goals

- No change to retrieval logic (ranking, thresholds, reranker behavior).
- No new telemetry backend; continue using the OTLP file exporter.
- **No PII redaction toggle.** The issue's acceptance criterion #2 said "redactable via env flag" — during design review the threat model was re-examined and found moot: corvia is single-user, local, non-hosted; the corpus is known-safe by construction; matched-pair regression works best with raw plaintext queries as join keys. The env-flag knob is pre-building for a hypothetical hosted deployment. If that materializes, adding the toggle is a ~10-line PR. This is an intentional departure from the issue text.
- No explicit `duration_ms` attrs per sub-span. The existing `ParsedTrace.elapsed_ms` (derived from span start/end in `trace.rs:230`) already exposes stage latency. Adding a redundant attribute is dead weight. This is also an intentional departure from the issue text.

## 4. Design

### 4.1 `SearchResult` struct change

`crates/corvia-core/src/types.rs`:

```rust
pub struct SearchResult {
    pub id: String,         // entry_id (unchanged name, avoid rename churn)
    pub chunk_id: String,   // NEW: source chunk within the entry
    pub kind: Kind,
    pub score: f32,
    pub content: String,
}
```

- Plain required field, no `#[serde(default)]`, no `Option<String>`.
- All construction sites (search.rs + tests) updated in one pass; `rustc` enforces completeness.
- No serde backward-compat concern: corvia is pre-1.0, no external consumers of the crate, JSON is never stored-and-reread.
- Schema change for users: index entries do not need to be re-ingested (the index stores chunk_id natively; only the user-facing API adds the field).

### 4.2 Span attribute additions

**Root span `corvia.search`:**

| Attr | Type | Value |
|---|---|---|
| `query` | string | `params.query` verbatim, always |
| `result_chunk_ids` | string | JSON array of final top-K chunk_ids after all truncation |

**Sub-spans** (`bm25`, `vector`, `fusion`, `rerank`):

| Attr | Type | Value |
|---|---|---|
| `chunk_ids` | string | JSON array of chunk_ids at this stage |
| `scores` | string | JSON array of scores parallel to `chunk_ids` |

- Each sub-span records exactly the chunks **that stage produced**, in the order it produced them. Not filtered to final top-K — stage-level regression diagnosis needs visibility into chunks that the stage saw but that a later stage dropped.
- BM25 stage: up to `retrieval_limit` chunks (already sorted by BM25 raw score) and their scores.
- Vector stage: up to `retrieval_limit` chunks (sorted by cosine similarity) and their cosine scores.
- Fusion stage: all fused candidates (the union across BM25 and vector), sorted by RRF score. RRF scores are `f64` internally — cast to `f32` for the telemetry array, acceptable precision loss for diagnostic use.
- Rerank stage: up to `reranker_candidates` chunks (sorted by reranker score). Preserves existing fallback-to-RRF path on reranker failure; the scores recorded will be whatever the code ends up using (reranker or RRF fallback).

The root-span `result_chunk_ids` is the **final** top-K (after min_score/max_tokens truncation), which is the eval join key. Per-stage arrays and the final array are deliberately different: stage arrays answer "what did this stage see?"; final array answers "what did we return?"

**Sub-span removal:** `corvia.search.bm25` currently records `query = %params.query`. This is redundant with the parent span's new `query` attr (join via trace ID). Remove it to avoid duplicating potentially-sensitive content.

### 4.3 Encoding

JSON-string encoding is forced by the OTLP file exporter (`trace.rs:72`): `opentelemetry::Value::Array` falls through to `format!("{:?}", value)` — debug-stringified, not round-trippable. By serializing arrays to JSON strings ourselves and recording as `opentelemetry::Value::String`, the exporter writes them as native `stringValue` and downstream consumers (`corvia_traces` MCP tool, Python eval harness) can `JSON.parse` them.

### 4.4 Helper function (Approach II)

```rust
/// Record parallel `chunk_ids` and `scores` JSON arrays on the given span.
///
/// Encoding choice: the OTLP file exporter serializes `Value::Array` as a
/// debug string, which is not machine-parseable. Encoding as a JSON string
/// round-trips cleanly through `parse_otlp_attribute_value`.
fn record_stage_scores(span: &Span, chunk_ids: &[String], scores: &[f32]) {
    let ids_json = serde_json::to_string(chunk_ids).unwrap_or_else(|_| "[]".to_string());
    let scores_json = serde_json::to_string(scores).unwrap_or_else(|_| "[]".to_string());
    span.record("chunk_ids", ids_json.as_str());
    span.record("scores", scores_json.as_str());
}
```

- Lives in `search.rs` as a private function.
- Sub-span macros must declare `chunk_ids = tracing::field::Empty, scores = tracing::field::Empty` in the `info_span!` call so `.record()` sees the fields.
- `unwrap_or_else` returns `"[]"` on the impossible serde error; preserves attr presence so eval harness code never hits "attr missing" branches.

### 4.5 Data flow in `search_with_handles`

Note: the `query` attr is recorded at step 1 (before cold-start returns early), so even cold-start and zero-result searches carry the query in their trace. Sub-span attrs and `result_chunk_ids` are only written when the respective stages execute, which is correct — empty stages honestly leave the attribute absent.

```
1. Enter `corvia.search` span (instrument macro)
   └── [NEW] Span::current().record("query", params.query.as_str())

2. Cold-start check, drift detection — unchanged

3. BM25 stage (existing info_span!)
   ├── Existing: result_count
   ├── [NEW] record_stage_scores(span, &chunk_ids_from_bm25, &bm25_scores)
   └── [CHANGED] remove `query = %params.query` from info_span! call

4. Vector stage
   ├── Existing: vector_count, result_count
   └── [NEW] record_stage_scores(span, &chunk_ids_from_vector, &cosine_scores)

5. Fusion stage
   ├── Existing: candidate_count
   └── [NEW] record_stage_scores(span, &fused_chunk_ids, &rrf_scores_as_f32)

6. Rerank stage
   ├── Existing: input_count, output_count
   └── [NEW] record_stage_scores(span, &rerank_chunk_ids, &rerank_scores)

7. Dedup, min_score, max_tokens truncation — unchanged

8. Build SearchResult
   └── [CHANGED] include chunk_id in struct literal

9. [NEW] Record result_chunk_ids on root span
   └── Span::current().record("result_chunk_ids", final_chunk_ids_json.as_str())

10. Quality span — unchanged
```

### 4.6 Testing

New unit test in `search.rs`:

1. Set up a fixture: tiny in-memory corpus (3 entries, 5 chunks), build redb+tantivy indexes, construct an Embedder against a mock or stubbed backend.
2. Configure tracing to write to a temp `.jsonl` file via `OtlpFileExporter`.
3. Issue a `search_with_handles()` call.
4. Force-flush the exporter; read the trace file.
5. Assert:
   - Root span `corvia.search` has `query = "<original>"` and `result_chunk_ids` is a parseable JSON array of strings.
   - Each sub-span has `chunk_ids` and `scores` as JSON-string attrs, parseable and same length.
   - `SearchResponse.results[i].chunk_id` is non-empty.

Existing tests (`rrf_fusion_*`, `quality_signal_*`, `compute_quality_signal`) unaffected by this change.

Test harness concern: `search_with_handles` requires a real `Embedder`, which calls fastembed-rs ONNX runtime. The integration test fixture must either:
- Use a small pre-downloaded model (existing pattern — see any existing search integration tests), or
- Skip if the model isn't cached (`#[ignore]`-gate for CI).

Investigation item for planning: check existing search integration tests to reuse their fixture approach.

### 4.7 Performance

Per-search added cost:
- 4 `serde_json::to_string` calls on `Vec<String>` and `Vec<f32>` of size ≤60 — microseconds.
- 8 `Span::record()` calls — negligible (tracing already sinks to a bounded queue).
- 1 additional JSON stringify for `result_chunk_ids` (size ≤10).

Total added latency: <1ms per search, well under the <1% p95 threshold in AC #6 (search p95 is tens to hundreds of ms).

### 4.8 Integration with `corvia_traces` MCP tool

Already handled by the existing implementation:
- `parse_otlp_attribute_value` in `trace.rs:182` returns strings for `stringValue` attrs.
- `TraceEntry.attributes: HashMap<String, serde_json::Value>` already carries arbitrary attr shapes.
- Downstream consumers (MCP clients, eval harness) do `json.loads(attrs["chunk_ids"])` to materialize the array.

No change needed to `corvia_traces` or the MCP tool schema. New fields surface automatically.

## 5. Acceptance criteria mapping

| Original AC | Coverage |
|---|---|
| 1. `SearchResult` includes chunk ID | §4.1 |
| 2. `corvia.search` exports raw query (redactable via env flag) | §4.2 — raw query yes; env flag dropped (see §3) |
| 3. Per-stage scores exported on sub-spans | §4.2, §4.4, §4.5 |
| 4. `corvia_traces` MCP tool shows new fields | §4.8 — automatic |
| 5. Unit test: search produces trace with all new fields | §4.6 |
| 6. No performance regression (<1% p95) | §4.7 |

## 6. Risks

- **Test-fixture complexity.** `search_with_handles` has many moving parts (redb, tantivy, Embedder); building a minimal trace-assert fixture may be the biggest implementation cost. Mitigated by reusing existing integration-test patterns.
- **JSON-string attr convention.** Introducing "JSON-encoded arrays as string attrs" establishes a pattern other spans may adopt. Alternatively, we could fix the OTLP exporter to serialize arrays natively. This design chooses the local workaround to keep scope contained; a proper exporter fix can come later.

## 7. Open items deferred to implementation plan

- Pick the exact test-fixture approach (in-memory subset of an existing test, or new tempdir-based fixture).
- Confirm no other `SearchResult {}` construction sites exist beyond `search.rs` + existing test module. If any, update them in the same PR.
