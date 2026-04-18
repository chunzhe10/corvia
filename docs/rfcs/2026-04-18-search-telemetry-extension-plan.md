# Search Telemetry Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add chunk-level identity, raw query, and per-stage scores to `corvia.search` telemetry so downstream eval tickets (#124–#129) can mine traces for recall@k, matched-pair regression, and stage-level diagnostics.

**Architecture:** Additive struct field on `SearchResult` (`chunk_id: String`); one private helper fn `record_stage_scores` on a `Span` that JSON-encodes parallel `chunk_ids` + `scores` arrays; root span records raw `query` + final `result_chunk_ids`. No retrieval-logic changes.

**Tech Stack:** Rust 2024, `tracing` + `tracing-opentelemetry` + `opentelemetry_sdk`, existing `OtlpFileExporter` (in `corvia-core/src/trace.rs`), existing `TestHarness` fixture in `corvia-core/tests/common/mod.rs`.

**Design doc:** `repos/corvia/docs/rfcs/2026-04-18-search-telemetry-extension-design.md`

**Branch:** `feat/123-telemetry-search-span` (already created; design doc committed at `9420b92`).

**Repo root for all commands:** `/workspaces/corvia-workspace/repos/corvia/`

---

## Context the engineer must know

1. **Crate layout**: `corvia-core` holds retrieval logic (search, types, trace exporter). `corvia-cli` holds binary, telemetry init (`init_telemetry`), and the MCP server. The unit-test-side subscriber wiring lives in `corvia-cli`; tests inside `corvia-core` cannot rely on `init_telemetry` and must stand up their own provider.

2. **Telemetry pipeline**: `tracing::info_span!` → `tracing-opentelemetry` bridge → `opentelemetry_sdk::SdkTracerProvider` → `simple_exporter(OtlpFileExporter)` → JSON line in `.corvia/traces.jsonl`. `simple_exporter` flushes synchronously on span close.

3. **Span recording rules**: `Span::current().record("name", value)` only works if the span declared `name = tracing::field::Empty` (or some literal) at creation time. Fields not pre-declared are silently dropped. Every new attr this plan adds MUST be declared in the `info_span!`/`#[tracing::instrument]` macro first.

4. **OTLP array quirk**: The exporter at `crates/corvia-core/src/trace.rs:72` serializes `opentelemetry::Value::Array` via `format!("{:?}", value)` — debug-stringified, not machine-parseable. This is why the design encodes arrays as JSON strings locally via `serde_json::to_string` and records them as `stringValue` attrs. Do NOT pass `&[String]` directly to `.record()`.

5. **Construction sites of `SearchResult`**: Verified grep shows exactly **one** production construction site: `crates/corvia-core/src/search.rs:418`. No Rust test code constructs `SearchResult` as a struct literal — tests read `.id`, `.score`, `.kind`, `.content` via field access. Adding a required `chunk_id` field will therefore only break that one site.

6. **Integration test gate**: Tests that need an `Embedder` are `#[ignore]`-gated throughout this codebase (see `tests/integration.rs:29`, `src/search.rs:604`). Follow that convention — CI doesn't run them; local runs use `cargo test -- --ignored`.

---

## File plan

| File | Change |
|---|---|
| `crates/corvia-core/src/types.rs` | Add `pub chunk_id: String` to `SearchResult`. Update serde roundtrip test if needed. |
| `crates/corvia-core/src/search.rs` | Six targeted changes (see Tasks 2–6). Add one private helper fn `record_stage_scores` + a pure `encode_stage_scores` for testability. |
| `crates/corvia-core/Cargo.toml` | Add `tracing-opentelemetry` and `tracing-subscriber` to `[dev-dependencies]` (the main `tracing-subscriber` is already a regular dep; confirm). |
| `crates/corvia-core/tests/integration.rs` | Add one new `#[ignore]` integration test `search_emits_eval_telemetry_attributes`. |

Total: 3 files modified + 1 test added.

---

## Task 1: Add `chunk_id` field to `SearchResult`

**Files:**
- Modify: `crates/corvia-core/src/types.rs` (struct at line 101)
- Modify: `crates/corvia-core/src/search.rs:418` (only construction site; will fail to compile until fixed → proves nothing else constructs this struct)

- [ ] **Step 1: Write the failing test**

Append this test at the end of the `tests` module in `crates/corvia-core/src/types.rs` (before the final `}`):

```rust
    #[test]
    fn search_result_carries_chunk_id() {
        let r = SearchResult {
            id: "entry-1".to_string(),
            chunk_id: "entry-1:3".to_string(),
            kind: Kind::Learning,
            score: 0.5,
            content: "body".to_string(),
        };
        assert_eq!(r.chunk_id, "entry-1:3");
        assert_eq!(r.id, "entry-1");
    }
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cargo test -p corvia-core --lib types::tests::search_result_carries_chunk_id 2>&1 | tail -30`

Expected: compile error `missing field 'chunk_id' in initializer of 'SearchResult'` OR `struct 'SearchResult' has no field named 'chunk_id'` (depending on which line rustc reaches first — both prove the field isn't there yet).

- [ ] **Step 3: Add the field**

Modify `crates/corvia-core/src/types.rs` — replace the `SearchResult` struct at line 101 with:

```rust
/// A single search result returned by the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Entry ID (UUIDv7) of the source entry.
    pub id: String,
    /// Chunk ID of the specific chunk retrieved from the entry. Format `<entry_id>:<chunk_index>`.
    pub chunk_id: String,
    pub kind: Kind,
    pub score: f32,
    pub content: String,
}
```

- [ ] **Step 4: Update the single production construction site**

Modify `crates/corvia-core/src/search.rs` — replace the block at lines 418-423:

```rust
        results.push(SearchResult {
            id: entry_id,
            chunk_id: chunk_id.clone(),
            kind,
            score,
            content,
        });
```

Note: the loop at line 412 destructures `(chunk_id, entry_id, score, content)` — `chunk_id` is already in scope. Clone it because the original binding is consumed by nothing else in this loop iteration but we take ownership of `entry_id`/`content` in the struct literal.

Actually verify: the tuple is destructured with `for (chunk_id, entry_id, score, content) in scored_results`, and the values are moved into the struct. Since `entry_id` and `content` are `String` (moved), and `score` is `f32` (Copy), the `chunk_id: String` is still available to be moved. We can move it in too, no clone needed. Use:

```rust
        results.push(SearchResult {
            id: entry_id,
            chunk_id,
            kind,
            score,
            content,
        });
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p corvia-core --lib types::tests::search_result_carries_chunk_id`

Expected: `test result: ok. 1 passed; 0 failed`.

- [ ] **Step 6: Verify no other callers broke**

Run: `cargo build --workspace 2>&1 | tail -20`

Expected: clean compile (no errors). If errors appear in `corvia-cli` or elsewhere, investigate — this means a `SearchResult { ... }` construction site exists that the grep missed; fix it by adding `chunk_id:` to the literal.

- [ ] **Step 7: Run full unit test suite**

Run: `cargo test -p corvia-core --lib 2>&1 | tail -10`

Expected: all pre-existing lib tests pass (the `#[ignore]` integration tests remain skipped).

- [ ] **Step 8: Commit**

```bash
git add crates/corvia-core/src/types.rs crates/corvia-core/src/search.rs
git commit -m "feat: add chunk_id to SearchResult for eval telemetry

Closes AC #1 of #123."
```

---

## Task 2: Pure JSON-encoding helper + unit tests

**Files:**
- Modify: `crates/corvia-core/src/search.rs` (add pure fn + unit tests in the existing `#[cfg(test)] mod tests`)

Rationale: the actual `Span::record` call needs a tracing subscriber to observe. Extracting the encoding as a pure fn lets us test the JSON output shape without any subscriber machinery.

- [ ] **Step 1: Write failing tests for the pure encoder**

Append to the `#[cfg(test)] mod tests { ... }` block in `crates/corvia-core/src/search.rs`, just before the closing `}`:

```rust
    #[test]
    fn encode_stage_scores_empty() {
        let (ids, scores) = encode_stage_scores(&[], &[]);
        assert_eq!(ids, "[]");
        assert_eq!(scores, "[]");
    }

    #[test]
    fn encode_stage_scores_parallel_arrays() {
        let chunk_ids = vec!["a:0".to_string(), "b:1".to_string(), "c:2".to_string()];
        let scores = vec![0.9f32, 0.5, 0.1];
        let (ids_json, scores_json) = encode_stage_scores(&chunk_ids, &scores);
        assert_eq!(ids_json, r#"["a:0","b:1","c:2"]"#);
        assert_eq!(scores_json, "[0.9,0.5,0.1]");
    }

    #[test]
    fn encode_stage_scores_length_mismatch_still_encodes_both() {
        // Caller's responsibility to pass parallel arrays; function must not crash.
        let chunk_ids = vec!["a".to_string()];
        let scores = vec![0.1f32, 0.2, 0.3];
        let (ids_json, scores_json) = encode_stage_scores(&chunk_ids, &scores);
        assert_eq!(ids_json, r#"["a"]"#);
        assert_eq!(scores_json, "[0.1,0.2,0.3]");
    }
```

- [ ] **Step 2: Confirm the test fails to compile**

Run: `cargo test -p corvia-core --lib search::tests::encode_stage_scores_empty 2>&1 | tail -15`

Expected: `error[E0425]: cannot find function 'encode_stage_scores' in this scope`.

- [ ] **Step 3: Add the pure encoder near the top of `search.rs` helpers**

Add after the `FusedCandidate` struct definition (around line 40) in `crates/corvia-core/src/search.rs`:

```rust
/// Encode parallel `chunk_ids` and `scores` arrays as JSON strings.
///
/// The OTLP file exporter serializes `opentelemetry::Value::Array` via debug
/// formatting (`trace.rs:72`), producing strings that are not machine-parseable.
/// Encoding as JSON strings and recording them as `stringValue` attrs lets
/// downstream consumers parse via `json::parse` cleanly.
///
/// Returns `("[]", "[]")` on any (impossible-for-these-types) serde error
/// so that attr presence is preserved even in unreachable edge cases.
fn encode_stage_scores(chunk_ids: &[String], scores: &[f32]) -> (String, String) {
    let ids = serde_json::to_string(chunk_ids).unwrap_or_else(|_| "[]".to_string());
    let sc = serde_json::to_string(scores).unwrap_or_else(|_| "[]".to_string());
    (ids, sc)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p corvia-core --lib search::tests::encode_stage_scores`

Expected: `test result: ok. 3 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add crates/corvia-core/src/search.rs
git commit -m "feat: add encode_stage_scores pure helper for span attrs

Encodes parallel chunk_ids+scores arrays as JSON strings to work around
the OTLP file exporter's debug-stringification of Value::Array. Tested
in isolation; Span::record wiring comes in subsequent tasks."
```

---

## Task 3: Record raw `query` on the root search span

**Files:**
- Modify: `crates/corvia-core/src/search.rs` — the `#[tracing::instrument]` attribute at line 177 and the function body.

- [ ] **Step 1: Declare `query` as an Empty field on the root span macro**

In `crates/corvia-core/src/search.rs`, replace the `#[tracing::instrument]` attribute at lines 177-183:

```rust
#[tracing::instrument(name = "corvia.search", skip(config, base_dir, embedder, params, redb, tantivy), fields(
    query = tracing::field::Empty,
    query_len = params.query.len(),
    limit = params.limit,
    kind_filter = ?params.kind,
    result_count = tracing::field::Empty,
    result_chunk_ids = tracing::field::Empty,
    confidence = tracing::field::Empty,
))]
```

(Adds `query` and `result_chunk_ids` as empty placeholders alongside existing fields. `result_chunk_ids` is filled in Task 5; `query` right now.)

- [ ] **Step 2: Record the raw query at the top of `search_with_handles`**

Insert as the first executable line inside `search_with_handles` (after the opening brace on line 191, before the cold-start check):

```rust
    // Record raw query for eval mining. Design RFC §4.2: no redaction toggle —
    // corvia is single-user local; raw query is the eval join key.
    Span::current().record("query", params.query.as_str());
```

- [ ] **Step 3: Build & test**

Run: `cargo build -p corvia-core 2>&1 | tail -10`

Expected: clean compile.

Run: `cargo test -p corvia-core --lib 2>&1 | tail -10`

Expected: all non-`#[ignore]` tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/corvia-core/src/search.rs
git commit -m "feat: record raw query on corvia.search root span

Per RFC §4.2. Raw query is the eval join key for matched-pair
regression downstream (#125-#129). No redaction toggle — corvia is
single-user local; threat model doesn't require it.

Partial: AC #2 of #123 (query attr; result_chunk_ids in Task 5)."
```

---

## Task 4: Wire per-stage score recording in sub-spans

**Files:**
- Modify: `crates/corvia-core/src/search.rs` — the bm25, vector, fusion, rerank sub-span blocks.

This task touches four stages. Each stage follows an identical pattern:
1. Declare `chunk_ids = Empty, scores = Empty` on the `info_span!`.
2. Collect chunk_ids and scores from the stage's output into parallel `Vec`s.
3. Call `encode_stage_scores` + `Span::current().record(...)` inside the stage block (before the scope closes).

Also: **remove** the redundant `query = %params.query` from the bm25 sub-span (now on parent).

- [ ] **Step 1: Add a private helper that records both attrs on the current span**

Insert right after the `encode_stage_scores` function added in Task 2 (same region of `search.rs`):

```rust
/// Record `chunk_ids` and `scores` as JSON-string attrs on the given span.
/// Both fields must have been declared on the `info_span!` with `tracing::field::Empty`.
fn record_stage_scores(span: &tracing::Span, chunk_ids: &[String], scores: &[f32]) {
    let (ids_json, scores_json) = encode_stage_scores(chunk_ids, scores);
    span.record("chunk_ids", ids_json.as_str());
    span.record("scores", scores_json.as_str());
}
```

- [ ] **Step 2: Wire BM25 stage**

Find lines 237-245 in `search.rs`. Replace the entire BM25 block with:

```rust
    // Step 5: BM25 search.
    let bm25_results = {
        let _span = info_span!(
            "corvia.search.bm25",
            result_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let results = tantivy
            .search(&params.query, params.kind, retrieval_limit)
            .context("BM25 search")?;
        Span::current().record("result_count", results.len());

        // Record chunk_ids + BM25 raw scores for eval mining.
        let ids: Vec<String> = results.iter().map(|(cid, _, _)| cid.clone()).collect();
        let scores: Vec<f32> = results.iter().map(|(_, _, s)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &scores);

        debug!(count = results.len(), "BM25 results");
        results
    };
```

Changes vs. original (lines 237-245):
- Removed `query = %params.query` from the span macro (redundant with root).
- Added `chunk_ids = Empty, scores = Empty` declarations.
- Added the `ids`/`scores` collection + `record_stage_scores` call.

- [ ] **Step 3: Wire vector stage**

Find the vector block starting at line 248. Replace the `info_span!` macro call and add the score-recording block. The replacement block:

```rust
    // Step 6: Vector search.
    let vector_scored = {
        let _span = info_span!(
            "corvia.search.vector",
            vector_count = tracing::field::Empty,
            result_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let query_vector = embedder
            .embed(&params.query)
            .context("embedding search query")?;

        let all_vectors = redb.all_vectors().context("loading all vectors from redb")?;
        let superseded_ids = redb.superseded_ids().context("loading superseded IDs")?;
        Span::current().record("vector_count", all_vectors.len());

        let mut scored: Vec<(String, String, f32)> = Vec::new();
        for (chunk_id, vector) in &all_vectors {
            let entry_id = match redb.chunk_entry_id(chunk_id)? {
                Some(eid) => eid,
                None => continue,
            };
            if superseded_ids.contains(&entry_id) {
                continue;
            }
            if let Some(ref kind_filter) = params.kind {
                if let Ok(Some(chunk_kind_str)) = redb.get_chunk_kind(chunk_id) {
                    if let Ok(chunk_kind) = chunk_kind_str.parse::<Kind>() {
                        if chunk_kind != *kind_filter {
                            continue;
                        }
                    }
                }
            }
            let similarity = Embedder::cosine_similarity(&query_vector, vector);
            scored.push((chunk_id.clone(), entry_id, similarity));
        }
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(retrieval_limit);
        Span::current().record("result_count", scored.len());

        // Record chunk_ids + cosine scores for eval mining.
        let ids: Vec<String> = scored.iter().map(|(cid, _, _)| cid.clone()).collect();
        let scores_vec: Vec<f32> = scored.iter().map(|(_, _, s)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        debug!(count = scored.len(), "vector results");
        scored
    };
```

- [ ] **Step 4: Wire fusion stage**

Find the fusion block at line 287. Replace with:

```rust
    // Step 7: RRF fusion.
    let fused = {
        let _span = info_span!(
            "corvia.search.fusion",
            candidate_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let result = rrf_fusion(&bm25_results, &vector_scored, config.search.rrf_k);
        Span::current().record("candidate_count", result.len());

        // Record chunk_ids + RRF scores (f64 → f32) for eval mining.
        let ids: Vec<String> = result.iter().map(|c| c.chunk_id.clone()).collect();
        let scores_vec: Vec<f32> = result.iter().map(|c| c.rrf_score as f32).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        debug!(count = result.len(), "fused candidates");
        result
    };
```

Note: RRF scores are `f64` internally (for rank sum precision); the cast to `f32` is a documented precision loss for telemetry only — retrieval ranking uses `f64` throughout.

- [ ] **Step 5: Wire rerank stage**

The rerank block starts at line 300 with `info_span!("corvia.search.rerank", input_count = ..., output_count = Empty)`. The pattern is the same but the bound variable is `results` (a `Vec<(String, String, f32, String)>` where `.0` is chunk_id, `.2` is score).

Replace the `info_span!` macro call and add score recording right before `Span::current().record("output_count", ...)`. The final span header:

```rust
        let _span = info_span!(
            "corvia.search.rerank",
            input_count = top_candidates.len(),
            output_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
```

At the END of the rerank block, just before `Span::current().record("output_count", results.len()); results`, insert:

```rust
        // Record chunk_ids + reranker (or RRF-fallback) scores for eval mining.
        let ids: Vec<String> = results.iter().map(|(cid, _, _, _)| cid.clone()).collect();
        let rerank_scores: Vec<f32> = results.iter().map(|(_, _, s, _)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &rerank_scores);
```

So the tail of the rerank block becomes:

```rust
        // Record chunk_ids + reranker (or RRF-fallback) scores for eval mining.
        let ids: Vec<String> = results.iter().map(|(cid, _, _, _)| cid.clone()).collect();
        let rerank_scores: Vec<f32> = results.iter().map(|(_, _, s, _)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &rerank_scores);

        Span::current().record("output_count", results.len());
        results
    };
```

- [ ] **Step 6: Build & test**

Run: `cargo build -p corvia-core 2>&1 | tail -10`

Expected: clean compile.

Run: `cargo test -p corvia-core --lib 2>&1 | tail -15`

Expected: all non-ignored tests pass; RRF/quality/encode tests still pass.

Run: `cargo clippy -p corvia-core --all-targets 2>&1 | tail -20`

Expected: no new warnings. Pay attention to any `unused_variables` — the `ids` and `scores_vec` shadowing across stage blocks should be fine since they're scope-local to each `let { ... }` block.

- [ ] **Step 7: Commit**

```bash
git add crates/corvia-core/src/search.rs
git commit -m "feat: record per-stage chunk_ids and scores on search sub-spans

Each of corvia.search.{bm25,vector,fusion,rerank} now carries
chunk_ids + scores as JSON-string attrs parallel to the stage's
output. Also drops the redundant query attr from bm25 (now on parent).

Closes AC #3 of #123."
```

---

## Task 5: Record `result_chunk_ids` on the root span

**Files:**
- Modify: `crates/corvia-core/src/search.rs` — around line 409 (after `final_scores` is collected, before/after `SearchResult` construction).

The instrument macro was already updated in Task 3 to declare `result_chunk_ids = Empty`.

- [ ] **Step 1: Collect final chunk_ids alongside `final_scores`**

Find line 409 in `search.rs`:

```rust
    // Step 12: Build SearchResult for each.
    let final_scores: Vec<f32> = scored_results.iter().map(|(_, _, s, _)| *s).collect();
```

Replace with:

```rust
    // Step 12: Build SearchResult for each.
    let final_scores: Vec<f32> = scored_results.iter().map(|(_, _, s, _)| *s).collect();
    let final_chunk_ids: Vec<String> =
        scored_results.iter().map(|(cid, _, _, _)| cid.clone()).collect();
```

- [ ] **Step 2: Record the JSON-encoded chunk_ids on the root span**

Find the line around 439-440:

```rust
    Span::current().record("result_count", results.len());
    Span::current().record("confidence", tracing::field::debug(quality.confidence));
```

Insert directly above these (after the quality block closes):

```rust
    // Record final (post-truncation) chunk_ids for downstream eval join.
    let result_chunk_ids_json =
        serde_json::to_string(&final_chunk_ids).unwrap_or_else(|_| "[]".to_string());
    Span::current().record("result_chunk_ids", result_chunk_ids_json.as_str());
```

Final section becomes:

```rust
    // Record final (post-truncation) chunk_ids for downstream eval join.
    let result_chunk_ids_json =
        serde_json::to_string(&final_chunk_ids).unwrap_or_else(|_| "[]".to_string());
    Span::current().record("result_chunk_ids", result_chunk_ids_json.as_str());

    Span::current().record("result_count", results.len());
    Span::current().record("confidence", tracing::field::debug(quality.confidence));

    info!(
        results = results.len(),
        confidence = ?quality.confidence,
        "search complete"
    );

    Ok(SearchResponse { results, quality })
```

- [ ] **Step 3: Build & test**

Run: `cargo build -p corvia-core 2>&1 | tail -10`

Expected: clean compile.

Run: `cargo test -p corvia-core --lib 2>&1 | tail -10`

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/corvia-core/src/search.rs
git commit -m "feat: record result_chunk_ids on corvia.search root span

Final (post-truncation) top-K chunk_ids encoded as JSON-string attr
for eval-harness join with an external golden set. Per RFC §4.2.

Completes AC #2 of #123 (with Task 3's query attr)."
```

---

## Task 6 [POC]: End-to-end trace-capture integration test

**[POC] justification:** This test exercises the full tracing stack (tracing + tracing-opentelemetry + SdkTracerProvider + OtlpFileExporter + tempfile I/O + tokio). Two assumptions the POC validates:

1. **`tracing::subscriber::with_default(subscriber, || { ... })` properly scopes OTel span emission to the closure.** In particular, that spans emitted from inside the closure are routed to the local subscriber's `OpenTelemetryLayer`, and that the closure's thread-local dispatch dance doesn't drop spans.
2. **A locally-constructed `SdkTracerProvider` with `with_simple_exporter(OtlpFileExporter)` flushes synchronously on span close, so that reading the tempfile immediately after the search call returns complete data.**

Both are likely true based on how the binary's `init_telemetry` is structured, but neither has been exercised from a pure unit-test context. If either fails, fall back to: mark the test `#[ignore]`, delete the file-readback, and cover the new attrs via a separate test that uses a custom `SpanExporter` mock storing spans in a `Mutex<Vec<SpanData>>`. That fallback is documented inline.

**Files:**
- Modify: `crates/corvia-core/Cargo.toml` — add `tracing-opentelemetry` dev-dep.
- Modify: `crates/corvia-core/tests/integration.rs` — add new test.

- [ ] **Step 1: Add `tracing-opentelemetry` as a dev-dep**

Modify `crates/corvia-core/Cargo.toml`. The existing `[dev-dependencies]` section has one line (`tempfile = "3"`). Replace it with:

```toml
[dev-dependencies]
tempfile = "3"
tracing-opentelemetry = { workspace = true }
```

(`workspace = true` uses the version pinned in the root `Cargo.toml` workspace deps table — already `"0.29"`.)

- [ ] **Step 2: Write the failing integration test**

Add to `crates/corvia-core/tests/integration.rs`, appending after the last test (the file is ~900 lines; add at the bottom, before the final `}` of any module if present, or at the top-level if tests are flat):

```rust
// ---------------------------------------------------------------------------
// Telemetry: #123 [eval 1/7]
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model + writes a tempfile
fn search_emits_eval_telemetry_attributes() {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::layer::SubscriberExt;
    use corvia_core::trace::{read_recent_traces, OtlpFileExporter};

    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    h.copy_fixtures();

    // Ingest fixtures so the search has something to retrieve.
    let _ingest = corvia_core::ingest::ingest(config, base, false)
        .expect("ingest failed");

    // Set up a local tracer provider pointing at a tempfile.
    let trace_dir = tempfile::tempdir().unwrap();
    let trace_path = trace_dir.path().join("traces.jsonl");
    let file_exporter = OtlpFileExporter::new(trace_path.clone())
        .expect("failed to create file exporter");
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(file_exporter)
        .build();
    let tracer = provider.tracer("corvia-test");

    // Compose: subscriber with the otel layer so tracing spans reach the provider.
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let subscriber = tracing_subscriber::registry().with(otel_layer);

    let embedder = make_embedder(config);

    // Run search inside the subscriber scope.
    tracing::subscriber::with_default(subscriber, || {
        let params = SearchParams {
            query: "why did we choose Redb".to_string(),
            limit: 5,
            max_tokens: None,
            min_score: None,
            kind: None,
        };
        let response = search(config, base, &embedder, &params).unwrap();
        assert!(
            !response.results.is_empty(),
            "search should return results from fixtures"
        );
        assert!(
            response.results.iter().all(|r| !r.chunk_id.is_empty()),
            "every SearchResult must carry a non-empty chunk_id"
        );
    });

    // Flush: drop the provider to shut down and flush buffered spans.
    drop(provider);

    // Read back the trace file and locate the corvia.search root span.
    let traces = read_recent_traces(&trace_path, 200);
    assert!(
        !traces.is_empty(),
        "trace file should contain at least one span; path: {}",
        trace_path.display()
    );

    let root = traces
        .iter()
        .find(|t| t.name == "corvia.search")
        .expect("corvia.search root span not found in trace file");

    // Root span: raw query present
    let query_attr = root
        .attributes
        .get("query")
        .and_then(|v| v.as_str())
        .expect("corvia.search.query attr missing or wrong type");
    assert_eq!(query_attr, "why did we choose Redb");

    // Root span: result_chunk_ids is a JSON-array string
    let ids_attr = root
        .attributes
        .get("result_chunk_ids")
        .and_then(|v| v.as_str())
        .expect("corvia.search.result_chunk_ids attr missing");
    let ids: Vec<String> = serde_json::from_str(ids_attr)
        .expect("result_chunk_ids must parse as JSON string array");
    assert!(
        !ids.is_empty(),
        "result_chunk_ids should be non-empty for a successful search"
    );
    assert!(
        ids.iter().all(|c| c.contains(':')),
        "chunk_ids should look like '<entry>:<idx>': {ids:?}"
    );

    // Each sub-span: chunk_ids + scores present as JSON-string arrays of equal length
    for stage in ["corvia.search.bm25", "corvia.search.vector", "corvia.search.fusion", "corvia.search.rerank"] {
        let sub = traces
            .iter()
            .find(|t| t.name == stage)
            .unwrap_or_else(|| panic!("sub-span {stage} not found"));
        let cids_raw = sub
            .attributes
            .get("chunk_ids")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{stage}.chunk_ids missing"));
        let scores_raw = sub
            .attributes
            .get("scores")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{stage}.scores missing"));
        let cids: Vec<String> = serde_json::from_str(cids_raw)
            .unwrap_or_else(|e| panic!("{stage}.chunk_ids bad JSON: {e}"));
        let scores: Vec<f32> = serde_json::from_str(scores_raw)
            .unwrap_or_else(|e| panic!("{stage}.scores bad JSON: {e}"));
        assert_eq!(
            cids.len(),
            scores.len(),
            "{stage}: chunk_ids and scores must be parallel (same length)"
        );
    }
}
```

- [ ] **Step 3: Verify the test fails on a clean build (if run)**

Run: `cargo test -p corvia-core --test integration search_emits_eval_telemetry_attributes -- --ignored 2>&1 | tail -25`

Expected: either:
- **PASS**: POC validated — wiring works. Proceed.
- **FAIL**: inspect the error. If the failure is "trace file is empty" → the scoped subscriber isn't routing to our provider. Fix: replace `with_default` with `global_default` and accept that this test cannot be run in parallel with other tracing tests (cargo runs `#[ignore]` tests serially when invoked with `-- --ignored --test-threads=1`). If the failure is "attr missing" in a particular stage → that stage's wiring is wrong, jump back to Task 4 for that stage and fix.

Document the outcome in the commit message (below).

- [ ] **Step 4: If POC needed adjustment, iterate**

If `with_default` didn't work, switch to this top-of-test pattern:

```rust
let _dispatch = tracing::dispatcher::set_default(&tracing::Dispatch::new(subscriber));
// ... run search ...
```

Or set globally (serial tests only):

```rust
tracing::subscriber::set_global_default(subscriber)
    .expect("only one test may set the global subscriber");
```

Pick whichever the error output suggests. Document the choice in a comment at the test head.

- [ ] **Step 5: Verify the test passes**

Run: `cargo test -p corvia-core --test integration search_emits_eval_telemetry_attributes -- --ignored --nocapture 2>&1 | tail -30`

Expected: `test result: ok. 1 passed; 0 failed`.

- [ ] **Step 6: Verify no other tests regressed**

Run: `cargo test -p corvia-core -- --ignored --test-threads=1 2>&1 | tail -30`

Expected: all pre-existing ignored integration tests continue to pass (they don't use the new attrs but they exercise the same search path).

- [ ] **Step 7: Commit**

```bash
git add crates/corvia-core/Cargo.toml crates/corvia-core/tests/integration.rs
git commit -m "test: add E2E trace-capture test for eval telemetry

Adds tracing-opentelemetry dev-dep and an ignored integration test
that asserts the new search telemetry attrs (query, result_chunk_ids,
per-stage chunk_ids/scores) appear in the OTLP file exporter output.

Closes AC #5 of #123."
```

---

## Task 7: Final verification & cleanup

- [ ] **Step 1: Full build**

Run: `cargo build --workspace 2>&1 | tail -10`

Expected: clean compile across `corvia-core` and `corvia-cli`.

- [ ] **Step 2: All lib + unit tests**

Run: `cargo test --workspace --lib 2>&1 | tail -10`

Expected: all pass, zero failures.

- [ ] **Step 3: Ignored integration tests**

Run: `cargo test --workspace -- --ignored --test-threads=1 2>&1 | tail -30`

Expected: all pass. If the pre-existing ingest/search tests fail unexpectedly, investigate — the changes should not affect retrieval behavior.

- [ ] **Step 4: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -30`

Expected: no warnings. If warnings appear, fix them before concluding. Most likely: `unused_variables` on helper-introduced bindings — add `_` prefix or the `#[allow(unused)]` attribute only if the binding is genuinely unused.

- [ ] **Step 5: Manual smoke (optional but recommended)**

Run one ingest + search against a live corpus to visually inspect the new trace file:

```bash
cd /workspaces/corvia-workspace/repos/corvia
rm -rf /tmp/corvia-smoke && mkdir -p /tmp/corvia-smoke/.corvia/entries
cp crates/corvia-core/tests/fixtures/*.md /tmp/corvia-smoke/.corvia/entries/
cargo run --release -- --trace-file /tmp/corvia-smoke/traces.jsonl ingest /tmp/corvia-smoke
cargo run --release -- --trace-file /tmp/corvia-smoke/traces.jsonl search "redb" --base-dir /tmp/corvia-smoke
# Inspect:
tail -n 20 /tmp/corvia-smoke/traces.jsonl | jq '.name, .attributes[] | select(.key=="query" or .key=="result_chunk_ids" or .key=="chunk_ids" or .key=="scores")'
```

Expected to see: `query` as a string, `result_chunk_ids` as a JSON-array-string, per-stage `chunk_ids`+`scores` on each sub-span.

(Verify the CLI actually takes `--trace-file` and `--base-dir` flags — if not, drop this step or adapt.)

- [ ] **Step 6: Confirm no commit was missed**

Run: `git status && git log --oneline origin/master..HEAD`

Expected: clean working tree; commits visible since branch-off.

- [ ] **Step 7: Summary commit (only if cleanup was needed)**

If any clippy fixes or smoke-test-driven adjustments were made, commit them:

```bash
git add -A
git commit -m "chore: polish per verification pass

<short description of what was fixed>"
```

If nothing was needed, skip this step.

---

## Self-review checklist (engineer to run before declaring plan complete)

- [ ] Every AC in the design doc (§5) maps to a task. AC #1 → Task 1; AC #2 → Task 5; AC #3 → Task 4; AC #4 → automatic via existing `corvia_traces` MCP tool (verify in Task 6 via smoke); AC #5 → Task 6; AC #6 → not explicitly benchmarked; the added code is a handful of JSON encodings per search (see RFC §4.7) — if perf is critical, add a `cargo bench` pass.
- [ ] No tasks depend on fields or functions defined in later tasks (dependency order: 1 → 2 → 3 → 4 → 5 → 6 → 7 is strictly forward).
- [ ] `encode_stage_scores` signature is consistent: `(&[String], &[f32]) -> (String, String)` everywhere used.
- [ ] `record_stage_scores` takes `&tracing::Span` (not `Span`) — verify in Task 4 Step 1.
- [ ] All new span fields (`query`, `result_chunk_ids`, `chunk_ids`, `scores`) are declared as `tracing::field::Empty` on their parent macro before any `.record()` call.

---

## Out-of-scope (deferred)

- Fixing `OtlpFileExporter` to handle `Value::Array` natively (RFC §6.2 risk). Tracked separately if adopted.
- AC #6 performance benchmark (<1% p95 delta). Added code cost is negligible by inspection; a formal bench lives in the broader #122 eval series (#128 `corvia bench` CLI).
- Adding redaction / env flag for raw query (deliberate YAGNI, RFC §3).
- Explicit `duration_ms` sub-span attrs (deliberate YAGNI, RFC §3).
