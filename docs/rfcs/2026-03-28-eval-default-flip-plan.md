# Phase 4: Evaluation + Default Flip -- Implementation Plan

> **Design:** `2026-03-28-eval-default-flip-design.md`

## Task Breakdown

### Task 1: Create eval.py (eval framework)
**Files:** `benchmarks/rag-retrieval/eval.py`
**Acceptance:** Script runs against live server, outputs valid JSON with all metrics.

- Define 15 known-answer queries with expected sources
- Implement `/v1/context` query runner
- Compute Source Recall@5 per query
- Compute MRR (Mean Reciprocal Rank)
- Extract latency from trace (including pipeline-mode fields)
- Compute latency percentiles (p50, p95, p99)
- Per-category aggregation
- CLI args: `--url`, `--scope-id`, `--output`, `--limit`
- Output structured JSON

### Task 2: Create compare.py (comparison tool)
**Files:** `benchmarks/rag-retrieval/compare.py`
**Acceptance:** Takes 2-3 eval JSONs, prints comparison table, checks ship gates.

- Load multiple result files
- Side-by-side aggregate metrics table
- Per-category comparison
- Per-query deltas (improved/regressed)
- Ship gate pass/fail checklist
- Exit code 0 if all gates pass, 1 if any fail

### Task 3: Create run-benchmark.sh (orchestrator)
**Files:** `benchmarks/rag-retrieval/run-benchmark.sh`
**Acceptance:** Runs full A/B benchmark end-to-end with hot-swap.

- Save current config
- Hot-swap to vector-only, run eval.py
- Hot-swap to hybrid (vector+BM25+RRF), run eval.py
- Hot-swap to BM25-only, run eval.py
- Restore original config
- Run compare.py on results
- Output results to timestamped directory

### Task 4: Run benchmarks and evaluate ship gates
**Acceptance:** Benchmark results recorded, ship gate verdict documented.

- Execute run-benchmark.sh against workspace server
- Review results
- Document findings

### Task 5: Flip defaults (conditional on ship gates)
**Files:** `crates/corvia-common/src/config.rs`
**Acceptance:** Default config uses hybrid search. Existing tests still pass.

- Change `default_pipeline_searchers()` to `["vector", "bm25"]`
- Change `default_pipeline_fusion()` to `"rrf"`
- Enable pipeline by default in RagConfig
- Run `cargo test` to verify no regressions

### Task 6: Update documentation
**Files:** `README.md`, `ARCHITECTURE.md`, MCP tool descriptions
**Acceptance:** Docs reflect hybrid search as default.

- README: add pipeline config section with benchmark numbers
- ARCHITECTURE.md: update retrieval pipeline section
- MCP tool descriptions: update corvia_search/context/ask descriptions
- corvia.toml template: add pipeline config comments

### Task 7: Commit and verify
- Commit after each logical unit
- Run full test suite (cargo test + clippy)
- Verify hot-swap still works with new defaults

## Execution Order
Tasks 1-3 are independent (can be parallelized).
Task 4 depends on 1-3.
Task 5 depends on 4 (conditional).
Task 6 depends on 5.
Task 7 runs throughout.
