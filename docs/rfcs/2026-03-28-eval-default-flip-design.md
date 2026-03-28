# Phase 4: Evaluation + Default Flip -- Design

> **Issue:** #43
> **Depends on:** Phase 3 (Integration + Hot-Swap) -- merged
> **Parent design:** `2026-03-27-hybrid-search-brainstorm.md` -- Evaluation Plan section

## Overview

Build an evaluation framework to A/B benchmark vector-only vs hybrid (vector+BM25+RRF)
vs BM25-only retrieval. If ship gates pass, flip the default config to hybrid.

## Deliverables

### 1. `benchmarks/rag-retrieval/eval.py`

Python script that runs known-answer queries against a running corvia server and
measures retrieval quality metrics.

**Input:** Running server at configurable `--url` (default `http://127.0.0.1:8020`)
**Output:** JSON file with per-query results and aggregate metrics

**Metrics computed:**
- **Source Recall@K**: fraction of expected sources found in top-K results
- **MRR** (Mean Reciprocal Rank): 1/rank of first relevant result, averaged
- **Per-category breakdown**: metrics grouped by query category
- **Latency percentiles**: p50, p95, p99 from retrieval trace

**Query set:** 15 known-answer queries across 5 categories (3 each):
- **architecture**: system design, component interaction, trait patterns
- **feature**: specific features, capabilities, what corvia does
- **config**: configuration keys, values, TOML options
- **api**: MCP tool names, REST endpoints, API behavior
- **performance**: numeric data, latency, resource usage

Each query has:
- `query`: the search string
- `category`: one of the 5 categories
- `expected_sources`: list of substrings that should appear in source file paths
  of relevant results (e.g., `["config.rs"]`, `["mcp.rs", "traits.rs"]`)
- `expected_keywords`: list of keywords that should appear in relevant result content

**Protocol:**
1. Hit `POST /v1/context` with `scope_id`, `limit=10`, query
2. Extract `sources[].source_file` and `sources[].content` from response
3. For each query, check which expected sources appear in top-5 results
4. Compute Recall@5 = |expected found in top 5| / |expected|
5. Compute RR = 1/rank of first expected source found
6. Extract latency from `trace.retrieval`

**Output JSON schema:**
```json
{
  "config": { "url": "...", "scope_id": "...", "searchers": [...], "fusion": "..." },
  "timestamp": "ISO8601",
  "queries": [
    {
      "id": 1,
      "query": "...",
      "category": "config",
      "expected_sources": ["config.rs"],
      "found_sources": ["config.rs", "traits.rs", ...],
      "recall_at_5": 1.0,
      "reciprocal_rank": 1.0,
      "latency_ms": 42,
      "top_5_files": ["config.rs", "lite_store.rs", ...],
      "bm25_latency_ms": null,
      "fusion_latency_ms": null
    }
  ],
  "aggregate": {
    "recall_at_5": 0.67,
    "mrr": 0.72,
    "latency_p50_ms": 35,
    "latency_p95_ms": 48,
    "latency_p99_ms": 52,
    "by_category": {
      "architecture": { "recall_at_5": 0.8, "mrr": 0.9 },
      "feature": { ... },
      ...
    }
  }
}
```

### 2. `benchmarks/rag-retrieval/compare.py`

Reads 2-3 eval result JSON files and produces a comparison report.

**Input:** `python compare.py baseline.json hybrid.json [bm25-only.json]`
**Output:** Printed table + ship gate verdict

**Report includes:**
- Side-by-side aggregate metrics (Recall@5, MRR, latency)
- Per-category comparison
- Per-query delta (which queries improved/regressed)
- Ship gate pass/fail checklist
- Recommendation: flip default or keep opt-in

### 3. `benchmarks/rag-retrieval/run-benchmark.sh`

Orchestrator script that:
1. Records current config
2. Hot-swaps to vector-only, runs eval.py
3. Hot-swaps to hybrid (vector+BM25+RRF), runs eval.py
4. Hot-swaps to BM25-only, runs eval.py
5. Restores original config
6. Runs compare.py on all 3 results

Uses MCP endpoint for hot-swap (with `_meta.confirmed`).

### 4. Default Config Flip (conditional)

If ship gates pass:
- `config.rs`: `default_pipeline_searchers()` -> `["vector", "bm25"]`
- `config.rs`: `default_pipeline_fusion()` -> `"rrf"`
- Enable pipeline by default in `RagConfig::default()`

### 5. Documentation Updates

- `README.md`: pipeline config section, benchmark numbers
- `corvia.toml` template: inline comments for pipeline options
- `ARCHITECTURE.md`: update retrieval pipeline section
- MCP tool descriptions: update to mention hybrid search

## Ship Gates (from RFC)

| Gate | Threshold | Baseline |
|------|-----------|----------|
| Source Recall@5 | > 50% | 37.5% |
| MRR | > 0.60 | 0.544 |
| No category regression | none below baseline | -- |
| Retrieval latency p95 | < 20ms | -- |

## Design Decisions

### D1: Python for eval scripts (not Rust)

Eval scripts are tooling, not product code. Python is faster to iterate,
has better data manipulation (json, statistics), and matches the RFC spec.

### D2: Known-answer queries with source file matching

Expected sources are substring-matched against `source_file` paths in results.
This is more robust than exact UUID matching (entries change across re-ingestion).

### D3: Hot-swap via MCP for identical conditions

All A/B runs use the same server instance with ArcSwap pipeline swap.
No restart between runs means identical HNSW index, tantivy index, and cache state.

### D4: Separate orchestrator from eval

`eval.py` is stateless (hit server, measure, output JSON). `run-benchmark.sh`
handles the hot-swap orchestration. This lets eval.py be reused independently.
