#!/usr/bin/env python3
"""
RAG Retrieval Evaluation Framework.

Runs known-answer queries against a corvia server and measures retrieval quality.
Outputs structured JSON with Source Recall@K, MRR, latency percentiles, and
per-category breakdowns.

Usage:
    python eval.py --output results/vector-only.json
    python eval.py --url http://localhost:8020 --scope-id corvia --output results.json
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Known-Answer Query Set ──────────────────────────────────────────────────
# 15 queries across 5 categories (3 each).
# expected_sources: substrings matched against source_file paths in results.
# expected_keywords: keywords matched against result content (case-insensitive).

QUERIES = [
    # ── Architecture (how components work together) ──
    {
        "id": 1,
        "query": "How does the RAG pipeline orchestrate retrieval, augmentation, and generation stages?",
        "category": "architecture",
        "expected_sources": ["rag_pipeline"],
        "expected_keywords": ["retriever", "augmenter", "generation"],
    },
    {
        "id": 2,
        "query": "How do LiteStore and PostgresStore implement the QueryableStore trait?",
        "category": "architecture",
        "expected_sources": ["lite_store", "store"],
        "expected_keywords": ["QueryableStore", "insert", "search"],
    },
    {
        "id": 3,
        "query": "How does the agent staging system isolate writes and handle merge conflicts?",
        "category": "architecture",
        "expected_sources": ["staging", "merge", "agent"],
        "expected_keywords": ["staging", "merge", "branch"],
    },
    # ── Feature (what features exist, capabilities) ──
    {
        "id": 4,
        "query": "What graph traversal and relationship operations does corvia support?",
        "category": "feature",
        "expected_sources": ["graph", "README"],
        "expected_keywords": ["edges", "traverse", "graph"],
    },
    {
        "id": 5,
        "query": "How does the chunking pipeline handle different document formats?",
        "category": "feature",
        "expected_sources": ["chunking"],
        "expected_keywords": ["chunk", "format"],
    },
    {
        "id": 6,
        "query": "What temporal queries does corvia support like as_of and history?",
        "category": "feature",
        "expected_sources": ["temporal", "README"],
        "expected_keywords": ["as_of", "history"],
    },
    # ── Config (configuration keys, values, TOML options) ──
    {
        "id": 7,
        "query": "What is graph_alpha and how does it affect retrieval scoring?",
        "category": "config",
        "expected_sources": ["retriever", "config", "rag"],
        "expected_keywords": ["graph_alpha", "0.3"],
    },
    {
        "id": 8,
        "query": "How do I configure pipeline searchers and fusion in corvia.toml?",
        "category": "config",
        "expected_sources": ["config", "pipeline", "hybrid"],
        "expected_keywords": ["searchers", "fusion", "pipeline"],
    },
    {
        "id": 9,
        "query": "What embedding model and dimensions does corvia use by default?",
        "category": "config",
        "expected_sources": ["config", "inference", "README"],
        "expected_keywords": ["nomic", "768", "embedding"],
    },
    # ── API (MCP tool names, REST endpoints, API behavior) ──
    {
        "id": 10,
        "query": "What parameters does corvia_search accept: query, scope_id, limit?",
        "category": "api",
        "expected_sources": ["mcp", "ARCHITECTURE", "README"],
        "expected_keywords": ["corvia_search", "query", "scope_id"],
    },
    {
        "id": 11,
        "query": "How does corvia_config_set handle hot-reload and ArcSwap pipeline swap?",
        "category": "api",
        "expected_sources": ["mcp", "config", "pipeline"],
        "expected_keywords": ["config_set", "pipeline", "swap"],
    },
    {
        "id": 12,
        "query": "What REST endpoints does the corvia server expose for health and context?",
        "category": "api",
        "expected_sources": ["server", "rest", "README"],
        "expected_keywords": ["health", "context"],
    },
    # ── Performance (numeric data, latency, resource usage) ──
    {
        "id": 13,
        "query": "What are the HNSW index parameters ef_construction and ef_search?",
        "category": "performance",
        "expected_sources": ["lite_store"],
        "expected_keywords": ["ef_construction", "ef_search"],
    },
    {
        "id": 14,
        "query": "How does the tiered storage system manage hot, warm, and cold entries?",
        "category": "performance",
        "expected_sources": ["ARCHITECTURE", "lite_store", "tier"],
        "expected_keywords": ["hot", "warm", "cold"],
    },
    {
        "id": 15,
        "query": "What is the tantivy BM25 index commit batching strategy?",
        "category": "performance",
        "expected_sources": ["tantivy", "lite_store", "hybrid"],
        "expected_keywords": ["tantivy", "commit"],
    },
]


def run_query(url: str, scope_id: str, query: str, limit: int = 10) -> dict:
    """Run a single context query against the server."""
    payload = json.dumps({
        "query": query,
        "scope_id": scope_id,
        "limit": limit,
        "expand_graph": True,
    }).encode()

    req = Request(
        f"{url}/v1/context",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    wall_ms = (time.monotonic() - start) * 1000

    return {**data, "_wall_ms": wall_ms}


def check_source_match(source_file: str, expected: str) -> bool:
    """Check if expected substring appears in source_file path."""
    if source_file is None:
        return False
    return expected.lower() in source_file.lower()


def check_keyword_match(content: str, keyword: str) -> bool:
    """Check if keyword appears in content (case-insensitive)."""
    if content is None:
        return False
    return keyword.lower() in content.lower()


def compute_query_metrics(query_def: dict, response: dict, k: int = 5) -> dict:
    """Compute Recall@K and Reciprocal Rank for a single query."""
    sources = response.get("sources", [])
    top_k = sources[:k]

    # Extract source files
    top_k_files = [s.get("source_file", "unknown") for s in top_k]
    all_files = [s.get("source_file", "unknown") for s in sources]

    # Source Recall@K: what fraction of expected sources appear in top-K results?
    expected = query_def["expected_sources"]
    found_in_top_k = 0
    for exp in expected:
        if any(check_source_match(f, exp) for f in top_k_files):
            found_in_top_k += 1
    recall_at_k = found_in_top_k / len(expected) if expected else 0.0

    # Reciprocal Rank: 1/rank of first relevant result
    rr = 0.0
    for rank, source in enumerate(sources, 1):
        sf = source.get("source_file", "")
        if any(check_source_match(sf, exp) for exp in expected):
            rr = 1.0 / rank
            break

    # Keyword coverage in top-K content
    expected_kw = query_def.get("expected_keywords", [])
    kw_found = 0
    for kw in expected_kw:
        for s in top_k:
            if check_keyword_match(s.get("content", ""), kw):
                kw_found += 1
                break
    keyword_recall = kw_found / len(expected_kw) if expected_kw else 0.0

    # Latency from trace
    trace = response.get("trace", {})
    retrieval = trace.get("retrieval", {})
    latency_ms = retrieval.get("latency_ms", 0)

    # Pipeline-mode fields (may be absent for legacy retriever)
    bm25_latency = retrieval.get("bm25_latency_ms")
    bm25_results = retrieval.get("bm25_results")
    fusion_method = retrieval.get("fusion_method")
    fusion_latency = retrieval.get("fusion_latency_ms")
    stages = retrieval.get("stages")

    return {
        "id": query_def["id"],
        "query": query_def["query"],
        "category": query_def["category"],
        "expected_sources": expected,
        "found_sources": all_files[:10],
        "recall_at_5": round(recall_at_k, 4),
        "reciprocal_rank": round(rr, 4),
        "keyword_recall": round(keyword_recall, 4),
        "latency_ms": latency_ms,
        "top_5_files": top_k_files,
        "retriever_name": retrieval.get("retriever_name", "unknown"),
        "vector_results": retrieval.get("vector_results", 0),
        "graph_expanded": retrieval.get("graph_expanded", 0),
        "bm25_latency_ms": bm25_latency,
        "bm25_results": bm25_results,
        "fusion_method": fusion_method,
        "fusion_latency_ms": fusion_latency,
        "stages": stages,
    }


def compute_aggregate(query_results: list) -> dict:
    """Compute aggregate metrics across all queries."""
    recalls = [q["recall_at_5"] for q in query_results]
    rrs = [q["reciprocal_rank"] for q in query_results]
    kw_recalls = [q["keyword_recall"] for q in query_results]
    latencies = [q["latency_ms"] for q in query_results if q["latency_ms"] > 0]

    # Per-category
    categories = {}
    for q in query_results:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = {"recalls": [], "rrs": [], "kw_recalls": []}
        categories[cat]["recalls"].append(q["recall_at_5"])
        categories[cat]["rrs"].append(q["reciprocal_rank"])
        categories[cat]["kw_recalls"].append(q["keyword_recall"])

    by_category = {}
    for cat, data in sorted(categories.items()):
        by_category[cat] = {
            "recall_at_5": round(statistics.mean(data["recalls"]), 4),
            "mrr": round(statistics.mean(data["rrs"]), 4),
            "keyword_recall": round(statistics.mean(data["kw_recalls"]), 4),
            "query_count": len(data["recalls"]),
        }

    latency_sorted = sorted(latencies) if latencies else [0]

    def percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        d = k - f
        return round(data[f] + d * (data[c] - data[f]), 1)

    return {
        "recall_at_5": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "keyword_recall": round(statistics.mean(kw_recalls), 4),
        "latency_p50_ms": percentile(latency_sorted, 50),
        "latency_p95_ms": percentile(latency_sorted, 95),
        "latency_p99_ms": percentile(latency_sorted, 99),
        "query_count": len(query_results),
        "by_category": by_category,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--url", default="http://127.0.0.1:8020", help="Server URL")
    parser.add_argument("--scope-id", default="corvia", help="Scope ID")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=10, help="Results per query")
    parser.add_argument("--label", default="", help="Label for this run (e.g., 'vector-only')")
    args = parser.parse_args()

    # Fetch current config for metadata
    try:
        req = Request(
            f"{args.url}/mcp",
            data=json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "corvia_config_get", "arguments": {
                    "scope_id": args.scope_id, "section": "rag"
                }}
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            mcp_resp = json.loads(resp.read())
        rag_config = json.loads(mcp_resp["result"]["content"][0]["text"])
    except Exception as e:
        print(f"Warning: could not fetch config: {e}", file=sys.stderr)
        rag_config = {}

    pipeline = rag_config.get("pipeline", {})
    config_meta = {
        "url": args.url,
        "scope_id": args.scope_id,
        "label": args.label,
        "searchers": pipeline.get("searchers", ["vector"]),
        "fusion": pipeline.get("fusion", "passthrough"),
        "expander": pipeline.get("expander", "graph"),
        "retriever": rag_config.get("retriever", "unknown"),
    }

    print(f"Eval config: searchers={config_meta['searchers']}, "
          f"fusion={config_meta['fusion']}, retriever={config_meta['retriever']}")

    # Run all queries
    query_results = []
    for qdef in QUERIES:
        print(f"  Query {qdef['id']:2d}/{len(QUERIES)}: {qdef['query'][:60]}...", end=" ")
        try:
            response = run_query(args.url, args.scope_id, qdef["query"], args.limit)
            metrics = compute_query_metrics(qdef, response)
            query_results.append(metrics)
            print(f"R@5={metrics['recall_at_5']:.2f}  RR={metrics['reciprocal_rank']:.2f}  "
                  f"{metrics['latency_ms']}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            query_results.append({
                "id": qdef["id"], "query": qdef["query"],
                "category": qdef["category"],
                "expected_sources": qdef["expected_sources"],
                "found_sources": [], "recall_at_5": 0, "reciprocal_rank": 0,
                "keyword_recall": 0, "latency_ms": 0, "top_5_files": [],
                "retriever_name": "error", "vector_results": 0,
                "graph_expanded": 0, "bm25_latency_ms": None,
                "bm25_results": None, "fusion_method": None,
                "fusion_latency_ms": None, "stages": None,
                "error": str(e),
            })

    # Compute aggregates
    aggregate = compute_aggregate(query_results)

    result = {
        "config": config_meta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries": query_results,
        "aggregate": aggregate,
    }

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print()
    print(f"{'='*60}")
    print(f"  Recall@5:  {aggregate['recall_at_5']:.4f}")
    print(f"  MRR:       {aggregate['mrr']:.4f}")
    print(f"  Keyword:   {aggregate['keyword_recall']:.4f}")
    print(f"  Latency:   p50={aggregate['latency_p50_ms']}ms  "
          f"p95={aggregate['latency_p95_ms']}ms  p99={aggregate['latency_p99_ms']}ms")
    print()
    for cat, data in sorted(aggregate["by_category"].items()):
        print(f"  {cat:15s}  R@5={data['recall_at_5']:.2f}  MRR={data['mrr']:.2f}  "
              f"KW={data['keyword_recall']:.2f}")
    print(f"{'='*60}")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
