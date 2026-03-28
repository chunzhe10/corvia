#!/usr/bin/env python3
"""
RAG Retrieval Benchmark Comparison.

Compares 2-3 eval result JSON files and checks ship gates.

Usage:
    python compare.py results/vector-only.json results/hybrid-rrf.json
    python compare.py results/vector-only.json results/hybrid-rrf.json results/bm25-only.json
"""

import argparse
import json
import sys

# ── Ship Gates ──────────────────────────────────────────────────────────────
SHIP_GATES = {
    "recall_at_5": {"threshold": 0.50, "direction": "above", "label": "Source Recall@5 > 50%"},
    "mrr": {"threshold": 0.60, "direction": "above", "label": "MRR > 0.60"},
    "latency_p95_ms": {"threshold": 20, "direction": "below", "label": "Latency p95 < 20ms"},
}

# Baseline values (from initial eval, 2026-03-19)
BASELINE = {
    "recall_at_5": 0.375,
    "mrr": 0.544,
}


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def label_for(result: dict) -> str:
    cfg = result.get("config", {})
    lbl = cfg.get("label", "")
    if lbl:
        return lbl
    searchers = cfg.get("searchers", [])
    fusion = cfg.get("fusion", "")
    if len(searchers) > 1:
        return f"{'+'.join(searchers)}+{fusion}"
    return searchers[0] if searchers else "unknown"


def print_header(labels: list[str]):
    col_w = 16
    header = f"{'Metric':<25s}"
    for lbl in labels:
        header += f"{lbl:>{col_w}s}"
    if len(labels) >= 2:
        header += f"{'Delta':>{col_w}s}"
    print(header)
    print("-" * len(header))


def print_row(metric: str, values: list, fmt: str = ".4f", delta_fmt: str = "+.4f",
              higher_is_better: bool = True):
    col_w = 16
    row = f"{metric:<25s}"
    for v in values:
        row += f"{v:{col_w}{fmt}}"
    if len(values) >= 2:
        delta = values[1] - values[0]
        indicator = ""
        if delta > 0.001:
            indicator = " ^" if higher_is_better else " v"
        elif delta < -0.001:
            indicator = " v" if higher_is_better else " ^"
        delta_str = f"{delta:{delta_fmt}}{indicator}"
        row += f"{delta_str:>{col_w}}"
    print(row)


def check_category_regression(baseline_result: dict, candidate_result: dict) -> list[str]:
    """Check if any category regressed below baseline."""
    regressions = []
    base_cats = baseline_result["aggregate"]["by_category"]
    cand_cats = candidate_result["aggregate"]["by_category"]

    for cat in base_cats:
        if cat in cand_cats:
            base_r = base_cats[cat]["recall_at_5"]
            cand_r = cand_cats[cat]["recall_at_5"]
            if cand_r < base_r - 0.01:  # 1% tolerance
                regressions.append(
                    f"{cat}: Recall@5 regressed {base_r:.2f} -> {cand_r:.2f}"
                )
            base_m = base_cats[cat]["mrr"]
            cand_m = cand_cats[cat]["mrr"]
            if cand_m < base_m - 0.01:
                regressions.append(
                    f"{cat}: MRR regressed {base_m:.2f} -> {cand_m:.2f}"
                )
    return regressions


def main():
    parser = argparse.ArgumentParser(description="Compare RAG retrieval benchmarks")
    parser.add_argument("files", nargs="+", help="Eval result JSON files (baseline first)")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Need at least 2 result files to compare.", file=sys.stderr)
        sys.exit(1)

    results = [load_result(f) for f in args.files]
    labels = [label_for(r) for r in results]

    print()
    print("=" * 70)
    print("  RAG Retrieval Benchmark Comparison")
    print("=" * 70)
    print()

    # ── Aggregate Metrics ──
    print("## Aggregate Metrics")
    print()
    print_header(labels)

    aggs = [r["aggregate"] for r in results]
    print_row("Recall@5", [a["recall_at_5"] for a in aggs])
    print_row("MRR", [a["mrr"] for a in aggs])
    print_row("Keyword Recall", [a["keyword_recall"] for a in aggs])
    print_row("Latency p50 (ms)", [a["latency_p50_ms"] for a in aggs],
              fmt=".1f", delta_fmt="+.1f", higher_is_better=False)
    print_row("Latency p95 (ms)", [a["latency_p95_ms"] for a in aggs],
              fmt=".1f", delta_fmt="+.1f", higher_is_better=False)
    print_row("Latency p99 (ms)", [a["latency_p99_ms"] for a in aggs],
              fmt=".1f", delta_fmt="+.1f", higher_is_better=False)
    print()

    # ── Per-Category Breakdown ──
    print("## Per-Category Breakdown")
    print()
    categories = sorted(set(
        cat for r in results for cat in r["aggregate"]["by_category"]
    ))

    for cat in categories:
        print(f"  {cat}:")
        for i, r in enumerate(results):
            cat_data = r["aggregate"]["by_category"].get(cat, {})
            print(f"    {labels[i]:>20s}:  R@5={cat_data.get('recall_at_5', 0):.2f}  "
                  f"MRR={cat_data.get('mrr', 0):.2f}  "
                  f"KW={cat_data.get('keyword_recall', 0):.2f}")
    print()

    # ── Per-Query Deltas ──
    if len(results) >= 2:
        print("## Per-Query Comparison (baseline vs candidate)")
        print()
        base_queries = {q["id"]: q for q in results[0]["queries"]}
        cand_queries = {q["id"]: q for q in results[1]["queries"]}

        improved = []
        regressed = []
        unchanged = []

        for qid in sorted(base_queries.keys()):
            bq = base_queries[qid]
            cq = cand_queries.get(qid)
            if not cq:
                continue
            delta_r = cq["recall_at_5"] - bq["recall_at_5"]
            delta_m = cq["reciprocal_rank"] - bq["reciprocal_rank"]

            entry = {
                "id": qid,
                "query": bq["query"][:55],
                "category": bq["category"],
                "base_r": bq["recall_at_5"],
                "cand_r": cq["recall_at_5"],
                "delta_r": delta_r,
                "delta_m": delta_m,
            }

            if delta_r > 0.01 or delta_m > 0.01:
                improved.append(entry)
            elif delta_r < -0.01 or delta_m < -0.01:
                regressed.append(entry)
            else:
                unchanged.append(entry)

        if improved:
            print(f"  Improved ({len(improved)} queries):")
            for e in improved:
                print(f"    Q{e['id']:2d} [{e['category']:12s}] R@5: {e['base_r']:.2f}->{e['cand_r']:.2f} "
                      f"(+{e['delta_r']:.2f})  {e['query']}")
        if regressed:
            print(f"  Regressed ({len(regressed)} queries):")
            for e in regressed:
                print(f"    Q{e['id']:2d} [{e['category']:12s}] R@5: {e['base_r']:.2f}->{e['cand_r']:.2f} "
                      f"({e['delta_r']:.2f})  {e['query']}")
        if unchanged:
            print(f"  Unchanged ({len(unchanged)} queries)")
        print()

    # ── Ship Gate Check ──
    print("=" * 70)
    print("  SHIP GATE CHECK")
    print("=" * 70)
    print()

    # Check gates against the candidate (second file, index 1)
    candidate = results[1] if len(results) >= 2 else results[0]
    candidate_agg = candidate["aggregate"]
    candidate_label = labels[1] if len(labels) >= 2 else labels[0]

    all_pass = True

    for key, gate in SHIP_GATES.items():
        value = candidate_agg.get(key, 0)
        if gate["direction"] == "above":
            passed = value >= gate["threshold"]
        else:
            passed = value <= gate["threshold"]

        status = "PASS" if passed else "FAIL"
        icon = "[x]" if passed else "[ ]"
        print(f"  {icon} {gate['label']:30s}  actual={value:.4f}  {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    # Category regression check
    if len(results) >= 2:
        regressions = check_category_regression(results[0], results[1])
        no_regression = len(regressions) == 0
        icon = "[x]" if no_regression else "[ ]"
        print(f"  {icon} {'No category regression':30s}  {'PASS' if no_regression else 'FAIL'}")
        if regressions:
            for r in regressions:
                print(f"      - {r}")
            all_pass = False
    else:
        print(f"  [?] {'No category regression':30s}  SKIP (need baseline)")

    print()

    # ── Verdict ──
    print("=" * 70)
    if all_pass:
        print(f"  VERDICT: ALL GATES PASS for '{candidate_label}'")
        print(f"  RECOMMENDATION: Flip default config to hybrid search")
    else:
        print(f"  VERDICT: GATES DID NOT PASS for '{candidate_label}'")
        print(f"  RECOMMENDATION: Keep BM25 as opt-in, investigate improvements")
    print("=" * 70)
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
