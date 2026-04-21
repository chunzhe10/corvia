"""Generate a Ragas synthetic testset from `.corvia/entries/*.md`.

See ../../docs/rfcs/2026-04-21-ragas-synthetic-testset-{design,plan}.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import corpus


_DEFAULT_DISTRIBUTION = "single=0.5,multi=0.25,abstract=0.25"

# Distribution short keys (CLI-facing) → Ragas synthesizer identifiers. Both
# the JSONL `query_type` field and the meta.json `query_distribution` dict
# use the Ragas names so downstream tooling sees a single vocabulary.
_DIST_KEY_MAP = {
    "single": "single_hop_specific",
    "multi": "multi_hop_specific",
    "abstract": "multi_hop_abstract",
}

# Collision guard ordering invariant (see tests/test_cli.py:
# test_output_exists_exits_four_without_force). The collision check MUST
# run before any provider init so the exit(4) path works without a key.
_COLLISION_BEFORE_PROVIDER_INIT = True


def _preflight_python() -> None:
    """Refuse to run on Python 3.13+ where Ragas wheels are not validated.

    POC (docs/rfcs/2026-04-21-ragas-synthetic-testset-poc-findings.md)
    confirmed Ragas 0.4.3 cleanly installs on 3.12; 3.13 is untested here.
    Operators should use `uv venv --python 3.12 .venv`.
    """
    if sys.version_info[:2] >= (3, 13):
        print(
            "bench/ragas: Python 3.13+ detected. Ragas 0.4.x is pinned for 3.12. "
            "Create a 3.12 venv: `uv venv --python 3.12 .venv && source .venv/bin/activate`.",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_distribution(spec: str) -> dict[str, float]:
    """Parse `single=0.5,multi=0.25,abstract=0.25` into a normalized dict.

    Output uses Ragas synthesizer keys (`single_hop_specific`, etc.). Rejects
    duplicate keys, unknown keys, and sums far from 1.0.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    raw: dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise SystemExit(
                f"bench/ragas: bad --distribution token '{part}'. "
                "Expected 'key=value,key=value,...'"
            )
        k, v = part.split("=", 1)
        k = k.strip()
        if k in raw:
            raise SystemExit(
                f"bench/ragas: duplicate --distribution key '{k}'"
            )
        raw[k] = float(v)

    unknown = set(raw) - set(_DIST_KEY_MAP)
    if unknown:
        raise SystemExit(
            f"bench/ragas: unknown --distribution keys {sorted(unknown)}. "
            f"Valid: {sorted(_DIST_KEY_MAP)}"
        )

    total = sum(raw.values())
    if total <= 0 or abs(total - 1.0) > 0.01:
        raise SystemExit(
            f"bench/ragas: --distribution must sum to ~1.0 (got {total:.3f})"
        )

    return {_DIST_KEY_MAP[k]: v for k, v in raw.items()}


def _read_corvia_version(cargo_toml: Path) -> str:
    import tomllib

    data = tomllib.loads(cargo_toml.read_text())
    return data["workspace"]["package"]["version"]


def _git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve_entries_default() -> Path | None:
    """Locate a .corvia/entries dir in either workspace or standalone layout.

    Workspace: corvia-workspace/repos/corvia/bench/ragas/generate.py → parents[4].
    Standalone: corvia/bench/ragas/generate.py → parents[2].
    Returns the first that exists, else None (caller emits a clear error).
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / ".corvia" / "entries",  # workspace
        here.parents[2] / ".corvia" / "entries",  # standalone repo
    ]
    return next((c for c in candidates if c.is_dir()), None)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate a Ragas synthetic testset from corvia entries.",
    )
    p.add_argument("--n", type=int, default=50, help="Testset size (default: 50)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "eval_sets",
        help="Output dir (default: ../eval_sets)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent / ".cache" / "ragas-llm.sqlite",
        help="LangChain SQLite LLM cache path",
    )
    default_entries = _resolve_entries_default()
    p.add_argument(
        "--entries",
        type=Path,
        default=default_entries,
        help="corvia entries directory "
        f"(default: {default_entries or '<pass explicitly>'})",
    )
    p.add_argument(
        "--provider",
        choices=("gemini", "openai", "anthropic"),
        default="gemini",
    )
    p.add_argument(
        "--generator-model",
        default=None,
        help="Override provider default model name",
    )
    p.add_argument(
        "--distribution",
        default=_DEFAULT_DISTRIBUTION,
        help=f"Query-type weights (default: {_DEFAULT_DISTRIBUTION})",
    )
    p.add_argument(
        "--min-rows-frac",
        type=float,
        default=0.8,
        help="Fail if Ragas produces fewer than --n * this many rows (default: 0.8)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, partition, hash — no LLM calls, no output written",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing testset for this corpus snapshot",
    )
    return p


def _content_query_id(query: str, answer: str) -> str:
    """Deterministic 12-char query ID from content (identical runs → identical IDs)."""
    h = hashlib.sha256(f"{query}\0{answer}".encode("utf-8")).hexdigest()
    return f"ragas-{h[:12]}"


def _build_rows(samples: list[Any], visible: list[corpus.Entry]) -> tuple[list[dict], Counter]:
    """Post-process Ragas samples into our JSONL row schema.

    Returns (rows, match_method_counts). Keeps the loop out of main()
    so it's unit-testable with fake samples.
    """
    import writer  # noqa: WPS433 — lazy import

    rows: list[dict] = []
    method_counts: Counter = Counter()
    for sample in samples:
        eval_sample = sample.eval_sample
        ref_contexts = list(eval_sample.reference_contexts or [])
        source_ids: set[str] = set()
        methods: list[str] = []
        for ctx in ref_contexts:
            result = writer.match_source_entries(ctx, visible)
            source_ids.update(result.ids)
            methods.append(result.method)
            method_counts[result.method] += 1
        query_text = eval_sample.user_input or ""
        answer_text = eval_sample.reference or ""
        rows.append(
            {
                "schema_version": writer.SCHEMA_VERSION,
                "query_id": _content_query_id(query_text, answer_text),
                "query": query_text,
                "reference_answer": answer_text,
                "reference_contexts": ref_contexts,
                "source_entry_ids": sorted(source_ids),
                "query_type": sample.synthesizer_name,
                "ragas_metadata": {
                    "persona_name": eval_sample.persona_name,
                    "query_style": eval_sample.query_style,
                    "query_length": eval_sample.query_length,
                    "match_methods": methods,
                },
            }
        )
    return rows, method_counts


def main(argv: list[str] | None = None) -> int:
    _preflight_python()
    args = _build_parser().parse_args(argv)

    # 1. Validate entries dir and load corpus.
    if args.entries is None or not args.entries.is_dir():
        print(
            f"bench/ragas: --entries dir does not exist: {args.entries}. "
            "Pass --entries explicitly if you're running from a standalone checkout.",
            file=sys.stderr,
        )
        return 3

    try:
        all_entries = corpus.load_all(args.entries)
    except ValueError as err:
        print(f"bench/ragas: corpus load failed: {err}", file=sys.stderr)
        return 3

    part = corpus.partition_visible(all_entries)
    visible = part.visible
    superseded_ids = part.superseded_ids
    dangling = sorted(part.dangling_superseded_refs)

    if not visible:
        print(
            f"bench/ragas: no visible entries to generate from (found "
            f"{len(all_entries)} on disk, {len(superseded_ids)} superseded)",
            file=sys.stderr,
        )
        return 3
    if dangling:
        print(
            f"bench/ragas: WARN {len(dangling)} supersedes references point to "
            f"unknown IDs: {dangling}",
            file=sys.stderr,
        )

    chash = corpus.corpus_hash(visible)
    repo_dir = Path(__file__).resolve().parents[2]  # repos/corvia
    corvia_version = _read_corvia_version(repo_dir / "Cargo.toml")
    target_path = args.out / corvia_version / f"{chash}.jsonl"

    print(
        f"bench/ragas: corpus_entry_count={len(all_entries)} "
        f"visible_entry_count={len(visible)} "
        f"superseded_entry_count={len(superseded_ids)} "
        f"dangling_superseded_refs={len(dangling)}"
    )
    print(f"bench/ragas: corvia_version={corvia_version}")
    print(f"bench/ragas: corpus_hash={chash}")
    print(f"bench/ragas: target_path={target_path}")

    distribution = _parse_distribution(args.distribution)

    if args.dry_run:
        print("bench/ragas: --dry-run set, exiting without LLM calls.")
        return 0

    # 2. Collision guard — MUST run before provider init (exit(4) without key).
    if target_path.exists() and not args.force:
        print(
            f"bench/ragas: {target_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 4

    # 3. Lazy imports so --dry-run and error paths stay fast.
    import providers  # noqa: WPS433
    import writer  # noqa: WPS433
    from langchain_community.cache import SQLiteCache  # noqa: WPS433
    from langchain_core.documents import Document  # noqa: WPS433
    from langchain_core.globals import set_llm_cache  # noqa: WPS433

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(args.cache)))

    bundle = providers.make_provider(args.provider, args.generator_model)

    documents = [
        Document(
            page_content=e.body,
            metadata={
                "entry_id": e.id,
                "kind": e.kind,
                "tags": e.tags,
                "created_at": e.created_at,
            },
        )
        for e in visible
    ]

    from ragas.testset import TestsetGenerator  # noqa: WPS433
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType  # noqa: WPS433
    from ragas.testset.synthesizers import (  # noqa: WPS433
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )
    from ragas.testset.transforms import (  # noqa: WPS433
        apply_transforms,
        default_transforms_for_prechunked,
    )
    import ragas  # noqa: WPS433

    generator = TestsetGenerator.from_langchain(bundle.llm, bundle.embedder)

    # We treat each corvia entry as an atomic CHUNK and bypass Ragas'
    # HeadlineSplitter (unreliable on ~500-token docs with varying
    # markdown structure — raises "'headlines' property not found"). Our
    # entries are small and self-contained, so pre-chunked semantics fit.
    nodes = [
        Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        for doc in documents
    ]
    generator.knowledge_graph = KnowledgeGraph(nodes=nodes)
    transforms = default_transforms_for_prechunked(
        llm=generator.llm,
        embedding_model=generator.embedding_model,
    )

    # distribution keys are Ragas names after _parse_distribution normalization.
    synth_map = {
        "single_hop_specific": SingleHopSpecificQuerySynthesizer(llm=generator.llm),
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer(llm=generator.llm),
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer(llm=generator.llm),
    }
    query_distribution = [(synth_map[k], distribution[k]) for k in distribution]

    start = time.monotonic()
    apply_transforms(generator.knowledge_graph, transforms)
    dataset = generator.generate(
        testset_size=args.n,
        query_distribution=query_distribution,
    )
    elapsed = time.monotonic() - start

    samples = list(dataset.samples)
    min_rows = max(1, int(args.n * args.min_rows_frac))
    if len(samples) < min_rows:
        print(
            f"bench/ragas: Ragas produced only {len(samples)} samples for --n {args.n} "
            f"(below --min-rows-frac {args.min_rows_frac} floor of {min_rows}). "
            "Not writing output. Check rate limits, distribution, or corpus size.",
            file=sys.stderr,
        )
        return 5

    rows, method_counts = _build_rows(samples, visible)
    empty_source_rows = sum(1 for r in rows if not r["source_entry_ids"])
    if empty_source_rows:
        print(
            f"bench/ragas: WARN {empty_source_rows}/{len(rows)} rows have empty "
            "source_entry_ids (recall@k cannot be computed for these)",
            file=sys.stderr,
        )

    meta = {
        "schema_version": writer.SCHEMA_VERSION,
        "corvia_version": corvia_version,
        "corpus_hash": chash,
        "corpus_entry_count": len(all_entries),
        "visible_entry_count": len(visible),
        "superseded_entry_count": len(superseded_ids),
        "dangling_superseded_refs": dangling,
        "generator_model": bundle.generator_model,
        "embedding_model": bundle.embedding_model,
        "ragas_version": ragas.__version__,
        "query_distribution": distribution,
        "testset_size": args.n,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generated_by": "bench/ragas/generate.py v1",
        "git_sha": _git_sha(repo_dir),
        "drift_policy": (
            "corpus_hash is audit-only; not validated by downstream eval harness"
        ),
        "match_method_counts": dict(method_counts),
        "rows_with_empty_source_ids": empty_source_rows,
        "generation_seconds": round(elapsed, 2),
    }
    writer.write_testset(args.out, meta, rows, force=args.force)

    print(
        f"bench/ragas: wrote {len(rows)} rows → {target_path}\n"
        f"bench/ragas: spot-check file at "
        f"{args.out / corvia_version / 'spotcheck.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
