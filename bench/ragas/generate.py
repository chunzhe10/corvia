"""Generate a Ragas synthetic testset from `.corvia/entries/*.md`.

See ../../docs/rfcs/2026-04-21-ragas-synthetic-testset-{design,plan}.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import uuid
from pathlib import Path

import corpus


_DEFAULT_DISTRIBUTION = "single=0.5,multi=0.25,abstract=0.25"


def _parse_distribution(spec: str) -> dict[str, float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise SystemExit(
                f"bench/ragas: bad --distribution token '{part}'. "
                "Expected 'key=value,key=value,...'"
            )
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    total = sum(out.values())
    if total <= 0 or abs(total - 1.0) > 0.01:
        raise SystemExit(
            f"bench/ragas: --distribution must sum to ~1.0 (got {total:.3f})"
        )
    return out


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
    p.add_argument(
        "--entries",
        type=Path,
        # generate.py → bench/ragas → bench → corvia → repos → corvia-workspace
        default=Path(__file__).resolve().parents[4] / ".corvia" / "entries",
        help="corvia entries directory (default: <workspace>/.corvia/entries)",
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # 1. Validate entries dir and load corpus.
    if not args.entries.is_dir():
        print(
            f"bench/ragas: --entries dir does not exist: {args.entries}",
            file=sys.stderr,
        )
        return 3

    all_entries = corpus.load_all(args.entries)
    visible, superseded_ids = corpus.partition_visible(all_entries)
    if not visible:
        print(
            f"bench/ragas: no visible entries to generate from (found "
            f"{len(all_entries)} on disk, {len(superseded_ids)} superseded)",
            file=sys.stderr,
        )
        return 3

    chash = corpus.corpus_hash(visible)
    repo_dir = Path(__file__).resolve().parents[2]  # repos/corvia
    corvia_version = _read_corvia_version(repo_dir / "Cargo.toml")
    target_path = args.out / corvia_version / f"{chash}.jsonl"

    print(
        f"bench/ragas: corpus_entry_count={len(all_entries)} "
        f"visible_entry_count={len(visible)} "
        f"superseded_entry_count={len(superseded_ids)}"
    )
    print(f"bench/ragas: corvia_version={corvia_version}")
    print(f"bench/ragas: corpus_hash={chash}")
    print(f"bench/ragas: target_path={target_path}")

    distribution = _parse_distribution(args.distribution)

    if args.dry_run:
        print("bench/ragas: --dry-run set, exiting without LLM calls.")
        return 0

    # 2. Collision guard (cheap; before LLM init).
    if target_path.exists() and not args.force:
        print(
            f"bench/ragas: {target_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 4

    # 3. Import heavy deps lazily so --dry-run stays fast and doesn't require them.
    import providers  # noqa: WPS433
    import writer  # noqa: WPS433
    from langchain_community.cache import SQLiteCache  # noqa: WPS433
    from langchain_core.documents import Document  # noqa: WPS433
    from langchain_core.globals import set_llm_cache  # noqa: WPS433

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(args.cache)))

    llm, embedder = providers.make_provider(args.provider, args.generator_model)

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
    from ragas.testset.synthesizers import (  # noqa: WPS433
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )
    import ragas  # noqa: WPS433 — for __version__

    # Map our short distribution keys → Ragas synthesizer instances.
    synth_map = {
        "single": SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi": MultiHopSpecificQuerySynthesizer(llm=llm),
        "abstract": MultiHopAbstractQuerySynthesizer(llm=llm),
    }
    unknown = set(distribution) - synth_map.keys()
    if unknown:
        print(
            f"bench/ragas: unknown distribution keys {unknown}. "
            f"Valid keys: {sorted(synth_map)}",
            file=sys.stderr,
        )
        return 2
    query_distribution = [
        (synth_map[k], distribution[k]) for k in distribution
    ]

    generator = TestsetGenerator.from_langchain(llm, embedder)
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=args.n,
        query_distribution=query_distribution,
    )

    # 4. Post-process Ragas output into our JSONL schema.
    rows = []
    for sample in dataset.samples:
        eval_sample = sample.eval_sample
        ref_contexts = list(eval_sample.reference_contexts or [])
        source_ids: set[str] = set()
        for ctx in ref_contexts:
            source_ids.update(writer.match_source_entries(ctx, visible))
        rows.append(
            {
                "query_id": f"ragas-{uuid.uuid4().hex[:12]}",
                "query": eval_sample.user_input,
                "reference_answer": eval_sample.reference,
                "reference_contexts": ref_contexts,
                "source_entry_ids": sorted(source_ids),
                "query_type": sample.synthesizer_name,
                "ragas_metadata": {
                    "persona_name": eval_sample.persona_name,
                    "query_style": eval_sample.query_style,
                    "query_length": eval_sample.query_length,
                },
            }
        )

    # 5. Build meta and write.
    meta = {
        "corvia_version": corvia_version,
        "corpus_hash": chash,
        "corpus_entry_count": len(all_entries),
        "visible_entry_count": len(visible),
        "superseded_entry_count": len(superseded_ids),
        "generator_model": f"{args.provider}:{args.generator_model or 'default'}",
        "embedding_model": f"{args.provider}:embedding-default",
        "ragas_version": ragas.__version__,
        "query_distribution": distribution,
        "testset_size": args.n,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generated_by": "bench/ragas/generate.py v1",
        "git_sha": _git_sha(repo_dir),
        "drift_policy": "corpus_hash is audit-only; not validated by downstream eval harness",
    }
    writer.write_testset(args.out, meta, rows, force=args.force)

    print(
        f"bench/ragas: wrote {len(rows)} rows → {target_path}\n"
        f"bench/ragas: spot-check file seeded at "
        f"{args.out / corvia_version / 'spotcheck.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
