"""Output writers + source-entry matcher for bench/ragas.

Writes `{out_dir}/{corvia_version}/{corpus_hash}.jsonl` (one row per line),
its `.meta.json` sidecar, and on first write a blank `spotcheck.md`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import corpus


SCHEMA_VERSION = 1
"""JSONL + meta.json schema version.

Additive changes (new optional fields) may keep the same major. Renames
or removals require a major bump — consumers MUST check this field.
"""

# Minimum length for a reference context to be considered a reliable
# substring anchor. Shorter than this, the match is too likely to hit
# boilerplate (headings like "## Summary", shared link text, etc.) and
# inflate source_entry_ids. Callers fall back to normalized match if
# the exact match fails at or above this threshold.
MIN_MATCH_CHARS = 40


def _spotcheck_template(corvia_version: str, corpus_hash: str, jsonl_name: str) -> str:
    header = (
        f"# Spot-check log — corvia {corvia_version} / {corpus_hash}\n"
        "\n"
        f"Sample 10 random queries from the JSONL "
        f"(e.g. `shuf -n 10 {jsonl_name} | jq .query`).\n"
        "Mark each `accept` or `reject` with a one-sentence reason.\n"
        "\n"
        "**Gate:** if more than 2 of 10 are `reject`, do NOT proceed to the retrieval "
        "harness. Regenerate with different `--distribution` or `--generator-model` "
        "until reject rate ≤ 2/10.\n"
        "\n"
    )
    slots = "\n".join(
        f"## {i}. [accept|reject] …\n\nQuery:\nReason:\n" for i in range(1, 11)
    )
    footer = "\n---\n\n**Summary:** __ accept / __ reject. Gate passed? [yes / no]\n"
    return header + slots + footer


def write_testset(
    out_dir: Path,
    meta: dict,
    rows: Iterable[dict],
    force: bool = False,
) -> None:
    """Write JSONL + meta.json + spotcheck.md under `{out_dir}/{corvia_version}/`.

    Raises FileExistsError if the JSONL already exists and `force=False`.
    The spotcheck template is seeded only on first write — it is never
    clobbered on a --force rerun, so an operator's annotations survive.
    """
    version_dir = out_dir / meta["corvia_version"]
    version_dir.mkdir(parents=True, exist_ok=True)

    jsonl_name = f"{meta['corpus_hash']}.jsonl"
    jsonl_path = version_dir / jsonl_name
    meta_path = version_dir / f"{meta['corpus_hash']}.meta.json"
    spotcheck_path = version_dir / "spotcheck.md"

    if jsonl_path.exists() and not force:
        raise FileExistsError(
            f"{jsonl_path} already exists. Use --force to overwrite."
        )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if not spotcheck_path.exists():
        spotcheck_path.write_text(
            _spotcheck_template(
                corvia_version=meta["corvia_version"],
                corpus_hash=meta["corpus_hash"],
                jsonl_name=jsonl_name,
            )
        )


_WS_RE = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """Collapse whitespace + strip for fuzzy matching."""
    return _WS_RE.sub(" ", s).strip()


@dataclass(frozen=True)
class MatchResult:
    """Outcome of `match_source_entries`.

    `method` is one of:
      - "exact"      — verbatim substring hit
      - "normalized" — whitespace-collapsed substring hit (HeadlineSplitter drift)
      - "none"       — nothing matched (rare; post-process logs warning)
      - "too_short"  — reference context below MIN_MATCH_CHARS; no match attempted
    """

    ids: list[str]
    method: str


def match_source_entries(
    reference_context: str,
    entries: list[corpus.Entry],
    *,
    min_chars: int = MIN_MATCH_CHARS,
) -> MatchResult:
    """Identify which original entries a Ragas reference_context came from.

    Ragas does not propagate LangChain Document metadata to the
    SingleTurnSample output. HeadlineSplitter produces verbatim-ish content
    slices, so we substring-match back to the source entry body. To avoid
    boilerplate false positives (headings, link text, shared phrases), we
    require the context to be at least `min_chars` characters long.

    Order of attempts:
      1. Exact substring (after strip).
      2. Whitespace-normalized substring (collapses CR/LF/tabs to single space).
      3. Give up.
    """
    needle = reference_context.strip()
    if not needle:
        return MatchResult(ids=[], method="none")
    if len(needle) < min_chars:
        return MatchResult(ids=[], method="too_short")

    # 1. Exact substring match.
    exact = [e.id for e in entries if needle in e.body]
    if exact:
        return MatchResult(ids=sorted(exact), method="exact")

    # 2. Whitespace-normalized fallback.
    norm_needle = _normalize_for_match(needle)
    if not norm_needle:
        return MatchResult(ids=[], method="none")
    norm_matches = [
        e.id for e in entries if norm_needle in _normalize_for_match(e.body)
    ]
    if norm_matches:
        return MatchResult(ids=sorted(norm_matches), method="normalized")

    return MatchResult(ids=[], method="none")
