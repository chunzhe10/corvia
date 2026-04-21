"""Output writers + source-entry matcher for bench/ragas.

Writes `{out_dir}/{corvia_version}/{corpus_hash}.jsonl` (one row per line),
its `.meta.json` sidecar, and on first write a blank `spotcheck.md` template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import corpus


_SPOTCHECK_TEMPLATE = """\
# Spot-check log — corvia {corvia_version} / {corpus_hash}

Sample 10 random queries from the JSONL (e.g. `shuf -n 10 {jsonl_name} | jq .query`).
Mark each `accept` or `reject` with a one-sentence reason.

**Gate:** if more than 2 of 10 are `reject`, do NOT proceed to the retrieval harness.
Regenerate with different `--distribution` or `--generator-model` until reject rate ≤ 2/10.

## 1. [accept|reject] …

Query: _paste query here_
Reason: _one sentence_

## 2. [accept|reject] …

Query:
Reason:

## 3. [accept|reject] …

Query:
Reason:

## 4. [accept|reject] …

Query:
Reason:

## 5. [accept|reject] …

Query:
Reason:

## 6. [accept|reject] …

Query:
Reason:

## 7. [accept|reject] …

Query:
Reason:

## 8. [accept|reject] …

Query:
Reason:

## 9. [accept|reject] …

Query:
Reason:

## 10. [accept|reject] …

Query:
Reason:

---

**Summary:** __ accept / __ reject. Gate passed? [yes / no]
"""


def write_testset(
    out_dir: Path,
    meta: dict,
    rows: Iterable[dict],
    force: bool = False,
) -> None:
    """Write JSONL + meta.json + spotcheck.md under `{out_dir}/{corvia_version}/`.

    Raises FileExistsError if the JSONL already exists and `force=False`.
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

    # Write JSONL (one row per line, no trailing newline on the last row).
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write meta.json (pretty-printed for human inspection).
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    # Seed the spotcheck template only if it doesn't exist. Never clobber
    # an operator-filled checklist on a re-run with --force.
    if not spotcheck_path.exists():
        spotcheck_path.write_text(
            _SPOTCHECK_TEMPLATE.format(
                corvia_version=meta["corvia_version"],
                corpus_hash=meta["corpus_hash"],
                jsonl_name=jsonl_name,
            )
        )


def match_source_entries(
    reference_context: str,
    entries: list[corpus.Entry],
) -> list[str]:
    """Identify which original entries a Ragas reference_context came from.

    Ragas does not propagate LangChain Document metadata to the SingleTurnSample
    output. Its HeadlineSplitter produces verbatim content slices, so we can
    substring-match the chunk back to the source entry body.

    Returns a list of entry IDs (typically 1 for SingleHop, 1–N for MultiHop).
    Empty list if no match (unexpected; log upstream).
    """
    needle = reference_context.strip()
    if not needle:
        return []
    return [e.id for e in entries if needle in e.body]
