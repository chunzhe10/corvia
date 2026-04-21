"""Corpus loader for bench/ragas.

Parses `.corvia/entries/*.md` files (TOML frontmatter + markdown body),
partitions visible from superseded entries, and computes a deterministic
corpus hash.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    id: str
    created_at: str
    kind: str
    tags: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    body: str = ""


_DELIM = "+++"


def load_entry(path: Path) -> Entry:
    """Parse a single .md file with TOML frontmatter into an Entry.

    Frontmatter is delimited by a line containing exactly '+++' (opening)
    and another line containing exactly '+++' (closing). Everything after
    the closing delimiter is the body; a single leading newline is stripped.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not lines or lines[0].strip() != _DELIM:
        raise ValueError(f"{path}: missing opening '+++' frontmatter delimiter")

    try:
        close_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == _DELIM)
    except StopIteration as err:
        raise ValueError(f"{path}: missing closing '+++' frontmatter delimiter") from err

    fm_toml = "\n".join(lines[1:close_idx])
    fm = tomllib.loads(fm_toml)
    body = "\n".join(lines[close_idx + 1 :])
    if body.startswith("\n"):
        body = body[1:]

    return Entry(
        id=fm["id"],
        created_at=fm["created_at"],
        kind=fm.get("kind", ""),
        tags=list(fm.get("tags", [])),
        supersedes=list(fm.get("supersedes", [])),
        body=body,
    )


def load_all(entries_dir: Path) -> list[Entry]:
    """Load every .md file in `entries_dir` as an Entry. Order is stable (sorted)."""
    md_files = sorted(entries_dir.glob("*.md"))
    return [load_entry(p) for p in md_files]


def partition_visible(entries: list[Entry]) -> tuple[list[Entry], set[str]]:
    """Return (visible_entries, superseded_ids).

    An entry is superseded iff its id appears in any other entry's `supersedes`
    list. `supersedes` is a forward link: new_entry.supersedes = [old_entry.id].
    """
    superseded_ids: set[str] = set()
    for e in entries:
        superseded_ids.update(e.supersedes)
    visible = [e for e in entries if e.id not in superseded_ids]
    return visible, superseded_ids


def corpus_hash(visible: list[Entry]) -> str:
    """Deterministic sha256 over the visible corpus.

    Hashes each entry's (id, body) pair, sorted — so load order and tag edits
    don't bust the hash, only content changes do.
    """
    lines = sorted(
        f"{e.id}:{hashlib.sha256(e.body.encode('utf-8')).hexdigest()}" for e in visible
    )
    h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return f"sha256:{h}"
