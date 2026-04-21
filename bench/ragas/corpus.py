"""Corpus loader for bench/ragas.

Parses `.corvia/entries/*.md` files (TOML frontmatter + markdown body),
partitions visible from superseded entries, and computes a deterministic
corpus hash.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections import Counter
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


def normalize_body(text: str) -> str:
    """Canonicalize a body string before hashing/matching.

    - CRLF/CR → LF (editor drift)
    - strip trailing whitespace + ensure a single terminal newline

    The same normalization is applied everywhere the body is consumed
    (hash, Document.page_content, match_source_entries) so the hash and
    the matcher see a consistent view regardless of editor.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    return cleaned + "\n" if cleaned else ""


def load_entry(path: Path) -> Entry:
    """Parse a single .md file with TOML frontmatter into an Entry.

    Frontmatter is delimited by a line containing exactly '+++' (opening)
    and another line containing exactly '+++' (closing). Body is
    normalized via `normalize_body` so hashes survive editor drift.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ValueError(f"{path}: cannot read: {err}") from err
    lines = text.split("\n")
    if not lines or lines[0].strip() != _DELIM:
        raise ValueError(f"{path}: missing opening '+++' frontmatter delimiter")

    try:
        close_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == _DELIM)
    except StopIteration as err:
        raise ValueError(
            f"{path}: missing closing '+++' frontmatter delimiter"
        ) from err

    fm_toml = "\n".join(lines[1:close_idx])
    try:
        fm = tomllib.loads(fm_toml)
    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"{path}: malformed TOML frontmatter: {err}") from err

    missing = [k for k in ("id", "created_at") if k not in fm]
    if missing:
        raise ValueError(
            f"{path}: frontmatter missing required keys: {sorted(missing)}"
        )

    body_raw = "\n".join(lines[close_idx + 1 :])
    if body_raw.startswith("\n"):
        body_raw = body_raw[1:]

    return Entry(
        id=fm["id"],
        created_at=fm["created_at"],
        kind=fm.get("kind", ""),
        tags=list(fm.get("tags", [])),
        supersedes=list(fm.get("supersedes", [])),
        body=normalize_body(body_raw),
    )


def load_all(entries_dir: Path) -> list[Entry]:
    """Load every .md file in `entries_dir` as an Entry. Order is stable (sorted).

    Raises ValueError if any two files share the same `id`.
    """
    md_files = sorted(entries_dir.glob("*.md"))
    entries = [load_entry(p) for p in md_files]
    dupes = [e_id for e_id, count in Counter(e.id for e in entries).items() if count > 1]
    if dupes:
        raise ValueError(
            f"{entries_dir}: duplicate entry IDs on disk: {sorted(dupes)}"
        )
    return entries


@dataclass(frozen=True)
class Partition:
    """Result of partition_visible: visible + superseded IDs + dangling refs."""

    visible: list[Entry]
    superseded_ids: set[str]
    dangling_superseded_refs: set[str]  # ids referenced by `supersedes=[...]` that don't exist


def partition_visible(entries: list[Entry]) -> Partition:
    """Split entries into visible + superseded; report dangling supersede refs.

    `supersedes` is a forward link: new_entry.supersedes = [old_entry.id].
    An entry is invisible iff its id appears in any other entry's list.
    Dangling refs (supersedes pointing to a nonexistent id) are recorded
    but do NOT count as superseded — they are diagnostic metadata for
    the operator.
    """
    all_ids: set[str] = {e.id for e in entries}
    referenced: set[str] = set()
    for e in entries:
        referenced.update(e.supersedes)
    dangling = referenced - all_ids
    superseded_ids = referenced & all_ids
    visible = [e for e in entries if e.id not in superseded_ids]
    return Partition(
        visible=visible,
        superseded_ids=superseded_ids,
        dangling_superseded_refs=dangling,
    )


def corpus_hash(visible: list[Entry]) -> str:
    """Deterministic sha256 over the visible corpus.

    Hashes each entry's (id, normalized_body) pair, sorted — so load order
    and tag edits don't bust the hash, only content changes do. Bodies
    are already normalized by `load_entry`, so CRLF/trailing-whitespace
    drift between editors does not change the hash either.
    """
    lines = sorted(
        f"{e.id}:{hashlib.sha256(e.body.encode('utf-8')).hexdigest()}" for e in visible
    )
    h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return f"sha256:{h}"
