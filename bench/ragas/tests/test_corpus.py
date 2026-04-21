"""Tests for bench/ragas/corpus.py."""

from pathlib import Path

import pytest

import corpus


def test_load_entry_parses_frontmatter_and_body(fixtures_dir: Path) -> None:
    entry = corpus.load_entry(fixtures_dir / "entry_a.md")
    assert entry.id == "fix-a"
    assert entry.kind == "learning"
    assert entry.tags == ["alpha"]
    assert entry.supersedes == []
    assert entry.created_at == "2026-01-01T00:00:00Z"
    assert "Entry A body" in entry.body
    assert "Original content about alpha" in entry.body


def test_load_entry_strips_plus_delimiters(fixtures_dir: Path) -> None:
    entry = corpus.load_entry(fixtures_dir / "entry_b.md")
    assert "+++" not in entry.body
    # leading whitespace / the blank line just after the closing +++ is stripped
    assert entry.body.lstrip().startswith("# Entry B body")


def test_load_entry_preserves_multiline_body(fixtures_dir: Path) -> None:
    entry = corpus.load_entry(fixtures_dir / "entry_b.md")
    assert "Second paragraph" in entry.body
    assert "\n" in entry.body


def test_load_all_returns_one_per_md_file(fixtures_dir: Path) -> None:
    entries = corpus.load_all(fixtures_dir)
    ids = {e.id for e in entries}
    assert ids == {"fix-a", "fix-b", "fix-c", "fix-d", "fix-e"}


def test_load_entry_missing_supersedes_is_empty_list(fixtures_dir: Path) -> None:
    entry = corpus.load_entry(fixtures_dir / "entry_d.md")
    assert entry.supersedes == []


def test_load_entry_with_supersedes_returns_list(fixtures_dir: Path) -> None:
    entry = corpus.load_entry(fixtures_dir / "entry_c.md")
    assert entry.supersedes == ["fix-a"]


def test_load_all_rejects_file_without_frontmatter(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("no frontmatter here\n")
    with pytest.raises(ValueError, match="frontmatter"):
        corpus.load_entry(bad)


# ─── partition_visible ─────────────────────────────────────────────────────────


def test_partition_visible_excludes_superseded(fixtures_dir: Path) -> None:
    entries = corpus.load_all(fixtures_dir)
    visible, superseded_ids = corpus.partition_visible(entries)
    ids = {e.id for e in visible}
    # fix-a superseded by fix-c; fix-c superseded by fix-e. Visible: b, d, e.
    assert ids == {"fix-b", "fix-d", "fix-e"}
    assert superseded_ids == {"fix-a", "fix-c"}


def test_partition_visible_transitive_chain() -> None:
    # A → B → C (C.supersedes=[B]; B.supersedes=[A]). Visible: [C].
    a = corpus.Entry(id="a", created_at="t", kind="x", body="ba")
    b = corpus.Entry(id="b", created_at="t", kind="x", supersedes=["a"], body="bb")
    c = corpus.Entry(id="c", created_at="t", kind="x", supersedes=["b"], body="bc")
    visible, superseded = corpus.partition_visible([a, b, c])
    assert [e.id for e in visible] == ["c"]
    assert superseded == {"a", "b"}


def test_partition_visible_empty_corpus() -> None:
    visible, superseded = corpus.partition_visible([])
    assert visible == []
    assert superseded == set()


# ─── corpus_hash ───────────────────────────────────────────────────────────────


def _make(id_: str, body: str, tags: list[str] | None = None) -> corpus.Entry:
    return corpus.Entry(
        id=id_, created_at="t", kind="x", tags=list(tags or []), body=body
    )


def test_corpus_hash_shape() -> None:
    h = corpus.corpus_hash([_make("a", "body-a"), _make("b", "body-b")])
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_corpus_hash_deterministic_across_order() -> None:
    a = _make("a", "body-a")
    b = _make("b", "body-b")
    assert corpus.corpus_hash([a, b]) == corpus.corpus_hash([b, a])


def test_corpus_hash_body_sensitive() -> None:
    h1 = corpus.corpus_hash([_make("a", "body-original")])
    h2 = corpus.corpus_hash([_make("a", "body-modified")])
    assert h1 != h2


def test_corpus_hash_tag_insensitive() -> None:
    h1 = corpus.corpus_hash([_make("a", "same-body", tags=["x"])])
    h2 = corpus.corpus_hash([_make("a", "same-body", tags=["y", "z"])])
    assert h1 == h2


def test_corpus_hash_empty_is_valid_sha256() -> None:
    h = corpus.corpus_hash([])
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
