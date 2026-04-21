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
    p = corpus.partition_visible(entries)
    ids = {e.id for e in p.visible}
    # fix-a superseded by fix-c; fix-c superseded by fix-e. Visible: b, d, e.
    assert ids == {"fix-b", "fix-d", "fix-e"}
    assert p.superseded_ids == {"fix-a", "fix-c"}
    assert p.dangling_superseded_refs == set()


def test_partition_visible_transitive_chain() -> None:
    # A → B → C (C.supersedes=[B]; B.supersedes=[A]). Visible: [C].
    a = corpus.Entry(id="a", created_at="t", kind="x", body="ba")
    b = corpus.Entry(id="b", created_at="t", kind="x", supersedes=["a"], body="bb")
    c = corpus.Entry(id="c", created_at="t", kind="x", supersedes=["b"], body="bc")
    p = corpus.partition_visible([a, b, c])
    assert [e.id for e in p.visible] == ["c"]
    assert p.superseded_ids == {"a", "b"}


def test_partition_visible_empty_corpus() -> None:
    p = corpus.partition_visible([])
    assert p.visible == []
    assert p.superseded_ids == set()
    assert p.dangling_superseded_refs == set()


def test_partition_visible_detects_dangling_refs() -> None:
    # b.supersedes=["does-not-exist"] should be flagged, not silently counted
    a = corpus.Entry(id="a", created_at="t", kind="x", body="body-a")
    b = corpus.Entry(
        id="b", created_at="t", kind="x", supersedes=["ghost"], body="body-b"
    )
    p = corpus.partition_visible([a, b])
    assert [e.id for e in p.visible] == ["a", "b"]  # neither superseded
    assert p.superseded_ids == set()
    assert p.dangling_superseded_refs == {"ghost"}


def test_load_all_rejects_duplicate_ids(tmp_path: Path) -> None:
    # Two files with the same frontmatter id → ValueError
    for name in ("one.md", "two.md"):
        (tmp_path / name).write_text(
            "+++\nid = \"dup\"\ncreated_at = \"2026-01-01\"\nkind = \"x\"\n+++\n\nbody\n"
        )
    with pytest.raises(ValueError, match="duplicate entry IDs"):
        corpus.load_all(tmp_path)


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


# ─── normalize_body (EOL / trailing-whitespace drift) ──────────────────────────


def test_normalize_body_crlf_to_lf() -> None:
    assert corpus.normalize_body("a\r\nb\r\n") == "a\nb\n"


def test_normalize_body_lone_cr_to_lf() -> None:
    assert corpus.normalize_body("a\rb\r") == "a\nb\n"


def test_normalize_body_strips_trailing_whitespace() -> None:
    assert corpus.normalize_body("body\n\n\n") == "body\n"
    assert corpus.normalize_body("body\n  \t\n") == "body\n"


def test_normalize_body_empty_stays_empty() -> None:
    assert corpus.normalize_body("") == ""
    assert corpus.normalize_body("   \n\n") == ""


def test_corpus_hash_insensitive_to_eol_drift() -> None:
    # Byte-equivalent content saved by LF vs CRLF editors should hash identically.
    lf = corpus.Entry(id="a", created_at="t", kind="x", body=corpus.normalize_body("line1\nline2\n"))
    crlf = corpus.Entry(id="a", created_at="t", kind="x", body=corpus.normalize_body("line1\r\nline2\r\n"))
    assert corpus.corpus_hash([lf]) == corpus.corpus_hash([crlf])


def test_corpus_hash_insensitive_to_trailing_newlines() -> None:
    one = corpus.Entry(id="a", created_at="t", kind="x", body=corpus.normalize_body("body"))
    three = corpus.Entry(id="a", created_at="t", kind="x", body=corpus.normalize_body("body\n\n\n"))
    assert corpus.corpus_hash([one]) == corpus.corpus_hash([three])
