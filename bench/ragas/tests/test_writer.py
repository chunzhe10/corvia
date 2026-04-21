"""Tests for bench/ragas/writer.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import corpus
import writer


def _fake_rows() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "query_id": "ragas-000",
            "query": "what is A?",
            "reference_answer": "A is about alpha.",
            "reference_contexts": ["Original content about alpha."],
            "source_entry_ids": ["fix-a"],
            "query_type": "single_hop_specific",
            "ragas_metadata": {},
        },
        {
            "schema_version": 1,
            "query_id": "ragas-001",
            "query": "how do A and B relate?",
            "reference_answer": "...",
            "reference_contexts": ["chunk-1", "chunk-2"],
            "source_entry_ids": ["fix-a", "fix-b"],
            "query_type": "multi_hop_specific",
            "ragas_metadata": {},
        },
    ]


def _fake_meta() -> dict:
    return {
        "schema_version": 1,
        "corvia_version": "1.0.0",
        "corpus_hash": "sha256:" + "0" * 64,
        "corpus_entry_count": 5,
        "visible_entry_count": 3,
        "superseded_entry_count": 2,
        "dangling_superseded_refs": [],
        "generator_model": "gemini:gemini-2.0-flash",
        "embedding_model": "gemini:models/text-embedding-004",
        "ragas_version": "0.4.3",
        "query_distribution": {
            "single_hop_specific": 0.5,
            "multi_hop_specific": 0.25,
            "multi_hop_abstract": 0.25,
        },
        "testset_size": 2,
        "generated_at": "2026-04-21T00:00:00Z",
        "generated_by": "bench/ragas/generate.py v1",
        "git_sha": "abcdef1234",
        "drift_policy": "corpus_hash is audit-only; not validated by downstream eval harness",
        "match_method_counts": {"exact": 2, "normalized": 0, "none": 0, "too_short": 0},
        "rows_with_empty_source_ids": 0,
        "generation_seconds": 12.3,
    }


def test_write_testset_creates_expected_files(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    version_dir = tmp_path / meta["corvia_version"]
    hash_ = meta["corpus_hash"]
    assert (version_dir / f"{hash_}.jsonl").exists()
    assert (version_dir / f"{hash_}.meta.json").exists()
    assert (version_dir / "spotcheck.md").exists()


def test_jsonl_one_row_per_line(tmp_path: Path) -> None:
    meta = _fake_meta()
    rows = _fake_rows()
    writer.write_testset(tmp_path, meta, rows)
    jsonl_path = tmp_path / meta["corvia_version"] / f"{meta['corpus_hash']}.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == len(rows)
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["query_id"] == "ragas-000"
    assert parsed[1]["source_entry_ids"] == ["fix-a", "fix-b"]
    assert all(row["schema_version"] == 1 for row in parsed)


def test_meta_json_has_required_keys(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    meta_path = tmp_path / meta["corvia_version"] / f"{meta['corpus_hash']}.meta.json"
    reloaded = json.loads(meta_path.read_text())
    required = {
        "schema_version",
        "corvia_version",
        "corpus_hash",
        "corpus_entry_count",
        "visible_entry_count",
        "superseded_entry_count",
        "generator_model",
        "embedding_model",
        "ragas_version",
        "query_distribution",
        "testset_size",
        "generated_at",
        "generated_by",
        "git_sha",
        "drift_policy",
        "match_method_counts",
        "rows_with_empty_source_ids",
        "generation_seconds",
        "dangling_superseded_refs",
    }
    assert required.issubset(reloaded.keys())
    assert reloaded["schema_version"] == 1


def test_spotcheck_template_has_ten_slots(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    sc = (tmp_path / meta["corvia_version"] / "spotcheck.md").read_text()
    for i in range(1, 11):
        assert f"## {i}. " in sc


def test_spotcheck_survives_force_overwrite(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    sc_path = tmp_path / meta["corvia_version"] / "spotcheck.md"
    sc_path.write_text("# OPERATOR ANNOTATED\n- hand entry 1\n")
    # Re-run with --force; hand annotations must survive.
    writer.write_testset(tmp_path, meta, _fake_rows(), force=True)
    assert "OPERATOR ANNOTATED" in sc_path.read_text()


def test_overwrite_refuses_without_force(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    with pytest.raises(FileExistsError):
        writer.write_testset(tmp_path, meta, _fake_rows())


def test_overwrite_honours_force(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    new_rows = _fake_rows()[:1]
    writer.write_testset(tmp_path, meta, new_rows, force=True)
    jsonl_path = tmp_path / meta["corvia_version"] / f"{meta['corpus_hash']}.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1


# ─── match_source_entries ──────────────────────────────────────────────────────


def test_match_source_entries_exact_substring() -> None:
    body = "prefix " + "x" * 100 + " alpha marker content trailing"
    entries = [
        corpus.Entry(id="a", created_at="t", kind="x", body=body),
        corpus.Entry(id="b", created_at="t", kind="x", body="unrelated short"),
    ]
    needle = "x" * 100 + " alpha marker content"
    result = writer.match_source_entries(needle, entries)
    assert result.ids == ["a"]
    assert result.method == "exact"


def test_match_source_entries_empty_needle() -> None:
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="anything")]
    assert writer.match_source_entries("", entries).method == "none"
    assert writer.match_source_entries("   ", entries).method == "none"


def test_match_source_entries_too_short_returns_too_short() -> None:
    # Needle below MIN_MATCH_CHARS (40) — we refuse to anchor on boilerplate.
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="## Summary is here")]
    result = writer.match_source_entries("## Summary", entries)
    assert result.method == "too_short"
    assert result.ids == []


def test_match_source_entries_heading_boilerplate_no_false_positives() -> None:
    # Two entries both contain "## Summary" heading. Short needle should refuse
    # to match either, rather than returning both.
    body_a = "## Summary\n\nEntry A describes alpha in detail."
    body_b = "## Summary\n\nEntry B describes beta in detail."
    entries = [
        corpus.Entry(id="a", created_at="t", kind="x", body=body_a),
        corpus.Entry(id="b", created_at="t", kind="x", body=body_b),
    ]
    # A real chunk would be long; the short heading is below the threshold.
    result = writer.match_source_entries("## Summary", entries)
    assert result.ids == []
    assert result.method == "too_short"


def test_match_source_entries_whitespace_normalized_fallback() -> None:
    # Ragas' splitter may collapse whitespace differently than our stored body.
    body = "paragraph one.\n\nparagraph two continues here with many specific words."
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body=body)]
    # Needle with a single space where body has \n\n.
    needle = "paragraph one. paragraph two continues here with many specific words."
    result = writer.match_source_entries(needle, entries)
    assert result.ids == ["a"]
    assert result.method == "normalized"


def test_match_source_entries_needle_longer_than_body_returns_none() -> None:
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="short body content here")]
    # A needle longer than any body content → no substring match possible.
    needle = "unrelated paraphrase that does not appear anywhere in the corpus body x" * 5
    result = writer.match_source_entries(needle, entries)
    assert result.method == "none"
    assert result.ids == []


def test_match_source_entries_across_two_entries() -> None:
    # A long shared passage present in two entries — both legitimately match.
    shared = "shared exact phrase that is unusually long and appears verbatim in both entries"
    entries = [
        corpus.Entry(id="a", created_at="t", kind="x", body="prefix " + shared + " and more"),
        corpus.Entry(id="b", created_at="t", kind="x", body="other prefix " + shared + " elsewhere"),
        corpus.Entry(id="c", created_at="t", kind="x", body="unrelated body text here"),
    ]
    result = writer.match_source_entries(shared, entries)
    assert set(result.ids) == {"a", "b"}
    assert result.method == "exact"


def test_match_source_entries_respects_custom_min_chars() -> None:
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="short snippet body")]
    # Override threshold to accept a shorter needle
    result = writer.match_source_entries("short snippet", entries, min_chars=5)
    assert result.method == "exact"
    assert result.ids == ["a"]
