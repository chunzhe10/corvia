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
            "query_id": "ragas-000",
            "query": "what is A?",
            "reference_answer": "A is about alpha.",
            "reference_contexts": ["Original content about alpha."],
            "source_entry_ids": ["fix-a"],
            "query_type": "single_hop_specific",
            "ragas_metadata": {},
        },
        {
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
        "corvia_version": "1.0.0",
        "corpus_hash": "sha256:" + "0" * 64,
        "corpus_entry_count": 5,
        "visible_entry_count": 3,
        "superseded_entry_count": 2,
        "generator_model": "gemini:gemini-2.0-flash",
        "embedding_model": "gemini:text-embedding-004",
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


def test_meta_json_has_required_keys(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    meta_path = tmp_path / meta["corvia_version"] / f"{meta['corpus_hash']}.meta.json"
    reloaded = json.loads(meta_path.read_text())
    required = {
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
    }
    assert required.issubset(reloaded.keys())


def test_spotcheck_template_has_ten_slots(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    sc = (tmp_path / meta["corvia_version"] / "spotcheck.md").read_text()
    # Ten numbered headings like "## 1", "## 2", ...
    for i in range(1, 11):
        assert f"## {i}. " in sc


def test_overwrite_refuses_without_force(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    with pytest.raises(FileExistsError):
        writer.write_testset(tmp_path, meta, _fake_rows())


def test_overwrite_honours_force(tmp_path: Path) -> None:
    meta = _fake_meta()
    writer.write_testset(tmp_path, meta, _fake_rows())
    # Write again with different rows under --force
    new_rows = _fake_rows()[:1]
    writer.write_testset(tmp_path, meta, new_rows, force=True)
    jsonl_path = tmp_path / meta["corvia_version"] / f"{meta['corpus_hash']}.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1


# ─── match_source_entries ──────────────────────────────────────────────────────


def test_match_source_entries_exact_substring() -> None:
    entries = [
        corpus.Entry(id="a", created_at="t", kind="x", body="hello alpha world"),
        corpus.Entry(id="b", created_at="t", kind="x", body="beta gamma"),
    ]
    matches = writer.match_source_entries("alpha world", entries)
    assert matches == ["a"]


def test_match_source_entries_empty_needle() -> None:
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="anything")]
    # Empty / whitespace needle → no spurious matches
    assert writer.match_source_entries("", entries) == []
    assert writer.match_source_entries("   ", entries) == []


def test_match_source_entries_across_two_entries() -> None:
    # Same needle present in two entries — both match
    entries = [
        corpus.Entry(id="a", created_at="t", kind="x", body="shared phrase and more"),
        corpus.Entry(id="b", created_at="t", kind="x", body="prefix shared phrase suffix"),
        corpus.Entry(id="c", created_at="t", kind="x", body="unrelated"),
    ]
    matches = writer.match_source_entries("shared phrase", entries)
    assert set(matches) == {"a", "b"}


def test_match_source_entries_strips_whitespace() -> None:
    entries = [corpus.Entry(id="a", created_at="t", kind="x", body="body with padding")]
    assert writer.match_source_entries("  body with padding  ", entries) == ["a"]
