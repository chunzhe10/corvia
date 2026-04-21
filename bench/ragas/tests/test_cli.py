"""CLI-level tests for generate.py (--dry-run only; non-dry-run hits the network)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent  # .../bench/ragas
GENERATE = BENCH_DIR / "generate.py"


def _scrub_env() -> dict[str, str]:
    """Subprocess env without any provider keys — so collision guard order is pinned
    independently of whether the dev machine has GEMINI_API_KEY set."""
    env = os.environ.copy()
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    return env


def _run_dry(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATE), "--dry-run", *args],
        capture_output=True,
        text=True,
        cwd=BENCH_DIR,
        env=_scrub_env(),
    )


def test_dry_run_exits_zero(fixtures_dir: Path, tmp_path: Path) -> None:
    result = _run_dry(["--entries", str(fixtures_dir), "--out", str(tmp_path)])
    assert result.returncode == 0, result.stderr


def test_dry_run_prints_counts(fixtures_dir: Path, tmp_path: Path) -> None:
    result = _run_dry(["--entries", str(fixtures_dir), "--out", str(tmp_path)])
    assert "visible_entry_count=3" in result.stdout  # b, d, e (after a, c superseded)
    assert "superseded_entry_count=2" in result.stdout
    assert "corpus_hash=sha256:" in result.stdout


def test_dry_run_does_not_write_output(fixtures_dir: Path, tmp_path: Path) -> None:
    _run_dry(["--entries", str(fixtures_dir), "--out", str(tmp_path)])
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_missing_entries_dir_exits_three(tmp_path: Path) -> None:
    result = _run_dry(
        ["--entries", str(tmp_path / "nonexistent"), "--out", str(tmp_path)]
    )
    assert result.returncode == 3


def test_empty_entries_dir_exits_three(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run_dry(["--entries", str(empty), "--out", str(tmp_path)])
    assert result.returncode == 3


def test_output_exists_exits_four_without_force(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    # Pre-create the JSONL that would be written.
    import corpus
    entries = corpus.load_all(fixtures_dir)
    chash = corpus.corpus_hash(corpus.partition_visible(entries).visible)
    version_dir = tmp_path / "1.0.0"
    version_dir.mkdir()
    (version_dir / f"{chash}.jsonl").write_text("[]")

    # Non-dry-run, env scrubbed of provider keys. If the collision guard is
    # correctly ordered BEFORE provider init, exit should be 4 (not 2 for
    # missing key). This locks the ordering invariant in generate.py:main().
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATE),
            "--entries",
            str(fixtures_dir),
            "--out",
            str(tmp_path),
            "--n",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=BENCH_DIR,
        env=_scrub_env(),
    )
    assert result.returncode == 4, f"stdout={result.stdout!r} stderr={result.stderr!r}"


# ─── _parse_distribution unit tests (module import path) ───────────────────────


def test_parse_distribution_happy_path() -> None:
    import generate  # noqa: WPS433 — imported under test
    out = generate._parse_distribution("single=0.5,multi=0.25,abstract=0.25")
    assert out == {
        "single_hop_specific": 0.5,
        "multi_hop_specific": 0.25,
        "multi_hop_abstract": 0.25,
    }


def test_parse_distribution_bad_token() -> None:
    import generate  # noqa: WPS433
    with pytest.raises(SystemExit):
        generate._parse_distribution("this_has_no_equals,single=0.5,abstract=0.5")


def test_parse_distribution_bad_sum() -> None:
    import generate  # noqa: WPS433
    with pytest.raises(SystemExit):
        generate._parse_distribution("single=0.5,multi=0.5,abstract=0.5")  # sums to 1.5


def test_parse_distribution_duplicate_key_rejected() -> None:
    import generate  # noqa: WPS433
    with pytest.raises(SystemExit):
        generate._parse_distribution("single=0.5,single=0.5")


def test_parse_distribution_unknown_key_rejected() -> None:
    import generate  # noqa: WPS433
    with pytest.raises(SystemExit):
        generate._parse_distribution("single=0.5,garbage=0.5")


# ─── _content_query_id determinism ─────────────────────────────────────────────


def test_content_query_id_deterministic() -> None:
    import generate  # noqa: WPS433
    a = generate._content_query_id("what is X?", "X is foo.")
    b = generate._content_query_id("what is X?", "X is foo.")
    c = generate._content_query_id("what is Y?", "X is foo.")
    assert a == b
    assert a != c
    assert a.startswith("ragas-")
    assert len(a) == len("ragas-") + 12


# ─── _build_rows unit test with fake Ragas samples ─────────────────────────────


class _FakeEvalSample:
    def __init__(self, q: str, a: str, ctxs: list[str]) -> None:
        self.user_input = q
        self.reference = a
        self.reference_contexts = ctxs
        self.persona_name = "tester"
        self.query_style = "formal"
        self.query_length = "medium"


class _FakeSample:
    def __init__(self, q: str, a: str, ctxs: list[str], synth: str) -> None:
        self.eval_sample = _FakeEvalSample(q, a, ctxs)
        self.synthesizer_name = synth


def test_build_rows_assembles_schema(fixtures_dir: Path) -> None:
    import corpus
    import generate  # noqa: WPS433

    entries = corpus.load_all(fixtures_dir)
    visible = corpus.partition_visible(entries).visible
    entry_b = next(e for e in visible if e.id == "fix-b")
    # Pull a long enough substring from entry_b.body so match_source_entries
    # clears MIN_MATCH_CHARS. Take the entire body — guaranteed to be ≥40 chars.
    ctx = entry_b.body
    assert len(ctx) >= 40

    samples = [_FakeSample("q1?", "a1", [ctx], "single_hop_specific")]
    rows, counts = generate._build_rows(samples, visible)

    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == 1
    assert row["query"] == "q1?"
    assert row["reference_answer"] == "a1"
    assert row["query_type"] == "single_hop_specific"
    assert row["source_entry_ids"] == ["fix-b"]
    assert counts["exact"] == 1
