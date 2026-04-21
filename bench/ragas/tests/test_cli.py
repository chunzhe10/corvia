"""CLI-level tests for generate.py (--dry-run only; non-dry-run hits the network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent  # .../bench/ragas
GENERATE = BENCH_DIR / "generate.py"


def _run_dry(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATE), "--dry-run", *args],
        capture_output=True,
        text=True,
        cwd=BENCH_DIR,
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
    result = _run_dry(["--entries", str(tmp_path / "nonexistent"), "--out", str(tmp_path)])
    assert result.returncode == 3


def test_empty_entries_dir_exits_three(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run_dry(["--entries", str(empty), "--out", str(tmp_path)])
    assert result.returncode == 3


def test_output_exists_exits_four_without_force(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    # Pre-create the output file that would be written.
    import corpus
    entries = corpus.load_all(fixtures_dir)
    visible, _ = corpus.partition_visible(entries)
    chash = corpus.corpus_hash(visible)
    version_dir = tmp_path / "1.0.0"
    version_dir.mkdir()
    (version_dir / f"{chash}.jsonl").write_text("[]")

    # Invoke non-dry-run to hit the collision guard. We use --n 0 so we don't
    # trigger LLM init before the collision check, and we DON'T pass --force.
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
    )
    assert result.returncode == 4, f"stdout={result.stdout!r} stderr={result.stderr!r}"
