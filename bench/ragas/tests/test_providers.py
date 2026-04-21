"""Tests for bench/ragas/providers.py — no real network calls."""

from __future__ import annotations

import pytest

import providers


def test_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc:
        providers.make_provider("garbage", None)
    assert exc.value.code == 2


def test_gemini_missing_key_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        providers.make_provider("gemini", None)
    assert exc.value.code == 2


def test_openai_missing_key_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        providers.make_provider("openai", None)
    assert exc.value.code == 2


def test_anthropic_missing_key_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        providers.make_provider("anthropic", None)
    assert exc.value.code == 2


def test_anthropic_needs_openai_key_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # Anthropic has no first-party embedder → must fall through to OpenAI.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        providers.make_provider("anthropic", None)
    assert exc.value.code == 2


def test_gemini_bundle_has_concrete_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key")
    bundle = providers.make_provider("gemini", None)
    assert hasattr(bundle.llm, "invoke")
    assert hasattr(bundle.embedder, "embed_documents")
    assert bundle.generator_model == "gemini:gemini-2.0-flash"
    assert bundle.embedding_model == "gemini:models/text-embedding-004"


def test_gemini_accepts_generator_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key")
    bundle = providers.make_provider("gemini", "gemini-2.0-flash-lite")
    assert bundle.generator_model == "gemini:gemini-2.0-flash-lite"


def test_google_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Users may have GOOGLE_API_KEY set instead of GEMINI_API_KEY.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-fake-key")
    bundle = providers.make_provider("gemini", None)
    assert hasattr(bundle.llm, "invoke")
