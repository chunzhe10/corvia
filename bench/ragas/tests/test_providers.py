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


def test_gemini_builds_with_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key")
    llm, embedder = providers.make_provider("gemini", None)
    # langchain-compat surface: LLMs expose `invoke` / embedders expose `embed_documents`.
    assert hasattr(llm, "invoke")
    assert hasattr(embedder, "embed_documents")


def test_gemini_accepts_generator_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key")
    llm, _ = providers.make_provider("gemini", "gemini-2.0-flash-lite")
    # Model name is stashed on the LangChain LLM wrapper under `.model` or `.model_name`.
    model_attr = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    assert model_attr is not None
    assert "gemini-2.0-flash-lite" in str(model_attr)


def test_google_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Users may have GOOGLE_API_KEY set instead of GEMINI_API_KEY.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-fake-key")
    llm, _ = providers.make_provider("gemini", None)
    assert hasattr(llm, "invoke")
