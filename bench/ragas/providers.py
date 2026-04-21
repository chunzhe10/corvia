"""LLM + embedder factory for bench/ragas.

Keyed by `--provider`. Keeps provider SDK imports lazy so the CLI can
start without pulling optional deps.
"""

from __future__ import annotations

import os
import sys
from typing import Any


_KEY_HELP = (
    "Set {env} to a valid key. Free Gemini keys: https://aistudio.google.com/app/apikey"
)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"bench/ragas: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _gemini_api_key() -> str:
    # langchain-google-genai accepts either GEMINI_API_KEY or GOOGLE_API_KEY;
    # we normalize to GEMINI_API_KEY.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        _fail(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
            + _KEY_HELP.format(env="GEMINI_API_KEY")
        )
    return key


def _make_gemini(generator_model: str | None) -> tuple[Any, Any]:
    from langchain_google_genai import (  # noqa: WPS433 — lazy import
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    key = _gemini_api_key()
    model = generator_model or "gemini-2.0-flash"
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=key)
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=key
    )
    return llm, embedder


def _make_openai(generator_model: str | None) -> tuple[Any, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        _fail("OPENAI_API_KEY is not set.")
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: WPS433
    except ImportError:
        _fail(
            "--provider openai requires `pip install langchain-openai`. "
            "Uncomment the line in requirements.txt."
        )
    model = generator_model or "gpt-4o-mini"
    return ChatOpenAI(model=model), OpenAIEmbeddings(model="text-embedding-3-small")


def _make_anthropic(generator_model: str | None) -> tuple[Any, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set.")
    if not os.environ.get("OPENAI_API_KEY"):
        # Anthropic does not ship a first-party embedder. We use OpenAI's.
        _fail(
            "--provider anthropic still needs OPENAI_API_KEY for embeddings "
            "(Anthropic has no first-party embedding model)."
        )
    try:
        from langchain_anthropic import ChatAnthropic  # noqa: WPS433
        from langchain_openai import OpenAIEmbeddings  # noqa: WPS433
    except ImportError:
        _fail(
            "--provider anthropic requires `pip install langchain-anthropic langchain-openai`. "
            "Uncomment the lines in requirements.txt."
        )
    model = generator_model or "claude-haiku-4-5"
    return ChatAnthropic(model=model), OpenAIEmbeddings(model="text-embedding-3-small")


def make_provider(name: str, generator_model: str | None) -> tuple[Any, Any]:
    """Return (llm, embedder) LangChain-compatible pair for the named provider.

    Exits with code 2 and a clear message if the required key isn't in env.
    """
    dispatch = {
        "gemini": _make_gemini,
        "openai": _make_openai,
        "anthropic": _make_anthropic,
    }
    if name not in dispatch:
        _fail(
            f"unknown --provider '{name}'. "
            f"Choose one of: {', '.join(sorted(dispatch))}"
        )
    return dispatch[name](generator_model)
