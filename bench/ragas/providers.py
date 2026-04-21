"""LLM + embedder factory for bench/ragas.

Keyed by `--provider`. Returns a `ProviderBundle` that names the concrete
models used so the caller can record them in meta.json without guessing.
SDK imports are lazy so the CLI can start without optional deps.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, NoReturn


_KEY_HELP = (
    "Set {env} to a valid key. Free Gemini keys: https://aistudio.google.com/app/apikey"
)


@dataclass(frozen=True)
class ProviderBundle:
    """LLM + embedder with the concrete model names used.

    The meta.json writer records `generator_model`/`embedding_model`
    verbatim from this bundle so future auditors can reproduce the run.
    """

    llm: Any
    embedder: Any
    generator_model: str  # e.g. "gemini:gemini-2.0-flash"
    embedding_model: str  # e.g. "gemini:models/text-embedding-004"


def _fail(msg: str) -> NoReturn:
    print(f"bench/ragas: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _gemini_api_key() -> str:
    # langchain-google-genai accepts either GEMINI_API_KEY or GOOGLE_API_KEY;
    # we normalize to GEMINI_API_KEY for error messages.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        _fail(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
            + _KEY_HELP.format(env="GEMINI_API_KEY")
        )
    return key


def _make_gemini(generator_model: str | None) -> ProviderBundle:
    from langchain_google_genai import (  # noqa: WPS433 — lazy import
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    key = _gemini_api_key()
    # Defaults validated against v1beta ListModels 2026-04-21: both in free tier.
    # gemini-flash-latest rotates to the current Flash; pin a stamp (e.g.
    # gemini-2.5-flash-lite) via --generator-model for snapshot reproducibility.
    gen_model = generator_model or "gemini-flash-latest"
    emb_model = "models/gemini-embedding-001"
    llm = ChatGoogleGenerativeAI(model=gen_model, google_api_key=key)
    embedder = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=key)
    return ProviderBundle(
        llm=llm,
        embedder=embedder,
        generator_model=f"gemini:{gen_model}",
        embedding_model=f"gemini:{emb_model}",
    )


def _make_openai(generator_model: str | None) -> ProviderBundle:
    if not os.environ.get("OPENAI_API_KEY"):
        _fail("OPENAI_API_KEY is not set.")
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: WPS433
    except ImportError:
        _fail(
            "--provider openai requires `pip install langchain-openai`. "
            "Uncomment the line in requirements.txt."
        )
    gen_model = generator_model or "gpt-4o-mini"
    emb_model = "text-embedding-3-small"
    return ProviderBundle(
        llm=ChatOpenAI(model=gen_model),
        embedder=OpenAIEmbeddings(model=emb_model),
        generator_model=f"openai:{gen_model}",
        embedding_model=f"openai:{emb_model}",
    )


def _make_groq(generator_model: str | None) -> ProviderBundle:
    # Groq's generous free-tier (14 400 RPD on llama-3.1-8b-instant as of 2026-04)
    # makes it the default for users without paid LLM API access. Groq has no
    # first-party embedder, so we run a local sentence-transformers model
    # instead — keeps the pipeline fully offline for embeddings.
    if not os.environ.get("GROQ_API_KEY"):
        _fail(
            "GROQ_API_KEY is not set. Free keys: https://console.groq.com/keys"
        )
    try:
        from langchain_groq import ChatGroq  # noqa: WPS433
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: WPS433
    except ImportError:
        _fail(
            "--provider groq requires `pip install langchain-groq "
            "langchain-huggingface sentence-transformers`. Uncomment the "
            "lines in requirements.txt."
        )
    gen_model = generator_model or "llama-3.1-8b-instant"
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    return ProviderBundle(
        llm=ChatGroq(model=gen_model),
        embedder=HuggingFaceEmbeddings(model_name=emb_model),
        generator_model=f"groq:{gen_model}",
        embedding_model=f"local:{emb_model}",
    )


def _make_anthropic(generator_model: str | None) -> ProviderBundle:
    # --provider anthropic requires BOTH keys because Anthropic ships no
    # first-party embedder. This is documented in README.md.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set.")
    if not os.environ.get("OPENAI_API_KEY"):
        _fail(
            "--provider anthropic still needs OPENAI_API_KEY for embeddings "
            "(Anthropic has no first-party embedding model). See README.md."
        )
    try:
        from langchain_anthropic import ChatAnthropic  # noqa: WPS433
        from langchain_openai import OpenAIEmbeddings  # noqa: WPS433
    except ImportError:
        _fail(
            "--provider anthropic requires `pip install langchain-anthropic langchain-openai`. "
            "Uncomment the lines in requirements.txt."
        )
    # Anthropic published `claude-haiku-4-5` as a stable alias in late 2025;
    # pin to the alias. Users wanting a frozen snapshot should pass
    # --generator-model claude-haiku-4-5-20251001 (or newer stamp).
    gen_model = generator_model or "claude-haiku-4-5"
    emb_model = "text-embedding-3-small"
    return ProviderBundle(
        llm=ChatAnthropic(model=gen_model),
        embedder=OpenAIEmbeddings(model=emb_model),
        generator_model=f"anthropic:{gen_model}",
        embedding_model=f"openai:{emb_model}",
    )


def make_provider(name: str, generator_model: str | None) -> ProviderBundle:
    """Return a ProviderBundle for the named provider.

    Exits with code 2 and a clear message if the required key isn't in env.
    """
    dispatch = {
        "gemini": _make_gemini,
        "groq": _make_groq,
        "openai": _make_openai,
        "anthropic": _make_anthropic,
    }
    if name not in dispatch:
        _fail(
            f"unknown --provider '{name}'. "
            f"Choose one of: {', '.join(sorted(dispatch))}"
        )
    return dispatch[name](generator_model)
