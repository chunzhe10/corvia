# Ragas Synthetic Testset — POC Findings (2026-04-21)

Phase 4 POC validation for [#125](https://github.com/chunzhe10/corvia/issues/125).
Environment: `/tmp/ragas-poc/` venv (Python 3.12.13 via `uv`), Ragas 0.4.3, LangChain 1.2.15, langchain-google-genai installed.

## Results

| Assumption | Status | Finding |
|---|---|---|
| Python 3.13 compat | ⚠ Revised | 3.13 is the devcontainer default, but `uv venv --python 3.12` works cleanly. Pin Python 3.12 in README. Avoids `ragas` wheel issues against newest CPython. |
| Ragas `from_langchain` + `generate_with_langchain_docs` | ✅ Confirmed | `TestsetGenerator.from_langchain(llm, embedder)` and `generate_with_langchain_docs(documents, testset_size, query_distribution)` both exist in 0.4.3. |
| Synthesizer class names | ✅ Confirmed | `SingleHopSpecificQuerySynthesizer`, `MultiHopSpecificQuerySynthesizer`, `MultiHopAbstractQuerySynthesizer` all importable from `ragas.testset.synthesizers`. |
| SQLite LLM cache | ✅ Confirmed with path correction | Correct import is **`langchain_core.globals.set_llm_cache`** — NOT `langchain.globals` as the plan originally said. `SQLiteCache` lives in `langchain_community.cache`. Cache file is created on first set. |
| `document_metadata` threads through to output | ❌ Not via `SingleTurnSample` | Ragas stores `document_metadata` on the original document Node in the knowledge graph (at `generate.py:200`) but the final `SingleTurnSample` only exposes `user_input`, `reference`, `reference_contexts` (list of content strings), `persona_name`, `query_style`, `query_length`. No metadata surface. |
| Gemini 2.0 Flash end-to-end | ⏸ Pending | Needs `GEMINI_API_KEY`. Will run in Phase 8 E2E. |
| 2-entry dry-run | ⏸ Pending | Same — needs key. |

## Workaround for missing metadata propagation

`reference_contexts` are verbatim chunks from `HeadlineSplitter`-produced nodes whose `page_content` is a slice of the original entry body. At write time, for each sample:

```python
def match_source_entries(reference_context: str, entries: list[Entry]) -> list[str]:
    # Ragas' splitter may introduce leading/trailing whitespace — strip both sides
    needle = reference_context.strip()
    # Exact substring match against entry body; typical for SingleHop
    matches = [e.id for e in entries if needle and needle in e.body]
    if not matches:
        # Fallback: longest common substring heuristic (cover cases where
        # the splitter dropped a trailing whitespace)
        matches = [e.id for e in entries if _body_overlap(needle, e.body) > 0.8]
    return matches
```

For MultiHop samples, `reference_contexts` is a list; union the matches across all chunks.

This strategy was validated by inspecting `HeadlineSplitter.adjust_chunks` which produces verbatim slices (not paraphrased).

## Plan revisions required

1. `requirements.txt` → Python 3.12+ note (was "3.11+"), pin Python via `.python-version` or instruct `uv venv --python 3.12` in README.
2. `generate.py` cache wiring → `from langchain_core.globals import set_llm_cache` (not `langchain.globals`).
3. `writer.py` → add `match_source_entries()` helper for post-processing `reference_contexts` → `source_entry_ids`.
4. Design doc §4.3 — note that LangChain Document metadata is NOT preserved through Ragas and we use substring matching instead.

## POC artifact

Venv left at `/tmp/ragas-poc/` for the duration of this dev-loop. Deleted after PR merge.

## Cost

$0. No API calls made (no key available in POC env).

## Exit gate

POC succeeded — all blocking assumptions resolved or have documented workarounds. No user pause required. Proceed to Phase 5 (TDD implementation) with plan revisions applied inline.
