# Ragas Synthetic Testset Generator

Generates a Ragas-synthesized `(query, reference_answer, reference_contexts, source_entry_ids)` testset from `.corvia/entries/*.md` for use by the retrieval harness (#126) and generation harness (#127).

See `../../docs/rfcs/2026-04-21-ragas-synthetic-testset-design.md` for full rationale.

## Quickstart

```bash
# 1. Create venv with Python 3.12 (NOT 3.13 — Ragas wheels pin <3.13).
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Get a free Gemini API key from https://aistudio.google.com/app/apikey
#    and export it.
export GEMINI_API_KEY=...

# 3. Dry run — confirms corpus loads, hashes, counts visible/superseded entries.
#    No LLM calls, no cost.
python generate.py --dry-run

# 4. Full generation — 50 queries, default distribution, writes JSONL + meta.json.
python generate.py --n 50
```

Output lands under `../eval_sets/{corvia_version}/{corpus_hash}.jsonl` with a sidecar `.meta.json` and a `spotcheck.md` template (fill in by hand after generation).

## Cost

Gemini 2.0 Flash free tier: ~1 500 requests/day, plenty for a 50-query run.
OpenAI `gpt-4o-mini` override: ~$3–5 per run.

## CLI

```
generate.py [--n N] [--out DIR] [--cache PATH] [--entries DIR]
            [--provider gemini|openai|anthropic] [--generator-model NAME]
            [--distribution single=0.5,multi=0.25,abstract=0.25]
            [--dry-run] [--force]
```

| Flag | Default | Purpose |
|---|---|---|
| `--n` | 50 | Testset size |
| `--out` | `../eval_sets` | Where to write `{corvia_version}/{corpus_hash}.jsonl` |
| `--cache` | `.cache/ragas-llm.sqlite` | LangChain SQLite cache path (gitignored) |
| `--entries` | `../../../.corvia/entries` | Source directory for entry `.md` files |
| `--provider` | `gemini` | LLM provider (see "Provider overrides" below) |
| `--generator-model` | provider-specific | Override model name |
| `--distribution` | `single=0.5,multi=0.25,abstract=0.25` | Query-type weights |
| `--dry-run` | off | Print counts and target path without LLM calls |
| `--force` | off | Overwrite an existing testset for this corpus snapshot |

## Output schema

Each JSONL row is a single JSON object. Readers MUST check `schema_version` and refuse unknown majors.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | `1`. Additive changes keep major; renames bump it. |
| `query_id` | str | `ragas-<sha256(query+answer)[:12]>` — deterministic across re-runs. |
| `query` | str | The synthesized query. |
| `reference_answer` | str | Ragas-generated answer grounded in `reference_contexts`. |
| `reference_contexts` | list[str] | Raw chunk(s) Ragas used as grounding. |
| `source_entry_ids` | list[str] | Entries whose bodies contain the context (for recall@k). May be empty if matching failed — consumers should report, not crash. |
| `query_type` | str | One of `single_hop_specific`, `multi_hop_specific`, `multi_hop_abstract`. |
| `ragas_metadata` | object | `persona_name`, `query_style`, `query_length`, and `match_methods` (`exact|normalized|none|too_short` per context). |

The sidecar `*.meta.json` pairs with the JSONL and records `corvia_version`, `corpus_hash`, entry counts, provider + embedding model names, `ragas_version`, generation time, and `match_method_counts` / `rows_with_empty_source_ids` for observability.

**Downstream consumers:** the retrieval harness (#126) keys on `source_entry_ids` and `query` to compute `recall@k`; the generation harness (#127) keys on `reference_answer` and `reference_contexts` for `faithfulness` / `answer_relevancy`.

## Provider overrides

- `--provider gemini` (default): needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- `--provider openai`: needs `OPENAI_API_KEY`. Uncomment `langchain-openai` in `requirements.txt` first.
- `--provider anthropic`: needs **BOTH** `ANTHROPIC_API_KEY` (LLM) **and** `OPENAI_API_KEY` (embeddings — Anthropic ships no first-party embedder). Uncomment `langchain-anthropic` + `langchain-openai`.

## Spot-check protocol

After generation the script emits `../eval_sets/{corvia_version}/spotcheck.md` with 10 numbered slots. Open 10 randomly sampled query lines (use `shuf -n 10` on the JSONL) and mark each `accept` / `reject` with a one-sentence reason.

**Gate:** if more than 2/10 are rejected, regenerate with a different `--distribution` or `--generator-model` before using this set downstream. Don't commit a set that fails the gate.

## Superseded entries

This script **excludes** superseded entries from the corpus it feeds to Ragas — they are invisible to the live `corvia_search` retriever (corvia#132), so queries generated from them would be unanswerable and depress recall@k artificially. When corvia#132 lands an `include_superseded` flag, we may revisit.

Visible count and superseded count are both recorded in the sidecar `.meta.json` for provenance.

## Third-party licenses

See `../../THIRD_PARTY_LICENSES.md`. Ragas is Apache 2.0 (compatible with corvia's AGPL-3.0 one-way).
