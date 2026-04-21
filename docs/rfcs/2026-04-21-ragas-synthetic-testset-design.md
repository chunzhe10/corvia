# Ragas Synthetic Testset Generator — Design

**Issue:** [#125](https://github.com/chunzhe10/corvia/issues/125) [eval 3/7]
**Parent:** #122 (RAG eval harness umbrella)
**Author:** chunzhe10 (brainstormed with Claude Code agent)
**Status:** Draft
**Date:** 2026-04-21

## 1. Problem

The eval harness (#122) needs a **broad-recall testset** for retrieval and generation metrics (recall@k, MRR, context_precision, faithfulness). The canary set (#124, shipped) is 20 frozen queries — too small for statistical power on aggregate metrics, and its whole point is to stay invariant.

We need a *regenerable, labeled* set of `(query, expected_context)` pairs sized in the tens-to-hundreds, derived from the actual corpus so the expected-context labels are correct by construction. Manually hand-labeling this would not scale.

**Ragas** (explodinggradients/ragas, Apache 2.0) provides a `TestsetGenerator` that LLM-synthesizes queries from a corpus with the chunk each query was derived from, giving us free `expected_context` labels. This is the standard approach in the RAG eval literature.

## 2. Goals

1. Produce `repos/corvia/bench/ragas/generate.py` — a Python script that generates an N-query synthetic testset from `.corvia/entries/*.md`.
2. Configurable query-type distribution (default: simple 50 %, reasoning 25 %, multi-context 25 %).
3. Write output as JSONL to `repos/corvia/bench/eval_sets/{corvia_version}/{corpus_hash}.jsonl` with a sidecar `.meta.json` recording generator model, Ragas version, corpus hash, visible/superseded counts, timestamp, and distribution.
4. Filter out superseded entries so every query is answerable by the live retriever (addresses corvia#132).
5. Deterministic `corpus_hash` for reproducibility.
6. LLM-call caching so reruns are free once the cache warms.
7. Spot-check template at `bench/eval_sets/{corvia_version}/spotcheck.md`.
8. Ship `THIRD_PARTY_LICENSES.md` (new file) with Ragas/LangChain/pydantic/openai attributions.

## 3. Non-goals

- **No runtime integration.** This is an offline Python tool; it does not link into the Rust binary and never runs at request time.
- **No scoring.** This issue generates the testset only. The retrieval harness (#126) and generation harness (#127) consume the output.
- **No CI integration.** That is #129 (eval-7).
- **No chunk-level granularity.** Entries are small (~100–5 000 tokens). Ragas handles its own internal chunking from the LangChain `Document`; we feed one `Document` per entry.
- **No reruns on every change.** Regeneration is intentional (costs money and requires human spot-check). Frequency: per release or when corpus composition shifts materially.
- **No corpus-drift gate.** The `corpus_hash` is audit metadata, not a validation guard. Different hash → different testset file. Old testset files stay on disk.

## 4. Design

### 4.1 File layout

```
repos/corvia/bench/ragas/
├── generate.py            # the entry-point script
├── requirements.txt       # pinned Python deps
├── README.md              # usage, cost, spot-check protocol
└── .gitignore             # ignores .cache/, .venv/

repos/corvia/bench/eval_sets/
└── {corvia_version}/
    ├── {corpus_hash}.jsonl      # one query per line
    ├── {corpus_hash}.meta.json  # generator model, ragas version, timestamps, stratification
    └── spotcheck.md             # hand-filled after generation (10 samples, accept/reject notes)

repos/corvia/THIRD_PARTY_LICENSES.md  # new file, Ragas + deps attribution
```

`bench/ragas/` matches the `bench/canary/` sibling pattern (per user preflight note, overriding the issue's `bench/ragas_eval/` naming).

### 4.2 Corpus loading & superseded filtering

Script walks `.corvia/entries/*.md`, parses TOML frontmatter (delimited by `+++` lines), extracts `id`, `kind`, `tags`, `supersedes`, `created_at`, and the markdown body.

**Visibility rule.** An entry is *invisible* (superseded) if its `id` appears in any other entry's `supersedes = [...]` list. `supersedes` is a forward link (new → old), so the visibility set is computed by a single linear pass:

```python
superseded_ids = set()
for entry in all_entries:
    superseded_ids.update(entry.frontmatter.get("supersedes", []))
visible_entries = [e for e in all_entries if e.id not in superseded_ids]
```

We feed **only visible entries** to Ragas. Rationale (per user preflight + canary authoring-trace, corvia#132):

- Superseded entries live on disk but are invisible to `corvia_search` — queries generated from them would be unanswerable by the live retriever, driving artificial false negatives on recall@k.
- The eval harness measures the *live retriever*, not all-on-disk content.
- When corvia#132 lands (`include_superseded` flag), we can revisit. Until then, excluding is correct.

The sidecar `.meta.json` records both `visible_entry_count` and `superseded_entry_count` for provenance.

### 4.3 LangChain Document construction

For each visible entry:

```python
Document(
    page_content=<markdown body, frontmatter stripped>,
    metadata={
        "entry_id": <id>,
        "kind": <kind>,
        "tags": <tags list>,
        "created_at": <created_at>,
    },
)
```

`entry_id` threads through Ragas' testset output via `reference_contexts` → enabling recall@k measurement in #126.

### 4.4 Ragas `TestsetGenerator` configuration

- API: `ragas.testset.TestsetGenerator.from_langchain(llm, embedding_model)` + `.generate_with_langchain_docs(documents, testset_size, query_distribution)`.
- Default `testset_size`: 50 (issue spec).
- Default `query_distribution` (Ragas ≥ 0.2 terminology):
  - `SingleHopSpecificQuerySynthesizer`: 0.50 (issue's "simple")
  - `MultiHopSpecificQuerySynthesizer`: 0.25 (issue's "multi-context")
  - `MultiHopAbstractQuerySynthesizer`: 0.25 (issue's "reasoning")
- Override via `--distribution simple=0.5,reasoning=0.25,multi=0.25` CLI flag.

Exact Ragas class names vary by version; the POC confirms the 0.4.x-series API.

### 4.5 LLM & embedding choice

**Default: Google Gemini free tier** (`gemini-2.0-flash` generator, `models/text-embedding-004` embedder) via `langchain-google-genai`.

- Free tier at https://aistudio.google.com/app/apikey requires no credit card; quota is ~1 500 req/day / 1 M TPM for 2.0 Flash — comfortably covers a 50-query testset run (a few hundred LLM calls).
- Gemini 2.0 Flash is a strong model for structured synthesis tasks; Ragas supports it via LangChain.
- Chosen because Claude Code Max is a CLI subscription without programmatic API credits, and the eval harness needs a path that works at $0 operator cost.
- Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment; `generate.py` fails fast with a clear message if unset.

**Override:** `--provider openai` or `--provider anthropic` for operators who have those keys. Each provider plugs in via `langchain-openai` / `langchain-anthropic`. Anthropic embedder is not first-party — if `--provider anthropic` is set, the embedder defaults to OpenAI's and errors out if neither `GEMINI_API_KEY` nor `OPENAI_API_KEY` is present.

### 4.6 LLM caching

Wrap the LangChain LLM in `SQLiteCache` at `bench/ragas/.cache/ragas-llm.sqlite` (gitignored). Per LangChain docs:

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".cache/ragas-llm.sqlite"))
```

Rerunning the generator with the same documents and model is then free (hits cache). Changing the corpus or model busts the cache naturally (different prompts → different keys).

### 4.7 Corpus hash

```
hash_input = "\n".join(sorted(f"{entry.id}:{sha256(entry.body)}" for entry in visible_entries))
corpus_hash = "sha256:" + hashlib.sha256(hash_input.encode()).hexdigest()
```

- Sorted so entry-creation order does not affect the hash.
- Hashes body only (not frontmatter) so tag edits do not bust the hash — only content changes do.
- Written into the `.meta.json` and embedded in the output filename.

### 4.8 Corvia version

Read `version` from `repos/corvia/Cargo.toml` `[workspace.package]`. Today: `1.0.0`. Output goes to `bench/eval_sets/1.0.0/{corpus_hash}.jsonl`.

### 4.9 JSONL output schema

One JSON object per line:

```json
{
  "query_id": "ragas-<uuid4>",
  "query": "...",
  "reference_answer": "...",
  "reference_contexts": ["...markdown chunk..."],
  "source_entry_ids": ["019d..."],
  "query_type": "single_hop_specific | multi_hop_specific | multi_hop_abstract",
  "ragas_metadata": { "...raw Ragas fields..." }
}
```

`source_entry_ids` is our extraction from Ragas' context-to-document map, threaded via the `entry_id` metadata set in §4.3. This is the field the retrieval harness (#126) will use for recall@k.

### 4.10 Sidecar `.meta.json` schema

```json
{
  "corvia_version": "1.0.0",
  "corpus_hash": "sha256:...",
  "corpus_entry_count": 79,
  "visible_entry_count": 72,
  "superseded_entry_count": 7,
  "generator_model": "openai:gpt-4o-mini",
  "embedding_model": "openai:text-embedding-3-small",
  "ragas_version": "0.4.3",
  "query_distribution": {"single_hop_specific": 0.5, "multi_hop_specific": 0.25, "multi_hop_abstract": 0.25},
  "testset_size": 50,
  "generated_at": "2026-04-21T...Z",
  "generated_by": "bench/ragas/generate.py v1",
  "git_sha": "...",
  "drift_policy": "corpus_hash is audit-only; not validated by downstream eval harness"
}
```

### 4.11 CLI surface

```
python generate.py \
    [--n 50] \
    [--out repos/corvia/bench/eval_sets] \
    [--cache bench/ragas/.cache/ragas-llm.sqlite] \
    [--entries .corvia/entries] \
    [--provider gemini|openai|anthropic]     # default gemini
    [--generator-model gemini-2.0-flash] \
    [--distribution single=0.5,multi=0.25,abstract=0.25] \
    [--dry-run]
```

`--dry-run` prints corpus-hash + visible/superseded counts + output path without calling the LLM. Useful for CI smoke checks and for deciding whether to regenerate.

### 4.12 Error handling

- Missing provider key (`GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` per `--provider`) → exit 2 with a clear message pointing at the README env-var section.
- Zero visible entries → exit 3 ("no corpus to generate from").
- Output file already exists for this `{corvia_version}/{corpus_hash}` → exit 4 unless `--force` ("testset already generated for this corpus snapshot"). Prevents accidental overwrite after an LLM run.
- Ragas raises → propagate stack trace; do not swallow.

### 4.13 Spot-check protocol

After each generation, the operator hand-reviews 10 random queries:

1. `python generate.py` completes, writes the JSONL + meta.json.
2. Operator opens `bench/eval_sets/{corvia_version}/spotcheck.md` (templated on first generation).
3. For each of 10 shuffled queries: mark `accept` / `reject` with one-line reason.
4. Commit the filled-out `spotcheck.md` in the same PR as the testset.
5. **Gate**: if > 2/10 rejected, tune the generator (e.g., swap model, tweak distribution, prune junk entries) and regenerate before proceeding to #126.

The spotcheck file is itself committed into `bench/eval_sets/{corvia_version}/` — the audit trail lives with the set.

### 4.14 Licensing

- Ragas is Apache 2.0 → compatible with corvia's AGPL-3.0.
- External Python script pulls Ragas at eval time only; no Rust linkage.
- New file `repos/corvia/THIRD_PARTY_LICENSES.md` lists: Ragas (Apache 2.0), LangChain (MIT), langchain-google-genai (MIT), langchain-openai (MIT, optional), langchain-anthropic (MIT, optional), Pydantic (MIT), google-generativeai SDK (Apache 2.0), datasets (Apache 2.0), nest-asyncio (BSD-2).

## 5. Assumptions to validate (POC tasks)

These will be tagged `[POC]` in the implementation plan:

1. **[POC] Ragas 0.4.x installs on Python 3.13.** Devcontainer ships Python 3.13 only. Ragas 0.4.3 was released Jan 2026 and typically supports 3.10–3.12. If 3.13 fails, add a pyenv/venv step to the README (or pin to Python 3.12 via `uv`).
2. **[POC] Ragas 0.4.x `TestsetGenerator.from_langchain` + `generate_with_langchain_docs` API.** Confirm the exact class names / kwargs on the currently published version.
3. **[POC] Ragas threads `entry_id` metadata through the testset output.** We rely on this to derive `source_entry_ids` for recall@k. If Ragas strips metadata, we need a fallback (e.g., match output contexts back to entries by substring).
4. **[POC] LangChain SQLite cache hits on Ragas' internal LLM calls.** Ragas may use its own LLM wrapper that bypasses global caches; if so, wrap the LLM explicitly at construction time.
5. **[POC] Gemini 2.0 Flash + `langchain-google-genai` works end-to-end for Ragas generation** on a 2-entry dry run (no rate-limit error, metadata preserved, output shape matches expectation). Free-tier quota accommodates the run.
6. **[POC] 2-entry dry-run completes in < 60 s and stays under Gemini free-tier quota.** Smoke test before running on the full 72-entry corpus.

POC runs on 2–3 entries and aborts early. Cost: $0 (Gemini free tier).

## 6. Success criteria (matches issue #125 acceptance)

- [ ] `bench/ragas/` with `requirements.txt`, `generate.py`, `README.md`.
- [ ] 50+ queries persisted with `.meta.json` sidecar.
- [ ] `spotcheck.md` committed (≤ 2/10 rejected).
- [ ] Corpus hash + model version in sidecar.
- [ ] `THIRD_PARTY_LICENSES.md` includes Ragas attribution.

## 7. Approaches considered

**A. Ragas Apache-2.0 synthetic generation (SELECTED).** Standard, battle-tested, mentioned by name in the issue and parent RFC.

**B. Hand-author everything like the canary set.** Rejected: can't scale to 50+ queries, and the #122 plan explicitly separates canary (hand) from broad-recall (synthetic).

**C. Roll our own generator on top of LangChain prompts.** Rejected: reinvents Ragas with no upside and bifurcates the eval toolchain ecosystem we want downstream metrics from.

## 8. Decisions recorded inline (flagged for user review)

| Decision | Choice | Rationale |
|---|---|---|
| Output dir | `bench/ragas/` + `bench/eval_sets/` | User preflight note overrides issue's `bench/ragas_eval/` path; matches `bench/canary/` sibling pattern. |
| Superseded entries | **Excluded** from corpus | Addresses corvia#132 (superseded invisible to retriever) → unanswerable queries. |
| Default LLM | Google Gemini 2.0 Flash (free tier) | Claude Code Max is a CLI sub without programmatic API credits; Gemini free tier is the only no-cost path. OpenAI + Anthropic supported as overrides. |
| Granularity | Entry-level (one `Document` per `.md`) | Chunking is internal to Ragas; chunk-level would require #123 telemetry trade-offs not yet worth the complexity. |
| Corpus hash | `sha256` of sorted `entry_id:sha256(body)` for visible entries | Bodies only — tag edits should not bust hash; sorted for determinism. |
| Regeneration | Manual, per release or major corpus shift | Costs money; spot-check is the gate, not a hash validator. |

## 9. Decisions locked in during brainstorming (user-approved, 2026-04-21)

| Q | Decision |
|---|---|
| Output path | `bench/ragas/` (matches `bench/canary/` sibling pattern). |
| Superseded-entry policy | **Exclude** (queries must be answerable by live retriever; revisit when corvia#132 lands). |
| LLM provider | **Gemini 2.0 Flash (free tier)** — only $0-cost path given Claude Code Max is not a programmatic API. User must create a free key at aistudio.google.com/app/apikey and export `GEMINI_API_KEY` before running. |
