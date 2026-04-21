# Ragas Synthetic Testset — E2E findings (2026-04-21)

Phase 8 E2E for [#125](https://github.com/chunzhe10/corvia/issues/125).
All 59 unit tests pass. Three code issues surfaced only at real-API time
and were fixed; the fourth blocker is environmental (quota) and is
explicitly recorded as deferred.

## E2E attempts (this session)

| # | Config | Outcome | Action |
|---|---|---|---|
| 1 | 68-entry corpus, `gemini-2.0-flash`, n=5, default transforms | Crashed at 17/47 HeadlinesExtractor calls with **quota `limit: 0`** for 2.0 flash free tier. Never reached query generation. | None to fix: this key tier has zero allowance for 2.0 flash. |
| 2 | 10-entry subset, `gemini-flash-latest`, prechunked transforms (my first fix — see §1) | `ValueError: 'headlines' property not found` from `HeadlineSplitter` in default_transforms. | Switched to `default_transforms_for_prechunked` + treat each entry as `NodeType.CHUNK` (see §1). |
| 3 | 10-entry subset, `gemini-flash-latest`, prechunked | `404 NOT_FOUND: models/text-embedding-004` — Google renamed/retired this embedder. | Default embedder → `models/gemini-embedding-001` (see §2). |
| 4 | 10-entry subset, `gemini-flash-latest` | `gemini-flash-latest` aliases to `gemini-3-flash`, which has **20 RPD** free-tier cap — exhausted. | Try a different stable-tier model. |
| 5 | 10-entry subset, `gemini-2.5-flash-lite` | Also `limit: 20 RPD`, also exhausted (prior attempts consumed). | Quota reset is 24h. |
| 6 | 2-entry tiny subset, `gemini-2.0-flash-lite` | `limit: 0` (same as 2.0 flash). | Stop. This key is on a reduced free-tier profile. |

## §1 Fix: HeadlineSplitter fragility → prechunked transforms

`default_transforms` decides based on corpus token distribution: if ≥25 % of docs are > 500 tokens, it installs `HeadlineSplitter(min_tokens=500)` after `HeadlinesExtractor`. The extractor sometimes fails to populate the `headlines` property (short, header-light docs), and the splitter then raises `ValueError: 'headlines' property not found in this node` with no graceful recovery.

Our entries are atomic memory units, not long articles — the "pre-chunked" path from Ragas is a better semantic fit:

```python
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms_for_prechunked, apply_transforms

nodes = [
    Node(type=NodeType.CHUNK,
         properties={"page_content": d.page_content, "document_metadata": d.metadata})
    for d in documents
]
generator.knowledge_graph = KnowledgeGraph(nodes=nodes)
transforms = default_transforms_for_prechunked(
    llm=generator.llm, embedding_model=generator.embedding_model,
)
apply_transforms(generator.knowledge_graph, transforms)
dataset = generator.generate(testset_size=n, query_distribution=...)
```

This path skips the splitter entirely and runs `summary/theme/ner` extractors on each chunk. Shipped in this PR (`bench/ragas/generate.py`).

## §2 Fix: Gemini embedder rename

Design + initial code used `models/text-embedding-004`; the v1beta ListModels API on this key returns only:

- `models/gemini-embedding-001`
- `models/gemini-embedding-2-preview`

`text-embedding-004` appears retired (or restricted). Default embedder switched to `gemini-embedding-001` in `providers.py`; test assertions updated.

## §3 Fix: `rapidfuzz` undeclared Ragas dep

Ragas 0.4.3 imports `rapidfuzz` inside `OverlapScoreBuilder.__post_init__` but does NOT declare it in `install_requires`. Pinned `rapidfuzz==3.14.5` in `requirements.txt`.

## §4 Deferred: free-tier quota too tight for 50-query generation

Ragas' `default_transforms_for_prechunked` still runs ≈ 4 extractors per doc (summary, theme, NER, summary-embedding, plus node_filter) — ~4× LLM call + 1× embedding per doc. On 50 docs that's ~200 LLM calls. Per-query generation adds ~3-5 calls per sample. A 50-query run on 50 docs = ~350-500 LLM calls total.

Google free-tier on this project tops out at either **0 or 20 RPD** per model (stable models + preview models tested). A 350-call run needs a paid-tier (pay-as-you-go) billing on the project or a key on a higher-quota org.

### Deferred: acceptance criterion "50+ queries generated + persisted"

Two viable paths out of this deferral, neither blocks the code-only merge:

1. **Enable billing on the Google Cloud project** backing the Gemini key. `pay-as-you-go` limits then rise to hundreds of RPS; a full run costs under $1 at `gemini-2.5-flash-lite` pricing.
2. **Use a different provider key** (OpenAI paid, Anthropic paid) by passing `--provider openai` or `--provider anthropic`. The generator supports both; requirements.txt lines are currently commented out and need uncomment.

Re-running the harness on this branch when quota is available:

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
cd repos/corvia/bench/ragas
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python generate.py --n 50   # writes ../eval_sets/1.0.0/<hash>.jsonl
```

Then fill `bench/eval_sets/1.0.0/spotcheck.md` manually, commit, close #125.

## Cache observation

`langchain_core.globals.set_llm_cache(SQLiteCache(...))` was set in every run, but the cache file (`.cache/ragas-llm.sqlite`) stayed at 0 rows across attempts. Ragas uses its own `LangchainLLMWrapper` path; the global LLM cache does not intercept. For subsequent retries this is probably fine (an operator running to completion once is enough), but an operator who wants to re-run cheaply after a partial crash will need to either modify Ragas' LLM wrapper or use Ragas' own caching layer. Non-blocking; documented here for future work.

## §5 Groq + local embedder path added (unblocks follow-up run)

Gemini's restricted free tier on this key made it impractical to run the 50-query generation in this session. Groq's free tier (14 400 RPD on `llama-3.1-8b-instant`) is ~720× more generous and easily covers a full run. Groq has no first-party embedder, so we pair it with a **local** `sentence-transformers/all-MiniLM-L6-v2` embedder (~80 MB model weights, one-time download) to keep the pipeline at zero marginal cost.

This PR ships:

- `providers._make_groq` dispatched by `--provider groq`.
- `langchain-groq`, `langchain-huggingface`, `sentence-transformers` added to `requirements.txt`.
- CLI: `--provider` choices now include `groq`.
- README has a "Provider overrides" section documenting the Groq + local embedder path.
- 3 new provider tests (62 total, all passing).

Follow-up PR to close the 50-query acceptance criterion:

```bash
export GROQ_API_KEY=...   # free at https://console.groq.com/keys
cd repos/corvia/bench/ragas
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt   # downloads all-MiniLM-L6-v2 on first run
python generate.py --n 50 --provider groq
# Fill bench/eval_sets/1.0.0/spotcheck.md manually (10-of-50 accept/reject)
git add bench/eval_sets/ bench/ragas/.cache/ && git commit -m "feat(bench): first Ragas testset"
```

## State of this branch at session end

- Code: complete, 59 tests passing, dry-run works on full 68-entry visible corpus.
- Real generation: blocked by per-model 20 RPD free-tier cap.
- Three environment-level code fixes (§1, §2, §3) are committed to this branch.
- The file `bench/eval_sets/1.0.0/*.jsonl` does NOT exist yet. Creating it is a follow-up PR when quota or paid-tier access is available.
