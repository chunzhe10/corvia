# Ragas Synthetic Testset Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Python script that produces a 50-query Ragas synthetic testset from `.corvia/entries/*.md`, excluding superseded entries, with a deterministic corpus hash and a sidecar manifest.

**Architecture:** Self-contained Python tool at `repos/corvia/bench/ragas/`. Parses entries → filters superseded → feeds LangChain `Document`s to `ragas.testset.TestsetGenerator` → writes JSONL + meta.json + spotcheck template. Default LLM: Gemini 2.0 Flash free tier.

**Tech Stack:** Python 3.11+ (3.13 tentatively), Ragas 0.4.x, langchain-core, langchain-google-genai, Pydantic, hashlib, argparse, pytest.

---

## File Structure

```
repos/corvia/bench/ragas/
├── generate.py           # entry point — orchestration + CLI
├── corpus.py             # entry loader + frontmatter parser + superseded filter + corpus hash
├── writer.py             # JSONL + meta.json + spotcheck.md writers
├── providers.py          # LLM/embedder factory keyed by --provider
├── requirements.txt      # pinned deps
├── README.md             # usage, cost, spot-check protocol, env-var setup
├── .gitignore            # .cache/, .venv/, __pycache__
└── tests/
    ├── __init__.py
    ├── conftest.py       # tmp_path fixtures for fake entries
    ├── test_corpus.py    # frontmatter parser, visibility filter, hash determinism
    ├── test_writer.py    # JSONL shape, meta.json schema, overwrite guard
    └── fixtures/         # handcrafted .md fixtures for tests

repos/corvia/bench/eval_sets/      # (git-kept .gitkeep; generated artifacts land here)
repos/corvia/THIRD_PARTY_LICENSES.md   # new: Ragas + LangChain + Gemini SDK attributions
```

Splitting corpus / writer / providers keeps each file under ~150 LoC and individually unit-testable. `generate.py` is pure glue.

---

## Task 0: POC — validate assumptions (Phase 4, runs BEFORE Phase 5)

**[POC]** Block implementation until these succeed or are revised.

**Files:** temporary scratch only; POC outputs feed back into the plan.

- [ ] **Step 1:** Create a throwaway venv: `python3 -m venv /tmp/ragas-poc && source /tmp/ragas-poc/bin/activate`.
- [ ] **Step 2:** `pip install ragas langchain-google-genai`. Record version installed.
  - If Python 3.13 install fails → try `python3.12` via `uv` or note the need for a pinned interpreter. Document the fix in requirements.md.
- [ ] **Step 3:** Confirm Ragas API on installed version. `python -c "from ragas.testset import TestsetGenerator; help(TestsetGenerator)"`. Record the exact from-langchain / generate-with-langchain-docs method names and kwargs in a `docs/rfcs/2026-04-21-ragas-synthetic-testset-poc-findings.md` file.
- [ ] **Step 4:** Run a 2-entry dry generation using a `GEMINI_API_KEY`. Feed two fake `Document`s. Assert output is non-empty and `reference_contexts` preserves `entry_id` in metadata.
  - If metadata is stripped, record the workaround (substring-match in writer.py).
- [ ] **Step 5:** Confirm `langchain.globals.set_llm_cache(SQLiteCache(...))` produces a cache hit on a second identical run. Record yes/no.
- [ ] **Step 6:** Commit POC findings file. If any assumption is invalidated, pause and report to user per dev-loop pivot protocol.

**Success gate:** all steps produce yes/no answers; invalidated assumptions trigger design revision.

**Cost bound:** $0 (Gemini free tier, 2-entry run).

---

## Task 1: Project scaffolding

**Files:**
- Create: `repos/corvia/bench/ragas/requirements.txt`
- Create: `repos/corvia/bench/ragas/README.md`
- Create: `repos/corvia/bench/ragas/.gitignore`
- Create: `repos/corvia/bench/ragas/tests/__init__.py`
- Create: `repos/corvia/bench/eval_sets/.gitkeep`

- [ ] **Step 1:** Write `requirements.txt` with pinned versions from Task 0:
  ```
  ragas==<poc-version>
  langchain-core==<poc-version>
  langchain-google-genai==<poc-version>
  langchain-community==<poc-version>   # for SQLiteCache
  pytest==8.3.*
  pytest-cov==5.*
  ```
  Optional providers pinned but commented out, gated by extras.
- [ ] **Step 2:** Write `README.md` covering: install (`python -m venv .venv && pip install -r requirements.txt`), env vars (`GEMINI_API_KEY` link to aistudio.google.com/app/apikey), CLI usage, cost, spot-check protocol, regeneration cadence.
- [ ] **Step 3:** Write `.gitignore`:
  ```
  .cache/
  .venv/
  __pycache__/
  *.pyc
  ```
- [ ] **Step 4:** Create empty `tests/__init__.py` and `bench/eval_sets/.gitkeep`.
- [ ] **Step 5:** Commit `chore(bench/ragas): scaffold project (requirements, README, gitignore)`.

---

## Task 2: Corpus loader + frontmatter parser (TDD)

**Files:**
- Create: `repos/corvia/bench/ragas/corpus.py`
- Create: `repos/corvia/bench/ragas/tests/conftest.py`
- Create: `repos/corvia/bench/ragas/tests/test_corpus.py`
- Create: `repos/corvia/bench/ragas/tests/fixtures/` with 5 handcrafted `.md` files (2 supersede older entries; 3 are fresh).

- [ ] **Step 1:** Write fixtures in `tests/fixtures/` — each `.md` has TOML frontmatter delimited by `+++` and a markdown body. Include: one entry with `supersedes = ["id-of-another-fixture"]`, one with multiple tags, one with multi-line body, one with no `supersedes`.
- [ ] **Step 2:** Write `conftest.py` with a `fixtures_dir` session-scoped fixture returning the path to `tests/fixtures/`.
- [ ] **Step 3:** Write failing tests in `test_corpus.py`:
  - `test_load_entry_parses_frontmatter_and_body` — asserts returned Entry has `id`, `kind`, `tags`, `supersedes`, `body` (frontmatter stripped).
  - `test_load_entry_strips_plus_delimiters` — body does not contain `+++`.
  - `test_load_all_returns_one_per_md_file` — fixtures dir contains N files → loader returns N entries.
  - `test_load_entry_handles_missing_supersedes_as_empty` — no `supersedes` key → `entry.supersedes == []`.
- [ ] **Step 4:** Run `pytest -q tests/test_corpus.py -k "parses or strips or one_per or missing"` → expect all fail with ImportError.
- [ ] **Step 5:** Implement `corpus.Entry` (dataclass) and `corpus.load_all(entries_dir: Path) -> list[Entry]`. Use `tomllib` (stdlib). Strip frontmatter between the first two `+++` lines. Minimal code to pass.
- [ ] **Step 6:** Run tests again → expect PASS.
- [ ] **Step 7:** Commit `feat(bench/ragas): corpus.load_all parses entry frontmatter and body`.

---

## Task 3: Superseded filter (TDD)

**Files:**
- Modify: `repos/corvia/bench/ragas/corpus.py`
- Modify: `repos/corvia/bench/ragas/tests/test_corpus.py`

- [ ] **Step 1:** Write failing tests:
  - `test_visible_entries_excludes_superseded` — given entries A, B, C where C.supersedes = [A.id], visible is `[B, C]`.
  - `test_visible_entries_transitive_chain` — A → B → C supersession chain (B.supersedes=[A], C.supersedes=[B]); visible is `[C]` only.
  - `test_visible_entries_reports_counts` — helper returns `(visible, superseded_ids)` tuple.
- [ ] **Step 2:** Run tests → expect FAIL.
- [ ] **Step 3:** Implement `corpus.partition_visible(entries) -> tuple[list[Entry], set[str]]`:
  ```python
  superseded_ids = set()
  for e in entries:
      superseded_ids.update(e.supersedes)
  visible = [e for e in entries if e.id not in superseded_ids]
  return visible, superseded_ids
  ```
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(bench/ragas): partition_visible filters superseded entries`.

---

## Task 4: Corpus hash (TDD)

**Files:**
- Modify: `repos/corvia/bench/ragas/corpus.py`
- Modify: `repos/corvia/bench/ragas/tests/test_corpus.py`

- [ ] **Step 1:** Write failing tests:
  - `test_corpus_hash_deterministic` — same entries in different load order → same hash.
  - `test_corpus_hash_body_sensitive` — modify one body → hash changes.
  - `test_corpus_hash_tag_insensitive` — change a tag (not body) → hash unchanged.
  - `test_corpus_hash_shape` — returns `sha256:<64 hex chars>`.
- [ ] **Step 2:** Run tests → FAIL.
- [ ] **Step 3:** Implement `corpus.corpus_hash(visible: list[Entry]) -> str`:
  ```python
  lines = sorted(f"{e.id}:{hashlib.sha256(e.body.encode()).hexdigest()}" for e in visible)
  h = hashlib.sha256("\n".join(lines).encode()).hexdigest()
  return f"sha256:{h}"
  ```
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(bench/ragas): deterministic corpus_hash over visible entries`.

---

## Task 5: Provider factory (TDD)

**Files:**
- Create: `repos/corvia/bench/ragas/providers.py`
- Create: `repos/corvia/bench/ragas/tests/test_providers.py`

- [ ] **Step 1:** Write failing tests (monkeypatch env vars; no network calls):
  - `test_provider_gemini_requires_key` — no `GEMINI_API_KEY` → `SystemExit(2)` with clear message.
  - `test_provider_openai_requires_key` — `--provider openai` without `OPENAI_API_KEY` → `SystemExit(2)`.
  - `test_provider_gemini_returns_llm_and_embedder` — with mock key, returns tuple of LangChain-compatible objects (type-check via `hasattr('invoke')` and `hasattr('embed_documents')`; no real API calls).
- [ ] **Step 2:** Run tests → FAIL.
- [ ] **Step 3:** Implement `providers.make_provider(name: str, generator_model: str | None) -> tuple[LLM, Embeddings]`.
  - Gemini: `ChatGoogleGenerativeAI(model=generator_model or "gemini-2.0-flash")` + `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`.
  - OpenAI: `ChatOpenAI(model=generator_model or "gpt-4o-mini")` + `OpenAIEmbeddings(model="text-embedding-3-small")`.
  - Anthropic: `ChatAnthropic(model=generator_model or "claude-haiku-4-5")` + OpenAI embedder (hard requirement; error if no OPENAI_API_KEY).
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(bench/ragas): provider factory for gemini/openai/anthropic`.

---

## Task 6: Output writers (TDD)

**Files:**
- Create: `repos/corvia/bench/ragas/writer.py`
- Create: `repos/corvia/bench/ragas/tests/test_writer.py`

- [ ] **Step 1:** Write failing tests:
  - `test_write_jsonl_one_per_line` — given fake Ragas rows, output file has N lines, each a valid JSON object with expected fields (`query_id`, `query`, `reference_answer`, `reference_contexts`, `source_entry_ids`, `query_type`).
  - `test_write_meta_json_schema` — meta.json contains required keys (corvia_version, corpus_hash, corpus_entry_count, visible_entry_count, superseded_entry_count, generator_model, embedding_model, ragas_version, query_distribution, testset_size, generated_at, generated_by, git_sha, drift_policy).
  - `test_write_spotcheck_template` — template file exists and contains 10 numbered slots.
  - `test_overwrite_refuses_without_force` — writing to an existing path raises `FileExistsError`.
  - `test_overwrite_honours_force` — with `force=True`, succeeds.
- [ ] **Step 2:** Run tests → FAIL.
- [ ] **Step 3:** Implement `writer.write_testset(out_dir, corvia_version, corpus_hash, rows, meta, force=False)` that creates `{out_dir}/{corvia_version}/{corpus_hash}.jsonl`, `.meta.json`, and seeds a `spotcheck.md` on first write.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(bench/ragas): JSONL + meta.json + spotcheck template writer`.

---

## Task 7: CLI + --dry-run (TDD)

**Files:**
- Create: `repos/corvia/bench/ragas/generate.py`
- Create: `repos/corvia/bench/ragas/tests/test_cli.py`

- [ ] **Step 1:** Write failing tests via `subprocess.run([sys.executable, generate.py, "--dry-run", "--entries", fixtures_dir, "--out", tmp_out])`:
  - `test_dry_run_exits_zero` — process exits 0.
  - `test_dry_run_prints_counts` — stdout contains `visible_entry_count=`, `superseded_entry_count=`, `corpus_hash=sha256:`.
  - `test_dry_run_does_not_write_output` — `tmp_out` directory has no `.jsonl` after run.
  - `test_missing_entries_dir_exits_three` — nonexistent dir → exit 3.
  - `test_output_exists_exits_four_without_force` — touch an existing output file → exit 4.
- [ ] **Step 2:** Run tests → FAIL.
- [ ] **Step 3:** Implement `generate.py` `main()`:
  - `argparse` for `--n`, `--out`, `--cache`, `--entries`, `--provider`, `--generator-model`, `--distribution`, `--dry-run`, `--force`.
  - On `--dry-run`: load, partition, hash, print counts + target path, exit 0.
  - Error handling matches design §4.12.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(bench/ragas): generate.py CLI with --dry-run`.

---

## Task 8: Ragas glue + end-to-end orchestration (no new tests; uses Task 0 POC as integration check)

**Files:**
- Modify: `repos/corvia/bench/ragas/generate.py`

- [ ] **Step 1:** Add non-dry-run path in `main()`:
  1. Partition corpus → visible + superseded counts.
  2. Build LangChain `Document` objects with `metadata={"entry_id": e.id, "kind": e.kind, "tags": e.tags, "created_at": e.created_at}`.
  3. Wire SQLiteCache via `set_llm_cache(SQLiteCache(database_path=args.cache))`.
  4. `llm, embedder = providers.make_provider(args.provider, args.generator_model)`.
  5. Parse `args.distribution` into Ragas's `query_distribution` mapping (per Task 0 POC findings).
  6. `generator = TestsetGenerator.from_langchain(llm, embedder)`.
  7. `dataset = generator.generate_with_langchain_docs(documents, testset_size=args.n, query_distribution=distribution)`.
  8. Map dataset rows → our JSONL schema, extracting `entry_id`s from each row's reference contexts metadata.
  9. Build meta dict per design §4.10 (git_sha from `git rev-parse HEAD`, corvia_version parsed from Cargo.toml, ragas_version from `ragas.__version__`).
  10. `writer.write_testset(...)`.
  11. Print summary: path, row count, cost=$0.
- [ ] **Step 2:** Run `python generate.py --dry-run --entries tests/fixtures` (smoke test).
- [ ] **Step 3:** Commit `feat(bench/ragas): end-to-end Ragas orchestration`.

---

## Task 9: THIRD_PARTY_LICENSES.md

**Files:**
- Create: `repos/corvia/THIRD_PARTY_LICENSES.md`

- [ ] **Step 1:** Author `THIRD_PARTY_LICENSES.md` listing for each eval-harness dep: name, upstream URL, license SPDX, short quoted NOTICE / copyright line where the license requires one. Include Ragas (Apache 2.0 — requires NOTICE preservation), LangChain family (MIT), Pydantic (MIT), google-generativeai SDK (Apache 2.0), datasets (Apache 2.0), nest-asyncio (BSD-2). Link back to each project's LICENSE file at a pinned commit SHA / version.
- [ ] **Step 2:** Commit `chore(license): add THIRD_PARTY_LICENSES for eval-harness deps`.

---

## Task 10: Real end-to-end generation (Phase 8 — E2E, only once `GEMINI_API_KEY` is set)

**Files:**
- Will create: `bench/eval_sets/1.0.0/sha256:<...>.jsonl`
- Will create: `bench/eval_sets/1.0.0/sha256:<...>.meta.json`
- Will create: `bench/eval_sets/1.0.0/spotcheck.md`

- [ ] **Step 1:** `export GEMINI_API_KEY=<user-provided>`.
- [ ] **Step 2:** `python bench/ragas/generate.py --n 50 --entries ../../.corvia/entries --out ../eval_sets --cache bench/ragas/.cache/ragas-llm.sqlite`.
- [ ] **Step 3:** Verify JSONL has ≥ 50 lines, each valid JSON. Verify meta.json matches schema.
- [ ] **Step 4:** Open `spotcheck.md`, sample 10 queries at random (use `shuf -n 10` on line numbers), judge each as `accept` / `reject` with one-sentence reason. Commit the filled-out template.
- [ ] **Step 5:** If > 2/10 rejected → tune (swap model, tighten distribution, prune junk entries), regenerate, re-spotcheck. Loop until ≤ 2/10 reject rate.
- [ ] **Step 6:** Commit `feat(bench): first Ragas testset for corvia v1.0.0`.

---

## Task 11: Register eval-set artifacts in corvia

- [ ] **Step 1:** After Task 10 success, write a corvia_write entry (kind=`learning`) summarizing: corvia_version, corpus_hash, visible/superseded counts, generator model, spot-check accept rate, file path.

---

## Rollback plan

- The only runtime-impacting change is `THIRD_PARTY_LICENSES.md`. All other files are under `bench/ragas/` and `bench/eval_sets/` (eval-only).
- If the generated testset turns out to be low-quality (> 2/10 rejects persistently), delete the JSONL + spotcheck, retry with a different `--distribution` or `--generator-model`. No code changes needed to revert.

## Non-goals reminders (do NOT do in this PR)

- No `corvia bench` CLI — that's #128.
- No retrieval harness — that's #126.
- No CI integration — that's #129.
- No runtime RAG changes.
- Do not touch `bench/canary/` — it's frozen.
