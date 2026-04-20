# Canary Query Set — Design

**Issue:** [#124](https://github.com/chunzhe10/corvia/issues/124) [eval 2/7]
**Parent:** #122 (RAG eval harness umbrella)
**Author:** chunzhe10 (brainstormed with Claude Code agent)
**Status:** Draft
**Date:** 2026-04-20

## 1. Problem

The eval harness (#122) needs a **calibration anchor**: a tiny set of queries whose expected results never change across corpus growth, synthetic-set regeneration (Ragas, #125), or model upgrades. Without this anchor, every retrieval regression is ambiguous — did the score move because the system got worse, or because the testset regenerated differently?

Industry term: **canary queries**. Every mature RAG team maintains one. They are:
- Tiny (≈20 queries, not thousands).
- Hand-curated (deliberate authorship anchored to specific entry IDs), not LLM-synthesized.
- Frozen — same queries + expected IDs forever, even as the corpus and models evolve.

This issue delivers that set.

## 2. Goals

1. Produce `repos/corvia/bench/canary/queries.toml` containing 20 queries with pinned `expected_entry_ids`.
2. Stratify by `query_type` (6 lookup / 6 multi-hop / 4 reasoning / 4 cross-kind) — the eval harness joins on this field for per-type metrics.
3. Stratify by `target_kind` proportionally to the actual corpus composition (approach A, below) so every stratum has ≥2 queries.
4. Record a deterministic `corpus_snapshot_hash` alongside the queries for provenance.
5. Write `bench/canary/README.md` codifying the "frozen forever" rule.

## 3. Non-goals

- **Not synthetic.** Queries are authored by deliberately reading entries and crafting anchored queries; they are **not** produced by an LLM pipeline that generates `(query, answer)` pairs from chunks. That pipeline is issue #125 (Ragas synthetic testset), and its whole purpose depends on a canary set existing separately.
- **No human review gate.** The agent performs the curation end-to-end (read entry → author query → verify via live `corvia_search` → commit). This is an intentional departure from the colloquial meaning of "hand-labeled." The issue text uses "hand-authored" to contrast with synthetic; in an agentic workflow, the agent is the hand.
- **No ongoing mutation.** Once merged, the file is read-only forever. A corpus entry being superseded does **not** trigger a canary update — stale expected_entry_ids are a feature, not a bug (they measure the retriever's ability to surface stable IDs).
- **No chunk-level IDs.** Issue #123 (search telemetry) adds chunk IDs to `SearchResult`, but this canary set uses **entry IDs** only. Rationale in §4.2.
- **Not used for training.** Eval-only.

## 4. Design

### 4.1 File locations

```
repos/corvia/bench/canary/
├── queries.toml     # the 20 queries + corpus snapshot hash
└── README.md        # frozen-forever rule + regeneration prohibition
```

The `bench/` directory is new. It will host the `corvia bench` CLI (issue #128) and additional fixtures. Placed inside the corvia repo (not the workspace) because it ships alongside the binary.

### 4.2 TOML schema

```toml
# Header — provenance
schema_version = 1
created_at = "2026-04-20"
corpus_snapshot_hash = "sha256:<64-hex>"
corpus_entry_count = 71
notes = "Frozen forever. See README.md before editing — do not mutate expected IDs."

[[query]]
id = "canary-001"
query = "how does corvia handle write deduplication"
expected_entry_ids = ["019d...a", "019d...b"]   # ranked
query_type = "lookup"           # lookup | multi-hop | reasoning | cross-kind
target_kind = "reference"       # decision | learning | instruction | reference | mixed
notes = "tests supersession retrieval path"
```

- `expected_entry_ids` is **ranked** (position 0 = ideal top result), so the eval harness can compute MRR, not just recall.
- `target_kind` uses a 5th value `mixed` for `query_type = "cross-kind"` queries whose expected entries span multiple kinds.
- `notes` is free-form, intended to document **what retrieval capability the query tests** (e.g., "synonym resolution", "temporal decision recall"). Required for every query — agents must articulate why each query exists, since there is no human in the loop to remember.

### 4.3 Stratification plan (Approach A: proportional to corpus)

Corpus composition as of 2026-04-20: **43 learning / 15 reference / 11 decision / 2 instruction** (71 total).

**Constraints:**
1. **Hard:** `query_type` distribution = 6 lookup / 6 multi-hop / 4 reasoning / 4 cross-kind (per issue spec).
2. **Hard:** every `target_kind` stratum has **n ≥ 2** queries — below this, a stratum has zero variance and per-kind metrics are uninformative. For instruction (corpus n=2), one query per entry.
3. **Soft:** remaining allocation roughly proportional to corpus mix; learning dominates.
4. **Special value:** `target_kind = "mixed"` is used for all 4 `cross-kind` queries, since cross-kind means the expected entries span multiple kinds by definition.

**Target allocation for 20 queries:**

| Kind        | Corpus count | Canary target | Rationale |
|-------------|-------------:|--------------:|-----------|
| learning    | 43           | 7             | dominant kind, ~35% |
| reference   | 15           | 3             | ~15% |
| decision    | 11           | 4             | ~20%, over corpus share because decisions anchor good reasoning queries |
| instruction | 2            | 2             | floor; one canary per available entry |
| mixed       | —            | 4             | all cross-kind queries |
| **Total**   | 71           | **20**        | |

The two stratifications (by `query_type` and by `target_kind`) are orthogonal. One feasible fill:

| query_type ↓ / target_kind → | learning | reference | decision | instruction | mixed | row total |
|---|---:|---:|---:|---:|---:|---:|
| lookup (6)    | 2 | 1 | 1 | 2 | 0 | 6 |
| multi-hop (6) | 3 | 2 | 1 | 0 | 0 | 6 |
| reasoning (4) | 2 | 0 | 2 | 0 | 0 | 4 |
| cross-kind (4) | 0 | 0 | 0 | 0 | 4 | 4 |
| **col total** | 7 | 3 | 4 | 2 | 4 | **20** |

The authoring agent may shift ±1 within a row/column if a better anchor emerges, **provided the hard constraints above are preserved** (row totals exact, every col ≥2). Final fill is whatever ends up in the committed `queries.toml`; this matrix is a starting plan, not a gate.

### 4.4 Authoring workflow (agent-curated)

Step-by-step per query (×20):

1. Pick a cell from the stratification matrix not yet filled.
2. Enumerate candidate entries of the required `target_kind` via `ls .corvia/entries/` + `grep "^kind = \"<kind>\""`.
3. Read 3–5 candidates, pick one (or two for multi-hop / cross-kind) that genuinely anchors a natural user question.
4. Author a query phrased as an agent or dev would actually ask it — not a regurgitation of the entry's title. Examples:
   - Bad: `"devcontainer taskfile workspace_root"` (keyword soup; too close to the entry's own words)
   - Good: `"why did my devcontainer post-start fail after copying .devcontainer to a new repo?"` (symptom-first phrasing the learning entry addresses)
5. **Verify via live corvia_search** with `limit=10`. The expected entry ID must appear in the top 10. If it doesn't, either:
   - The query is bad — rephrase and re-verify.
   - The retriever is bad — but that's not this issue's scope. Pick a different entry that *does* surface, and note the poor-retrieval entry separately for later investigation (do NOT paper over it by stretching expectations).
6. For multi-hop (2+ expected entries): all expected IDs must appear in top-10; their ranks become the `expected_entry_ids` order.
7. Write the `notes` field explaining what retrieval capability the query probes.

**Freshness caveat.** The verification step uses **current** corvia retrieval as an adequacy check, not as a source of truth. The point of a canary is that expected IDs stay pinned even if future retrievers rank them differently. Verification here ensures the authored queries are *reasonable anchors today*; any future ranking changes are measured against these same pins.

### 4.5 Corpus snapshot hash

Deterministic hash over sorted `{entry_id}\t{sha256_of_file_bytes}` pairs:

```python
# Pseudocode — actual impl likely a small bash/awk or Rust one-shot
entries = sorted(glob(".corvia/entries/*.md"))
pairs = [(basename(f).removesuffix(".md"), sha256(read_bytes(f))) for f in entries]
snapshot = sha256("\n".join(f"{eid}\t{h.hexdigest()}" for eid, h in pairs))
print(f"sha256:{snapshot.hexdigest()}")
```

Properties:
- Filesystem-order-independent (sorted input).
- Ignores mtime, inode, hidden files.
- Changes if any entry's content changes or if entries are added/removed.

The snapshot is recorded at **authoring time** for provenance — it lets future readers of `queries.toml` answer "what was the corpus when these queries were calibrated?" The eval harness does **not** validate the hash at run time (that would defeat frozen-forever: the whole point is that the queries remain valid even when the corpus drifts). The hash is an audit trail, not a gate.

### 4.6 README contents

`bench/canary/README.md` explains, in order:

1. **What this is.** A frozen calibration anchor for the eval harness.
2. **Frozen-forever rule.** Never mutate `expected_entry_ids`. Never add queries. Never remove queries. If the canary feels insufficient, add a **second** canary file (e.g., `canary-v2.toml`) — don't edit this one.
3. **Why frozen.** Mutation defeats the purpose: you can't measure regression against a moving target.
4. **What to do if an expected ID no longer exists.** Nothing. A superseded entry is still a valid ID; corvia's storage is append-only-ish (supersession, not deletion). If the file is truly gone, that's a corpus integrity bug — file it separately, don't edit the canary.
5. **Relationship to #125 (Ragas).** Ragas generates a large synthetic testset that regenerates every run. The canary is immune to that churn. They are complementary: Ragas measures broad recall; canary measures anchored stability.

### 4.7 Implementation vehicle

The authoring itself is a **one-shot agent task**, not a long-lived Rust feature. No crate code changes. Deliverables are three text files committed to the corvia repo:

- `bench/canary/queries.toml`
- `bench/canary/README.md`
- (Optional) `bench/canary/snapshot.sh` — reproducible snapshot-hash computation script for future audits

This keeps the issue scoped to **data authoring**, not infrastructure. The infrastructure that *consumes* this file lives in #126 (retrieval harness) and #128 (`corvia bench` CLI).

## 5. Acceptance criteria mapping

| Original AC | Coverage |
|---|---|
| 20 queries in `bench/canary/queries.toml` | §4.2, §4.3, §4.4 |
| Each has expected IDs verified against current `.corvia/entries/` | §4.4 step 5 |
| Stratified per distribution (6/6/4/4 by query_type) | §4.3 matrix |
| README explains the "frozen forever" rule | §4.6 |
| Corpus snapshot hash recorded alongside | §4.5 |

## 6. Risks

- **Retrieval quality ceiling.** If the current retriever cannot surface a natural anchor entry for some `(query_type, target_kind)` cell, the matrix fill will force either (a) a contrived query that the retriever happens to handle well (biasing the canary toward easy queries) or (b) a genuinely hard query whose top-10 doesn't include the target (violating verification). Mitigation: if forced into this corner, prefer (b) — record the hard query, mark it with `notes = "known-hard — see issue #N"`, and file the retrieval gap as a separate issue rather than papering over.
- **Cross-kind query scarcity.** The corpus has limited decision↔reference linkage; cross-kind queries may feel manufactured. If after 30 min of exploration no natural cross-kind query surfaces for a cell, reduce cross-kind count to 3 and bump reasoning to 5 — flag as a design-time deviation in the PR.
- **Entry-ID stability across supersessions.** UUIDv7 IDs are stable — supersession creates a new entry with `supersedes = [<old_id>]` and the old entry keeps its ID. So `expected_entry_ids` remain valid even after content is superseded. Confirmed by code inspection of `corvia_write` semantics.
- **Agent authoring bias.** The agent curating the canary may inadvertently favor queries the *authoring agent* would ask, not the ones *other agents* will ask. Partial mitigation: rotate the agent persona during authoring (learning-focused vs. decision-focused), and explicitly include queries that test bad phrasing, typos, and symptom-first framing.

## 7. Open items deferred to plan

- Decide exact snapshot-hash tooling: inline one-liner shell, committed `snapshot.sh`, or a tiny `corvia` subcommand. Prefer committed `snapshot.sh` for reproducibility.
- Confirm whether `bench/` should also contain a top-level `bench/README.md` explaining the eval harness (or defer that to #128 when the CLI lands). Defer.
- Decide whether the README should cross-reference `expected_entry_ids` to the entries' `corvia_search` scores at authoring time (helpful for debugging later regressions, adds maintenance burden). **Decision: yes, record as a separate `bench/canary/authoring-trace.md` file**, so `queries.toml` stays minimal and the authoring trace is audit-only.
