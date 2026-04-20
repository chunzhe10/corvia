# Canary Query Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen 20-query canary calibration anchor for the corvia RAG eval harness, committed as static files under `repos/corvia/bench/canary/`.

**Architecture:** Three deliverables — `queries.toml` (the 20 canary queries with pinned entry IDs + provenance hash), `README.md` (the frozen-forever rule), and `snapshot.sh` (reproducible corpus hash). No crate code changes. The authoring agent reads candidate entries, drafts symptom-first queries, and verifies via the live `mcp__corvia__corvia_search` tool that each expected entry ID appears in the top-10 retrieval results before pinning.

**Tech Stack:** TOML (`toml` Python stdlib for validator), bash/sha256sum for the snapshot script, `mcp__corvia__corvia_search` for authoring-time verification.

**Design doc:** `repos/corvia/docs/rfcs/2026-04-20-canary-query-set-design.md` — read §4 before starting.

**Working directory for all tasks:** `/workspaces/corvia-workspace/repos/corvia` (the corvia subtree git root). Entries live at `/workspaces/corvia-workspace/.corvia/entries/` (workspace root, NOT inside repos/corvia). This path split is intentional: canary queries reference workspace corpus IDs and ship with the corvia binary.

**Corpus reality:** As of 2026-04-20, there are 71 entries (43 learning / 15 reference / 11 decision / 2 instruction). The two instruction entry filenames are `019da109-faed-7970-bddd-06e12ef65cbe.md` and `019da147-8881-7971-89f0-a3157c95e8b4.md` — confirm these still exist at Task 1.

**Stratification target matrix (from design §4.3):**

| query_type ↓ / target_kind → | learning | reference | decision | instruction | mixed | row total |
|---|---:|---:|---:|---:|---:|---:|
| lookup     | 2 | 1 | 1 | 2 | 0 | 6 |
| multi-hop  | 3 | 2 | 1 | 0 | 0 | 6 |
| reasoning  | 2 | 0 | 2 | 0 | 0 | 4 |
| cross-kind | 0 | 0 | 0 | 0 | 4 | 4 |
| **col total** | 7 | 3 | 4 | 2 | 4 | **20** |

Row totals (6/6/4/4) are **hard** constraints from the issue spec. Per-cell values are a starting plan; authoring may shift ±1 within a row/column as long as (a) row totals stay exact and (b) every column ≥2.

---

## Task 1: Scaffold directory and snapshot script

**Files:**
- Create: `repos/corvia/bench/canary/snapshot.sh`
- Create: `repos/corvia/bench/canary/.gitkeep` (temporary — removed after Task 3)

- [ ] **Step 1.1: Create directory**

```bash
mkdir -p /workspaces/corvia-workspace/repos/corvia/bench/canary
touch /workspaces/corvia-workspace/repos/corvia/bench/canary/.gitkeep
```

- [ ] **Step 1.2: Write `snapshot.sh`**

Create `/workspaces/corvia-workspace/repos/corvia/bench/canary/snapshot.sh` with the following exact contents. This script emits a deterministic `sha256:<hex>` string over the workspace entry corpus.

```bash
#!/usr/bin/env bash
# snapshot.sh — emit a deterministic corpus snapshot hash for the canary query set.
#
# Hash is sha256 over sorted lines of "<entry_id>\t<sha256_of_file_bytes>".
# Filesystem-order-independent, mtime-independent.
# Invoke from anywhere; entries path is resolved relative to the workspace root.
set -euo pipefail

WORKSPACE_ROOT="${CORVIA_WORKSPACE:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
ENTRIES_DIR="${WORKSPACE_ROOT}/.corvia/entries"

if [[ ! -d "${ENTRIES_DIR}" ]]; then
  echo "error: entries dir not found: ${ENTRIES_DIR}" >&2
  exit 1
fi

# shellcheck disable=SC2012
find "${ENTRIES_DIR}" -maxdepth 1 -name '*.md' -type f \
  | sort \
  | while read -r f; do
      eid="$(basename "$f" .md)"
      h="$(sha256sum "$f" | awk '{print $1}')"
      printf '%s\t%s\n' "${eid}" "${h}"
    done \
  | sha256sum \
  | awk '{print "sha256:" $1}'
```

- [ ] **Step 1.3: Make it executable**

```bash
chmod +x /workspaces/corvia-workspace/repos/corvia/bench/canary/snapshot.sh
```

- [ ] **Step 1.4: Verify determinism**

Run the script twice; the output must match byte-for-byte.

```bash
cd /workspaces/corvia-workspace/repos/corvia/bench/canary
A="$(./snapshot.sh)"
B="$(./snapshot.sh)"
[[ "$A" == "$B" ]] && echo "OK determinism: $A" || { echo "FAIL: $A != $B"; exit 1; }
```

Expected output: `OK determinism: sha256:<64 hex chars>`. Record the hash value — it will be pasted into `queries.toml` in Task 2.

- [ ] **Step 1.5: Verify entry count**

```bash
ls /workspaces/corvia-workspace/.corvia/entries/*.md | wc -l
```

Expected: `71` (matches design §4.3). If different, update `corpus_entry_count` in Task 2 accordingly.

- [ ] **Step 1.6: Verify the 2 instruction entries still exist**

```bash
grep -l '^kind = "instruction"' /workspaces/corvia-workspace/.corvia/entries/*.md | wc -l
```

Expected: `2`. If 0 or 1, stratification must change — STOP and inform the user.

- [ ] **Step 1.7: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/snapshot.sh bench/canary/.gitkeep
git commit -m "feat(bench): add corpus snapshot script for canary set (#124)"
```

---

## Task 2: Write `queries.toml` header

**Files:**
- Create: `repos/corvia/bench/canary/queries.toml`

- [ ] **Step 2.1: Compute the snapshot hash**

```bash
SNAP="$(/workspaces/corvia-workspace/repos/corvia/bench/canary/snapshot.sh)"
echo "$SNAP"
```

Copy the `sha256:<hex>` string — use it verbatim in Step 2.2.

- [ ] **Step 2.2: Write the header**

Create `/workspaces/corvia-workspace/repos/corvia/bench/canary/queries.toml` with the following contents. Replace `<SNAP>` with the exact string from Step 2.1.

```toml
# Canary query set for corvia RAG eval harness.
# FROZEN FOREVER. See README.md before touching this file.
# Issue: https://github.com/chunzhe10/corvia/issues/124

schema_version = 1
created_at = "2026-04-20"
corpus_snapshot_hash = "<SNAP>"
corpus_entry_count = 71
notes = "Hand-curated (agent-authored) canary set. Expected entry IDs verified via live corvia_search top-10 at authoring time. Do not mutate."

# Query blocks appended in Tasks 3–6.
```

- [ ] **Step 2.3: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/queries.toml
git commit -m "feat(bench): canary queries.toml header + corpus snapshot (#124)"
```

---

## Task 3: Author the 6 lookup queries

**Files:**
- Modify: `repos/corvia/bench/canary/queries.toml` (append 6 `[[query]]` blocks)

Target allocation: 2 learning, 1 reference, 1 decision, 2 instruction.

**Per-query authoring protocol** (repeat 6×):

- [ ] **Step 3.N.a: Pick target kind for this slot**

Example for the first lookup query: target_kind = "learning". Select candidate entries:

```bash
grep -l '^kind = "learning"' /workspaces/corvia-workspace/.corvia/entries/*.md | head -10
```

Pick one whose content anchors a natural single-entry question. Read it:

```bash
cat /workspaces/corvia-workspace/.corvia/entries/<picked-id>.md
```

- [ ] **Step 3.N.b: Draft the query**

Phrase as a developer would actually ask it — symptom-first, not keyword soup. Example format:

> Bad: `"devcontainer taskfile workspace_root hardcoded"` (keyword echo)
> Good: `"why does my devcontainer post-start fail when I copy .devcontainer to a different repo?"` (symptom-first)

- [ ] **Step 3.N.c: Verify via live corvia_search**

Call the `mcp__corvia__corvia_search` tool (not bash) with:

```
query: "<drafted query>"
limit: 10
min_score: (omit)
```

The picked entry ID MUST appear in the returned `results[].id` within the top 10. If it doesn't:

- Option 1: rephrase the query and re-verify (preferred).
- Option 2: pick a different candidate entry and re-verify.
- Do NOT lower expectations or pick a more obscure anchor just to make retrieval work.

- [ ] **Step 3.N.d: Record the block**

Append to `repos/corvia/bench/canary/queries.toml`:

```toml
[[query]]
id = "canary-001"                # canary-001 through canary-006 for lookup
query = "<the verified query>"
expected_entry_ids = ["<anchor-id>"]
query_type = "lookup"
target_kind = "learning"          # or reference/decision/instruction per matrix
notes = "<what retrieval capability this probes — 1 sentence>"
```

- [ ] **Step 3.N.e: Record the authoring trace row**

Keep a running scratch file (will become `authoring-trace.md` in Task 7). For each query record:

```
canary-001 | target: <entry-id> | top-10 ranks: [<id1>, <id2>, …] | note: <anchor appeared at rank N>
```

**Lookup slot plan (adjust ±1 if needed, preserve column totals):**

| # | target_kind | cell |
|---|---|---|
| canary-001 | learning | row 1 col 1 |
| canary-002 | learning | row 1 col 1 |
| canary-003 | reference | row 1 col 2 |
| canary-004 | decision | row 1 col 3 |
| canary-005 | instruction | row 1 col 4 (use 019da109-faed-7970-bddd-06e12ef65cbe) |
| canary-006 | instruction | row 1 col 4 (use 019da147-8881-7971-89f0-a3157c95e8b4) |

- [ ] **Step 3.7: Commit after all 6 authored**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/queries.toml
git commit -m "feat(bench): canary queries — 6 lookup (#124)"
```

---

## Task 4: Author the 6 multi-hop queries

**Files:**
- Modify: `repos/corvia/bench/canary/queries.toml`

Target allocation: 3 learning, 2 reference, 1 decision. Each multi-hop query has **2 expected entry IDs** (sometimes 3 if the content genuinely requires it). Both must appear in top-10 for the verification to pass.

**Per-query protocol** (repeat 6×) — same as Task 3, with these differences:

- **Step 4.N.a (pick):** Identify 2 entries of the target_kind whose contents, when combined, answer a natural question that neither answers alone. Example: two learning entries about different workarounds for the same root cause.
- **Step 4.N.c (verify):** BOTH expected IDs must appear in top-10. If only one appears, either rephrase (so both surface), pick different paired entries, or downgrade to lookup + move this slot's count. Do not downgrade the row total — rebalance within the row.
- **Step 4.N.d (record):** `expected_entry_ids` has 2 elements, ranked (first = more central).

**Multi-hop slot plan:**

| # | target_kind | count of IDs |
|---|---|---|
| canary-007 | learning | 2 |
| canary-008 | learning | 2 |
| canary-009 | learning | 2 |
| canary-010 | reference | 2 |
| canary-011 | reference | 2 |
| canary-012 | decision | 2 |

- [ ] **Step 4.7: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/queries.toml
git commit -m "feat(bench): canary queries — 6 multi-hop (#124)"
```

---

## Task 5: Author the 4 reasoning queries

**Files:**
- Modify: `repos/corvia/bench/canary/queries.toml`

Target allocation: 2 learning, 2 decision. Reasoning queries require *inferring* an answer from entry content — the entry does not state the answer literally. Example: query asks "what should I use instead of X for Y?"; the entry explains X's flaw and mentions Y exists, leaving the inference "Y is the replacement" to the retriever's consumer.

**Per-query protocol** — same as Task 3, plus:

- **Step 5.N.a (pick):** Pick an entry whose content, while not directly stating an answer, contains the premises needed to infer one. Decision entries are good anchors because they describe tradeoffs that enable inference.
- **Step 5.N.c (verify):** The anchor entry must appear in top-10. Do not verify whether the inference "would be correct" — that's the generation harness's job (#127), not retrieval's.

**Reasoning slot plan:**

| # | target_kind |
|---|---|
| canary-013 | learning |
| canary-014 | learning |
| canary-015 | decision |
| canary-016 | decision |

- [ ] **Step 5.5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/queries.toml
git commit -m "feat(bench): canary queries — 4 reasoning (#124)"
```

---

## Task 6: Author the 4 cross-kind queries

**Files:**
- Modify: `repos/corvia/bench/canary/queries.toml`

All 4 use `target_kind = "mixed"`. Each query's `expected_entry_ids` spans **at least 2 different kinds** — e.g., one learning + one decision, or one reference + one instruction.

**Per-query protocol** — same as Task 4 (multi-hop) for structural purposes, plus:

- **Step 6.N.a (pick):** Identify 2–3 entries from **different** kinds whose combination answers a natural question. Verify the kind diversity:

```bash
for id in <id1> <id2> <id3>; do
  grep '^kind' /workspaces/corvia-workspace/.corvia/entries/${id}.md
done
```

The output must show at least 2 distinct `kind = "..."` values.

- **Step 6.N.c (verify):** ALL expected IDs must appear in top-10. These are the hardest queries to land — budget extra time for rephrasing.

**Cross-kind slot plan:**

| # | expected_entry_ids count | kinds represented |
|---|---:|---|
| canary-017 | 2 | learning + decision |
| canary-018 | 2 | learning + reference |
| canary-019 | 2 | decision + reference |
| canary-020 | 3 | learning + decision + reference (if feasible; else 2) |

- [ ] **Step 6.5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/queries.toml
git commit -m "feat(bench): canary queries — 4 cross-kind (#124)"
```

---

## Task 7: Write `authoring-trace.md`

**Files:**
- Create: `repos/corvia/bench/canary/authoring-trace.md`

Consolidate the per-query scratch records from Steps 3.N.e / 4.N.e / 5.N.e / 6.N.e.

- [ ] **Step 7.1: Write the file**

Create `/workspaces/corvia-workspace/repos/corvia/bench/canary/authoring-trace.md`:

```markdown
# Canary Authoring Trace

Records the live `corvia_search` top-10 observed at authoring time for each canary query.
Audit-only. Not consumed by the eval harness. Not mutated after merge — if retrieval
behavior drifts, that drift is the signal the eval harness measures against the frozen
`queries.toml`, not a reason to edit this file.

**Authored:** 2026-04-20
**Corpus snapshot:** <paste the same sha256:... from queries.toml>
**Retrieval stack at authoring time:** corvia server v1.0.0, nomic-embed-text-v1.5 (768d), BM25+vector+cross-encoder rerank.

## Per-query traces

### canary-001 — lookup / learning
- Query: `<exact query string>`
- Expected: `<anchor-id>` at rank **N** (out of 10)
- Other top-10 (ranks 1–10): `[<id1>, <id2>, …]`
- Notes: `<anything noteworthy — e.g., "rank 1 was a superseded duplicate; anchor was rank 2">`

### canary-002 — lookup / learning
<same structure>

… (through canary-020)
```

- [ ] **Step 7.2: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/authoring-trace.md
git commit -m "docs(bench): authoring-trace for canary queries (#124)"
```

---

## Task 8: Write `README.md` (frozen-forever rule)

**Files:**
- Create: `repos/corvia/bench/canary/README.md`
- Delete: `repos/corvia/bench/canary/.gitkeep` (no longer needed)

- [ ] **Step 8.1: Write the README**

Create `/workspaces/corvia-workspace/repos/corvia/bench/canary/README.md`:

```markdown
# Canary Query Set — Frozen Forever

This directory contains the **frozen calibration anchor** for the corvia RAG eval harness.

## Rule 1: Never mutate `queries.toml`

- Never change an `expected_entry_ids` list.
- Never add a `[[query]]` block.
- Never remove a `[[query]]` block.
- Never edit a `query` string.

If the canary set feels insufficient, add a **new** file (e.g., `canary-v2.toml`) — do not edit this one. The frozen set's value is its invariance.

## Rule 2: Corpus drift does not trigger updates

- If a corpus entry referenced in `expected_entry_ids` is superseded, **do nothing**. Supersession preserves the original entry ID (corvia uses append-only semantics); the ID is still valid.
- If a corpus entry is genuinely deleted (not superseded), that's a corpus integrity bug — file it separately. Do not edit the canary.

## Rule 3: The `corpus_snapshot_hash` is an audit trail, not a gate

The hash recorded in `queries.toml` answers the question *"what was the corpus when these queries were calibrated?"*. The eval harness does **not** validate the hash at run time — doing so would defeat the point of a frozen set (the eval is supposed to keep running even as the corpus grows).

## Why these rules?

Regression metrics require an invariant baseline. If the baseline moves, every regression becomes ambiguous: did retrieval get worse, or did the testset shift? The canary set exists precisely to be immune to that ambiguity.

## Relationship to other eval artifacts

| File | Mutable? | Purpose |
|---|---|---|
| `queries.toml` (this dir) | **No — frozen** | Stability anchor; 20 queries, 20 pins |
| Ragas synthetic testset (#125) | Yes — regenerates | Broad recall measurement |
| `authoring-trace.md` | No — historical record | Audit trail for the original authoring |

## Files in this directory

- `queries.toml` — the frozen 20 canary queries.
- `README.md` — this file.
- `snapshot.sh` — reproducible corpus-hash computation (used at authoring time, kept for audit).
- `authoring-trace.md` — per-query retrieval trace observed at authoring time.

## Schema

See `queries.toml` comments and the design doc at `docs/rfcs/2026-04-20-canary-query-set-design.md`.
```

- [ ] **Step 8.2: Delete `.gitkeep`**

```bash
rm /workspaces/corvia-workspace/repos/corvia/bench/canary/.gitkeep
```

- [ ] **Step 8.3: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/README.md bench/canary/.gitkeep
git commit -m "docs(bench): canary README with frozen-forever rule (#124)"
```

---

## Task 9: Validator + final structural check (TDD)

**Files:**
- Create: `repos/corvia/bench/canary/validate.sh`

A small validation script that parses `queries.toml`, checks the schema, counts strata, and confirms every expected entry ID exists on disk. This is run now (to catch any authoring errors before merge) and can be re-run in CI by issue #128.

- [ ] **Step 9.1: Write the failing validator**

Create `/workspaces/corvia-workspace/repos/corvia/bench/canary/validate.sh`:

```bash
#!/usr/bin/env bash
# validate.sh — structural check on queries.toml. Exits non-zero on violation.
# Usage: ./validate.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="${CORVIA_WORKSPACE:-$(cd "${HERE}/../../../.." && pwd)}"
ENTRIES_DIR="${WORKSPACE_ROOT}/.corvia/entries"
TOML="${HERE}/queries.toml"

python3 - "$TOML" "$ENTRIES_DIR" <<'PY'
import sys, os, re
try:
    import tomllib
except ImportError:
    import tomli as tomllib

toml_path, entries_dir = sys.argv[1], sys.argv[2]
with open(toml_path, 'rb') as f:
    data = tomllib.load(f)

errors = []

# Header checks
for key in ("schema_version", "created_at", "corpus_snapshot_hash", "corpus_entry_count"):
    if key not in data:
        errors.append(f"missing header key: {key}")

if data.get("schema_version") != 1:
    errors.append(f"schema_version != 1: {data.get('schema_version')}")

snap = data.get("corpus_snapshot_hash", "")
if not re.match(r"^sha256:[0-9a-f]{64}$", snap):
    errors.append(f"corpus_snapshot_hash malformed: {snap}")

queries = data.get("query", [])
if len(queries) != 20:
    errors.append(f"expected 20 queries, got {len(queries)}")

# Per-query checks
required_fields = {"id", "query", "expected_entry_ids", "query_type", "target_kind", "notes"}
valid_types = {"lookup", "multi-hop", "reasoning", "cross-kind"}
valid_kinds = {"learning", "reference", "decision", "instruction", "mixed"}
seen_ids = set()
type_counts = {t: 0 for t in valid_types}
kind_counts = {k: 0 for k in valid_kinds}

for i, q in enumerate(queries):
    missing = required_fields - q.keys()
    if missing:
        errors.append(f"query[{i}] missing: {sorted(missing)}")
        continue
    if q["id"] in seen_ids:
        errors.append(f"duplicate query id: {q['id']}")
    seen_ids.add(q["id"])
    if q["query_type"] not in valid_types:
        errors.append(f"{q['id']}: invalid query_type {q['query_type']}")
    else:
        type_counts[q["query_type"]] += 1
    if q["target_kind"] not in valid_kinds:
        errors.append(f"{q['id']}: invalid target_kind {q['target_kind']}")
    else:
        kind_counts[q["target_kind"]] += 1
    if not isinstance(q["expected_entry_ids"], list) or not q["expected_entry_ids"]:
        errors.append(f"{q['id']}: expected_entry_ids must be non-empty list")
        continue
    for eid in q["expected_entry_ids"]:
        path = os.path.join(entries_dir, f"{eid}.md")
        if not os.path.isfile(path):
            errors.append(f"{q['id']}: entry file not found: {eid}.md")
    if q["query_type"] == "multi-hop" and len(q["expected_entry_ids"]) < 2:
        errors.append(f"{q['id']}: multi-hop needs ≥2 expected_entry_ids")
    if q["query_type"] == "cross-kind":
        kinds_seen = set()
        for eid in q["expected_entry_ids"]:
            path = os.path.join(entries_dir, f"{eid}.md")
            try:
                with open(path) as ef:
                    for line in ef:
                        m = re.match(r'^kind\s*=\s*"([^"]+)"', line)
                        if m:
                            kinds_seen.add(m.group(1))
                            break
            except FileNotFoundError:
                pass
        if len(kinds_seen) < 2:
            errors.append(f"{q['id']}: cross-kind requires ≥2 distinct kinds among expected; got {kinds_seen}")
        if q["target_kind"] != "mixed":
            errors.append(f"{q['id']}: cross-kind must have target_kind=mixed")

# Strata totals
expected_type = {"lookup": 6, "multi-hop": 6, "reasoning": 4, "cross-kind": 4}
for t, want in expected_type.items():
    if type_counts[t] != want:
        errors.append(f"query_type {t}: want {want}, got {type_counts[t]}")

for k in valid_kinds:
    if kind_counts[k] < 2:
        errors.append(f"target_kind {k}: below floor-of-2 (got {kind_counts[k]})")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"VALIDATION OK: 20 queries, types={type_counts}, kinds={kind_counts}")
PY
```

- [ ] **Step 9.2: Make it executable and run it**

```bash
chmod +x /workspaces/corvia-workspace/repos/corvia/bench/canary/validate.sh
/workspaces/corvia-workspace/repos/corvia/bench/canary/validate.sh
```

Expected: `VALIDATION OK: 20 queries, types={...}, kinds={...}`.

- [ ] **Step 9.3: If validation fails, fix the queries.toml**

Read the error list. Common fixes:
- Missing entry ID on disk → either the ID is typo'd (fix it in queries.toml) or the entry was deleted between Task 1 and now (pick a different anchor and re-verify via corvia_search).
- Strata imbalance → find the offending query and re-classify or swap with another query in the same row/column.
- Cross-kind not mixed → ensure all 4 cross-kind entries have `target_kind = "mixed"`.

Re-run Step 9.2 until it passes.

- [ ] **Step 9.4: Commit the validator**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add bench/canary/validate.sh
git commit -m "feat(bench): queries.toml structural validator (#124)"
```

---

## Self-Review Checklist (run after implementation)

- [ ] All 5 AC items from design §5 covered in `queries.toml` + `README.md` + snapshot hash.
- [ ] Row totals exact: 6 lookup / 6 multi-hop / 4 reasoning / 4 cross-kind.
- [ ] Every `target_kind` stratum has ≥2 queries (including `mixed`).
- [ ] Every `expected_entry_ids` entry exists as a file in `.corvia/entries/`.
- [ ] `corpus_snapshot_hash` matches current output of `snapshot.sh`.
- [ ] `validate.sh` exits 0.
- [ ] All 4 cross-kind queries have `target_kind = "mixed"` AND span ≥2 distinct kinds in their expected IDs.
- [ ] `README.md` contains the three rules (never mutate, corpus drift doesn't trigger updates, hash is audit-only).
- [ ] `authoring-trace.md` has 20 entries, one per canary.

## Notes for the executing agent

- **Use `mcp__corvia__corvia_search`, not a bash wrapper.** The MCP tool runs against the live corvia server and reflects the current index. A bash `corvia search` might hit a stale CLI path.
- **Rank the `expected_entry_ids`.** For multi-hop and cross-kind, order matters (position 0 = most central). The eval harness uses this order for MRR and top-k recall gap metrics.
- **Do not skip verification to save time.** A canary query whose target never appears in top-10 is worse than useless — it locks in a false baseline. If verification keeps failing for a slot, rebalance within the row (move a count from this slot to an adjacent cell) and document the deviation in the PR description.
- **Budget:** rough estimate is 10–15 minutes per query × 20 = 3–5 hours of dedicated work. Budget accordingly; do not rush.
