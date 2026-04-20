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
        errors.append(f"{q['id']}: multi-hop needs >=2 expected_entry_ids")
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
            errors.append(f"{q['id']}: cross-kind requires >=2 distinct kinds among expected; got {kinds_seen}")
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
