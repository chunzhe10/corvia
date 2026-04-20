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
- `validate.sh` — structural validator; run to confirm queries.toml is well-formed.

## Schema

See `queries.toml` comments and the design doc at `docs/rfcs/2026-04-20-canary-query-set-design.md`.
