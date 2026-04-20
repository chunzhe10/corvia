# Canary Authoring Trace

Records the live `corvia_search` top-10 observed at authoring time for each canary query.
Audit-only. Not consumed by the eval harness. Not mutated after merge — if retrieval
behavior drifts, that drift is the signal the eval harness measures against the frozen
`queries.toml`, not a reason to edit this file.

**Authored:** 2026-04-20
**Corpus snapshot:** sha256:e4e6596393906e518812a6f3449efe51fd11840dd3edb4657924a409940dfa2d
**Retrieval stack at authoring time:** corvia server v1.0.1+, nomic-embed-text-v1.5 (768d), BM25+vector+cross-encoder rerank.

## Per-query traces

### canary-001 — lookup / learning
- Query: `why does my devcontainer leave stray .corvia directories inside subdirectories after using corvia`
- Expected: `019da398-62f2-7122-b945-f1176f08a3d2` at rank **1** (out of 10)
- Other top-10 (partial): `019d99c3-a54e-75d1-93ec-fbee046f78cd`, `019d9661-b6f2-7f52-b791-4b1349dd3ec3`, `019d9661-e6ce-7ab0-b8ab-db024a5d0b41`
- Notes: anchor surfaced at rank 1 with high confidence; symptom-first phrasing worked immediately

### canary-002 — lookup / learning
- Query: `how do I fix a second corvia serve process failing with database lock already open`
- Expected: `019d9661-e6ce-7ab0-b8ab-db024a5d0b41` at rank **1** (out of 10)
- Other top-10 (partial): `019d9661-b6f2-7f52-b791-4b1349dd3ec3`, `019d99c3-a54e-75d1-93ec-fbee046f78cd`
- Notes: operational gotcha lookup; anchor surfaced at rank 1 with high confidence; "database lock" phrasing strong signal

### canary-003 — lookup / reference
- Query: `what is the L0 constrained decoding approach and how does it compare to other LLM enforcement techniques`
- Expected: `019d9fa8-0a51-7b13-a0ce-d2a4eff7e0c7` at rank **1** (out of 10)
- Other top-10 (partial): `019d9fa6-03ea-7b73-b8e3-2069b2a9afe6`, `019d9f72-cbc3-7eb3-b2d3-94de53569b2a`, `019d9fe4-2da0-7041-bdc9-7a53fd00b5e7`
- Notes: reference lookup for LLM enforcement ladder L0 slide; anchor surfaced at rank 1; "constrained decoding" is a distinctive term in the entry title

### canary-004 — lookup / decision
- Query: `why was the release.yml workflow deleted from corvia-workspace`
- Expected: `019d99da-d4c3-7581-881b-85dbc7cc9009` at rank **1** (out of 10)
- Other top-10 (partial): `019d99db-0c8e-718e-af7a-4e5a5dd3e86e`, `019d99c3-a54e-75d1-93ec-fbee046f78cd`
- Notes: deletion rationale lookup; anchor is kind=decision (not learning as originally noted in plan); surfaced at rank 1 with high confidence

### canary-005 — lookup / instruction
- Query: `when should I use a subagent for carousel edits versus making the change directly in the main context`
- Expected: `019da147-8881-7971-89f0-a3157c95e8b4` at rank **1** (out of 10)
- Other top-10 (partial): `019da109-faed-7970-bddd-06e12ef65cbe`, `019da124-1786-7fb1-a33f-53bdc4e70818`
- Notes: workflow instruction lookup; superseding entry 019da147 surfaced at rank 1; superseded entry 019da109 also in top-10 as expected

### canary-006 — lookup / instruction
- Query: `what is the rule for when to spin subagents versus doing carousel edits directly and why was the original blanket rule changed`
- Expected: `019da147-8881-7971-89f0-a3157c95e8b4` at rank **1** (out of 10)
- Other top-10 (partial): `019da109-faed-7970-bddd-06e12ef65cbe`, `019da124-1786-7fb1-a33f-53bdc4e70818`
- Notes: second instruction lookup with different framing; probes rationale retrieval for refined workflow directive; "blanket rule changed" phrasing draws out the refinement narrative

### canary-007 — multi-hop / learning
- Query: `why did corvia switch from stdio MCP to HTTP transport and what operational gotchas should I know about the running server`
- Expected: `019d9661-b6f2-7f52-b791-4b1349dd3ec3` (rank 1), `019d9661-e6ce-7ab0-b8ab-db024a5d0b41` (in top-10)
- Other top-10 (partial): `019d99c3-a54e-75d1-93ec-fbee046f78cd`, `019da398-62f2-7122-b945-f1176f08a3d2`
- Notes: multi-hop architecture rationale (decision) + operational gotchas (learning); target_kind=learning refers to dominant retrieval nature of the pair; both surfaced in top-10

### canary-008 — multi-hop / learning
- Query: `what were the root causes of the corvia-workspace stray .corvia dirs problem and how did the corvia binary become outdated in the devcontainer`
- Expected: `019da398-62f2-7122-b945-f1176f08a3d2` (rank 1-2), `019d99c3-a54e-75d1-93ec-fbee046f78cd` (in top-10)
- Other top-10 (partial): `019d9661-e6ce-7ab0-b8ab-db024a5d0b41`, `019d9661-b6f2-7f52-b791-4b1349dd3ec3`
- Notes: multi-hop stray dirs root cause (learning) + devcontainer HTTP default and minimum binary version (decision); both surface in top-10

### canary-009 — multi-hop / learning
- Query: `why is a ±1% latency acceptance criterion unverifiable on the corvia_search workload and what alternatives work`
- Expected: `019d9fa3-4d75-7c20-a36b-b4cc6e50b579` (rank 1), `019d9f72-cbc3-7eb3-b2d3-94de53569b2a` (in top-10)
- Other top-10 (partial): `019d9fa6-03ea-7b73-b8e3-2069b2a9afe6`, `019d9f8b-70fe-7211-a3ce-6b9ef7fe44e3`
- Notes: multi-hop perf bench precision floor (learning) + phase 8 E2E validation (learning); both surface for inference-bound perf query; "unverifiable" phrasing is distinctive

### canary-010 — multi-hop / reference
- Query: `what are the research findings for LLM enforcement layers L7 policy engines and L8 session sandboxing`
- Expected: `019da02a-7ee8-7ff3-b52d-36444bc09123` (rank 1-2), `019da02a-f600-7801-af3c-1e88aa764cde` (in top-10)
- Other top-10 (partial): `019da02b-6e7c-7d30-89d5-b6cbcb66db09`, `019da02a-3555-73e6-bd82-cd51e2dfe91d`
- Notes: multi-hop L7 Policy Engines reference + L8 Session Sandboxing reference; both carousel research entries surface for policy-engine sandbox query

### canary-011 — multi-hop / reference
- Query: `what reference material exists for the LLM enforcement ladder covering levels L5 runtime reminders and L3 skills`
- Expected: `019d9ff3-ce77-75b3-aa18-5f9bf34b79fe` (rank 1), `019d9fe4-2da0-7041-bdc9-7a53fd00b5e7` (in top-10)
- Other top-10 (partial): `019d9fa8-0a51-7b13-a0ce-d2a4eff7e0c7`, `019d9fa6-03ea-7b73-b8e3-2069b2a9afe6`
- Notes: multi-hop L5 Runtime Reminders reference + L3 Skills reference; both surface together on enforcement ladder mid-tier query; L3 and L5 share enforcement ladder taxonomy

### canary-012 — multi-hop / decision
- Query: `what decisions were made about keeping temporal freshness out of the corvia entry schema and how should the v2 schema handle date-awareness`
- Expected: `019d99f6-0018-7963-8728-6122f3319e9d` (rank 1), `019d99f5-f883-7f32-89cc-638b8bb49be9` (in top-10)
- Other top-10 (partial): `019d99c3-a54e-75d1-93ec-fbee046f78cd`, `019d99da-d4c3-7581-881b-85dbc7cc9009`
- Notes: multi-hop temporal freshness decision + v1-to-v2 schema simplification history; "date-awareness" and "v2 schema" phrasing draws both entries into top-10

### canary-013 — reasoning / learning
- Query: `if my carousel vector PDF is over 40MB, what is causing the bloat and which CSS element should I target first to fix it`
- Expected: `019da124-1786-7fb1-a33f-53bdc4e70818` at rank **1** (out of 10)
- Other top-10 (partial): `019da147-8881-7971-89f0-a3157c95e8b4`, `019da109-faed-7970-bddd-06e12ef65cbe`
- Notes: reasoning query asking for root cause inference; entry explains grain noise SVG is the dominant contributor, not glows as first suspected; query asks "which CSS element" while entry provides the inferential basis

### canary-014 — reasoning / learning
- Query: `if a UserPromptSubmit hook injects a reminder every turn, can the model reliably be forced to comply with it, or is there a stronger enforcement option`
- Expected: `019da0de-aca2-7d52-a8cd-5eb39ea2113c` at rank **1** (out of 10)
- Other top-10 (partial): `019d9ff3-ce77-75b3-aa18-5f9bf34b79fe`, `019d9fe4-2da0-7041-bdc9-7a53fd00b5e7`
- Notes: reasoning requires inferring from soft/hard hook distinction that L5 is bypassable and L6 PreToolUse is the upgrade; entry provides the premises without stating the answer directly

### canary-015 — reasoning / decision
- Query: `should I add valid_from and valid_to fields to new corvia entries to help the agent reason about freshness`
- Expected: `019d99f6-0018-7963-8728-6122f3319e9d` at rank **1** (out of 10)
- Other top-10 (partial): `019d99f5-f883-7f32-89cc-638b8bb49be9`, `019d99c3-a54e-75d1-93ec-fbee046f78cd`
- Notes: reasoning query proposes a schema change the decision explicitly rejected; requires reading the decision to infer the correct answer is no; "valid_from valid_to" phrasing maps to schema temporal fields discussed in the entry

### canary-016 — reasoning / decision
- Query: `why would running corvia serve in HTTP mode let multiple MCP clients connect without lock contention, when stdio mode could not`
- Expected: `019d9661-b6f2-7f52-b791-4b1349dd3ec3` at rank **1** (out of 10)
- Other top-10 (partial): `019d9661-e6ce-7ab0-b8ab-db024a5d0b41`, `019d99c3-a54e-75d1-93ec-fbee046f78cd`
- Notes: reasoning requires inferring from redb/Tantivy lock semantics that HTTP holds handles open for the server lifetime; entry has the premises (flock, O_CREAT|O_EXCL, IndexHandles) but not a direct statement

### canary-017 — cross-kind / mixed
- Query: `what decisions were made about how corvia handles its MCP transport and what operational gotchas should a developer know when running the server`
- Expected: `019d9661-b6f2-7f52-b791-4b1349dd3ec3` (rank 1, kind=decision), `019d9661-e6ce-7ab0-b8ab-db024a5d0b41` (in top-10, kind=learning)
- Other top-10 (partial): `019d99c3-a54e-75d1-93ec-fbee046f78cd`, `019da398-62f2-7122-b945-f1176f08a3d2`
- Notes: cross-kind spanning decision + learning; both surface in top-10; kind diversity confirmed from entry frontmatter

### canary-018 — cross-kind / mixed
- Query: `how does corvia devcontainer start the server and what minimum binary version is needed for MCP to work, and what reference documents explain the L5 and L6 enforcement levels`
- Expected: `019d99c3-a54e-75d1-93ec-fbee046f78cd` (rank 1, kind=decision), `019d9ff3-ce77-75b3-aa18-5f9bf34b79fe` (in top-10, kind=reference)
- Other top-10 (partial): `019d9661-b6f2-7f52-b791-4b1349dd3ec3`, `019d9fe4-2da0-7041-bdc9-7a53fd00b5e7`
- Notes: cross-kind spanning decision + reference; compound query bridges devcontainer setup and enforcement ladder research; both surface in top-10

### canary-019 — cross-kind / mixed
- Query: `what is the policy for how corvia_search handles temporal freshness in its response envelope, and what reference material exists for L9 external verification to ensure outputs are grounded`
- Expected: `019d99f6-0018-7963-8728-6122f3319e9d` (rank 1, kind=decision), `019da02b-6e7c-7d30-89d5-b6cbcb66db09` (rank 2, kind=reference)
- Other top-10 (partial): `019d99f5-f883-7f32-89cc-638b8bb49be9`, `019da02a-7ee8-7ff3-b52d-36444bc09123`
- Notes: cross-kind spanning decision + reference; temporal freshness policy + L9 External Verifier reference; both surfaced at ranks 1 and 2 respectively

### canary-020 — cross-kind / mixed
- Query: `what did we learn from fixing the stray .corvia directory bug, what architecture decision was made about the HTTP transport that relates to lock contention, and what reference exists for L7 policy engines that could be applied to corvia`
- Expected: `019da398-62f2-7122-b945-f1176f08a3d2` (rank 2, kind=learning), `019d9661-b6f2-7f52-b791-4b1349dd3ec3` (rank 1, kind=decision), `019da02a-7ee8-7ff3-b52d-36444bc09123` (in top-5, kind=reference)
- Other top-10 (partial): `019d9661-e6ce-7ab0-b8ab-db024a5d0b41`, `019d99c3-a54e-75d1-93ec-fbee046f78cd`
- Notes: cross-kind spanning learning + decision + reference; the hardest cross-kind query (3 anchors, 3 kinds); all three surfaced in top-5 at authoring time; triple-compound phrasing needed to pull all three simultaneously

## Known limitations

Recorded here so downstream eval work (#126 retrieval harness, #128 bench CLI) can account
for them when interpreting canary metrics. These cannot be fixed under the frozen-forever
rule — document, don't mutate.

### 1. Correlated anchor reuse

Several anchors appear in 2+ canaries:

- `019d9661-b6f2-7f52-b791-4b1349dd3ec3` (MCP HTTP transport decision) → canary-007, 016, 017, 020
- `019d9661-e6ce-7ab0-b8ab-db024a5d0b41` (serve operational gotchas) → canary-002, 007, 017
- `019da147-8881-7971-89f0-a3157c95e8b4` (subagent workflow instruction) → canary-005, 006
- `019d99f6-0018-7963-8728-6122f3319e9d` (temporal freshness decision) → canary-012, 015, 019
- `019d99c3-a54e-75d1-93ec-fbee046f78cd` (devcontainer MCP HTTP default) → canary-008, 018
- `019da398-62f2-7122-b945-f1176f08a3d2` (stray .corvia dirs) → canary-001, 008, 020
- `019d99f5-f883-7f32-89cc-638b8bb49be9` (v1→v2 schema) → canary-012 (+ secondary in 015)

Effective distinct anchor count across 20 queries ≈ 13–14, not 20. A retrieval regression
on a single entry will drop recall across several canaries simultaneously, so aggregate
`recall@k` and `MRR` mean are correlated statistics. Downstream analyses should report
per-anchor recall alongside per-query metrics and weight-adjust when summarising. Corpus
invisibility (7 superseded/hidden entries of 71 on disk — see corvia#132) forced anchor
reuse; fixing requires broader visible-corpus coverage.

### 2. Cross-kind queries are conjunctive, not semantic bridges

canary-017/018/019/020 are structurally "X and Y (and Z)" conjunctions. Real agentic
cross-kind retrieval tends to look like a single symptom-first question whose answer
*happens to* require multiple kinds (e.g., "how do I diagnose foo?" → needs decision
rationale + learning workaround + reference spec). Current canaries measure **keyword
union** behavior, not **semantic bridging**. Eval harness should label cross-kind
metrics as "conjunctive cross-kind" to set correct expectations; genuine semantic
cross-kind should be covered via the Ragas synthetic testset (#125).

### 3. canary-005 and canary-006 share a single anchor

Both point to `019da147-8881-7971-89f0-a3157c95e8b4` with different query framings.
The instruction corpus has only 2 entries and one (`019da109-faed-7970-bddd-06e12ef65cbe`)
is superseded/invisible, forcing anchor reuse. Effective lookup coverage is 5 distinct
anchors, not 6. Document in eval reports that the lookup / instruction cell has n=1.

### 4. No adversarial query phrasings

All 20 queries are grammatical full-sentence English. Real agent input includes keyword
fragments ("mcp http lock"), typos, acronym-only, and pure-symptom phrasing with no
technical term. This canary does not probe robustness to such input. Gap recorded for
future adversarial canary set (propose `canary-adversarial.toml` in a follow-up — do
NOT edit this one).

### 5. Ceiling effect on MRR

Every anchor surfaced at rank 1–2 at authoring time. MRR will start near 1.0 and has
only downward headroom. That's the point (regression detection) but aggregate MRR
becomes a noisy drift signal. Recommend eval harness track per-query rank histograms
in addition to MRR means. At launch, report per-query Δrank from the authoring-time
baseline recorded above; any positive Δ (rank worsens) is the calibration signal.

### 6. All anchors are known-easy

Design §6 anticipated some `notes = "known-hard"` queries to extend regression headroom.
None were authored — the agent conservatively picked only confirmed-landable anchors
after the server-storage-path bug (workspace#55) consumed authoring time. A second
harder canary set is a reasonable follow-up.

### Corpus snapshot drift at review time

Snapshot hash at authoring: `sha256:e4e6596393...` (71 entries).
Snapshot hash at review: `sha256:21536bb21c...` (72 entries — one new entry written
between authoring and review).

This is expected and explicitly allowed by design §4.5: the hash is an audit trail,
not a validation gate. The drift does NOT invalidate any of the 20 canaries — their
expected_entry_ids are still present on disk.
