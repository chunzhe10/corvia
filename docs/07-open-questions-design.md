# Corvia V2 Open Questions -- Design Decisions

**Date**: 2026-04-15
**Status**: Approved (revised after 5-persona review)
**Author**: chunzhe + Claude (brainstorming session)
**Version**: 1.0.0

## Summary

This document resolves the open questions from the v2 design phase and incorporates
findings from the 5-persona review (SWE, QA, PM, AI Engineer, IR Engineer). Each
decision was evaluated against v2's core principles: local-first, single binary, lean,
maintainable by one person.

## Decision Table

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Storage format | Flat files + Redb indexes | Human-readable, git-syncable, Redb proven in v1 |
| 2 | Versioning | Git + lightweight frontmatter | Git is the temporal layer, frontmatter for supersession |
| 3 | File format | TOML frontmatter + markdown body | Unambiguous, toml crate already in deps |
| 4 | Graph | Dropped | Vector + BM25 + reranking sufficient. YAGNI. |
| 5 | Cedar/RBAC | Dropped | Local-first, one corvia per product |
| 6 | Agent tracking | Dropped | Knowledge matters, not author. Git blame for provenance. |
| 7 | Classification | Single `kind` field, 4 values | Reduced from 7 after review: LLMs classify more reliably with fewer categories |
| 8 | Supersession | Auto-dedup + manual override | Near-duplicate writes auto-supersede. Manual `supersedes` for semantic contradictions. |
| 9 | Auto-classification | Caller LLM classifies via MCP tool description | Zero cost to corvia |
| 10 | Tiers | Dropped for v1.0 | Corpus is small (<5K). Search everything. Add tiers in v1.1 if needed. |
| 11 | CLI commands | 5: ingest, search, write, status, mcp | Lean surface |
| 12 | Testing | Unit + integration | Per module + full pipeline |
| 13 | Deletion | Delete the file, ingest cleans up | Git has the history. No soft-delete state. |
| 14 | Dedup on write | Auto-supersede near-duplicates | If similar entry exists, new entry supersedes it automatically |
| 15 | IDs | UUIDv7, lowercase | Time-ordered, zero collision risk, case-safe on all filesystems |
| 16 | Version | 1.0.0 | Fresh start, no migration from v1 |

---

## 1. Storage and Data Model

### Knowledge Entries (git-tracked)

Flat files at `.corvia/entries/<id>.md` with TOML frontmatter + markdown body:

```
+++
id = "01963f1a-7b2d-7000-8000-1a2b3c4d5e6f"
created_at = "2026-04-15T10:00:00Z"
kind = "decision"
supersedes = ["01963f1a-1234-7000-8000-aabbccddeeff"]
tags = ["storage", "architecture"]
+++

Corvia v2 uses Redb for indexes and flat files for knowledge entries.
The index is rebuilt from entries via `corvia ingest`.
```

**Frontmatter fields:**
- `id` (string, required) -- UUIDv7, lowercase. Time-ordered, 128-bit, zero collision risk. Generated on write.
- `created_at` (datetime, required) -- ISO 8601 timestamp
- `kind` (string, optional, default "learning") -- one of: `decision`, `learning`, `instruction`, `reference`
- `supersedes` (string[], optional) -- IDs of entries this one replaces. Set automatically on dedup, or manually by caller.
- `tags` (string[], optional) -- freeform labels for filtering

**`kind` taxonomy (4 values):**
- `decision` -- choices made with rationale, architecture decisions, trade-off evaluations
- `learning` -- discovered insights, gotchas, patterns, workarounds, debug findings (default)
- `instruction` -- how-to guides, setup procedures, workflows, processes
- `reference` -- implementation patterns, code examples, API docs, config documentation

### ID Generation

UUIDv7 (RFC 9562). Time-ordered (first 48 bits are millisecond timestamp), 128-bit
random suffix. Zero collision risk. Always lowercase. Case-safe on macOS (HFS+)
and Windows (NTFS) filesystems.

Generated via the `uuid` crate with `uuid::Uuid::now_v7()`.

### Index (gitignored, rebuilt from entries)

At `.corvia/index/`:
- `store.redb` -- vectors (entry_id to f32 embedding), chunk-to-entry mappings, supersession state (which entries are current vs superseded)
- `tantivy/` -- BM25 full-text index directory

### Lifecycle

- `.corvia/entries/` is git-tracked. Human-readable, diffable, syncable across machines.
- `.corvia/index/` is gitignored. Rebuilt from entries via `corvia ingest`.
- `corvia ingest` reads all entry files, strips frontmatter, chunks markdown body, embeds, and populates both indexes.
- Superseded entries are marked in the index at ingest time and excluded from search candidates.

### Deletion

Delete the entry file from `.corvia/entries/`. The next `corvia ingest` removes it
from the index. `corvia ingest --fresh` rebuilds everything from scratch.
Git retains the file history for recovery.

No soft-delete mechanism. No delete CLI command in v1.0.

### Entry/Index Drift Detection

On search, compare entry file count against indexed entry count stored in Redb.
If they differ, include a warning in the quality signal:
`suggestion: "Index may be stale. Run 'corvia ingest' to update."`

This catches manual file edits, `git pull` with new entries, and file deletions
between ingests.

---

## 2. Write Safety

### Atomicity: File-First Ordering

`corvia_write` follows this order:

1. Generate UUIDv7 ID
2. Serialize TOML frontmatter + markdown body
3. Write to temp file (`.corvia/entries/.<id>.md.tmp`)
4. Atomic rename to final path (`.corvia/entries/<id>.md`)
5. Update Redb index (vector, chunk mappings, supersession state)
6. Update tantivy index

**Recovery invariant**: the flat file is the source of truth. If the process crashes
after step 4 but before step 5/6, the entry exists as a file but is not indexed.
The next `corvia ingest` picks it up. Orphan index entries (step 5 without step 4)
cannot happen because file-write comes first.

### Concurrency Model

- **stdio MCP is inherently serial**: one request at a time over stdin/stdout. No concurrent MCP tool calls within a single `corvia mcp` process.
- **CLI vs running MCP**: `corvia write` from CLI while `corvia mcp` is running could race. Redb uses file-level locking (single-writer). The second writer blocks until the first commits. Tantivy's `IndexWriter` is held by a single process. If two processes attempt concurrent writes, the second gets a lock error.
- **Documented behavior**: "Run one `corvia mcp` process per project. CLI commands that write (ingest, write) should not be run concurrently with a running MCP server."

### Supersession Validation

- If `supersedes` references a non-existent ID: write succeeds with a warning in the response (`"warning": "superseded entry 'xxx' not found"`). The entry is still created. This prevents blocking writes due to stale references.
- Circular supersession (A supersedes B, B supersedes A): resolved by `created_at` ordering. The most recently created entry wins. Older entry is marked superseded. Detected and resolved during `corvia ingest`.

---

## 3. Dedup on Write (Auto-Supersession)

When `corvia_write` is called:

1. Embed the incoming content
2. Search existing entries for similarity > 0.85 threshold
3. **If near-duplicate found**: create new entry with `supersedes = [matched_id]` automatically. The old entry becomes superseded.
4. **If no match**: create new entry with no supersedes
5. **If caller explicitly passed `supersedes`**: use caller's value (overrides auto-detection)

**Response always includes:**
```json
{
  "id": "01963f1a-7b2d-7000-8000-1a2b3c4d5e6f",
  "action": "created",
  "superseded": []
}
// or
{
  "id": "01963f1a-7b2d-7000-8000-1a2b3c4d5e6f",
  "action": "superseded",
  "superseded": ["01963f1a-1234-7000-8000-aabbccddeeff"],
  "similarity": 0.91
}
```

The calling LLM does not need to search-then-write for updates. Just write. Corvia
handles the versioning chain. Manual `supersedes` remains available for the case
where replacement content is semantically different (contradiction, not update).

**Dedup threshold**: 0.85 cosine similarity. Configurable in `corvia.toml`.

---

## 4. Retrieval Pipeline

```
query -> embed (fastembed, nomic-embed-text-v1.5)
      -> [BM25 search (tantivy, pre-filtered: kind, superseded excluded)]  \
                                                                             -> RRF fusion (k=30)
      -> [vector search (cosine, pre-filtered: superseded excluded)]        /
      -> cross-encoder rerank (fastembed, ms-marco-MiniLM-L6-v2)
      -> quality signal
      -> results
```

### Steps

1. **Embed query** via fastembed (nomic-embed-text-v1.5, 768d)
2. **Pre-filter**: exclude superseded entries from both search paths. Apply `kind` filter if specified (with 3x oversampling to compensate for post-filter elimination).
3. **BM25 search** via tantivy. Good for exact terms, code patterns, config names.
4. **Vector search** via cosine similarity. Brute-force for <10K vectors, HNSW above that threshold. Good for semantic/conceptual queries.
5. **RRF fusion** merges both result sets. Score: `sum(1 / (k + rank_i))`, k=30 (tuned for small corpora; configurable in corvia.toml).
6. **Cross-encoder rerank** top 50 candidates down to `limit`. Note: the +33% accuracy figure is from web search benchmarks (MS MARCO). Actual gains on technical knowledge entries may be lower (~15-20%) due to domain gap. ms-marco-MiniLM-L6-v2 has a 512-token combined limit (query + passage); chunks are sized to respect this.
7. **Quality signal** computed on reranker scores (not post-weighted). Confidence thresholds are provisional for v1.0 and will be recalibrated after integration testing with real workloads.
8. **Return** results with quality signal.

### Superseded Entry Filtering

Superseded entries are marked in Redb during ingest. Both BM25 and vector searches
exclude them at query time. This prevents superseded entries from consuming reranker
candidate slots.

### Vector Search: Brute-Force vs HNSW

For corpus sizes typical of corvia (100-5000 entries, up to 25K chunks):
- **<10K vectors**: brute-force cosine similarity. 100% recall, <1ms latency at this scale.
- **>=10K vectors**: switch to HNSW (hnsw_rs). ef_construction=200, ef_search=64, max_connections=16.

This eliminates HNSW overhead for the common case while preserving scalability.

### Chunking and Frontmatter

- **Strip frontmatter before chunking.** TOML frontmatter is metadata, not semantic content. Including it in chunks wastes ~10-15% of the 512-token window and pollutes embeddings.
- **Store `kind` and `tags` as structured metadata** in Redb alongside each chunk. Used for pre-retrieval filtering.
- **Chunk-to-entry mapping**: each chunk record in Redb includes `source_entry_id` so search results trace back to the original file.
- **Short entries** (<512 tokens body): single chunk, no splitting needed.

### Config Defaults

| Parameter | Value | Source |
|-----------|-------|--------|
| Chunk max_tokens | 512 | FloTorch 2026 benchmark |
| Chunk overlap | 64 (~12.5%) | NVIDIA 15% optimum |
| Chunk min_tokens | 32 | Merge threshold |
| RRF k | 30 | Tuned for small corpora (configurable) |
| Dedup threshold | 0.85 | Cosine similarity for auto-supersession (configurable) |
| Reranker candidates | 50 | Retrieve 50, return top `limit` |
| Embedding model | nomic-embed-text-v1.5 | 62.4 MTEB, 768d, proven in v1 |
| Reranker model | ms-marco-MiniLM-L6-v2 | Smallest, fastest. Domain gap acknowledged. |
| Brute-force threshold | 10,000 vectors | Below: brute-force cosine. Above: HNSW. |

---

## 5. MCP Tool Schemas

### corvia_search

Hybrid semantic + keyword search with cross-encoder reranking.

```json
{
  "name": "corvia_search",
  "description": "Hybrid semantic + keyword search with reranking across organizational knowledge. Returns ranked results with quality signals. Results are raw chunks (no LLM synthesis) for the calling agent to interpret directly.",
  "inputSchema": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query (natural language or keywords)"
      },
      "limit": {
        "type": "integer",
        "default": 5,
        "description": "Maximum results to return (default 5)"
      },
      "max_tokens": {
        "type": "integer",
        "description": "Maximum total tokens across all results. Truncates or reduces result count to fit budget."
      },
      "min_score": {
        "type": "number",
        "description": "Minimum relevance score threshold (0.0-1.0). If omitted, no floor is applied."
      },
      "kind": {
        "type": "string",
        "enum": ["decision", "learning", "instruction", "reference"],
        "description": "Filter by knowledge kind."
      }
    }
  }
}
```

**Response schema:**
```json
{
  "results": [
    {
      "id": "01963f1a-7b2d-7000-8000-1a2b3c4d5e6f",
      "kind": "decision",
      "score": 0.82,
      "content": "...matched chunk content..."
    }
  ],
  "quality": {
    "confidence": "high",
    "suggestion": null
  }
}
```

Response design decisions:
- Return matched **chunks**, not full entries. Caps per-result payload at ~512 tokens.
- Exclude `tags`, `created_at`, and `tier` from results. Tags are write-time metadata. The LLM rarely needs timestamps.
- `id` is included so the LLM can reference entries in supersession.
- `kind` is included for context.
- Quality signal computed on reranker scores. Thresholds provisional for v1.0.

### corvia_write

Write a knowledge entry. Auto-detects near-duplicates and supersedes them.

```json
{
  "name": "corvia_write",
  "description": "Write a knowledge entry. If similar content already exists, the old entry is automatically superseded. Use 'supersedes' to explicitly replace entries with different content. Classify with 'kind' to aid future retrieval.",
  "inputSchema": {
    "type": "object",
    "required": ["content"],
    "properties": {
      "content": {
        "type": "string",
        "description": "Knowledge content (markdown)"
      },
      "kind": {
        "type": "string",
        "enum": ["decision", "learning", "instruction", "reference"],
        "default": "learning",
        "description": "Knowledge type. decision = choices with rationale. learning = insight, gotcha, pattern (default). instruction = how-to, setup, workflow. reference = code pattern, API doc, config."
      },
      "tags": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Freeform labels for categorization"
      },
      "supersedes": {
        "type": "array",
        "items": { "type": "string" },
        "description": "IDs of entries to explicitly replace. Use when the new content contradicts (not just updates) existing knowledge."
      }
    }
  }
}
```

**Response schema:**
```json
{
  "id": "01963f1a-7b2d-7000-8000-1a2b3c4d5e6f",
  "action": "created",
  "superseded": [],
  "warning": null
}
```

`action` is `"created"` (new entry) or `"superseded"` (auto-dedup triggered).
`superseded` lists IDs of entries that were superseded.
`warning` is set if `supersedes` references a non-existent ID.

### corvia_status

System health and entry statistics.

```json
{
  "name": "corvia_status",
  "description": "System status: entry counts, index health, storage location.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**Response schema:**
```json
{
  "entry_count": 142,
  "superseded_count": 23,
  "index_health": {
    "bm25_docs": 142,
    "vector_count": 380,
    "last_ingest": "2026-04-15T10:00:00Z",
    "stale": false
  },
  "storage_path": "/home/user/project/.corvia"
}
```

---

## 6. CLI Commands

| Command | Description | Notes |
|---------|-------------|-------|
| `corvia ingest [path]` | Index documents into knowledge store | `--fresh` for full rebuild. Auto-creates `.corvia/` on first run. |
| `corvia search <query>` | Hybrid search from terminal | `--limit`, `--kind`, `--max-tokens` flags. |
| `corvia write <content>` | Write a knowledge entry | `--kind`, `--tags`, `--supersedes` flags. Shows dedup result. |
| `corvia status` | Show system health | Entry counts, index health, staleness. |
| `corvia mcp` | Start stdio MCP server | Primary interface for AI agents. |

### First-Run / Cold Start Behavior

- `corvia ingest`: auto-creates `.corvia/entries/` and `.corvia/index/` if they don't exist.
  Downloads embedding model (nomic-embed-text-v1.5, ~274MB) and reranker model
  (ms-marco-MiniLM-L6-v2, ~80MB) on first run. Progress shown on stderr.
  Use `--model-path <dir>` for airgapped environments with pre-downloaded models.
- `corvia search` on empty index: returns zero results with
  `quality.suggestion = "No entries indexed. Run 'corvia ingest' first."`.
- `corvia mcp` with no index: MCP tools work but return appropriate empty/error responses.
  `corvia_search` returns the suggestion above. `corvia_write` works (creates entries).
  `corvia_status` shows `entry_count: 0, stale: true`.

### Malformed Entry Handling

During `corvia ingest`, if an entry file has invalid TOML frontmatter, missing `+++`
delimiters, or missing required `id` field:
- Skip the entry and continue ingesting.
- Log a warning to stderr.
- At the end, print summary: "Ingested 142 entries. Skipped 2 malformed: entry-xyz.md (missing id), entry-abc.md (invalid TOML)."

---

## 7. Crate Structure

```
repos/corvia2/
  Cargo.toml              # workspace root
  AGENTS.md
  .gitignore
  crates/
    corvia-core/           # storage, chunking, embedding, search, reranking
      src/
        lib.rs
        chunk.rs           # semantic sub-splitting, frontmatter stripping
        config.rs          # corvia.toml parsing
        embed.rs           # fastembed wrapper (embed + rerank)
        ingest.rs          # read entries -> chunk -> embed -> index
        search.rs          # hybrid BM25+vector, RRF fusion, rerank, quality signal
        store.rs           # flat file I/O + Redb index + chunk-to-entry mapping
    corvia-cli/            # CLI + MCP server
      src/
        main.rs            # clap CLI (5 commands)
        mcp.rs             # rmcp stdio server (3 tools)
```

**Dependencies:**
- tantivy 0.22 (BM25)
- hnsw_rs 0.3 (vector search, >=10K threshold only)
- fastembed 4 (embedding + cross-encoder reranking)
- rmcp 0.1 (stdio MCP server)
- redb 2 (index storage)
- uuid 1 (UUIDv7 generation)
- clap 4 (CLI)
- serde + serde_json + toml (serialization)
- tokio (async runtime)
- anyhow + thiserror (errors)
- tracing (logging)

**Dependency risk note:** hnsw_rs (0.3, ~100 stars) is the least mature dependency.
If it proves unstable, fallback options: qdrant-embedded or tantivy's vector search
feature. For v1.0, hnsw_rs is only used above the 10K vector threshold; below that,
brute-force cosine is used.

---

## 8. Testing Strategy

### Unit Tests (per module)

| Module | Tests |
|--------|-------|
| `chunk` | Split/merge, overlap, min/max token enforcement, frontmatter stripping, short entries (single chunk) |
| `embed` | Model loading, vector dimensions, deterministic output |
| `search` | BM25 scoring, vector cosine, RRF fusion math (k=30), reranker ordering, superseded filtering, kind pre-filtering, quality signal thresholds |
| `store` | Redb read/write, entry file serialization, TOML frontmatter parsing, round-trip fidelity (write then read back, assert match), UUIDv7 generation, chunk-to-entry mapping, atomic file write |
| `config` | TOML parsing, defaults, validation |

### Integration Tests (tests/ directory)

- Ingest a small test corpus (5-10 fixture entries)
- Search with known-answer queries, verify expected entries in top results
- Write an entry, verify it appears in subsequent search
- Supersession: write B superseding A, verify A no longer in results
- Auto-dedup: write near-duplicate, verify auto-supersession and response
- Supersession chains: A superseded by B, B superseded by C, verify only C in results
- Circular supersession: A supersedes B, B supersedes A, verify last-writer-wins
- Empty content: write with `content: ""`, verify graceful handling
- Long content: write >8K tokens, verify chunking produces multiple chunks and dedup still works
- Tags with TOML-special characters: verify round-trip serialization
- Cold start: search on empty index, verify quality signal suggestion
- Malformed entries: ingest with bad TOML, verify skip-and-warn
- Entry/index drift: add file manually, verify staleness detection on search
- Deletion: delete file, ingest, verify entry removed from results
- Supersession validation: supersede non-existent ID, verify warning in response
- Full pipeline: ingest -> search -> rerank -> verify ordering

### MCP E2E Tests

- Each of the 3 tools with valid input
- Each tool with invalid/edge-case input (empty query, missing content)
- Full lifecycle: write -> search for written content -> write superseding entry -> verify old excluded
- Large response: search returning entries with long content, verify truncation to max_tokens
- Cold start: search and status on empty store

### Test Fixtures

- `tests/fixtures/*.md` -- entry files covering each `kind`, various sizes
- Reusable test harness: creates temp `.corvia/`, ingests fixtures, runs assertions, cleans up
- Round-trip property tests: generate random frontmatter fields, write entry, read back, assert equality

---

## 9. What's NOT in V1.0

Explicitly deferred or permanently dropped:

| Feature | Status | Rationale |
|---------|--------|-----------|
| Tiered lifecycle (hot/warm/cold) | Deferred v1.1 | Corpus is small. Search everything. Add when scale justifies. |
| Graph (petgraph) | Dropped | Reranking covers accuracy needs. No multi-hop queries without graph. |
| Cedar policies | Dropped | Local-first, single user, no RBAC needed |
| Agent identity tracking | Dropped | Knowledge matters, not author. Git blame for provenance. |
| Bench CLI command | Deferred v1.1 | Integration tests cover quality. |
| Dashboard | Dropped | No server, no UI. Agents are the UI. |
| Inference server | Dropped | fastembed embedded in binary. |
| Contextual Retrieval | Unlikely | Reranking gets most of the benefit without LLM-at-ingest cost. |
| HyDE query expansion | Deferred v1.1 | Medium effort, selective improvement. |
| Multi-workspace | Deferred v1.2 | One corvia per product for v1.0. |
| V1 migration | Dropped | Fresh start. V1 and v2 are different architectures. |
| Delete CLI command | Deferred v1.1 | Delete file + ingest is sufficient for v1.0. |
| Multi-model indexing | Deferred v1.1 | Changing embedding model requires `ingest --fresh`. |
