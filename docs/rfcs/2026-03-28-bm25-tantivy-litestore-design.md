# BM25 Hybrid Search Phase 2a: Tantivy Integration (LiteStore)

**Issue:** #39
**Date:** 2026-03-28
**Status:** Approved
**Depends on:** Phase 1 (Composable Pipeline) -- PR #55, merged

## Summary

Add full-text BM25 search to LiteStore via embedded tantivy. Implement a
`FullTextSearchable` trait with code-aware tokenization, batched commits,
and staleness detection. Register a `Bm25Searcher` in the pipeline registry
so it can run alongside the existing `VectorSearcher`.

## Architecture

```
LiteStore
  +-- hnsw_rs (vector search)        [existing]
  +-- Redb (metadata, indexes)       [existing]
  +-- petgraph (graph)               [existing]
  +-- TantivyIndex (BM25 search)     [NEW]
       +-- .corvia/cache/tantivy/    [cache, rebuildable]
       +-- code-aware tokenizer
       +-- batched commit (100 entries / 500ms)
       +-- write-generation staleness detection
```

## Design Decisions

### D1: FullTextSearchable trait (traits.rs)

New async trait alongside existing kernel traits:

```rust
#[async_trait]
pub trait FullTextSearchable: Send + Sync {
    async fn search_text(&self, query: &str, scope_id: &str, limit: usize) -> Result<Vec<TextSearchResult>>;
    async fn index_entry(&self, entry: &KnowledgeEntry) -> Result<()>;
    async fn remove_entry(&self, entry_id: &uuid::Uuid) -> Result<()>;
    async fn flush(&self) -> Result<()>;
    async fn rebuild_from_store(&self, entries: &[KnowledgeEntry]) -> Result<usize>;
    async fn entry_count(&self) -> Result<u64>;
}
```

Algorithm-agnostic naming (`search_text` not `search_bm25`). Default no-op
for `flush()` so stores that don't need it can skip.

### D2: TextSearchResult type

```rust
pub struct TextSearchResult {
    pub entry_id: uuid::Uuid,
    pub score: f32,
}
```

Lightweight. The Bm25Searcher looks up full KnowledgeEntry from the store
using the entry_id. This keeps the FTS layer decoupled from full entry storage.

### D3: TantivyIndex implementation

**Schema fields:**
- `content` (TEXT) -- the knowledge entry content, tokenized with code-aware tokenizer
- `entry_id` (STRING, stored, indexed) -- UUID for result lookup
- `scope_id` (STRING, indexed) -- for scope filtering in queries
- `content_role` (STRING, indexed) -- optional metadata filter
- `source_origin` (STRING, indexed) -- optional metadata filter

**Storage:** `.corvia/cache/tantivy/` -- this is a cache, not truth. Can be
fully rebuilt from knowledge JSON files.

### D4: Code-aware tokenizer

Custom tantivy tokenizer registered as `"corvia_code"`:

| Input | Tokens |
|-------|--------|
| `snake_case` | `["snake_case", "snake", "case"]` |
| `CamelCase` | `["camelcase", "camel", "case"]` |
| `EF_SEARCH` | `["ef_search", "ef", "search"]` |
| `corvia_search` | `["corvia_search", "corvia", "search"]` |
| `retriever.rs` | `["retriever.rs", "retriever", "rs"]` |
| `path/to/file.rs` | `["path/to/file.rs", "path", "to", "file.rs", "file", "rs"]` |
| `v0.4.5` | `["v0.4.5", "0.4.5", "v0", "4", "5"]` |
| `550e8400-e29b-...` | UUID split on `-` |

All tokens lowercased. Originals preserved for exact match.

### D5: Batched commit strategy

- `index_entry()` calls `index_writer.add_document()` (buffered, no fsync)
- Auto-flush triggered by:
  - 100 buffered entries (counter), OR
  - 500ms since last flush (tokio interval timer)
- `flush()` calls `index_writer.commit()` then `reader.reload()`
- `rebuild_from_store()` does a single commit at the end
- Progress logging every 10K entries during rebuild

### D6: Write-generation staleness detection

- Monotonic `fts_write_generation` counter in Redb META table
- Incremented on every `insert()` and `delete_scope()` that touches tantivy
- `last_synced_generation` stored in a small JSON file alongside the tantivy index
- On startup: compare the two. Mismatch triggers rebuild.
- After rebuild: update `last_synced_generation` to current `fts_write_generation`

### D7: First-run bootstrap

- Missing `.corvia/cache/tantivy/` directory + BM25 in config -> synchronous rebuild
- `< 50K entries`: synchronous (expected < 10s)
- `>= 50K entries`: async background rebuild, BM25 returns empty until done
- Progress logging every 10K entries

### D8: LiteStore integration

```rust
pub struct LiteStore {
    // ... existing fields ...
    tantivy: Option<TantivyIndex>,  // None when BM25 not configured
}
```

- `insert()` mirrors to `tantivy.index_entry()` when present
- `delete_scope()` mirrors scope removal to tantivy
- `rebuild_from_files()` rebuilds tantivy in parallel via `tokio::join!`
- Zero overhead when BM25 disabled (Option is None)

### D9: Bm25Searcher pipeline integration

```rust
pub struct Bm25Searcher {
    store: Arc<dyn QueryableStore>,  // for entry lookup by ID
    fts: Arc<dyn FullTextSearchable>,
}
```

- Implements `Searcher` trait
- `needs_embedding() -> false` (BM25 uses raw query text)
- Registered as `"bm25"` in PipelineRegistry
- Uses `SearchContext.query` for text search
- Looks up full entries from store, applies tier weighting, normalizes scores

### D10: Dependency

```toml
tantivy = "0.22"  # Pure Rust, no C deps
```

If incremental compile time exceeds 45s, add a `bm25` feature flag.

## Files Changed

| File | Change |
|------|--------|
| `crates/corvia-kernel/Cargo.toml` | Add `tantivy = "0.22"` |
| `crates/corvia-kernel/src/traits.rs` | Add `FullTextSearchable` trait + `TextSearchResult` |
| `crates/corvia-kernel/src/tantivy_index.rs` | New: TantivyIndex struct, code-aware tokenizer |
| `crates/corvia-kernel/src/pipeline/searcher.rs` | Add `Bm25Searcher` |
| `crates/corvia-kernel/src/pipeline/registry.rs` | Register `"bm25"` searcher |
| `crates/corvia-kernel/src/lite_store.rs` | Add `Option<TantivyIndex>`, mirror insert/delete |
| `crates/corvia-kernel/src/lib.rs` | Wire TantivyIndex into LiteStore construction, update `build_pipeline_retriever` |
| `crates/corvia-kernel/src/ops.rs` | Update `corvia rebuild` for dual-index |

## Non-Goals

- RRF fusion (Phase 2c)
- PostgresStore tsvector integration (Phase 2b)
- Runtime A/B benchmarking (Phase 3)

## Risks

- **Compile time:** tantivy is a large crate. Mitigated by measuring and adding feature flag if needed.
- **Memory:** ~20-50MB at 10K entries. Lazy-load when BM25 disabled = zero cost for vector-only.
