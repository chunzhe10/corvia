# ArcSwap HNSW Migration — Design

**Issue:** #17
**Date:** 2026-03-27
**Status:** Approved (issue prompt is the spec)

## Problem

LiteStore's HNSW index uses `Arc<Mutex<Hnsw>>`. At 100K entries, rebuild takes
30-90s. All searches block during rebuild because they contend on the same mutex.

## Solution

Replace `Arc<Mutex<Hnsw>>` with `Arc<ArcSwap<Arc<Hnsw>>>` using the `arc-swap` crate.

- **Search**: `load()` — wait-free pointer read, zero contention
- **Insert**: `load()` + `insert()` — hnsw_rs uses internal RwLock, thread-safe
- **Rebuild**: Build new Hnsw in background, `store(Arc::new(new_hnsw))` for atomic swap
- **Flush**: `load()` + `file_dump()` — snapshot current index for persistence
- **Drop**: Old index deallocated via `tokio::task::spawn_blocking(move || drop(old))`

## Type Change

```rust
// Before
hnsw: Arc<Mutex<Hnsw<'static, f32, DistCosine>>>

// After
hnsw: Arc<ArcSwap<Arc<Hnsw<'static, f32, DistCosine>>>>
```

The outer `Arc` enables sharing across `&self` methods. `ArcSwap` enables atomic
pointer swap. The inner `Arc` enables the old index to be moved to `spawn_blocking`
for deferred deallocation.

## Lock Site Migration (4 sites)

| Site | Before | After |
|------|--------|-------|
| `index_entry()` L272 | `.lock().unwrap(); hnsw.insert()` | `.load(); hnsw.insert()` |
| `search()` L633 | `.lock().unwrap(); hnsw.search()` | `.load(); hnsw.search()` |
| `flush_hnsw()` L348 | `.lock().unwrap(); hnsw.file_dump()` | `.load(); hnsw.file_dump()` |
| `rebuild_from_files()` L443 | `.lock().unwrap(); *hnsw = new` | Build new, `.store(Arc::new(new))` |

## Unsafe Transmute

The existing `unsafe { std::mem::transmute(loaded) }` for lifetime erasure in
`open()` is preserved identically. The transmute happens on the bare `Hnsw` value
before wrapping in `Arc<ArcSwap<Arc<...>>>`.

## Blue-Green Rebuild

```
1. Build new Hnsw (MAX_NB_CONNECTION, MAX_ELEMENTS, etc.)
2. Insert all entries into new index (from knowledge files)
3. old = self.hnsw.swap(Arc::new(new_hnsw))
4. tokio::task::spawn_blocking(move || drop(old))
```

During step 1-2, searches continue using the old index via `load()`.
Step 3 is an atomic pointer swap — searches after this use the new index.
Step 4 deallocates the old index off the async runtime.

## Files Modified

- `crates/corvia-kernel/Cargo.toml` — add `arc-swap = "1"`
- `crates/corvia-kernel/src/lite_store.rs` — all HNSW access (4 sites + struct def + open)

## Acceptance Criteria

From issue #17 — all preserved as-is.
