# ArcSwap HNSW Migration — Design

**Issue:** #17
**Date:** 2026-03-27
**Status:** Approved (issue prompt is the spec)

## Problem

LiteStore's HNSW index uses `Arc<Mutex<Hnsw>>`. At 100K entries, rebuild takes
30-90s. All searches block during rebuild because they contend on the same mutex.

## Solution

Replace `Arc<Mutex<Hnsw>>` with `Arc<ArcSwap<Hnsw>>` using the `arc-swap` crate.
(`ArcSwap<T>` stores `Arc<T>` internally via `from_pointee`.)

- **Search**: `load()` — wait-free pointer read, zero contention
- **Insert**: `load()` + `insert()` — hnsw_rs uses internal RwLock, thread-safe
- **Rebuild**: Build new Hnsw fully, then `swap(Arc::new(new_hnsw))` for atomic swap
- **Flush**: `load_full()` + `file_dump()` — snapshot current index for persistence
- **Drop**: Old index deallocated via `std::thread::spawn(move || drop(old))`

## Type Change

```rust
// Before
hnsw: Arc<Mutex<Hnsw<'static, f32, DistCosine>>>

// After
hnsw: Arc<ArcSwap<Hnsw<'static, f32, DistCosine>>>
// ArcSwap<T> stores Arc<T> internally — no double-Arc needed
```

The outer `Arc` enables sharing across `&self` methods. `ArcSwap` enables atomic
pointer swap. `swap()` returns the old `Arc<T>` for deferred deallocation.

## Lock Site Migration (4 sites + rebuild refactor)

| Site | Before | After |
|------|--------|-------|
| `index_entry()` | `.lock().unwrap(); hnsw.insert()` | `.load(); hnsw.insert()` |
| `search()` | `.lock().unwrap(); hnsw.search()` | `.load(); hnsw.search()` |
| `flush_hnsw()` | `.lock().unwrap(); hnsw.file_dump()` | `.load_full(); hnsw.file_dump()` |
| `rebuild_from_files()` | `.lock().unwrap(); *hnsw = new` | Build full → `.swap(Arc::new(new))` |

`flush_hnsw` uses `load_full()` (clones Arc, releases slot) instead of `load()`
to avoid pinning the ArcSwap slot during potentially long I/O.

## Unsafe Transmute

The existing `unsafe { std::mem::transmute(loaded) }` for lifetime erasure in
`open()` is preserved identically. The transmute happens on the bare `Hnsw` value
before wrapping in `ArcSwap`. Safety argument is unchanged: without mmap, all
HNSW data is owned `Vec<T>`, making the lifetime a phantom artifact.

## Blue-Green Rebuild

```
1. Build new Hnsw (MAX_NB_CONNECTION, MAX_ELEMENTS, etc.)
2. Insert all entries into new_hnsw directly (via index_entry_into)
3. old = self.hnsw.swap(Arc::new(new_hnsw))  // atomic pointer swap
4. std::thread::spawn(move || drop(old))      // deferred deallocation
```

During steps 1-2, searches continue using the old index via `load()`.
Step 3 is an atomic pointer swap — searches after this use the new index.
Step 4 deallocates the old index on a background thread.

## Files Modified

- `crates/corvia-kernel/Cargo.toml` — add `arc-swap = "1"`
- `crates/corvia-kernel/src/lite_store.rs` — all HNSW access (4 sites + struct def + open + rebuild refactor)

## Acceptance Criteria

From issue #17 — all preserved as-is.
