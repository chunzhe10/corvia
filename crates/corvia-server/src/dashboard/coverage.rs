use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use corvia_kernel::knowledge_files;
use corvia_kernel::lite_store::LiteStore;
use corvia_kernel::traits::QueryableStore;

/// Snapshot of coverage state returned to callers.
#[derive(Debug, Clone)]
pub struct CoverageSnapshot {
    pub coverage: Option<f64>,
    pub stale: Option<bool>,
    pub disk_count: u64,
    pub store_count: u64,
    pub hnsw_count: u64,
    pub threshold: f64,
    pub checked_at: Option<String>,
}

impl From<CoverageSnapshot> for corvia_common::dashboard::CoverageResponse {
    fn from(s: CoverageSnapshot) -> Self {
        Self {
            index_coverage: s.coverage,
            index_stale: s.stale,
            index_disk_count: s.disk_count,
            index_store_count: s.store_count,
            index_hnsw_count: s.hnsw_count,
            index_stale_threshold: s.threshold,
            index_coverage_checked_at: s.checked_at,
        }
    }
}

/// Cached index coverage metrics with TTL-based refresh.
///
/// Uses `std::sync::Mutex` with brief lock holds (never across `.await`).
/// The `get_cached` method is fully synchronous for use in axum handlers
/// (avoids complex handler futures with dual axum 0.7/0.8).
/// Recomputation happens via `refresh()` (async) or at startup.
pub struct IndexCoverageCache {
    inner: Mutex<CacheInner>,
    ttl: Duration,
    threshold: f64,
}

struct CacheInner {
    snapshot: CoverageSnapshot,
    last_computed: Option<Instant>,
}

impl IndexCoverageCache {
    pub fn new(threshold: f64, ttl_secs: u64) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                snapshot: CoverageSnapshot {
                    coverage: None,
                    stale: None,
                    disk_count: 0,
                    store_count: 0,
                    hnsw_count: 0,
                    threshold,
                    checked_at: None,
                },
                last_computed: None,
            }),
            ttl: Duration::from_secs(ttl_secs),
            threshold,
        }
    }

    /// Return the cached snapshot. If TTL has expired, returns stale data and
    /// marks the cache as needing refresh (callers can trigger `refresh()`
    /// separately). This is synchronous and safe to call from axum handlers.
    pub fn get_cached(&self) -> CoverageSnapshot {
        let inner = self.inner.lock().unwrap_or_else(|e| {
            warn!("coverage cache mutex poisoned, recovering");
            e.into_inner()
        });
        debug!(
            disk = inner.snapshot.disk_count,
            hnsw = inner.snapshot.hnsw_count,
            fresh = inner.last_computed.is_some_and(|t| t.elapsed() < self.ttl),
            "index coverage cache read"
        );
        inner.snapshot.clone()
    }

    /// Returns true if the cache needs refreshing (TTL expired or never computed).
    pub fn needs_refresh(&self) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|e| {
            warn!("coverage cache mutex poisoned, recovering");
            e.into_inner()
        });
        inner.last_computed.is_none_or(|t| t.elapsed() >= self.ttl)
    }

    /// Recompute coverage from disk, store, and HNSW. Async because it uses
    /// `spawn_blocking` for disk I/O and `store.count().await`.
    pub async fn refresh(
        &self,
        data_dir: &Path,
        scope_id: &str,
        store: &dyn QueryableStore,
    ) -> CoverageSnapshot {
        let start = Instant::now();

        // Disk count via spawn_blocking (avoid blocking async runtime)
        let scope_dir = knowledge_files::scope_dir(data_dir, scope_id);
        let disk_count = tokio::task::spawn_blocking(move || count_json_files(&scope_dir))
            .await
            .unwrap_or_else(|e| {
                warn!("spawn_blocking for disk count failed: {e}");
                0
            });

        // Store count (Redb SCOPE_INDEX)
        let store_count = store.count(scope_id).await.unwrap_or_else(|e| {
            warn!("store.count failed: {e}");
            0
        });

        // HNSW count (Redb HNSW_TO_UUID). This is a sync Redb metadata read
        // (table.len()), not a full scan — fast enough to not need spawn_blocking.
        let hnsw_count = if let Some(lite) = store.as_any().downcast_ref::<LiteStore>() {
            lite.hnsw_entry_count().unwrap_or_else(|e| {
                warn!("hnsw_entry_count failed: {e}");
                0
            })
        } else {
            // PostgresStore: pgvector manages its own index, use store_count
            store_count
        };

        // Compute coverage
        let (coverage, stale) = if disk_count == 0 {
            (None, None)
        } else {
            let ratio = (hnsw_count as f64 / disk_count as f64).min(1.0);
            (Some(ratio), Some(ratio < self.threshold))
        };

        // Orphan detection
        if hnsw_count > disk_count || store_count > disk_count {
            warn!(
                disk = disk_count,
                store = store_count,
                hnsw = hnsw_count,
                "index has more entries than knowledge files on disk (orphaned entries)"
            );
        }

        let checked_at = chrono::Utc::now().to_rfc3339();
        let elapsed_ms = start.elapsed().as_millis();

        info!(
            disk = disk_count,
            store = store_count,
            hnsw = hnsw_count,
            ?coverage,
            ?stale,
            compute_ms = elapsed_ms,
            "index coverage recomputed"
        );

        let snapshot = CoverageSnapshot {
            coverage,
            stale,
            disk_count,
            store_count,
            hnsw_count,
            threshold: self.threshold,
            checked_at: Some(checked_at),
        };

        // Store under lock (brief hold, no await)
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| {
            warn!("coverage cache mutex poisoned, recovering");
            e.into_inner()
        });
            inner.snapshot = snapshot.clone();
            inner.last_computed = Some(Instant::now());
        }

        snapshot
    }
}

/// Count `.json` files in a directory (non-recursive, flat layout).
/// Returns 0 if the directory does not exist or is unreadable.
/// Exposed for testing.
#[cfg(test)]
pub(crate) fn test_count_json_files(dir: &Path) -> u64 {
    count_json_files(dir)
}

fn count_json_files(dir: &Path) -> u64 {
    if !dir.exists() {
        return 0;
    }
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str()) == Some("json")
            })
            .count() as u64,
        Err(e) => {
            warn!("Failed to read knowledge dir {}: {e}", dir.display());
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_kernel::lite_store::LiteStore;
    use corvia_kernel::traits::QueryableStore;
    use tempfile::tempdir;

    fn write_dummy_json_files(data_dir: &Path, scope_id: &str, count: u64) {
        let dir = data_dir.join("knowledge").join(scope_id);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..count {
            std::fs::write(
                dir.join(format!("dummy-{i}.json")),
                format!(r#"{{"dummy":{i}}}"#),
            )
            .unwrap();
        }
    }

    fn make_store(dir: &Path) -> LiteStore {
        LiteStore::open(dir, 3).unwrap()
    }

    async fn insert_entries(store: &LiteStore, scope_id: &str, count: u64) {
        for _ in 0..count {
            let mut entry = corvia_common::types::KnowledgeEntry::new(
                "test content".into(),
                scope_id.into(),
                "test.rs".into(),
            );
            entry.embedding = Some(vec![0.1, 0.2, 0.3]);
            store.insert(&entry).await.unwrap();
        }
    }

    #[test]
    fn test_count_json_files_no_dir() {
        let tmp = tempdir().unwrap();
        let missing = tmp.path().join("nonexistent");
        assert_eq!(count_json_files(&missing), 0);
    }

    #[test]
    fn test_count_json_files_empty_dir() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(count_json_files(&dir), 0);
    }

    #[test]
    fn test_count_json_files_mixed() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("mixed");
        std::fs::create_dir_all(&dir).unwrap();
        // 3 json files
        for i in 0..3 {
            std::fs::write(dir.join(format!("{i}.json")), "{}").unwrap();
        }
        // 2 non-json files
        std::fs::write(dir.join("readme.txt"), "hi").unwrap();
        std::fs::write(dir.join("data.csv"), "a,b").unwrap();
        assert_eq!(count_json_files(&dir), 3);
    }

    #[tokio::test]
    async fn test_fresh_workspace_no_files() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), "test", &store).await;

        assert!(snap.coverage.is_none());
        assert!(snap.stale.is_none());
        assert_eq!(snap.disk_count, 0);
        assert_eq!(snap.store_count, 0);
        assert_eq!(snap.hnsw_count, 0);
    }

    #[tokio::test]
    async fn test_knowledge_dir_missing() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();
        // Don't create knowledge/ dir at all
        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), "missing_scope", &store).await;

        assert!(snap.coverage.is_none());
        assert!(snap.stale.is_none());
        assert_eq!(snap.disk_count, 0);
    }

    #[tokio::test]
    async fn test_full_coverage() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let scope = "full";
        // insert_entries creates both store entries AND knowledge JSON files
        insert_entries(&store, scope, 10).await;

        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), scope, &store).await;

        assert_eq!(snap.coverage, Some(1.0));
        assert_eq!(snap.stale, Some(false));
        assert_eq!(snap.disk_count, 10);
        assert_eq!(snap.store_count, 10);
        assert_eq!(snap.hnsw_count, 10);
    }

    #[tokio::test]
    async fn test_partial_coverage() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let scope = "partial";
        // 7 fully indexed entries (creates 7 JSON files on disk)
        insert_entries(&store, scope, 7).await;
        // 3 extra JSON files on disk (simulating partial ingest that didn't embed)
        write_dummy_json_files(tmp.path(), scope, 3);

        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), scope, &store).await;

        assert_eq!(snap.coverage, Some(0.7));
        assert_eq!(snap.stale, Some(true));
        assert_eq!(snap.disk_count, 10);
        assert_eq!(snap.store_count, 7);
        assert_eq!(snap.hnsw_count, 7);
    }

    #[tokio::test]
    async fn test_threshold_boundary_exact() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let scope = "boundary";
        // 9 indexed + 1 unindexed = coverage 9/10 = 0.9
        insert_entries(&store, scope, 9).await;
        write_dummy_json_files(tmp.path(), scope, 1);

        // threshold=0.9, coverage=0.9 → NOT stale (< not <=)
        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), scope, &store).await;

        assert_eq!(snap.coverage, Some(0.9));
        assert_eq!(snap.stale, Some(false));
    }

    #[tokio::test]
    async fn test_threshold_boundary_below() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let scope = "below";
        // 89 indexed + 11 unindexed = coverage 89/100 = 0.89
        insert_entries(&store, scope, 89).await;
        write_dummy_json_files(tmp.path(), scope, 11);

        // threshold=0.9, coverage=0.89 → stale
        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), scope, &store).await;

        assert_eq!(snap.coverage, Some(0.89));
        assert_eq!(snap.stale, Some(true));
    }

    #[tokio::test]
    async fn test_hnsw_gt_disk_orphaned() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let scope = "orphan";
        // 10 indexed entries create 10 JSON files
        insert_entries(&store, scope, 10).await;
        // Delete 5 JSON files from disk to simulate orphaned index entries
        let dir = tmp.path().join("knowledge").join(scope);
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            if count >= 5 { break; }
            std::fs::remove_file(entry.unwrap().path()).unwrap();
            count += 1;
        }

        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.refresh(tmp.path(), scope, &store).await;

        // Coverage clamped to 1.0
        assert_eq!(snap.coverage, Some(1.0));
        assert_eq!(snap.stale, Some(false));
        // But raw counts reveal the orphans
        assert_eq!(snap.disk_count, 5);
        assert_eq!(snap.hnsw_count, 10);
    }

    #[tokio::test]
    async fn test_get_cached_returns_default_before_refresh() {
        let cache = IndexCoverageCache::new(0.9, 60);
        let snap = cache.get_cached();

        assert!(snap.coverage.is_none());
        assert!(snap.stale.is_none());
        assert_eq!(snap.disk_count, 0);
    }

    #[tokio::test]
    async fn test_needs_refresh_before_first_compute() {
        let cache = IndexCoverageCache::new(0.9, 60);
        assert!(cache.needs_refresh());
    }

    #[tokio::test]
    async fn test_needs_refresh_after_compute() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        let cache = IndexCoverageCache::new(0.9, 60);
        cache.refresh(tmp.path(), "test", &store).await;

        assert!(!cache.needs_refresh());
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let tmp = tempdir().unwrap();
        let store = std::sync::Arc::new(make_store(tmp.path()));
        store.init_schema().await.unwrap();

        write_dummy_json_files(tmp.path(), "conc", 5);

        let cache = std::sync::Arc::new(IndexCoverageCache::new(0.9, 60));
        let mut handles = Vec::new();

        for _ in 0..10 {
            let c = cache.clone();
            let s = store.clone();
            let d = tmp.path().to_path_buf();
            handles.push(tokio::spawn(async move {
                c.refresh(&d, "conc", &*s).await;
                c.get_cached();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let snap = cache.get_cached();
        assert_eq!(snap.disk_count, 5);
        assert!(snap.coverage.is_some(), "coverage should be computed after concurrent refreshes");
        assert!(snap.checked_at.is_some(), "checked_at should be set");
    }

    #[tokio::test]
    async fn test_needs_refresh_after_ttl_expires() {
        let tmp = tempdir().unwrap();
        let store = make_store(tmp.path());
        store.init_schema().await.unwrap();

        // TTL of 5s (minimum). After refresh, needs_refresh should be false,
        // then true after we manually expire via a 0s TTL cache.
        let cache = IndexCoverageCache::new(0.9, 0); // will be clamped to 5 at config level, but raw 0 works here
        cache.refresh(tmp.path(), "test", &store).await;
        // Immediately after refresh with 0s TTL, needs_refresh is true
        assert!(cache.needs_refresh());
    }

    #[test]
    fn test_count_json_files_ignores_subdirectories() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("scope");
        std::fs::create_dir_all(&dir).unwrap();
        // 2 json files at top level
        std::fs::write(dir.join("a.json"), "{}").unwrap();
        std::fs::write(dir.join("b.json"), "{}").unwrap();
        // 3 json files in a subdirectory (should NOT be counted)
        let sub = dir.join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("c.json"), "{}").unwrap();
        std::fs::write(sub.join("d.json"), "{}").unwrap();
        std::fs::write(sub.join("e.json"), "{}").unwrap();
        assert_eq!(count_json_files(&dir), 2);
    }

    #[test]
    fn test_checked_at_is_valid_rfc3339() {
        // Verify CoverageSnapshot.checked_at is valid RFC 3339 after refresh
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = tempdir().unwrap();
            let store = make_store(tmp.path());
            store.init_schema().await.unwrap();

            let cache = IndexCoverageCache::new(0.9, 60);
            let snap = cache.refresh(tmp.path(), "test", &store).await;
            let ts = snap.checked_at.expect("checked_at should be set after refresh");
            assert!(
                chrono::DateTime::parse_from_rfc3339(&ts).is_ok(),
                "checked_at should be valid RFC 3339: {ts}"
            );
        });
    }
}
