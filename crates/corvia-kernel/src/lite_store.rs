use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{KnowledgeEntry, SearchResult};
use hnsw_rs::prelude::*;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::knowledge_files;
use crate::traits::QueryableStore;

// Redb table definitions
const ENTRIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");
const SCOPE_INDEX: TableDefinition<&str, &str> = TableDefinition::new("scope_index");
const HNSW_TO_UUID: TableDefinition<u64, &str> = TableDefinition::new("hnsw_to_uuid");
const UUID_TO_HNSW: TableDefinition<&str, u64> = TableDefinition::new("uuid_to_hnsw");
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

/// Temporal index: compound key (scope_id, valid_from_millis, entry_id) → (valid_to_millis, recorded_at_millis)
/// Enables O(log n) range scans for bi-temporal queries via Redb B-tree.
const TEMPORAL_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("temporal_index");

/// Secondary index: "{scope_id}:{source_version}" → UUID string.
/// Enables O(log n) lookup of entries by source_version within a scope.
/// Used for cross-batch graph edge resolution (e.g., spawned_by parent lookup).
const SOURCE_VERSION_INDEX: TableDefinition<&str, &str> =
    TableDefinition::new("source_version_index");

/// HNSW tuning constants
const MAX_NB_CONNECTION: usize = 16;
const MAX_LAYER: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 64;
const MAX_ELEMENTS: usize = 100_000;

/// LiteStore implements `QueryableStore` using hnsw_rs + Redb + knowledge files.
/// Zero Docker required. Suitable for single-machine workloads up to ~100K entries.
pub struct LiteStore {
    data_dir: PathBuf,
    db: Arc<Database>,
    hnsw: Arc<Mutex<Hnsw<'static, f32, DistCosine>>>,
    next_hnsw_id: AtomicU64,
    dimensions: usize,
    graph: crate::graph_store::LiteGraphStore,
}

impl LiteStore {
    /// Create or open a LiteStore at the given directory.
    /// Creates Redb database and a fresh HNSW index.
    /// Call `rebuild_from_files()` after open to restore state from knowledge JSONs.
    pub fn open(data_dir: &Path, dimensions: usize) -> Result<Self> {
        // Ensure data directory exists
        std::fs::create_dir_all(data_dir)
            .map_err(|e| CorviaError::Storage(format!("Failed to create data dir: {e}")))?;

        let db_path = data_dir.join("lite_store.redb");
        let db = Arc::new(Database::create(&db_path)
            .map_err(|e| CorviaError::Storage(format!("Failed to open Redb: {e}")))?
        );

        // Read persisted next_hnsw_id from meta table, default to 0
        let next_id = {
            let read_txn = db
                .begin_read()
                .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
            match read_txn.open_table(META) {
                Ok(table) => match table.get("next_hnsw_id") {
                    Ok(Some(val)) => val.value(),
                    _ => 0,
                },
                Err(_) => 0, // Table doesn't exist yet
            }
        };

        // Try loading persisted HNSW from disk first (fast path)
        let hnsw_dir = data_dir.join("hnsw");
        let hnsw_graph_file = hnsw_dir.join("litestore.hnsw.graph");
        let hnsw_data_file = hnsw_dir.join("litestore.hnsw.data");

        let (hnsw, loaded_from_disk) = if next_id > 0
            && hnsw_graph_file.exists()
            && hnsw_data_file.exists()
        {
            let start = std::time::Instant::now();
            let mut hnswio = HnswIo::new(&hnsw_dir, "litestore");
            match hnswio.load_hnsw::<f32, DistCosine>() {
                Ok(loaded) => {
                    // SAFETY: Without mmap (default ReloadOptions), all point data is
                    // deserialized into owned Vec<T>, not borrowed from HnswIo.
                    // The lifetime parameter is a compile-time artifact that doesn't
                    // reflect actual borrowing in the non-mmap case.
                    let loaded: Hnsw<'static, f32, DistCosine> =
                        unsafe { std::mem::transmute(loaded) };
                    let elapsed = start.elapsed();
                    info!(
                        "HNSW loaded from disk in {:.2}s ({} entries)",
                        elapsed.as_secs_f64(),
                        next_id
                    );
                    (loaded, true)
                }
                Err(e) => {
                    warn!("Failed to load HNSW from disk, will rebuild: {e}");
                    let fresh = Hnsw::<f32, DistCosine>::new(
                        MAX_NB_CONNECTION,
                        MAX_ELEMENTS,
                        MAX_LAYER,
                        EF_CONSTRUCTION,
                        DistCosine {},
                    );
                    (fresh, false)
                }
            }
        } else {
            let fresh = Hnsw::<f32, DistCosine>::new(
                MAX_NB_CONNECTION,
                MAX_ELEMENTS,
                MAX_LAYER,
                EF_CONSTRUCTION,
                DistCosine {},
            );
            (fresh, false)
        };

        info!(
            "LiteStore opened at {} (dimensions={}, next_hnsw_id={}, hnsw_from_disk={})",
            data_dir.display(),
            dimensions,
            next_id,
            loaded_from_disk
        );

        let graph = crate::graph_store::LiteGraphStore::new(Arc::clone(&db))?;

        let store = Self {
            data_dir: data_dir.to_path_buf(),
            db,
            hnsw: Arc::new(Mutex::new(hnsw)),
            next_hnsw_id: AtomicU64::new(next_id),
            dimensions,
            graph,
        };

        // If HNSW was loaded from disk, rebuild only the temporal index (cheap, from Redb).
        // If not loaded, fall back to full rebuild from knowledge files (slow).
        if next_id > 0 && !loaded_from_disk {
            let start = std::time::Instant::now();
            let rebuilt = store.rebuild_from_files()?;
            let elapsed = start.elapsed();
            if rebuilt > 0 {
                info!(
                    "Rebuilt HNSW index from {} knowledge files in {:.2}s",
                    rebuilt,
                    elapsed.as_secs_f64()
                );
                // Persist the rebuilt HNSW so next startup is fast
                if let Err(e) = store.flush_hnsw() {
                    warn!("Failed to persist rebuilt HNSW index: {e}");
                }
            }
        }

        Ok(store)
    }

    /// Fetch all knowledge entries from the Redb ENTRIES table in a single pass.
    /// Used for migration export — avoids re-reading JSON files from disk.
    pub fn fetch_all_entries(&self) -> Result<Vec<KnowledgeEntry>> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let entries_table = match read_txn.open_table(ENTRIES) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut entries = Vec::new();
        for item in entries_table
            .iter()
            .map_err(|e| CorviaError::Storage(format!("Failed to iterate ENTRIES: {e}")))?
        {
            let (_key, value) = item
                .map_err(|e| CorviaError::Storage(format!("Failed to read entry: {e}")))?;
            match serde_json::from_slice::<KnowledgeEntry>(value.value()) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("Skipping malformed entry in Redb: {e}");
                }
            }
        }

        Ok(entries)
    }

    /// Access the underlying graph store (for migration export).
    pub fn graph(&self) -> &crate::graph_store::LiteGraphStore {
        &self.graph
    }

    /// Allocate the next HNSW data ID (atomic counter).
    fn allocate_hnsw_id(&self) -> u64 {
        self.next_hnsw_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Persist the current next_hnsw_id counter to Redb.
    fn persist_next_id(&self) -> Result<()> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn
                .open_table(META)
                .map_err(|e| CorviaError::Storage(format!("Failed to open META table: {e}")))?;
            let val = self.next_hnsw_id.load(Ordering::SeqCst);
            table
                .insert("next_hnsw_id", val)
                .map_err(|e| CorviaError::Storage(format!("Failed to write next_hnsw_id: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit next_id: {e}")))?;
        Ok(())
    }

    /// Write a temporal index entry to Redb for bi-temporal range queries.
    fn write_temporal_index(&self, entry: &KnowledgeEntry) -> Result<()> {
        let valid_from_millis = entry.valid_from.timestamp_millis();
        let recorded_at_millis = entry.recorded_at.timestamp_millis();
        let valid_to_millis = entry.valid_to.map(|t| t.timestamp_millis());

        let key = format!(
            "{}:{:020}:{}",
            entry.scope_id, valid_from_millis, entry.id
        );
        let value = serde_json::to_vec(&serde_json::json!({
            "valid_to_millis": valid_to_millis,
            "recorded_at_millis": recorded_at_millis,
        }))
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize temporal value: {e}")))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn
                .open_table(TEMPORAL_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;
            table
                .insert(key.as_str(), value.as_slice())
                .map_err(|e| CorviaError::Storage(format!("Failed to insert temporal index: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit temporal index: {e}")))?;
        Ok(())
    }

    /// Index an entry into HNSW + Redb (does NOT write the knowledge JSON file).
    fn index_entry(&self, entry: &KnowledgeEntry, embedding: &[f32]) -> Result<()> {
        let uuid_str = entry.id.to_string();
        let hnsw_id = self.allocate_hnsw_id();

        // Insert into HNSW
        {
            let hnsw = self
                .hnsw
                .lock()
                .map_err(|e| CorviaError::Storage(format!("HNSW lock poisoned: {e}")))?;
            hnsw.insert((embedding, hnsw_id as usize));
        }

        // Write metadata to Redb
        let entry_json = serde_json::to_vec(entry)
            .map_err(|e| CorviaError::Storage(format!("Failed to serialize entry: {e}")))?;

        let scope_key = format!("{}:{}", entry.scope_id, uuid_str);

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let mut entries_table = write_txn
                .open_table(ENTRIES)
                .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;
            entries_table
                .insert(uuid_str.as_str(), entry_json.as_slice())
                .map_err(|e| CorviaError::Storage(format!("Failed to insert entry: {e}")))?;

            let mut scope_table = write_txn
                .open_table(SCOPE_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to open SCOPE_INDEX: {e}")))?;
            scope_table
                .insert(scope_key.as_str(), uuid_str.as_str())
                .map_err(|e| CorviaError::Storage(format!("Failed to insert scope index: {e}")))?;

            let mut h2u_table = write_txn
                .open_table(HNSW_TO_UUID)
                .map_err(|e| CorviaError::Storage(format!("Failed to open HNSW_TO_UUID: {e}")))?;
            h2u_table
                .insert(hnsw_id, uuid_str.as_str())
                .map_err(|e| CorviaError::Storage(format!("Failed to insert hnsw_to_uuid: {e}")))?;

            let mut u2h_table = write_txn
                .open_table(UUID_TO_HNSW)
                .map_err(|e| CorviaError::Storage(format!("Failed to open UUID_TO_HNSW: {e}")))?;
            u2h_table
                .insert(uuid_str.as_str(), hnsw_id)
                .map_err(|e| CorviaError::Storage(format!("Failed to insert uuid_to_hnsw: {e}")))?;

            // Source version secondary index
            if !entry.source_version.is_empty() {
                let sv_key = format!("{}:{}", entry.scope_id, entry.source_version);
                let mut sv_table = write_txn
                    .open_table(SOURCE_VERSION_INDEX)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open SOURCE_VERSION_INDEX: {e}")))?;
                sv_table
                    .insert(sv_key.as_str(), uuid_str.as_str())
                    .map_err(|e| CorviaError::Storage(format!("Failed to insert source_version index: {e}")))?;
            }
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit index_entry: {e}")))?;

        Ok(())
    }

    /// Dump the HNSW index to disk for persistence.
    pub fn flush_hnsw(&self) -> Result<()> {
        let hnsw_dir = self.data_dir.join("hnsw");
        std::fs::create_dir_all(&hnsw_dir)
            .map_err(|e| CorviaError::Storage(format!("Failed to create hnsw dir: {e}")))?;

        // Remove old dump files before writing new ones.
        // hnsw_rs skips overwrite when datamap_opt is true (set after load_hnsw),
        // so we clean up first to ensure a fresh dump with the canonical name.
        let _ = std::fs::remove_file(hnsw_dir.join("litestore.hnsw.graph"));
        let _ = std::fs::remove_file(hnsw_dir.join("litestore.hnsw.data"));

        let hnsw = self
            .hnsw
            .lock()
            .map_err(|e| CorviaError::Storage(format!("HNSW lock poisoned: {e}")))?;
        hnsw.file_dump(&hnsw_dir, "litestore")
            .map_err(|e| CorviaError::Storage(format!("Failed to dump HNSW: {e}")))?;

        self.persist_next_id()?;
        info!("HNSW flushed to {}", hnsw_dir.display());
        Ok(())
    }

    /// Clear the TEMPORAL_INDEX table in Redb by dropping and re-creating it.
    fn clear_source_version_index(&self) -> Result<()> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        let _ = write_txn.delete_table(SOURCE_VERSION_INDEX);
        write_txn
            .open_table(SOURCE_VERSION_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to re-create SOURCE_VERSION_INDEX: {e}")))?;
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit clear sv index: {e}")))?;
        Ok(())
    }

    fn write_source_version_index(&self, entry: &KnowledgeEntry) -> Result<()> {
        if entry.source_version.is_empty() {
            return Ok(());
        }
        let sv_key = format!("{}:{}", entry.scope_id, entry.source_version);
        let uuid_str = entry.id.to_string();
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let mut sv_table = write_txn
                .open_table(SOURCE_VERSION_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to open SOURCE_VERSION_INDEX: {e}")))?;
            sv_table
                .insert(sv_key.as_str(), uuid_str.as_str())
                .map_err(|e| CorviaError::Storage(format!("Failed to insert sv index: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit sv index write: {e}")))?;
        Ok(())
    }

    fn clear_temporal_index(&self) -> Result<()> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        let existed = write_txn
            .delete_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to delete TEMPORAL_INDEX: {e}")))?;
        // Re-create the table so subsequent writes succeed
        write_txn
            .open_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to re-create TEMPORAL_INDEX: {e}")))?;
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit clear temporal: {e}")))?;

        if existed {
            info!("Cleared temporal index table");
        }
        Ok(())
    }

    /// Rebuild the HNSW index, Redb metadata, and temporal index from knowledge JSON files.
    /// Graph edges in Redb are preserved — petgraph is rebuilt from Redb during open().
    /// This is used to restore state after opening a fresh LiteStore.
    /// Returns the number of entries re-indexed.
    pub fn rebuild_from_files(&self) -> Result<usize> {
        let all_entries = knowledge_files::read_all(&self.data_dir)?;
        let count = all_entries.len();

        // Reset HNSW to a fresh index
        {
            let mut hnsw = self
                .hnsw
                .lock()
                .map_err(|e| CorviaError::Storage(format!("HNSW lock poisoned: {e}")))?;
            *hnsw = Hnsw::<f32, DistCosine>::new(
                MAX_NB_CONNECTION,
                MAX_ELEMENTS,
                MAX_LAYER,
                EF_CONSTRUCTION,
                DistCosine {},
            );
        }

        // Reset the counter
        self.next_hnsw_id.store(0, Ordering::SeqCst);

        // Clear the temporal index and source_version index before rebuilding
        self.clear_temporal_index()?;
        self.clear_source_version_index()?;

        for entry in &all_entries {
            if let Some(ref embedding) = entry.embedding {
                self.index_entry(entry, embedding)?;
            } else {
                // For entries without embeddings, still populate source_version index
                self.write_source_version_index(entry)?;
            }
            // Rebuild temporal index for every entry (even those without embeddings)
            self.write_temporal_index(entry)?;
        }

        self.persist_next_id()?;

        info!(
            "Rebuilt LiteStore from {} entries (HNSW + metadata + temporal index; graph edges preserved in Redb)",
            count
        );
        Ok(count)
    }

    /// Mark an entry as superseded by another. Updates valid_to, superseded_by,
    /// and the temporal index for the old entry.
    pub async fn supersede(&self, old_id: &uuid::Uuid, new_id: &uuid::Uuid) -> Result<()> {
        let now = chrono::Utc::now();

        // Read old entry, update fields, write back
        let mut old_entry = self
            .get(old_id)
            .await?
            .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", old_id)))?;

        old_entry.valid_to = Some(now);
        old_entry.superseded_by = Some(*new_id);

        // Update ENTRIES table
        let entry_json = serde_json::to_vec(&old_entry)
            .map_err(|e| CorviaError::Storage(format!("Failed to serialize: {e}")))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn
                .open_table(ENTRIES)
                .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;
            table
                .insert(old_id.to_string().as_str(), entry_json.as_slice())
                .map_err(|e| CorviaError::Storage(format!("Failed to update entry: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit supersede: {e}")))?;

        // Update temporal index for old entry (overwrites existing key since
        // scope_id:valid_from:entry_id is unchanged, but value now has valid_to set)
        self.write_temporal_index(&old_entry)?;

        // Update knowledge file
        knowledge_files::write_entry(&self.data_dir, &old_entry)?;

        Ok(())
    }
}

impl Drop for LiteStore {
    fn drop(&mut self) {
        if let Err(e) = self.flush_hnsw() {
            tracing::warn!(error = %e, "LiteStore: failed to flush HNSW on drop");
        } else {
            tracing::info!("LiteStore: HNSW index persisted to disk");
        }
    }
}

#[async_trait]
impl super::traits::QueryableStore for LiteStore {
    async fn init_schema(&self) -> Result<()> {
        // Create all Redb tables by opening them in a write transaction
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
        {
            let _ = write_txn
                .open_table(ENTRIES)
                .map_err(|e| CorviaError::Storage(format!("Failed to create ENTRIES: {e}")))?;
            let _ = write_txn
                .open_table(SCOPE_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to create SCOPE_INDEX: {e}")))?;
            let _ = write_txn
                .open_table(HNSW_TO_UUID)
                .map_err(|e| CorviaError::Storage(format!("Failed to create HNSW_TO_UUID: {e}")))?;
            let _ = write_txn
                .open_table(UUID_TO_HNSW)
                .map_err(|e| CorviaError::Storage(format!("Failed to create UUID_TO_HNSW: {e}")))?;
            let _ = write_txn
                .open_table(META)
                .map_err(|e| CorviaError::Storage(format!("Failed to create META: {e}")))?;
            let _ = write_txn
                .open_table(TEMPORAL_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to create TEMPORAL_INDEX: {e}")))?;
            let _ = write_txn
                .open_table(SOURCE_VERSION_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to create SOURCE_VERSION_INDEX: {e}")))?;
            let _ = write_txn
                .open_table(crate::graph_store::GRAPH_EDGES)
                .map_err(|e| CorviaError::Storage(format!("Failed to create GRAPH_EDGES: {e}")))?;
        }
        write_txn
            .commit()
            .map_err(|e| CorviaError::Storage(format!("Failed to commit init_schema: {e}")))?;

        info!("LiteStore schema initialized (dimensions={})", self.dimensions);
        Ok(())
    }

    #[tracing::instrument(name = "corvia.store.insert", skip(self, entry), fields(entry_id = %entry.id, scope_id = %entry.scope_id))]
    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()> {
        let embedding = entry
            .embedding
            .as_ref()
            .ok_or_else(|| CorviaError::Storage("Entry must have embedding set".into()))?;

        if embedding.len() != self.dimensions {
            warn!(
                "Embedding dimension mismatch: got {}d, store expects {}d. \
                 Run 'corvia rebuild' to re-index.",
                embedding.len(),
                self.dimensions
            );
            return Err(CorviaError::Storage(format!(
                "Embedding dimension mismatch: got {}d, store expects {}d. \
                 Run 'corvia rebuild' to re-index.",
                embedding.len(),
                self.dimensions
            )));
        }

        // 1. Write knowledge JSON file
        knowledge_files::write_entry(&self.data_dir, entry)?;

        // 2. Index into HNSW + Redb
        self.index_entry(entry, embedding)?;

        // 3. Persist the HNSW ID counter so auto-rebuild triggers on reopen
        self.persist_next_id()?;

        // 4. Write temporal index entry
        self.write_temporal_index(entry)?;

        Ok(())
    }

    #[tracing::instrument(name = "corvia.store.search", skip(self, embedding), fields(scope_id))]
    async fn search(
        &self,
        embedding: &[f32],
        scope_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Fetch K*2 candidates (min 10) to allow for scope filtering.
        // The floor of 10 ensures small limits (e.g. limit=1) still fetch enough
        // candidates to survive post-filter elimination of cross-scope and stale entries.
        let fetch_count = (limit * 2).max(10);

        let neighbours = {
            let hnsw = self
                .hnsw
                .lock()
                .map_err(|e| CorviaError::Storage(format!("HNSW lock poisoned: {e}")))?;
            hnsw.search(embedding, fetch_count, EF_SEARCH)
        };

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let h2u_table = read_txn
            .open_table(HNSW_TO_UUID)
            .map_err(|e| CorviaError::Storage(format!("Failed to open HNSW_TO_UUID: {e}")))?;
        let entries_table = read_txn
            .open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

        let mut results = Vec::new();

        for neighbour in neighbours {
            let hnsw_id = neighbour.d_id as u64;
            let distance = neighbour.distance;

            // Look up UUID from HNSW ID
            let uuid_str = match h2u_table.get(hnsw_id) {
                Ok(Some(val)) => val.value().to_string(),
                _ => continue, // Stale entry (deleted), skip
            };

            // Look up entry from UUID
            let entry_bytes = match entries_table.get(uuid_str.as_str()) {
                Ok(Some(val)) => val.value().to_vec(),
                _ => continue, // Entry not found in Redb, skip
            };

            let entry: KnowledgeEntry = match serde_json::from_slice(&entry_bytes) {
                Ok(e) => e,
                Err(_) => continue, // Corrupt entry, skip
            };

            // Post-filter by scope
            if entry.scope_id != scope_id {
                continue;
            }

            // Convert cosine distance to similarity score
            let score = 1.0 - distance;

            results.push(SearchResult { entry, score });

            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    #[tracing::instrument(name = "corvia.store.get", skip(self))]
    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>> {
        let uuid_str = id.to_string();
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let entries_table = read_txn
            .open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

        match entries_table.get(uuid_str.as_str()) {
            Ok(Some(val)) => {
                let entry: KnowledgeEntry = serde_json::from_slice(val.value())
                    .map_err(|e| CorviaError::Storage(format!("Failed to deserialize entry: {e}")))?;
                Ok(Some(entry))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(CorviaError::Storage(format!("Failed to get entry: {e}"))),
        }
    }

    async fn count(&self, scope_id: &str) -> Result<u64> {
        let prefix_start = format!("{scope_id}:");
        // Use the character after ':' as the exclusive upper bound for the range scan.
        // ';' is the next ASCII character after ':', so "scope_id;" is just past all "scope_id:*" keys.
        let prefix_end = format!("{scope_id};");

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let scope_table = read_txn
            .open_table(SCOPE_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to open SCOPE_INDEX: {e}")))?;

        let range = scope_table
            .range(prefix_start.as_str()..prefix_end.as_str())
            .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

        let count = range.count() as u64;
        Ok(count)
    }

    async fn delete_scope(&self, scope_id: &str) -> Result<()> {
        let prefix_start = format!("{scope_id}:");
        let prefix_end = format!("{scope_id};");

        // Collect UUIDs to delete
        let uuids_to_delete: Vec<String> = {
            let read_txn = self
                .db
                .begin_read()
                .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
            let scope_table = read_txn
                .open_table(SCOPE_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to open SCOPE_INDEX: {e}")))?;

            let range = scope_table
                .range(prefix_start.as_str()..prefix_end.as_str())
                .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

            range
                .filter_map(|item| item.ok().map(|(_, v)| v.value().to_string()))
                .collect()
        };

        if !uuids_to_delete.is_empty() {
            // Remove from Redb tables
            let write_txn = self
                .db
                .begin_write()
                .map_err(|e| CorviaError::Storage(format!("Failed to begin write txn: {e}")))?;
            {
                let mut entries_table = write_txn
                    .open_table(ENTRIES)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;
                let mut scope_table = write_txn
                    .open_table(SCOPE_INDEX)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open SCOPE_INDEX: {e}")))?;
                let mut h2u_table = write_txn
                    .open_table(HNSW_TO_UUID)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open HNSW_TO_UUID: {e}")))?;
                let mut u2h_table = write_txn
                    .open_table(UUID_TO_HNSW)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open UUID_TO_HNSW: {e}")))?;

                // Clear source_version index entries for this scope (range scan)
                let mut sv_table = write_txn
                    .open_table(SOURCE_VERSION_INDEX)
                    .map_err(|e| CorviaError::Storage(format!("Failed to open SOURCE_VERSION_INDEX: {e}")))?;
                let sv_keys: Vec<String> = {
                    let range = sv_table
                        .range(prefix_start.as_str()..prefix_end.as_str())
                        .map_err(|e| CorviaError::Storage(format!("Failed to range scan sv index: {e}")))?;
                    range.filter_map(|item| item.ok().map(|(k, _)| k.value().to_string())).collect()
                };
                for key in &sv_keys {
                    let _ = sv_table.remove(key.as_str());
                }

                for uuid_str in &uuids_to_delete {
                    // Remove entry
                    let _ = entries_table.remove(uuid_str.as_str());

                    // Remove scope index
                    let scope_key = format!("{scope_id}:{uuid_str}");
                    let _ = scope_table.remove(scope_key.as_str());

                    // Remove HNSW ID mappings
                    if let Ok(Some(hnsw_id_val)) = u2h_table.remove(uuid_str.as_str()) {
                        let _ = h2u_table.remove(hnsw_id_val.value());
                    }
                }
            }
            write_txn
                .commit()
                .map_err(|e| CorviaError::Storage(format!("Failed to commit delete_scope: {e}")))?;
        }

        // Delete knowledge files for this scope
        knowledge_files::delete_scope_files(&self.data_dir, scope_id)?;

        info!("Deleted scope '{}' ({} entries)", scope_id, uuids_to_delete.len());
        Ok(())
    }

    async fn get_by_source_version(
        &self,
        scope_id: &str,
        source_version: &str,
    ) -> Result<Option<KnowledgeEntry>> {
        if source_version.is_empty() {
            return Ok(None);
        }
        let sv_key = format!("{scope_id}:{source_version}");

        let uuid_str = {
            let read_txn = self
                .db
                .begin_read()
                .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
            let sv_table = read_txn
                .open_table(SOURCE_VERSION_INDEX)
                .map_err(|e| CorviaError::Storage(format!("Failed to open SOURCE_VERSION_INDEX: {e}")))?;
            match sv_table.get(sv_key.as_str()) {
                Ok(Some(val)) => Some(val.value().to_string()),
                Ok(None) => None,
                Err(e) => return Err(CorviaError::Storage(format!("SOURCE_VERSION_INDEX lookup failed: {e}"))),
            }
        };

        match uuid_str {
            Some(id_str) => {
                let id = uuid::Uuid::parse_str(&id_str)
                    .map_err(|e| CorviaError::Storage(format!("Invalid UUID in sv index: {e}")))?;
                self.get(&id).await
            }
            None => Ok(None),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl crate::traits::TemporalStore for LiteStore {
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        let ts_millis = timestamp.timestamp_millis();
        // Range scan: all keys from "{scope_id}:" up to "{scope_id}:{ts_millis:020}:~"
        let range_start = format!("{}:", scope_id);
        // ~ (ASCII 126) is the highest printable char, above all UUID hex digits and hyphens
        let range_end_key = format!("{}:{:020}:~", scope_id, ts_millis);

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn
            .open_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;

        let entries_table = read_txn
            .open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

        let range = table
            .range(range_start.as_str()..=range_end_key.as_str())
            .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

        let mut results = Vec::new();
        for item in range {
            let (key, value) = item
                .map_err(|e| CorviaError::Storage(format!("Failed to read range item: {e}")))?;
            let key_str = key.value();

            // Entry valid for: [valid_from, valid_to) — valid_to is exclusive upper bound
            // Parse valid_to from value to check if entry is still valid at timestamp
            let val_json: serde_json::Value = serde_json::from_slice(value.value())
                .map_err(|e| {
                    CorviaError::Storage(format!("Failed to parse temporal value: {e}"))
                })?;

            if let Some(valid_to_millis) = val_json["valid_to_millis"].as_i64()
                && valid_to_millis <= ts_millis {
                    continue; // Entry expired before the query timestamp
                }

            // Extract entry_id from compound key (UUID is after the last colon group)
            let parts: Vec<&str> = key_str.rsplitn(2, ':').collect();
            if parts.len() < 2 {
                tracing::warn!("Malformed temporal index key (no entry_id segment): {}", key_str);
                continue;
            }
            let entry_uuid_str = parts[0];

            // Look up entry from ENTRIES table
            if let Ok(Some(entry_val)) = entries_table.get(entry_uuid_str)
                && let Ok(entry) = serde_json::from_slice::<KnowledgeEntry>(entry_val.value()) {
                    results.push(entry);
                    if results.len() >= limit {
                        break;
                    }
                }
        }

        Ok(results)
    }

    async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>> {
        let mut chain = Vec::new();

        // Start with the given entry (newest in chain)
        let start = self
            .get(entry_id)
            .await?
            .ok_or_else(|| CorviaError::Storage(format!("Entry {} not found", entry_id)))?;
        chain.push(start);

        // Build reverse lookup: superseded_by -> entry (i.e., "who was replaced by X?")
        // Simple scan approach — for large stores, a reverse index would be better (YAGNI).
        let all_entries = knowledge_files::read_all(&self.data_dir)?;
        let mut by_superseded: std::collections::HashMap<uuid::Uuid, KnowledgeEntry> =
            all_entries
                .into_iter()
                .filter_map(|e| e.superseded_by.map(|s| (s, e)))
                .collect();

        // Walk backward: find who was superseded by current entry, then who was
        // superseded by that, etc.
        let mut current = entry_id;
        while let Some(predecessor) = by_superseded.remove(current) {
            chain.push(predecessor);
            current = &chain.last().unwrap().id;
        }

        Ok(chain)
    }

    async fn evolution(
        &self,
        scope_id: &str,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<KnowledgeEntry>> {
        let from_millis = from.timestamp_millis();
        let to_millis = to.timestamp_millis();

        let range_start = format!("{}:{:020}:", scope_id, from_millis);
        // ~ (ASCII 126) is the highest printable char, above all UUID hex digits and hyphens
        let range_end = format!("{}:{:020}:~", scope_id, to_millis);

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| CorviaError::Storage(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn
            .open_table(TEMPORAL_INDEX)
            .map_err(|e| CorviaError::Storage(format!("Failed to open TEMPORAL_INDEX: {e}")))?;
        let entries_table = read_txn
            .open_table(ENTRIES)
            .map_err(|e| CorviaError::Storage(format!("Failed to open ENTRIES: {e}")))?;

        let range = table
            .range(range_start.as_str()..=range_end.as_str())
            .map_err(|e| CorviaError::Storage(format!("Failed to range scan: {e}")))?;

        let mut results = Vec::new();
        for item in range {
            let (key, _value) = item
                .map_err(|e| CorviaError::Storage(format!("Failed to read range item: {e}")))?;
            let key_str = key.value();
            let parts: Vec<&str> = key_str.rsplitn(2, ':').collect();
            if parts.len() < 2 {
                tracing::warn!("Malformed temporal index key (no entry_id segment): {}", key_str);
                continue;
            }
            let entry_uuid_str = parts[0];

            if let Ok(Some(entry_val)) = entries_table.get(entry_uuid_str)
                && let Ok(entry) = serde_json::from_slice::<KnowledgeEntry>(entry_val.value()) {
                    results.push(entry);
                }
        }

        Ok(results)
    }
}

#[async_trait]
impl crate::traits::GraphStore for LiteStore {
    async fn relate(
        &self,
        from: &uuid::Uuid,
        relation: &str,
        to: &uuid::Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.graph.relate(from, relation, to, metadata).await
    }

    async fn edges(
        &self,
        entry_id: &uuid::Uuid,
        direction: corvia_common::types::EdgeDirection,
    ) -> Result<Vec<corvia_common::types::GraphEdge>> {
        self.graph.edges(entry_id, direction).await
    }

    async fn traverse(
        &self,
        start: &uuid::Uuid,
        relation: Option<&str>,
        direction: corvia_common::types::EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>> {
        let ids = self.graph.traverse_ids(start, relation, direction, max_depth)?;
        let mut entries = Vec::new();
        for id in ids {
            if let Some(entry) = self.get(&id).await? {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    async fn shortest_path(
        &self,
        from: &uuid::Uuid,
        to: &uuid::Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>> {
        let ids = match self.graph.shortest_path_ids(from, to)? {
            Some(ids) => ids,
            None => return Ok(None),
        };
        let mut entries = Vec::new();
        for id in ids {
            if let Some(entry) = self.get(&id).await? {
                entries.push(entry);
            }
        }
        Ok(Some(entries))
    }

    async fn remove_edges(&self, entry_id: &uuid::Uuid) -> Result<()> {
        self.graph.remove_edges(entry_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{QueryableStore, TemporalStore};
    use corvia_common::types::KnowledgeEntry;

    #[tokio::test]
    async fn test_lite_store_insert_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let entry = KnowledgeEntry::new("fn hello() {}".into(), "test".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        store.insert(&entry).await.unwrap();
        assert_eq!(store.count("test").await.unwrap(), 1);
        assert_eq!(store.count("other").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_lite_store_get() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let entry = KnowledgeEntry::new("test content".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        let id = entry.id;
        store.insert(&entry).await.unwrap();
        let retrieved = store.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
        assert!(store.get(&uuid::Uuid::now_v7()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_lite_store_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let e1 = KnowledgeEntry::new("auth function".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = KnowledgeEntry::new("database query".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        let results = store.search(&[0.9, 0.1, 0.0], "scope", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].entry.content.contains("auth"));
    }

    #[tokio::test]
    async fn test_lite_store_scope_filter() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let e1 = KnowledgeEntry::new("in scope A".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = KnowledgeEntry::new("in scope B".into(), "scope-b".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        let results = store.search(&[1.0, 0.0, 0.0], "scope-a", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].entry.content.contains("scope A"));
    }

    #[tokio::test]
    async fn test_lite_store_delete_scope() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let entry = KnowledgeEntry::new("temp data".into(), "temp".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        store.insert(&entry).await.unwrap();
        assert_eq!(store.count("temp").await.unwrap(), 1);
        store.delete_scope("temp").await.unwrap();
        assert_eq!(store.count("temp").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_lite_store_knowledge_files_written() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let entry = KnowledgeEntry::new("file test".into(), "files".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        store.insert(&entry).await.unwrap();
        let file_path = dir
            .path()
            .join("knowledge")
            .join("files")
            .join(format!("{}.json", entry.id));
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_lite_store_dimension_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let entry = KnowledgeEntry::new("wrong dims".into(), "test".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2]); // 2 dims, store expects 3
        let result = store.insert(&entry).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
    }

    #[tokio::test]
    async fn test_lite_store_survives_reopen() {
        // Regression: next_hnsw_id must be persisted so auto-rebuild triggers on reopen.
        let dir = tempfile::tempdir().unwrap();
        {
            let store = LiteStore::open(dir.path(), 3).unwrap();
            store.init_schema().await.unwrap();
            let e1 = KnowledgeEntry::new("auth module".into(), "scope".into(), "v1".into())
                .with_embedding(vec![1.0, 0.0, 0.0]);
            let e2 = KnowledgeEntry::new("database layer".into(), "scope".into(), "v1".into())
                .with_embedding(vec![0.0, 1.0, 0.0]);
            store.insert(&e1).await.unwrap();
            store.insert(&e2).await.unwrap();
            assert_eq!(store.count("scope").await.unwrap(), 2);
            // store dropped here — simulates process exit
        }
        {
            // Reopen the same store directory
            let store = LiteStore::open(dir.path(), 3).unwrap();
            assert_eq!(store.count("scope").await.unwrap(), 2);
            let results = store.search(&[0.9, 0.1, 0.0], "scope", 1).await.unwrap();
            assert!(results.len() >= 1, "search should find results after reopen");
        }
    }

    #[tokio::test]
    async fn test_temporal_index_populated_on_insert() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let entry = KnowledgeEntry::new("temporal test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let id = entry.id;
        let valid_from = entry.valid_from;
        store.insert(&entry).await.unwrap();

        // Verify temporal index was written
        let results = store.as_of("scope", valid_from + chrono::Duration::seconds(1), 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[tokio::test]
    async fn test_temporal_history_follows_chain() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Create original entry
        let e1 = KnowledgeEntry::new("version 1".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e1_id = e1.id;
        store.insert(&e1).await.unwrap();

        // Create superseding entry
        let e2 = KnowledgeEntry::new("version 2".into(), "scope".into(), "v2".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2_id = e2.id;
        store.insert(&e2).await.unwrap();

        // Mark e1 as superseded by e2
        store.supersede(&e1_id, &e2_id).await.unwrap();

        let history = store.history(&e2_id).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].id, e2_id); // newest first
        assert_eq!(history[1].id, e1_id);
    }

    #[tokio::test]
    async fn test_temporal_evolution_time_range() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let before = chrono::Utc::now();
        let e1 = KnowledgeEntry::new("entry 1".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert(&e1).await.unwrap();
        let after = chrono::Utc::now();

        let results = store.evolution("scope", before, after).await.unwrap();
        assert!(results.len() >= 1);

        // Query a range before the entry was created — should be empty
        let old_results = store
            .evolution(
                "scope",
                before - chrono::Duration::hours(2),
                before - chrono::Duration::hours(1),
            )
            .await
            .unwrap();
        assert_eq!(old_results.len(), 0);
    }

    #[tokio::test]
    async fn test_lite_store_search_limit_one() {
        // Verifies the .max(10) fetch floor: even with limit=1,
        // search should find the correct result among multiple scopes.
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Insert entries in two different scopes with similar embeddings
        for i in 0..5 {
            let entry = KnowledgeEntry::new(
                format!("other scope {i}"), "other".into(), "v1".into(),
            ).with_embedding(vec![1.0, 0.0, 0.0]);
            store.insert(&entry).await.unwrap();
        }
        let target = KnowledgeEntry::new(
            "target entry".into(), "target-scope".into(), "v1".into(),
        ).with_embedding(vec![1.0, 0.0, 0.0]);
        store.insert(&target).await.unwrap();

        // Search with limit=1 in target-scope — must find the entry
        // despite 5 other-scope entries being closer in HNSW insertion order
        let results = store.search(&[1.0, 0.0, 0.0], "target-scope", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.content, "target entry");
    }

    #[tokio::test]
    async fn test_lite_store_graph_integration() {
        use crate::traits::GraphStore;
        use corvia_common::types::EdgeDirection;

        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Insert two entries
        let e1 = KnowledgeEntry::new("module A".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = KnowledgeEntry::new("module B".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);
        let id1 = e1.id;
        let id2 = e2.id;
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        // Create a graph edge
        store.relate(&id1, "imports", &id2, None).await.unwrap();

        // Query edges
        let edges = store.edges(&id1, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to, id2);

        // Traverse — should return actual KnowledgeEntry objects now
        let traversed = store.traverse(&id1, None, EdgeDirection::Outgoing, 1).await.unwrap();
        assert_eq!(traversed.len(), 1);
        assert_eq!(traversed[0].content, "module B");

        // Shortest path with entry lookup
        let path = store.shortest_path(&id1, &id2).await.unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].content, "module A");
        assert_eq!(path[1].content, "module B");
    }

    #[tokio::test]
    async fn test_rebuild_restores_temporal_index() {
        // Insert entries with temporal data, clear the temporal index,
        // call rebuild_from_files(), verify temporal queries still work.
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let before = chrono::Utc::now();

        let e1 = KnowledgeEntry::new("rebuild temporal A".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = KnowledgeEntry::new("rebuild temporal B".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);
        let id1 = e1.id;
        let id2 = e2.id;
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        let after = chrono::Utc::now();

        // Verify temporal queries work before corruption
        let results = store.as_of("scope", after, 10).await.unwrap();
        assert_eq!(results.len(), 2);

        let evolution = store.evolution("scope", before, after).await.unwrap();
        assert_eq!(evolution.len(), 2);

        // Manually clear the TEMPORAL_INDEX table to simulate corruption
        store.clear_temporal_index().unwrap();

        // Verify temporal queries return nothing after corruption
        let results_after_clear = store.as_of("scope", after, 10).await.unwrap();
        assert_eq!(results_after_clear.len(), 0, "temporal index should be empty after clearing");

        let evolution_after_clear = store.evolution("scope", before, after).await.unwrap();
        assert_eq!(evolution_after_clear.len(), 0, "evolution should be empty after clearing");

        // Rebuild from knowledge files — should restore the temporal index
        let rebuilt = store.rebuild_from_files().unwrap();
        assert_eq!(rebuilt, 2);

        // Verify temporal queries work again after rebuild
        let results_rebuilt = store.as_of("scope", after, 10).await.unwrap();
        assert_eq!(results_rebuilt.len(), 2, "temporal index should be restored after rebuild");

        let evolution_rebuilt = store.evolution("scope", before, after).await.unwrap();
        assert_eq!(evolution_rebuilt.len(), 2, "evolution should be restored after rebuild");

        // Verify the correct entries are returned
        let ids: Vec<uuid::Uuid> = results_rebuilt.iter().map(|e| e.id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[tokio::test]
    async fn test_rebuild_preserves_graph_edges() {
        use crate::traits::GraphStore;
        use corvia_common::types::EdgeDirection;

        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Insert entries and create graph edges
        let e1 = KnowledgeEntry::new("graph node A".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = KnowledgeEntry::new("graph node B".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);
        let e3 = KnowledgeEntry::new("graph node C".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 0.0, 1.0]);
        let id1 = e1.id;
        let id2 = e2.id;
        let id3 = e3.id;
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        store.insert(&e3).await.unwrap();

        store.relate(&id1, "imports", &id2, None).await.unwrap();
        store.relate(&id2, "calls", &id3, None).await.unwrap();

        // Verify edges exist before rebuild
        let edges_before = store.edges(&id1, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(edges_before.len(), 1);
        assert_eq!(edges_before[0].to, id2);

        let path_before = store.shortest_path(&id1, &id3).await.unwrap();
        assert!(path_before.is_some());
        assert_eq!(path_before.unwrap().len(), 3);

        // Rebuild from files — graph edges should survive
        let rebuilt = store.rebuild_from_files().unwrap();
        assert_eq!(rebuilt, 3);

        // Verify edges still exist after rebuild
        let edges_after = store.edges(&id1, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(edges_after.len(), 1, "graph edges should survive rebuild");
        assert_eq!(edges_after[0].to, id2);
        assert_eq!(edges_after[0].relation, "imports");

        let edges_b = store.edges(&id2, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(edges_b.len(), 1, "B->C edge should survive rebuild");
        assert_eq!(edges_b[0].to, id3);

        // Verify traversal still works
        let traversed = store.traverse(&id1, None, EdgeDirection::Outgoing, 2).await.unwrap();
        assert_eq!(traversed.len(), 2, "traversal should find B and C after rebuild");

        // Verify shortest path still works
        let path_after = store.shortest_path(&id1, &id3).await.unwrap();
        assert!(path_after.is_some(), "shortest path should still work after rebuild");
        let path = path_after.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].content, "graph node A");
        assert_eq!(path[1].content, "graph node B");
        assert_eq!(path[2].content, "graph node C");
    }

    /// Traverse depth-2 with LiteStore (entry-aware): verifies depth limits,
    /// direction filtering, and relation-type filtering.
    #[tokio::test]
    async fn test_lite_store_traverse_depth_2() {
        use crate::traits::GraphStore;

        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Insert A, B, C with embeddings
        let ea = KnowledgeEntry::new("node A".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let eb = KnowledgeEntry::new("node B".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);
        let ec = KnowledgeEntry::new("node C".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.0, 0.0, 1.0]);

        let id_a = ea.id;
        let id_b = eb.id;
        let id_c = ec.id;

        store.insert(&ea).await.unwrap();
        store.insert(&eb).await.unwrap();
        store.insert(&ec).await.unwrap();

        // Edges: A→B (imports), B→C (imports)
        store.relate(&id_a, "imports", &id_b, None).await.unwrap();
        store.relate(&id_b, "imports", &id_c, None).await.unwrap();

        // Depth 2, Both direction: should find B and C
        let result = store
            .traverse(&id_a, None, corvia_common::types::EdgeDirection::Both, 2)
            .await
            .unwrap();
        assert_eq!(result.len(), 2, "depth 2 Both should find B and C");
        let contents: Vec<&str> = result.iter().map(|e| e.content.as_str()).collect();
        assert!(contents.contains(&"node B"));
        assert!(contents.contains(&"node C"));

        // Depth 1: should find only B
        let result_d1 = store
            .traverse(&id_a, None, corvia_common::types::EdgeDirection::Both, 1)
            .await
            .unwrap();
        assert_eq!(result_d1.len(), 1, "depth 1 should find only B");
        assert_eq!(result_d1[0].content, "node B");

        // Relation filter: only "imports" Outgoing depth 2
        let result_filtered = store
            .traverse(
                &id_a,
                Some("imports"),
                corvia_common::types::EdgeDirection::Outgoing,
                2,
            )
            .await
            .unwrap();
        assert_eq!(
            result_filtered.len(),
            2,
            "filtered 'imports' Outgoing depth 2 should find B and C"
        );

        // Relation filter: "calls" should find nothing (no calls edges)
        let result_no_match = store
            .traverse(
                &id_a,
                Some("calls"),
                corvia_common::types::EdgeDirection::Outgoing,
                2,
            )
            .await
            .unwrap();
        assert!(
            result_no_match.is_empty(),
            "filtering by non-existent relation should return empty"
        );

        // Returned entries have correct content
        for entry in &result {
            assert!(!entry.content.is_empty());
            assert_eq!(entry.scope_id, "scope");
        }
    }

    #[tokio::test]
    async fn test_lite_store_get_by_source_version() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let entry = KnowledgeEntry::new(
            "session turn 1".into(),
            "user-history".into(),
            "ses-abc123:turn-1".into(),
        )
        .with_embedding(vec![0.1, 0.2, 0.3]);
        let expected_id = entry.id;
        store.insert(&entry).await.unwrap();

        // Found by exact scope + source_version
        let result = store
            .get_by_source_version("user-history", "ses-abc123:turn-1")
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, expected_id);

        // Not found: wrong scope
        let result = store
            .get_by_source_version("other-scope", "ses-abc123:turn-1")
            .await
            .unwrap();
        assert!(result.is_none());

        // Not found: wrong source_version
        let result = store
            .get_by_source_version("user-history", "ses-nonexistent:turn-1")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_lite_store_get_by_source_version_empty_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        // Entry with empty source_version should not be indexed
        let entry = KnowledgeEntry::new("no version".into(), "scope".into(), "".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        store.insert(&entry).await.unwrap();

        let result = store
            .get_by_source_version("scope", "")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_lite_store_get_by_source_version_survives_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let entry = KnowledgeEntry::new(
            "rebuild test".into(),
            "scope".into(),
            "ses-rebuild:turn-1".into(),
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);
        let expected_id = entry.id;
        store.insert(&entry).await.unwrap();

        // Rebuild clears and repopulates indexes
        store.rebuild_from_files().unwrap();

        let result = store
            .get_by_source_version("scope", "ses-rebuild:turn-1")
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, expected_id);
    }

    #[tokio::test]
    async fn test_lite_store_get_by_source_version_cleaned_on_delete_scope() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();

        let entry = KnowledgeEntry::new(
            "deletable".into(),
            "temp-scope".into(),
            "ses-del:turn-1".into(),
        )
        .with_embedding(vec![0.5, 0.5, 0.5]);
        store.insert(&entry).await.unwrap();

        // Verify it exists
        assert!(store
            .get_by_source_version("temp-scope", "ses-del:turn-1")
            .await
            .unwrap()
            .is_some());

        // Delete scope
        store.delete_scope("temp-scope").await.unwrap();

        // Source version index should be cleaned
        assert!(store
            .get_by_source_version("temp-scope", "ses-del:turn-1")
            .await
            .unwrap()
            .is_none());
    }
}
