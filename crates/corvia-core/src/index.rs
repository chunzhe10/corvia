//! Redb-backed index for vectors, chunk-to-entry mappings, supersession state,
//! and metadata.
//!
//! All data is stored in a single Redb database file with five tables:
//!
//! - `VECTORS`: chunk_id -> f32 vector bytes
//! - `CHUNK_MAP`: chunk_id -> source entry ID
//! - `SUPERSESSION`: entry_id -> 0 (current) or 1 (superseded)
//! - `CHUNK_KIND`: chunk_id -> knowledge kind string
//! - `META`: string key -> string value (e.g. "last_ingest", "entry_count")

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Chunk ID -> vector bytes (f32 array serialized as little-endian bytes).
const VECTORS: TableDefinition<&str, &[u8]> = TableDefinition::new("vectors");

/// Chunk ID -> source entry ID.
const CHUNK_MAP: TableDefinition<&str, &str> = TableDefinition::new("chunk_map");

/// Entry ID -> supersession state (0 = current, 1 = superseded).
const SUPERSESSION: TableDefinition<&str, u8> = TableDefinition::new("supersession");

/// Chunk ID -> knowledge kind string (e.g. "decision", "learning").
const CHUNK_KIND: TableDefinition<&str, &str> = TableDefinition::new("chunk_kind");

/// Arbitrary metadata key-value pairs.
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

// ---------------------------------------------------------------------------
// Byte conversion helpers
// ---------------------------------------------------------------------------

/// Convert an f32 slice to little-endian bytes.
fn f32_to_bytes(floats: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Convert little-endian bytes back to an f32 vec.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// RedbIndex
// ---------------------------------------------------------------------------

/// Persistent index backed by a Redb embedded database.
///
/// Stores vectors, chunk-to-entry mappings, supersession state, and metadata
/// in a single file. Thread-safe: Redb handles internal locking.
pub struct RedbIndex {
    db: Database,
}

impl RedbIndex {
    /// Create or open a Redb database at `path`.
    ///
    /// All four tables are created on first open via a write transaction.
    pub fn open(path: &Path) -> Result<Self> {
        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create dir for redb: {}", parent.display()))?;
        }

        let db = Database::create(path)
            .with_context(|| format!("failed to open redb database: {}", path.display()))?;

        // Create all tables on first open.
        {
            let txn = db.begin_write().context("failed to begin write txn for table creation")?;
            // Opening a table in a write transaction creates it if it doesn't exist.
            let _vectors = txn.open_table(VECTORS).context("failed to create VECTORS table")?;
            let _chunk_map = txn.open_table(CHUNK_MAP).context("failed to create CHUNK_MAP table")?;
            let _supersession = txn.open_table(SUPERSESSION).context("failed to create SUPERSESSION table")?;
            let _chunk_kind = txn.open_table(CHUNK_KIND).context("failed to create CHUNK_KIND table")?;
            let _meta = txn.open_table(META).context("failed to create META table")?;
            // Must drop tables before committing.
            drop(_vectors);
            drop(_chunk_map);
            drop(_supersession);
            drop(_chunk_kind);
            drop(_meta);
            txn.commit().context("failed to commit table creation")?;
        }

        Ok(Self { db })
    }

    // -----------------------------------------------------------------------
    // Vector operations
    // -----------------------------------------------------------------------

    /// Store a vector and its chunk-to-entry mapping in a single transaction.
    pub fn put_vector(&self, chunk_id: &str, entry_id: &str, vector: &[f32]) -> Result<()> {
        let bytes = f32_to_bytes(vector);
        let txn = self.db.begin_write().context("put_vector: begin_write")?;
        {
            let mut vectors = txn.open_table(VECTORS).context("put_vector: open VECTORS")?;
            vectors.insert(chunk_id, bytes.as_slice()).context("put_vector: insert vector")?;
        }
        {
            let mut chunk_map = txn.open_table(CHUNK_MAP).context("put_vector: open CHUNK_MAP")?;
            chunk_map.insert(chunk_id, entry_id).context("put_vector: insert chunk mapping")?;
        }
        txn.commit().context("put_vector: commit")?;
        Ok(())
    }

    /// Get all vectors with their chunk IDs (for brute-force search).
    pub fn all_vectors(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let txn = self.db.begin_read().context("all_vectors: begin_read")?;
        let table = txn.open_table(VECTORS).context("all_vectors: open VECTORS")?;
        let mut results = Vec::new();
        for entry in table.iter().context("all_vectors: iter")? {
            let (key, value) = entry.context("all_vectors: read entry")?;
            let chunk_id = key.value().to_string();
            let vector = bytes_to_f32(value.value());
            results.push((chunk_id, vector));
        }
        Ok(results)
    }

    /// Get source entry ID for a chunk.
    pub fn chunk_entry_id(&self, chunk_id: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read().context("chunk_entry_id: begin_read")?;
        let table = txn.open_table(CHUNK_MAP).context("chunk_entry_id: open CHUNK_MAP")?;
        match table.get(chunk_id).context("chunk_entry_id: get")? {
            Some(guard) => Ok(Some(guard.value().to_string())),
            None => Ok(None),
        }
    }

    /// Store the kind classification for a chunk.
    pub fn put_chunk_kind(&self, chunk_id: &str, kind: &str) -> Result<()> {
        let txn = self.db.begin_write().context("put_chunk_kind: begin_write")?;
        {
            let mut table = txn.open_table(CHUNK_KIND).context("put_chunk_kind: open")?;
            table.insert(chunk_id, kind).context("put_chunk_kind: insert")?;
        }
        txn.commit().context("put_chunk_kind: commit")?;
        Ok(())
    }

    /// Get the kind classification for a chunk.
    pub fn get_chunk_kind(&self, chunk_id: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read().context("get_chunk_kind: begin_read")?;
        let table = txn.open_table(CHUNK_KIND).context("get_chunk_kind: open")?;
        match table.get(chunk_id).context("get_chunk_kind: get")? {
            Some(guard) => Ok(Some(guard.value().to_string())),
            None => Ok(None),
        }
    }

    /// Total number of stored vectors.
    pub fn vector_count(&self) -> Result<u64> {
        let txn = self.db.begin_read().context("vector_count: begin_read")?;
        let table = txn.open_table(VECTORS).context("vector_count: open VECTORS")?;
        table.len().context("vector_count: len")
    }

    // -----------------------------------------------------------------------
    // Supersession tracking
    // -----------------------------------------------------------------------

    /// Mark an entry as superseded (true) or current (false).
    pub fn set_superseded(&self, entry_id: &str, superseded: bool) -> Result<()> {
        let val: u8 = if superseded { 1 } else { 0 };
        let txn = self.db.begin_write().context("set_superseded: begin_write")?;
        {
            let mut table = txn.open_table(SUPERSESSION).context("set_superseded: open")?;
            table.insert(entry_id, val).context("set_superseded: insert")?;
        }
        txn.commit().context("set_superseded: commit")?;
        Ok(())
    }

    /// Check if an entry is superseded.
    pub fn is_superseded(&self, entry_id: &str) -> Result<bool> {
        let txn = self.db.begin_read().context("is_superseded: begin_read")?;
        let table = txn.open_table(SUPERSESSION).context("is_superseded: open")?;
        match table.get(entry_id).context("is_superseded: get")? {
            Some(guard) => Ok(guard.value() == 1),
            None => Ok(false),
        }
    }

    /// Get the set of all superseded entry IDs.
    pub fn superseded_ids(&self) -> Result<HashSet<String>> {
        let txn = self.db.begin_read().context("superseded_ids: begin_read")?;
        let table = txn.open_table(SUPERSESSION).context("superseded_ids: open")?;
        let mut ids = HashSet::new();
        for entry in table.iter().context("superseded_ids: iter")? {
            let (key, value) = entry.context("superseded_ids: read entry")?;
            if value.value() == 1 {
                ids.insert(key.value().to_string());
            }
        }
        Ok(ids)
    }

    /// Check if an entry exists in the supersession table.
    pub fn entry_exists(&self, entry_id: &str) -> Result<bool> {
        let txn = self.db.begin_read().context("entry_exists: begin_read")?;
        let table = txn.open_table(SUPERSESSION).context("entry_exists: open")?;
        Ok(table.get(entry_id).context("entry_exists: get")?.is_some())
    }

    /// Total number of entries in the supersession table.
    pub fn entry_count(&self) -> Result<u64> {
        let txn = self.db.begin_read().context("entry_count: begin_read")?;
        let table = txn.open_table(SUPERSESSION).context("entry_count: open")?;
        table.len().context("entry_count: len")
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Set a metadata key-value pair.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let txn = self.db.begin_write().context("set_meta: begin_write")?;
        {
            let mut table = txn.open_table(META).context("set_meta: open")?;
            table.insert(key, value).context("set_meta: insert")?;
        }
        txn.commit().context("set_meta: commit")?;
        Ok(())
    }

    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let txn = self.db.begin_read().context("get_meta: begin_read")?;
        let table = txn.open_table(META).context("get_meta: open")?;
        match table.get(key).context("get_meta: get")? {
            Some(guard) => Ok(Some(guard.value().to_string())),
            None => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Clear all tables (for --fresh ingest).
    ///
    /// Uses `retain(|_, _| false)` to remove every row from each table.
    pub fn clear_all(&self) -> Result<()> {
        let txn = self.db.begin_write().context("clear_all: begin_write")?;
        {
            let mut vectors = txn.open_table(VECTORS).context("clear_all: open VECTORS")?;
            vectors.retain(|_, _| false).context("clear_all: clear VECTORS")?;
        }
        {
            let mut chunk_map = txn.open_table(CHUNK_MAP).context("clear_all: open CHUNK_MAP")?;
            chunk_map.retain(|_, _| false).context("clear_all: clear CHUNK_MAP")?;
        }
        {
            let mut supersession = txn.open_table(SUPERSESSION).context("clear_all: open SUPERSESSION")?;
            supersession.retain(|_, _| false).context("clear_all: clear SUPERSESSION")?;
        }
        {
            let mut chunk_kind = txn.open_table(CHUNK_KIND).context("clear_all: open CHUNK_KIND")?;
            chunk_kind.retain(|_, _| false).context("clear_all: clear CHUNK_KIND")?;
        }
        {
            let mut meta = txn.open_table(META).context("clear_all: open META")?;
            meta.retain(|_, _| false).context("clear_all: clear META")?;
        }
        txn.commit().context("clear_all: commit")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Open a RedbIndex in a fresh temp directory.
    fn open_temp_index() -> (RedbIndex, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let index = RedbIndex::open(&db_path).unwrap();
        (index, dir)
    }

    #[test]
    fn put_and_get_vector() {
        let (index, _dir) = open_temp_index();

        let vec_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.5];
        index.put_vector("chunk-1", "entry-1", &vec_data).unwrap();

        let all = index.all_vectors().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "chunk-1");
        assert_eq!(all[0].1, vec_data);
    }

    #[test]
    fn chunk_entry_mapping() {
        let (index, _dir) = open_temp_index();

        index.put_vector("chunk-a", "entry-x", &[0.1, 0.2]).unwrap();

        let entry_id = index.chunk_entry_id("chunk-a").unwrap();
        assert_eq!(entry_id, Some("entry-x".to_string()));

        let missing = index.chunk_entry_id("nonexistent").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn supersession_tracking() {
        let (index, _dir) = open_temp_index();

        // Initially, entries don't exist.
        assert!(!index.is_superseded("e1").unwrap());
        assert!(!index.entry_exists("e1").unwrap());

        // Set e1 as current, e2 as superseded.
        index.set_superseded("e1", false).unwrap();
        index.set_superseded("e2", true).unwrap();
        index.set_superseded("e3", true).unwrap();

        assert!(!index.is_superseded("e1").unwrap());
        assert!(index.is_superseded("e2").unwrap());
        assert!(index.is_superseded("e3").unwrap());

        assert!(index.entry_exists("e1").unwrap());
        assert!(index.entry_exists("e2").unwrap());

        let superseded = index.superseded_ids().unwrap();
        assert_eq!(superseded.len(), 2);
        assert!(superseded.contains("e2"));
        assert!(superseded.contains("e3"));
        assert!(!superseded.contains("e1"));

        // Flip e2 back to current.
        index.set_superseded("e2", false).unwrap();
        assert!(!index.is_superseded("e2").unwrap());

        let superseded = index.superseded_ids().unwrap();
        assert_eq!(superseded.len(), 1);
        assert!(superseded.contains("e3"));

        // Entry count should be 3 (e1, e2, e3 all in the table).
        assert_eq!(index.entry_count().unwrap(), 3);
    }

    #[test]
    fn clear_all_empties_tables() {
        let (index, _dir) = open_temp_index();

        // Populate all tables.
        index.put_vector("c1", "e1", &[1.0, 2.0]).unwrap();
        index.put_vector("c2", "e2", &[3.0, 4.0]).unwrap();
        index.set_superseded("e1", false).unwrap();
        index.set_superseded("e2", true).unwrap();
        index.set_meta("last_ingest", "2026-04-15T00:00:00Z").unwrap();

        // Verify non-zero counts.
        assert_eq!(index.vector_count().unwrap(), 2);
        assert_eq!(index.entry_count().unwrap(), 2);
        assert!(index.get_meta("last_ingest").unwrap().is_some());

        // Clear everything.
        index.clear_all().unwrap();

        assert_eq!(index.vector_count().unwrap(), 0);
        assert_eq!(index.entry_count().unwrap(), 0);
        assert!(index.get_meta("last_ingest").unwrap().is_none());
        assert!(index.all_vectors().unwrap().is_empty());
        assert!(index.chunk_entry_id("c1").unwrap().is_none());
    }

    #[test]
    fn metadata_set_and_get() {
        let (index, _dir) = open_temp_index();

        // Missing key returns None.
        assert!(index.get_meta("missing").unwrap().is_none());

        // Set and retrieve.
        index.set_meta("last_ingest", "2026-04-15T12:00:00Z").unwrap();
        assert_eq!(
            index.get_meta("last_ingest").unwrap(),
            Some("2026-04-15T12:00:00Z".to_string())
        );

        // Overwrite.
        index.set_meta("last_ingest", "2026-04-15T13:00:00Z").unwrap();
        assert_eq!(
            index.get_meta("last_ingest").unwrap(),
            Some("2026-04-15T13:00:00Z".to_string())
        );

        // Multiple keys.
        index.set_meta("entry_count", "42").unwrap();
        assert_eq!(
            index.get_meta("entry_count").unwrap(),
            Some("42".to_string())
        );
        // Previous key still there.
        assert!(index.get_meta("last_ingest").unwrap().is_some());
    }

    #[test]
    fn vector_roundtrip_precision() {
        let (index, _dir) = open_temp_index();

        // Create a 768-dimensional vector (nomic-embed-text dimensionality).
        let vector: Vec<f32> = (0..768)
            .map(|i| {
                // Use a variety of float values: small, large, negative, fractional.
                let x = i as f32;
                (x * 0.00123 - 0.5).sin() * 1.234e-5 + x * 0.001
            })
            .collect();

        index.put_vector("chunk-768", "entry-768", &vector).unwrap();

        let all = index.all_vectors().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "chunk-768");
        assert_eq!(all[0].1.len(), 768);

        // Verify exact bit-for-bit precision (f32 -> bytes -> f32 is lossless).
        for (i, (&original, &retrieved)) in vector.iter().zip(all[0].1.iter()).enumerate() {
            assert_eq!(
                original.to_bits(),
                retrieved.to_bits(),
                "precision loss at dimension {i}: {original} != {retrieved}"
            );
        }
    }
}
