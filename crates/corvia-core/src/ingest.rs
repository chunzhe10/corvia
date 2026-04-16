//! Document ingestion pipeline: read -> chunk -> embed -> index.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, info_span, warn, Span};

use crate::chunk::chunk_entry;
use crate::config::Config;
use crate::embed::Embedder;
use crate::entry::{now_iso8601, read_entry, scan_entries};
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::Entry;

// ---------------------------------------------------------------------------
// IngestResult
// ---------------------------------------------------------------------------

/// Summary of an ingest run.
pub struct IngestResult {
    pub entries_ingested: usize,
    pub chunks_indexed: usize,
    pub entries_skipped: Vec<(String, String)>, // (filename, reason)
    pub superseded_count: usize,
}

// ---------------------------------------------------------------------------
// Supersession logic
// ---------------------------------------------------------------------------

/// Build the set of entry IDs that are superseded by other entries.
///
/// Rules:
/// - Any ID listed in an entry's `supersedes` field is considered superseded.
/// - Circular supersession (A supersedes B and B supersedes A) is resolved by
///   keeping the entry with the later `created_at` timestamp as current.
///   The entry with the earlier timestamp is the one that gets superseded.
pub fn build_superseded_set(entries: &[Entry]) -> HashSet<String> {
    // Map from entry ID to its created_at timestamp for circular resolution.
    let mut created_at_map: HashMap<&str, &str> = HashMap::new();
    for entry in entries {
        created_at_map.insert(&entry.meta.id, &entry.meta.created_at);
    }

    // Collect all IDs that appear in any entry's supersedes field.
    let mut superseded: HashSet<String> = HashSet::new();
    for entry in entries {
        for sid in &entry.meta.supersedes {
            superseded.insert(sid.clone());
        }
    }

    // Detect and resolve circular supersession.
    // If A supersedes B and B supersedes A, the one with the later created_at
    // wins (stays current); the other stays in the superseded set.
    let entry_ids: HashSet<&str> = entries.iter().map(|e| e.meta.id.as_str()).collect();

    for entry in entries {
        if !superseded.contains(&entry.meta.id) {
            continue;
        }
        // This entry is marked superseded. Check if it also supersedes any of
        // the entries that supersede it (circular).
        for sid in &entry.meta.supersedes {
            if !entry_ids.contains(sid.as_str()) {
                continue;
            }
            // entry supersedes sid, and entry is itself superseded.
            // Check if sid is one of the entries that supersedes entry.
            let sid_entry = entries.iter().find(|e| e.meta.id == *sid);
            if let Some(other) = sid_entry {
                let other_supersedes_us = other.meta.supersedes.contains(&entry.meta.id);
                if other_supersedes_us {
                    // Circular: entry <-> other. The one with the later
                    // created_at wins (remove it from superseded set).
                    let our_ts = created_at_map.get(entry.meta.id.as_str()).unwrap_or(&"");
                    let their_ts = created_at_map.get(sid.as_str()).unwrap_or(&"");
                    if our_ts >= their_ts {
                        // We are newer or equal: we win (remove us from superseded).
                        superseded.remove(&entry.meta.id);
                    } else {
                        // They are newer: they win (remove them from superseded).
                        superseded.remove(sid);
                    }
                }
            }
        }
    }

    superseded
}

// ---------------------------------------------------------------------------
// Ingest pipeline
// ---------------------------------------------------------------------------

/// Run the full ingest pipeline: scan entries, parse, chunk, embed, and index.
///
/// Steps:
/// 1. Resolve paths from config
/// 2. Create directories if needed
/// 3. Open Redb and Tantivy indexes
/// 4. If `fresh`: clear both indexes
/// 5. Initialize embedder
/// 6. Scan and parse entry files
/// 7. Build supersession set
/// 8. For each entry: chunk, embed, store vectors, add to Tantivy
/// 9. Commit Tantivy writer
/// 10. Store metadata in Redb
/// 11. Return summary
#[tracing::instrument(name = "corvia.ingest", skip(config, base_dir), fields(
    fresh,
    entries_ingested = tracing::field::Empty,
    chunks_indexed = tracing::field::Empty,
    superseded_count = tracing::field::Empty,
))]
pub fn ingest(config: &Config, base_dir: &Path, fresh: bool) -> Result<IngestResult> {
    let entries_dir = base_dir.join(config.entries_dir());
    let index_dir = base_dir.join(config.index_dir());

    // Step 2: Create directories if they don't exist.
    std::fs::create_dir_all(&entries_dir)
        .with_context(|| format!("creating entries dir: {}", entries_dir.display()))?;
    std::fs::create_dir_all(&index_dir)
        .with_context(|| format!("creating index dir: {}", index_dir.display()))?;

    // Step 3: Open indexes.
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))
        .context("opening redb index")?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))
        .context("opening tantivy index")?;

    // Step 4: If fresh, clear both indexes.
    if fresh {
        info!("fresh ingest: clearing existing indexes");
        redb.clear_all().context("clearing redb index")?;
        tantivy.clear().context("clearing tantivy index")?;
    }

    // Step 5: Initialize embedder.
    let cache_dir = config.embedding.model_path.as_deref();
    let embedder = Embedder::new(cache_dir, &config.embedding.model, &config.embedding.reranker_model)
        .context("initializing embedder")?;

    // Step 6: Scan and parse entries.
    let (entries, skipped) = {
        let _span = info_span!("corvia.ingest.parse", file_count = tracing::field::Empty, skipped_count = tracing::field::Empty).entered();

        let entry_paths = scan_entries(&entries_dir).context("scanning entries")?;
        Span::current().record("file_count", entry_paths.len());
        info!(count = entry_paths.len(), "found entry files");

        let mut entries: Vec<Entry> = Vec::new();
        let mut skipped: Vec<(String, String)> = Vec::new();

        for path in &entry_paths {
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());

            match read_entry(path) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    warn!(file = %filename, error = %e, "skipping unparseable entry");
                    skipped.push((filename, format!("{e}")));
                }
            }
        }

        Span::current().record("skipped_count", skipped.len());
        (entries, skipped)
    };

    // Step 7: Build supersession set.
    let superseded_set = build_superseded_set(&entries);
    let superseded_count = superseded_set.len();
    info!(
        total = entries.len(),
        superseded = superseded_count,
        "parsed entries"
    );

    // Step 8: Process each entry (chunk, embed, index).
    let chunks_indexed = {
        let _span = info_span!("corvia.ingest.embed", chunk_count = tracing::field::Empty).entered();

        let mut writer = tantivy.writer().context("creating tantivy writer")?;
        let mut count: usize = 0;

        for entry in &entries {
            let is_superseded = superseded_set.contains(&entry.meta.id);

            // 8a: Mark superseded or current in Redb.
            redb.set_superseded(&entry.meta.id, is_superseded)
                .with_context(|| format!("setting supersession for {}", entry.meta.id))?;

            // 8b: Chunk the entry.
            let chunks = chunk_entry(
                entry,
                config.chunking.max_tokens,
                config.chunking.overlap_tokens,
                config.chunking.min_tokens,
            );

            for chunk in &chunks {
                // 8c: Generate chunk_id.
                let chunk_id = format!("{}:{}", entry.meta.id, chunk.chunk_index);

                // 8d: Skip embedding if chunk text is empty.
                if chunk.text.is_empty() {
                    continue;
                }

                // Embed chunk text.
                let vector = embedder
                    .embed(&chunk.text)
                    .with_context(|| format!("embedding chunk {chunk_id}"))?;

                // 8e: Store vector and kind in Redb.
                redb.put_vector(&chunk_id, &entry.meta.id, &vector)
                    .with_context(|| format!("storing vector for {chunk_id}"))?;
                redb.put_chunk_kind(&chunk_id, &chunk.kind.to_string())
                    .with_context(|| format!("storing kind for {chunk_id}"))?;

                // 8f: Add document to Tantivy.
                tantivy
                    .add_doc(
                        &writer,
                        &chunk_id,
                        &entry.meta.id,
                        &chunk.text,
                        entry.meta.kind,
                        is_superseded,
                    )
                    .with_context(|| format!("adding tantivy doc for {chunk_id}"))?;

                count += 1;
            }
        }

        // Step 9: Commit Tantivy writer.
        writer.commit().context("committing tantivy writer")?;
        tantivy
            .reload_reader()
            .context("reloading tantivy reader after ingest")?;

        Span::current().record("chunk_count", count);
        count
    };

    // Step 10: Store metadata in Redb.
    let timestamp = now_iso8601();
    redb.set_meta("last_ingest", &timestamp)
        .context("setting last_ingest metadata")?;
    redb.set_meta("entry_count", &entries.len().to_string())
        .context("setting entry_count metadata")?;

    // Step 11: Record on parent span and log summary.
    Span::current().record("entries_ingested", entries.len());
    Span::current().record("chunks_indexed", chunks_indexed);
    Span::current().record("superseded_count", superseded_count);

    info!(
        entries = entries.len(),
        chunks = chunks_indexed,
        skipped = skipped.len(),
        superseded = superseded_count,
        "ingest complete"
    );

    Ok(IngestResult {
        entries_ingested: entries.len(),
        chunks_indexed,
        entries_skipped: skipped,
        superseded_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntryMeta, Kind};

    /// Helper: build a minimal entry for testing supersession logic.
    fn entry(id: &str, created_at: &str, supersedes: Vec<&str>) -> Entry {
        Entry {
            meta: EntryMeta {
                id: id.to_string(),
                created_at: created_at.to_string(),
                kind: Kind::Learning,
                supersedes: supersedes.into_iter().map(String::from).collect(),
                tags: vec![],
            },
            body: String::new(),
        }
    }

    #[test]
    fn superseded_set_simple() {
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec![]),
            entry("B", "2026-01-02T00:00:00Z", vec!["A"]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(superseded.contains("A"), "A should be superseded by B");
        assert!(!superseded.contains("B"), "B should be current");
    }

    #[test]
    fn superseded_set_chain() {
        // A -> B -> C (C supersedes B, B supersedes A)
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec![]),
            entry("B", "2026-01-02T00:00:00Z", vec!["A"]),
            entry("C", "2026-01-03T00:00:00Z", vec!["B"]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(superseded.contains("A"));
        assert!(superseded.contains("B"));
        assert!(!superseded.contains("C"));
    }

    #[test]
    fn superseded_set_circular_later_wins() {
        // A supersedes B and B supersedes A. B is newer, so B wins.
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec!["B"]),
            entry("B", "2026-01-02T00:00:00Z", vec!["A"]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(
            superseded.contains("A"),
            "A should be superseded (older in circular)"
        );
        assert!(
            !superseded.contains("B"),
            "B should be current (newer in circular)"
        );
    }

    #[test]
    fn superseded_set_references_nonexistent() {
        // B supersedes "ghost" which doesn't exist as an entry.
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec![]),
            entry("B", "2026-01-02T00:00:00Z", vec!["ghost"]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(
            superseded.contains("ghost"),
            "ghost should be in superseded set even if not present as entry"
        );
        assert!(!superseded.contains("A"));
        assert!(!superseded.contains("B"));
    }

    #[test]
    fn superseded_set_empty_entries() {
        let entries: Vec<Entry> = vec![];
        let superseded = build_superseded_set(&entries);
        assert!(superseded.is_empty());
    }

    #[test]
    fn superseded_set_no_supersessions() {
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec![]),
            entry("B", "2026-01-02T00:00:00Z", vec![]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(superseded.is_empty());
    }

    #[test]
    fn superseded_set_multiple_supersede_same() {
        // Both B and C supersede A.
        let entries = vec![
            entry("A", "2026-01-01T00:00:00Z", vec![]),
            entry("B", "2026-01-02T00:00:00Z", vec!["A"]),
            entry("C", "2026-01-03T00:00:00Z", vec!["A"]),
        ];

        let superseded = build_superseded_set(&entries);
        assert!(superseded.contains("A"), "A should be superseded");
        assert!(!superseded.contains("B"));
        assert!(!superseded.contains("C"));
    }
}
