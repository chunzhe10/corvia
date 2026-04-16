//! Write pipeline: validate -> deduplicate -> store -> index.
//!
//! Creates a new knowledge entry, optionally detecting near-duplicate content
//! via cosine similarity and automatically superseding the matched entry.
//!
//! Steps:
//! 1. Resolve paths and ensure directories exist
//! 2. Open RedbIndex and TantivyIndex
//! 3. Auto-dedup check (if no explicit supersedes and content is non-empty)
//! 4. Validate supersedes references
//! 5. Create and write entry atomically
//! 6. Update indexes (Redb vectors + Tantivy BM25)
//! 7. Update entry count metadata
//! 8. Return WriteResponse

use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, info_span, warn, Span};

use crate::chunk::chunk_entry;
use crate::config::Config;
use crate::embed::Embedder;
use crate::entry::{new_entry, write_entry_atomic};
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::{Kind, WriteResponse};

// ---------------------------------------------------------------------------
// WriteParams
// ---------------------------------------------------------------------------

/// Parameters for writing a new knowledge entry.
pub struct WriteParams {
    /// The body content of the entry.
    pub content: String,
    /// Classification of the entry.
    pub kind: Kind,
    /// Tags for the entry.
    pub tags: Vec<String>,
    /// Explicit list of entry IDs this new entry supersedes.
    /// If empty and content is non-empty, auto-dedup will attempt to find
    /// a near-duplicate to supersede.
    pub supersedes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Auto-dedup logic
// ---------------------------------------------------------------------------

/// Result of the auto-dedup check.
struct DedupResult {
    /// The entry ID that was matched, if any.
    matched_id: Option<String>,
    /// The cosine similarity score of the match.
    similarity: f32,
}

/// Check incoming content against all non-superseded vectors in the index.
///
/// Returns the best match above the dedup threshold, or None.
fn auto_dedup_check(
    embedder: &Embedder,
    redb: &RedbIndex,
    content: &str,
    threshold: f32,
) -> Result<DedupResult> {
    // Embed the incoming content.
    let incoming_vec = embedder
        .embed(content)
        .context("embedding content for dedup check")?;

    // Get all stored vectors and the set of superseded entry IDs.
    let all_vectors = redb.all_vectors().context("loading vectors for dedup")?;
    let superseded_ids = redb.superseded_ids().context("loading superseded IDs for dedup")?;

    let mut best_match: Option<(String, f32)> = None;

    for (chunk_id, stored_vec) in &all_vectors {
        // Resolve chunk_id to entry_id.
        let entry_id = match redb.chunk_entry_id(chunk_id)? {
            Some(eid) => eid,
            None => continue,
        };

        // Skip superseded entries.
        if superseded_ids.contains(&entry_id) {
            continue;
        }

        let sim = Embedder::cosine_similarity(&incoming_vec, stored_vec);

        if sim >= threshold {
            // Track the highest similarity match.
            let dominated = match &best_match {
                Some((_, best_sim)) => sim > *best_sim,
                None => true,
            };
            if dominated {
                best_match = Some((entry_id, sim));
            }
        }
    }

    match best_match {
        Some((id, sim)) => Ok(DedupResult {
            matched_id: Some(id),
            similarity: sim,
        }),
        None => Ok(DedupResult {
            matched_id: None,
            similarity: 0.0,
        }),
    }
}

/// Validate supersedes references and build a warning string for missing entries.
fn validate_supersedes(redb: &RedbIndex, supersedes: &[String]) -> Result<Option<String>> {
    let mut missing: Vec<String> = Vec::new();

    for id in supersedes {
        if !redb.entry_exists(id).context("checking entry existence")? {
            missing.push(id.clone());
        }
    }

    if missing.is_empty() {
        Ok(None)
    } else {
        let warning = missing
            .iter()
            .map(|id| format!("superseded entry '{}' not found", id))
            .collect::<Vec<_>>()
            .join("; ");
        Ok(Some(warning))
    }
}

// ---------------------------------------------------------------------------
// Write pipeline
// ---------------------------------------------------------------------------

/// Write a new knowledge entry using pre-opened index handles.
///
/// Callers must ensure the entries directory and index directory already exist.
/// For one-shot callers, use [`write`] which creates directories and opens handles.
#[tracing::instrument(name = "corvia.write", skip(config, base_dir, embedder, params, redb, tantivy), fields(
    kind = %params.kind,
    content_len = params.content.len(),
    action = tracing::field::Empty,
    superseded_count = tracing::field::Empty,
))]
pub fn write_with_handles(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: WriteParams,
    redb: &RedbIndex,
    tantivy: &TantivyIndex,
) -> Result<WriteResponse> {
    let entries_dir = base_dir.join(config.entries_dir());

    // Step 3: Determine supersedes list and action.
    let caller_provided_supersedes = !params.supersedes.is_empty();
    let mut supersedes = params.supersedes;
    let action: String;
    let mut dedup_similarity: Option<f32> = None;

    if !caller_provided_supersedes && !params.content.is_empty() {
        let _dedup_span = info_span!("corvia.write.dedup",
            threshold = config.search.dedup_threshold,
            matched = tracing::field::Empty,
            similarity = tracing::field::Empty,
        ).entered();

        let dedup = auto_dedup_check(embedder, redb, &params.content, config.search.dedup_threshold)
            .context("auto-dedup check")?;

        if let Some(matched_id) = dedup.matched_id {
            info!(
                matched_id = %matched_id,
                similarity = dedup.similarity,
                "auto-dedup: superseding existing entry"
            );
            Span::current().record("matched", true);
            Span::current().record("similarity", dedup.similarity as f64);
            dedup_similarity = Some(dedup.similarity);
            supersedes = vec![matched_id];
            action = "superseded".to_string();
        } else {
            Span::current().record("matched", false);
            Span::current().record("similarity", 0.0f64);
            action = "created".to_string();
        }
        drop(_dedup_span);
    } else if !supersedes.is_empty() {
        action = "superseded".to_string();
    } else {
        action = "created".to_string();
    }

    // Step 4: Validate supersedes references.
    let warning = validate_supersedes(redb, &supersedes)?;
    if let Some(ref w) = warning {
        warn!("{}", w);
    }

    // Step 5: Create entry and write atomically.
    let entry = new_entry(
        params.content,
        params.kind,
        params.tags,
        supersedes.clone(),
    );
    let entry_id = entry.meta.id.clone();

    write_entry_atomic(&entries_dir, &entry)
        .with_context(|| format!("writing entry {}", entry_id))?;

    info!(id = %entry_id, action = %action, "entry written to disk");

    // Step 6: Update indexes.
    // 6a: Mark superseded entries in Redb.
    for sup_id in &supersedes {
        redb.set_superseded(sup_id, true)
            .with_context(|| format!("marking {} as superseded", sup_id))?;
    }

    // 6b: Mark the new entry as current.
    redb.set_superseded(&entry_id, false)
        .with_context(|| format!("marking {} as current", entry_id))?;

    // 6b2: Delete superseded entries from Tantivy.
    {
        let mut sup_writer = tantivy.writer().context("creating tantivy writer for supersession cleanup")?;
        for sup_id in &supersedes {
            tantivy.delete_by_entry_id(&sup_writer, sup_id);
        }
        sup_writer.commit().context("committing supersession deletes")?;
        tantivy
            .reload_reader()
            .context("reloading tantivy reader after supersession deletes")?;
    }

    // 6c-6e: Chunk, embed, index.
    {
        let _span = info_span!("corvia.write.index", chunk_count = tracing::field::Empty).entered();

        let chunks = chunk_entry(
            &entry,
            config.chunking.max_tokens,
            config.chunking.overlap_tokens,
            config.chunking.min_tokens,
        );
        Span::current().record("chunk_count", chunks.len());

        let mut writer = tantivy.writer().context("creating tantivy writer")?;

        for chunk in &chunks {
            let chunk_id = format!("{}:{}", entry_id, chunk.chunk_index);
            if chunk.text.is_empty() {
                continue;
            }
            let vector = embedder
                .embed(&chunk.text)
                .with_context(|| format!("embedding chunk {}", chunk_id))?;
            redb.put_vector(&chunk_id, &entry_id, &vector)
                .with_context(|| format!("storing vector for {}", chunk_id))?;
            redb.put_chunk_kind(&chunk_id, &chunk.kind.to_string())
                .with_context(|| format!("storing kind for {}", chunk_id))?;
            tantivy
                .add_doc(
                    &writer,
                    &chunk_id,
                    &entry_id,
                    &chunk.text,
                    entry.meta.kind,
                    false,
                )
                .with_context(|| format!("adding tantivy doc for {}", chunk_id))?;
        }

        writer.commit().context("committing tantivy writer")?;
        tantivy
            .reload_reader()
            .context("reloading tantivy reader after write")?;
    }

    // Step 7: Update entry count in Redb meta.
    let actual_count = crate::entry::scan_entries(&entries_dir)
        .context("scanning entries for count update")?
        .len();
    redb.set_meta("entry_count", &actual_count.to_string())
        .context("updating entry_count metadata")?;

    Span::current().record("action", action.as_str());
    Span::current().record("superseded_count", supersedes.len());

    info!(
        id = %entry_id,
        action = %action,
        superseded_count = supersedes.len(),
        "write pipeline complete"
    );

    Ok(WriteResponse {
        id: entry_id,
        action,
        superseded: supersedes,
        similarity: dedup_similarity,
        warning,
    })
}

/// Write a new knowledge entry with auto-dedup detection.
///
/// This is the primary write path for creating knowledge entries. It handles:
/// - Automatic near-duplicate detection via cosine similarity
/// - Explicit supersession when the caller provides `supersedes` IDs
/// - Atomic file writes
/// - Index updates (Redb vectors + Tantivy BM25)
/// - Entry count metadata updates
/// Write a new knowledge entry with auto-dedup detection.
///
/// Opens index handles internally. For callers that hold persistent handles,
/// use [`write_with_handles`] directly.
pub fn write(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: WriteParams,
) -> Result<WriteResponse> {
    // Step 1: Resolve paths and ensure directories exist.
    let entries_dir = base_dir.join(config.entries_dir());
    let index_dir = base_dir.join(config.index_dir());
    std::fs::create_dir_all(&entries_dir)
        .with_context(|| format!("creating entries dir: {}", entries_dir.display()))?;
    std::fs::create_dir_all(&index_dir)
        .with_context(|| format!("creating index dir: {}", index_dir.display()))?;

    // Step 2: Open indexes.
    let redb = RedbIndex::open(&base_dir.join(config.redb_path())).context("opening redb index")?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir())).context("opening tantivy index")?;
    write_with_handles(config, base_dir, embedder, params, &redb, &tantivy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_supersedes_all_missing() {
        // Open a fresh Redb with no entries.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        let ids = vec!["ghost-1".to_string(), "ghost-2".to_string()];
        let warning = validate_supersedes(&redb, &ids).unwrap();

        assert!(warning.is_some(), "should produce a warning");
        let w = warning.unwrap();
        assert!(
            w.contains("ghost-1"),
            "warning should mention ghost-1: {w}"
        );
        assert!(
            w.contains("ghost-2"),
            "warning should mention ghost-2: {w}"
        );
    }

    #[test]
    fn validate_supersedes_some_exist() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        // Insert one entry so it exists.
        redb.set_superseded("existing-1", false).unwrap();

        let ids = vec!["existing-1".to_string(), "ghost-1".to_string()];
        let warning = validate_supersedes(&redb, &ids).unwrap();

        assert!(warning.is_some(), "should produce a warning for ghost-1");
        let w = warning.unwrap();
        assert!(
            !w.contains("existing-1"),
            "should not mention existing-1: {w}"
        );
        assert!(w.contains("ghost-1"), "should mention ghost-1: {w}");
    }

    #[test]
    fn validate_supersedes_all_exist() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        redb.set_superseded("entry-a", false).unwrap();
        redb.set_superseded("entry-b", true).unwrap();

        let ids = vec!["entry-a".to_string(), "entry-b".to_string()];
        let warning = validate_supersedes(&redb, &ids).unwrap();

        assert!(warning.is_none(), "should produce no warning");
    }

    #[test]
    fn validate_supersedes_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        let warning = validate_supersedes(&redb, &[]).unwrap();
        assert!(warning.is_none(), "empty supersedes should produce no warning");
    }

    #[test]
    fn write_params_default_kind() {
        let params = WriteParams {
            content: "test content".to_string(),
            kind: Kind::default(),
            tags: vec![],
            supersedes: vec![],
        };
        assert_eq!(params.kind, Kind::Learning);
    }

    #[test]
    fn warning_message_format() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        let ids = vec!["abc-123".to_string()];
        let warning = validate_supersedes(&redb, &ids).unwrap().unwrap();

        assert_eq!(
            warning,
            "superseded entry 'abc-123' not found",
            "single missing entry should produce exact message"
        );
    }

    #[test]
    fn warning_message_multiple() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let redb = RedbIndex::open(&db_path).unwrap();

        let ids = vec!["id-1".to_string(), "id-2".to_string(), "id-3".to_string()];
        let warning = validate_supersedes(&redb, &ids).unwrap().unwrap();

        assert_eq!(
            warning,
            "superseded entry 'id-1' not found; superseded entry 'id-2' not found; superseded entry 'id-3' not found"
        );
    }

    // Integration-level tests that require the embedding model.
    // Run with: cargo test -p corvia-core -- --ignored

    #[test]
    #[ignore]
    fn write_creates_entry_and_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path();
        let config = Config::default();

        // Create the required directories.
        std::fs::create_dir_all(base_dir.join(config.entries_dir())).unwrap();
        std::fs::create_dir_all(base_dir.join(config.index_dir())).unwrap();

        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");

        let params = WriteParams {
            content: "Rust is a systems programming language focused on safety.".to_string(),
            kind: Kind::Learning,
            tags: vec!["rust".to_string(), "programming".to_string()],
            supersedes: vec![],
        };

        let response = write(&config, base_dir, &embedder, params).unwrap();

        assert!(!response.id.is_empty(), "should return an entry ID");
        assert_eq!(response.action, "created");
        assert!(response.superseded.is_empty());
        assert!(response.warning.is_none());

        // Verify the entry file was created.
        let entry_path = base_dir
            .join(config.entries_dir())
            .join(format!("{}.md", response.id));
        assert!(entry_path.exists(), "entry file should exist on disk");

        // Verify indexes were updated.
        let redb = RedbIndex::open(&base_dir.join(config.redb_path())).unwrap();
        assert!(
            redb.entry_exists(&response.id).unwrap(),
            "entry should exist in Redb"
        );
        assert!(
            !redb.is_superseded(&response.id).unwrap(),
            "new entry should not be superseded"
        );
        assert!(
            redb.vector_count().unwrap() > 0,
            "should have at least one vector"
        );
    }

    #[test]
    #[ignore]
    fn write_with_explicit_supersedes() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path();
        let config = Config::default();

        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");

        // Write the first entry.
        let first = write(
            &config,
            base_dir,
            &embedder,
            WriteParams {
                content: "Original content about Rust memory safety.".to_string(),
                kind: Kind::Learning,
                tags: vec![],
                supersedes: vec![],
            },
        )
        .unwrap();

        // Write a second entry that explicitly supersedes the first.
        let second = write(
            &config,
            base_dir,
            &embedder,
            WriteParams {
                content: "Updated content about Rust memory safety and ownership.".to_string(),
                kind: Kind::Learning,
                tags: vec![],
                supersedes: vec![first.id.clone()],
            },
        )
        .unwrap();

        assert_eq!(second.action, "superseded");
        assert_eq!(second.superseded, vec![first.id.clone()]);
        assert!(second.warning.is_none());

        // Verify the first entry is now superseded in Redb.
        let redb = RedbIndex::open(&base_dir.join(config.redb_path())).unwrap();
        assert!(
            redb.is_superseded(&first.id).unwrap(),
            "first entry should be superseded"
        );
        assert!(
            !redb.is_superseded(&second.id).unwrap(),
            "second entry should be current"
        );
    }

    #[test]
    fn write_with_handles_signature_exists() {
        let _fn: fn(
            &crate::config::Config,
            &std::path::Path,
            &crate::embed::Embedder,
            WriteParams,
            &crate::index::RedbIndex,
            &crate::tantivy_index::TantivyIndex,
        ) -> anyhow::Result<crate::types::WriteResponse> = write_with_handles;
        let _ = _fn;
    }

    #[test]
    #[ignore]
    fn write_auto_dedup_detects_near_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path();
        let config = Config::default();

        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");

        // Write an entry.
        let first = write(
            &config,
            base_dir,
            &embedder,
            WriteParams {
                content: "Rust is a systems programming language focused on safety and concurrency.".to_string(),
                kind: Kind::Learning,
                tags: vec![],
                supersedes: vec![],
            },
        )
        .unwrap();

        assert_eq!(first.action, "created");

        // Write a near-duplicate (very similar content, no explicit supersedes).
        let second = write(
            &config,
            base_dir,
            &embedder,
            WriteParams {
                content: "Rust is a systems programming language focused on safety and concurrency.".to_string(),
                kind: Kind::Learning,
                tags: vec![],
                supersedes: vec![],
            },
        )
        .unwrap();

        // Auto-dedup should have detected the near-duplicate.
        assert_eq!(
            second.action, "superseded",
            "identical content should trigger auto-dedup"
        );
        assert_eq!(second.superseded, vec![first.id.clone()]);
    }
}
