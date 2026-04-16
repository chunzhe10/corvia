//! Entry persistence (read, write, delete).
//!
//! Entries are stored as markdown files with TOML frontmatter delimited by `+++`.
//! Atomic writes use a temp-file-then-rename strategy to avoid partial reads.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};

use crate::types::{Entry, EntryMeta, Kind, new_entry_id};

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

/// Returns the current time as an ISO 8601 string: `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Uses `std::time::SystemTime` (no chrono dependency).
pub fn now_iso8601() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch");

    let secs = dur.as_secs();

    // Break epoch seconds into calendar components.
    // Algorithm: civil_from_days (Howard Hinnant, public domain).
    let days = (secs / 86400) as i64;
    let day_secs = (secs % 86400) as u32;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;

    // days since 1970-01-01 -> civil date
    let z = days + 719468; // shift epoch from 1970-01-01 to 0000-03-01
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// ---------------------------------------------------------------------------
// Entry construction
// ---------------------------------------------------------------------------

/// Create a new [`Entry`] with a generated UUIDv7 id and the current timestamp.
pub fn new_entry(
    body: String,
    kind: Kind,
    tags: Vec<String>,
    supersedes: Vec<String>,
) -> Entry {
    Entry {
        meta: EntryMeta {
            id: new_entry_id(),
            created_at: now_iso8601(),
            kind,
            supersedes,
            tags,
        },
        body,
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

const DELIMITER: &str = "+++";

/// Serialize an [`Entry`] to TOML-frontmatter + markdown body.
///
/// ```text
/// +++
/// id = "..."
/// created_at = "..."
/// kind = "..."
/// supersedes = ["..."]
/// tags = ["..."]
/// +++
///
/// body text here
/// ```
pub fn serialize_entry(entry: &Entry) -> Result<String> {
    let toml = toml::to_string_pretty(&entry.meta)
        .context("failed to serialize entry meta to TOML")?;

    let mut out = String::new();
    out.push_str(DELIMITER);
    out.push('\n');
    out.push_str(toml.trim_end());
    out.push('\n');
    out.push_str(DELIMITER);
    out.push('\n');

    if !entry.body.is_empty() {
        out.push('\n');
        out.push_str(&entry.body);
        out.push('\n');
    }

    Ok(out)
}

/// Parse file content (TOML frontmatter + body) back into an [`Entry`].
pub fn parse_entry(content: &str) -> Result<Entry> {
    // Find opening delimiter.
    let content = content.trim_start_matches('\u{feff}'); // strip optional BOM
    let rest = content
        .strip_prefix(DELIMITER)
        .with_context(|| "missing opening +++ delimiter")?;

    // Find closing delimiter.
    let (toml_block, after) = rest
        .split_once(&format!("\n{DELIMITER}"))
        .with_context(|| "missing closing +++ delimiter")?;

    let meta: EntryMeta = toml::from_str(toml_block.trim())
        .context("failed to parse entry TOML frontmatter")?;

    if meta.id.is_empty() {
        bail!("entry id must not be empty");
    }

    // Body is everything after the closing delimiter line, trimmed.
    let body = after
        .strip_prefix('\n')
        .unwrap_or(after)
        .trim()
        .to_string();

    Ok(Entry { meta, body })
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Read an entry from a file on disk.
pub fn read_entry(path: &Path) -> Result<Entry> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read entry file: {}", path.display()))?;
    parse_entry(&content)
}

/// Atomically write an entry to `<entries_dir>/<id>.md`.
///
/// Strategy: write to a hidden temp file first (`.< id>.md.tmp`), then rename.
/// Returns the final path on success.
pub fn write_entry_atomic(entries_dir: &Path, entry: &Entry) -> Result<PathBuf> {
    fs::create_dir_all(entries_dir)
        .with_context(|| format!("failed to create entries dir: {}", entries_dir.display()))?;

    let final_name = format!("{}.md", entry.meta.id);
    let tmp_name = format!(".{}.md.tmp", entry.meta.id);

    let final_path = entries_dir.join(&final_name);
    let tmp_path = entries_dir.join(&tmp_name);

    let serialized = serialize_entry(entry)?;

    fs::write(&tmp_path, &serialized)
        .with_context(|| format!("failed to write temp file: {}", tmp_path.display()))?;

    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("failed to rename temp file to: {}", final_path.display()))?;

    Ok(final_path)
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

/// List all `.md` entry files in a directory, skipping hidden files.
///
/// Returns an empty vec if the directory does not exist.
/// Results are sorted by filename.
pub fn scan_entries(entries_dir: &Path) -> Result<Vec<PathBuf>> {
    if !entries_dir.exists() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();

    for dir_entry in fs::read_dir(entries_dir)
        .with_context(|| format!("failed to read entries dir: {}", entries_dir.display()))?
    {
        let dir_entry = dir_entry?;
        let file_name = dir_entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden files (starting with '.')
        if name.starts_with('.') {
            continue;
        }

        // Only include .md files
        if name.ends_with(".md") {
            paths.push(dir_entry.path());
        }
    }

    paths.sort();
    Ok(paths)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Kind;

    /// Helper: build an entry with all fields populated.
    fn sample_entry() -> Entry {
        Entry {
            meta: EntryMeta {
                id: "01961234-5678-7abc-def0-123456789abc".to_string(),
                created_at: "2026-04-15T12:30:00Z".to_string(),
                kind: Kind::Decision,
                supersedes: vec!["old-entry-1".to_string(), "old-entry-2".to_string()],
                tags: vec!["architecture".to_string(), "v2".to_string()],
            },
            body: "This is the body of the entry.\n\nIt has multiple paragraphs.".to_string(),
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let entry = sample_entry();
        let serialized = serialize_entry(&entry).unwrap();
        let parsed = parse_entry(&serialized).unwrap();

        assert_eq!(parsed.meta.id, entry.meta.id);
        assert_eq!(parsed.meta.created_at, entry.meta.created_at);
        assert_eq!(parsed.meta.kind, entry.meta.kind);
        assert_eq!(parsed.meta.supersedes, entry.meta.supersedes);
        assert_eq!(parsed.meta.tags, entry.meta.tags);
        assert_eq!(parsed.body, entry.body);
    }

    #[test]
    fn parse_missing_delimiter_errors() {
        let bad = "id = \"abc\"\nbody text";
        let result = parse_entry(bad);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("delimiter"), "error should mention delimiter: {msg}");
    }

    #[test]
    fn parse_missing_id_errors() {
        let content = "+++\nkind = \"learning\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n+++\nbody";
        let result = parse_entry(content);
        // toml::from_str should fail because `id` is a required String field
        // with no default, or our empty-id check catches it.
        assert!(result.is_err(), "should error on missing id");
    }

    #[test]
    fn parse_empty_body() {
        let content = "+++\nid = \"abc\"\ncreated_at = \"2026-01-01T00:00:00Z\"\nkind = \"learning\"\n+++\n";
        let entry = parse_entry(content).unwrap();
        assert_eq!(entry.meta.id, "abc");
        assert!(entry.body.is_empty(), "body should be empty");
    }

    #[test]
    fn atomic_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");

        let entry = sample_entry();
        let path = write_entry_atomic(&entries_dir, &entry).unwrap();

        assert!(path.exists(), "file should exist after atomic write");

        let read_back = read_entry(&path).unwrap();
        assert_eq!(read_back.meta.id, entry.meta.id);
        assert_eq!(read_back.meta.created_at, entry.meta.created_at);
        assert_eq!(read_back.meta.kind, entry.meta.kind);
        assert_eq!(read_back.meta.supersedes, entry.meta.supersedes);
        assert_eq!(read_back.meta.tags, entry.meta.tags);
        assert_eq!(read_back.body, entry.body);
    }

    #[test]
    fn atomic_write_no_tmp_left() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");

        let entry = sample_entry();
        write_entry_atomic(&entries_dir, &entry).unwrap();

        // Check no .tmp files remain
        let tmp_files: Vec<_> = fs::read_dir(&entries_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();

        assert!(tmp_files.is_empty(), "no .tmp files should remain after write");
    }

    #[test]
    fn scan_entries_skips_hidden_files() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");
        fs::create_dir_all(&entries_dir).unwrap();

        // Create a visible entry file
        fs::write(entries_dir.join("visible.md"), "+++\nid = \"v\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n+++\n").unwrap();

        // Create a hidden temp file
        fs::write(entries_dir.join(".hidden.md.tmp"), "temp data").unwrap();

        let paths = scan_entries(&entries_dir).unwrap();
        assert_eq!(paths.len(), 1, "should only return visible .md files");
        assert!(
            paths[0].file_name().unwrap().to_string_lossy() == "visible.md",
            "should return visible.md"
        );
    }

    #[test]
    fn scan_entries_nonexistent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");

        let paths = scan_entries(&missing).unwrap();
        assert!(paths.is_empty(), "nonexistent dir should return empty vec");
    }

    #[test]
    fn tags_with_special_chars_roundtrip() {
        let entry = Entry {
            meta: EntryMeta {
                id: "special-chars-test".to_string(),
                created_at: "2026-04-15T00:00:00Z".to_string(),
                kind: Kind::Learning,
                supersedes: vec![],
                tags: vec![
                    "key=value".to_string(),
                    "has \"quotes\"".to_string(),
                    "has [brackets]".to_string(),
                    "normal-tag".to_string(),
                ],
            },
            body: "body".to_string(),
        };

        let serialized = serialize_entry(&entry).unwrap();
        let parsed = parse_entry(&serialized).unwrap();

        assert_eq!(parsed.meta.tags, entry.meta.tags);
    }

    #[test]
    fn now_iso8601_format() {
        let ts = now_iso8601();
        // Should match YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20, "ISO 8601 timestamp should be 20 chars: {ts}");
        assert!(ts.ends_with('Z'), "should end with Z: {ts}");
        assert_eq!(&ts[4..5], "-", "dash at position 4");
        assert_eq!(&ts[7..8], "-", "dash at position 7");
        assert_eq!(&ts[10..11], "T", "T at position 10");
        assert_eq!(&ts[13..14], ":", "colon at position 13");
        assert_eq!(&ts[16..17], ":", "colon at position 16");
    }

    #[test]
    fn new_entry_generates_id_and_timestamp() {
        let entry = new_entry(
            "test body".to_string(),
            Kind::Decision,
            vec!["tag1".to_string()],
            vec![],
        );

        assert!(!entry.meta.id.is_empty(), "id should be set");
        assert_eq!(entry.meta.id.len(), 36, "id should be a UUID");
        assert!(!entry.meta.created_at.is_empty(), "created_at should be set");
        assert_eq!(entry.meta.kind, Kind::Decision);
        assert_eq!(entry.meta.tags, vec!["tag1"]);
        assert!(entry.meta.supersedes.is_empty());
        assert_eq!(entry.body, "test body");
    }
}
