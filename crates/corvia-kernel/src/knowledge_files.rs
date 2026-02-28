use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::KnowledgeEntry;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Validate that a scope_id is safe to use in file paths.
/// Rejects path separators, `..`, and empty strings to prevent path traversal.
fn validate_scope_id(scope_id: &str) -> Result<()> {
    if scope_id.is_empty() {
        return Err(CorviaError::Validation("scope_id must not be empty".into()));
    }
    if scope_id.contains('/') || scope_id.contains('\\') || scope_id.contains("..") {
        return Err(CorviaError::Validation(format!(
            "scope_id contains invalid characters: {scope_id:?}"
        )));
    }
    Ok(())
}

fn entry_path(data_dir: &Path, scope_id: &str, entry_id: &Uuid) -> PathBuf {
    data_dir
        .join("knowledge")
        .join(scope_id)
        .join(format!("{}.json", entry_id))
}

fn scope_dir(data_dir: &Path, scope_id: &str) -> PathBuf {
    data_dir.join("knowledge").join(scope_id)
}

/// Write a KnowledgeEntry as a pretty-printed JSON file. Creates directories as needed.
pub fn write_entry(data_dir: &Path, entry: &KnowledgeEntry) -> Result<()> {
    validate_scope_id(&entry.scope_id)?;
    let path = entry_path(data_dir, &entry.scope_id, &entry.id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| CorviaError::Storage(format!("Failed to create dir: {e}")))?;
    }
    let json = serde_json::to_string_pretty(entry)
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize entry: {e}")))?;
    std::fs::write(&path, json)
        .map_err(|e| CorviaError::Storage(format!("Failed to write {}: {e}", path.display())))?;
    Ok(())
}

/// Read a single KnowledgeEntry from its JSON file.
pub fn read_entry(data_dir: &Path, scope_id: &str, entry_id: &Uuid) -> Result<KnowledgeEntry> {
    validate_scope_id(scope_id)?;
    let path = entry_path(data_dir, scope_id, entry_id);
    let json = std::fs::read_to_string(&path)
        .map_err(|e| CorviaError::NotFound(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&json)
        .map_err(|e| CorviaError::Storage(format!("Failed to parse {}: {e}", path.display())))
}

/// Read all KnowledgeEntries in a scope directory.
pub fn read_scope(data_dir: &Path, scope_id: &str) -> Result<Vec<KnowledgeEntry>> {
    validate_scope_id(scope_id)?;
    let dir = scope_dir(data_dir, scope_id);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for file in std::fs::read_dir(&dir)
        .map_err(|e| CorviaError::Storage(format!("Failed to read dir: {e}")))?
    {
        let file = file.map_err(|e| CorviaError::Storage(e.to_string()))?;
        let path = file.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let json = std::fs::read_to_string(&path).map_err(|e| {
                CorviaError::Storage(format!("Failed to read {}: {e}", path.display()))
            })?;
            if let Ok(entry) = serde_json::from_str::<KnowledgeEntry>(&json) {
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

/// Read all KnowledgeEntries across all scopes.
pub fn read_all(data_dir: &Path) -> Result<Vec<KnowledgeEntry>> {
    let knowledge_dir = data_dir.join("knowledge");
    if !knowledge_dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for scope_entry in std::fs::read_dir(&knowledge_dir)
        .map_err(|e| CorviaError::Storage(format!("Failed to read knowledge dir: {e}")))?
    {
        let scope_entry = scope_entry.map_err(|e| CorviaError::Storage(e.to_string()))?;
        if scope_entry.path().is_dir() {
            let scope_id = scope_entry.file_name().to_string_lossy().to_string();
            let scope_entries = read_scope(data_dir, &scope_id)?;
            entries.extend(scope_entries);
        }
    }
    Ok(entries)
}

/// Delete a single entry's JSON file.
pub fn delete_entry(data_dir: &Path, scope_id: &str, entry_id: &Uuid) -> Result<()> {
    validate_scope_id(scope_id)?;
    let path = entry_path(data_dir, scope_id, entry_id);
    std::fs::remove_file(&path)
        .map_err(|e| CorviaError::Storage(format!("Failed to delete {}: {e}", path.display())))?;
    Ok(())
}

/// Delete all entry files in a scope (removes the scope directory).
pub fn delete_scope_files(data_dir: &Path, scope_id: &str) -> Result<()> {
    validate_scope_id(scope_id)?;
    let dir = scope_dir(data_dir, scope_id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|e| CorviaError::Storage(format!("Failed to delete scope dir: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::types::KnowledgeEntry;

    #[test]
    fn test_write_and_read_entry() {
        let dir = tempfile::tempdir().unwrap();
        let entry = KnowledgeEntry::new(
            "fn hello() {}".into(),
            "test-scope".into(),
            "abc123".into(),
        )
        .with_embedding(vec![0.1, 0.2, 0.3]);
        write_entry(dir.path(), &entry).unwrap();
        let loaded = read_entry(dir.path(), "test-scope", &entry.id).unwrap();
        assert_eq!(loaded.content, entry.content);
        assert_eq!(loaded.embedding, entry.embedding);
    }

    #[test]
    fn test_read_scope() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3 {
            let entry = KnowledgeEntry::new(
                format!("entry {i}"),
                "my-scope".into(),
                "v1".into(),
            )
            .with_embedding(vec![0.1; 3]);
            write_entry(dir.path(), &entry).unwrap();
        }
        let entries = read_scope(dir.path(), "my-scope").unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_read_scope_empty() {
        let dir = tempfile::tempdir().unwrap();
        let entries = read_scope(dir.path(), "nonexistent").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_read_all() {
        let dir = tempfile::tempdir().unwrap();
        let e1 = KnowledgeEntry::new("a".into(), "scope-a".into(), "v1".into())
            .with_embedding(vec![0.1]);
        let e2 = KnowledgeEntry::new("b".into(), "scope-b".into(), "v1".into())
            .with_embedding(vec![0.2]);
        write_entry(dir.path(), &e1).unwrap();
        write_entry(dir.path(), &e2).unwrap();
        let all = read_all(dir.path()).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_delete_entry() {
        let dir = tempfile::tempdir().unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        write_entry(dir.path(), &entry).unwrap();
        delete_entry(dir.path(), "scope", &entry.id).unwrap();
        assert!(read_entry(dir.path(), "scope", &entry.id).is_err());
    }

    #[test]
    fn test_delete_scope() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3 {
            let entry = KnowledgeEntry::new(
                format!("entry {i}"),
                "doomed".into(),
                "v1".into(),
            );
            write_entry(dir.path(), &entry).unwrap();
        }
        delete_scope_files(dir.path(), "doomed").unwrap();
        let entries = read_scope(dir.path(), "doomed").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_scope_id_path_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();

        let bad_ids = ["../etc", "foo/bar", "scope\\sub", "..", ""];
        for bad_id in bad_ids {
            let entry = KnowledgeEntry::new("test".into(), bad_id.into(), "v1".into());
            let result = write_entry(dir.path(), &entry);
            assert!(result.is_err(), "should reject scope_id={bad_id:?}");
        }
    }

    #[test]
    fn test_scope_id_valid_names_accepted() {
        let dir = tempfile::tempdir().unwrap();

        let good_ids = ["my-scope", "scope_v2", "org:team:project", "scope.name"];
        for good_id in good_ids {
            let entry = KnowledgeEntry::new("test".into(), good_id.into(), "v1".into());
            assert!(write_entry(dir.path(), &entry).is_ok(), "should accept scope_id={good_id:?}");
        }
    }
}
