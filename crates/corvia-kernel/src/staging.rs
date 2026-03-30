use corvia_common::agent_types::sanitize_agent_id;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::KnowledgeEntry;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use uuid::Uuid;

/// Manages staging directories for agent writes and git branch operations.
/// Staging dirs live under `{data_dir}/staging/{sanitized_agent}/{session_short}/`.
/// Knowledge files live under `{data_dir}/knowledge/{scope_id}/{entry_id}.json`.
pub struct StagingManager {
    data_dir: PathBuf,
}

impl StagingManager {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Resolve a stored staging dir string (e.g., ".corvia/staging/agent/sess-abc")
    /// into an absolute path under data_dir by stripping the ".corvia/" prefix.
    pub fn resolve_staging_path(&self, staging_dir_str: &str) -> PathBuf {
        self.data_dir.join(
            staging_dir_str.strip_prefix(".corvia/").unwrap_or(staging_dir_str)
        )
    }

    /// Create a staging directory for an agent session.
    /// Path: `{data_dir}/staging/{sanitized_agent}/{session_short}/`
    pub fn create_staging_dir(&self, agent_id: &str, session_short: &str) -> Result<PathBuf> {
        let sanitized = sanitize_agent_id(agent_id);
        let path = self.data_dir.join("staging").join(&sanitized).join(session_short);
        std::fs::create_dir_all(&path)
            .map_err(|e| CorviaError::Agent(format!("Failed to create staging dir: {e}")))?;
        Ok(path)
    }

    /// Write a knowledge entry to a staging file as `{entry_id}.json`.
    /// Also writes a companion `.vec` file with raw embedding bytes,
    /// since the embedding field is skip_serializing in JSON.
    pub fn write_staging_file(&self, staging_dir: &Path, entry: &KnowledgeEntry) -> Result<()> {
        let path = staging_dir.join(format!("{}.json", entry.id));
        let json = serde_json::to_string_pretty(entry)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize entry: {e}")))?;
        std::fs::write(&path, json)
            .map_err(|e| CorviaError::Agent(format!("Failed to write staging file: {e}")))?;

        // Write companion .vec file with raw embedding bytes
        if let Some(ref embedding) = entry.embedding {
            let vec_path = staging_dir.join(format!("{}.vec", entry.id));
            let vec_bytes: &[u8] = bytemuck::cast_slice(embedding);
            std::fs::write(&vec_path, vec_bytes)
                .map_err(|e| CorviaError::Agent(format!("Failed to write staging vec file: {e}")))?;
        }
        Ok(())
    }

    /// Read a knowledge entry from a staging file.
    /// Also reads the companion `.vec` file to restore the embedding,
    /// since the embedding field is skip_serializing in JSON.
    pub fn read_staging_file(&self, staging_dir: &Path, entry_id: &Uuid) -> Result<KnowledgeEntry> {
        let path = staging_dir.join(format!("{entry_id}.json"));
        let content = std::fs::read_to_string(&path)
            .map_err(|e| CorviaError::Agent(format!("Failed to read staging file: {e}")))?;
        let mut entry: KnowledgeEntry = serde_json::from_str(&content)
            .map_err(|e| CorviaError::Agent(format!("Failed to deserialize staging entry: {e}")))?;

        // Restore embedding from companion .vec file
        let vec_path = staging_dir.join(format!("{entry_id}.vec"));
        if let Ok(bytes) = std::fs::read(&vec_path)
            && let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(&bytes)
        {
            entry.embedding = Some(floats.to_vec());
        }
        Ok(entry)
    }

    /// List all entry IDs in a staging directory (from `*.json` filenames).
    pub fn list_staging_files(&self, staging_dir: &Path) -> Result<Vec<Uuid>> {
        if !staging_dir.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(staging_dir)
            .map_err(|e| CorviaError::Agent(format!("Failed to read staging dir: {e}")))?;
        for entry in entries {
            let entry = entry
                .map_err(|e| CorviaError::Agent(format!("Failed to read dir entry: {e}")))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(stem) = name_str.strip_suffix(".json")
                && let Ok(uuid) = Uuid::parse_str(stem) {
                    ids.push(uuid);
                }
        }
        Ok(ids)
    }

    /// Move a staging file to the main knowledge directory.
    /// From: `{staging_dir}/{entry_id}.json`
    /// To:   `{data_dir}/knowledge/{scope_id}/{entry_id}.json`
    /// Also cleans up the companion `.vec` file (embeddings are in VECTORS table).
    pub fn move_to_knowledge(
        &self,
        staging_dir: &Path,
        entry_id: &Uuid,
        scope_id: &str,
    ) -> Result<()> {
        let src = staging_dir.join(format!("{entry_id}.json"));
        let dest_dir = self.data_dir.join("knowledge").join(scope_id);
        std::fs::create_dir_all(&dest_dir)
            .map_err(|e| CorviaError::Agent(format!("Failed to create knowledge dir: {e}")))?;
        let dest = dest_dir.join(format!("{entry_id}.json"));
        std::fs::rename(&src, &dest)
            .map_err(|e| CorviaError::Agent(format!("Failed to move staging file to knowledge: {e}")))?;

        // Clean up companion .vec file (embedding is now in VECTORS table)
        let vec_src = staging_dir.join(format!("{entry_id}.vec"));
        let _ = std::fs::remove_file(vec_src);
        Ok(())
    }

    /// Remove a staging directory and all its contents.
    pub fn cleanup_staging_dir(&self, staging_dir: &Path) -> Result<()> {
        if staging_dir.exists() {
            std::fs::remove_dir_all(staging_dir)
                .map_err(|e| CorviaError::Agent(format!("Failed to cleanup staging dir: {e}")))?;
        }
        Ok(())
    }

    // --- Git operations ---
    // These shell out to `git` and are no-ops if not in a git repo.

    /// Create a git branch from HEAD.
    pub fn create_branch(&self, branch_name: &str) -> Result<()> {
        if !self.is_git_repo() {
            warn!("Not a git repo, skipping branch creation for {branch_name}");
            return Ok(());
        }
        run_git(&self.data_dir, &["branch", branch_name])?;
        info!(branch_name, "git_branch_created");
        Ok(())
    }

    /// Commit files on an agent branch.
    /// Checks out the branch, stages files, commits, then checks out the previous branch.
    pub fn commit_on_branch(
        &self,
        branch_name: &str,
        message: &str,
        files: &[&str],
    ) -> Result<()> {
        if !self.is_git_repo() {
            warn!("Not a git repo, skipping commit on {branch_name}");
            return Ok(());
        }
        run_git(&self.data_dir, &["checkout", branch_name])?;
        let mut add_args = vec!["add"];
        add_args.extend_from_slice(files);
        run_git(&self.data_dir, &add_args)?;
        run_git(&self.data_dir, &["commit", "-m", message])?;
        run_git(&self.data_dir, &["checkout", "-"])?;
        info!(branch_name, "git_commit_on_branch");
        Ok(())
    }

    /// Merge an agent branch into the current branch with --no-ff.
    pub fn merge_branch(&self, branch_name: &str) -> Result<()> {
        if !self.is_git_repo() {
            warn!("Not a git repo, skipping merge for {branch_name}");
            return Ok(());
        }
        let message = format!("merge: {branch_name}");
        run_git(&self.data_dir, &["merge", branch_name, "--no-ff", "-m", &message])?;
        info!(branch_name, "git_branch_merged");
        Ok(())
    }

    /// Delete a git branch.
    pub fn delete_branch(&self, branch_name: &str) -> Result<()> {
        if !self.is_git_repo() {
            warn!("Not a git repo, skipping branch deletion for {branch_name}");
            return Ok(());
        }
        run_git(&self.data_dir, &["branch", "-d", branch_name])?;
        info!(branch_name, "git_branch_deleted");
        Ok(())
    }

    /// Check if the data_dir is inside a git repository.
    fn is_git_repo(&self) -> bool {
        // Walk up from data_dir looking for .git
        let mut dir = self.data_dir.as_path();
        loop {
            if dir.join(".git").exists() {
                return true;
            }
            match dir.parent() {
                Some(parent) => dir = parent,
                None => return false,
            }
        }
    }
}

/// Run a git command in the given directory.
fn run_git(dir: &Path, args: &[&str]) -> Result<()> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| CorviaError::Agent(format!("Failed to run git {}: {e}", args.join(" "))))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CorviaError::Agent(format!(
            "git {} failed: {stderr}",
            args.join(" ")
        )));
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_staging_in(dir: &Path) -> StagingManager {
        StagingManager::new(dir)
    }

    #[test]
    fn test_create_staging_dir() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let path = staging.create_staging_dir("test::agent", "sess-abc123").unwrap();
        assert!(path.exists());
        assert!(path.is_dir());
    }

    #[test]
    fn test_write_and_read_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let staging_dir = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        staging.write_staging_file(&staging_dir, &entry).unwrap();
        let files = staging.list_staging_files(&staging_dir).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_cleanup_staging_dir() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let path = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        assert!(path.exists());
        staging.cleanup_staging_dir(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_move_to_knowledge() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let staging_dir = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        let entry_id = entry.id;
        staging.write_staging_file(&staging_dir, &entry).unwrap();
        staging.move_to_knowledge(&staging_dir, &entry_id, "scope").unwrap();

        // Staging file gone, knowledge file exists
        assert_eq!(staging.list_staging_files(&staging_dir).unwrap().len(), 0);
        let knowledge_path = dir.path()
            .join("knowledge").join("scope").join(format!("{entry_id}.json"));
        assert!(knowledge_path.exists());
    }

    #[test]
    fn test_staging_vec_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let staging_dir = staging.create_staging_dir("test::agent", "sess-vec").unwrap();

        // Write entry WITH embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let entry = KnowledgeEntry::new("vec test".into(), "scope".into(), "v1".into())
            .with_embedding(embedding.clone());
        let entry_id = entry.id;
        staging.write_staging_file(&staging_dir, &entry).unwrap();

        // Verify .vec companion file exists
        let vec_path = staging_dir.join(format!("{entry_id}.vec"));
        assert!(vec_path.exists(), ".vec file should be created for entries with embeddings");

        // Read back and verify embedding is restored
        let loaded = staging.read_staging_file(&staging_dir, &entry_id).unwrap();
        assert_eq!(loaded.embedding, Some(embedding), "embedding should roundtrip via .vec file");

        // Entry without embedding should NOT create .vec file
        let no_emb = KnowledgeEntry::new("no vec".into(), "scope".into(), "v1".into());
        let no_emb_id = no_emb.id;
        staging.write_staging_file(&staging_dir, &no_emb).unwrap();
        let no_vec_path = staging_dir.join(format!("{no_emb_id}.vec"));
        assert!(!no_vec_path.exists(), ".vec file should not exist for entries without embeddings");

        // move_to_knowledge cleans up .vec file
        staging.move_to_knowledge(&staging_dir, &entry_id, "scope").unwrap();
        assert!(!vec_path.exists(), ".vec file should be removed after move_to_knowledge");
    }

    #[test]
    fn test_git_branch_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        // Initialize a git repo with a valid initial commit
        std::process::Command::new("git").args(["init"]).current_dir(dir.path()).output().unwrap();
        std::process::Command::new("git")
            .args(["-c", "user.name=test", "-c", "user.email=test@test.com", "commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path()).output().unwrap();

        let staging = test_staging_in(dir.path());
        staging.create_branch("agents/test/sess-abc").unwrap();
        staging.delete_branch("agents/test/sess-abc").unwrap();
    }
}
