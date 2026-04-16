//! Project root discovery — walk up to find `.corvia/corvia.toml`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Walk up from `start` to find a directory containing `.corvia/corvia.toml`.
/// Returns the project root (parent of `.corvia/`).
pub fn find_project_root(start: &Path) -> Result<PathBuf> {
    let mut current = start
        .canonicalize()
        .unwrap_or_else(|_| start.to_path_buf());

    loop {
        let candidate = current.join(".corvia").join("corvia.toml");
        if candidate.is_file() {
            return Ok(current);
        }
        if !current.pop() {
            bail!(
                "No .corvia/ found (searched from {}). Run 'corvia init' to set up.",
                start.display()
            );
        }
    }
}

/// Resolve the project root: use `--base-dir` if provided, otherwise discover.
pub fn resolve_base_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    match explicit {
        Some(dir) => {
            let config = dir.join(".corvia").join("corvia.toml");
            if !config.is_file() {
                bail!(
                    "No .corvia/corvia.toml in {}. Run 'corvia init' first.",
                    dir.display()
                );
            }
            Ok(dir.to_path_buf())
        }
        None => find_project_root(&std::env::current_dir()?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn finds_root_in_current_dir() {
        let dir = TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        std::fs::write(corvia.join("corvia.toml"), "").unwrap();

        let root = find_project_root(dir.path()).unwrap();
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn finds_root_from_subdirectory() {
        let dir = TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        std::fs::write(corvia.join("corvia.toml"), "").unwrap();

        let sub = dir.path().join("src").join("deep");
        std::fs::create_dir_all(&sub).unwrap();

        let root = find_project_root(&sub).unwrap();
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn errors_when_no_corvia_dir() {
        let dir = TempDir::new().unwrap();
        let result = find_project_root(dir.path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("No .corvia/ found"));
    }

    #[test]
    fn resolve_explicit_base_dir() {
        let dir = TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        std::fs::write(corvia.join("corvia.toml"), "").unwrap();

        let root = resolve_base_dir(Some(dir.path())).unwrap();
        assert_eq!(root, dir.path().to_path_buf());
    }

    #[test]
    fn resolve_explicit_missing_errors() {
        let dir = TempDir::new().unwrap();
        let result = resolve_base_dir(Some(dir.path()));
        assert!(result.is_err());
    }
}
