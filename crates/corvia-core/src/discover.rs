//! Project root discovery — walk up to find `.corvia/corvia.toml`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Walk up from `start` to find a directory containing `.corvia/corvia.toml`.
/// Returns the project root (parent of `.corvia/`).
///
/// If additional `.corvia/corvia.toml` files exist further up the directory
/// tree (the closer one shadowing one or more enclosing stores), emits a
/// `tracing::warn!` so the operator notices the misconfiguration. The closest
/// root is still returned so behavior is unchanged — only visibility improves.
pub fn find_project_root(start: &Path) -> Result<PathBuf> {
    let (root, shadowed) = find_project_root_with_ancestors(start)?;
    if !shadowed.is_empty() {
        let ancestors = shadowed
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::warn!(
            target: "corvia.discover",
            active_root = %root.display(),
            shadowed_ancestors = %ancestors,
            "Multiple .corvia/ roots found on walk-up path. Using the closest one ({}); \
             the following enclosing .corvia/ are being shadowed: {}. \
             This usually means `corvia init` ran in a subdirectory by mistake. \
             Consider removing the inner store to consolidate state.",
            root.display(),
            ancestors
        );
    }
    Ok(root)
}

/// Internal variant of [`find_project_root`] that surfaces shadowed ancestor
/// roots instead of warning about them. The first element of the returned tuple
/// is the closest root; the second is every enclosing root that would be
/// shadowed by it, in walk-up order.
fn find_project_root_with_ancestors(start: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    let mut current = start
        .canonicalize()
        .unwrap_or_else(|_| start.to_path_buf());

    let mut primary: Option<PathBuf> = None;
    let mut shadowed: Vec<PathBuf> = Vec::new();

    loop {
        let candidate = current.join(".corvia").join("corvia.toml");
        if candidate.is_file() {
            if primary.is_none() {
                primary = Some(current.clone());
            } else {
                shadowed.push(current.clone());
            }
        }
        if !current.pop() {
            break;
        }
    }

    match primary {
        Some(root) => Ok((root, shadowed)),
        None => bail!(
            "No .corvia/ found (searched from {}). Run 'corvia init' to set up.",
            start.display()
        ),
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

    #[test]
    fn shadowed_enclosing_root_is_detected() {
        // Outer .corvia/ at dir/, inner .corvia/ at dir/subdir/.
        let dir = TempDir::new().unwrap();
        let outer = dir.path().join(".corvia");
        std::fs::create_dir_all(&outer).unwrap();
        std::fs::write(outer.join("corvia.toml"), "").unwrap();

        let inner_root = dir.path().join("subdir");
        let inner = inner_root.join(".corvia");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join("corvia.toml"), "").unwrap();

        let (primary, shadowed) = find_project_root_with_ancestors(&inner_root).unwrap();
        let canonical_outer = dir.path().canonicalize().unwrap();
        let canonical_inner = inner_root.canonicalize().unwrap();

        assert_eq!(primary, canonical_inner, "closest root wins");
        assert_eq!(
            shadowed,
            vec![canonical_outer],
            "enclosing root is reported as shadowed"
        );

        // Public API returns the same primary and doesn't error.
        assert_eq!(find_project_root(&inner_root).unwrap(), canonical_inner);
    }

    #[test]
    fn no_shadowed_ancestors_when_only_one_root() {
        let dir = TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        std::fs::create_dir_all(&corvia).unwrap();
        std::fs::write(corvia.join("corvia.toml"), "").unwrap();

        let sub = dir.path().join("src").join("deep");
        std::fs::create_dir_all(&sub).unwrap();

        let (_, shadowed) = find_project_root_with_ancestors(&sub).unwrap();
        assert!(shadowed.is_empty());
    }
}
