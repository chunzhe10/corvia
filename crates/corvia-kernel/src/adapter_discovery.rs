//! Runtime adapter discovery and resolution (D74/D76).
//!
//! Scans `~/.config/corvia/adapters/` and `$PATH` for executables named
//! `corvia-adapter-*`. Each candidate is probed with `--corvia-metadata`
//! to extract its [`AdapterMetadata`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{debug, warn};

use crate::adapter_protocol::AdapterMetadata;

/// A discovered adapter binary with its parsed metadata.
#[derive(Debug, Clone)]
pub struct DiscoveredAdapter {
    pub binary_path: PathBuf,
    pub metadata: AdapterMetadata,
}

/// Discover all available adapters by scanning known directories.
///
/// Discovery order (D74): config dir first, then PATH. First match per
/// adapter name wins (config dir takes priority over PATH).
pub fn discover_adapters(extra_dirs: &[String]) -> Vec<DiscoveredAdapter> {
    discover_adapters_inner(extra_dirs, true)
}

/// Internal discovery with option to skip PATH and config dir scanning.
/// Used by tests to avoid picking up real adapter binaries from the host.
fn discover_adapters_inner(extra_dirs: &[String], scan_system: bool) -> Vec<DiscoveredAdapter> {
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut adapters = Vec::new();

    // 1. Extra dirs from config (e.g., [adapters].search_dirs)
    for dir in extra_dirs {
        let expanded = shellexpand_home(dir);
        scan_dir(&expanded, &mut seen_names, &mut adapters);
    }

    if scan_system {
        // 2. Default config dir: ~/.config/corvia/adapters/
        if let Some(home) = home_dir() {
            let config_dir = home.join(".config").join("corvia").join("adapters");
            scan_dir(&config_dir, &mut seen_names, &mut adapters);
        }

        // 3. $PATH
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                scan_dir(&dir, &mut seen_names, &mut adapters);
            }
        }
    }

    adapters
}

/// Resolve which adapter to use for a given source path.
///
/// Resolution order (D76):
/// 1. Explicit adapter name from [[sources]] config -> find by name
/// 2. Path has .git/ -> use "git" adapter if discovered
/// 3. Default from config -> find by name
/// 4. Fall back to "basic"
pub fn resolve_adapter<'a>(
    source_path: &str,
    discovered: &'a [DiscoveredAdapter],
    explicit_adapter: Option<&str>,
    default_adapter: Option<&str>,
) -> Option<&'a DiscoveredAdapter> {
    // 1. Explicit config match
    if let Some(name) = explicit_adapter {
        if let Some(a) = find_by_name(discovered, name) {
            return Some(a);
        }
        warn!("Configured adapter '{}' not found, falling through", name);
    }

    // 2. Auto-detect: .git/ -> git adapter
    if Path::new(source_path).join(".git").exists() {
        if let Some(a) = find_by_name(discovered, "git") {
            debug!("Auto-detected git adapter for {}", source_path);
            return Some(a);
        }
    }

    // 3. Config default
    if let Some(name) = default_adapter {
        if let Some(a) = find_by_name(discovered, name) {
            return Some(a);
        }
    }

    // 4. Fall back to "basic"
    find_by_name(discovered, "basic")
}

/// Find a discovered adapter by name.
pub fn find_by_name<'a>(discovered: &'a [DiscoveredAdapter], name: &str) -> Option<&'a DiscoveredAdapter> {
    discovered.iter().find(|a| a.metadata.name == name)
}

/// Format a human-readable adapter list for `corvia adapters list`.
pub fn format_adapter_list(adapters: &[DiscoveredAdapter]) -> String {
    if adapters.is_empty() {
        return "No adapters found.\n\
                Install adapters to ~/.config/corvia/adapters/ or ensure they are on PATH.\n\
                Adapter binaries must be named corvia-adapter-<name>.".to_string();
    }
    let mut out = String::new();
    for a in adapters {
        out.push_str(&format!(
            "  {:<12} v{:<8} {:<44} {}\n",
            a.metadata.name,
            a.metadata.version,
            a.metadata.description,
            a.binary_path.display(),
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn scan_dir(dir: &Path, seen: &mut HashSet<String>, out: &mut Vec<DiscoveredAdapter>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if !name.starts_with("corvia-adapter-") {
            continue;
        }

        // Extract adapter name: corvia-adapter-git -> git
        let adapter_name = name.trim_start_matches("corvia-adapter-");
        // Strip platform extension (.exe on Windows)
        let adapter_name = adapter_name.trim_end_matches(".exe");

        if adapter_name.is_empty() || seen.contains(adapter_name) {
            continue;
        }

        // Check executable
        if !is_executable(&path) {
            continue;
        }

        // Probe metadata
        match probe_metadata(&path) {
            Ok(meta) => {
                seen.insert(adapter_name.to_string());
                out.push(DiscoveredAdapter {
                    binary_path: path,
                    metadata: meta,
                });
            }
            Err(e) => {
                warn!("Failed to probe {}: {}", path.display(), e);
            }
        }
    }
}

fn probe_metadata(binary: &Path) -> Result<AdapterMetadata, String> {
    let output = Command::new(binary)
        .arg("--corvia-metadata")
        .output()
        .map_err(|e| format!("spawn failed: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "exit code {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("invalid metadata JSON: {e}"))
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file()
        && std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

fn shellexpand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fn create_mock_adapter(dir: &Path, name: &str, metadata_json: &str) {
        let script_path = dir.join(name);
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"--corvia-metadata\" ]; then echo '{}'; fi\n",
            metadata_json.replace('\'', "'\\''")
        );
        fs::write(&script_path, script).unwrap();
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn test_discover_from_extra_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_mock_adapter(
            dir.path(),
            "corvia-adapter-test",
            r#"{"name":"test","version":"0.1.0","domain":"test","protocol_version":1,"description":"Test adapter","supported_extensions":["txt"],"chunking_extensions":[]}"#,
        );

        let adapters = discover_adapters_inner(&[dir.path().to_string_lossy().to_string()], false);
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].metadata.name, "test");
        assert_eq!(adapters[0].metadata.version, "0.1.0");
    }

    #[test]
    fn test_discover_deduplicates_by_name() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let meta = r#"{"name":"test","version":"0.1.0","domain":"test","protocol_version":1,"description":"Test","supported_extensions":[],"chunking_extensions":[]}"#;
        create_mock_adapter(dir1.path(), "corvia-adapter-test", meta);
        create_mock_adapter(dir2.path(), "corvia-adapter-test", meta);

        let adapters = discover_adapters_inner(&[
            dir1.path().to_string_lossy().to_string(),
            dir2.path().to_string_lossy().to_string(),
        ], false);
        assert_eq!(adapters.len(), 1, "should deduplicate by name");
    }

    #[test]
    fn test_discover_skips_non_executable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corvia-adapter-test");
        fs::write(&path, "not executable").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        let adapters = discover_adapters_inner(&[dir.path().to_string_lossy().to_string()], false);
        assert!(adapters.is_empty());
    }

    #[test]
    fn test_resolve_explicit_adapter() {
        let adapters = vec![
            DiscoveredAdapter {
                binary_path: PathBuf::from("/usr/bin/corvia-adapter-git"),
                metadata: AdapterMetadata {
                    name: "git".into(),
                    version: "0.3.1".into(),
                    domain: "git".into(),
                    protocol_version: 1,
                    description: "Git adapter".into(),
                    supported_extensions: vec!["rs".into()],
                    chunking_extensions: vec!["rs".into()],
                },
            },
            DiscoveredAdapter {
                binary_path: PathBuf::from("/usr/bin/corvia-adapter-basic"),
                metadata: AdapterMetadata {
                    name: "basic".into(),
                    version: "0.1.0".into(),
                    domain: "filesystem".into(),
                    protocol_version: 1,
                    description: "Basic adapter".into(),
                    supported_extensions: vec!["txt".into()],
                    chunking_extensions: vec![],
                },
            },
        ];

        // Explicit name takes priority
        let resolved = resolve_adapter("/some/path", &adapters, Some("basic"), None);
        assert_eq!(resolved.unwrap().metadata.name, "basic");
    }

    #[test]
    fn test_resolve_auto_detect_git() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();

        let adapters = vec![DiscoveredAdapter {
            binary_path: PathBuf::from("/usr/bin/corvia-adapter-git"),
            metadata: AdapterMetadata {
                name: "git".into(),
                version: "0.3.1".into(),
                domain: "git".into(),
                protocol_version: 1,
                description: "Git adapter".into(),
                supported_extensions: vec![],
                chunking_extensions: vec![],
            },
        }];

        let resolved = resolve_adapter(
            &dir.path().to_string_lossy(),
            &adapters,
            None,
            None,
        );
        assert_eq!(resolved.unwrap().metadata.name, "git");
    }

    #[test]
    fn test_resolve_falls_back_to_basic() {
        let adapters = vec![DiscoveredAdapter {
            binary_path: PathBuf::from("/usr/bin/corvia-adapter-basic"),
            metadata: AdapterMetadata {
                name: "basic".into(),
                version: "0.1.0".into(),
                domain: "filesystem".into(),
                protocol_version: 1,
                description: "Basic adapter".into(),
                supported_extensions: vec![],
                chunking_extensions: vec![],
            },
        }];

        let resolved = resolve_adapter("/no/git/here", &adapters, None, None);
        assert_eq!(resolved.unwrap().metadata.name, "basic");
    }

    #[test]
    fn test_resolve_returns_none_when_empty() {
        let resolved = resolve_adapter("/any/path", &[], None, None);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_format_adapter_list_empty() {
        let output = format_adapter_list(&[]);
        assert!(output.contains("No adapters found"));
    }

    #[test]
    fn test_format_adapter_list() {
        let adapters = vec![DiscoveredAdapter {
            binary_path: PathBuf::from("/usr/local/bin/corvia-adapter-git"),
            metadata: AdapterMetadata {
                name: "git".into(),
                version: "0.3.1".into(),
                domain: "git".into(),
                protocol_version: 1,
                description: "Git + tree-sitter code ingestion".into(),
                supported_extensions: vec![],
                chunking_extensions: vec![],
            },
        }];
        let output = format_adapter_list(&adapters);
        assert!(output.contains("git"));
        assert!(output.contains("0.3.1"));
        assert!(output.contains("/usr/local/bin/corvia-adapter-git"));
    }

    #[test]
    fn test_shellexpand_home() {
        let expanded = shellexpand_home("~/test/path");
        // Should start with the home directory
        if let Some(home) = home_dir() {
            assert_eq!(expanded, home.join("test/path"));
        }
    }

    #[test]
    fn test_shellexpand_no_tilde() {
        let expanded = shellexpand_home("/absolute/path");
        assert_eq!(expanded, PathBuf::from("/absolute/path"));
    }
}
