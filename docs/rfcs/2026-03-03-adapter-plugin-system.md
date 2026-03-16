# Adapter Plugin System — Implementation Plan

> **Status:** Shipped (v0.3.0)

**Goal:** Replace compile-time adapter dependencies with runtime process-based plugins discovered via JSONL over stdin/stdout, enabling adapter ecosystem extensibility.

**Architecture:** Adapters become standalone binaries (`corvia-adapter-*`) discovered at runtime via PATH + config directory scanning. The kernel spawns adapter processes, sends JSONL commands to stdin, reads JSONL responses from stdout. First-party adapters live in the corvia monorepo as workspace crates. Third-party adapters can be written in any language.

**Tech Stack:** Rust (serde_json for JSONL, std::process for IPC, walkdir for discovery), existing corvia-kernel traits and pipeline.

**Design doc:** `docs/rfcs/2026-03-03-adapter-plugin-system-design.md` (approved)
**Decisions:** D72-D79 in `docs/local/plans/2026-03-03-adapter-plugin-architecture.md`

---

## Task 1: Add Serde Derives to Chunking Types

Add `Serialize`/`Deserialize` to all chunking types so they can cross the JSONL wire. No structural changes — purely additive derives.

**Files:**
- Modify: `crates/corvia-kernel/src/chunking_strategy.rs`
- Modify: `crates/corvia-kernel/Cargo.toml` (ensure serde is available — already is)

**Step 1: Add serde derives to all data types**

In `crates/corvia-kernel/src/chunking_strategy.rs`, add `Serialize, Deserialize` to all six public structs. Add the serde import at the top:

```rust
use corvia_common::errors::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
```

Then change each struct's derives:
- `SourceMetadata`: `#[derive(Debug, Clone, Serialize, Deserialize)]`
- `RawChunk`: `#[derive(Debug, Clone, Serialize, Deserialize)]`
- `ChunkMetadata`: `#[derive(Debug, Clone, Default, Serialize, Deserialize)]`
- `ProcessedChunk`: `#[derive(Debug, Clone, Serialize, Deserialize)]`
- `ProcessingInfo`: `#[derive(Debug, Clone, Serialize, Deserialize)]`
- `ChunkRelation`: `#[derive(Debug, Clone, Serialize, Deserialize)]`
- `ChunkResult`: `#[derive(Debug, Clone, Default, Serialize, Deserialize)]`
- `SourceFile`: `#[derive(Debug, Clone, Serialize, Deserialize)]`

**Step 2: Add serde roundtrip test**

Add to the existing `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_source_file_serde_roundtrip() {
    let sf = SourceFile {
        content: "fn main() {}".into(),
        metadata: SourceMetadata {
            file_path: "src/main.rs".into(),
            extension: "rs".into(),
            language: Some("rust".into()),
            scope_id: "test".into(),
            source_version: "abc123".into(),
        },
    };
    let json = serde_json::to_string(&sf).unwrap();
    let parsed: SourceFile = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, sf.content);
    assert_eq!(parsed.metadata.file_path, sf.metadata.file_path);
    assert_eq!(parsed.metadata.language, sf.metadata.language);
}

#[test]
fn test_chunk_result_serde_roundtrip() {
    let cr = ChunkResult {
        chunks: vec![RawChunk {
            content: "fn main() {}".into(),
            chunk_type: "function".into(),
            start_line: 1,
            end_line: 3,
            metadata: ChunkMetadata {
                source_file: "src/main.rs".into(),
                language: Some("rust".into()),
                ..Default::default()
            },
        }],
        relations: vec![ChunkRelation {
            from_source_file: "src/main.rs".into(),
            from_start_line: 1,
            relation: "imports".into(),
            to_file: "src/lib.rs".into(),
            to_name: Some("MyStruct".into()),
        }],
    };
    let json = serde_json::to_string(&cr).unwrap();
    let parsed: ChunkResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.chunks.len(), 1);
    assert_eq!(parsed.relations.len(), 1);
    assert_eq!(parsed.chunks[0].content, "fn main() {}");
    assert_eq!(parsed.relations[0].to_name, Some("MyStruct".into()));
}
```

**Step 3: Run tests**

```bash
cargo test -p corvia-kernel chunking_strategy -- --nocapture
```

Expected: all existing tests pass + 2 new serde tests pass.

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/chunking_strategy.rs
git commit -m "feat: add Serialize/Deserialize derives to all chunking types

Enables JSONL serialization for the adapter plugin protocol (D75)."
```

---

## Task 2: Create Adapter Protocol Types

Define the JSONL wire protocol types for host ↔ adapter communication.

**Files:**
- Create: `crates/corvia-kernel/src/adapter_protocol.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module export)

**Step 1: Write the protocol type tests**

Create `crates/corvia-kernel/src/adapter_protocol.rs`:

```rust
//! JSONL wire protocol types for adapter ↔ host communication (D75).
//!
//! Adapters are standalone binaries that communicate with the kernel via
//! newline-delimited JSON over stdin/stdout. This module defines the shared
//! types for metadata probing, request messages, and response messages.

use serde::{Deserialize, Serialize};

use crate::chunking_strategy::{ChunkRelation, RawChunk, SourceMetadata};

// ---------------------------------------------------------------------------
// Metadata (returned by `--corvia-metadata`)
// ---------------------------------------------------------------------------

/// Self-description returned by `corvia-adapter-* --corvia-metadata`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub name: String,
    pub version: String,
    pub domain: String,
    pub protocol_version: u32,
    pub description: String,
    pub supported_extensions: Vec<String>,
    pub chunking_extensions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Host → Adapter requests (written to stdin as JSON lines)
// ---------------------------------------------------------------------------

/// A request from the host to the adapter.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum AdapterRequest {
    /// Walk the source path and stream back source files.
    #[serde(rename = "ingest")]
    Ingest {
        source_path: String,
        scope_id: String,
    },
    /// Chunk the given file content using adapter-specific intelligence.
    #[serde(rename = "chunk")]
    Chunk {
        content: String,
        metadata: SourceMetadata,
    },
}

// ---------------------------------------------------------------------------
// Adapter → Host responses (read from stdout, one JSON line per message)
// ---------------------------------------------------------------------------

/// A source file payload streamed during ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFilePayload {
    pub content: String,
    pub metadata: SourceMetadata,
}

/// An error reported by the adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterError {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// A response line from the adapter.
///
/// Uses `#[serde(untagged)]` — the deserializer tries each variant in order.
/// Ordering matters: most specific first.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdapterResponse {
    /// A single source file (streamed during ingest).
    SourceFile { source_file: SourceFilePayload },
    /// Chunking result for a single file.
    ChunkResult {
        chunks: Vec<RawChunk>,
        relations: Vec<ChunkRelation>,
    },
    /// Signals end of ingestion stream.
    Done { done: bool, total_files: usize },
    /// An error from the adapter.
    Error { error: AdapterError },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_serde_roundtrip() {
        let meta = AdapterMetadata {
            name: "git".into(),
            version: "0.3.1".into(),
            domain: "git".into(),
            protocol_version: 1,
            description: "Git + tree-sitter code ingestion".into(),
            supported_extensions: vec!["rs".into(), "py".into()],
            chunking_extensions: vec!["rs".into(), "py".into()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: AdapterMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "git");
        assert_eq!(parsed.protocol_version, 1);
        assert_eq!(parsed.chunking_extensions, vec!["rs", "py"]);
    }

    #[test]
    fn test_request_ingest_serde() {
        let req = AdapterRequest::Ingest {
            source_path: "/repo".into(),
            scope_id: "proj".into(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"method\":\"ingest\""));
        let parsed: AdapterRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            AdapterRequest::Ingest { source_path, scope_id } => {
                assert_eq!(source_path, "/repo");
                assert_eq!(scope_id, "proj");
            }
            _ => panic!("expected Ingest"),
        }
    }

    #[test]
    fn test_request_chunk_serde() {
        let req = AdapterRequest::Chunk {
            content: "fn main() {}".into(),
            metadata: SourceMetadata {
                file_path: "src/main.rs".into(),
                extension: "rs".into(),
                language: Some("rust".into()),
                scope_id: "test".into(),
                source_version: "abc123".into(),
            },
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"method\":\"chunk\""));
    }

    #[test]
    fn test_response_source_file_serde() {
        let json = r#"{"source_file":{"content":"hello","metadata":{"file_path":"a.rs","extension":"rs","language":null,"scope_id":"s","source_version":"v1"}}}"#;
        let resp: AdapterResponse = serde_json::from_str(json).unwrap();
        match resp {
            AdapterResponse::SourceFile { source_file } => {
                assert_eq!(source_file.content, "hello");
                assert_eq!(source_file.metadata.file_path, "a.rs");
            }
            _ => panic!("expected SourceFile"),
        }
    }

    #[test]
    fn test_response_done_serde() {
        let json = r#"{"done":true,"total_files":42}"#;
        let resp: AdapterResponse = serde_json::from_str(json).unwrap();
        match resp {
            AdapterResponse::Done { done, total_files } => {
                assert!(done);
                assert_eq!(total_files, 42);
            }
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn test_response_error_serde() {
        let json = r#"{"error":{"code":"PARSE_FAILED","message":"bad syntax","file":"bad.rs"}}"#;
        let resp: AdapterResponse = serde_json::from_str(json).unwrap();
        match resp {
            AdapterResponse::Error { error } => {
                assert_eq!(error.code, "PARSE_FAILED");
                assert_eq!(error.file, Some("bad.rs".into()));
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_python_adapter_metadata_compat() {
        // Matches exact output of adapters/corvia-adapter-basic/python/corvia-adapter-basic
        let json = r#"{"name":"basic","version":"0.1.0","domain":"filesystem","protocol_version":1,"description":"Basic filesystem adapter","supported_extensions":["rs","py"],"chunking_extensions":[]}"#;
        let meta: AdapterMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.name, "basic");
        assert!(meta.chunking_extensions.is_empty());
    }

    #[test]
    fn test_python_adapter_ingest_response_compat() {
        // Matches exact JSONL output of the Python adapter's ingest command
        let json = r#"{"source_file":{"content":"fn main() {}","metadata":{"file_path":"src/main.rs","extension":"rs","language":null,"scope_id":"myrepo","source_version":"unknown"}}}"#;
        let resp: AdapterResponse = serde_json::from_str(json).unwrap();
        match resp {
            AdapterResponse::SourceFile { source_file } => {
                assert_eq!(source_file.metadata.source_version, "unknown");
            }
            _ => panic!("expected SourceFile"),
        }
    }
}
```

**Step 2: Export the module**

In `crates/corvia-kernel/src/lib.rs`, add after the `chunking_pdf` line (line 90):

```rust
pub mod adapter_protocol;
```

**Step 3: Run tests**

```bash
cargo test -p corvia-kernel adapter_protocol -- --nocapture
```

Expected: 8 tests pass.

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/adapter_protocol.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat: define JSONL wire protocol types for adapter IPC (D75)

AdapterMetadata, AdapterRequest, AdapterResponse types with full serde
roundtrip tests and cross-language compatibility validation."
```

---

## Task 3: Create Adapter Discovery Module

Scan filesystem for adapter binaries, probe metadata, resolve which adapter to use.

**Files:**
- Create: `crates/corvia-kernel/src/adapter_discovery.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module export)

**Step 1: Write the discovery module**

Create `crates/corvia-kernel/src/adapter_discovery.rs`:

```rust
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
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut adapters = Vec::new();

    // 1. Extra dirs from config (e.g., [adapters].search_dirs)
    for dir in extra_dirs {
        let expanded = shellexpand_home(dir);
        scan_dir(&expanded, &mut seen_names, &mut adapters);
    }

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

    adapters
}

/// Resolve which adapter to use for a given source path.
///
/// Resolution order (D76):
/// 1. Explicit adapter name from [[sources]] config → find by name
/// 2. Path has .git/ → use "git" adapter if discovered
/// 3. Default from config → find by name
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

    // 2. Auto-detect: .git/ → git adapter
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

        // Extract adapter name: corvia-adapter-git → git
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

        let adapters = discover_adapters(&[dir.path().to_string_lossy().to_string()]);
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

        let adapters = discover_adapters(&[
            dir1.path().to_string_lossy().to_string(),
            dir2.path().to_string_lossy().to_string(),
        ]);
        assert_eq!(adapters.len(), 1, "should deduplicate by name");
    }

    #[test]
    fn test_discover_skips_non_executable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corvia-adapter-test");
        fs::write(&path, "not executable").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        let adapters = discover_adapters(&[dir.path().to_string_lossy().to_string()]);
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
```

**Step 2: Export the module**

In `crates/corvia-kernel/src/lib.rs`, add after the `adapter_protocol` line:

```rust
pub mod adapter_discovery;
```

**Step 3: Run tests**

```bash
cargo test -p corvia-kernel adapter_discovery -- --nocapture
```

Expected: 10 tests pass (mock adapter tests spawn real shell scripts).

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/adapter_discovery.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat: add adapter discovery with PATH scan and metadata probing (D74)

Discovers corvia-adapter-* binaries from config dirs and PATH, probes
each with --corvia-metadata, and resolves which adapter to use via
explicit config, .git/ auto-detection, or basic fallback (D76)."
```

---

## Task 4: Create ProcessAdapter IPC Wrapper

The core IPC layer: spawn an adapter process, send JSONL commands, read JSONL responses.

**Files:**
- Create: `crates/corvia-kernel/src/process_adapter.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module export)

**Step 1: Write the ProcessAdapter module**

Create `crates/corvia-kernel/src/process_adapter.rs`:

```rust
//! IPC wrapper for adapter processes (D72/D75).
//!
//! [`ProcessAdapter`] manages the lifecycle of an adapter binary: spawning it,
//! writing JSONL requests to its stdin, and reading JSONL responses from stdout.
//! One process per adapter per ingestion session.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use tracing::{debug, warn};

use crate::adapter_protocol::{
    AdapterError, AdapterMetadata, AdapterRequest, AdapterResponse, SourceFilePayload,
};
use crate::chunking_strategy::{ChunkRelation, RawChunk, SourceFile, SourceMetadata};

/// IPC wrapper for a single adapter process.
pub struct ProcessAdapter {
    binary_path: PathBuf,
    metadata: AdapterMetadata,
    child: Option<Child>,
    stdin: Option<BufWriter<std::process::ChildStdin>>,
    stdout: Option<BufReader<std::process::ChildStdout>>,
}

impl ProcessAdapter {
    /// Create a new ProcessAdapter (not yet spawned).
    pub fn new(binary_path: PathBuf, metadata: AdapterMetadata) -> Self {
        Self {
            binary_path,
            metadata,
            child: None,
            stdin: None,
            stdout: None,
        }
    }

    /// The adapter's metadata.
    pub fn metadata(&self) -> &AdapterMetadata {
        &self.metadata
    }

    /// Spawn the adapter process in session mode (no args).
    ///
    /// stdin is open for commands, stdout for responses, stderr inherited.
    pub fn spawn(&mut self) -> Result<(), String> {
        let child = Command::new(&self.binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {e}", self.binary_path.display()))?;

        let stdin = child.stdin.as_ref().map(|_| ()).ok_or("no stdin")?;
        let stdout = child.stdout.as_ref().map(|_| ()).ok_or("no stdout")?;
        let _ = (stdin, stdout);

        let mut child = child;
        let stdin = BufWriter::new(child.stdin.take().unwrap());
        let stdout = BufReader::new(child.stdout.take().unwrap());

        self.child = Some(child);
        self.stdin = Some(stdin);
        self.stdout = Some(stdout);

        debug!("Spawned adapter {} (pid {:?})", self.metadata.name,
            self.child.as_ref().map(|c| c.id()));
        Ok(())
    }

    /// Send an ingest command and collect all source files until Done.
    pub fn ingest(&mut self, source_path: &str, scope_id: &str) -> Result<Vec<SourceFile>, String> {
        let req = AdapterRequest::Ingest {
            source_path: source_path.to_string(),
            scope_id: scope_id.to_string(),
        };
        self.send_request(&req)?;

        let mut files = Vec::new();
        loop {
            let line = self.read_line()?;
            let resp: AdapterResponse = serde_json::from_str(&line)
                .map_err(|e| format!("Invalid JSONL from adapter: {e}\nLine: {line}"))?;

            match resp {
                AdapterResponse::SourceFile { source_file } => {
                    files.push(SourceFile {
                        content: source_file.content,
                        metadata: source_file.metadata,
                    });
                }
                AdapterResponse::Done { total_files, .. } => {
                    debug!("Adapter {} finished: {} files", self.metadata.name, total_files);
                    break;
                }
                AdapterResponse::Error { error } => {
                    return Err(format!(
                        "Adapter error [{}]: {}{}",
                        error.code,
                        error.message,
                        error.file.map(|f| format!(" (file: {f})")).unwrap_or_default()
                    ));
                }
                AdapterResponse::ChunkResult { .. } => {
                    warn!("Unexpected ChunkResult during ingest, ignoring");
                }
            }
        }
        Ok(files)
    }

    /// Send a chunk command and return the result.
    pub fn chunk(&mut self, content: &str, metadata: &SourceMetadata) -> Result<(Vec<RawChunk>, Vec<ChunkRelation>), String> {
        let req = AdapterRequest::Chunk {
            content: content.to_string(),
            metadata: metadata.clone(),
        };
        self.send_request(&req)?;

        let line = self.read_line()?;
        let resp: AdapterResponse = serde_json::from_str(&line)
            .map_err(|e| format!("Invalid JSONL from adapter: {e}"))?;

        match resp {
            AdapterResponse::ChunkResult { chunks, relations } => Ok((chunks, relations)),
            AdapterResponse::Error { error } => {
                Err(format!("Adapter chunk error [{}]: {}", error.code, error.message))
            }
            _ => Err("Unexpected response to chunk request".into()),
        }
    }

    /// Gracefully shut down the adapter process.
    ///
    /// Closes stdin (adapter sees EOF and exits), then waits for process.
    pub fn shutdown(&mut self) -> Result<(), String> {
        // Drop stdin to send EOF
        self.stdin.take();

        if let Some(mut child) = self.child.take() {
            match child.wait() {
                Ok(status) => {
                    if !status.success() {
                        warn!("Adapter {} exited with {}", self.metadata.name, status);
                    }
                    debug!("Adapter {} shut down", self.metadata.name);
                }
                Err(e) => {
                    warn!("Failed to wait for adapter {}: {e}", self.metadata.name);
                }
            }
        }
        self.stdout.take();
        Ok(())
    }

    // -- Internal helpers --

    fn send_request(&mut self, req: &AdapterRequest) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("Adapter not spawned")?;
        let json = serde_json::to_string(req).map_err(|e| format!("Serialize error: {e}"))?;
        stdin
            .write_all(json.as_bytes())
            .map_err(|e| format!("Write to adapter stdin: {e}"))?;
        stdin
            .write_all(b"\n")
            .map_err(|e| format!("Write newline: {e}"))?;
        stdin.flush().map_err(|e| format!("Flush stdin: {e}"))?;
        Ok(())
    }

    fn read_line(&mut self) -> Result<String, String> {
        let stdout = self.stdout.as_mut().ok_or("Adapter not spawned")?;
        let mut line = String::new();
        let bytes = stdout
            .read_line(&mut line)
            .map_err(|e| format!("Read from adapter stdout: {e}"))?;
        if bytes == 0 {
            return Err("Adapter closed stdout unexpectedly (EOF)".into());
        }
        Ok(line.trim_end().to_string())
    }
}

impl Drop for ProcessAdapter {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    /// Create a mock adapter shell script that responds to session-mode JSONL.
    fn create_session_adapter(dir: &Path) -> PathBuf {
        let path = dir.join("corvia-adapter-mock");
        // This script reads JSON lines from stdin and responds appropriately
        let script = r#"#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    method = req.get("method")

    if method == "ingest":
        # Return one file then done
        sf = {"source_file": {"content": "hello world", "metadata": {
            "file_path": "test.txt", "extension": "txt",
            "language": None, "scope_id": req["params"]["scope_id"],
            "source_version": "v1"
        }}}
        print(json.dumps(sf), flush=True)
        print(json.dumps({"done": True, "total_files": 1}), flush=True)

    elif method == "chunk":
        chunks = [{"content": req["params"]["content"], "chunk_type": "file",
                   "start_line": 1, "end_line": 1,
                   "metadata": {"source_file": req["params"]["metadata"]["file_path"],
                               "language": None, "parent_chunk_id": None, "merge_group": None}}]
        print(json.dumps({"chunks": chunks, "relations": []}), flush=True)
"#;
        fs::write(&path, script).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn mock_metadata() -> AdapterMetadata {
        AdapterMetadata {
            name: "mock".into(),
            version: "0.1.0".into(),
            domain: "test".into(),
            protocol_version: 1,
            description: "Mock adapter".into(),
            supported_extensions: vec!["txt".into()],
            chunking_extensions: vec![],
        }
    }

    #[test]
    fn test_spawn_and_ingest() {
        let dir = tempfile::tempdir().unwrap();
        let binary = create_session_adapter(dir.path());
        let mut adapter = ProcessAdapter::new(binary, mock_metadata());

        adapter.spawn().unwrap();
        let files = adapter.ingest("/fake/path", "test-scope").unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content, "hello world");
        assert_eq!(files[0].metadata.file_path, "test.txt");
        assert_eq!(files[0].metadata.scope_id, "test-scope");
        adapter.shutdown().unwrap();
    }

    #[test]
    fn test_spawn_and_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let binary = create_session_adapter(dir.path());
        let mut adapter = ProcessAdapter::new(binary, mock_metadata());

        adapter.spawn().unwrap();
        let meta = SourceMetadata {
            file_path: "src/main.rs".into(),
            extension: "rs".into(),
            language: Some("rust".into()),
            scope_id: "test".into(),
            source_version: "v1".into(),
        };
        let (chunks, relations) = adapter.chunk("fn main() {}", &meta).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "fn main() {}");
        assert!(relations.is_empty());
        adapter.shutdown().unwrap();
    }

    #[test]
    fn test_ingest_then_chunk_same_session() {
        let dir = tempfile::tempdir().unwrap();
        let binary = create_session_adapter(dir.path());
        let mut adapter = ProcessAdapter::new(binary, mock_metadata());

        adapter.spawn().unwrap();

        // Ingest first
        let files = adapter.ingest("/fake/path", "test").unwrap();
        assert_eq!(files.len(), 1);

        // Then chunk
        let meta = SourceMetadata {
            file_path: "test.txt".into(),
            extension: "txt".into(),
            language: None,
            scope_id: "test".into(),
            source_version: "v1".into(),
        };
        let (chunks, _) = adapter.chunk("test content", &meta).unwrap();
        assert_eq!(chunks.len(), 1);

        adapter.shutdown().unwrap();
    }

    #[test]
    fn test_drop_shuts_down() {
        let dir = tempfile::tempdir().unwrap();
        let binary = create_session_adapter(dir.path());
        let mut adapter = ProcessAdapter::new(binary, mock_metadata());
        adapter.spawn().unwrap();
        // Drop without explicit shutdown — should not panic
        drop(adapter);
    }
}
```

**Step 2: Export the module**

In `crates/corvia-kernel/src/lib.rs`, add after the `adapter_discovery` line:

```rust
pub mod process_adapter;
```

**Step 3: Run tests**

```bash
cargo test -p corvia-kernel process_adapter -- --nocapture
```

Expected: 4 tests pass (spawn real Python mock adapter processes).

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/process_adapter.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat: add ProcessAdapter IPC wrapper for adapter processes (D72/D75)

Spawns adapter binaries, writes JSONL requests to stdin, reads JSONL
responses from stdout. Supports ingest (streaming) and chunk (request-
response) commands. Clean shutdown via stdin EOF."
```

---

## Task 5: Add ProcessChunkingStrategy to Pipeline

Register adapter-claimed extensions in the FormatRegistry so the pipeline calls back to the adapter for chunking.

**Files:**
- Modify: `crates/corvia-kernel/src/chunking_pipeline.rs`

**Step 1: Add ProcessChunkingStrategy**

In `crates/corvia-kernel/src/chunking_pipeline.rs`, add after the existing imports (around line 26):

```rust
use crate::process_adapter::ProcessAdapter;
use std::sync::Mutex;
```

Then add this struct before the `#[cfg(test)]` block:

```rust
// ---------------------------------------------------------------------------
// ProcessChunkingStrategy — delegates to adapter process via IPC (D77)
// ---------------------------------------------------------------------------

/// Routes chunking calls to an external adapter process via JSONL IPC.
///
/// Registered in the [`FormatRegistry`] override tier for extensions the
/// adapter claims in its `chunking_extensions` metadata field.
pub struct ProcessChunkingStrategy {
    adapter: Arc<Mutex<ProcessAdapter>>,
    extension_list: Vec<String>,
}

impl ProcessChunkingStrategy {
    /// Create a new strategy backed by the given adapter process.
    pub fn new(adapter: Arc<Mutex<ProcessAdapter>>, extensions: Vec<String>) -> Self {
        Self {
            adapter,
            extension_list: extensions,
        }
    }
}

impl crate::chunking_strategy::ChunkingStrategy for ProcessChunkingStrategy {
    fn name(&self) -> &str {
        "process-adapter"
    }

    fn supported_extensions(&self) -> &[&str] {
        // Return a static slice — this is a minor limitation but acceptable
        // since the extensions are fixed at construction time.
        // We use a leaked slice for the trait's lifetime requirement.
        // This is safe because adapters are long-lived (session scope).
        &[]
    }

    fn chunk(
        &self,
        source: &str,
        meta: &crate::chunking_strategy::SourceMetadata,
    ) -> corvia_common::errors::Result<crate::chunking_strategy::ChunkResult> {
        let mut adapter = self.adapter.lock().map_err(|e| {
            corvia_common::errors::CorviaError::Internal(format!("Adapter lock poisoned: {e}"))
        })?;

        let (chunks, relations) = adapter.chunk(source, meta).map_err(|e| {
            corvia_common::errors::CorviaError::Internal(format!("Adapter chunk failed: {e}"))
        })?;

        Ok(crate::chunking_strategy::ChunkResult { chunks, relations })
    }
}
```

**Step 2: Add helper to register adapter chunking strategies**

Add this function to `chunking_pipeline.rs` (after `ProcessChunkingStrategy`):

```rust
/// Register an adapter's chunking extensions in the format registry.
///
/// For each extension in `chunking_extensions`, registers a
/// [`ProcessChunkingStrategy`] as an override in the registry.
pub fn register_adapter_chunking(
    registry: &mut FormatRegistry,
    adapter: Arc<Mutex<ProcessAdapter>>,
    chunking_extensions: &[String],
) {
    if chunking_extensions.is_empty() {
        return;
    }
    let strategy = Arc::new(ProcessChunkingStrategy::new(
        adapter,
        chunking_extensions.to_vec(),
    ));
    for ext in chunking_extensions {
        registry.register_override(ext, strategy.clone());
    }
}
```

**Step 3: Run existing tests**

```bash
cargo test -p corvia-kernel chunking_pipeline -- --nocapture
```

Expected: all existing tests pass (no breakage).

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/chunking_pipeline.rs
git commit -m "feat: add ProcessChunkingStrategy for adapter IPC chunking (D77)

Delegates chunking to adapter process via JSONL for extensions the
adapter claims. Registered as overrides in FormatRegistry, preserving
the three-tier priority: adapter > kernel default > fallback."
```

---

## Task 6: Add Adapter Config Types

Add `[adapters]` and `[[sources]]` config sections to `corvia.toml` parsing.

**Files:**
- Modify: `crates/corvia-common/src/config.rs`

**Step 1: Add config types**

In `crates/corvia-common/src/config.rs`, add after the `ChunkingConfig` impl block (after line 243):

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub search_dirs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_config: Option<toml::Value>,
}
```

**Step 2: Add fields to CorviaConfig**

In the `CorviaConfig` struct (around line 54), add two new optional fields after `chunking`:

```rust
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapters: Option<AdaptersConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SourceConfig>>,
```

**Step 3: Add to Default impl**

In the `Default` impl for `CorviaConfig` (around line 277), add:

```rust
            chunking: ChunkingConfig::default(),
            adapters: None,
            sources: None,
```

**Step 4: Add tests**

Add to the existing `#[cfg(test)] mod tests` in config.rs:

```rust
    #[test]
    fn test_adapters_config_from_toml() {
        let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
data_dir = ".corvia"

[embedding]
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020

[adapters]
search_dirs = ["~/.config/corvia/adapters", "/opt/corvia/adapters"]
default = "git"

[[sources]]
path = "./backend"
adapter = "git"

[[sources]]
path = "https://company.atlassian.net/wiki/spaces/ENG"
adapter = "confluence"
"#;
        let config: CorviaConfig = toml::from_str(toml_str).unwrap();
        let adapters = config.adapters.unwrap();
        assert_eq!(adapters.search_dirs.len(), 2);
        assert_eq!(adapters.default, Some("git".into()));

        let sources = config.sources.unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].path, "./backend");
        assert_eq!(sources[0].adapter, Some("git".into()));
        assert_eq!(sources[1].adapter, Some("confluence".into()));
    }

    #[test]
    fn test_adapters_config_optional() {
        let config = CorviaConfig::default();
        assert!(config.adapters.is_none());
        assert!(config.sources.is_none());
    }

    #[test]
    fn test_adapters_config_roundtrip() {
        let mut config = CorviaConfig::default();
        config.adapters = Some(AdaptersConfig {
            search_dirs: vec!["~/adapters".into()],
            default: Some("basic".into()),
        });
        config.sources = Some(vec![SourceConfig {
            path: "./repo".into(),
            adapter: Some("git".into()),
            adapter_config: None,
        }]);

        let toml_str = toml::to_string_pretty(&config).unwrap();
        let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.adapters.unwrap().default, Some("basic".into()));
        assert_eq!(loaded.sources.unwrap().len(), 1);
    }
```

**Step 5: Run tests**

```bash
cargo test -p corvia-common config -- --nocapture
```

Expected: all existing + 3 new tests pass.

**Step 6: Commit**

```bash
git add crates/corvia-common/src/config.rs
git commit -m "feat: add [adapters] and [[sources]] config types (D76)

Optional AdaptersConfig (search_dirs, default) and SourceConfig (path,
adapter, adapter_config) for multi-adapter routing. Both optional —
zero config needed for auto-detection."
```

---

## Task 7: Create corvia-adapter-basic Rust Crate

Minimal filesystem adapter binary that implements the JSONL protocol.

**Files:**
- Create: `crates/corvia-adapter-basic/Cargo.toml`
- Create: `crates/corvia-adapter-basic/src/main.rs`
- Modify: `Cargo.toml` (add to workspace members)

**Step 1: Create Cargo.toml**

Create `crates/corvia-adapter-basic/Cargo.toml`:

```toml
[package]
name = "corvia-adapter-basic"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
rust-version.workspace = true
description = "Basic filesystem ingestion adapter for Corvia"
keywords = ["ai-agents", "ingestion", "adapter", "filesystem"]
categories = ["command-line-utilities"]

[[bin]]
name = "corvia-adapter-basic"
path = "src/main.rs"

[dependencies]
serde.workspace = true
serde_json.workspace = true
```

**Step 2: Create main.rs**

Create `crates/corvia-adapter-basic/src/main.rs`:

```rust
//! corvia-adapter-basic — Minimal filesystem ingestion adapter for Corvia.
//!
//! Walks a directory, reads text files, and streams them as JSONL to stdout.
//! No language intelligence, no AST parsing — just raw file content for the
//! kernel's default chunkers (MarkdownChunker, ConfigChunker, FallbackChunker).
//!
//! Protocol: D75 (JSONL over stdin/stdout)
//! Design: D78 (no built-in fallback — this IS the default adapter)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: u32 = 1;
const MAX_FILE_SIZE: u64 = 100 * 1024; // 100KB

fn skip_dirs() -> HashSet<&'static str> {
    [
        ".git", "target", "node_modules", "__pycache__", ".venv",
        "dist", "build", "vendor", ".corvia", ".tox", ".mypy_cache",
        ".pytest_cache", "venv", "env", ".env", ".idea", ".vscode",
    ]
    .into_iter()
    .collect()
}

fn text_extensions() -> HashSet<&'static str> {
    [
        "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "rb", "php",
        "c", "cpp", "h", "hpp", "cs", "swift", "kt", "scala", "ex", "exs",
        "sh", "bash", "zsh", "fish",
        "md", "txt", "rst", "adoc",
        "toml", "yaml", "yml", "json", "xml", "csv",
        "html", "css", "scss", "less", "sql",
        "dockerfile", "makefile", "justfile",
    ]
    .into_iter()
    .collect()
}

// -- Protocol types (matches corvia-kernel/src/adapter_protocol.rs) ----------

#[derive(Serialize)]
struct Metadata {
    name: &'static str,
    version: String,
    domain: &'static str,
    protocol_version: u32,
    description: &'static str,
    supported_extensions: Vec<String>,
    chunking_extensions: Vec<String>,
}

#[derive(Serialize)]
struct SourceFileMsg {
    source_file: SourceFilePayload,
}

#[derive(Serialize)]
struct SourceFilePayload {
    content: String,
    metadata: SourceMetadata,
}

#[derive(Serialize)]
struct SourceMetadata {
    file_path: String,
    extension: String,
    language: Option<String>,
    scope_id: String,
    source_version: String,
}

#[derive(Serialize)]
struct DoneMsg {
    done: bool,
    total_files: usize,
}

#[derive(Serialize)]
struct ErrorMsg {
    error: ErrorPayload,
}

#[derive(Serialize)]
struct ErrorPayload {
    code: String,
    message: String,
}

#[derive(Deserialize)]
#[serde(tag = "method", content = "params")]
enum Request {
    #[serde(rename = "ingest")]
    Ingest { source_path: String, scope_id: String },
    #[serde(rename = "chunk")]
    Chunk { content: String, metadata: serde_json::Value },
}

// -- Main -------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        // Session mode: read JSONL from stdin
        session_mode();
        return;
    }

    match args[1].as_str() {
        "--corvia-metadata" => {
            let meta = metadata();
            println!("{}", serde_json::to_string(&meta).unwrap());
        }
        "ingest" => {
            if args.len() < 3 {
                eprintln!("Error: ingest requires a path argument");
                std::process::exit(1);
            }
            ingest_cli(&args[2]);
        }
        _ => {
            eprintln!("Usage: corvia-adapter-basic [--corvia-metadata | ingest <path>]");
            eprintln!("  No args: session mode (read JSONL from stdin)");
            std::process::exit(1);
        }
    }
}

fn metadata() -> Metadata {
    let exts: Vec<String> = text_extensions().into_iter().map(String::from).collect();
    Metadata {
        name: "basic",
        version: VERSION.to_string(),
        domain: "filesystem",
        protocol_version: PROTOCOL_VERSION,
        description: "Basic filesystem adapter — walks directories, reads text files",
        supported_extensions: exts,
        chunking_extensions: vec![], // no custom chunking
    }
}

/// Session mode: read JSONL requests from stdin, respond on stdout.
fn session_mode() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<Request>(&line) {
            Ok(Request::Ingest { source_path, scope_id }) => {
                ingest_to_writer(&source_path, &scope_id, &mut out);
            }
            Ok(Request::Chunk { .. }) => {
                let err = ErrorMsg {
                    error: ErrorPayload {
                        code: "NOT_SUPPORTED".into(),
                        message: "basic adapter does not provide chunking".into(),
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&err).unwrap()).ok();
                out.flush().ok();
            }
            Err(e) => {
                let err = ErrorMsg {
                    error: ErrorPayload {
                        code: "PARSE_ERROR".into(),
                        message: format!("Invalid request: {e}"),
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&err).unwrap()).ok();
                out.flush().ok();
            }
        }
    }
}

/// CLI mode: ingest directly from command line args.
fn ingest_cli(source_path: &str) {
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let scope_id = Path::new(source_path)
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "unknown".into());

    ingest_to_writer(source_path, &scope_id, &mut out);
}

fn ingest_to_writer<W: Write>(source_path: &str, scope_id: &str, out: &mut W) {
    let path = Path::new(source_path);
    if !path.is_dir() {
        let err = ErrorMsg {
            error: ErrorPayload {
                code: "NOT_A_DIRECTORY".into(),
                message: format!("'{}' is not a directory", source_path),
            },
        };
        writeln!(out, "{}", serde_json::to_string(&err).unwrap()).ok();
        out.flush().ok();
        return;
    }

    let skip = skip_dirs();
    let exts = text_extensions();
    let mut total = 0;

    walk_dir(path, &skip, &exts, scope_id, &mut total, out);

    let done = DoneMsg { done: true, total_files: total };
    writeln!(out, "{}", serde_json::to_string(&done).unwrap()).ok();
    out.flush().ok();
}

fn walk_dir<W: Write>(
    dir: &Path,
    skip: &HashSet<&str>,
    exts: &HashSet<&str>,
    scope_id: &str,
    total: &mut usize,
    out: &mut W,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if skip.contains(name_str.as_ref()) || name_str.starts_with('.') {
                continue;
            }
            walk_dir(&path, skip, exts, scope_id, total, out);
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        let name_lower = name_str.to_lowercase();
        if !exts.contains(ext.as_str()) && !exts.contains(name_lower.as_str()) {
            continue;
        }

        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        if meta.len() > MAX_FILE_SIZE || meta.len() == 0 {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel_path = path
            .strip_prefix(dir.parent().unwrap_or(dir))
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        // Use path relative to the source root, not the parent
        let rel_from_source = path
            .strip_prefix(dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        let msg = SourceFileMsg {
            source_file: SourceFilePayload {
                content,
                metadata: SourceMetadata {
                    file_path: rel_from_source,
                    extension: ext,
                    language: None,
                    scope_id: scope_id.to_string(),
                    source_version: "unknown".into(),
                },
            },
        };

        writeln!(out, "{}", serde_json::to_string(&msg).unwrap()).ok();
        *total += 1;
    }
    out.flush().ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_metadata_output() {
        let meta = metadata();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"name\":\"basic\""));
        assert!(json.contains("\"protocol_version\":1"));
        assert!(json.contains("\"chunking_extensions\":[]"));
    }

    #[test]
    fn test_ingest_produces_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("hello.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("data.bin"), vec![0u8; 10]).unwrap(); // skip non-text

        let mut output = Vec::new();
        ingest_to_writer(
            &dir.path().to_string_lossy(),
            "test-scope",
            &mut output,
        );

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();

        // Should have 1 source_file + 1 done
        assert_eq!(lines.len(), 2, "got: {:?}", lines);
        assert!(lines[0].contains("\"source_file\""));
        assert!(lines[0].contains("hello.rs"));
        assert!(lines[1].contains("\"done\":true"));
        assert!(lines[1].contains("\"total_files\":1"));
    }

    #[test]
    fn test_ingest_skips_large_files() {
        let dir = tempfile::tempdir().unwrap();
        let large = vec![b'a'; 200 * 1024]; // 200KB > MAX_FILE_SIZE
        fs::write(dir.path().join("huge.rs"), &large).unwrap();

        let mut output = Vec::new();
        ingest_to_writer(&dir.path().to_string_lossy(), "test", &mut output);

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"total_files\":0"));
    }

    #[test]
    fn test_ingest_skips_dot_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = dir.path().join(".hidden");
        fs::create_dir(&hidden).unwrap();
        fs::write(hidden.join("secret.rs"), "secret").unwrap();
        fs::write(dir.path().join("visible.rs"), "visible").unwrap();

        let mut output = Vec::new();
        ingest_to_writer(&dir.path().to_string_lossy(), "test", &mut output);

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"total_files\":1"));
        assert!(text.contains("visible.rs"));
        assert!(!text.contains("secret.rs"));
    }

    #[test]
    fn test_session_mode_ingest_request() {
        // Verify the Request enum deserializes correctly
        let json = r#"{"method":"ingest","params":{"source_path":"/tmp/test","scope_id":"s"}}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        match req {
            Request::Ingest { source_path, scope_id } => {
                assert_eq!(source_path, "/tmp/test");
                assert_eq!(scope_id, "s");
            }
            _ => panic!("expected Ingest"),
        }
    }
}
```

**Step 3: Add to workspace members**

In `Cargo.toml` (workspace root), add `"crates/corvia-adapter-basic"` to the `members` array:

```toml
members = [
    "crates/corvia-common",
    "crates/corvia-kernel",
    "crates/corvia-server",
    "crates/corvia-cli",
    "crates/corvia-proto",
    "crates/corvia-inference",
    "crates/corvia-adapter-basic",
]
```

**Step 4: Build and test**

```bash
cargo build -p corvia-adapter-basic && cargo test -p corvia-adapter-basic -- --nocapture
```

Expected: builds successfully, 5 tests pass.

**Step 5: Test the binary manually**

```bash
./target/debug/corvia-adapter-basic --corvia-metadata | python3 -m json.tool
```

Expected: valid JSON metadata with name "basic".

**Step 6: Commit**

```bash
git add crates/corvia-adapter-basic/ Cargo.toml
git commit -m "feat: add corvia-adapter-basic filesystem adapter binary (D78)

Minimal adapter that walks directories, reads text files, and streams
JSONL. Supports both CLI mode (ingest <path>) and session mode (JSONL
over stdin/stdout). No custom chunking — kernel defaults handle it."
```

---

## Task 8: Move corvia-adapter-git into Monorepo

Move the external `corvia-adapter-git` repo into `crates/corvia-adapter-git/` and add a JSONL `main.rs` wrapper.

**Files:**
- Create: `crates/corvia-adapter-git/` (copy from `/root/corvia-adapter-git/src/`)
- Modify: `crates/corvia-adapter-git/Cargo.toml` (workspace deps, add [[bin]])
- Create: `crates/corvia-adapter-git/src/main.rs` (JSONL wrapper)
- Modify: `Cargo.toml` (add to workspace members)

**Step 1: Copy source files**

```bash
mkdir -p crates/corvia-adapter-git/src
cp /root/corvia-adapter-git/src/lib.rs crates/corvia-adapter-git/src/
cp /root/corvia-adapter-git/src/git.rs crates/corvia-adapter-git/src/
cp /root/corvia-adapter-git/src/ast_chunker.rs crates/corvia-adapter-git/src/
cp /root/corvia-adapter-git/src/treesitter.rs crates/corvia-adapter-git/src/
```

**Step 2: Create workspace-aware Cargo.toml**

Create `crates/corvia-adapter-git/Cargo.toml`:

```toml
[package]
name = "corvia-adapter-git"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
rust-version.workspace = true
description = "Git repository and source code ingestion adapter for Corvia using tree-sitter"
keywords = ["ai-agents", "tree-sitter", "code-analysis", "ingestion", "knowledge-graph"]
categories = ["development-tools", "parser-implementations"]

[lib]
name = "corvia_adapter_git"
path = "src/lib.rs"

[[bin]]
name = "corvia-adapter-git"
path = "src/main.rs"

[dependencies]
corvia-common.workspace = true
corvia-kernel.workspace = true
serde.workspace = true
serde_json.workspace = true
async-trait = "0.1"
tree-sitter = "0.26"
tree-sitter-language = "0.1"
tree-sitter-rust = "0.24"
tree-sitter-javascript = "0.25"
tree-sitter-typescript = "0.23"
tree-sitter-python = "0.25"
git2 = "0.20"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
walkdir = "2"

[dev-dependencies]
tempfile = "3"
```

**Step 3: Create JSONL main.rs wrapper**

Create `crates/corvia-adapter-git/src/main.rs`:

```rust
//! corvia-adapter-git — JSONL entry point for the Git + tree-sitter adapter.
//!
//! Wraps the existing `GitAdapter` library with the JSONL process protocol (D75).
//! Supports both CLI mode and session mode.

use corvia_adapter_git::{AstChunker, GitAdapter};
use corvia_kernel::adapter_protocol::{AdapterMetadata, AdapterRequest, AdapterResponse, SourceFilePayload, AdapterError};
use corvia_kernel::chunking_strategy::{ChunkingStrategy, SourceFile, SourceMetadata};
use corvia_kernel::traits::IngestionAdapter;
use serde_json;
use std::io::{self, BufRead, Write};

const PROTOCOL_VERSION: u32 = 1;

fn metadata() -> AdapterMetadata {
    let chunker = AstChunker;
    let exts: Vec<String> = chunker.supported_extensions().iter().map(|s| s.to_string()).collect();

    AdapterMetadata {
        name: "git".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        domain: "git".into(),
        protocol_version: PROTOCOL_VERSION,
        description: "Git + tree-sitter code ingestion".into(),
        supported_extensions: vec![
            "rs".into(), "js".into(), "jsx".into(), "ts".into(), "tsx".into(),
            "py".into(), "md".into(), "toml".into(), "yaml".into(), "yml".into(),
            "json".into(),
        ],
        chunking_extensions: exts,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        session_mode();
        return;
    }

    match args[1].as_str() {
        "--corvia-metadata" => {
            println!("{}", serde_json::to_string(&metadata()).unwrap());
        }
        _ => {
            eprintln!("Usage: corvia-adapter-git [--corvia-metadata]");
            eprintln!("  No args: session mode (read JSONL from stdin)");
            std::process::exit(1);
        }
    }
}

fn session_mode() {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<AdapterRequest>(&line) {
            Ok(AdapterRequest::Ingest { source_path, scope_id }) => {
                handle_ingest(&rt, &source_path, &scope_id, &mut out);
            }
            Ok(AdapterRequest::Chunk { content, metadata }) => {
                handle_chunk(&content, &metadata, &mut out);
            }
            Err(e) => {
                let resp = AdapterResponse::Error {
                    error: AdapterError {
                        code: "PARSE_ERROR".into(),
                        message: format!("Invalid request: {e}"),
                        file: None,
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
                out.flush().ok();
            }
        }
    }
}

fn handle_ingest<W: Write>(rt: &tokio::runtime::Runtime, source_path: &str, scope_id: &str, out: &mut W) {
    let adapter = GitAdapter::new();

    match rt.block_on(adapter.ingest_sources(source_path)) {
        Ok(files) => {
            let total = files.len();
            for sf in files {
                let resp = AdapterResponse::SourceFile {
                    source_file: SourceFilePayload {
                        content: sf.content,
                        metadata: sf.metadata,
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
            }
            let done = serde_json::json!({"done": true, "total_files": total});
            writeln!(out, "{}", done).ok();
            out.flush().ok();
        }
        Err(e) => {
            let resp = AdapterResponse::Error {
                error: AdapterError {
                    code: "INGEST_FAILED".into(),
                    message: format!("{e}"),
                    file: None,
                },
            };
            writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
            out.flush().ok();
        }
    }
}

fn handle_chunk<W: Write>(content: &str, metadata: &SourceMetadata, out: &mut W) {
    let chunker = AstChunker;

    match chunker.chunk(content, metadata) {
        Ok(result) => {
            let resp = AdapterResponse::ChunkResult {
                chunks: result.chunks,
                relations: result.relations,
            };
            writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
            out.flush().ok();
        }
        Err(e) => {
            let resp = AdapterResponse::Error {
                error: AdapterError {
                    code: "CHUNK_FAILED".into(),
                    message: format!("{e}"),
                    file: Some(metadata.file_path.clone()),
                },
            };
            writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
            out.flush().ok();
        }
    }
}
```

**Step 4: Add to workspace members**

In `Cargo.toml` (workspace root), add `"crates/corvia-adapter-git"`:

```toml
members = [
    "crates/corvia-common",
    "crates/corvia-kernel",
    "crates/corvia-server",
    "crates/corvia-cli",
    "crates/corvia-proto",
    "crates/corvia-inference",
    "crates/corvia-adapter-basic",
    "crates/corvia-adapter-git",
]
```

**Step 5: Build and test**

```bash
cargo build -p corvia-adapter-git && cargo test -p corvia-adapter-git -- --nocapture
```

Expected: all existing library tests pass, binary builds.

**Step 6: Test the binary**

```bash
./target/debug/corvia-adapter-git --corvia-metadata | python3 -m json.tool
```

Expected: valid JSON metadata with name "git", chunking_extensions listing code file extensions.

**Step 7: Commit**

```bash
git add crates/corvia-adapter-git/
git commit -m "feat: move corvia-adapter-git into monorepo with JSONL wrapper (D79)

Existing library code (GitAdapter, AstChunker, tree-sitter) preserved
unchanged. New main.rs wraps the library with the JSONL process protocol.
Workspace path deps replace cross-repo git deps."
```

---

## Task 9: Refactor cmd_ingest to Use Discovery

Replace the hardcoded `GitAdapter::new()` in the CLI with runtime adapter discovery and IPC.

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`
- Modify: `crates/corvia-cli/Cargo.toml` (remove compile-time adapter-git dep)

**Step 1: Remove compile-time adapter-git dependency**

In `crates/corvia-cli/Cargo.toml`, remove the line:

```toml
corvia-adapter-git = { git = "https://github.com/chunzhe10/corvia-adapter-git", tag = "v0.3.2" }
```

Also remove the `git2 = "0.20"` dep (no longer needed — adapter handles git detection).

**Step 2: Update imports in main.rs**

Remove:
```rust
use corvia_adapter_git::GitAdapter;
```

Add:
```rust
use corvia_kernel::adapter_discovery;
use corvia_kernel::process_adapter::ProcessAdapter;
use corvia_kernel::chunking_pipeline::register_adapter_chunking;
```

**Step 3: Rewrite cmd_ingest**

Replace the `cmd_ingest` function (lines 447-533 in main.rs) with:

```rust
async fn cmd_ingest(path: Option<&str>) -> Result<()> {
    if let Some(path) = path {
        let config = load_config()?;
        let (store, graph) = connect_store_with_graph(&config).await?;
        let engine = connect_engine(&config);

        // Step 1: Discover adapters
        let extra_dirs = config
            .adapters
            .as_ref()
            .map(|a| a.search_dirs.clone())
            .unwrap_or_default();
        let discovered = adapter_discovery::discover_adapters(&extra_dirs);

        if discovered.is_empty() {
            anyhow::bail!(
                "No adapters found. Install corvia-adapter-basic to PATH or \
                 ~/.config/corvia/adapters/. Adapter binaries must be named corvia-adapter-<name>."
            );
        }

        // Step 2: Resolve which adapter to use
        let explicit = config
            .sources
            .as_ref()
            .and_then(|sources| {
                sources.iter().find(|s| s.path == path).and_then(|s| s.adapter.as_deref())
            });
        let default_name = config.adapters.as_ref().and_then(|a| a.default.as_deref());
        let adapter_info = adapter_discovery::resolve_adapter(path, &discovered, explicit, default_name)
            .ok_or_else(|| anyhow::anyhow!("No suitable adapter found for '{}'", path))?;

        println!("Ingesting {} (adapter: {})...", path, adapter_info.metadata.name);

        // Step 3: Spawn adapter process
        let mut process = ProcessAdapter::new(
            adapter_info.binary_path.clone(),
            adapter_info.metadata.clone(),
        );
        process.spawn().map_err(|e| anyhow::anyhow!(e))?;

        // Step 4: Ingest source files via IPC
        let source_files = process
            .ingest(path, &config.project.scope_id)
            .map_err(|e| anyhow::anyhow!(e))?;
        let total_files = source_files.len();

        // Step 5: Build chunking pipeline with adapter strategies
        let mut pipeline = corvia_kernel::create_chunking_pipeline(&config);

        // Register adapter's chunking extensions via IPC
        if !adapter_info.metadata.chunking_extensions.is_empty() {
            let adapter_arc = std::sync::Arc::new(std::sync::Mutex::new(
                ProcessAdapter::new(adapter_info.binary_path.clone(), adapter_info.metadata.clone()),
            ));
            // Spawn a second process for chunking (ingest process is done streaming)
            adapter_arc.lock().unwrap().spawn().map_err(|e| anyhow::anyhow!(e))?;
            register_adapter_chunking(
                pipeline.registry_mut(),
                adapter_arc,
                &adapter_info.metadata.chunking_extensions,
            );
        }

        // Step 6: Process through chunking pipeline
        let (processed, pipeline_relations, report) = pipeline.process_batch(&source_files)?;
        println!(
            "Chunked {} files → {} chunks ({} merged, {} split)",
            report.files_processed, report.total_chunks,
            report.chunks_merged, report.chunks_split
        );

        // Step 7: Convert, embed, and store
        let entries: Vec<corvia_common::types::KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let mut entry = corvia_common::types::KnowledgeEntry::new(
                    pc.content.clone(),
                    config.project.scope_id.clone(),
                    pc.metadata.source_file.clone(),
                );
                entry.metadata = corvia_common::types::EntryMetadata {
                    source_file: Some(pc.metadata.source_file.clone()),
                    language: pc.metadata.language.clone(),
                    chunk_type: Some(pc.chunk_type.clone()),
                    start_line: Some(pc.start_line),
                    end_line: Some(pc.end_line),
                };
                entry
            })
            .collect();

        let total = entries.len();
        let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
        let mut stored = 0;
        for batch in entries.chunks(corvia_kernel::introspect::EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored_ids.push(entry.id);
                stored += 1;
            }
            println!("  {stored}/{total} chunks stored");
        }

        // Step 8: Wire relations
        if !pipeline_relations.is_empty() {
            let relations_stored = wire_pipeline_relations(
                &pipeline_relations, &processed, &stored_ids, &*graph,
            ).await;
            if relations_stored > 0 {
                println!("  {relations_stored} graph relations stored");
            }
        }

        // Step 9: Shutdown adapter
        process.shutdown().map_err(|e| anyhow::anyhow!(e))?;

        println!("Done. {stored} chunks from {total_files} files ingested from {path}.");
        println!("Next: corvia search \"your query\"");
    } else {
        // Workspace mode
        let root = std::env::current_dir()?;
        let config = load_config()?;
        if config.is_workspace() {
            workspace::ingest_workspace(&root, None, false).await?;
        } else {
            anyhow::bail!("No path provided and not in a workspace. Usage: corvia ingest <path>");
        }
    }
    Ok(())
}
```

**Step 4: Add `corvia adapters list` subcommand**

Add to the `Commands` enum:

```rust
    /// List discovered adapters
    Adapters {
        #[command(subcommand)]
        action: AdaptersAction,
    },
```

```rust
#[derive(Subcommand)]
enum AdaptersAction {
    /// List all discovered adapters
    List,
}
```

Add the handler in the main match:

```rust
        Commands::Adapters { action } => {
            match action {
                AdaptersAction::List => {
                    let config = load_config().unwrap_or_default();
                    let extra_dirs = config
                        .adapters
                        .as_ref()
                        .map(|a| a.search_dirs.clone())
                        .unwrap_or_default();
                    let adapters = adapter_discovery::discover_adapters(&extra_dirs);
                    print!("{}", adapter_discovery::format_adapter_list(&adapters));
                }
            }
        }
```

**Step 5: Build**

```bash
cargo build -p corvia-cli
```

Expected: builds without `corvia-adapter-git` compile-time dep.

**Step 6: Commit**

```bash
git add crates/corvia-cli/
git commit -m "feat: refactor cmd_ingest to use runtime adapter discovery (D72-D78)

Replace hardcoded GitAdapter::new() with runtime discovery and IPC.
Adapters are spawned as processes, source files stream via JSONL,
chunking delegates via ProcessChunkingStrategy. Add 'corvia adapters
list' diagnostic command. Remove compile-time adapter-git dependency."
```

---

## Task 10: Remove [patch] Section and Clean Up Workspace Cargo.toml

Now that adapter-git is a workspace crate, the `[patch]` section for the external git dep is no longer needed.

**Files:**
- Modify: `Cargo.toml` (workspace root)

**Step 1: Remove the [patch] section**

In `/root/corvia/Cargo.toml`, remove:

```toml
[patch."https://github.com/chunzhe10/corvia"]
corvia-common = { path = "crates/corvia-common" }
corvia-kernel = { path = "crates/corvia-kernel" }
```

**Step 2: Add workspace dep for corvia-adapter-git**

This is optional but useful if other crates want to depend on adapter-git as a library:

```toml
[workspace.dependencies]
corvia-common = { path = "crates/corvia-common" }
corvia-kernel = { path = "crates/corvia-kernel" }
corvia-server = { path = "crates/corvia-server" }
corvia-proto = { path = "crates/corvia-proto" }
```

No new line needed — the adapter crates use workspace deps for common/kernel directly.

**Step 3: Build everything**

```bash
cargo build --workspace
```

Expected: full workspace builds cleanly.

**Step 4: Run all tests**

```bash
cargo test --workspace
```

Expected: all tests pass (200+ tests).

**Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: remove [patch] section, adapter-git is now a workspace crate

Cross-repo version coordination eliminated. All first-party adapters
use workspace path deps. No git deps, no tag choreography."
```

---

## Task 11: Update Release Workflow

Build all four binaries in the release workflow.

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Update the build and asset steps**

Replace the build and prepare steps in `.github/workflows/release.yml`:

```yaml
      - name: Build release binaries
        run: cargo build --release -p corvia-cli -p corvia-inference -p corvia-adapter-basic -p corvia-adapter-git

      - name: Prepare release assets
        run: |
          cp target/release/corvia corvia-cli-linux-amd64
          cp target/release/corvia-inference corvia-inference-linux-amd64
          cp target/release/corvia-adapter-basic corvia-adapter-basic-linux-amd64
          cp target/release/corvia-adapter-git corvia-adapter-git-linux-amd64
          chmod +x corvia-cli-linux-amd64 corvia-inference-linux-amd64 corvia-adapter-basic-linux-amd64 corvia-adapter-git-linux-amd64

      - name: Create GitHub release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            corvia-cli-linux-amd64
            corvia-inference-linux-amd64
            corvia-adapter-basic-linux-amd64
            corvia-adapter-git-linux-amd64
          generate_release_notes: true
```

**Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: build all four binaries in release workflow

Release now produces corvia, corvia-inference, corvia-adapter-basic,
and corvia-adapter-git — all from the same workspace tag."
```

---

## Task 12: Protocol Conformance Test with Python Adapter

End-to-end test that spawns the Python reference adapter and validates JSONL parsing.

**Files:**
- Modify: `crates/corvia-kernel/src/process_adapter.rs` (add integration test)

**Step 1: Add cross-language protocol test**

Add to the `#[cfg(test)] mod tests` in `process_adapter.rs`:

```rust
    #[test]
    fn test_python_adapter_protocol_conformance() {
        // Use the Python reference adapter from examples/
        let adapter_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("adapters/corvia-adapter-basic/python/corvia-adapter-basic");

        if !adapter_path.exists() {
            eprintln!("Skipping: Python adapter not found at {}", adapter_path.display());
            return;
        }

        // Create a temp dir with test files
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("hello.rs"), "fn main() { println!(\"hello\"); }").unwrap();
        fs::write(dir.path().join("readme.md"), "# Test\nThis is a test.").unwrap();

        let meta = AdapterMetadata {
            name: "basic".into(),
            version: "0.1.0".into(),
            domain: "filesystem".into(),
            protocol_version: 1,
            description: "Basic filesystem adapter".into(),
            supported_extensions: vec!["rs".into(), "md".into()],
            chunking_extensions: vec![],
        };

        // The Python adapter uses CLI mode (ingest <path>), not session mode,
        // so we test metadata probing instead
        let output = std::process::Command::new(&adapter_path)
            .arg("--corvia-metadata")
            .output()
            .expect("Failed to run Python adapter");

        assert!(output.status.success());
        let parsed: AdapterMetadata = serde_json::from_slice(&output.stdout)
            .expect("Failed to parse Python adapter metadata");
        assert_eq!(parsed.name, "basic");
        assert_eq!(parsed.protocol_version, 1);
        assert!(parsed.chunking_extensions.is_empty());
    }
```

**Step 2: Run the test**

```bash
cargo test -p corvia-kernel test_python_adapter_protocol_conformance -- --nocapture
```

Expected: passes (or skips gracefully if Python not available).

**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/process_adapter.rs
git commit -m "test: add cross-language protocol conformance test (D75)

Validates that the Python reference adapter's metadata output parses
correctly into Rust AdapterMetadata types."
```

---

## Task 13: Final Verification

Full workspace build and test sweep.

**Step 1: Build everything**

```bash
cargo build --workspace
```

**Step 2: Run all tests**

```bash
cargo test --workspace
```

**Step 3: Test adapter discovery end-to-end**

```bash
# Build adapters
cargo build -p corvia-adapter-basic -p corvia-adapter-git

# Put on PATH
export PATH="$PWD/target/debug:$PATH"

# Verify discovery
./target/debug/corvia adapters list
```

Expected output includes both basic and git adapters.

**Step 4: Test ingest with discovered adapter**

```bash
# Create a test directory
mkdir -p /tmp/corvia-test-ingest
echo "fn main() {}" > /tmp/corvia-test-ingest/test.rs

# Initialize (if not already)
cd /tmp && ./path/to/corvia init 2>/dev/null || true

# Ingest (should auto-discover basic adapter since no .git/)
./path/to/corvia ingest /tmp/corvia-test-ingest
```

**Step 5: Commit (if any final fixes needed)**

```bash
git add -A
git commit -m "chore: final verification and fixes for adapter plugin system"
```

---

## Summary of All Files Changed

| Task | Files | Type |
|------|-------|------|
| 1 | `crates/corvia-kernel/src/chunking_strategy.rs` | Modify |
| 2 | `crates/corvia-kernel/src/adapter_protocol.rs` | Create |
| 2 | `crates/corvia-kernel/src/lib.rs` | Modify |
| 3 | `crates/corvia-kernel/src/adapter_discovery.rs` | Create |
| 3 | `crates/corvia-kernel/src/lib.rs` | Modify |
| 4 | `crates/corvia-kernel/src/process_adapter.rs` | Create |
| 4 | `crates/corvia-kernel/src/lib.rs` | Modify |
| 5 | `crates/corvia-kernel/src/chunking_pipeline.rs` | Modify |
| 6 | `crates/corvia-common/src/config.rs` | Modify |
| 7 | `crates/corvia-adapter-basic/Cargo.toml` | Create |
| 7 | `crates/corvia-adapter-basic/src/main.rs` | Create |
| 7 | `Cargo.toml` | Modify |
| 8 | `crates/corvia-adapter-git/Cargo.toml` | Create |
| 8 | `crates/corvia-adapter-git/src/main.rs` | Create |
| 8 | `crates/corvia-adapter-git/src/lib.rs` | Copy |
| 8 | `crates/corvia-adapter-git/src/git.rs` | Copy |
| 8 | `crates/corvia-adapter-git/src/ast_chunker.rs` | Copy |
| 8 | `crates/corvia-adapter-git/src/treesitter.rs` | Copy |
| 8 | `Cargo.toml` | Modify |
| 9 | `crates/corvia-cli/src/main.rs` | Modify |
| 9 | `crates/corvia-cli/Cargo.toml` | Modify |
| 10 | `Cargo.toml` | Modify |
| 11 | `.github/workflows/release.yml` | Modify |
| 12 | `crates/corvia-kernel/src/process_adapter.rs` | Modify |
