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
    Chunk { #[allow(dead_code)] content: String, #[allow(dead_code)] metadata: serde_json::Value },
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

        let _rel_path = path
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
