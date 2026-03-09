//! corvia-adapter-git — JSONL entry point for the Git + tree-sitter adapter.
//!
//! Wraps the existing `GitAdapter` library with the JSONL process protocol (D75).
//! Supports both CLI mode and session mode.

use corvia_adapter_git::{AstChunker, GitAdapter};
use corvia_kernel::adapter_protocol::{AdapterMetadata, AdapterRequest, AdapterResponse, SourceFilePayload, AdapterError};
use corvia_kernel::chunking_strategy::{ChunkingStrategy, SourceMetadata};
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

fn handle_ingest<W: Write>(rt: &tokio::runtime::Runtime, source_path: &str, _scope_id: &str, out: &mut W) {
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
