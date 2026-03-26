//! JSONL wire protocol types for adapter <-> host communication (D75).
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
// Host -> Adapter requests (written to stdin as JSON lines)
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
// Adapter -> Host responses (read from stdout, one JSON line per message)
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
                workstream: None,
                content_role: None,
                source_origin: None,
                    parent_session_id: None,
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
        let json = r#"{"name":"basic","version":"0.1.0","domain":"filesystem","protocol_version":1,"description":"Basic filesystem adapter","supported_extensions":["rs","py"],"chunking_extensions":[]}"#;
        let meta: AdapterMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.name, "basic");
        assert!(meta.chunking_extensions.is_empty());
    }

    #[test]
    fn test_python_adapter_ingest_response_compat() {
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
