//! IPC wrapper for adapter processes (D72/D75).
//!
//! [`ProcessAdapter`] manages the lifecycle of an adapter binary: spawning it,
//! writing JSONL requests to its stdin, and reading JSONL responses from stdout.
//! One process per adapter per ingestion session.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use tracing::{debug, warn};

use crate::adapter_protocol::{AdapterMetadata, AdapterRequest, AdapterResponse};
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
    use std::path::Path;

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
            workstream: None,
            content_role: None,
            source_origin: None,
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
            workstream: None,
            content_role: None,
            source_origin: None,
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

    #[test]
    fn test_python_adapter_protocol_conformance() {
        // Use the Python reference adapter
        let adapter_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("adapters/corvia-adapter-basic/python/corvia-adapter-basic");

        if !adapter_path.exists() {
            eprintln!("Skipping: Python adapter not found at {}", adapter_path.display());
            return;
        }

        // Probe metadata via --corvia-metadata
        let output = std::process::Command::new(&adapter_path)
            .arg("--corvia-metadata")
            .output()
            .expect("Failed to run Python adapter");

        assert!(output.status.success(), "Python adapter should exit 0 for --corvia-metadata");
        let parsed: AdapterMetadata = serde_json::from_slice(&output.stdout)
            .expect("Failed to parse Python adapter metadata");
        assert_eq!(parsed.name, "basic");
        assert_eq!(parsed.protocol_version, 1);
        assert!(parsed.chunking_extensions.is_empty());
    }
}
