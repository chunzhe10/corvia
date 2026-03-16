# M1: "Point it at your repo, search your code" — Implementation Plan

> **Status:** Shipped (v0.1.0)

**Goal:** Build a working CLI tool that provisions SurrealDB, ingests a Git repo via tree-sitter AST chunking, embeds chunks via Ollama, and returns semantic search results over code.

**Architecture:** Cargo workspace with 4 crates (`corvia-common`, `corvia-kernel`, `corvia-server`, `corvia-cli`) in the main repo + a separate `corvia-adapter-git` repo. The kernel exposes `QueryableStore` and `InferenceEngine` traits. SurrealDB is the queryable store, Ollama provides embeddings. The CLI wires everything together.

**Tech Stack:** Rust 1.89+, SurrealDB 3.0 (Docker), Ollama (local embeddings via nomic-embed-text), tree-sitter (AST parsing), Axum (REST), Clap (CLI), bollard (Docker API), git2 (Git reading), serde/serde_json, tokio, uuid.

**Demo after M1:** `corvia init && corvia ingest ./my-repo && corvia search "how does auth work?"`

**What is NOT in M1:** Agent coordination (M2), MCP server (M2), merge worker (M2), temporal queries as_of/history (M3), graph traversal (M3), watch mode, commit message ingestion, embedding cache in Redb, VS Code extension (M5).

---

## Task 1: Project Scaffold

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/corvia-common/Cargo.toml`
- Create: `crates/corvia-common/src/lib.rs`
- Create: `crates/corvia-kernel/Cargo.toml`
- Create: `crates/corvia-kernel/src/lib.rs`
- Create: `crates/corvia-server/Cargo.toml`
- Create: `crates/corvia-server/src/lib.rs`
- Create: `crates/corvia-cli/Cargo.toml`
- Create: `crates/corvia-cli/src/main.rs`
- Create: `docker/docker-compose.yml`
- Create: `.gitignore`
- Move: brainstorm/design files into `docs/rfcs/`

**Step 1: Initialize git repo and move design docs**

```bash
cd /root/corvia
git init
mv 2026-02-25-corvia-brainstorm.md docs/rfcs/
mv 2026-02-25-corvia-design.md docs/rfcs/
mv 2026-02-27-corvia-v0.2.0-brainstorm.md docs/rfcs/
mv "organizational reasoning memory system v3.pdf" docs/rfcs/
```

**Step 2: Create workspace Cargo.toml**

```toml
# /root/corvia/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/corvia-common",
    "crates/corvia-kernel",
    "crates/corvia-server",
    "crates/corvia-cli",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "AGPL-3.0-only"
repository = "https://github.com/corvia/corvia"

[workspace.dependencies]
corvia-common = { path = "crates/corvia-common" }
corvia-kernel = { path = "crates/corvia-kernel" }
corvia-server = { path = "crates/corvia-server" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
uuid = { version = "1", features = ["v7", "serde"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

**Step 3: Create crate skeletons**

Each crate gets a minimal `Cargo.toml` and `src/lib.rs` (or `src/main.rs` for CLI).

`crates/corvia-common/Cargo.toml`:
```toml
[package]
name = "corvia-common"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
serde.workspace = true
serde_json.workspace = true
uuid.workspace = true
thiserror.workspace = true
chrono = { version = "0.4", features = ["serde"] }
```

`crates/corvia-common/src/lib.rs`:
```rust
pub mod types;
pub mod namespace;
pub mod config;
pub mod errors;
pub mod events;
```

`crates/corvia-kernel/Cargo.toml`:
```toml
[package]
name = "corvia-kernel"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
corvia-common.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
uuid.workspace = true
anyhow.workspace = true
thiserror.workspace = true
tracing.workspace = true
async-trait = "0.1"
surrealdb = { version = "3", features = ["protocol-ws"] }
bollard = "0.19"
```

`crates/corvia-server/Cargo.toml`:
```toml
[package]
name = "corvia-server"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
corvia-common.workspace = true
corvia-kernel.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
uuid.workspace = true
anyhow.workspace = true
tracing.workspace = true
axum = "0.8"
tower-http = { version = "0.6", features = ["cors", "trace"] }
```

`crates/corvia-cli/Cargo.toml`:
```toml
[package]
name = "corvia"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "corvia"
path = "src/main.rs"

[dependencies]
corvia-common.workspace = true
corvia-kernel.workspace = true
corvia-server.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
clap = { version = "4", features = ["derive"] }
```

`crates/corvia-cli/src/main.rs`:
```rust
fn main() {
    println!("corvia — code memory for AI agents");
}
```

**Step 4: Create Docker Compose for SurrealDB**

`docker/docker-compose.yml`:
```yaml
services:
  surrealdb:
    image: surrealdb/surrealdb:v3
    container_name: corvia-surrealdb
    command: start --log=info --user root --pass root
    ports:
      - "8000:8000"
    volumes:
      - corvia-surreal-data:/data
    restart: unless-stopped

volumes:
  corvia-surreal-data:
```

**Step 5: Create .gitignore**

```
/target
.env
*.swp
*.swo
*~
.DS_Store
```

**Step 6: Verify workspace compiles**

Run: `cargo check`
Expected: Compiles with no errors (only warnings about unused modules)

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: initialize cargo workspace with 4 crates and docker config

Corvia M1 scaffold: corvia-common, corvia-kernel, corvia-server, corvia-cli.
Existing design docs moved to docs/rfcs/."
```

---

## Task 2: corvia-common — Core Types

**Files:**
- Create: `crates/corvia-common/src/types.rs`
- Create: `crates/corvia-common/src/errors.rs`
- Test: `crates/corvia-common/src/types.rs` (inline tests)

**Step 1: Write the types with failing tests**

`crates/corvia-common/src/types.rs`:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single unit of knowledge stored in Corvia.
/// Bi-temporal: tracks both when the knowledge was true (valid_from/valid_to)
/// and when it was recorded (recorded_at). See design doc D14.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeEntry {
    pub id: Uuid,
    pub content: String,
    pub source_version: String,
    pub scope_id: String,
    pub workstream: String,
    pub recorded_at: DateTime<Utc>,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub superseded_by: Option<Uuid>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: EntryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EntryMetadata {
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

impl KnowledgeEntry {
    pub fn new(content: String, scope_id: String, source_version: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            content,
            source_version,
            scope_id,
            workstream: "main".to_string(),
            recorded_at: now,
            valid_from: now,
            valid_to: None,
            superseded_by: None,
            embedding: None,
            metadata: EntryMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: EntryMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Returns true if this entry is currently valid (not superseded).
    pub fn is_current(&self) -> bool {
        self.valid_to.is_none() && self.superseded_by.is_none()
    }
}

/// Result from a semantic search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub entry: KnowledgeEntry,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_entry_new_defaults() {
        let entry = KnowledgeEntry::new(
            "fn hello() {}".to_string(),
            "my-repo".to_string(),
            "abc123".to_string(),
        );
        assert_eq!(entry.content, "fn hello() {}");
        assert_eq!(entry.scope_id, "my-repo");
        assert_eq!(entry.workstream, "main");
        assert!(entry.is_current());
        assert!(entry.embedding.is_none());
    }

    #[test]
    fn test_knowledge_entry_with_embedding() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        assert_eq!(entry.embedding.unwrap().len(), 3);
    }

    #[test]
    fn test_knowledge_entry_is_current() {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        assert!(entry.is_current());

        entry.valid_to = Some(Utc::now());
        assert!(!entry.is_current());
    }

    #[test]
    fn test_knowledge_entry_serialization() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: KnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry.id, deserialized.id);
        assert_eq!(entry.content, deserialized.content);
    }
}
```

**Step 2: Write error types**

`crates/corvia-common/src/errors.rs`:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CorviaError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Ingestion error: {0}")]
    Ingestion(String),

    #[error("Docker error: {0}")]
    Docker(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, CorviaError>;
```

**Step 3: Run tests**

Run: `cargo test -p corvia-common`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/corvia-common/src/types.rs crates/corvia-common/src/errors.rs
git commit -m "feat(common): add KnowledgeEntry, SearchResult, and error types

Bi-temporal schema per D14. Builder pattern for optional fields."
```

---

## Task 3: corvia-common — Namespace & Config

**Files:**
- Create: `crates/corvia-common/src/namespace.rs`
- Create: `crates/corvia-common/src/config.rs`
- Create: `crates/corvia-common/src/events.rs`
- Test: inline tests in each file

**Step 1: Write namespace struct with tests (D17)**

`crates/corvia-common/src/namespace.rs`:
```rust
use serde::{Deserialize, Serialize};
use std::fmt;
use crate::errors::{CorviaError, Result};

/// Five-segment hierarchical namespace (D17).
/// Format: {org}:{scope_id}:{workstream}:{source}:{version_ref}
/// Version ref: @{hash} (immutable) or :{label} (mutable)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace {
    pub org: String,
    pub scope: String,
    pub workstream: String,
    pub source: String,
    pub version_ref: VersionRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionRef {
    Immutable(String),
    Mutable(String),
}

impl Namespace {
    pub fn new(org: &str, scope: &str, workstream: &str, source: &str, version_ref: VersionRef) -> Self {
        Self {
            org: org.to_string(),
            scope: scope.to_string(),
            workstream: workstream.to_string(),
            source: source.to_string(),
            version_ref,
        }
    }

    /// Default local namespace for single-user use.
    pub fn local(scope: &str, source: &str) -> Self {
        Self::new("local", scope, "main", source, VersionRef::Mutable("latest".into()))
    }

    /// Parse from colon-separated string at system boundaries.
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(5, ':').collect();
        if parts.len() != 5 {
            return Err(CorviaError::Config(format!(
                "Invalid namespace '{}': expected 5 colon-separated segments", s
            )));
        }
        let version_ref = if parts[4].starts_with('@') {
            VersionRef::Immutable(parts[4][1..].to_string())
        } else if parts[4].starts_with(':') {
            VersionRef::Mutable(parts[4][1..].to_string())
        } else {
            VersionRef::Mutable(parts[4].to_string())
        };
        Ok(Self {
            org: parts[0].to_string(),
            scope: parts[1].to_string(),
            workstream: parts[2].to_string(),
            source: parts[3].to_string(),
            version_ref,
        })
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ref_str = match &self.version_ref {
            VersionRef::Immutable(hash) => format!("@{hash}"),
            VersionRef::Mutable(label) => format!(":{label}"),
        };
        write!(f, "{}:{}:{}:{}{}", self.org, self.scope, self.workstream, self.source, ref_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_local_default() {
        let ns = Namespace::local("my-repo", "code");
        assert_eq!(ns.org, "local");
        assert_eq!(ns.workstream, "main");
        assert_eq!(ns.version_ref, VersionRef::Mutable("latest".into()));
    }

    #[test]
    fn test_namespace_parse_immutable() {
        let ns = Namespace::parse("local:project-alpha:main:my-repo:@abc123").unwrap();
        assert_eq!(ns.org, "local");
        assert_eq!(ns.scope, "project-alpha");
        assert_eq!(ns.workstream, "main");
        assert_eq!(ns.source, "my-repo");
        assert_eq!(ns.version_ref, VersionRef::Immutable("abc123".into()));
    }

    #[test]
    fn test_namespace_parse_mutable() {
        let ns = Namespace::parse("local:project-alpha:main:my-repo::latest").unwrap();
        assert_eq!(ns.version_ref, VersionRef::Mutable("latest".into()));
    }

    #[test]
    fn test_namespace_display_roundtrip() {
        let ns = Namespace::new("acme", "prod", "feature-x", "contracts", VersionRef::Immutable("def456".into()));
        let s = ns.to_string();
        let parsed = Namespace::parse(&s).unwrap();
        assert_eq!(ns, parsed);
    }

    #[test]
    fn test_namespace_parse_invalid() {
        assert!(Namespace::parse("only:two:parts").is_err());
    }
}
```

**Step 2: Write config**

`crates/corvia-common/src/config.rs`:
```rust
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::errors::{CorviaError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorviaConfig {
    pub project: ProjectConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub scope_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub surrealdb_url: String,
    pub surrealdb_ns: String,
    pub surrealdb_db: String,
    pub surrealdb_user: String,
    pub surrealdb_pass: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub url: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for CorviaConfig {
    fn default() -> Self {
        Self {
            project: ProjectConfig {
                name: "default".into(),
                scope_id: "default".into(),
            },
            storage: StorageConfig {
                surrealdb_url: "ws://localhost:8000".into(),
                surrealdb_ns: "corvia".into(),
                surrealdb_db: "main".into(),
                surrealdb_user: "root".into(),
                surrealdb_pass: "root".into(),
            },
            embedding: EmbeddingConfig {
                provider: "ollama".into(),
                model: "nomic-embed-text".into(),
                url: "http://localhost:11434".into(),
                dimensions: 768,
            },
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8020,
            },
        }
    }
}

impl CorviaConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CorviaError::Config(format!("Failed to read {}: {}", path.display(), e)))?;
        toml::from_str(&content)
            .map_err(|e| CorviaError::Config(format!("Failed to parse config: {e}")))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| CorviaError::Config(format!("Failed to serialize config: {e}")))?;
        std::fs::write(path, content)
            .map_err(|e| CorviaError::Config(format!("Failed to write {}: {}", path.display(), e)))?;
        Ok(())
    }

    pub fn config_path() -> PathBuf {
        PathBuf::from("corvia.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = CorviaConfig::default();
        assert_eq!(config.storage.surrealdb_url, "ws://localhost:8000");
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.server.port, 8020);
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let config = CorviaConfig::default();
        let mut file = NamedTempFile::new().unwrap();
        config.save(file.path()).unwrap();
        let loaded = CorviaConfig::load(file.path()).unwrap();
        assert_eq!(config.project.name, loaded.project.name);
        assert_eq!(config.storage.surrealdb_url, loaded.storage.surrealdb_url);
    }
}
```

Note: Add `toml = "0.8"` and `tempfile = "3"` (dev) to `corvia-common/Cargo.toml`:
```toml
[dependencies]
# ... existing deps ...
toml = "0.8"

[dev-dependencies]
tempfile = "3"
```

**Step 3: Write events module (minimal for M1)**

`crates/corvia-common/src/events.rs`:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use crate::errors::{CorviaError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub ts: DateTime<Utc>,
    #[serde(rename = "type")]
    pub event_type: EventType,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    ChunksIndexed,
    SearchPerformed,
    IngestionStarted,
    IngestionCompleted,
}

impl Event {
    pub fn new(event_type: EventType, data: serde_json::Value) -> Self {
        Self {
            ts: Utc::now(),
            event_type,
            data,
        }
    }
}

/// Append an event to events.jsonl
pub fn append_event(path: &Path, event: &Event) -> Result<()> {
    let line = serde_json::to_string(event)
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize event: {e}")))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| CorviaError::Storage(format!("Failed to open events file: {e}")))?;
    writeln!(file, "{line}")
        .map_err(|e| CorviaError::Storage(format!("Failed to write event: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_event_serialization() {
        let event = Event::new(EventType::ChunksIndexed, serde_json::json!({"count": 42}));
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ChunksIndexed"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_append_event() {
        let file = NamedTempFile::new().unwrap();
        let event = Event::new(EventType::SearchPerformed, serde_json::json!({"query": "test"}));
        append_event(file.path(), &event).unwrap();
        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("SearchPerformed"));
    }
}
```

**Step 4: Update lib.rs to export modules**

`crates/corvia-common/src/lib.rs` should already have the module declarations from Task 1.

**Step 5: Run all tests**

Run: `cargo test -p corvia-common`
Expected: All tests pass

**Step 6: Commit**

```bash
git add crates/corvia-common/
git commit -m "feat(common): add namespace (D17), config, and event log

Five-segment namespace with parse/display roundtrip.
TOML config with defaults for SurrealDB, Ollama, and REST server.
Append-only event log for audit trail (D14)."
```

---

## Task 4: corvia-kernel — Trait Definitions

**Files:**
- Create: `crates/corvia-kernel/src/traits.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Define QueryableStore and InferenceEngine traits**

`crates/corvia-kernel/src/traits.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::{KnowledgeEntry, SearchResult};

/// Queryable store for knowledge entries (D15).
/// SurrealDB is the first implementation. Trait-bounded for future swap.
#[async_trait]
pub trait QueryableStore: Send + Sync {
    /// Insert a knowledge entry (must have embedding set).
    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()>;

    /// Semantic search by embedding vector.
    async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<Vec<SearchResult>>;

    /// Get a single entry by ID.
    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>>;

    /// Count entries in a scope.
    async fn count(&self, scope_id: &str) -> Result<u64>;

    /// Initialize the store schema (create tables, indexes).
    async fn init_schema(&self) -> Result<()>;
}

/// Embedding and inference provider (D5).
/// Ollama is the first implementation. Provider-agnostic via trait.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Return the embedding dimension for this model.
    fn dimensions(&self) -> usize;
}

/// Ingestion adapter for domain-specific sources (D11, Section 4).
/// Git/code adapter is the first implementation.
#[async_trait]
pub trait IngestionAdapter: Send + Sync {
    /// Domain name (e.g., "git", "general").
    fn domain(&self) -> &str;

    /// Ingest a source and return knowledge entries (without embeddings).
    /// Embeddings are added by the kernel after ingestion.
    async fn ingest(&self, source_path: &str) -> Result<Vec<KnowledgeEntry>>;
}
```

**Step 2: Update lib.rs**

`crates/corvia-kernel/src/lib.rs`:
```rust
pub mod traits;
pub mod knowledge_store;
pub mod embedding_pipeline;
pub mod docker;
```

**Step 3: Create placeholder modules**

Create empty files so the crate compiles:
- `crates/corvia-kernel/src/knowledge_store.rs`: `// SurrealDB implementation — Task 6`
- `crates/corvia-kernel/src/embedding_pipeline.rs`: `// Ollama implementation — Task 7`
- `crates/corvia-kernel/src/docker.rs`: `// Docker provisioning — Task 5`

**Step 4: Verify it compiles**

Run: `cargo check -p corvia-kernel`
Expected: Compiles (warnings about unused modules are fine)

**Step 5: Commit**

```bash
git add crates/corvia-kernel/
git commit -m "feat(kernel): define QueryableStore, InferenceEngine, and IngestionAdapter traits

Core trait boundaries per D15, D5, D11. No implementations yet."
```

---

## Task 5: Docker Auto-Provisioning

**Files:**
- Create: `crates/corvia-kernel/src/docker.rs`
- Test: inline tests (requires Docker running)

**Step 1: Write Docker provisioning with bollard**

`crates/corvia-kernel/src/docker.rs`:
```rust
use bollard::container::{Config, CreateContainerOptions, ListContainersOptions, StartContainerOptions};
use bollard::image::CreateImageOptions;
use bollard::models::{HostConfig, PortBinding};
use bollard::Docker;
use corvia_common::errors::{CorviaError, Result};
use futures_util::TryStreamExt;
use std::collections::HashMap;
use tracing::{info, warn};

const CONTAINER_NAME: &str = "corvia-surrealdb";
const SURREALDB_IMAGE: &str = "surrealdb/surrealdb:v3";
const SURREALDB_PORT: u16 = 8000;

pub struct DockerProvisioner {
    docker: Docker,
}

impl DockerProvisioner {
    pub fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| CorviaError::Docker(format!("Failed to connect to Docker: {e}")))?;
        Ok(Self { docker })
    }

    /// Check if SurrealDB container is already running.
    pub async fn is_running(&self) -> Result<bool> {
        let filters: HashMap<String, Vec<String>> = HashMap::from([
            ("name".into(), vec![CONTAINER_NAME.into()]),
        ]);
        let options = ListContainersOptions {
            filters,
            ..Default::default()
        };
        let containers = self.docker.list_containers(Some(options)).await
            .map_err(|e| CorviaError::Docker(format!("Failed to list containers: {e}")))?;
        Ok(!containers.is_empty())
    }

    /// Pull the SurrealDB image if not present.
    pub async fn pull_image(&self) -> Result<()> {
        info!("Pulling SurrealDB image: {SURREALDB_IMAGE}");
        let options = CreateImageOptions {
            from_image: SURREALDB_IMAGE,
            ..Default::default()
        };
        self.docker.create_image(Some(options), None, None)
            .try_collect::<Vec<_>>().await
            .map_err(|e| CorviaError::Docker(format!("Failed to pull image: {e}")))?;
        info!("Image pulled successfully");
        Ok(())
    }

    /// Start SurrealDB container. Pulls image if needed.
    pub async fn start(&self, user: &str, pass: &str) -> Result<()> {
        if self.is_running().await? {
            info!("SurrealDB container already running");
            return Ok(());
        }

        // Pull image (idempotent)
        self.pull_image().await?;

        let port_str = format!("{SURREALDB_PORT}/tcp");
        let host_config = HostConfig {
            port_bindings: Some(HashMap::from([(
                port_str.clone(),
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".into()),
                    host_port: Some(SURREALDB_PORT.to_string()),
                }]),
            )])),
            ..Default::default()
        };

        let config = Config {
            image: Some(SURREALDB_IMAGE.to_string()),
            cmd: Some(vec![
                "start".into(),
                "--log=info".into(),
                format!("--user={user}"),
                format!("--pass={pass}"),
            ]),
            exposed_ports: Some(HashMap::from([(port_str, HashMap::new())])),
            host_config: Some(host_config),
            ..Default::default()
        };

        let options = CreateContainerOptions {
            name: CONTAINER_NAME,
            ..Default::default()
        };

        // Remove existing stopped container if present
        let _ = self.docker.remove_container(CONTAINER_NAME, None).await;

        info!("Creating SurrealDB container: {CONTAINER_NAME}");
        self.docker.create_container(Some(options), config).await
            .map_err(|e| CorviaError::Docker(format!("Failed to create container: {e}")))?;

        self.docker.start_container(CONTAINER_NAME, None::<StartContainerOptions<String>>).await
            .map_err(|e| CorviaError::Docker(format!("Failed to start container: {e}")))?;

        info!("SurrealDB started on port {SURREALDB_PORT}");

        // Wait for SurrealDB to be ready
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        Ok(())
    }

    /// Stop and remove the SurrealDB container.
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping SurrealDB container");
        self.docker.stop_container(CONTAINER_NAME, None).await
            .map_err(|e| CorviaError::Docker(format!("Failed to stop container: {e}")))?;
        self.docker.remove_container(CONTAINER_NAME, None).await
            .map_err(|e| CorviaError::Docker(format!("Failed to remove container: {e}")))?;
        Ok(())
    }
}
```

Note: Add `futures-util = "0.3"` to `corvia-kernel/Cargo.toml` dependencies.

**Step 2: Verify it compiles**

Run: `cargo check -p corvia-kernel`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/docker.rs crates/corvia-kernel/Cargo.toml
git commit -m "feat(kernel): add Docker auto-provisioning for SurrealDB via bollard

Handles pull, create, start, stop. Checks if already running."
```

---

## Task 6: corvia-kernel — SurrealDB QueryableStore

**Files:**
- Create: `crates/corvia-kernel/src/knowledge_store.rs`
- Test: `tests/integration/store_test.rs` (requires running SurrealDB)

**Step 1: Write the SurrealDB implementation**

`crates/corvia-kernel/src/knowledge_store.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::{KnowledgeEntry, SearchResult};
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::opt::auth::Root;
use surrealdb::Surreal;
use tracing::info;

pub struct SurrealStore {
    db: Surreal<Client>,
    dimensions: usize,
}

impl SurrealStore {
    pub async fn connect(url: &str, ns: &str, db_name: &str, user: &str, pass: &str, dimensions: usize) -> Result<Self> {
        let db = Surreal::new::<Ws>(url).await
            .map_err(|e| CorviaError::Storage(format!("Failed to connect to SurrealDB: {e}")))?;

        db.signin(Root { username: user, password: pass }).await
            .map_err(|e| CorviaError::Storage(format!("Failed to sign in: {e}")))?;

        db.use_ns(ns).use_db(db_name).await
            .map_err(|e| CorviaError::Storage(format!("Failed to select namespace/db: {e}")))?;

        info!("Connected to SurrealDB at {url} (ns={ns}, db={db_name})");

        Ok(Self { db, dimensions })
    }
}

#[async_trait]
impl super::traits::QueryableStore for SurrealStore {
    async fn init_schema(&self) -> Result<()> {
        let dim = self.dimensions;
        self.db.query(format!(
            "DEFINE TABLE IF NOT EXISTS knowledge SCHEMALESS;
             DEFINE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge
                FIELDS embedding HNSW DIMENSION {dim} DIST COSINE;
             DEFINE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge
                FIELDS scope_id;"
        )).await
            .map_err(|e| CorviaError::Storage(format!("Failed to init schema: {e}")))?;
        info!("Schema initialized (embedding dim={dim})");
        Ok(())
    }

    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()> {
        let id_str = entry.id.to_string();
        self.db.query(
            "CREATE type::thing('knowledge', $id) CONTENT {
                content: $content,
                source_version: $source_version,
                scope_id: $scope_id,
                workstream: $workstream,
                recorded_at: $recorded_at,
                valid_from: $valid_from,
                valid_to: $valid_to,
                superseded_by: $superseded_by,
                embedding: $embedding,
                metadata: $metadata
            };"
        )
        .bind(("id", &id_str))
        .bind(("content", &entry.content))
        .bind(("source_version", &entry.source_version))
        .bind(("scope_id", &entry.scope_id))
        .bind(("workstream", &entry.workstream))
        .bind(("recorded_at", entry.recorded_at.to_rfc3339()))
        .bind(("valid_from", entry.valid_from.to_rfc3339()))
        .bind(("valid_to", entry.valid_to.map(|t| t.to_rfc3339())))
        .bind(("superseded_by", entry.superseded_by.map(|u| u.to_string())))
        .bind(("embedding", &entry.embedding))
        .bind(("metadata", serde_json::to_value(&entry.metadata).unwrap()))
        .await
        .map_err(|e| CorviaError::Storage(format!("Failed to insert entry: {e}")))?;
        Ok(())
    }

    async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let response = self.db.query(
            "SELECT *, vector::distance::knn() AS distance
             FROM knowledge
             WHERE embedding <|$limit,COSINE|> $embedding
             AND scope_id = $scope_id
             ORDER BY distance;"
        )
        .bind(("embedding", embedding.to_vec()))
        .bind(("scope_id", scope_id))
        .bind(("limit", limit))
        .await
        .map_err(|e| CorviaError::Storage(format!("Search failed: {e}")))?;

        // Parse results — SurrealDB returns records that we map to KnowledgeEntry
        let results: Vec<serde_json::Value> = response.take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse search results: {e}")))?;

        let search_results = results.into_iter()
            .filter_map(|v| {
                let distance = v.get("distance")?.as_f64()? as f32;
                let score = 1.0 - distance; // cosine distance to similarity
                let entry: KnowledgeEntry = serde_json::from_value(v.clone()).ok()?;
                Some(SearchResult { entry, score })
            })
            .collect();

        Ok(search_results)
    }

    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>> {
        let id_str = id.to_string();
        let result: Option<KnowledgeEntry> = self.db
            .select(("knowledge", id_str.as_str()))
            .await
            .map_err(|e| CorviaError::Storage(format!("Failed to get entry: {e}")))?;
        Ok(result)
    }

    async fn count(&self, scope_id: &str) -> Result<u64> {
        let response = self.db.query(
            "SELECT count() FROM knowledge WHERE scope_id = $scope_id GROUP ALL;"
        )
        .bind(("scope_id", scope_id))
        .await
        .map_err(|e| CorviaError::Storage(format!("Count failed: {e}")))?;

        let result: Option<serde_json::Value> = response.take(0)
            .map_err(|e| CorviaError::Storage(format!("Failed to parse count: {e}")))?;

        match result {
            Some(v) => Ok(v.get("count").and_then(|c| c.as_u64()).unwrap_or(0)),
            None => Ok(0),
        }
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p corvia-kernel`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/knowledge_store.rs
git commit -m "feat(kernel): implement QueryableStore for SurrealDB

HNSW vector index, bi-temporal fields stored, cosine similarity search.
Schema auto-created on init. Entries stored with full D14 temporal fields."
```

> **Note for implementation:** The SurrealDB query syntax for vector search may need adjustment based on the actual v3.0 API. The implementer should test against a running SurrealDB instance and adapt the queries as needed. The trait contract is what matters — the SQL can be fixed.

---

## Task 7: corvia-kernel — Ollama Embedding Pipeline

**Files:**
- Create: `crates/corvia-kernel/src/embedding_pipeline.rs`
- Test: inline tests (mock HTTP for unit, real Ollama for integration)

**Step 1: Write Ollama InferenceEngine implementation**

`crates/corvia-kernel/src/embedding_pipeline.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

pub struct OllamaEngine {
    url: String,
    model: String,
    dimensions: usize,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEngine {
    pub fn new(url: &str, model: &str, dimensions: usize) -> Self {
        Self {
            url: url.to_string(),
            model: model.to_string(),
            dimensions,
        }
    }
}

#[async_trait]
impl super::traits::InferenceEngine for OllamaEngine {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results.into_iter().next()
            .ok_or_else(|| CorviaError::Embedding("Empty embedding response".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/embed", self.url);
        let request = EmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let response = client.post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| CorviaError::Embedding(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CorviaError::Embedding(format!("Ollama returned {status}: {body}")));
        }

        let embed_response: EmbedResponse = response.json().await
            .map_err(|e| CorviaError::Embedding(format!("Failed to parse response: {e}")))?;

        Ok(embed_response.embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
```

Note: Add `reqwest = { version = "0.12", features = ["json"] }` to `corvia-kernel/Cargo.toml`.

**Step 2: Verify it compiles**

Run: `cargo check -p corvia-kernel`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/embedding_pipeline.rs crates/corvia-kernel/Cargo.toml
git commit -m "feat(kernel): implement InferenceEngine for Ollama

HTTP client for Ollama /api/embed endpoint. Supports single and batch embedding.
Uses nomic-embed-text (768 dimensions) by default."
```

---

## Task 8: corvia-adapter-git — Tree-sitter Chunking

**Files:**
- Create: `/root/corvia-adapter-git/` (new repo)
- Create: `Cargo.toml`, `src/lib.rs`, `src/treesitter.rs`, `src/git.rs`
- Test: inline tests

**Step 1: Initialize the adapter repo**

```bash
mkdir -p /root/corvia-adapter-git/src
cd /root/corvia-adapter-git
git init
```

`/root/corvia-adapter-git/Cargo.toml`:
```toml
[package]
name = "corvia-adapter-git"
version = "0.1.0"
edition = "2024"
license = "AGPL-3.0-only"

[dependencies]
corvia-common = { path = "../corvia/crates/corvia-common" }
corvia-kernel = { path = "../corvia/crates/corvia-kernel" }
async-trait = "0.1"
tree-sitter = "0.26"
tree-sitter-rust = "0.23"
tree-sitter-javascript = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-python = "0.23"
git2 = "0.20"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
walkdir = "2"
```

**Step 2: Write tree-sitter chunker**

`/root/corvia-adapter-git/src/treesitter.rs`:
```rust
use corvia_common::types::{EntryMetadata, KnowledgeEntry};
use tree_sitter::{Parser, Query, QueryCursor};
use tracing::debug;

/// Supported languages and their tree-sitter grammars + queries.
struct LangConfig {
    language: tree_sitter::Language,
    /// Query to match top-level constructs (functions, classes, structs, etc.)
    query: &'static str,
}

fn lang_config_for(extension: &str) -> Option<LangConfig> {
    match extension {
        "rs" => Some(LangConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            query: "(function_item) @chunk
                    (struct_item) @chunk
                    (enum_item) @chunk
                    (impl_item) @chunk
                    (trait_item) @chunk
                    (mod_item) @chunk",
        }),
        "js" | "jsx" => Some(LangConfig {
            language: tree_sitter_javascript::LANGUAGE.into(),
            query: "(function_declaration) @chunk
                    (class_declaration) @chunk
                    (export_statement) @chunk
                    (lexical_declaration) @chunk",
        }),
        "ts" | "tsx" => Some(LangConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            query: "(function_declaration) @chunk
                    (class_declaration) @chunk
                    (export_statement) @chunk
                    (interface_declaration) @chunk
                    (type_alias_declaration) @chunk
                    (lexical_declaration) @chunk",
        }),
        "py" => Some(LangConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            query: "(function_definition) @chunk
                    (class_definition) @chunk",
        }),
        _ => None,
    }
}

/// A chunk of code extracted from a source file via tree-sitter AST parsing.
pub struct CodeChunk {
    pub content: String,
    pub file_path: String,
    pub language: String,
    pub chunk_type: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Parse a source file and extract AST-aware chunks.
/// Falls back to full-file chunk if language is unsupported.
pub fn chunk_file(file_path: &str, source: &str, extension: &str) -> Vec<CodeChunk> {
    let Some(config) = lang_config_for(extension) else {
        // Unsupported language: return entire file as one chunk
        let line_count = source.lines().count() as u32;
        return vec![CodeChunk {
            content: source.to_string(),
            file_path: file_path.to_string(),
            language: extension.to_string(),
            chunk_type: "file".to_string(),
            start_line: 1,
            end_line: line_count,
        }];
    };

    let mut parser = Parser::new();
    if parser.set_language(&config.language).is_err() {
        return vec![];
    }

    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let Ok(query) = Query::new(&config.language, config.query) else {
        return vec![];
    };

    let mut cursor = QueryCursor::new();
    let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

    let mut chunks = Vec::new();
    for m in matches {
        for capture in m.captures {
            let node = capture.node;
            let content = &source[node.byte_range()];
            // Skip very small chunks (one-liners that are trivial)
            if content.lines().count() < 2 {
                continue;
            }
            chunks.push(CodeChunk {
                content: content.to_string(),
                file_path: file_path.to_string(),
                language: extension.to_string(),
                chunk_type: node.kind().to_string(),
                start_line: node.start_position().row as u32 + 1,
                end_line: node.end_position().row as u32 + 1,
            });
        }
    }

    // If no AST chunks found (e.g., file with only imports), return whole file
    if chunks.is_empty() {
        let line_count = source.lines().count() as u32;
        chunks.push(CodeChunk {
            content: source.to_string(),
            file_path: file_path.to_string(),
            language: extension.to_string(),
            chunk_type: "file".to_string(),
            start_line: 1,
            end_line: line_count,
        });
    }

    chunks
}

impl CodeChunk {
    /// Convert to a KnowledgeEntry (without embedding — kernel adds that).
    pub fn to_knowledge_entry(&self, scope_id: &str, source_version: &str) -> KnowledgeEntry {
        KnowledgeEntry::new(
            self.content.clone(),
            scope_id.to_string(),
            source_version.to_string(),
        ).with_metadata(EntryMetadata {
            source_file: Some(self.file_path.clone()),
            language: Some(self.language.clone()),
            chunk_type: Some(self.chunk_type.clone()),
            start_line: Some(self.start_line),
            end_line: Some(self.end_line),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_rust_function() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}
"#;
        let chunks = chunk_file("src/main.rs", source, "rs");
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[1].content.contains("world"));
        assert_eq!(chunks[0].chunk_type, "function_item");
    }

    #[test]
    fn test_chunk_python_class() {
        let source = r#"
class MyClass:
    def method(self):
        pass

def standalone():
    return 42
"#;
        let chunks = chunk_file("app.py", source, "py");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_unsupported_language() {
        let source = "some content\nin a file\nwith multiple lines";
        let chunks = chunk_file("data.txt", source, "txt");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "file");
    }

    #[test]
    fn test_chunk_to_knowledge_entry() {
        let chunk = CodeChunk {
            content: "fn test() { 42 }".into(),
            file_path: "src/lib.rs".into(),
            language: "rs".into(),
            chunk_type: "function_item".into(),
            start_line: 1,
            end_line: 1,
        };
        let entry = chunk.to_knowledge_entry("my-repo", "abc123");
        assert_eq!(entry.scope_id, "my-repo");
        assert_eq!(entry.metadata.source_file.unwrap(), "src/lib.rs");
        assert_eq!(entry.metadata.language.unwrap(), "rs");
    }
}
```

**Step 3: Run tests**

Run: `cd /root/corvia-adapter-git && cargo test`
Expected: All tests pass

**Step 4: Commit**

```bash
cd /root/corvia-adapter-git
git add -A
git commit -m "feat: tree-sitter AST chunking for Rust, JS, TS, Python

Extracts functions, classes, structs, enums, traits, impls as chunks.
Falls back to full-file for unsupported languages.
Converts chunks to KnowledgeEntry for kernel consumption."
```

---

## Task 9: corvia-adapter-git — Git Repo Ingestion

**Files:**
- Create: `/root/corvia-adapter-git/src/git.rs`
- Modify: `/root/corvia-adapter-git/src/lib.rs`
- Test: inline tests

**Step 1: Write git repo reader + IngestionAdapter impl**

`/root/corvia-adapter-git/src/git.rs`:
```rust
use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::traits::IngestionAdapter;
use git2::Repository;
use std::path::Path;
use tracing::{info, debug, warn};
use walkdir::WalkDir;

use crate::treesitter;

pub struct GitAdapter;

impl GitAdapter {
    pub fn new() -> Self {
        Self
    }
}

/// File extensions we attempt to parse.
const SUPPORTED_EXTENSIONS: &[&str] = &["rs", "js", "jsx", "ts", "tsx", "py", "md", "toml", "yaml", "yml", "json"];

/// Directories to skip during ingestion.
const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", ".corvia", "dist", "build", "__pycache__", ".venv", "vendor"];

#[async_trait]
impl IngestionAdapter for GitAdapter {
    fn domain(&self) -> &str {
        "git"
    }

    async fn ingest(&self, source_path: &str) -> Result<Vec<KnowledgeEntry>> {
        let path = Path::new(source_path);
        if !path.exists() {
            return Err(CorviaError::Ingestion(format!("Path does not exist: {source_path}")));
        }

        // Try to get git info for source_version
        let source_version = get_head_sha(path).unwrap_or_else(|| "unknown".to_string());
        let scope_id = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("Ingesting {} (version: {}, scope: {})", source_path, source_version, scope_id);

        let mut entries = Vec::new();

        for entry in WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !SKIP_DIRS.iter().any(|skip| name == *skip)
            })
        {
            let entry = entry.map_err(|e| CorviaError::Ingestion(format!("Walk error: {e}")))?;
            if !entry.file_type().is_file() {
                continue;
            }

            let file_path = entry.path();
            let extension = file_path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if !SUPPORTED_EXTENSIONS.contains(&extension) {
                continue;
            }

            let Ok(source) = std::fs::read_to_string(file_path) else {
                debug!("Skipping binary or unreadable file: {}", file_path.display());
                continue;
            };

            // Skip very large files (>100KB)
            if source.len() > 100_000 {
                warn!("Skipping large file ({}KB): {}", source.len() / 1024, file_path.display());
                continue;
            }

            let relative_path = file_path.strip_prefix(path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            let chunks = treesitter::chunk_file(&relative_path, &source, extension);
            for chunk in chunks {
                entries.push(chunk.to_knowledge_entry(&scope_id, &source_version));
            }
        }

        info!("Ingested {} chunks from {}", entries.len(), source_path);
        Ok(entries)
    }
}

fn get_head_sha(path: &Path) -> Option<String> {
    let repo = Repository::discover(path).ok()?;
    let head = repo.head().ok()?;
    let commit = head.peel_to_commit().ok()?;
    Some(commit.id().to_string()[..8].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ingest_nonexistent_path() {
        let adapter = GitAdapter::new();
        let result = adapter.ingest("/nonexistent/path").await;
        assert!(result.is_err());
    }
}
```

**Step 2: Write lib.rs**

`/root/corvia-adapter-git/src/lib.rs`:
```rust
pub mod treesitter;
pub mod git;

pub use git::GitAdapter;
```

**Step 3: Run tests**

Run: `cd /root/corvia-adapter-git && cargo test`
Expected: All tests pass

**Step 4: Commit**

```bash
cd /root/corvia-adapter-git
git add src/git.rs src/lib.rs
git commit -m "feat: Git repo ingestion via walkdir + tree-sitter

Walks repo, skips common noise dirs (target, node_modules, .git).
Chunks supported files via tree-sitter, falls back to full-file.
Gets HEAD SHA as source_version from git2."
```

---

## Task 10: corvia-server — REST API

**Files:**
- Create: `crates/corvia-server/src/rest.rs`
- Modify: `crates/corvia-server/src/lib.rs`

**Step 1: Write REST API routes**

`crates/corvia-server/src/rest.rs`:
```rust
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use corvia_common::types::{KnowledgeEntry, SearchResult};
use corvia_kernel::traits::{InferenceEngine, QueryableStore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

pub struct AppState {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
}

#[derive(Deserialize)]
pub struct WriteRequest {
    pub content: String,
    pub scope_id: String,
    pub source_version: Option<String>,
    pub metadata: Option<corvia_common::types::EntryMetadata>,
}

#[derive(Serialize)]
pub struct WriteResponse {
    pub id: String,
    pub embedded: bool,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub scope_id: String,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultDto>,
    pub count: usize,
}

#[derive(Serialize)]
pub struct SearchResultDto {
    pub content: String,
    pub score: f32,
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

impl From<SearchResult> for SearchResultDto {
    fn from(r: SearchResult) -> Self {
        Self {
            content: r.entry.content,
            score: r.score,
            source_file: r.entry.metadata.source_file,
            language: r.entry.metadata.language,
            chunk_type: r.entry.metadata.chunk_type,
            start_line: r.entry.metadata.start_line,
            end_line: r.entry.metadata.end_line,
        }
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/memories/write", post(write_memory))
        .route("/v1/memories/search", post(search_memories))
        .route("/health", get(health))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn write_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WriteRequest>,
) -> std::result::Result<Json<WriteResponse>, (StatusCode, String)> {
    // Generate embedding
    let embedding = state.engine.embed(&req.content).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;

    // Create entry
    let mut entry = KnowledgeEntry::new(
        req.content,
        req.scope_id,
        req.source_version.unwrap_or_else(|| "manual".into()),
    ).with_embedding(embedding);

    if let Some(metadata) = req.metadata {
        entry = entry.with_metadata(metadata);
    }

    let id = entry.id.to_string();

    // Store
    state.store.insert(&entry).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Storage failed: {e}")))?;

    Ok(Json(WriteResponse { id, embedded: true }))
}

async fn search_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> std::result::Result<Json<SearchResponse>, (StatusCode, String)> {
    let limit = req.limit.unwrap_or(10);

    // Embed the query
    let query_embedding = state.engine.embed(&req.query).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;

    // Search
    let results = state.store.search(&query_embedding, &req.scope_id, limit).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Search failed: {e}")))?;

    let count = results.len();
    let results: Vec<SearchResultDto> = results.into_iter().map(Into::into).collect();

    Ok(Json(SearchResponse { results, count }))
}
```

**Step 2: Update lib.rs**

`crates/corvia-server/src/lib.rs`:
```rust
pub mod rest;
```

**Step 3: Verify it compiles**

Run: `cargo check -p corvia-server`
Expected: Compiles

**Step 4: Commit**

```bash
git add crates/corvia-server/
git commit -m "feat(server): REST API with /v1/memories/write and /v1/memories/search

Axum routes. Write embeds content via InferenceEngine then stores.
Search embeds query then performs vector similarity search.
/health endpoint for liveness checks."
```

---

## Task 11: corvia-cli — CLI Commands

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`
- Modify: `crates/corvia-cli/Cargo.toml` (add corvia-adapter-git dependency)

**Step 1: Write CLI with all M1 subcommands**

Update `crates/corvia-cli/Cargo.toml` to add adapter dependency:
```toml
[dependencies]
# ... existing deps ...
corvia-adapter-git = { path = "../../../corvia-adapter-git" }
```

`crates/corvia-cli/src/main.rs`:
```rust
use anyhow::Result;
use clap::{Parser, Subcommand};
use corvia_common::config::CorviaConfig;
use corvia_kernel::docker::DockerProvisioner;
use corvia_kernel::embedding_pipeline::OllamaEngine;
use corvia_kernel::knowledge_store::SurrealStore;
use corvia_kernel::traits::{InferenceEngine, IngestionAdapter, QueryableStore};
use corvia_adapter_git::GitAdapter;
use std::sync::Arc;
use tracing::info;

#[derive(Parser)]
#[command(name = "corvia")]
#[command(about = "Code memory for AI agents — point, ingest, search")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Corvia: provision SurrealDB, create config
    Init,

    /// Start the REST API server
    Serve,

    /// Ingest a Git repository
    Ingest {
        /// Path to the repository to ingest
        path: String,
    },

    /// Search ingested knowledge
    Search {
        /// The search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Show status of Corvia services
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corvia=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init => cmd_init().await?,
        Commands::Serve => cmd_serve().await?,
        Commands::Ingest { path } => cmd_ingest(&path).await?,
        Commands::Search { query, limit } => cmd_search(&query, limit).await?,
        Commands::Status => cmd_status().await?,
    }

    Ok(())
}

async fn cmd_init() -> Result<()> {
    println!("Initializing Corvia...");

    // Create default config
    let config = CorviaConfig::default();
    let config_path = CorviaConfig::config_path();
    if !config_path.exists() {
        config.save(&config_path)?;
        println!("Created {}", config_path.display());
    } else {
        println!("Config already exists: {}", config_path.display());
    }

    // Provision SurrealDB
    let docker = DockerProvisioner::new()?;
    docker.start(&config.storage.surrealdb_user, &config.storage.surrealdb_pass).await?;

    // Connect and init schema
    let store = connect_store(&config).await?;
    store.init_schema().await?;

    println!("Corvia initialized. SurrealDB running on port 8000.");
    println!("Next: corvia ingest <path-to-repo>");
    Ok(())
}

async fn cmd_serve() -> Result<()> {
    let config = load_config()?;
    let store = Arc::new(connect_store(&config).await?) as Arc<dyn QueryableStore>;
    let engine = Arc::new(connect_engine(&config)) as Arc<dyn InferenceEngine>;

    let state = Arc::new(corvia_server::rest::AppState { store, engine });
    let app = corvia_server::rest::router(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    println!("Corvia server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn cmd_ingest(path: &str) -> Result<()> {
    let config = load_config()?;
    let store = connect_store(&config).await?;
    let engine = connect_engine(&config);

    let adapter = GitAdapter::new();
    println!("Ingesting {}...", path);

    let entries = adapter.ingest(path).await?;
    let total = entries.len();
    println!("Parsed {} chunks. Embedding and storing...", total);

    // Embed and store in batches
    let batch_size = 32;
    let mut stored = 0;
    for batch in entries.chunks(batch_size) {
        let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
        let embeddings = engine.embed_batch(&texts).await?;

        for (entry, embedding) in batch.iter().zip(embeddings) {
            let mut entry = entry.clone();
            entry.embedding = Some(embedding);
            store.insert(&entry).await?;
            stored += 1;
        }
        println!("  {stored}/{total} chunks stored");
    }

    println!("Done. {stored} chunks ingested from {path}.");
    println!("Next: corvia search \"your query\"");
    Ok(())
}

async fn cmd_search(query: &str, limit: usize) -> Result<()> {
    let config = load_config()?;
    let store = connect_store(&config).await?;
    let engine = connect_engine(&config);

    let embedding = engine.embed(query).await?;
    let results = store.search(&embedding, &config.project.scope_id, limit).await?;

    if results.is_empty() {
        println!("No results found for: {query}");
        return Ok(());
    }

    println!("Found {} results for: {query}\n", results.len());
    for (i, result) in results.iter().enumerate() {
        let file = result.entry.metadata.source_file.as_deref().unwrap_or("unknown");
        let lines = match (result.entry.metadata.start_line, result.entry.metadata.end_line) {
            (Some(s), Some(e)) => format!(":{s}-{e}"),
            _ => String::new(),
        };
        println!("--- Result {} (score: {:.3}) ---", i + 1, result.score);
        println!("File: {file}{lines}");
        // Show first 5 lines of content
        let preview: String = result.entry.content.lines().take(5).collect::<Vec<_>>().join("\n");
        println!("{preview}");
        if result.entry.content.lines().count() > 5 {
            println!("  ...");
        }
        println!();
    }

    Ok(())
}

async fn cmd_status() -> Result<()> {
    let docker = DockerProvisioner::new()?;
    let running = docker.is_running().await?;
    println!("SurrealDB: {}", if running { "running" } else { "stopped" });

    if running {
        let config = load_config()?;
        let store = connect_store(&config).await?;
        let count = store.count(&config.project.scope_id).await?;
        println!("Entries in scope '{}': {count}", config.project.scope_id);
    }

    Ok(())
}

fn load_config() -> Result<CorviaConfig> {
    let path = CorviaConfig::config_path();
    if !path.exists() {
        anyhow::bail!("No corvia.toml found. Run 'corvia init' first.");
    }
    Ok(CorviaConfig::load(&path)?)
}

async fn connect_store(config: &CorviaConfig) -> Result<SurrealStore> {
    Ok(SurrealStore::connect(
        &config.storage.surrealdb_url,
        &config.storage.surrealdb_ns,
        &config.storage.surrealdb_db,
        &config.storage.surrealdb_user,
        &config.storage.surrealdb_pass,
        config.embedding.dimensions,
    ).await?)
}

fn connect_engine(config: &CorviaConfig) -> OllamaEngine {
    OllamaEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    )
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p corvia`
Expected: Compiles

**Step 3: Commit**

```bash
cd /root/corvia
git add crates/corvia-cli/
git commit -m "feat(cli): implement init, serve, ingest, search, status commands

init: provisions SurrealDB via Docker, creates config, inits schema.
ingest: walks repo via GitAdapter, embeds via Ollama, stores in SurrealDB.
search: embeds query, vector search, displays results with file/line info.
serve: starts Axum REST API on :8020.
status: checks SurrealDB container and entry count."
```

---

## Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/integration/e2e_test.rs` (in main repo)

**Step 1: Write the end-to-end test**

This test requires: Docker running, SurrealDB available, Ollama running with nomic-embed-text.

`tests/integration/e2e_test.rs`:
```rust
//! End-to-end integration test for M1.
//! Requires: Docker running, Ollama running with nomic-embed-text pulled.
//!
//! Run with: cargo test --test e2e_test -- --nocapture

use corvia_common::config::CorviaConfig;
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::docker::DockerProvisioner;
use corvia_kernel::embedding_pipeline::OllamaEngine;
use corvia_kernel::knowledge_store::SurrealStore;
use corvia_kernel::traits::{InferenceEngine, QueryableStore};

#[tokio::test]
async fn test_write_and_search() {
    // Use test namespace to avoid polluting real data
    let config = CorviaConfig::default();

    let store = SurrealStore::connect(
        &config.storage.surrealdb_url,
        "corvia_test",
        "e2e",
        &config.storage.surrealdb_user,
        &config.storage.surrealdb_pass,
        config.embedding.dimensions,
    ).await.expect("Failed to connect to SurrealDB — is it running?");

    store.init_schema().await.expect("Failed to init schema");

    let engine = OllamaEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    // Embed and store a test entry
    let content = "fn authenticate(token: &str) -> Result<User> { verify_jwt(token) }";
    let embedding = engine.embed(content).await
        .expect("Failed to embed — is Ollama running with nomic-embed-text?");

    assert_eq!(embedding.len(), config.embedding.dimensions);

    let entry = KnowledgeEntry::new(
        content.to_string(),
        "test-repo".to_string(),
        "abc123".to_string(),
    ).with_embedding(embedding);

    store.insert(&entry).await.expect("Failed to insert");

    // Search for it
    let query_embedding = engine.embed("how does authentication work?").await.unwrap();
    let results = store.search(&query_embedding, "test-repo", 5).await.unwrap();

    assert!(!results.is_empty(), "Expected at least one search result");
    assert!(results[0].entry.content.contains("authenticate"));

    println!("E2E test passed: wrote 1 entry, searched, found it.");
    println!("Score: {:.3}", results[0].score);
}
```

Add to `Cargo.toml` workspace root:
```toml
[[test]]
name = "e2e_test"
path = "tests/integration/e2e_test.rs"
```

**Step 2: Run the test (requires infrastructure)**

Run: `cargo test --test e2e_test -- --nocapture`
Expected: PASS (if Docker + SurrealDB + Ollama are running)

**Step 3: Commit**

```bash
git add tests/ Cargo.toml
git commit -m "test: end-to-end integration test for M1

Verifies: connect to SurrealDB, embed via Ollama, insert, vector search.
Requires Docker + SurrealDB + Ollama running."
```

---

## Task Summary

| Task | What | Files | Depends on |
|------|------|-------|-----------|
| 1 | Project scaffold | Cargo workspace, docker-compose, .gitignore | — |
| 2 | Core types | KnowledgeEntry, SearchResult, errors | 1 |
| 3 | Namespace, config, events | Namespace (D17), TOML config, event log | 1 |
| 4 | Kernel traits | QueryableStore, InferenceEngine, IngestionAdapter | 2 |
| 5 | Docker provisioning | bollard SurrealDB management | 4 |
| 6 | SurrealDB store | QueryableStore impl, HNSW index, bi-temporal | 4 |
| 7 | Ollama embeddings | InferenceEngine impl, HTTP client | 4 |
| 8 | Tree-sitter chunking | AST parsing for Rust/JS/TS/Python | 2 |
| 9 | Git repo ingestion | walkdir + tree-sitter + IngestionAdapter | 8 |
| 10 | REST API | Axum /v1/memories/write and /search | 6, 7 |
| 11 | CLI commands | init, serve, ingest, search, status | 5, 6, 7, 9, 10 |
| 12 | E2E test | Full pipeline integration test | 11 |

**Parallelism:** Tasks 5/6/7 can be developed in parallel (all depend on traits from Task 4). Tasks 8/9 can be developed in parallel with 5/6/7 (adapter has no kernel dependency beyond common types).

## Prerequisites

Before starting implementation:
1. **Install Rust 1.89+**: `rustup update`
2. **Install Docker**: Must be running for SurrealDB
3. **Install Ollama**: `curl -fsSL https://ollama.com/install.sh | sh`
4. **Pull embedding model**: `ollama pull nomic-embed-text`
5. **Verify Ollama running**: `curl http://localhost:11434/api/tags`
