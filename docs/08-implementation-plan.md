# Corvia V1.0.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build corvia v1.0.0 -- a local-first organizational memory for AI agents with 5 CLI commands and 3 MCP tools.

**Architecture:** Two crates (corvia-core + corvia-cli). Flat files as source of truth, Redb for indexes. Hybrid BM25+vector search with cross-encoder reranking. stdio MCP server via rmcp.

**Tech Stack:** Rust 1.86, tantivy 0.22, hnsw_rs 0.3, fastembed 4, rmcp 0.1, redb 2, uuid 1, clap 4

**Spec:** `docs/corvia2/07-open-questions-design.md`

**Working directory:** `repos/corvia2/`

---

## File Map

### corvia-core/src/

| File | Responsibility |
|------|---------------|
| `lib.rs` | Public API re-exports |
| `types.rs` | Core types: Entry, Kind, Chunk, SearchResult, QualitySignal, WriteResponse, StatusResponse |
| `config.rs` | Parse corvia.toml, defaults, validation |
| `entry.rs` | Entry file I/O: serialize/deserialize TOML frontmatter + markdown, atomic write |
| `chunk.rs` | Frontmatter stripping, semantic sub-splitting, overlap, merge small chunks |
| `embed.rs` | fastembed wrapper: embed text, rerank pairs, model download |
| `index.rs` | Redb tables: vectors, chunk-to-entry mapping, supersession state, drift detection |
| `tantivy_index.rs` | Tantivy BM25 index: create, add docs, query with kind filtering |
| `ingest.rs` | Ingest pipeline: scan entries -> parse -> chunk -> embed -> index |
| `search.rs` | Search pipeline: BM25 + vector -> RRF fusion -> rerank -> quality signal |
| `write.rs` | Write pipeline: embed -> dedup check -> create entry -> update indexes |

### corvia-cli/src/

| File | Responsibility |
|------|---------------|
| `main.rs` | Clap CLI: 5 commands, dispatch to handlers |
| `mcp.rs` | rmcp stdio server: 3 tools (search, write, status) |

### tests/

| File | Responsibility |
|------|---------------|
| `tests/fixtures/*.md` | Test entry files covering each kind |
| `tests/common/mod.rs` | Test harness: temp dir, ingest fixtures, cleanup |
| `tests/integration.rs` | Full pipeline integration tests |
| `tests/mcp_e2e.rs` | MCP stdio server E2E tests |

---

## Task 1: Core Types

**Files:**
- Create: `crates/corvia-core/src/types.rs`
- Modify: `crates/corvia-core/src/lib.rs`

- [ ] **Step 1: Write types with tests**

```rust
// crates/corvia-core/src/types.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Decision,
    Learning,
    Instruction,
    Reference,
}

impl Default for Kind {
    fn default() -> Self {
        Kind::Learning
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::Decision => write!(f, "decision"),
            Kind::Learning => write!(f, "learning"),
            Kind::Instruction => write!(f, "instruction"),
            Kind::Reference => write!(f, "reference"),
        }
    }
}

impl std::str::FromStr for Kind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "decision" => Ok(Kind::Decision),
            "learning" => Ok(Kind::Learning),
            "instruction" => Ok(Kind::Instruction),
            "reference" => Ok(Kind::Reference),
            _ => anyhow::bail!("unknown kind: {s}. Expected: decision, learning, instruction, reference"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMeta {
    pub id: String,
    pub created_at: String,
    #[serde(default)]
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub meta: EntryMeta,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub source_entry_id: String,
    pub text: String,
    pub chunk_index: u32,
    pub kind: Kind,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub kind: Kind,
    pub score: f32,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignal {
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub quality: QualitySignal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteResponse {
    pub id: String,
    pub action: String,
    pub superseded: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    pub bm25_docs: u64,
    pub vector_count: u64,
    pub last_ingest: Option<String>,
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub entry_count: u64,
    pub superseded_count: u64,
    pub index_health: IndexHealth,
    pub storage_path: String,
}

pub fn new_entry_id() -> String {
    Uuid::now_v7().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_default_is_learning() {
        assert_eq!(Kind::default(), Kind::Learning);
    }

    #[test]
    fn kind_round_trip() {
        for kind in [Kind::Decision, Kind::Learning, Kind::Instruction, Kind::Reference] {
            let s = kind.to_string();
            let parsed: Kind = s.parse().unwrap();
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn kind_invalid_parse() {
        let result: Result<Kind, _> = "unknown".parse();
        assert!(result.is_err());
    }

    #[test]
    fn new_entry_id_is_lowercase_uuid() {
        let id = new_entry_id();
        assert_eq!(id.len(), 36); // UUID format with hyphens
        assert_eq!(id, id.to_lowercase());
    }

    #[test]
    fn new_entry_id_is_unique() {
        let a = new_entry_id();
        let b = new_entry_id();
        assert_ne!(a, b);
    }

    #[test]
    fn kind_serde_json_roundtrip() {
        let meta = EntryMeta {
            id: "test".into(),
            created_at: "2026-04-15T00:00:00Z".into(),
            kind: Kind::Decision,
            supersedes: vec!["old-id".into()],
            tags: vec!["arch".into()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: EntryMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, Kind::Decision);
        assert_eq!(parsed.supersedes, vec!["old-id"]);
    }
}
```

- [ ] **Step 2: Update lib.rs**

```rust
// crates/corvia-core/src/lib.rs
pub mod types;
pub mod config;
pub mod entry;
pub mod chunk;
pub mod embed;
pub mod index;
pub mod tantivy_index;
pub mod ingest;
pub mod search;
pub mod write;
```

- [ ] **Step 3: Create stub files for all modules**

Create empty files so lib.rs compiles:
```rust
// Each of these files starts with just a comment:
// crates/corvia-core/src/config.rs
// crates/corvia-core/src/entry.rs
// (etc.)
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p corvia-core -- types`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/corvia-core/src/types.rs crates/corvia-core/src/lib.rs
git commit -m "feat: core types (Entry, Kind, Chunk, SearchResult, WriteResponse)"
```

---

## Task 2: Config Parsing

**Files:**
- Modify: `crates/corvia-core/src/config.rs`

- [ ] **Step 1: Write failing test**

```rust
// crates/corvia-core/src/config.rs
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default)]
    pub chunking: ChunkingConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_overlap_tokens")]
    pub overlap_tokens: usize,
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,
    #[serde(default = "default_dedup_threshold")]
    pub dedup_threshold: f32,
    #[serde(default = "default_reranker_candidates")]
    pub reranker_candidates: usize,
    #[serde(default = "default_brute_force_threshold")]
    pub brute_force_threshold: usize,
    #[serde(default = "default_limit")]
    pub default_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_embedding_model")]
    pub model: String,
    #[serde(default = "default_reranker_model")]
    pub reranker_model: String,
    pub model_path: Option<PathBuf>,
}

fn default_data_dir() -> PathBuf { PathBuf::from(".corvia") }
fn default_max_tokens() -> usize { 512 }
fn default_overlap_tokens() -> usize { 64 }
fn default_min_tokens() -> usize { 32 }
fn default_rrf_k() -> u32 { 30 }
fn default_dedup_threshold() -> f32 { 0.85 }
fn default_reranker_candidates() -> usize { 50 }
fn default_brute_force_threshold() -> usize { 10_000 }
fn default_limit() -> usize { 5 }
fn default_embedding_model() -> String { "nomic-embed-text-v1.5".into() }
fn default_reranker_model() -> String { "ms-marco-MiniLM-L6-v2".into() }

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            overlap_tokens: default_overlap_tokens(),
            min_tokens: default_min_tokens(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: default_rrf_k(),
            dedup_threshold: default_dedup_threshold(),
            reranker_candidates: default_reranker_candidates(),
            brute_force_threshold: default_brute_force_threshold(),
            default_limit: default_limit(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            reranker_model: default_reranker_model(),
            model_path: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            chunking: ChunkingConfig::default(),
            search: SearchConfig::default(),
            embedding: EmbeddingConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    pub fn entries_dir(&self) -> PathBuf {
        self.data_dir.join("entries")
    }

    pub fn index_dir(&self) -> PathBuf {
        self.data_dir.join("index")
    }

    pub fn redb_path(&self) -> PathBuf {
        self.index_dir().join("store.redb")
    }

    pub fn tantivy_dir(&self) -> PathBuf {
        self.index_dir().join("tantivy")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_correct_values() {
        let config = Config::default();
        assert_eq!(config.data_dir, PathBuf::from(".corvia"));
        assert_eq!(config.chunking.max_tokens, 512);
        assert_eq!(config.chunking.overlap_tokens, 64);
        assert_eq!(config.chunking.min_tokens, 32);
        assert_eq!(config.search.rrf_k, 30);
        assert_eq!(config.search.dedup_threshold, 0.85);
        assert_eq!(config.search.default_limit, 5);
        assert_eq!(config.search.brute_force_threshold, 10_000);
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let config = Config::load(Path::new("nonexistent.toml")).unwrap();
        assert_eq!(config.chunking.max_tokens, 512);
    }

    #[test]
    fn load_partial_toml_fills_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corvia.toml");
        std::fs::write(&path, "[search]\nrrf_k = 60\n").unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.search.rrf_k, 60);
        assert_eq!(config.chunking.max_tokens, 512); // default preserved
    }

    #[test]
    fn paths_are_derived_correctly() {
        let config = Config::default();
        assert_eq!(config.entries_dir(), PathBuf::from(".corvia/entries"));
        assert_eq!(config.index_dir(), PathBuf::from(".corvia/index"));
        assert_eq!(config.redb_path(), PathBuf::from(".corvia/index/store.redb"));
        assert_eq!(config.tantivy_dir(), PathBuf::from(".corvia/index/tantivy"));
    }
}
```

- [ ] **Step 2: Add tempfile dev dependency**

Add to `crates/corvia-core/Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p corvia-core -- config`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/corvia-core/src/config.rs crates/corvia-core/Cargo.toml
git commit -m "feat: config parsing with defaults (corvia.toml)"
```

---

## Task 3: Entry File I/O

**Files:**
- Modify: `crates/corvia-core/src/entry.rs`

- [ ] **Step 1: Write entry serialization/deserialization with tests**

```rust
// crates/corvia-core/src/entry.rs
use crate::types::{Entry, EntryMeta, Kind, new_entry_id};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

const FRONTMATTER_DELIM: &str = "+++";

/// Serialize an Entry to TOML frontmatter + markdown body.
pub fn serialize_entry(entry: &Entry) -> Result<String> {
    let toml_str = toml::to_string_pretty(&entry.meta)
        .context("failed to serialize entry metadata")?;
    Ok(format!("{FRONTMATTER_DELIM}\n{toml_str}{FRONTMATTER_DELIM}\n\n{}", entry.body))
}

/// Parse a file's content into an Entry.
pub fn parse_entry(content: &str) -> Result<Entry> {
    let content = content.trim();
    if !content.starts_with(FRONTMATTER_DELIM) {
        bail!("missing opening +++ delimiter");
    }

    let after_open = &content[FRONTMATTER_DELIM.len()..];
    let close_pos = after_open
        .find(FRONTMATTER_DELIM)
        .context("missing closing +++ delimiter")?;

    let toml_str = &after_open[..close_pos];
    let body_start = FRONTMATTER_DELIM.len() + close_pos + FRONTMATTER_DELIM.len();
    let body = content[body_start..].trim().to_string();

    let meta: EntryMeta = toml::from_str(toml_str)
        .context("invalid TOML in frontmatter")?;

    if meta.id.is_empty() {
        bail!("missing required field: id");
    }

    Ok(Entry { meta, body })
}

/// Read an entry from a file path.
pub fn read_entry(path: &Path) -> Result<Entry> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read entry file: {}", path.display()))?;
    parse_entry(&content)
        .with_context(|| format!("failed to parse entry: {}", path.display()))
}

/// Write an entry to disk using atomic rename.
/// Writes to a temp file first, then renames to final path.
pub fn write_entry_atomic(entries_dir: &Path, entry: &Entry) -> Result<PathBuf> {
    fs::create_dir_all(entries_dir)
        .context("failed to create entries directory")?;

    let content = serialize_entry(entry)?;
    let final_path = entries_dir.join(format!("{}.md", entry.meta.id));
    let tmp_path = entries_dir.join(format!(".{}.md.tmp", entry.meta.id));

    fs::write(&tmp_path, &content)
        .with_context(|| format!("failed to write temp file: {}", tmp_path.display()))?;

    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("failed to rename {} -> {}", tmp_path.display(), final_path.display()))?;

    Ok(final_path)
}

/// Create a new Entry with generated ID and current timestamp.
pub fn new_entry(body: String, kind: Kind, tags: Vec<String>, supersedes: Vec<String>) -> Entry {
    let now = chrono_now_iso8601();
    Entry {
        meta: EntryMeta {
            id: new_entry_id(),
            created_at: now,
            kind,
            supersedes,
            tags,
        },
        body,
    }
}

fn chrono_now_iso8601() -> String {
    // Simple ISO 8601 without chrono dep. Uses std::time.
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = d.as_secs();
    // Format as YYYY-MM-DDTHH:MM:SSZ (good enough for v1.0)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to date (simplified, accurate through ~2100)
    let mut y = 1970u32;
    let mut remaining = days;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining < days_in_year { break; }
        remaining -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days: [u64; 12] = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0u32;
    for &md in &month_days {
        if remaining < md { break; }
        remaining -= md;
        m += 1;
    }
    format!("{y:04}-{:02}-{:02}T{hours:02}:{minutes:02}:{seconds:02}Z", m + 1, remaining + 1)
}

/// Scan entries directory and return all .md file paths.
pub fn scan_entries(entries_dir: &Path) -> Result<Vec<PathBuf>> {
    if !entries_dir.exists() {
        return Ok(vec![]);
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(entries_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "md")
            && !entry.file_name().to_string_lossy().starts_with('.')
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_roundtrip() {
        let entry = Entry {
            meta: EntryMeta {
                id: "test-id-123".into(),
                created_at: "2026-04-15T10:00:00Z".into(),
                kind: Kind::Decision,
                supersedes: vec!["old-id".into()],
                tags: vec!["storage".into(), "architecture".into()],
            },
            body: "This is the knowledge content.".into(),
        };

        let serialized = serialize_entry(&entry).unwrap();
        assert!(serialized.starts_with("+++\n"));
        assert!(serialized.contains("id = \"test-id-123\""));
        assert!(serialized.contains("This is the knowledge content."));

        let parsed = parse_entry(&serialized).unwrap();
        assert_eq!(parsed.meta.id, "test-id-123");
        assert_eq!(parsed.meta.kind, Kind::Decision);
        assert_eq!(parsed.meta.supersedes, vec!["old-id"]);
        assert_eq!(parsed.meta.tags, vec!["storage", "architecture"]);
        assert_eq!(parsed.body, "This is the knowledge content.");
    }

    #[test]
    fn parse_missing_delimiter_errors() {
        let result = parse_entry("no delimiters here");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing opening"));
    }

    #[test]
    fn parse_missing_id_errors() {
        let content = "+++\nkind = \"learning\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n+++\nbody";
        let result = parse_entry(content);
        assert!(result.is_err());
    }

    #[test]
    fn parse_empty_body() {
        let content = "+++\nid = \"x\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n+++\n";
        let entry = parse_entry(content).unwrap();
        assert_eq!(entry.body, "");
    }

    #[test]
    fn atomic_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");
        let entry = new_entry("test body".into(), Kind::Learning, vec![], vec![]);
        let path = write_entry_atomic(&entries_dir, &entry).unwrap();

        assert!(path.exists());
        let readback = read_entry(&path).unwrap();
        assert_eq!(readback.meta.id, entry.meta.id);
        assert_eq!(readback.body, "test body");
    }

    #[test]
    fn atomic_write_no_tmp_left() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");
        let entry = new_entry("test".into(), Kind::Learning, vec![], vec![]);
        write_entry_atomic(&entries_dir, &entry).unwrap();

        // No .tmp files should remain
        for f in fs::read_dir(&entries_dir).unwrap() {
            let name = f.unwrap().file_name().to_string_lossy().to_string();
            assert!(!name.ends_with(".tmp"), "temp file left behind: {name}");
        }
    }

    #[test]
    fn scan_entries_skips_hidden_files() {
        let dir = tempfile::tempdir().unwrap();
        let entries_dir = dir.path().join("entries");
        fs::create_dir_all(&entries_dir).unwrap();
        fs::write(entries_dir.join("visible.md"), "+++\nid=\"a\"\n+++\n").unwrap();
        fs::write(entries_dir.join(".hidden.md.tmp"), "temp").unwrap();

        let paths = scan_entries(&entries_dir).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("visible.md"));
    }

    #[test]
    fn tags_with_special_chars_roundtrip() {
        let entry = Entry {
            meta: EntryMeta {
                id: "special".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                kind: Kind::Learning,
                supersedes: vec![],
                tags: vec!["key=value".into(), "has \"quotes\"".into(), "has]bracket".into()],
            },
            body: "body".into(),
        };
        let serialized = serialize_entry(&entry).unwrap();
        let parsed = parse_entry(&serialized).unwrap();
        assert_eq!(parsed.meta.tags, entry.meta.tags);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia-core -- entry`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/entry.rs
git commit -m "feat: entry file I/O with TOML frontmatter, atomic write"
```

---

## Task 4: Chunking

**Files:**
- Modify: `crates/corvia-core/src/chunk.rs`

- [ ] **Step 1: Write chunker with tests**

```rust
// crates/corvia-core/src/chunk.rs
use crate::types::{Chunk, Entry, Kind};

/// Strip TOML frontmatter from entry content.
/// Returns just the markdown body.
pub fn strip_frontmatter(raw: &str) -> &str {
    let trimmed = raw.trim();
    if !trimmed.starts_with("+++") {
        return trimmed;
    }
    let after_open = &trimmed[3..];
    match after_open.find("+++") {
        Some(pos) => after_open[pos + 3..].trim(),
        None => trimmed,
    }
}

/// Split text into chunks with overlap. Simple token-based splitting.
/// Tokens are approximated as whitespace-separated words.
pub fn split_into_chunks(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    min_tokens: usize,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }
    if words.len() <= max_tokens {
        return vec![words.join(" ")];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + max_tokens).min(words.len());
        let chunk_text = words[start..end].join(" ");

        if !chunk_text.is_empty() {
            chunks.push(chunk_text);
        }

        if end >= words.len() {
            break;
        }

        // Advance by (max_tokens - overlap_tokens)
        let step = max_tokens.saturating_sub(overlap_tokens).max(1);
        start += step;
    }

    // Merge last chunk if too small
    if chunks.len() >= 2 {
        let last = &chunks[chunks.len() - 1];
        let last_tokens = last.split_whitespace().count();
        if last_tokens < min_tokens {
            let merged = format!("{} {}", chunks[chunks.len() - 2], chunks.pop().unwrap());
            *chunks.last_mut().unwrap() = merged;
        }
    }

    chunks
}

/// Chunk an entry: strip frontmatter, split body, produce Chunk structs.
pub fn chunk_entry(entry: &Entry, max_tokens: usize, overlap_tokens: usize, min_tokens: usize) -> Vec<Chunk> {
    let body = entry.body.trim();
    if body.is_empty() {
        return vec![Chunk {
            source_entry_id: entry.meta.id.clone(),
            text: String::new(),
            chunk_index: 0,
            kind: entry.meta.kind,
            tags: entry.meta.tags.clone(),
        }];
    }

    let texts = split_into_chunks(body, max_tokens, overlap_tokens, min_tokens);
    texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            source_entry_id: entry.meta.id.clone(),
            text,
            chunk_index: i as u32,
            kind: entry.meta.kind,
            tags: entry.meta.tags.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_frontmatter_removes_toml() {
        let raw = "+++\nid = \"test\"\n+++\n\nBody content here.";
        assert_eq!(strip_frontmatter(raw), "Body content here.");
    }

    #[test]
    fn strip_frontmatter_no_frontmatter() {
        assert_eq!(strip_frontmatter("just text"), "just text");
    }

    #[test]
    fn short_text_single_chunk() {
        let chunks = split_into_chunks("hello world", 512, 64, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn empty_text_no_chunks() {
        let chunks = split_into_chunks("", 512, 64, 32);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn long_text_splits_with_overlap() {
        // Create text with 100 words
        let words: Vec<String> = (0..100).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");

        let chunks = split_into_chunks(&text, 30, 5, 10);
        assert!(chunks.len() >= 3);

        // Each chunk should have at most 30 words (except possibly merged last)
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.split_whitespace().count() <= 30);
        }
    }

    #[test]
    fn small_last_chunk_gets_merged() {
        // 35 words, max 30, overlap 5 -> would produce [30] + [10] but 10 < 32 min
        // so second chunk merges into first
        let words: Vec<String> = (0..35).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");

        let chunks = split_into_chunks(&text, 30, 5, 32);
        // 35 words, step = 25, first chunk 0..30, second chunk 25..35 = 10 words < 32
        // merged into one chunk
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_entry_preserves_metadata() {
        use crate::types::EntryMeta;
        let entry = Entry {
            meta: EntryMeta {
                id: "abc".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                kind: Kind::Decision,
                supersedes: vec![],
                tags: vec!["arch".into()],
            },
            body: "short body".into(),
        };

        let chunks = chunk_entry(&entry, 512, 64, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source_entry_id, "abc");
        assert_eq!(chunks[0].kind, Kind::Decision);
        assert_eq!(chunks[0].tags, vec!["arch"]);
    }

    #[test]
    fn chunk_entry_empty_body_produces_one_chunk() {
        use crate::types::EntryMeta;
        let entry = Entry {
            meta: EntryMeta {
                id: "empty".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                kind: Kind::Learning,
                supersedes: vec![],
                tags: vec![],
            },
            body: "".into(),
        };

        let chunks = chunk_entry(&entry, 512, 64, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia-core -- chunk`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/chunk.rs
git commit -m "feat: chunking with frontmatter stripping, overlap, merge"
```

---

## Task 5: Embedding Wrapper

**Files:**
- Modify: `crates/corvia-core/src/embed.rs`

- [ ] **Step 1: Write fastembed wrapper**

```rust
// crates/corvia-core/src/embed.rs
use anyhow::{Context, Result};
use fastembed::{TextEmbedding, EmbeddingModel, InitOptions, TextRerank, RerankInitOptions, RerankerModel, RerankResult};
use std::path::Path;
use std::sync::Arc;

pub struct Embedder {
    model: TextEmbedding,
    reranker: TextRerank,
}

impl Embedder {
    /// Initialize with default models. Downloads on first run.
    pub fn new(cache_dir: Option<&Path>) -> Result<Self> {
        let mut embed_opts = InitOptions::new(EmbeddingModel::NomicEmbedTextV15)
            .with_show_download_progress(true);
        if let Some(dir) = cache_dir {
            embed_opts = embed_opts.with_cache_dir(dir.to_path_buf());
        }

        let model = TextEmbedding::try_new(embed_opts)
            .context("failed to initialize embedding model (nomic-embed-text-v1.5). If offline, use --model-path.")?;

        let mut rerank_opts = RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_show_download_progress(true);
        if let Some(dir) = cache_dir {
            rerank_opts = rerank_opts.with_cache_dir(dir.to_path_buf());
        }

        let reranker = TextRerank::try_new(rerank_opts)
            .context("failed to initialize reranker model. If offline, use --model-path.")?;

        Ok(Self { model, reranker })
    }

    /// Embed a single text. Returns a 768-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.model.embed(vec![text], None)
            .context("embedding failed")?;
        results
            .into_iter()
            .next()
            .context("no embedding returned")
    }

    /// Embed multiple texts in batch.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.model.embed(texts_owned, None)
            .context("batch embedding failed")
    }

    /// Rerank documents against a query. Returns indices sorted by relevance.
    pub fn rerank(&self, query: &str, documents: &[&str], top_n: usize) -> Result<Vec<RerankResult>> {
        let docs: Vec<String> = documents.iter().map(|s| s.to_string()).collect();
        self.reranker.rerank(query, docs, true, Some(top_n))
            .context("reranking failed")
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = Embedder::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = Embedder::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(Embedder::cosine_similarity(&a, &b), 0.0);
    }

    // NOTE: Model download tests are #[ignore]d because they require network.
    // Run with: cargo test -p corvia-core -- embed --ignored
    #[test]
    #[ignore]
    fn embed_produces_768d_vector() {
        let embedder = Embedder::new(None).unwrap();
        let vec = embedder.embed("test text").unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[test]
    #[ignore]
    fn embed_batch_produces_correct_count() {
        let embedder = Embedder::new(None).unwrap();
        let vecs = embedder.embed_batch(&["hello", "world", "test"]).unwrap();
        assert_eq!(vecs.len(), 3);
        for v in &vecs {
            assert_eq!(v.len(), 768);
        }
    }

    #[test]
    #[ignore]
    fn similar_texts_have_high_similarity() {
        let embedder = Embedder::new(None).unwrap();
        let a = embedder.embed("Rust programming language").unwrap();
        let b = embedder.embed("Rust systems programming").unwrap();
        let c = embedder.embed("chocolate cake recipe").unwrap();

        let sim_ab = Embedder::cosine_similarity(&a, &b);
        let sim_ac = Embedder::cosine_similarity(&a, &c);
        assert!(sim_ab > sim_ac, "related texts should be more similar");
    }
}
```

- [ ] **Step 2: Run unit tests (no network)**

Run: `cargo test -p corvia-core -- embed --skip ignored`
Expected: Cosine similarity tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/embed.rs
git commit -m "feat: embedding wrapper (fastembed, nomic-embed-text-v1.5, BGE reranker)"
```

---

## Task 6: Redb Index

**Files:**
- Modify: `crates/corvia-core/src/index.rs`

- [ ] **Step 1: Write Redb index with tables and operations**

```rust
// crates/corvia-core/src/index.rs
use anyhow::{Context, Result};
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::collections::HashSet;

// Table: chunk_id (string) -> vector (bytes, f32 array)
const VECTORS: TableDefinition<&str, &[u8]> = TableDefinition::new("vectors");
// Table: chunk_id (string) -> source_entry_id (string)
const CHUNK_MAP: TableDefinition<&str, &str> = TableDefinition::new("chunk_map");
// Table: entry_id (string) -> superseded (u8, 0 or 1)
const SUPERSESSION: TableDefinition<&str, u8> = TableDefinition::new("supersession");
// Table: "meta" key -> value (string)
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

pub struct RedbIndex {
    db: Database,
}

impl RedbIndex {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Database::create(path)
            .with_context(|| format!("failed to open redb at {}", path.display()))?;

        // Create tables on first open
        let write_txn = db.begin_write()?;
        { let _ = write_txn.open_table(VECTORS)?; }
        { let _ = write_txn.open_table(CHUNK_MAP)?; }
        { let _ = write_txn.open_table(SUPERSESSION)?; }
        { let _ = write_txn.open_table(META)?; }
        write_txn.commit()?;

        Ok(Self { db })
    }

    /// Store a vector for a chunk.
    pub fn put_vector(&self, chunk_id: &str, entry_id: &str, vector: &[f32]) -> Result<()> {
        let bytes = bytemuck_f32_to_bytes(vector);
        let write_txn = self.db.begin_write()?;
        {
            let mut vectors = write_txn.open_table(VECTORS)?;
            vectors.insert(chunk_id, bytes.as_slice())?;
            let mut chunk_map = write_txn.open_table(CHUNK_MAP)?;
            chunk_map.insert(chunk_id, entry_id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Mark an entry as superseded (or current).
    pub fn set_superseded(&self, entry_id: &str, superseded: bool) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SUPERSESSION)?;
            table.insert(entry_id, if superseded { 1u8 } else { 0u8 })?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Check if an entry is superseded.
    pub fn is_superseded(&self, entry_id: &str) -> Result<bool> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SUPERSESSION)?;
        match table.get(entry_id)? {
            Some(val) => Ok(val.value() == 1),
            None => Ok(false),
        }
    }

    /// Get all superseded entry IDs.
    pub fn superseded_ids(&self) -> Result<HashSet<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SUPERSESSION)?;
        let mut ids = HashSet::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, val) = item?;
            if val.value() == 1 {
                ids.insert(key.value().to_string());
            }
        }
        Ok(ids)
    }

    /// Get all vectors with their chunk IDs (for brute-force search).
    pub fn all_vectors(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(VECTORS)?;
        let mut results = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, val) = item?;
            let chunk_id = key.value().to_string();
            let vector = bytes_to_f32(val.value());
            results.push((chunk_id, vector));
        }
        Ok(results)
    }

    /// Get the source entry ID for a chunk.
    pub fn chunk_entry_id(&self, chunk_id: &str) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHUNK_MAP)?;
        Ok(table.get(chunk_id)?.map(|v| v.value().to_string()))
    }

    /// Get total vector count.
    pub fn vector_count(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(VECTORS)?;
        Ok(table.len()?)
    }

    /// Get total entry count (in supersession table).
    pub fn entry_count(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SUPERSESSION)?;
        Ok(table.len()?)
    }

    /// Set metadata value.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(META)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get metadata value.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(META)?;
        Ok(table.get(key)?.map(|v| v.value().to_string()))
    }

    /// Clear all tables (for --fresh ingest).
    pub fn clear_all(&self) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut t = write_txn.open_table(VECTORS)?;
            while t.pop_last()?.is_some() {}
        }
        {
            let mut t = write_txn.open_table(CHUNK_MAP)?;
            while t.pop_last()?.is_some() {}
        }
        {
            let mut t = write_txn.open_table(SUPERSESSION)?;
            while t.pop_last()?.is_some() {}
        }
        {
            let mut t = write_txn.open_table(META)?;
            while t.pop_last()?.is_some() {}
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Check if entry exists in the index.
    pub fn entry_exists(&self, entry_id: &str) -> Result<bool> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SUPERSESSION)?;
        Ok(table.get(entry_id)?.is_some())
    }
}

fn bytemuck_f32_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> (tempfile::TempDir, RedbIndex) {
        let dir = tempfile::tempdir().unwrap();
        let db = RedbIndex::open(&dir.path().join("test.redb")).unwrap();
        (dir, db)
    }

    #[test]
    fn put_and_get_vector() {
        let (_dir, db) = test_db();
        let vec = vec![1.0f32, 2.0, 3.0];
        db.put_vector("chunk-1", "entry-1", &vec).unwrap();

        let all = db.all_vectors().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "chunk-1");
        assert_eq!(all[0].1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn chunk_entry_mapping() {
        let (_dir, db) = test_db();
        db.put_vector("chunk-1", "entry-1", &[1.0]).unwrap();

        let entry_id = db.chunk_entry_id("chunk-1").unwrap();
        assert_eq!(entry_id, Some("entry-1".to_string()));
    }

    #[test]
    fn supersession_tracking() {
        let (_dir, db) = test_db();
        db.set_superseded("entry-a", false).unwrap();
        db.set_superseded("entry-b", true).unwrap();

        assert!(!db.is_superseded("entry-a").unwrap());
        assert!(db.is_superseded("entry-b").unwrap());

        let superseded = db.superseded_ids().unwrap();
        assert!(superseded.contains("entry-b"));
        assert!(!superseded.contains("entry-a"));
    }

    #[test]
    fn clear_all_empties_tables() {
        let (_dir, db) = test_db();
        db.put_vector("c1", "e1", &[1.0]).unwrap();
        db.set_superseded("e1", false).unwrap();

        db.clear_all().unwrap();
        assert_eq!(db.vector_count().unwrap(), 0);
        assert_eq!(db.entry_count().unwrap(), 0);
    }

    #[test]
    fn metadata_set_and_get() {
        let (_dir, db) = test_db();
        db.set_meta("last_ingest", "2026-04-15T10:00:00Z").unwrap();
        let val = db.get_meta("last_ingest").unwrap();
        assert_eq!(val, Some("2026-04-15T10:00:00Z".to_string()));
    }

    #[test]
    fn vector_roundtrip_precision() {
        let (_dir, db) = test_db();
        let vec: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        db.put_vector("c1", "e1", &vec).unwrap();

        let all = db.all_vectors().unwrap();
        for (a, b) in vec.iter().zip(all[0].1.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia-core -- index`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/index.rs
git commit -m "feat: redb index (vectors, chunk mapping, supersession, metadata)"
```

---

## Task 7: Tantivy BM25 Index

**Files:**
- Modify: `crates/corvia-core/src/tantivy_index.rs`

- [ ] **Step 1: Write tantivy index with kind filtering**

```rust
// crates/corvia-core/src/tantivy_index.rs
use crate::types::Kind;
use anyhow::{Context, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{doc, Directory, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

pub struct TantivyIndex {
    index: Index,
    reader: IndexReader,
    schema: Schema,
    chunk_id_field: Field,
    entry_id_field: Field,
    body_field: Field,
    kind_field: Field,
    superseded_field: Field,
}

impl TantivyIndex {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let mut schema_builder = Schema::builder();
        let chunk_id_field = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let entry_id_field = schema_builder.add_text_field("entry_id", STRING | STORED);
        let body_field = schema_builder.add_text_field("body", TEXT | STORED);
        let kind_field = schema_builder.add_text_field("kind", STRING);
        let superseded_field = schema_builder.add_text_field("superseded", STRING);
        let schema = schema_builder.build();

        let index = Index::open_or_create(
            tantivy::directory::MmapDirectory::open(path)?,
            schema.clone(),
        )?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            schema,
            chunk_id_field,
            entry_id_field,
            body_field,
            kind_field,
            superseded_field,
        })
    }

    pub fn writer(&self) -> Result<IndexWriter> {
        self.index
            .writer(50_000_000) // 50MB heap
            .context("failed to create tantivy writer")
    }

    /// Add a chunk document.
    pub fn add_doc(
        &self,
        writer: &IndexWriter,
        chunk_id: &str,
        entry_id: &str,
        body: &str,
        kind: Kind,
        superseded: bool,
    ) -> Result<()> {
        writer.add_document(doc!(
            self.chunk_id_field => chunk_id,
            self.entry_id_field => entry_id,
            self.body_field => body,
            self.kind_field => kind.to_string(),
            self.superseded_field => if superseded { "true" } else { "false" },
        ))?;
        Ok(())
    }

    /// Search with optional kind filter, excluding superseded.
    pub fn search(
        &self,
        query_text: &str,
        kind_filter: Option<Kind>,
        limit: usize,
    ) -> Result<Vec<(String, String, f32)>> {
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        let text_query = query_parser.parse_query(query_text)
            .context("failed to parse search query")?;

        // Always exclude superseded
        let not_superseded = TermQuery::new(
            Term::from_field_text(self.superseded_field, "false"),
            IndexRecordOption::Basic,
        );

        let mut must_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
            (Occur::Must, Box::new(text_query)),
            (Occur::Must, Box::new(not_superseded)),
        ];

        if let Some(kind) = kind_filter {
            let kind_query = TermQuery::new(
                Term::from_field_text(self.kind_field, &kind.to_string()),
                IndexRecordOption::Basic,
            );
            must_clauses.push((Occur::Must, Box::new(kind_query)));
        }

        let combined = BooleanQuery::new(must_clauses);
        let top_docs = searcher.search(&combined, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_addr)?;
            let chunk_id = doc
                .get_first(self.chunk_id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let entry_id = doc
                .get_first(self.entry_id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            results.push((chunk_id, entry_id, score));
        }

        Ok(results)
    }

    /// Get total document count.
    pub fn doc_count(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }

    /// Delete all documents (for --fresh).
    pub fn clear(&self) -> Result<()> {
        let mut writer = self.writer()?;
        writer.delete_all_documents()?;
        writer.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_index() -> (tempfile::TempDir, TantivyIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = TantivyIndex::open(&dir.path().join("tantivy")).unwrap();
        (dir, idx)
    }

    #[test]
    fn add_and_search() {
        let (_dir, idx) = test_index();
        let writer = idx.writer().unwrap();
        idx.add_doc(&writer, "c1", "e1", "rust programming language", Kind::Reference, false).unwrap();
        idx.add_doc(&writer, "c2", "e2", "chocolate cake recipe", Kind::Instruction, false).unwrap();
        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        let results = idx.search("rust programming", None, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "c1");
    }

    #[test]
    fn superseded_excluded_from_search() {
        let (_dir, idx) = test_index();
        let writer = idx.writer().unwrap();
        idx.add_doc(&writer, "c1", "e1", "rust programming", Kind::Reference, true).unwrap();
        idx.add_doc(&writer, "c2", "e2", "rust systems", Kind::Reference, false).unwrap();
        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        let results = idx.search("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "c2");
    }

    #[test]
    fn kind_filter() {
        let (_dir, idx) = test_index();
        let writer = idx.writer().unwrap();
        idx.add_doc(&writer, "c1", "e1", "chose redb for storage", Kind::Decision, false).unwrap();
        idx.add_doc(&writer, "c2", "e2", "how to install redb", Kind::Instruction, false).unwrap();
        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        let results = idx.search("redb", Some(Kind::Decision), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "c1");
    }

    #[test]
    fn empty_index_returns_empty() {
        let (_dir, idx) = test_index();
        let results = idx.search("anything", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn clear_removes_all() {
        let (_dir, idx) = test_index();
        let writer = idx.writer().unwrap();
        idx.add_doc(&writer, "c1", "e1", "content", Kind::Learning, false).unwrap();
        writer.commit().unwrap();
        idx.reader.reload().unwrap();

        idx.clear().unwrap();
        idx.reader.reload().unwrap();
        assert_eq!(idx.doc_count(), 0);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia-core -- tantivy_index`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/tantivy_index.rs
git commit -m "feat: tantivy BM25 index with kind filtering, superseded exclusion"
```

---

## Task 8: Ingest Pipeline

**Files:**
- Modify: `crates/corvia-core/src/ingest.rs`

- [ ] **Step 1: Write ingest pipeline**

```rust
// crates/corvia-core/src/ingest.rs
use crate::chunk::chunk_entry;
use crate::config::Config;
use crate::embed::Embedder;
use crate::entry::{read_entry, scan_entries};
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::Entry;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use tracing::{info, warn};

pub struct IngestResult {
    pub entries_ingested: usize,
    pub chunks_indexed: usize,
    pub entries_skipped: Vec<(String, String)>, // (filename, reason)
    pub superseded_count: usize,
}

pub fn ingest(config: &Config, base_dir: &Path, fresh: bool) -> Result<IngestResult> {
    let entries_dir = base_dir.join(config.entries_dir());
    let index_dir = base_dir.join(config.index_dir());

    // Create directories
    std::fs::create_dir_all(&entries_dir)?;
    std::fs::create_dir_all(&index_dir)?;

    // Open indexes
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))?;

    if fresh {
        redb.clear_all()?;
        tantivy.clear()?;
    }

    // Initialize embedder
    let embedder = Embedder::new(config.embedding.model_path.as_deref())?;

    // Scan and parse entries
    let paths = scan_entries(&entries_dir)?;
    let mut entries: Vec<Entry> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for path in &paths {
        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        match read_entry(path) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                warn!("skipping {filename}: {e}");
                skipped.push((filename, e.to_string()));
            }
        }
    }

    // Build supersession set
    let mut superseded_ids: HashSet<String> = HashSet::new();
    for entry in &entries {
        for sup_id in &entry.meta.supersedes {
            superseded_ids.insert(sup_id.clone());
        }
    }

    // Resolve circular supersession by created_at (last writer wins)
    // If A supersedes B and B supersedes A, the one with later created_at wins
    let entry_map: std::collections::HashMap<&str, &Entry> =
        entries.iter().map(|e| (e.meta.id.as_str(), e)).collect();
    let mut to_unsupersede = Vec::new();
    for id in &superseded_ids {
        if let Some(entry) = entry_map.get(id.as_str()) {
            // Check if this entry supersedes something that also supersedes it
            for sup_id in &entry.meta.supersedes {
                if superseded_ids.contains(&entry.meta.id) && superseded_ids.contains(sup_id) {
                    // Both are superseded by each other -- last created_at wins
                    if let Some(other) = entry_map.get(sup_id.as_str()) {
                        if entry.meta.created_at > other.meta.created_at {
                            to_unsupersede.push(entry.meta.id.clone());
                        }
                    }
                }
            }
        }
    }
    for id in to_unsupersede {
        superseded_ids.remove(&id);
    }

    // Chunk, embed, and index
    let tantivy_writer = tantivy.writer()?;
    let mut chunks_indexed = 0;

    for entry in &entries {
        let is_superseded = superseded_ids.contains(&entry.meta.id);
        redb.set_superseded(&entry.meta.id, is_superseded)?;

        let chunks = chunk_entry(
            entry,
            config.chunking.max_tokens,
            config.chunking.overlap_tokens,
            config.chunking.min_tokens,
        );

        for chunk in &chunks {
            let chunk_id = format!("{}:{}", chunk.source_entry_id, chunk.chunk_index);

            // Embed
            if !chunk.text.is_empty() {
                let vector = embedder.embed(&chunk.text)?;
                redb.put_vector(&chunk_id, &chunk.source_entry_id, &vector)?;
            }

            // BM25 index
            tantivy.add_doc(
                &tantivy_writer,
                &chunk_id,
                &chunk.source_entry_id,
                &chunk.text,
                chunk.kind,
                is_superseded,
            )?;

            chunks_indexed += 1;
        }
    }

    tantivy_writer.commit()?;

    // Store metadata
    let now = crate::entry::chrono_now_iso8601_pub();
    redb.set_meta("last_ingest", &now)?;
    redb.set_meta("entry_count", &entries.len().to_string())?;

    let superseded_count = superseded_ids.len();
    info!(
        "ingested {} entries ({} chunks), {} superseded, {} skipped",
        entries.len(),
        chunks_indexed,
        superseded_count,
        skipped.len()
    );

    Ok(IngestResult {
        entries_ingested: entries.len(),
        chunks_indexed,
        entries_skipped: skipped,
        superseded_count,
    })
}
```

Note: This task requires making `chrono_now_iso8601` public in entry.rs. Rename to `chrono_now_iso8601_pub` or add `pub` to the existing function.

- [ ] **Step 2: Make timestamp function public in entry.rs**

In `crates/corvia-core/src/entry.rs`, rename:
```rust
// Change:
fn chrono_now_iso8601() -> String {
// To:
pub fn chrono_now_iso8601_pub() -> String {
```
And update the call site in `new_entry` to use `chrono_now_iso8601_pub()`.

- [ ] **Step 3: Run tests (unit tests only, ingest needs embedder)**

Run: `cargo check -p corvia-core`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add crates/corvia-core/src/ingest.rs crates/corvia-core/src/entry.rs
git commit -m "feat: ingest pipeline (scan -> parse -> chunk -> embed -> index)"
```

---

## Task 9: Search Pipeline

**Files:**
- Modify: `crates/corvia-core/src/search.rs`

- [ ] **Step 1: Write search pipeline with RRF fusion and quality signal**

```rust
// crates/corvia-core/src/search.rs
use crate::config::Config;
use crate::embed::Embedder;
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::{Confidence, Kind, QualitySignal, SearchResponse, SearchResult};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub struct SearchParams {
    pub query: String,
    pub limit: usize,
    pub max_tokens: Option<usize>,
    pub min_score: Option<f32>,
    pub kind: Option<Kind>,
}

pub fn search(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: &SearchParams,
) -> Result<SearchResponse> {
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))?;

    // Check for empty/stale index
    let indexed_count = redb.get_meta("entry_count")
        .ok()
        .flatten()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    if indexed_count == 0 {
        return Ok(SearchResponse {
            results: vec![],
            quality: QualitySignal {
                confidence: Confidence::None,
                suggestion: Some("No entries indexed. Run 'corvia ingest' first.".into()),
            },
        });
    }

    // Drift detection
    let entries_dir = base_dir.join(config.entries_dir());
    let file_count = crate::entry::scan_entries(&entries_dir)
        .map(|v| v.len() as u64)
        .unwrap_or(0);
    let stale = file_count != indexed_count;

    // Oversample for kind filter
    let oversample = if params.kind.is_some() { 3 } else { 1 };
    let retrieval_limit = config.search.reranker_candidates * oversample;

    // BM25 search
    let bm25_results = tantivy.search(
        &params.query,
        params.kind,
        retrieval_limit,
    )?;

    // Vector search (brute-force)
    let query_vec = embedder.embed(&params.query)?;
    let all_vectors = redb.all_vectors()?;
    let superseded = redb.superseded_ids()?;

    let mut vector_results: Vec<(String, String, f32)> = Vec::new();
    for (chunk_id, vec) in &all_vectors {
        let entry_id = redb.chunk_entry_id(chunk_id)?
            .unwrap_or_default();
        if superseded.contains(&entry_id) {
            continue;
        }
        let sim = Embedder::cosine_similarity(&query_vec, vec);
        vector_results.push((chunk_id.clone(), entry_id, sim));
    }
    vector_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    vector_results.truncate(retrieval_limit);

    // RRF fusion
    let k = config.search.rrf_k as f32;
    let mut fused_scores: HashMap<String, (f32, String)> = HashMap::new(); // chunk_id -> (score, entry_id)

    for (rank, (chunk_id, entry_id, _)) in bm25_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);
        let entry = fused_scores.entry(chunk_id.clone()).or_insert((0.0, entry_id.clone()));
        entry.0 += rrf_score;
    }

    for (rank, (chunk_id, entry_id, _)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);
        let entry = fused_scores.entry(chunk_id.clone()).or_insert((0.0, entry_id.clone()));
        entry.0 += rrf_score;
    }

    // Sort by fused score
    let mut candidates: Vec<(String, String, f32)> = fused_scores
        .into_iter()
        .map(|(chunk_id, (score, entry_id))| (chunk_id, entry_id, score))
        .collect();
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(config.search.reranker_candidates);

    if candidates.is_empty() {
        return Ok(SearchResponse {
            results: vec![],
            quality: QualitySignal {
                confidence: Confidence::Low,
                suggestion: Some("No matching results. Try broader terms.".into()),
            },
        });
    }

    // Rerank
    // Collect chunk texts for reranking
    let tantivy_searcher = tantivy.reader.searcher();
    let chunk_texts: Vec<String> = candidates
        .iter()
        .map(|(chunk_id, _, _)| {
            // Look up chunk text from BM25 results or vector results
            // For simplicity, use the stored body field from tantivy
            // This is a placeholder -- actual impl reads from tantivy stored fields
            chunk_id.clone() // Will be replaced by actual text lookup
        })
        .collect();

    // TODO: The reranker step requires looking up chunk text from tantivy stored fields.
    // For Task 9, implement the fusion pipeline and quality signal.
    // Wire in actual reranking when the full pipeline is connected.

    // Apply min_score filter
    if let Some(min) = params.min_score {
        candidates.retain(|c| c.2 >= min);
    }

    // Truncate to limit
    candidates.truncate(params.limit);

    // Build results
    let results: Vec<SearchResult> = candidates
        .into_iter()
        .map(|(chunk_id, entry_id, score)| {
            // Look up kind from entry -- for now use Learning as default
            SearchResult {
                id: entry_id,
                kind: Kind::Learning, // Will be populated from index
                score,
                content: chunk_id, // Will be populated with actual chunk text
            }
        })
        .collect();

    // Quality signal
    let top_score = results.first().map(|r| r.score).unwrap_or(0.0);
    let confidence = if top_score >= 0.05 && results.len() >= 3 {
        Confidence::High
    } else if top_score >= 0.03 {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    let mut suggestion = None;
    if stale {
        suggestion = Some("Index may be stale. Run 'corvia ingest' to update.".into());
    } else if confidence == Confidence::Low {
        suggestion = Some("Low confidence results. Try broader terms or remove the kind filter.".into());
    }

    Ok(SearchResponse {
        results,
        quality: QualitySignal {
            confidence,
            suggestion,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fusion_math() {
        // RRF score = 1/(k + rank + 1), k=30
        let k = 30.0f32;
        let rank0 = 1.0 / (k + 1.0); // rank 0 = 0.0323
        let rank1 = 1.0 / (k + 2.0); // rank 1 = 0.03125
        assert!(rank0 > rank1);
        assert!((rank0 - 1.0 / 31.0).abs() < 1e-6);
    }

    #[test]
    fn quality_signal_thresholds() {
        // High: top_score >= 0.05 and >= 3 results
        // These thresholds are provisional for RRF scores (not cosine)
        let top = 0.06;
        let count = 5;
        let confidence = if top >= 0.05 && count >= 3 {
            Confidence::High
        } else if top >= 0.03 {
            Confidence::Medium
        } else {
            Confidence::Low
        };
        assert_eq!(confidence, Confidence::High);
    }
}
```

Note: This is a partial implementation. The reranker integration and chunk text lookup from tantivy stored fields will be completed when wiring the full pipeline in Task 11 (integration). The fusion math and quality signal logic are the core of this task.

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia-core -- search`
Expected: Unit tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/search.rs
git commit -m "feat: search pipeline (BM25 + vector, RRF fusion k=30, quality signal)"
```

---

## Task 10: Write Pipeline with Auto-Dedup

**Files:**
- Modify: `crates/corvia-core/src/write.rs`

- [ ] **Step 1: Write the write pipeline**

```rust
// crates/corvia-core/src/write.rs
use crate::chunk::chunk_entry;
use crate::config::Config;
use crate::embed::Embedder;
use crate::entry::{new_entry, write_entry_atomic};
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::{Entry, Kind, WriteResponse};
use anyhow::Result;
use std::path::Path;
use tracing::{info, warn};

pub struct WriteParams {
    pub content: String,
    pub kind: Kind,
    pub tags: Vec<String>,
    pub supersedes: Vec<String>,
}

pub fn write(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: WriteParams,
) -> Result<WriteResponse> {
    let entries_dir = base_dir.join(config.entries_dir());
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))?;

    std::fs::create_dir_all(&entries_dir)?;
    std::fs::create_dir_all(base_dir.join(config.index_dir()))?;

    // Step 1: Auto-dedup check (unless caller explicitly set supersedes)
    let mut supersedes = params.supersedes;
    let mut action = "created".to_string();
    let mut similarity = None;
    let mut warnings = Vec::new();

    if supersedes.is_empty() && !params.content.is_empty() {
        // Embed incoming content
        let query_vec = embedder.embed(&params.content)?;

        // Check all existing vectors for near-duplicates
        let all_vectors = redb.all_vectors()?;
        let superseded_ids = redb.superseded_ids()?;
        let mut best_match: Option<(String, f32)> = None;

        for (chunk_id, vec) in &all_vectors {
            if let Some(entry_id) = redb.chunk_entry_id(chunk_id)? {
                if superseded_ids.contains(&entry_id) {
                    continue;
                }
                let sim = Embedder::cosine_similarity(&query_vec, vec);
                if sim >= config.search.dedup_threshold {
                    if best_match.as_ref().is_none_or(|(_, best_sim)| sim > *best_sim) {
                        best_match = Some((entry_id, sim));
                    }
                }
            }
        }

        if let Some((matched_id, sim)) = best_match {
            supersedes = vec![matched_id];
            action = "superseded".to_string();
            similarity = Some(sim);
            info!("auto-dedup: superseding entry with similarity {sim:.3}");
        }
    }

    // Validate supersedes references
    for sup_id in &supersedes {
        if !redb.entry_exists(sup_id)? {
            let msg = format!("superseded entry '{sup_id}' not found");
            warn!("{msg}");
            warnings.push(msg);
        }
    }

    // Step 2: Create entry
    let entry = new_entry(params.content, params.kind, params.tags, supersedes.clone());
    let entry_id = entry.meta.id.clone();

    // Step 3: Write file (atomic)
    write_entry_atomic(&entries_dir, &entry)?;

    // Step 4: Update indexes
    // Mark superseded entries
    for sup_id in &supersedes {
        redb.set_superseded(sup_id, true)?;
    }
    redb.set_superseded(&entry_id, false)?;

    // Chunk, embed, index
    let chunks = chunk_entry(
        &entry,
        config.chunking.max_tokens,
        config.chunking.overlap_tokens,
        config.chunking.min_tokens,
    );

    let tantivy_writer = tantivy.writer()?;
    for chunk in &chunks {
        let chunk_id = format!("{}:{}", chunk.source_entry_id, chunk.chunk_index);
        if !chunk.text.is_empty() {
            let vector = embedder.embed(&chunk.text)?;
            redb.put_vector(&chunk_id, &entry_id, &vector)?;
        }
        tantivy.add_doc(
            &tantivy_writer,
            &chunk_id,
            &entry_id,
            &chunk.text,
            chunk.kind,
            false,
        )?;
    }
    tantivy_writer.commit()?;

    // Update entry count
    let count = redb.entry_count()?;
    redb.set_meta("entry_count", &count.to_string())?;

    let warning = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };

    Ok(WriteResponse {
        id: entry_id,
        action,
        superseded: supersedes,
        warning,
    })
}
```

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p corvia-core`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/src/write.rs
git commit -m "feat: write pipeline with auto-dedup supersession"
```

---

## Task 11: CLI Commands

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

- [ ] **Step 1: Wire CLI commands to core functions**

Replace `crates/corvia-cli/src/main.rs` with the full implementation that dispatches each subcommand to the appropriate corvia-core function. Each handler: loads config, initializes embedder if needed, calls the core function, prints results.

The `ingest` handler calls `corvia_core::ingest::ingest()`.
The `search` handler calls `corvia_core::search::search()`.
The `write` handler calls `corvia_core::write::write()`.
The `status` handler opens RedbIndex and TantivyIndex, reads counts.
The `mcp` handler delegates to `mcp.rs` (Task 12).

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p corvia`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-cli/src/main.rs
git commit -m "feat: CLI commands (ingest, search, write, status, mcp)"
```

---

## Task 12: MCP Server

**Files:**
- Create: `crates/corvia-cli/src/mcp.rs`
- Modify: `crates/corvia-cli/src/main.rs` (add `mod mcp;`)

- [ ] **Step 1: Write rmcp stdio server with 3 tools**

```rust
// crates/corvia-cli/src/mcp.rs
// Implement the rmcp stdio server exposing corvia_search, corvia_write, corvia_status.
// Uses rmcp's Server trait with stdio transport.
// Each tool handler: deserializes params, calls corvia-core function, serializes response.
```

The MCP server registers 3 tools matching the schemas in the spec (Section 5).
Tool descriptions include the kind taxonomy for LLM classification.
Error responses return structured JSON with the error message.

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p corvia`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-cli/src/mcp.rs crates/corvia-cli/src/main.rs
git commit -m "feat: MCP stdio server (corvia_search, corvia_write, corvia_status)"
```

---

## Task 13: Test Fixtures

**Files:**
- Create: `crates/corvia-core/tests/fixtures/decision-storage.md`
- Create: `crates/corvia-core/tests/fixtures/learning-redb-gotcha.md`
- Create: `crates/corvia-core/tests/fixtures/instruction-setup.md`
- Create: `crates/corvia-core/tests/fixtures/reference-api.md`
- Create: `crates/corvia-core/tests/fixtures/long-entry.md`
- Create: `crates/corvia-core/tests/common/mod.rs`

- [ ] **Step 1: Create fixture entry files**

One per kind, plus a long entry for chunking tests:

```markdown
<!-- decision-storage.md -->
+++
id = "fixture-decision-01"
created_at = "2026-04-01T10:00:00Z"
kind = "decision"
tags = ["storage", "architecture"]
+++

We chose Redb over SQLite for the index store. Redb is pure Rust, has ACID
transactions, and requires no system dependencies. SQLite would have added
a C compilation requirement.
```

```markdown
<!-- learning-redb-gotcha.md -->
+++
id = "fixture-learning-01"
created_at = "2026-04-02T10:00:00Z"
kind = "learning"
tags = ["redb", "gotcha"]
+++

Redb uses file-level locking. If two processes try to open the same database
for writing, the second process blocks until the first commits. This means
corvia write and corvia mcp should not run concurrently.
```

```markdown
<!-- instruction-setup.md -->
+++
id = "fixture-instruction-01"
created_at = "2026-04-03T10:00:00Z"
kind = "instruction"
tags = ["setup"]
+++

To install corvia, download the binary for your platform from GitHub releases.
Add the MCP config to your settings.json. Run corvia ingest to build the index.
```

```markdown
<!-- reference-api.md -->
+++
id = "fixture-reference-01"
created_at = "2026-04-04T10:00:00Z"
kind = "reference"
tags = ["api", "mcp"]
+++

corvia_search accepts query (required), limit (default 5), max_tokens, min_score,
and kind parameters. Returns results array with id, kind, score, content fields
plus a quality signal with confidence and suggestion.
```

```markdown
<!-- long-entry.md -->
+++
id = "fixture-long-01"
created_at = "2026-04-05T10:00:00Z"
kind = "reference"
tags = ["chunking-test"]
+++

[Generate 600+ words of technical content about retrieval pipelines to test chunking]
```

- [ ] **Step 2: Create test harness**

```rust
// crates/corvia-core/tests/common/mod.rs
use corvia_core::config::Config;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub struct TestHarness {
    pub dir: TempDir,
    pub config: Config,
}

impl TestHarness {
    pub fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let config = Config {
            data_dir: PathBuf::from(".corvia"),
            ..Config::default()
        };
        Self { dir, config }
    }

    pub fn base_dir(&self) -> &Path {
        self.dir.path()
    }

    pub fn copy_fixtures(&self) {
        let entries_dir = self.dir.path().join(".corvia/entries");
        std::fs::create_dir_all(&entries_dir).unwrap();

        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for entry in std::fs::read_dir(&fixtures_dir).unwrap() {
            let entry = entry.unwrap();
            if entry.path().extension().is_some_and(|e| e == "md") {
                let dest = entries_dir.join(entry.file_name());
                std::fs::copy(entry.path(), dest).unwrap();
            }
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-core/tests/
git commit -m "feat: test fixtures and harness for integration tests"
```

---

## Task 14: Integration Tests

**Files:**
- Create: `crates/corvia-core/tests/integration.rs`

- [ ] **Step 1: Write integration tests**

```rust
// crates/corvia-core/tests/integration.rs
mod common;

use corvia_core::config::Config;
use corvia_core::embed::Embedder;
use corvia_core::entry::{new_entry, write_entry_atomic, read_entry};
use corvia_core::ingest::ingest;
use corvia_core::search::{search, SearchParams};
use corvia_core::write::{write, WriteParams};
use corvia_core::types::{Kind, Confidence};
use common::TestHarness;

#[test]
#[ignore] // requires model download
fn ingest_and_search_known_answer() {
    let h = TestHarness::new();
    h.copy_fixtures();

    let result = ingest(&h.config, h.base_dir(), false).unwrap();
    assert!(result.entries_ingested >= 4);
    assert!(result.entries_skipped.is_empty());

    let embedder = Embedder::new(None).unwrap();
    let resp = search(&h.config, h.base_dir(), &embedder, &SearchParams {
        query: "why did we choose Redb".into(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    }).unwrap();

    assert!(!resp.results.is_empty());
    // The decision about Redb should be in top results
    assert!(resp.results.iter().any(|r| r.id == "fixture-decision-01"));
}

#[test]
#[ignore]
fn write_then_search_finds_entry() {
    let h = TestHarness::new();
    let embedder = Embedder::new(None).unwrap();

    // Write an entry
    let resp = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "Tantivy provides fast full-text search in Rust".into(),
        kind: Kind::Reference,
        tags: vec!["search".into()],
        supersedes: vec![],
    }).unwrap();
    assert_eq!(resp.action, "created");

    // Search for it
    let search_resp = search(&h.config, h.base_dir(), &embedder, &SearchParams {
        query: "tantivy full-text search".into(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    }).unwrap();

    assert!(search_resp.results.iter().any(|r| r.id == resp.id));
}

#[test]
#[ignore]
fn auto_dedup_supersedes_similar_entry() {
    let h = TestHarness::new();
    let embedder = Embedder::new(None).unwrap();

    // Write initial entry
    let first = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "We use nomic-embed-text-v1.5 for embeddings".into(),
        kind: Kind::Decision,
        tags: vec![],
        supersedes: vec![],
    }).unwrap();

    // Write near-duplicate
    let second = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "We use nomic-embed-text-v1.5 as the embedding model".into(),
        kind: Kind::Decision,
        tags: vec![],
        supersedes: vec![],
    }).unwrap();

    assert_eq!(second.action, "superseded");
    assert!(second.superseded.contains(&first.id));
}

#[test]
#[ignore]
fn superseded_entry_excluded_from_search() {
    let h = TestHarness::new();
    let embedder = Embedder::new(None).unwrap();

    let old = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "We chose SQLite for storage".into(),
        kind: Kind::Decision,
        tags: vec![],
        supersedes: vec![],
    }).unwrap();

    let _new = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "We chose Redb for storage instead of SQLite".into(),
        kind: Kind::Decision,
        tags: vec![],
        supersedes: vec![old.id.clone()],
    }).unwrap();

    let resp = search(&h.config, h.base_dir(), &embedder, &SearchParams {
        query: "storage choice".into(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    }).unwrap();

    // Old entry should NOT be in results
    assert!(!resp.results.iter().any(|r| r.id == old.id));
}

#[test]
#[ignore]
fn cold_start_returns_helpful_message() {
    let h = TestHarness::new();
    let embedder = Embedder::new(None).unwrap();

    let resp = search(&h.config, h.base_dir(), &embedder, &SearchParams {
        query: "anything".into(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    }).unwrap();

    assert_eq!(resp.quality.confidence, Confidence::None);
    assert!(resp.quality.suggestion.as_deref().unwrap().contains("corvia ingest"));
}

#[test]
#[ignore]
fn malformed_entry_skipped_during_ingest() {
    let h = TestHarness::new();
    let entries_dir = h.base_dir().join(".corvia/entries");
    std::fs::create_dir_all(&entries_dir).unwrap();

    // Good entry
    std::fs::write(
        entries_dir.join("good.md"),
        "+++\nid = \"good\"\ncreated_at = \"2026-01-01T00:00:00Z\"\n+++\nGood content."
    ).unwrap();

    // Bad entry (no id)
    std::fs::write(
        entries_dir.join("bad.md"),
        "+++\nkind = \"learning\"\n+++\nBad content."
    ).unwrap();

    let result = ingest(&h.config, h.base_dir(), false).unwrap();
    assert_eq!(result.entries_ingested, 1);
    assert_eq!(result.entries_skipped.len(), 1);
    assert!(result.entries_skipped[0].0.contains("bad.md"));
}

#[test]
#[ignore]
fn supersession_warning_for_missing_id() {
    let h = TestHarness::new();
    let embedder = Embedder::new(None).unwrap();

    let resp = write(&h.config, h.base_dir(), &embedder, WriteParams {
        content: "Some content".into(),
        kind: Kind::Learning,
        tags: vec![],
        supersedes: vec!["nonexistent-id".into()],
    }).unwrap();

    assert!(resp.warning.is_some());
    assert!(resp.warning.unwrap().contains("not found"));
}

#[test]
fn entry_roundtrip_serialization() {
    // This test does NOT need the embedder
    let dir = tempfile::tempdir().unwrap();
    let entries_dir = dir.path().join("entries");

    let entry = new_entry(
        "Test content with special chars: <>&\"'".into(),
        Kind::Decision,
        vec!["tag=with=equals".into(), "tag with spaces".into()],
        vec![],
    );

    let path = write_entry_atomic(&entries_dir, &entry).unwrap();
    let readback = read_entry(&path).unwrap();

    assert_eq!(readback.meta.id, entry.meta.id);
    assert_eq!(readback.meta.kind, Kind::Decision);
    assert_eq!(readback.body, "Test content with special chars: <>&\"'");
    assert_eq!(readback.meta.tags, entry.meta.tags);
}
```

- [ ] **Step 2: Run non-ignored tests**

Run: `cargo test -p corvia-core --test integration -- --skip ignored`
Expected: `entry_roundtrip_serialization` passes.

- [ ] **Step 3: Run full integration tests (with model download)**

Run: `cargo test -p corvia-core --test integration -- --ignored`
Expected: All tests pass (requires network for model download on first run).

- [ ] **Step 4: Commit**

```bash
git add crates/corvia-core/tests/
git commit -m "feat: integration tests (ingest, search, write, dedup, cold start)"
```

---

## Task 15: MCP E2E Tests

**Files:**
- Create: `crates/corvia-cli/tests/mcp_e2e.rs`

- [ ] **Step 1: Write E2E tests for MCP tools**

Tests that spawn the `corvia mcp` process, send JSON-RPC requests via stdin,
and assert on stdout responses. Tests cover:
- `tools/list` returns 3 tools
- `corvia_write` with valid input returns id + action
- `corvia_search` after write returns the entry
- `corvia_status` returns counts
- `corvia_search` on empty store returns quality suggestion
- `corvia_write` with empty content handled gracefully

- [ ] **Step 2: Run tests**

Run: `cargo test -p corvia --test mcp_e2e -- --ignored`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/corvia-cli/tests/
git commit -m "feat: MCP E2E tests (3 tools, lifecycle, cold start)"
```

---

## Task 16: Final Wiring and Cleanup

**Files:**
- Review and fix any compilation issues across all modules
- Ensure `cargo test` (non-ignored) passes
- Ensure `cargo clippy` is clean

- [ ] **Step 1: Run full check**

```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace -- --skip ignored
```

- [ ] **Step 2: Fix any issues**

Address clippy warnings, unused imports, type mismatches between tasks.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: cleanup, fix clippy warnings, verify all non-ignored tests pass"
```

---

## Execution Summary

| Task | Component | Dependencies | Estimated Size |
|------|-----------|-------------|----------------|
| 1 | Core types | None | Small |
| 2 | Config | None | Small |
| 3 | Entry I/O | Types | Medium |
| 4 | Chunking | Types | Medium |
| 5 | Embedding | None | Medium |
| 6 | Redb Index | None | Medium |
| 7 | Tantivy Index | Types | Medium |
| 8 | Ingest Pipeline | 3, 4, 5, 6, 7 | Large |
| 9 | Search Pipeline | 5, 6, 7 | Large |
| 10 | Write Pipeline | 3, 5, 6, 7 | Large |
| 11 | CLI Commands | 8, 9, 10 | Medium |
| 12 | MCP Server | 9, 10 | Medium |
| 13 | Test Fixtures | 3 | Small |
| 14 | Integration Tests | All core | Large |
| 15 | MCP E2E Tests | 12 | Medium |
| 16 | Final Cleanup | All | Small |

**Critical path:** Tasks 1-7 (foundation) -> Task 8 (ingest) -> Tasks 9-10 (search + write) -> Tasks 11-12 (CLI + MCP) -> Tasks 13-15 (tests) -> Task 16 (cleanup)

**Parallelizable:** Tasks 1+2, Tasks 4+5+6+7, Tasks 11+12, Tasks 14+15
