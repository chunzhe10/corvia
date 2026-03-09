# Adapter Plugin System — Design Document

**Status:** Approved
**Date:** 2026-03-03
**Author:** Lim Chun Zhe + Claude (brainstorming session)
**Decisions:** D72-D79 (see `docs/local/plans/2026-03-03-adapter-plugin-architecture.md`)
**Approach:** Bottom-up — build IPC layer first, test with adapter-basic, then migrate repos

---

## Overview

Adapters become standalone binaries discovered at runtime via JSONL over stdin/stdout.
The kernel never touches the filesystem for source discovery. All first-party adapters
live in the corvia monorepo as workspace crates. Third-party adapters can be written in
any language.

## 1. JSONL Protocol Types

New module: `corvia-kernel/src/adapter_protocol.rs`

### Metadata (returned by `--corvia-metadata`)

```rust
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
```

### Host → Adapter (written to stdin as JSON)

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum AdapterRequest {
    #[serde(rename = "ingest")]
    Ingest { source_path: String, scope_id: String },
    #[serde(rename = "chunk")]
    Chunk { content: String, metadata: SourceMetadata },
}
```

### Adapter → Host (read from stdout, one JSON line per message)

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdapterResponse {
    SourceFile { source_file: SourceFilePayload },
    ChunkResult { chunks: Vec<RawChunk>, relations: Vec<ChunkRelation> },
    Done { done: bool, total_files: usize },
    Error { error: AdapterError },
}
```

### Serde Derives Required

Add `#[derive(Serialize, Deserialize)]` to existing types in `chunking_strategy.rs`:
- `SourceFile`
- `SourceMetadata`
- `RawChunk`
- `ChunkMetadata`
- `ChunkResult`
- `ChunkRelation`

No structural changes — just serde derives.

## 2. Adapter Discovery

New module: `corvia-kernel/src/adapter_discovery.rs`

### Discovery

```
discover_adapters() -> Vec<DiscoveredAdapter>
  1. Scan ~/.config/corvia/adapters/ for corvia-adapter-* executables
  2. Scan $PATH for corvia-adapter-* executables (skip duplicates)
  3. For each: spawn `corvia-adapter-foo --corvia-metadata`, parse JSON
  4. Return Vec<DiscoveredAdapter> { binary_path, metadata }
```

Results cached for the session — discovery runs once at CLI startup.

### Auto-Detection

```
resolve_adapter(path, discovered, config) -> DiscoveredAdapter
  1. If [[sources]] config matches path → use that adapter
  2. If path has .git/ and git adapter installed → use git
  3. Otherwise → use [adapters].default or adapter-basic
```

The `.git/` check is `Path::join(".git").exists()` — no `git2` in the kernel.

### CLI Diagnostics

```
$ corvia adapters list
  git       v0.3.1  Git + tree-sitter code ingestion    /usr/local/bin/corvia-adapter-git
  basic     v0.1.0  Basic filesystem adapter             /usr/local/bin/corvia-adapter-basic
```

## 3. ProcessAdapter — IPC Wrapper

New module: `corvia-kernel/src/process_adapter.rs`

### Struct

```rust
pub struct ProcessAdapter {
    binary_path: PathBuf,
    metadata: AdapterMetadata,
    child: Option<Child>,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: Option<BufReader<ChildStdout>>,
}
```

### Session Lifecycle

```
spawn()     → start adapter process (no args = session mode)
              stdin open for commands, stdout for responses, stderr inherited
ingest()    → write Ingest request, read SourceFile lines until Done
chunk()     → write Chunk request, read single ChunkResult response
shutdown()  → close stdin (adapter sees EOF, exits), wait() for process
```

One process per adapter per session. Adapters are spawned fresh for each ingestion
session and killed on completion.

### ProcessChunkingStrategy

Registered in `FormatRegistry` for extensions the adapter claims in its metadata:

```rust
struct ProcessChunkingStrategy {
    adapter: Arc<Mutex<ProcessAdapter>>,
}

impl ChunkingStrategy for ProcessChunkingStrategy {
    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        self.adapter.lock().chunk(source, meta)
    }
}
```

Three-tier priority preserved:
1. Adapter override (ProcessChunkingStrategy via IPC) — e.g., `.rs` → AstChunker
2. Kernel default (MarkdownChunker, ConfigChunker) — e.g., `.md`, `.toml`
3. Fallback (FallbackChunker line-split) — unknown extensions

Adapters declare what they're smart about. Everything else falls to kernel defaults.

## 4. Refactored cmd_ingest

```
cmd_ingest(path)
  1. discover_adapters()
  2. resolve_adapter(path, ...)
  3. adapter.spawn()
  4. adapter.ingest(path, scope)       → Vec<SourceFile> via JSONL
  5. Register ProcessChunkingStrategy for adapter's chunking_extensions
  6. pipeline.process_batch(&files)    → IPC calls for adapter-claimed exts
  7. embed → store → wire relations
  8. adapter.shutdown()
```

Multi-adapter workspace mode:
```
cmd_ingest_workspace()
  1. discover_adapters()
  2. For each source/repo:
     a. resolve_adapter(repo.path) → pick adapter
     b. spawn → ingest → pipeline → shutdown
```

Sequential per source. No concurrent adapter processes (embedding is the bottleneck).

## 5. Workspace Layout

### New Crates

```
crates/
  corvia-adapter-basic/
    Cargo.toml            # [[bin]] name = "corvia-adapter-basic"
    src/main.rs           # walk dirs, read files, JSONL output (~150 lines)
  corvia-adapter-git/
    Cargo.toml            # [[bin]] name = "corvia-adapter-git"
    src/main.rs           # NEW: JSONL entry point wrapping existing lib
    src/lib.rs            # existing GitAdapter, AstChunker
    src/git.rs            # existing
    src/ast_chunker.rs    # existing
    src/treesitter.rs     # existing
```

### Cargo.toml Changes

- Workspace root: add both adapter crates to `members`
- `corvia-cli/Cargo.toml`: remove `corvia-adapter-git` git dep and `[patch]` section
- Both adapter Cargo.tomls: `corvia-common.workspace = true`, `corvia-kernel.workspace = true`

### Release Workflow

```yaml
cargo build --release -p corvia-cli -p corvia-inference -p corvia-adapter-basic -p corvia-adapter-git
```

Four binaries in one release, all guaranteed compatible.

### External Repo

Archive `chunzhe10/corvia-adapter-git` with README pointing to monorepo location.

## 6. Config Extensions

Add to `corvia-common/src/config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub search_dirs: Vec<String>,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub path: String,
    pub adapter: Option<String>,
    pub adapter_config: Option<toml::Value>,
}
```

Add to `CorviaConfig`:

```rust
pub adapters: Option<AdaptersConfig>,
pub sources: Option<Vec<SourceConfig>>,
```

Both optional. Zero config needed — auto-detection handles everything.

## 7. Testing Strategy

### Protocol Conformance (unit)

Spawn Python `adapters/corvia-adapter-basic/python/corvia-adapter-basic`, send ingest command, validate
JSONL response parses into `Vec<SourceFile>`. Proves cross-language protocol correctness.

### Round-Trip Integration (integration)

Put Rust `corvia-adapter-basic` on PATH, create temp dir with files, run `cmd_ingest`
with discovery, assert entries end up in the store. Exercises the full stack:
discovery → spawn → ingest → pipeline → store.

### Adapter-Specific (unit, existing)

Existing tree-sitter/chunking tests in `corvia-adapter-git` stay as library unit tests.
The `main.rs` JSONL wrapper is thin — protocol conformance test covers it.

## Files Affected

| File | Change |
|------|--------|
| `Cargo.toml` | Add adapter crates to workspace members |
| `crates/corvia-kernel/src/adapter_protocol.rs` | NEW — protocol types |
| `crates/corvia-kernel/src/adapter_discovery.rs` | NEW — discovery + auto-detect |
| `crates/corvia-kernel/src/process_adapter.rs` | NEW — IPC wrapper |
| `crates/corvia-kernel/src/chunking_strategy.rs` | Add serde derives |
| `crates/corvia-kernel/src/chunking_pipeline.rs` | Add ProcessChunkingStrategy |
| `crates/corvia-kernel/src/lib.rs` | Export new modules |
| `crates/corvia-cli/src/main.rs` | Refactor cmd_ingest to use discovery |
| `crates/corvia-cli/Cargo.toml` | Remove adapter-git dep and [patch] |
| `crates/corvia-common/src/config.rs` | Add AdaptersConfig, SourceConfig |
| `crates/corvia-adapter-basic/` | NEW crate — filesystem adapter binary |
| `crates/corvia-adapter-git/` | MOVED from external repo, add main.rs wrapper |
| `.github/workflows/release.yml` | Build all four binaries |
| `adapters/corvia-adapter-basic/python/corvia-adapter-basic` | Python reference adapter (already exists) |
