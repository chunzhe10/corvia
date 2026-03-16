# M3.1 + M3.2 + M3.3 Unified Parallel Implementation Plan

> **Status:** Shipped (v0.3.0)

**Goal:** Coordinate parallel implementation of three milestones — gRPC Inference Server (M3.1), RAG Pipeline (M3.2), and Embedding & Chunking (M3.3) — with explicit handoff points where one milestone unblocks another.

**Architecture:** Three independent workstreams converge at defined handoff points. Shared code (traits, types, config) is extracted early as "foundation tasks" that unblock all three workstreams. Each workstream has its own detailed implementation plan; this document defines the sequencing, handoff contracts, and integration tasks.

**Tech Stack:** Rust workspace, async-trait, tonic/prost (M3.1), axum (M3.2), tree-sitter (M3.3)

**Individual plans:**
- M3.1: `docs/rfcs/2026-03-02-grpc-inference-server-impl.md` (14 tasks)
- M3.2: `docs/rfcs/2026-03-02-m3.2-rag-pipeline-impl.md` (tasks TBD)
- M3.3: `docs/rfcs/2026-03-02-m3.3-embedding-chunking-design.md` (impl plan below)

---

## Dependency Graph

```
PHASE 0: Foundation (shared traits + types)
├── F1: GenerationEngine trait + ChatMessage types     ── unblocks M3.1, M3.2
├── F2: TokenEstimator trait + CharDivFourEstimator     ── unblocks M3.2, M3.3
├── F3: ChunkingStrategy trait + core types             ── unblocks M3.3
├── F4: RAG types (RetrievalOpts, TokenBudget, etc.)   ── unblocks M3.2
├── F5: Config additions (RagConfig, ChunkingConfig)    ── unblocks M3.2, M3.3
└── F6: InferenceProvider::Corvia enum variant          ── unblocks M3.1

PHASE 1: Independent workstreams (parallel)
├── M3.1: gRPC Inference Server (Tasks 1-14)
│   ├── corvia-proto crate + proto files
│   ├── GrpcInferenceEngine, GrpcChatEngine, GrpcVllmEngine
│   ├── corvia-inference binary (ONNX + candle)
│   └── InferenceProvisioner
│
├── M3.2: RAG Pipeline (partial — context mode)
│   ├── Retriever trait + VectorRetriever
│   ├── GraphExpandRetriever
│   ├── Augmenter trait + StructuredAugmenter
│   ├── RagPipeline (context mode only)
│   └── REST/MCP endpoints (context + search fix)
│
└── M3.3: Chunking Framework
    ├── ChunkingPipeline + FormatRegistry
    ├── FallbackChunker, MarkdownChunker, ConfigChunker
    ├── PdfChunker
    ├── D69 IngestionAdapter revision
    └── AstChunker (corvia-adapter-git)

HANDOFF H1: GenerationEngine trait lands (F1)
  → M3.2 can define ask() signature
  → M3.1 builds implementations against the trait

HANDOFF H2: GrpcChatEngine lands (M3.1 Task 6)
  → M3.2 can wire ask() mode with real GenerationEngine

HANDOFF H3: ChunkingPipeline lands (M3.3)
  → Ingestion flow can be rewired: adapter → chunk → embed → store

PHASE 2: Integration
├── I1: Wire ask() mode with GenerationEngine (needs H2)
├── I2: Wire ingestion through ChunkingPipeline (needs H3)
├── I3: Update create_engine() for Corvia provider
├── I4: E2E integration tests (all three milestones)
└── I5: Final server wiring + CLI updates
```

## Task Sequencing Rules

1. **All Phase 0 (Foundation) tasks must complete before Phase 1 begins.**
   Foundation tasks are small (trait definitions, type structs, config additions) and
   can themselves be parallelized since they touch different files.

2. **Phase 1 workstreams run in parallel.** No cross-dependencies between M3.1, M3.2,
   and M3.3 during Phase 1 — they all code against traits, not implementations.

3. **Phase 2 integration tasks require specific handoffs.** Each integration task lists
   its prerequisites explicitly.

---

## Phase 0: Foundation Tasks

These establish the shared contracts all three workstreams build against. Small,
trait-definition-only tasks that land first.

### Task F1: GenerationEngine Trait + ChatMessage Types

**Files:**
- Modify: `crates/corvia-kernel/src/traits.rs`

**Unblocks:** M3.1 (builds implementations), M3.2 (uses in ask() mode)

**Step 1: Add GenerationEngine trait and ChatMessage to traits.rs**

Append to `crates/corvia-kernel/src/traits.rs`:

```rust
/// Chat message for generation requests.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,    // "system", "user", "assistant"
    pub content: String,
}

/// Text generation engine for RAG answers and LLM-assisted merge (D63).
///
/// Wire protocol name (`ChatService` in proto) is separate from this
/// Rust capability trait. Implementations: GrpcChatEngine (corvia-inference),
/// OllamaChatEngine (HTTP), GrpcVllmChatEngine (vLLM).
#[async_trait]
pub trait GenerationEngine: Send + Sync {
    /// Generate a text response from a system prompt and user message.
    async fn generate(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<GenerationResult>;

    /// Return the model's context window size in tokens.
    fn context_window(&self) -> usize;
}

/// Result from a GenerationEngine call.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}
```

**Step 2: Update lib.rs re-exports**

In `crates/corvia-kernel/src/lib.rs`, the traits module is already public. Verify
`GenerationEngine` and `GenerationResult` are accessible as
`corvia_kernel::traits::GenerationEngine`.

**Step 3: Write test**

In `crates/corvia-kernel/src/traits.rs`, add to the existing test module (or create
`#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time test: verify trait is object-safe
    fn _assert_object_safe(_: &dyn GenerationEngine) {}

    #[test]
    fn test_chat_message_construction() {
        let msg = ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn test_generation_result_construction() {
        let result = GenerationResult {
            text: "answer".into(),
            model: "test-model".into(),
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(result.text, "answer");
        assert_eq!(result.input_tokens, 100);
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p corvia-kernel --lib traits`
Expected: PASS (3 new tests)

**Step 5: Commit**

```bash
git add crates/corvia-kernel/src/traits.rs
git commit -m "feat(kernel): add GenerationEngine trait and ChatMessage types (D63)"
```

---

### Task F2: TokenEstimator Trait

**Files:**
- Create: `crates/corvia-kernel/src/token_estimator.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module)

**Unblocks:** M3.2 (StructuredAugmenter token budget), M3.3 (ChunkingPipeline budget enforcement)

**Step 1: Write the failing test**

Create `crates/corvia-kernel/src/token_estimator.rs`:

```rust
/// Pluggable token estimation (shared by M3.2 RAG pipeline and M3.3 chunking).
///
/// Default implementation: chars / 4 heuristic. Swap in tiktoken or
/// sentencepiece later without changing consumers.
pub trait TokenEstimator: Send + Sync {
    fn estimate(&self, text: &str) -> usize;
}

/// Default estimator: character count divided by 4.
/// Industry-standard heuristic for English text. Overestimates slightly
/// for code (good — conservative budget enforcement).
pub struct CharDivFourEstimator;

impl TokenEstimator for CharDivFourEstimator {
    fn estimate(&self, text: &str) -> usize {
        // Minimum 1 token for non-empty strings
        if text.is_empty() {
            0
        } else {
            (text.len() / 4).max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let est = CharDivFourEstimator;
        assert_eq!(est.estimate(""), 0);
    }

    #[test]
    fn test_short_string() {
        let est = CharDivFourEstimator;
        // "hi" = 2 chars, 2/4 = 0, but min 1
        assert_eq!(est.estimate("hi"), 1);
    }

    #[test]
    fn test_typical_text() {
        let est = CharDivFourEstimator;
        let text = "The quick brown fox jumps over the lazy dog.";
        // 44 chars / 4 = 11
        assert_eq!(est.estimate(text), 11);
    }

    #[test]
    fn test_code_text() {
        let est = CharDivFourEstimator;
        let code = "fn main() {\n    println!(\"hello\");\n}";
        let tokens = est.estimate(code);
        assert!(tokens > 0);
        assert_eq!(tokens, code.len() / 4);
    }

    #[test]
    fn test_object_safety() {
        let est: Box<dyn TokenEstimator> = Box::new(CharDivFourEstimator);
        assert_eq!(est.estimate("test"), 1);
    }
}
```

**Step 2: Add module declaration**

In `crates/corvia-kernel/src/lib.rs`, add:
```rust
pub mod token_estimator;
```

**Step 3: Run tests**

Run: `cargo test -p corvia-kernel token_estimator`
Expected: PASS (5 tests)

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/token_estimator.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(kernel): add TokenEstimator trait with chars/4 default (D64/D65)"
```

---

### Task F3: ChunkingStrategy Trait + Core Types

**Files:**
- Create: `crates/corvia-kernel/src/chunking_strategy.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module)

**Unblocks:** M3.3 (all chunker implementations + pipeline)

**Step 1: Write the trait and types**

Create `crates/corvia-kernel/src/chunking_strategy.rs`:

```rust
use corvia_common::errors::Result;
use uuid::Uuid;

/// Metadata about the source file being chunked.
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub file_path: String,
    pub extension: String,
    pub language: Option<String>,
    pub scope_id: String,
    pub source_version: String,
}

/// Raw chunk produced by a ChunkingStrategy (pre-budget, pre-merge).
#[derive(Debug, Clone)]
pub struct RawChunk {
    pub content: String,
    /// Semantic type: "function_item", "heading_section", "key_value", "file", etc.
    pub chunk_type: String,
    pub start_line: u32,
    pub end_line: u32,
    pub metadata: ChunkMetadata,
}

/// Additional chunk metadata for provenance and graph linking.
#[derive(Debug, Clone, Default)]
pub struct ChunkMetadata {
    pub source_file: String,
    pub language: Option<String>,
    /// For split chunks — links back to original.
    pub parent_chunk_id: Option<Uuid>,
    /// Hint for which chunks can merge (e.g., same impl block).
    pub merge_group: Option<String>,
}

/// Final chunk after pipeline processing — ready for embedding.
#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    /// Content including overlap prefix (for embedding).
    pub content: String,
    /// Original content without overlap (for display/citation).
    pub original_content: String,
    pub chunk_type: String,
    pub start_line: u32,
    pub end_line: u32,
    pub metadata: ChunkMetadata,
    pub token_estimate: usize,
    pub processing: ProcessingInfo,
}

/// How this chunk was processed by the pipeline.
#[derive(Debug, Clone)]
pub struct ProcessingInfo {
    pub strategy_name: String,
    pub was_split: bool,
    pub was_merged: bool,
    pub overlap_tokens: usize,
}

/// Raw source file returned by IngestionAdapter (D69).
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub content: String,
    pub metadata: SourceMetadata,
}

/// Domain-specific chunking strategy (D65).
///
/// Template method pattern: adapters implement domain logic (chunk, split,
/// merge, overlap). The kernel's `ChunkingPipeline` enforces universal
/// concerns (budget limits, merge ordering, split recursion).
pub trait ChunkingStrategy: Send + Sync {
    /// Identifier for metrics attribution and benchmarking.
    fn name(&self) -> &str;

    /// Supported file extensions this strategy handles.
    fn supported_extensions(&self) -> &[&str];

    /// Domain-specific: extract semantic boundaries from source.
    /// Returns raw chunks WITHOUT embeddings.
    fn chunk(
        &self,
        source: &str,
        meta: &SourceMetadata,
    ) -> Result<Vec<RawChunk>>;

    /// Domain-specific: split an oversized chunk that exceeds token budget.
    /// Default: naive midpoint split.
    fn split_oversized(
        &self,
        chunk: &RawChunk,
        max_tokens: usize,
    ) -> Result<Vec<RawChunk>> {
        let mid = chunk.content.len() / 2;
        // Find nearest newline to avoid splitting mid-line
        let split_pos = chunk.content[..mid]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(mid);

        let mid_line = chunk.start_line
            + chunk.content[..split_pos].matches('\n').count() as u32;

        Ok(vec![
            RawChunk {
                content: chunk.content[..split_pos].to_string(),
                chunk_type: chunk.chunk_type.clone(),
                start_line: chunk.start_line,
                end_line: mid_line,
                metadata: ChunkMetadata {
                    parent_chunk_id: Some(Uuid::now_v7()),
                    ..chunk.metadata.clone()
                },
            },
            RawChunk {
                content: chunk.content[split_pos..].to_string(),
                chunk_type: chunk.chunk_type.clone(),
                start_line: mid_line + 1,
                end_line: chunk.end_line,
                metadata: ChunkMetadata {
                    parent_chunk_id: Some(Uuid::now_v7()),
                    ..chunk.metadata.clone()
                },
            },
        ])
    }

    /// Domain-specific: merge small adjacent chunks up to budget.
    /// Return None to use kernel's default merge pass.
    fn merge_small(
        &self,
        _chunks: &[RawChunk],
        _max_tokens: usize,
    ) -> Option<Vec<RawChunk>> {
        None // default: kernel handles merge
    }

    /// Domain-specific: what context to carry between adjacent chunks.
    /// Default: no overlap.
    fn overlap_context(
        &self,
        _prev: &RawChunk,
        _next: &RawChunk,
    ) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify trait is object-safe
    fn _assert_object_safe(_: &dyn ChunkingStrategy) {}

    #[test]
    fn test_source_metadata_construction() {
        let meta = SourceMetadata {
            file_path: "src/main.rs".into(),
            extension: "rs".into(),
            language: Some("rust".into()),
            scope_id: "test".into(),
            source_version: "abc123".into(),
        };
        assert_eq!(meta.extension, "rs");
    }

    #[test]
    fn test_raw_chunk_construction() {
        let chunk = RawChunk {
            content: "fn main() {}".into(),
            chunk_type: "function_item".into(),
            start_line: 1,
            end_line: 1,
            metadata: ChunkMetadata {
                source_file: "main.rs".into(),
                language: Some("rust".into()),
                ..Default::default()
            },
        };
        assert_eq!(chunk.chunk_type, "function_item");
    }

    #[test]
    fn test_processed_chunk_construction() {
        let chunk = ProcessedChunk {
            content: "// overlap\nfn main() {}".into(),
            original_content: "fn main() {}".into(),
            chunk_type: "function_item".into(),
            start_line: 1,
            end_line: 1,
            metadata: ChunkMetadata::default(),
            token_estimate: 5,
            processing: ProcessingInfo {
                strategy_name: "ast".into(),
                was_split: false,
                was_merged: false,
                overlap_tokens: 2,
            },
        };
        assert_eq!(chunk.processing.strategy_name, "ast");
        assert!(!chunk.processing.was_split);
    }

    /// Test the default split_oversized implementation
    struct TestStrategy;
    impl ChunkingStrategy for TestStrategy {
        fn name(&self) -> &str { "test" }
        fn supported_extensions(&self) -> &[&str] { &["txt"] }
        fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<Vec<RawChunk>> {
            Ok(vec![RawChunk {
                content: source.to_string(),
                chunk_type: "file".into(),
                start_line: 1,
                end_line: source.lines().count() as u32,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    ..Default::default()
                },
            }])
        }
    }

    #[test]
    fn test_default_split_oversized() {
        let strategy = TestStrategy;
        let chunk = RawChunk {
            content: "line one\nline two\nline three\nline four\n".into(),
            chunk_type: "file".into(),
            start_line: 1,
            end_line: 4,
            metadata: ChunkMetadata::default(),
        };
        let parts = strategy.split_oversized(&chunk, 5).unwrap();
        assert_eq!(parts.len(), 2);
        assert!(!parts[0].content.is_empty());
        assert!(!parts[1].content.is_empty());
    }

    #[test]
    fn test_default_merge_returns_none() {
        let strategy = TestStrategy;
        assert!(strategy.merge_small(&[], 100).is_none());
    }

    #[test]
    fn test_default_overlap_returns_none() {
        let strategy = TestStrategy;
        let chunk = RawChunk {
            content: "test".into(),
            chunk_type: "file".into(),
            start_line: 1,
            end_line: 1,
            metadata: ChunkMetadata::default(),
        };
        assert!(strategy.overlap_context(&chunk, &chunk).is_none());
    }
}
```

**Step 2: Add module declaration**

In `crates/corvia-kernel/src/lib.rs`, add:
```rust
pub mod chunking_strategy;
```

**Step 3: Run tests**

Run: `cargo test -p corvia-kernel chunking_strategy`
Expected: PASS (7 tests)

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/chunking_strategy.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(kernel): add ChunkingStrategy trait + chunking types (D65)"
```

---

### Task F4: RAG Types

**Files:**
- Create: `crates/corvia-kernel/src/rag_types.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add module)

**Unblocks:** M3.2 (all RAG pipeline tasks)

Follow Task 1 from `docs/rfcs/2026-03-02-m3.2-rag-pipeline-impl.md` exactly.
That plan defines `RetrievalOpts`, `TokenBudget`, `RetrievalResult`, `RetrievalMetrics`,
`AugmentedContext`, `AugmentationMetrics`, `GenerationMetrics`, `PipelineTrace`,
`RagResponse`, and `RagConfig`.

**Step 1: Implement per M3.2 Task 1**

Create the file as specified in the M3.2 impl plan.

**Step 2: Run tests**

Run: `cargo test -p corvia-kernel rag_types`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/corvia-kernel/src/rag_types.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(kernel): add RAG pipeline types (D61)"
```

---

### Task F5: Config Additions (RagConfig + ChunkingConfig)

**Files:**
- Modify: `crates/corvia-common/src/config.rs`

**Unblocks:** M3.2 (RagPipeline config), M3.3 (ChunkingPipeline config)

**Step 1: Add RagConfig**

In `crates/corvia-common/src/config.rs`, add:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default = "default_rag_limit")]
    pub default_limit: usize,
    #[serde(default = "default_graph_expand")]
    pub graph_expand: bool,
    #[serde(default = "default_graph_depth")]
    pub graph_depth: usize,
    #[serde(default = "default_graph_alpha")]
    pub graph_alpha: f32,
    #[serde(default = "default_reserve_for_answer")]
    pub reserve_for_answer: f32,
    #[serde(default)]
    pub max_context_tokens: usize, // 0 = model-aware auto-sizing
    #[serde(default)]
    pub system_prompt: String,     // empty = built-in default
}

fn default_rag_limit() -> usize { 10 }
fn default_graph_expand() -> bool { true }
fn default_graph_depth() -> usize { 2 }
fn default_graph_alpha() -> f32 { 0.3 }
fn default_reserve_for_answer() -> f32 { 0.2 }

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            default_limit: default_rag_limit(),
            graph_expand: default_graph_expand(),
            graph_depth: default_graph_depth(),
            graph_alpha: default_graph_alpha(),
            reserve_for_answer: default_reserve_for_answer(),
            max_context_tokens: 0,
            system_prompt: String::new(),
        }
    }
}
```

**Step 2: Add ChunkingConfig**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    #[serde(default = "default_max_chunk_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_min_chunk_tokens")]
    pub min_tokens: usize,
    #[serde(default = "default_overlap_tokens")]
    pub overlap_tokens: usize,
    #[serde(default = "default_chunking_strategy")]
    pub strategy: String, // "auto" = format registry
}

fn default_max_chunk_tokens() -> usize { 512 }
fn default_min_chunk_tokens() -> usize { 32 }
fn default_overlap_tokens() -> usize { 64 }
fn default_chunking_strategy() -> String { "auto".into() }

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_max_chunk_tokens(),
            min_tokens: default_min_chunk_tokens(),
            overlap_tokens: default_overlap_tokens(),
            strategy: default_chunking_strategy(),
        }
    }
}
```

**Step 3: Add fields to CorviaConfig**

Add to the `CorviaConfig` struct:
```rust
#[serde(default)]
pub rag: RagConfig,
#[serde(default)]
pub chunking: ChunkingConfig,
```

Update `Default for CorviaConfig` to include:
```rust
rag: RagConfig::default(),
chunking: ChunkingConfig::default(),
```

**Step 4: Write tests**

```rust
#[test]
fn test_rag_config_defaults() {
    let config = RagConfig::default();
    assert_eq!(config.default_limit, 10);
    assert!(config.graph_expand);
    assert_eq!(config.graph_depth, 2);
    assert!((config.graph_alpha - 0.3).abs() < f32::EPSILON);
    assert!((config.reserve_for_answer - 0.2).abs() < f32::EPSILON);
    assert_eq!(config.max_context_tokens, 0);
    assert!(config.system_prompt.is_empty());
}

#[test]
fn test_chunking_config_defaults() {
    let config = ChunkingConfig::default();
    assert_eq!(config.max_tokens, 512);
    assert_eq!(config.min_tokens, 32);
    assert_eq!(config.overlap_tokens, 64);
    assert_eq!(config.strategy, "auto");
}

#[test]
fn test_rag_config_roundtrip() {
    let mut config = CorviaConfig::default();
    config.rag.default_limit = 20;
    config.rag.graph_alpha = 0.5;
    let toml_str = toml::to_string_pretty(&config).unwrap();
    let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(loaded.rag.default_limit, 20);
    assert!((loaded.rag.graph_alpha - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_chunking_config_roundtrip() {
    let mut config = CorviaConfig::default();
    config.chunking.max_tokens = 1024;
    config.chunking.overlap_tokens = 128;
    let toml_str = toml::to_string_pretty(&config).unwrap();
    let loaded: CorviaConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(loaded.chunking.max_tokens, 1024);
    assert_eq!(loaded.chunking.overlap_tokens, 128);
}

#[test]
fn test_existing_config_without_rag_or_chunking_still_parses() {
    // Backward compat: configs without [rag] or [chunking] get defaults
    let toml_str = r#"
[project]
name = "test"
scope_id = "test"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020
"#;
    let config: CorviaConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.rag.default_limit, 10);
    assert_eq!(config.chunking.max_tokens, 512);
}
```

**Step 5: Run tests**

Run: `cargo test -p corvia-common config`
Expected: PASS (all existing + 5 new tests)

**Step 6: Commit**

```bash
git add crates/corvia-common/src/config.rs
git commit -m "feat(config): add RagConfig and ChunkingConfig sections (D61/D65)"
```

---

### Task F6: InferenceProvider::Corvia Enum Variant

**Files:**
- Modify: `crates/corvia-common/src/config.rs`

**Unblocks:** M3.1 (Corvia gRPC provider selection)

**Step 1: Add variant**

In `InferenceProvider` enum, add:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceProvider {
    #[default]
    Ollama,
    Vllm,
    Corvia,
}
```

**Step 2: Add MergeConfig provider field**

In `MergeConfig`, add:
```rust
#[serde(default)]
pub provider: InferenceProvider,
```

And update `Default for MergeConfig`:
```rust
provider: InferenceProvider::Ollama,
```

**Step 3: Write test**

```rust
#[test]
fn test_inference_provider_corvia_serde() {
    let json = serde_json::to_string(&InferenceProvider::Corvia).unwrap();
    assert_eq!(json, "\"corvia\"");
    let parsed: InferenceProvider = serde_json::from_str("\"corvia\"").unwrap();
    assert_eq!(parsed, InferenceProvider::Corvia);
}
```

**Step 4: Run tests**

Run: `cargo test -p corvia-common config`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/corvia-common/src/config.rs
git commit -m "feat(config): add InferenceProvider::Corvia variant (D60)"
```

---

## Phase 0 Completion Checkpoint

Run: `cargo test --workspace`
Expected: All 203+ existing tests pass, plus ~20 new Foundation tests.
All three workstreams can now begin.

```bash
git tag phase-0-foundation
```

---

## Phase 1: Parallel Workstreams

### Workstream A: M3.1 — gRPC Inference Server

**Detailed plan:** `docs/rfcs/2026-03-02-grpc-inference-server-impl.md`
**Tasks:** 14 (numbered 1-14 in that plan)
**Produces:** corvia-proto crate, corvia-inference binary, GrpcInferenceEngine,
  GrpcChatEngine, GrpcVllmEngine, InferenceProvisioner

**Phase 1 scope (Tasks A1-A12):**

| Task | Description | Deps | Files |
|------|-------------|------|-------|
| A1 | Create corvia-proto crate + 3 proto files | F1, F6 | `crates/corvia-proto/**` |
| A2 | Refactor MergeWorker to use GenerationEngine trait | F1 | `merge_worker.rs` |
| A3 | GrpcInferenceEngine client (InferenceEngine impl) | A1 | `engines/grpc_inference_engine.rs` |
| A4 | GrpcChatEngine client (GenerationEngine impl) | A1, F1 | `engines/grpc_chat_engine.rs` |
| A5 | InferenceProvisioner lifecycle manager | A1 | `provisioners/inference_provisioner.rs` |
| A6 | Scaffold corvia-inference binary + health endpoint | A1 | `crates/corvia-inference/**` |
| A7 | EmbeddingService (fastembed/ONNX) | A6 | `corvia-inference/src/embedding.rs` |
| A8 | ChatService scaffold (candle stub) | A6 | `corvia-inference/src/chat.rs` |
| A9 | Wire Corvia provider into create_engine() | A3, A4, A5, F6 | `lib.rs` |
| A10 | gRPC integration tests | A3, A4, A7 | `tests/` |
| A11 | GrpcVllmEngine (vLLM gRPC client) | A1 | `engines/grpc_vllm_engine.rs` |
| A12 | Candle chat implementation | A8 | `corvia-inference/src/chat.rs` |

**Handoff H1 produced:** After A4, `GrpcChatEngine` implements `GenerationEngine`.
M3.2 can now wire `ask()` mode.

Follow the M3.1 impl plan for detailed step-by-step instructions per task.

---

### Workstream B: M3.2 — RAG Pipeline (context mode)

**Detailed plan:** `docs/rfcs/2026-03-02-m3.2-rag-pipeline-impl.md`
**Phase 1 scope:** Everything EXCEPT `ask()` mode wiring (that's Phase 2).

| Task | Description | Deps | Files |
|------|-------------|------|-------|
| B1 | RAG types (done as F4) | — | `rag_types.rs` |
| B2 | Retriever trait + VectorRetriever | F4, F2 | `retriever.rs` |
| B3 | GraphExpandRetriever | B2 | `retriever.rs` |
| B4 | Augmenter trait + StructuredAugmenter | F2, F4 | `augmenter.rs` |
| B5 | RagPipeline orchestrator (context mode) | B2, B4 | `rag_pipeline.rs` |
| B6 | Fix search routing (REST + MCP) | B2 | `rest.rs`, `mcp.rs` |
| B7 | REST endpoint: POST /v1/context | B5 | `rest.rs` |
| B8 | MCP tool: corvia_context | B5 | `mcp.rs` |
| B9 | Unit + integration tests | B5, B6 | `tests/` |

Follow the M3.2 impl plan for detailed step-by-step instructions per task.
Skip `ask()` endpoint until Phase 2 (I1).

---

### Workstream C: M3.3 — Chunking Framework

**No existing impl plan — tasks defined here.**

| Task | Description | Deps | Files |
|------|-------------|------|-------|
| C1 | ChunkingStrategy trait (done as F3) | — | `chunking_strategy.rs` |
| C2 | FormatRegistry | F3 | `chunking_pipeline.rs` |
| C3 | FallbackChunker | F3, F2 | `chunking_fallback.rs` |
| C4 | ChunkingPipeline orchestrator | C2, C3, F2 | `chunking_pipeline.rs` |
| C5 | MarkdownChunker | F3 | `chunking_markdown.rs` |
| C6 | ConfigChunker | F3 | `chunking_config_fmt.rs` |
| C7 | PdfChunker | F3 | `chunking_pdf.rs` |
| C8 | D69 IngestionAdapter trait revision | F3 | `traits.rs` |
| C9 | AstChunker (corvia-adapter-git) | F3, C8 | `adapter-git/src/ast_chunker.rs` |
| C10 | Pipeline integration tests | C4, C5, C6, C7 | `tests/` |

#### Task C2: FormatRegistry

**Files:**
- Create: `crates/corvia-kernel/src/chunking_pipeline.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Write FormatRegistry + ProcessingReport types**

Create `crates/corvia-kernel/src/chunking_pipeline.rs`:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use corvia_common::config::ChunkingConfig;
use corvia_common::errors::Result;

use crate::chunking_strategy::{
    ChunkingStrategy, ProcessedChunk, ProcessingInfo, RawChunk, SourceFile, SourceMetadata,
};
use crate::token_estimator::{CharDivFourEstimator, TokenEstimator};

/// Routes file extensions to chunking strategies.
///
/// Resolution priority: adapter override > adapter register > kernel default > fallback.
pub struct FormatRegistry {
    defaults: HashMap<String, Arc<dyn ChunkingStrategy>>,
    overrides: HashMap<String, Arc<dyn ChunkingStrategy>>,
    fallback: Arc<dyn ChunkingStrategy>,
}

impl FormatRegistry {
    pub fn new(fallback: Arc<dyn ChunkingStrategy>) -> Self {
        Self {
            defaults: HashMap::new(),
            overrides: HashMap::new(),
            fallback,
        }
    }

    /// Register a kernel default strategy for an extension.
    pub fn register_default(&mut self, ext: &str, strategy: Arc<dyn ChunkingStrategy>) {
        self.defaults.insert(ext.to_lowercase(), strategy);
    }

    /// Adapter registers a strategy for formats it handles.
    pub fn register(&mut self, ext: &str, strategy: Arc<dyn ChunkingStrategy>) {
        self.defaults.insert(ext.to_lowercase(), strategy);
    }

    /// Adapter overrides a kernel default strategy.
    pub fn override_default(&mut self, ext: &str, strategy: Arc<dyn ChunkingStrategy>) {
        self.overrides.insert(ext.to_lowercase(), strategy);
    }

    /// Resolve the strategy for a file extension.
    /// Priority: override > default > fallback.
    pub fn resolve(&self, ext: &str) -> Arc<dyn ChunkingStrategy> {
        let ext_lower = ext.to_lowercase();
        if let Some(s) = self.overrides.get(&ext_lower) {
            return s.clone();
        }
        if let Some(s) = self.defaults.get(&ext_lower) {
            return s.clone();
        }
        self.fallback.clone()
    }
}

/// Per-strategy statistics for the processing report.
#[derive(Debug, Clone, Default)]
pub struct StrategyStats {
    pub files: usize,
    pub chunks: usize,
    pub splits: usize,
    pub merges: usize,
}

/// Token distribution statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    pub min: usize,
    pub max: usize,
    pub mean: usize,
    pub median: usize,
}

/// Report returned from batch processing.
#[derive(Debug, Clone, Default)]
pub struct ProcessingReport {
    pub files_processed: usize,
    pub total_chunks: usize,
    pub chunks_split: usize,
    pub chunks_merged: usize,
    pub token_stats: TokenStats,
    pub per_strategy: HashMap<String, StrategyStats>,
}

// ChunkingPipeline is defined in Task C4.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking_strategy::ChunkMetadata;

    /// Minimal test strategy
    struct MockStrategy {
        name: &'static str,
        extensions: Vec<&'static str>,
    }

    impl ChunkingStrategy for MockStrategy {
        fn name(&self) -> &str { self.name }
        fn supported_extensions(&self) -> &[&str] { &self.extensions }
        fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<Vec<RawChunk>> {
            Ok(vec![RawChunk {
                content: source.to_string(),
                chunk_type: "file".into(),
                start_line: 1,
                end_line: source.lines().count() as u32,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    ..Default::default()
                },
            }])
        }
    }

    #[test]
    fn test_registry_resolve_default() {
        let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
        let mut registry = FormatRegistry::new(fallback);
        let md_strategy = Arc::new(MockStrategy { name: "markdown", extensions: vec!["md"] });
        registry.register_default("md", md_strategy);

        assert_eq!(registry.resolve("md").name(), "markdown");
        assert_eq!(registry.resolve("unknown").name(), "fallback");
    }

    #[test]
    fn test_registry_override_takes_priority() {
        let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
        let mut registry = FormatRegistry::new(fallback);
        let default_rs = Arc::new(MockStrategy { name: "default-rs", extensions: vec!["rs"] });
        let override_rs = Arc::new(MockStrategy { name: "ast-rs", extensions: vec!["rs"] });

        registry.register_default("rs", default_rs);
        registry.override_default("rs", override_rs);

        assert_eq!(registry.resolve("rs").name(), "ast-rs");
    }

    #[test]
    fn test_registry_case_insensitive() {
        let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
        let mut registry = FormatRegistry::new(fallback);
        let md = Arc::new(MockStrategy { name: "markdown", extensions: vec!["md"] });
        registry.register_default("MD", md);

        assert_eq!(registry.resolve("md").name(), "markdown");
        assert_eq!(registry.resolve("Md").name(), "markdown");
    }

    #[test]
    fn test_registry_fallback() {
        let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
        let registry = FormatRegistry::new(fallback);
        assert_eq!(registry.resolve("xyz").name(), "fallback");
    }
}
```

**Step 2: Add module to lib.rs**

```rust
pub mod chunking_pipeline;
```

**Step 3: Run tests**

Run: `cargo test -p corvia-kernel chunking_pipeline`
Expected: PASS (4 tests)

**Step 4: Commit**

```bash
git add crates/corvia-kernel/src/chunking_pipeline.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(kernel): add FormatRegistry for chunking strategy routing (D66)"
```

---

#### Task C3: FallbackChunker

**Files:**
- Create: `crates/corvia-kernel/src/chunking_fallback.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

**Step 1: Implement FallbackChunker**

Create `crates/corvia-kernel/src/chunking_fallback.rs`:

```rust
use corvia_common::errors::Result;

use crate::chunking_strategy::{ChunkMetadata, ChunkingStrategy, RawChunk, SourceMetadata};
use crate::token_estimator::TokenEstimator;
use std::sync::Arc;

/// Recursive text splitting for unrecognized formats.
///
/// Separator cascade: "\n\n" → "\n" → " " → character-level.
/// Produces chunks up to max_tokens.
pub struct FallbackChunker {
    estimator: Arc<dyn TokenEstimator>,
    max_tokens: usize,
}

impl FallbackChunker {
    pub fn new(estimator: Arc<dyn TokenEstimator>, max_tokens: usize) -> Self {
        Self { estimator, max_tokens }
    }

    fn recursive_split(&self, text: &str, separators: &[&str], base_line: u32) -> Vec<RawChunk> {
        // Base case: fits in budget
        if self.estimator.estimate(text) <= self.max_tokens {
            return vec![RawChunk {
                content: text.to_string(),
                chunk_type: "text_block".into(),
                start_line: base_line,
                end_line: base_line + text.matches('\n').count() as u32,
                metadata: ChunkMetadata::default(),
            }];
        }

        // Try splitting by current separator
        if let Some((&sep, rest_seps)) = separators.split_first() {
            let parts: Vec<&str> = text.split(sep).collect();
            if parts.len() > 1 {
                let mut chunks = Vec::new();
                let mut current = String::new();
                let mut current_start = base_line;

                for part in parts {
                    let candidate = if current.is_empty() {
                        part.to_string()
                    } else {
                        format!("{}{}{}", current, sep, part)
                    };

                    if self.estimator.estimate(&candidate) <= self.max_tokens {
                        current = candidate;
                    } else {
                        if !current.is_empty() {
                            let line_count = current.matches('\n').count() as u32;
                            chunks.push(RawChunk {
                                content: current.clone(),
                                chunk_type: "text_block".into(),
                                start_line: current_start,
                                end_line: current_start + line_count,
                                metadata: ChunkMetadata::default(),
                            });
                            current_start = current_start + line_count + 1;
                        }
                        // If the part itself is too big, recurse with finer separators
                        if self.estimator.estimate(part) > self.max_tokens {
                            let sub = self.recursive_split(part, rest_seps, current_start);
                            if let Some(last) = sub.last() {
                                current_start = last.end_line + 1;
                            }
                            chunks.extend(sub);
                            current = String::new();
                        } else {
                            current = part.to_string();
                        }
                    }
                }

                if !current.is_empty() {
                    let line_count = current.matches('\n').count() as u32;
                    chunks.push(RawChunk {
                        content: current,
                        chunk_type: "text_block".into(),
                        start_line: current_start,
                        end_line: current_start + line_count,
                        metadata: ChunkMetadata::default(),
                    });
                }

                return chunks;
            }
        }

        // Final fallback: character-level split
        let mut chunks = Vec::new();
        let target_chars = self.max_tokens * 4; // reverse the chars/4 heuristic
        let mut pos = 0;
        let content = text.as_bytes();
        while pos < content.len() {
            let end = (pos + target_chars).min(content.len());
            let chunk_str = String::from_utf8_lossy(&content[pos..end]).to_string();
            let lines_before = text[..pos].matches('\n').count() as u32;
            let lines_in_chunk = chunk_str.matches('\n').count() as u32;
            chunks.push(RawChunk {
                content: chunk_str,
                chunk_type: "text_block".into(),
                start_line: base_line + lines_before,
                end_line: base_line + lines_before + lines_in_chunk,
                metadata: ChunkMetadata::default(),
            });
            pos = end;
        }
        chunks
    }
}

impl ChunkingStrategy for FallbackChunker {
    fn name(&self) -> &str { "fallback" }

    fn supported_extensions(&self) -> &[&str] { &[] } // matches everything via fallback

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<Vec<RawChunk>> {
        let separators = ["\n\n", "\n", " "];
        let mut chunks = self.recursive_split(source, &separators, 1);

        // Set source_file metadata
        for chunk in &mut chunks {
            chunk.metadata.source_file = meta.file_path.clone();
            chunk.metadata.language = meta.language.clone();
        }

        // Filter out empty chunks
        chunks.retain(|c| !c.content.trim().is_empty());

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_estimator::CharDivFourEstimator;

    fn make_fallback(max_tokens: usize) -> FallbackChunker {
        FallbackChunker::new(Arc::new(CharDivFourEstimator), max_tokens)
    }

    fn test_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "test.txt".into(),
            extension: "txt".into(),
            language: None,
            scope_id: "test".into(),
            source_version: "abc".into(),
        }
    }

    #[test]
    fn test_small_file_single_chunk() {
        let chunker = make_fallback(100);
        let chunks = chunker.chunk("Hello, world!", &test_meta()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello, world!");
    }

    #[test]
    fn test_paragraph_splitting() {
        let chunker = make_fallback(20); // ~80 chars max
        let text = "First paragraph with enough text to matter.\n\nSecond paragraph also has content.\n\nThird paragraph here.";
        let chunks = chunker.chunk(text, &test_meta()).unwrap();
        assert!(chunks.len() >= 2, "Should split on paragraph boundaries");
    }

    #[test]
    fn test_line_splitting_fallback() {
        let chunker = make_fallback(10); // ~40 chars max
        let text = "Short line one\nShort line two\nShort line three\nShort line four\nShort line five\nShort line six";
        let chunks = chunker.chunk(text, &test_meta()).unwrap();
        assert!(chunks.len() >= 2, "Should split on line boundaries");
    }

    #[test]
    fn test_empty_input() {
        let chunker = make_fallback(100);
        let chunks = chunker.chunk("", &test_meta()).unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_metadata_propagated() {
        let chunker = make_fallback(100);
        let chunks = chunker.chunk("content", &test_meta()).unwrap();
        assert_eq!(chunks[0].metadata.source_file, "test.txt");
    }
}
```

**Step 2: Add module, run tests, commit**

Run: `cargo test -p corvia-kernel chunking_fallback`
Expected: PASS (5 tests)

```bash
git add crates/corvia-kernel/src/chunking_fallback.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(kernel): add FallbackChunker — recursive text splitting (D68)"
```

---

#### Task C4: ChunkingPipeline Orchestrator

**Files:**
- Modify: `crates/corvia-kernel/src/chunking_pipeline.rs`

**Step 1: Add ChunkingPipeline to chunking_pipeline.rs**

Add to the existing `chunking_pipeline.rs` (which already has FormatRegistry):

```rust
/// 6-step chunking orchestrator (D65).
///
/// 1. Resolve strategy via FormatRegistry
/// 2. strategy.chunk() → raw chunks
/// 3. strategy.merge_small() or kernel default merge
/// 4. Token budget enforcement
/// 5. strategy.split_oversized() for violations
/// 6. strategy.overlap_context() between final chunks
pub struct ChunkingPipeline {
    registry: FormatRegistry,
    estimator: Arc<dyn TokenEstimator>,
    config: ChunkingConfig,
}

impl ChunkingPipeline {
    pub fn new(
        registry: FormatRegistry,
        estimator: Arc<dyn TokenEstimator>,
        config: ChunkingConfig,
    ) -> Self {
        Self { registry, estimator, config }
    }

    pub fn process(
        &self,
        source: &str,
        meta: &SourceMetadata,
    ) -> Result<Vec<ProcessedChunk>> {
        // Step 1: Resolve strategy
        let strategy = self.registry.resolve(&meta.extension);
        let strategy_name = strategy.name().to_string();

        // Step 2: Domain chunking
        let raw_chunks = strategy.chunk(source, meta)?;

        // Step 3: Merge small chunks
        let merged = match strategy.merge_small(&raw_chunks, self.config.max_tokens) {
            Some(custom_merged) => custom_merged,
            None => self.default_merge(&raw_chunks),
        };

        // Step 4 + 5: Enforce budget, split oversized
        let mut budget_ok = Vec::new();
        for chunk in &merged {
            let tokens = self.estimator.estimate(&chunk.content);
            if tokens > self.config.max_tokens {
                let splits = strategy.split_oversized(chunk, self.config.max_tokens)?;
                // Recurse: split results might still be oversized
                for split in splits {
                    self.recursive_split_until_fit(&strategy, &split, &mut budget_ok)?;
                }
            } else {
                budget_ok.push(chunk.clone());
            }
        }

        // Step 6: Add overlap context
        let mut final_chunks = Vec::new();
        for i in 0..budget_ok.len() {
            let overlap_prefix = if i > 0 {
                strategy.overlap_context(&budget_ok[i - 1], &budget_ok[i])
            } else {
                None
            };

            let original = &budget_ok[i];
            let content_with_overlap = match &overlap_prefix {
                Some(prefix) => format!("{}\n{}", prefix, original.content),
                None => original.content.clone(),
            };

            let overlap_tokens = overlap_prefix
                .as_ref()
                .map(|p| self.estimator.estimate(p))
                .unwrap_or(0);

            final_chunks.push(ProcessedChunk {
                content: content_with_overlap,
                original_content: original.content.clone(),
                chunk_type: original.chunk_type.clone(),
                start_line: original.start_line,
                end_line: original.end_line,
                metadata: original.metadata.clone(),
                token_estimate: self.estimator.estimate(&original.content) + overlap_tokens,
                processing: ProcessingInfo {
                    strategy_name: strategy_name.clone(),
                    was_split: original.metadata.parent_chunk_id.is_some(),
                    was_merged: false, // TODO: track merge flag
                    overlap_tokens,
                },
            });
        }

        Ok(final_chunks)
    }

    pub fn process_batch(
        &self,
        files: &[SourceFile],
    ) -> Result<(Vec<ProcessedChunk>, ProcessingReport)> {
        let mut all_chunks = Vec::new();
        let mut report = ProcessingReport::default();

        for file in files {
            let chunks = self.process(&file.content, &file.metadata)?;
            let strategy_name = if let Some(first) = chunks.first() {
                first.processing.strategy_name.clone()
            } else {
                "unknown".into()
            };

            let stats = report.per_strategy.entry(strategy_name).or_default();
            stats.files += 1;
            stats.chunks += chunks.len();
            stats.splits += chunks.iter().filter(|c| c.processing.was_split).count();

            report.files_processed += 1;
            report.total_chunks += chunks.len();
            report.chunks_split += chunks.iter().filter(|c| c.processing.was_split).count();

            all_chunks.extend(chunks);
        }

        // Compute token stats
        if !all_chunks.is_empty() {
            let mut token_counts: Vec<usize> = all_chunks.iter()
                .map(|c| c.token_estimate)
                .collect();
            token_counts.sort();
            report.token_stats.min = token_counts[0];
            report.token_stats.max = *token_counts.last().unwrap();
            report.token_stats.mean = token_counts.iter().sum::<usize>() / token_counts.len();
            report.token_stats.median = token_counts[token_counts.len() / 2];
        }

        Ok((all_chunks, report))
    }

    /// Kernel default merge: combine adjacent chunks where sum fits budget.
    fn default_merge(&self, chunks: &[RawChunk]) -> Vec<RawChunk> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let mut merged = Vec::new();
        let mut current = chunks[0].clone();

        for next in &chunks[1..] {
            let combined_tokens = self.estimator.estimate(&current.content)
                + self.estimator.estimate(&next.content);

            // Only merge if same merge_group (or both None) and fits budget
            let same_group = current.metadata.merge_group == next.metadata.merge_group;
            let below_min = self.estimator.estimate(&current.content) < self.config.min_tokens;

            if same_group && below_min && combined_tokens <= self.config.max_tokens {
                current.content = format!("{}\n{}", current.content, next.content);
                current.end_line = next.end_line;
            } else {
                merged.push(current);
                current = next.clone();
            }
        }
        merged.push(current);
        merged
    }

    fn recursive_split_until_fit(
        &self,
        strategy: &Arc<dyn ChunkingStrategy>,
        chunk: &RawChunk,
        output: &mut Vec<RawChunk>,
    ) -> Result<()> {
        let tokens = self.estimator.estimate(&chunk.content);
        if tokens <= self.config.max_tokens {
            output.push(chunk.clone());
            return Ok(());
        }
        let splits = strategy.split_oversized(chunk, self.config.max_tokens)?;
        if splits.len() <= 1 {
            // Can't split further — accept as-is
            output.push(chunk.clone());
            return Ok(());
        }
        for split in &splits {
            self.recursive_split_until_fit(strategy, split, output)?;
        }
        Ok(())
    }
}
```

**Step 2: Add pipeline tests**

```rust
#[test]
fn test_pipeline_single_small_file() {
    let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
    let registry = FormatRegistry::new(fallback);
    let pipeline = ChunkingPipeline::new(
        registry,
        Arc::new(CharDivFourEstimator),
        ChunkingConfig::default(),
    );
    let meta = SourceMetadata {
        file_path: "test.txt".into(),
        extension: "txt".into(),
        language: None,
        scope_id: "test".into(),
        source_version: "abc".into(),
    };
    let chunks = pipeline.process("Hello, world!", &meta).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].original_content, "Hello, world!");
    assert_eq!(chunks[0].processing.strategy_name, "fallback");
}

#[test]
fn test_pipeline_batch_processing() {
    let fallback = Arc::new(MockStrategy { name: "fallback", extensions: vec![] });
    let registry = FormatRegistry::new(fallback);
    let pipeline = ChunkingPipeline::new(
        registry,
        Arc::new(CharDivFourEstimator),
        ChunkingConfig::default(),
    );
    let files = vec![
        SourceFile {
            content: "File one".into(),
            metadata: SourceMetadata {
                file_path: "a.txt".into(),
                extension: "txt".into(),
                language: None,
                scope_id: "test".into(),
                source_version: "abc".into(),
            },
        },
        SourceFile {
            content: "File two".into(),
            metadata: SourceMetadata {
                file_path: "b.txt".into(),
                extension: "txt".into(),
                language: None,
                scope_id: "test".into(),
                source_version: "abc".into(),
            },
        },
    ];
    let (chunks, report) = pipeline.process_batch(&files).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(report.files_processed, 2);
    assert_eq!(report.total_chunks, 2);
}
```

**Step 3: Run tests, commit**

Run: `cargo test -p corvia-kernel chunking_pipeline`
Expected: PASS (6 tests: 4 FormatRegistry + 2 pipeline)

```bash
git add crates/corvia-kernel/src/chunking_pipeline.rs
git commit -m "feat(kernel): add ChunkingPipeline 6-step orchestrator (D65)"
```

---

#### Task C5: MarkdownChunker

**Files:**
- Create: `crates/corvia-kernel/src/chunking_markdown.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

Heading-based splitting: `#`, `##`, `###`. Each section = heading + body.
`split_oversized()`: paragraph splitting (`\n\n`), sentence fallback.
`overlap_context()`: carry heading hierarchy.

Implementation details: see M3.3 design doc "MarkdownChunker" section.

---

#### Task C6: ConfigChunker

**Files:**
- Create: `crates/corvia-kernel/src/chunking_config_fmt.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

Top-level key/section chunking for `.toml`, `.yaml`, `.yml`, `.json`.
`split_oversized()`: recurse into nested keys.

---

#### Task C7: PdfChunker

**Files:**
- Create: `crates/corvia-kernel/src/chunking_pdf.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`
- Modify: `crates/corvia-kernel/Cargo.toml` (add `pdf-extract` or `lopdf` dep, feature-gated)

Page-based extraction → paragraph splitting. Feature-gated if dependency is heavy.

---

#### Task C8: D69 IngestionAdapter Trait Revision

**Files:**
- Modify: `crates/corvia-kernel/src/traits.rs`
- Modify: `crates/corvia-kernel/src/chunking_pipeline.rs`

**Step 1: Replace IngestionAdapter trait**

In `traits.rs`, replace the existing `IngestionAdapter`:

```rust
/// Ingestion adapter for domain-specific sources (D11/D69).
///
/// D69 clean break: adapters return raw source files and register
/// chunking strategies. The kernel's ChunkingPipeline handles
/// chunking, budget enforcement, and embedding.
#[async_trait]
pub trait IngestionAdapter: Send + Sync {
    /// Domain name (e.g., "git", "general").
    fn domain(&self) -> &str;

    /// Register chunking strategies for formats this adapter handles.
    fn register_chunking(&self, registry: &mut crate::chunking_pipeline::FormatRegistry);

    /// Return raw source files for the kernel's ChunkingPipeline.
    async fn ingest_sources(&self, source_path: &str) -> Result<Vec<crate::chunking_strategy::SourceFile>>;
}
```

**Step 2: Compile check**

Run: `cargo check --workspace`

This will fail because `corvia-adapter-git` still implements the old trait.
That's expected — Task C9 updates the adapter.

**Step 3: Commit (kernel only)**

```bash
git add crates/corvia-kernel/src/traits.rs
git commit -m "feat(kernel): revise IngestionAdapter trait for D69 clean break"
```

---

#### Task C9: AstChunker (corvia-adapter-git)

**Files:**
- Create: `corvia-adapter-git/src/ast_chunker.rs`
- Modify: `corvia-adapter-git/src/lib.rs`
- Modify: `corvia-adapter-git/src/git.rs`

Wraps existing tree-sitter parsing as a `ChunkingStrategy` implementation.
Updates `GitAdapter` to implement the revised `IngestionAdapter` trait.
Registers `AstChunker` for `.rs`, `.py`, `.js`, `.jsx`, `.ts`, `.tsx`.

---

#### Task C10: Pipeline Integration Tests

**Files:**
- Create: `crates/corvia-kernel/tests/integration/chunking_test.rs` (or in-module)

Tests:
- Round-trip: known Markdown file → expected chunk count/content
- Round-trip: known TOML file → expected chunk count
- Oversized file → split and all chunks fit budget
- Merge pass: many tiny chunks merge into fewer
- FormatRegistry priority chain: adapter override > default > fallback
- ProcessingReport statistics correctness

---

## Phase 2: Integration Tasks

These tasks wire the three workstreams together.

### Task I1: Wire ask() Mode with GenerationEngine

**Prerequisite:** Handoff H2 (M3.1 Task A4 — GrpcChatEngine lands)

**Files:**
- Modify: `crates/corvia-kernel/src/rag_pipeline.rs`
- Modify: `crates/corvia-server/src/rest.rs`
- Modify: `crates/corvia-server/src/mcp.rs`

**Step 1: Add ask() to RagPipeline**

The `ask()` method calls `generator.generate()` with the augmented context.
Error if `generator` is None.

**Step 2: Add REST endpoint POST /v1/ask**

**Step 3: Add MCP tool corvia_ask**

**Step 4: E2E test**

Test with mock GenerationEngine, then optional e2e with Ollama (skip if unavailable).

---

### Task I2: Wire Ingestion Through ChunkingPipeline

**Prerequisite:** Handoff H3 (M3.3 ChunkingPipeline + D69 adapter revision)

**Files:**
- Modify: `crates/corvia-cli/src/main.rs` (or wherever `ingest` command lives)
- Modify: `crates/corvia-server/src/rest.rs` (if ingestion endpoint exists)

**Step 1: Update ingestion flow**

Replace:
```
adapter.ingest() → Vec<KnowledgeEntry> → embed → store
```
With:
```
adapter.ingest_sources() → Vec<SourceFile>
  → pipeline.process_batch() → Vec<ProcessedChunk>
  → engine.embed_batch() → Vec<Vec<f32>>
  → convert to KnowledgeEntry → store.insert()
```

---

### Task I3: Update create_engine() for Corvia Provider

**Prerequisite:** M3.1 Tasks A3, A5 (GrpcInferenceEngine, InferenceProvisioner)

**Files:**
- Modify: `crates/corvia-kernel/src/lib.rs`

Add `InferenceProvider::Corvia` match arm in `create_engine()`:
```rust
InferenceProvider::Corvia => Box::new(
    engines::grpc_inference_engine::GrpcInferenceEngine::new(
        &config.embedding.url,
        config.embedding.dimensions,
    )
),
```

---

### Task I4: E2E Integration Tests

**Prerequisite:** All Phase 1 complete + I1-I3

**Files:**
- Create: `tests/integration/e2e_rag_test.rs` (or similar)

Tests:
- Full ingest → chunk → embed → store → retrieve → augment → generate flow
- Verify ChunkingPipeline produces chunks that embed and retrieve correctly
- Verify graph expansion in GraphExpandRetriever works with chunked entries
- Performance: ingest a real repo, verify chunk count and token budget compliance

---

### Task I5: Final Server Wiring + CLI Updates

**Files:**
- Modify: `crates/corvia-server/src/rest.rs` (AppState gains `rag` field)
- Modify: `crates/corvia-cli/src/main.rs` (add `--provider corvia` flag)
- Modify: `crates/corvia-server/src/lib.rs` (startup initializes RagPipeline)

Wire everything together:
- Server startup creates `RagPipeline` via `create_rag_pipeline()` factory
- AppState includes `rag: Arc<RagPipeline>`
- CLI recognizes `--provider corvia` for inference

---

## Phase 2 Completion Checkpoint

Run: `cargo test --workspace`
Expected: All tests pass (203 existing + ~60-80 new from all three milestones).

Run: `make test-full` (if SurrealDB available)
Expected: All tests pass.

---

## Parallel Execution Summary

```
Timeline    │ Agent 1 (M3.1)        │ Agent 2 (M3.2)        │ Agent 3 (M3.3)
────────────┼───────────────────────┼───────────────────────┼──────────────────
Phase 0     │ ← F1 (GenEngine) + F6 (Corvia variant) ──────────────────────→ │
(shared)    │ ← F2 (TokenEstimator) + F4 (RAG types) + F5 (configs) ───────→ │
            │ ← F3 (ChunkingStrategy trait) ────────────────────────────────→ │
────────────┼───────────────────────┼───────────────────────┼──────────────────
Phase 1     │ A1: corvia-proto      │ B2: VectorRetriever   │ C2: FormatReg
(parallel)  │ A2: MergeWorker refac │ B3: GraphExpandRetr   │ C3: FallbackChkr
            │ A3: GrpcInferEngine   │ B4: StructuredAugmnt  │ C4: Pipeline
            │ A4: GrpcChatEngine    │ B5: RagPipeline ctx   │ C5: MarkdownChkr
            │     ──── H1 ────→     │ B6: search routing fix│ C6: ConfigChkr
            │ A5: InferProvisioner  │ B7: REST /v1/context  │ C7: PdfChunker
            │ A6: corvia-inference  │ B8: MCP corvia_context│ C8: D69 trait
            │ A7: EmbeddingService  │ B9: tests             │ C9: AstChunker
            │ A8: ChatService stub  │                       │ C10: tests
            │ A9: CLI wiring        │                       │
            │ A10: gRPC tests       │                       │
            │     ──── H2 ────→     │                       │     ── H3 ──→
            │ A11: GrpcVllmEngine   │                       │
            │ A12: Candle full      │                       │
────────────┼───────────────────────┼───────────────────────┼──────────────────
Phase 2     │                   I1: Wire ask() mode (needs H2)               │
(integrate) │                   I2: Wire ingestion (needs H3)                │
            │                   I3: create_engine() Corvia                   │
            │                   I4: E2E integration tests                    │
            │                   I5: Final server + CLI wiring                │
```

## Handoff Contracts

| Handoff | Producer | Consumer | Contract |
|---------|----------|----------|----------|
| **H1** | M3.1 (A4) | M3.2 | `GrpcChatEngine` implements `GenerationEngine` trait from F1. M3.2 can wire `ask()`. |
| **H2** | M3.1 (A4+A7) | Phase 2 (I1) | GrpcChatEngine + EmbeddingService are functional. Real `ask()` endpoint works e2e. |
| **H3** | M3.3 (C4+C8+C9) | Phase 2 (I2) | `ChunkingPipeline` + revised `IngestionAdapter` + `AstChunker` are functional. Ingestion can route through pipeline. |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| M3.1 candle/ONNX compilation issues | A12 (candle) is last task — stub first, real impl later. M3.2 works with Ollama fallback. |
| M3.3 PdfChunker heavy deps | Feature-gate `pdf-extract`. FallbackChunker handles PDFs as text in the meantime. |
| TokenEstimator divergence | Single implementation in `token_estimator.rs`. Both M3.2 and M3.3 import from same module. |
| Adapter-git compilation breaks from D69 | C8 (trait revision) and C9 (adapter update) are paired. Commit together or use feature flag. |
| Phase 2 integration surprises | Each workstream has its own test suite. Integration tests (I4) catch cross-workstream issues. |
