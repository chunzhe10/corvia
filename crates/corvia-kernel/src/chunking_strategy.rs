//! Chunking strategy trait and core types for the Corvia ingestion pipeline (D65).
//!
//! The [`ChunkingStrategy`] trait uses a template-method pattern: adapters implement
//! domain-specific `chunk()` logic while the kernel's `ChunkingPipeline` (future)
//! enforces universal concerns (token budget, merging, splitting).
//!
//! # Default Behaviours
//!
//! - `split_oversized` — splits at nearest newline to midpoint, links via `parent_chunk_id`
//! - `merge_small` — returns `None` (opt-in per strategy)
//! - `overlap_context` — returns `None` (opt-in per strategy)

use corvia_common::errors::Result;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Metadata about a source file being chunked.
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub file_path: String,
    pub extension: String,
    pub language: Option<String>,
    pub scope_id: String,
    pub source_version: String,
}

/// A raw chunk produced by a [`ChunkingStrategy`] before pipeline processing.
#[derive(Debug, Clone)]
pub struct RawChunk {
    pub content: String,
    pub chunk_type: String,
    pub start_line: u32,
    pub end_line: u32,
    pub metadata: ChunkMetadata,
}

/// Per-chunk metadata carried through the pipeline.
#[derive(Debug, Clone, Default)]
pub struct ChunkMetadata {
    pub source_file: String,
    pub language: Option<String>,
    pub parent_chunk_id: Option<Uuid>,
    pub merge_group: Option<String>,
}

/// A fully-processed chunk ready for embedding and storage.
#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    pub content: String,
    pub original_content: String,
    pub chunk_type: String,
    pub start_line: u32,
    pub end_line: u32,
    pub metadata: ChunkMetadata,
    pub token_estimate: usize,
    pub processing: ProcessingInfo,
}

/// Records what the pipeline did to a chunk.
#[derive(Debug, Clone)]
pub struct ProcessingInfo {
    pub strategy_name: String,
    pub was_split: bool,
    pub was_merged: bool,
    pub overlap_tokens: usize,
}

/// A source file with its metadata, returned by [`IngestionAdapter::ingest_sources`] (D69).
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub content: String,
    pub metadata: SourceMetadata,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Strategy for splitting source text into chunks.
///
/// Implementors provide the domain-specific `chunk()` method. The kernel's
/// `ChunkingPipeline` calls the default `split_oversized`, `merge_small`, and
/// `overlap_context` hooks — override them to customise behaviour.
pub trait ChunkingStrategy: Send + Sync {
    /// Human-readable name for metrics attribution (D62).
    fn name(&self) -> &str;

    /// File extensions this strategy handles (e.g. `["rs", "go", "py"]`).
    fn supported_extensions(&self) -> &[&str];

    /// Split source text into [`RawChunk`]s.
    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<Vec<RawChunk>>;

    /// Split a chunk that exceeds `max_tokens` into smaller pieces.
    ///
    /// Default: binary-split at nearest newline to the midpoint, linking the
    /// children back to the original via `parent_chunk_id`.
    fn split_oversized(&self, chunk: &RawChunk, max_tokens: usize) -> Result<Vec<RawChunk>> {
        let _ = max_tokens; // used conceptually; split is by line count
        let lines: Vec<&str> = chunk.content.split('\n').collect();
        if lines.len() <= 1 {
            // Cannot split a single line further — return as-is.
            return Ok(vec![chunk.clone()]);
        }

        let mid = lines.len() / 2;

        // Find the nearest newline boundary to the midpoint.
        let first_half = lines[..mid].join("\n");
        let second_half = lines[mid..].join("\n");

        let parent_id = Uuid::now_v7();
        let mid_line = chunk.start_line + mid as u32;

        let mut meta_a = chunk.metadata.clone();
        meta_a.parent_chunk_id = Some(parent_id);

        let mut meta_b = chunk.metadata.clone();
        meta_b.parent_chunk_id = Some(parent_id);

        Ok(vec![
            RawChunk {
                content: first_half,
                chunk_type: chunk.chunk_type.clone(),
                start_line: chunk.start_line,
                end_line: mid_line.saturating_sub(1),
                metadata: meta_a,
            },
            RawChunk {
                content: second_half,
                chunk_type: chunk.chunk_type.clone(),
                start_line: mid_line,
                end_line: chunk.end_line,
                metadata: meta_b,
            },
        ])
    }

    /// Optionally merge adjacent small chunks.
    ///
    /// Default: `None` (no merging). Override to combine related fragments.
    fn merge_small(&self, _chunks: &[RawChunk], _max_tokens: usize) -> Option<Vec<RawChunk>> {
        None
    }

    /// Optionally produce overlap/context text between consecutive chunks.
    ///
    /// Default: `None` (no overlap). Override to prepend trailing lines of `prev`.
    fn overlap_context(&self, _prev: &RawChunk, _next: &RawChunk) -> Option<String> {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time object-safety assertion — if this compiles, the trait is
    /// object-safe and can be used as `dyn ChunkingStrategy`.
    #[test]
    fn assert_object_safe() {
        fn _check(_: &dyn ChunkingStrategy) {}
    }

    // -- Struct construction tests ------------------------------------------

    #[test]
    fn test_source_metadata_construction() {
        let meta = SourceMetadata {
            file_path: "src/main.rs".into(),
            extension: "rs".into(),
            language: Some("rust".into()),
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
        };
        assert_eq!(meta.file_path, "src/main.rs");
        assert_eq!(meta.extension, "rs");
        assert_eq!(meta.language.as_deref(), Some("rust"));
        assert_eq!(meta.scope_id, "org:proj:ws:git:v1");
        assert_eq!(meta.source_version, "abc123");
    }

    #[test]
    fn test_raw_chunk_construction() {
        let chunk = RawChunk {
            content: "fn main() {}".into(),
            chunk_type: "function".into(),
            start_line: 1,
            end_line: 3,
            metadata: ChunkMetadata {
                source_file: "src/main.rs".into(),
                language: Some("rust".into()),
                ..Default::default()
            },
        };
        assert_eq!(chunk.content, "fn main() {}");
        assert_eq!(chunk.chunk_type, "function");
        assert_eq!(chunk.start_line, 1);
        assert_eq!(chunk.end_line, 3);
        assert_eq!(chunk.metadata.source_file, "src/main.rs");
        assert!(chunk.metadata.parent_chunk_id.is_none());
        assert!(chunk.metadata.merge_group.is_none());
    }

    #[test]
    fn test_processed_chunk_construction() {
        let pc = ProcessedChunk {
            content: "processed content".into(),
            original_content: "original content".into(),
            chunk_type: "function".into(),
            start_line: 10,
            end_line: 20,
            metadata: ChunkMetadata::default(),
            token_estimate: 42,
            processing: ProcessingInfo {
                strategy_name: "test".into(),
                was_split: false,
                was_merged: true,
                overlap_tokens: 5,
            },
        };
        assert_eq!(pc.content, "processed content");
        assert_eq!(pc.original_content, "original content");
        assert_eq!(pc.token_estimate, 42);
        assert!(pc.processing.was_merged);
        assert!(!pc.processing.was_split);
        assert_eq!(pc.processing.overlap_tokens, 5);
        assert_eq!(pc.processing.strategy_name, "test");
    }

    // -- Default trait method tests -----------------------------------------

    /// Minimal strategy for exercising default methods.
    struct TestStrategy;

    impl ChunkingStrategy for TestStrategy {
        fn name(&self) -> &str {
            "test"
        }

        fn supported_extensions(&self) -> &[&str] {
            &["txt"]
        }

        fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<Vec<RawChunk>> {
            // Trivial: one chunk = entire source
            Ok(vec![RawChunk {
                content: source.to_string(),
                chunk_type: "file".into(),
                start_line: 1,
                end_line: source.lines().count() as u32,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    language: meta.language.clone(),
                    ..Default::default()
                },
            }])
        }
    }

    fn test_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "test.txt".into(),
            extension: "txt".into(),
            language: None,
            scope_id: "test:scope".into(),
            source_version: "v1".into(),
        }
    }

    #[test]
    fn test_default_split_oversized() {
        let strategy = TestStrategy;
        let source = "line one\nline two\nline three\nline four";
        let chunks = strategy.chunk(source, &test_meta()).unwrap();
        assert_eq!(chunks.len(), 1);

        let split = strategy.split_oversized(&chunks[0], 10).unwrap();
        assert_eq!(split.len(), 2, "expected 2 chunks after split");

        // Both halves should reference the same parent id.
        let parent_a = split[0].metadata.parent_chunk_id.unwrap();
        let parent_b = split[1].metadata.parent_chunk_id.unwrap();
        assert_eq!(parent_a, parent_b);

        // Content should be preserved across the split.
        let rejoined = format!("{}\n{}", split[0].content, split[1].content);
        assert_eq!(rejoined, source);
    }

    #[test]
    fn test_default_merge_returns_none() {
        let strategy = TestStrategy;
        let chunks = strategy.chunk("hello", &test_meta()).unwrap();
        assert!(strategy.merge_small(&chunks, 100).is_none());
    }

    #[test]
    fn test_default_overlap_returns_none() {
        let strategy = TestStrategy;
        let chunks = strategy.chunk("a\nb", &test_meta()).unwrap();
        let a = &chunks[0];
        assert!(strategy.overlap_context(a, a).is_none());
    }
}
