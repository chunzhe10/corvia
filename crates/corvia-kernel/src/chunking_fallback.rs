//! Fallback chunking strategy using a recursive separator cascade (D68).
//!
//! [`FallbackChunker`] splits source text into token-budget-sized chunks by
//! trying separators in order of decreasing granularity:
//!
//! 1. `\n\n` — paragraph boundaries
//! 2. `\n` — line boundaries
//! 3. ` ` — word boundaries
//! 4. Return as-is — the pipeline's `split_oversized` handles the rest
//!
//! This strategy has no `supported_extensions` — it matches everything via the
//! `FormatRegistry` fallback mechanism, never by extension.

use corvia_common::errors::Result;

use crate::chunking_strategy::{ChunkMetadata, ChunkResult, ChunkingStrategy, RawChunk, SourceMetadata};

/// The separator cascade, tried in order from coarsest to finest.
const SEPARATORS: &[&str] = &["\n\n", "\n", " "];

/// Recursive separator cascade chunker.
///
/// Constructor takes `max_tokens` — the token budget per chunk.
/// Token estimation uses the chars/4 heuristic (same as
/// [`CharDivFourEstimator`](crate::token_estimator::CharDivFourEstimator)).
pub struct FallbackChunker {
    max_tokens: usize,
}

impl FallbackChunker {
    /// Create a new `FallbackChunker` with the given per-chunk token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Estimate tokens using chars/4 heuristic.
    fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            (text.len() / 4).max(1)
        }
    }

    /// Recursively split `text` using the separator cascade.
    ///
    /// `sep_index` is the current position in [`SEPARATORS`]. Each piece that
    /// fits within the budget is collected; oversized pieces recurse with the
    /// next separator.
    fn recursive_split(text: &str, max_tokens: usize, sep_index: usize) -> Vec<String> {
        // Fits in budget — return as a single piece.
        if Self::estimate_tokens(text) <= max_tokens {
            return vec![text.to_owned()];
        }

        // No separators left — return as-is (the pipeline's split_oversized handles it).
        let Some(&sep) = SEPARATORS.get(sep_index) else {
            return vec![text.to_owned()];
        };

        let pieces: Vec<&str> = text.split(sep).collect();

        // Separator not present — skip to the next finer separator.
        if pieces.len() <= 1 {
            return Self::recursive_split(text, max_tokens, sep_index + 1);
        }

        // Greedily accumulate consecutive pieces into chunks that fit the budget.
        let mut result: Vec<String> = Vec::new();
        let mut accumulator = String::new();

        for (i, piece) in pieces.iter().enumerate() {
            // Build the candidate: accumulator + sep + piece (or just piece if empty).
            let candidate = if accumulator.is_empty() {
                piece.to_string()
            } else {
                format!("{}{}{}", accumulator, sep, piece)
            };

            if Self::estimate_tokens(&candidate) <= max_tokens {
                accumulator = candidate;
            } else {
                // Flush whatever we accumulated so far.
                if !accumulator.is_empty() {
                    result.push(accumulator);
                    accumulator = String::new();
                }

                // Try this piece alone.
                if Self::estimate_tokens(piece) <= max_tokens {
                    accumulator = piece.to_string();
                } else {
                    // Piece itself is too large — recurse with the next separator.
                    let sub = Self::recursive_split(piece, max_tokens, sep_index + 1);
                    result.extend(sub);
                }
            }

            // If this is the last piece, flush the accumulator.
            if i == pieces.len() - 1 && !accumulator.is_empty() {
                result.push(accumulator.clone());
                accumulator.clear();
            }
        }

        // Safety flush (should not be needed, but just in case).
        if !accumulator.is_empty() {
            result.push(accumulator);
        }

        result
    }

    /// Count the number of `\n` characters in a string.
    fn newline_count(s: &str) -> u32 {
        s.bytes().filter(|&b| b == b'\n').count() as u32
    }
}

impl ChunkingStrategy for FallbackChunker {
    fn name(&self) -> &str {
        "fallback"
    }

    fn supported_extensions(&self) -> &[&str] {
        &[]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        if source.is_empty() {
            return Ok(ChunkResult::default());
        }

        let text_chunks = Self::recursive_split(source, self.max_tokens, 0);

        let mut chunks = Vec::with_capacity(text_chunks.len());
        let mut current_line: u32 = 1;

        for content in text_chunks {
            let line_count = Self::newline_count(&content);
            let start_line = current_line;
            let end_line = current_line + line_count;

            chunks.push(RawChunk {
                content,
                chunk_type: "text".into(),
                start_line,
                end_line,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    language: meta.language.clone(),
                    ..Default::default()
                },
            });

            current_line = end_line + 1;
        }

        // Fix line tracking: recompute from the original source for accuracy.
        let mut search_offset: usize = 0;
        let mut line_at_offset: u32 = 1;

        for chunk in &mut chunks {
            if let Some(pos) = source[search_offset..].find(&chunk.content) {
                let abs_pos = search_offset + pos;
                let skipped = &source[search_offset..abs_pos];
                line_at_offset += Self::newline_count(skipped);

                chunk.start_line = line_at_offset;
                chunk.end_line = line_at_offset + Self::newline_count(&chunk.content);

                search_offset = abs_pos + chunk.content.len();
                line_at_offset = chunk.end_line;
            }
        }

        Ok(ChunkResult { chunks, relations: vec![] })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "docs/notes.txt".into(),
            extension: "txt".into(),
            language: None,
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
        }
    }

    #[test]
    fn test_small_file_single_chunk() {
        let chunker = FallbackChunker::new(1000);
        let source = "Hello, world! This is a small file.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, source);
        assert_eq!(chunks[0].chunk_type, "text");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
    }

    #[test]
    fn test_paragraph_splitting() {
        // Each paragraph is ~25 chars -> ~6 tokens. Budget of 10 tokens forces
        // the chunker to split at paragraph boundaries.
        let source = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunker = FallbackChunker::new(10);
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        // All content should be chunk_type "text".
        for c in &chunks {
            assert_eq!(c.chunk_type, "text");
        }

        // Reassembled content (joining with \n\n) should reconstruct the source.
        let reassembled: String = chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        assert_eq!(reassembled, source);
    }

    #[test]
    fn test_line_splitting_fallback() {
        // No paragraph breaks (\n\n), only line breaks (\n). Should fall back
        // to line splitting.
        let source =
            "Line one content.\nLine two content.\nLine three content.\nLine four content.";
        let chunker = FallbackChunker::new(10);
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks from line splitting, got {}",
            chunks.len()
        );

        for c in &chunks {
            assert_eq!(c.chunk_type, "text");
        }
    }

    #[test]
    fn test_empty_source() {
        let chunker = FallbackChunker::new(100);
        let chunks = chunker.chunk("", &test_meta()).unwrap().chunks;
        assert!(chunks.is_empty(), "expected empty vec for empty source");
    }

    #[test]
    fn test_metadata_populated() {
        let chunker = FallbackChunker::new(1000);
        let source = "Some content here.";
        let meta = test_meta();
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.source_file, "docs/notes.txt");
        assert_eq!(chunks[0].metadata.language, None);
        assert!(chunks[0].metadata.parent_chunk_id.is_none());
    }

    #[test]
    fn test_name_and_extensions() {
        let chunker = FallbackChunker::new(100);
        assert_eq!(chunker.name(), "fallback");
        assert!(chunker.supported_extensions().is_empty());
    }
}
