//! PDF chunking strategy — paragraph-level splitting of extracted text (D68).
//!
//! [`PdfChunker`] operates on pre-extracted text (not raw PDF bytes). It splits
//! on double-newline (`\n\n`) paragraph boundaries, producing one chunk per
//! non-empty paragraph with `chunk_type = "paragraph"`.
//!
//! The companion [`extract_pdf_text`] function handles the byte-level PDF
//! extraction using `pdf-extract` and is intended to be called by the ingestion
//! layer before passing the resulting text to `PdfChunker::chunk()`.

use corvia_common::errors::{CorviaError, Result};

use crate::chunking_strategy::{ChunkMetadata, ChunkResult, ChunkingStrategy, RawChunk, SourceMetadata};

/// Paragraph-level PDF text chunker.
///
/// Expects pre-extracted text from a PDF. Splits on `\n\n` boundaries and
/// produces one [`RawChunk`] per non-empty paragraph.
#[derive(Default)]
pub struct PdfChunker;

impl PdfChunker {
    /// Create a new `PdfChunker`.
    pub fn new() -> Self {
        Self
    }
}

impl ChunkingStrategy for PdfChunker {
    fn name(&self) -> &str {
        "pdf"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        if source.is_empty() {
            return Ok(ChunkResult::default());
        }

        // Split on double-newline paragraph boundaries.
        let paragraphs: Vec<&str> = source.split("\n\n").collect();

        let mut chunks = Vec::new();
        let mut current_line: u32 = 1;

        for para in &paragraphs {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                // Advance past the empty paragraph's lines.
                let line_count = para.lines().count().max(1) as u32;
                // +1 for the \n\n separator that was consumed by split.
                current_line += line_count + 1;
                continue;
            }

            let line_count = para.lines().count().max(1) as u32;
            let start_line = current_line;
            let end_line = current_line + line_count - 1;

            chunks.push(RawChunk {
                content: trimmed.to_string(),
                chunk_type: "paragraph".into(),
                start_line,
                end_line,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    language: meta.language.clone(),
                    ..Default::default()
                },
            });

            // Advance past this paragraph's lines + the \n\n separator (2 newlines).
            current_line = end_line + 2;
        }

        Ok(ChunkResult { chunks, relations: vec![] })
    }
}

/// Extract text from PDF bytes.
///
/// Uses `pdf_extract::extract_text_from_mem` to read the PDF content. Returns
/// the extracted text or a [`CorviaError::Ingestion`] for encrypted, scanned,
/// or malformed PDFs.
pub fn extract_pdf_text(bytes: &[u8]) -> Result<String> {
    pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| CorviaError::Ingestion(format!("PDF text extraction failed: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "docs/report.pdf".into(),
            extension: "pdf".into(),
            language: None,
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
            workstream: None,
            content_role: None,
            source_origin: None,
        }
    }

    #[test]
    fn test_paragraph_splitting() {
        let chunker = PdfChunker::new();
        let source = "First paragraph of the PDF.\n\nSecond paragraph with more detail.\n\nThird and final paragraph.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 3, "expected 3 chunks, got {}", chunks.len());
        assert_eq!(chunks[0].content, "First paragraph of the PDF.");
        assert_eq!(chunks[1].content, "Second paragraph with more detail.");
        assert_eq!(chunks[2].content, "Third and final paragraph.");
    }

    #[test]
    fn test_single_paragraph() {
        let chunker = PdfChunker::new();
        let source = "This is a single paragraph with no double-newline separators.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 1, "expected 1 chunk, got {}", chunks.len());
        assert_eq!(chunks[0].content, source);
    }

    #[test]
    fn test_empty_text() {
        let chunker = PdfChunker::new();
        let chunks = chunker.chunk("", &test_meta()).unwrap().chunks;
        assert!(chunks.is_empty(), "expected empty vec for empty source");
    }

    #[test]
    fn test_chunk_type() {
        let chunker = PdfChunker::new();
        let source = "Para one.\n\nPara two.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 2);
        for c in &chunks {
            assert_eq!(c.chunk_type, "paragraph", "expected chunk_type 'paragraph', got '{}'", c.chunk_type);
        }
    }

    #[test]
    fn test_name_and_extensions() {
        let chunker = PdfChunker::new();
        assert_eq!(chunker.name(), "pdf");
        assert_eq!(chunker.supported_extensions(), &["pdf"]);
    }
}
