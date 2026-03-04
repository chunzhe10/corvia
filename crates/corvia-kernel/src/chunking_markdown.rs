//! Markdown chunking strategy — heading-based section splitting (D68).
//!
//! [`MarkdownChunker`] splits Markdown source text into sections at heading
//! boundaries (`# `, `## `, `### `, etc.). Content before the first heading
//! becomes its own chunk with `chunk_type = "text"`, while headed sections
//! use `chunk_type = "heading_section"`.
//!
//! # Override Behaviours
//!
//! - [`overlap_context`](ChunkingStrategy::overlap_context) — extracts heading
//!   hierarchy lines from the previous chunk so the next chunk knows its
//!   position in the document structure.
//! - [`merge_small`](ChunkingStrategy::merge_small) — merges adjacent heading
//!   sections at the same heading depth when both are small.

use std::collections::HashSet;

use corvia_common::errors::Result;
use pulldown_cmark::{Event, Parser, Tag, TagEnd};

use crate::chunking_strategy::{ChunkMetadata, ChunkRelation, ChunkResult, ChunkingStrategy, RawChunk, SourceMetadata};

/// Heading-based Markdown section chunker.
pub struct MarkdownChunker;

impl MarkdownChunker {
    /// Create a new `MarkdownChunker`.
    pub fn new() -> Self {
        Self
    }

    /// Return the heading depth of a line, or `None` if it is not a heading.
    ///
    /// `# Title` returns `Some(1)`, `## Section` returns `Some(2)`, etc.
    fn heading_depth(line: &str) -> Option<usize> {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('#') {
            return None;
        }
        let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
        // A heading requires a space after the hashes (e.g. "# Title"),
        // or the line is exactly "#" characters (edge case we ignore).
        let rest = &trimmed[hashes..];
        if rest.is_empty() || rest.starts_with(' ') {
            Some(hashes)
        } else {
            None
        }
    }

    /// Check whether a code-span symbol is a meaningful CamelCase identifier
    /// worth creating a cross-reference relation for.
    fn is_meaningful_symbol(text: &str) -> bool {
        if text.len() < 3 {
            return false;
        }
        // Must start with an uppercase letter.
        let first = text.chars().next().unwrap();
        if !first.is_ascii_uppercase() {
            return false;
        }
        // Must consist only of valid identifier chars (alphanumeric + underscore).
        if !text.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return false;
        }
        // Skip common / standard-library types and traits.
        const SKIP: &[&str] = &[
            "String", "Vec", "HashMap", "HashSet", "BTreeMap", "BTreeSet",
            "Option", "Result", "Box", "Arc", "Mutex", "RwLock", "Rc", "Cell",
            "RefCell", "Pin", "Future", "Stream", "Iterator", "Display", "Debug",
            "Clone", "Copy", "Send", "Sync", "Default", "From", "Into",
            "TryFrom", "TryInto", "AsRef", "AsMut", "Deref", "DerefMut", "Drop",
            "Fn", "FnMut", "FnOnce", "Error", "Ord", "PartialOrd", "Eq",
            "PartialEq", "Hash", "Sized", "Unpin", "Serialize", "Deserialize",
            "None", "Some", "Ok", "Err", "Self", "TRUE", "FALSE",
        ];
        !SKIP.contains(&text)
    }

    /// Return `true` if the string looks like a relative file-path reference
    /// (not an HTTP URL) with a recognised source-code extension.
    fn is_file_path_reference(s: &str) -> bool {
        if s.starts_with("http://") || s.starts_with("https://") {
            return false;
        }
        const EXTENSIONS: &[&str] = &[
            ".rs", ".ts", ".tsx", ".js", ".jsx", ".py",
            ".toml", ".yaml", ".yml", ".json", ".md",
        ];
        EXTENSIONS.iter().any(|ext| s.ends_with(ext))
    }

    /// Find the chunk whose line range contains the given 1-indexed line number.
    fn find_chunk_for_line<'a>(chunks: &'a [RawChunk], line: u32) -> Option<&'a RawChunk> {
        chunks
            .iter()
            .find(|c| line >= c.start_line && line <= c.end_line)
            .or_else(|| chunks.last())
    }

    /// Parse `source` with pulldown-cmark and extract cross-reference relations
    /// (code-span symbols and file-path links) mapped back to their owning chunks.
    fn extract_references(
        source: &str,
        chunks: &[RawChunk],
        meta: &SourceMetadata,
    ) -> Vec<ChunkRelation> {
        if chunks.is_empty() {
            return vec![];
        }

        // Pre-compute a line-start byte-offset table so we can convert byte
        // offsets from pulldown-cmark into 1-indexed line numbers.
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        let byte_offset_to_line = |offset: usize| -> u32 {
            match line_starts.binary_search(&offset) {
                Ok(idx) => (idx + 1) as u32,
                Err(idx) => idx as u32,  // idx is insertion point; line = idx (1-indexed)
            }
        };

        let mut relations: Vec<ChunkRelation> = Vec::new();
        let mut seen: HashSet<(u32, String, String)> = HashSet::new();
        let mut in_code_block = false;

        let parser = Parser::new(source);
        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(Tag::CodeBlock(_)) => {
                    in_code_block = true;
                }
                Event::End(TagEnd::CodeBlock) => {
                    in_code_block = false;
                }
                Event::Code(code) => {
                    let symbol = code.as_ref();
                    if Self::is_meaningful_symbol(symbol) {
                        let line = byte_offset_to_line(range.start);
                        if let Some(chunk) = Self::find_chunk_for_line(chunks, line) {
                            let key = (
                                chunk.start_line,
                                String::new(),
                                symbol.to_string(),
                            );
                            if seen.insert(key) {
                                relations.push(ChunkRelation {
                                    from_source_file: meta.file_path.clone(),
                                    from_start_line: chunk.start_line,
                                    relation: "references".into(),
                                    to_file: String::new(),
                                    to_name: Some(symbol.to_string()),
                                });
                            }
                        }
                    }
                }
                Event::Start(Tag::Link { dest_url, .. }) => {
                    let url = dest_url.as_ref();
                    if Self::is_file_path_reference(url) {
                        let clean_path = url.split('#').next().unwrap_or(url).to_string();
                        let line = byte_offset_to_line(range.start);
                        if let Some(chunk) = Self::find_chunk_for_line(chunks, line) {
                            let key = (
                                chunk.start_line,
                                clean_path.clone(),
                                String::new(),
                            );
                            if seen.insert(key) {
                                relations.push(ChunkRelation {
                                    from_source_file: meta.file_path.clone(),
                                    from_start_line: chunk.start_line,
                                    relation: "references".into(),
                                    to_file: clean_path,
                                    to_name: None,
                                });
                            }
                        }
                    }
                }
                Event::Text(text) if in_code_block => {
                    // Scan code block text for file-path patterns.
                    let line = byte_offset_to_line(range.start);
                    for word in text.split_whitespace() {
                        // Strip surrounding quotes/punctuation.
                        let cleaned = word.trim_matches(|c: char| {
                            c == '"' || c == '\'' || c == ',' || c == ';' || c == '('  || c == ')'
                        });
                        if Self::is_file_path_reference(cleaned) {
                            if let Some(chunk) = Self::find_chunk_for_line(chunks, line) {
                                let key = (
                                    chunk.start_line,
                                    cleaned.to_string(),
                                    String::new(),
                                );
                                if seen.insert(key) {
                                    relations.push(ChunkRelation {
                                        from_source_file: meta.file_path.clone(),
                                        from_start_line: chunk.start_line,
                                        relation: "references".into(),
                                        to_file: cleaned.to_string(),
                                        to_name: None,
                                    });
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        relations
    }
}

impl ChunkingStrategy for MarkdownChunker {
    fn name(&self) -> &str {
        "markdown"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md"]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        if source.is_empty() {
            return Ok(ChunkResult::default());
        }

        let lines: Vec<&str> = source.lines().collect();
        let mut chunks: Vec<RawChunk> = Vec::new();

        // Accumulator for the current section.
        let mut section_lines: Vec<&str> = Vec::new();
        let mut section_start: u32 = 1; // 1-indexed
        let mut section_is_headed = false;
        let mut section_depth: Option<usize> = None;

        for (i, &line) in lines.iter().enumerate() {
            let line_num = (i + 1) as u32; // 1-indexed

            if let Some(depth) = Self::heading_depth(line) {
                // We hit a heading boundary — emit the accumulated section first.
                if !section_lines.is_empty() {
                    let content = section_lines.join("\n");
                    let chunk_type = if section_is_headed {
                        "heading_section"
                    } else {
                        "text"
                    };
                    let merge_group = section_depth.map(|d| format!("h{}", d));

                    chunks.push(RawChunk {
                        content,
                        chunk_type: chunk_type.into(),
                        start_line: section_start,
                        end_line: line_num - 1,
                        metadata: ChunkMetadata {
                            source_file: meta.file_path.clone(),
                            language: meta.language.clone(),
                            merge_group,
                            ..Default::default()
                        },
                    });
                }

                // Start a new headed section.
                section_lines = vec![line];
                section_start = line_num;
                section_is_headed = true;
                section_depth = Some(depth);
            } else {
                section_lines.push(line);
            }
        }

        // Emit the final accumulated section.
        if !section_lines.is_empty() {
            let content = section_lines.join("\n");
            let chunk_type = if section_is_headed {
                "heading_section"
            } else {
                "text"
            };
            let merge_group = section_depth.map(|d| format!("h{}", d));
            let end_line = lines.len() as u32;

            chunks.push(RawChunk {
                content,
                chunk_type: chunk_type.into(),
                start_line: section_start,
                end_line,
                metadata: ChunkMetadata {
                    source_file: meta.file_path.clone(),
                    language: meta.language.clone(),
                    merge_group,
                    ..Default::default()
                },
            });
        }

        let relations = Self::extract_references(source, &chunks, meta);
        Ok(ChunkResult { chunks, relations })
    }

    fn overlap_context(&self, prev: &RawChunk, _next: &RawChunk) -> Option<String> {
        // Extract heading lines from the previous chunk to provide
        // structural context for the next chunk.
        let heading_lines: Vec<&str> = prev
            .content
            .lines()
            .filter(|line| Self::heading_depth(line).is_some())
            .collect();

        if heading_lines.is_empty() {
            return None;
        }

        // Join heading lines with newline, plus a trailing newline.
        let mut context = heading_lines.join("\n");
        context.push('\n');
        Some(context)
    }

    fn merge_small(&self, chunks: &[RawChunk], max_tokens: usize) -> Option<Vec<RawChunk>> {
        if chunks.len() < 2 {
            return None;
        }

        // Simple chars/4 token estimation (matches CharDivFourEstimator).
        fn estimate(text: &str) -> usize {
            if text.is_empty() { 0 } else { (text.len() / 4).max(1) }
        }

        let mut result: Vec<RawChunk> = Vec::new();
        let mut merged_any = false;
        let mut i = 0;

        while i < chunks.len() {
            if i + 1 < chunks.len() {
                let current = &chunks[i];
                let next = &chunks[i + 1];

                // Only merge if both are heading_section with the same merge_group.
                let same_group = match (&current.metadata.merge_group, &next.metadata.merge_group) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };

                if same_group {
                    let combined = format!("{}\n\n{}", current.content, next.content);
                    if estimate(&combined) <= max_tokens {
                        result.push(RawChunk {
                            content: combined,
                            chunk_type: current.chunk_type.clone(),
                            start_line: current.start_line,
                            end_line: next.end_line,
                            metadata: current.metadata.clone(),
                        });
                        merged_any = true;
                        i += 2;
                        continue;
                    }
                }
            }

            result.push(chunks[i].clone());
            i += 1;
        }

        if merged_any {
            Some(result)
        } else {
            None
        }
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
            file_path: "docs/guide.md".into(),
            extension: "md".into(),
            language: Some("markdown".into()),
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
        }
    }

    #[test]
    fn test_splits_on_h1_headings() {
        let chunker = MarkdownChunker::new();
        let source = "# A\n\ntext under A\n\n# B\n\ntext under B";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 2, "expected 2 chunks, got {}", chunks.len());
        assert!(chunks[0].content.contains("# A"));
        assert!(chunks[0].content.contains("text under A"));
        assert!(chunks[1].content.contains("# B"));
        assert!(chunks[1].content.contains("text under B"));
    }

    #[test]
    fn test_splits_on_mixed_heading_levels() {
        let chunker = MarkdownChunker::new();
        let source = "# Top\n\nIntro.\n\n## A\n\nSection A body.\n\n## B\n\nSection B body.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        // All should be heading sections.
        for c in &chunks {
            assert_eq!(c.chunk_type, "heading_section");
        }
    }

    #[test]
    fn test_no_headings_returns_single_chunk() {
        let chunker = MarkdownChunker::new();
        let source = "Just plain text.\nNo headings here.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 1, "expected 1 chunk, got {}", chunks.len());
        assert_eq!(chunks[0].chunk_type, "text");
        assert_eq!(chunks[0].content, source);
    }

    #[test]
    fn test_chunk_type_is_heading_section() {
        let chunker = MarkdownChunker::new();
        let source = "## Auth\n\nAuth details here.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "heading_section");
    }

    #[test]
    fn test_empty_source() {
        let chunker = MarkdownChunker::new();
        let chunks = chunker.chunk("", &test_meta()).unwrap().chunks;
        assert!(chunks.is_empty(), "expected empty vec for empty source");
    }

    #[test]
    fn test_overlap_carries_heading_hierarchy() {
        let chunker = MarkdownChunker::new();
        let prev = RawChunk {
            content: "## Auth\n\n### JWT\n\nJWT details here.".into(),
            chunk_type: "heading_section".into(),
            start_line: 1,
            end_line: 5,
            metadata: ChunkMetadata {
                source_file: "docs/guide.md".into(),
                language: Some("markdown".into()),
                ..Default::default()
            },
        };
        let next = RawChunk {
            content: "### OAuth\n\nOAuth details.".into(),
            chunk_type: "heading_section".into(),
            start_line: 6,
            end_line: 8,
            metadata: ChunkMetadata {
                source_file: "docs/guide.md".into(),
                language: Some("markdown".into()),
                ..Default::default()
            },
        };

        let overlap = chunker.overlap_context(&prev, &next);
        assert!(overlap.is_some(), "expected Some overlap context");
        let text = overlap.unwrap();
        assert!(
            text.contains("## Auth"),
            "overlap should contain parent heading '## Auth', got: {:?}",
            text
        );
        assert!(
            text.contains("### JWT"),
            "overlap should contain sub-heading '### JWT', got: {:?}",
            text
        );
    }

    #[test]
    fn test_name_and_extensions() {
        let chunker = MarkdownChunker::new();
        assert_eq!(chunker.name(), "markdown");
        assert_eq!(chunker.supported_extensions(), &["md"]);
    }

    // -- Additional edge-case tests ----------------------------------------

    #[test]
    fn test_content_before_first_heading_gets_own_chunk() {
        let chunker = MarkdownChunker::new();
        let source = "Preamble text.\n\n# First Heading\n\nBody.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 2, "expected 2 chunks, got {}", chunks.len());
        assert_eq!(chunks[0].chunk_type, "text");
        assert!(chunks[0].content.contains("Preamble text."));
        assert_eq!(chunks[1].chunk_type, "heading_section");
        assert!(chunks[1].content.contains("# First Heading"));
    }

    #[test]
    fn test_line_numbers_are_accurate() {
        let chunker = MarkdownChunker::new();
        let source = "Preamble.\n\n# Heading\n\nBody line 1.\nBody line 2.";
        let chunks = chunker.chunk(source, &test_meta()).unwrap().chunks;

        assert_eq!(chunks.len(), 2);
        // Preamble: lines 1..2 (line 1 = "Preamble.", line 2 = "")
        assert_eq!(chunks[0].start_line, 1);
        // Heading section starts at line 3 ("# Heading")
        assert_eq!(chunks[1].start_line, 3);
        // Last line of source is line 6.
        assert_eq!(chunks[1].end_line, 6);
    }

    #[test]
    fn test_metadata_source_and_language() {
        let chunker = MarkdownChunker::new();
        let source = "# Title\n\nContent.";
        let meta = test_meta();
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.source_file, "docs/guide.md");
        assert_eq!(chunks[0].metadata.language, Some("markdown".into()));
    }

    #[test]
    fn test_merge_small_combines_same_depth() {
        let chunker = MarkdownChunker::new();
        let chunks = vec![
            RawChunk {
                content: "## A\n\nSmall.".into(),
                chunk_type: "heading_section".into(),
                start_line: 1,
                end_line: 3,
                metadata: ChunkMetadata {
                    source_file: "test.md".into(),
                    merge_group: Some("h2".into()),
                    ..Default::default()
                },
            },
            RawChunk {
                content: "## B\n\nAlso small.".into(),
                chunk_type: "heading_section".into(),
                start_line: 4,
                end_line: 6,
                metadata: ChunkMetadata {
                    source_file: "test.md".into(),
                    merge_group: Some("h2".into()),
                    ..Default::default()
                },
            },
        ];

        // max_tokens large enough to hold both combined.
        let result = chunker.merge_small(&chunks, 1000);
        assert!(result.is_some(), "expected Some merged result");
        let merged = result.unwrap();
        assert_eq!(merged.len(), 1, "expected 1 merged chunk, got {}", merged.len());
        assert!(merged[0].content.contains("## A"));
        assert!(merged[0].content.contains("## B"));
    }

    #[test]
    fn test_merge_small_does_not_cross_depth() {
        let chunker = MarkdownChunker::new();
        let chunks = vec![
            RawChunk {
                content: "## A\n\nSmall.".into(),
                chunk_type: "heading_section".into(),
                start_line: 1,
                end_line: 3,
                metadata: ChunkMetadata {
                    source_file: "test.md".into(),
                    merge_group: Some("h2".into()),
                    ..Default::default()
                },
            },
            RawChunk {
                content: "### B\n\nAlso small.".into(),
                chunk_type: "heading_section".into(),
                start_line: 4,
                end_line: 6,
                metadata: ChunkMetadata {
                    source_file: "test.md".into(),
                    merge_group: Some("h3".into()),
                    ..Default::default()
                },
            },
        ];

        let result = chunker.merge_small(&chunks, 1000);
        // Different merge groups should not merge, so either None or unchanged length.
        match result {
            None => {} // fine
            Some(ref v) => assert_eq!(v.len(), 2, "should not merge across heading depths"),
        }
    }

    // -- Cross-reference extraction tests -----------------------------------

    #[test]
    fn test_extract_code_span_references() {
        let chunker = MarkdownChunker::new();
        let source = "# Architecture\n\nThe `QueryableStore` trait provides search.";
        let meta = test_meta();
        let result = chunker.chunk(source, &meta).unwrap();

        let refs: Vec<_> = result
            .relations
            .iter()
            .filter(|r| r.relation == "references" && r.to_name.is_some())
            .collect();

        assert!(
            !refs.is_empty(),
            "expected at least one code-span reference, got none"
        );
        assert_eq!(refs[0].to_name.as_deref(), Some("QueryableStore"));
        assert_eq!(refs[0].to_file, "");
        assert_eq!(refs[0].from_source_file, "docs/guide.md");
    }

    #[test]
    fn test_extract_link_references() {
        let chunker = MarkdownChunker::new();
        let source = "# Traits\n\nSee [trait file](src/traits.rs) for details.";
        let meta = test_meta();
        let result = chunker.chunk(source, &meta).unwrap();

        let refs: Vec<_> = result
            .relations
            .iter()
            .filter(|r| r.relation == "references" && !r.to_file.is_empty())
            .collect();

        assert!(
            !refs.is_empty(),
            "expected at least one file-path reference, got none"
        );
        assert_eq!(refs[0].to_file, "src/traits.rs");
        assert!(refs[0].to_name.is_none());
    }

    #[test]
    fn test_skip_common_types() {
        let chunker = MarkdownChunker::new();
        let source =
            "# Types\n\nUses `String` and `Vec` internally, wraps `QueryableStore`.";
        let meta = test_meta();
        let result = chunker.chunk(source, &meta).unwrap();

        let names: Vec<_> = result
            .relations
            .iter()
            .filter_map(|r| r.to_name.as_deref())
            .collect();

        assert!(
            !names.contains(&"String"),
            "String should be skipped, but was found in relations"
        );
        assert!(
            !names.contains(&"Vec"),
            "Vec should be skipped, but was found in relations"
        );
        assert!(
            names.contains(&"QueryableStore"),
            "QueryableStore should be kept, but was not found in relations"
        );
    }

    #[test]
    fn test_skip_http_urls() {
        let chunker = MarkdownChunker::new();
        let source =
            "# Links\n\nSee [docs](https://example.com/foo.rs) and [local](src/main.rs).";
        let meta = test_meta();
        let result = chunker.chunk(source, &meta).unwrap();

        let files: Vec<_> = result
            .relations
            .iter()
            .filter(|r| !r.to_file.is_empty())
            .map(|r| r.to_file.as_str())
            .collect();

        assert!(
            !files.contains(&"https://example.com/foo.rs"),
            "HTTP URL should be skipped, but was found: {:?}",
            files
        );
        assert!(
            files.contains(&"src/main.rs"),
            "local file reference should be kept: {:?}",
            files
        );
    }
}
