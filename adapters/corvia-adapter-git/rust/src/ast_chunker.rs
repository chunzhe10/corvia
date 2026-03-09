//! AstChunker -- tree-sitter AST-aware chunking strategy (D65/D68).
//!
//! Wraps the existing [`treesitter::chunk_file`] parser as a
//! [`ChunkingStrategy`] so the kernel's [`ChunkingPipeline`] can route
//! code files through tree-sitter AST parsing.

use corvia_common::errors::Result;
use corvia_kernel::chunking_strategy::*;

use crate::treesitter;

/// AST-aware chunking strategy powered by tree-sitter.
///
/// Delegates to [`treesitter::chunk_file_with_relations`] for the actual
/// parsing, then maps the resulting [`CodeChunk`]s and [`CodeRelation`]s
/// to the kernel's [`RawChunk`] and [`ChunkRelation`] types.
pub struct AstChunker;

impl AstChunker {
    pub fn new() -> Self {
        Self
    }
}

impl ChunkingStrategy for AstChunker {
    fn name(&self) -> &str {
        "ast"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs", "js", "jsx", "ts", "tsx", "py"]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        let result = treesitter::chunk_file_with_relations(&meta.file_path, source, &meta.extension);

        let chunks: Vec<RawChunk> = result
            .chunks
            .iter()
            .map(|cc| RawChunk {
                content: cc.content.clone(),
                chunk_type: cc.chunk_type.clone(),
                start_line: cc.start_line,
                end_line: cc.end_line,
                metadata: ChunkMetadata {
                    source_file: cc.file_path.clone(),
                    language: Some(cc.language.clone()),
                    ..Default::default()
                },
            })
            .collect();

        // Convert chunk-index-based CodeRelations to stable (source_file, start_line) ChunkRelations
        let relations: Vec<ChunkRelation> = result
            .relations
            .iter()
            .filter_map(|cr| {
                let source_chunk = result.chunks.get(cr.from_chunk_index)?;
                Some(ChunkRelation {
                    from_source_file: source_chunk.file_path.clone(),
                    from_start_line: source_chunk.start_line,
                    relation: cr.relation.clone(),
                    to_file: cr.to_file.clone(),
                    to_name: cr.to_name.clone(),
                })
            })
            .collect();

        Ok(ChunkResult { chunks, relations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_for(file_path: &str, ext: &str) -> SourceMetadata {
        SourceMetadata {
            file_path: file_path.into(),
            extension: ext.into(),
            language: None,
            scope_id: "test:scope".into(),
            source_version: "v1".into(),
        }
    }

    #[test]
    fn test_rust_function_chunks() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}
"#;
        let chunker = AstChunker::new();
        let meta = meta_for("src/main.rs", "rs");
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        assert_eq!(chunks.len(), 2, "expected 2 function_item chunks");
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[1].content.contains("world"));
        assert_eq!(chunks[0].chunk_type, "function_item");
        assert_eq!(chunks[1].chunk_type, "function_item");
    }

    #[test]
    fn test_python_class_chunks() {
        let source = r#"
class MyClass:
    def method(self):
        pass

def standalone():
    return 42
"#;
        let chunker = AstChunker::new();
        let meta = meta_for("app.py", "py");
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        // Should produce at least a class and a function chunk.
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks (class + function), got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_unsupported_extension_returns_whole_file() {
        let source = "some content\nin a file\nwith multiple lines";
        let chunker = AstChunker::new();
        let meta = meta_for("data.txt", "txt");
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        assert_eq!(chunks.len(), 1, "unsupported extension should return 1 whole-file chunk");
        assert_eq!(chunks[0].chunk_type, "file");
        assert_eq!(chunks[0].content, source);
    }

    #[test]
    fn test_metadata_populated() {
        let source = r#"
fn greet() {
    println!("hi");
}
"#;
        let chunker = AstChunker::new();
        let meta = meta_for("src/lib.rs", "rs");
        let chunks = chunker.chunk(source, &meta).unwrap().chunks;

        assert!(!chunks.is_empty());
        let first = &chunks[0];
        assert_eq!(first.metadata.source_file, "src/lib.rs");
        assert_eq!(first.metadata.language.as_deref(), Some("rs"));
    }

    #[test]
    fn test_name_and_extensions() {
        let chunker = AstChunker::new();
        assert_eq!(chunker.name(), "ast");
        assert_eq!(
            chunker.supported_extensions(),
            &["rs", "js", "jsx", "ts", "tsx", "py"]
        );
    }
}
