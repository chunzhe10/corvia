use async_trait::async_trait;
use corvia_common::errors::{CorviaError, Result};
use corvia_common::types::KnowledgeEntry;
use corvia_kernel::traits::IngestionAdapter;
use git2::Repository;
use std::path::Path;
use tracing::{info, debug, warn};
use walkdir::WalkDir;

use crate::treesitter::{self, CodeRelation};

/// Result of relation-aware ingestion: knowledge entries plus structural relations.
pub struct IngestionResult {
    pub entries: Vec<KnowledgeEntry>,
    pub relations: Vec<CodeRelation>,
}

pub struct GitAdapter;

impl Default for GitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GitAdapter {
    pub fn new() -> Self {
        Self
    }
}

/// File extensions we attempt to parse.
const SUPPORTED_EXTENSIONS: &[&str] = &["rs", "js", "jsx", "ts", "tsx", "py", "md", "toml", "yaml", "yml", "json"];

/// Directories to skip during ingestion.
const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", ".corvia", ".claude", "dist", "build", "__pycache__", ".venv", "vendor", "benchmark-results"];

#[async_trait]
impl IngestionAdapter for GitAdapter {
    fn domain(&self) -> &str {
        "git"
    }

    fn register_chunking(&self, registry: &mut corvia_kernel::chunking_pipeline::FormatRegistry) {
        use corvia_kernel::chunking_strategy::ChunkingStrategy;
        use std::sync::Arc;
        let ast = Arc::new(crate::ast_chunker::AstChunker::new());
        for ext in ast.supported_extensions() {
            registry.register_override(ext, ast.clone());
        }
    }

    async fn ingest_sources(
        &self,
        source_path: &str,
    ) -> Result<Vec<corvia_kernel::chunking_strategy::SourceFile>> {
        use corvia_kernel::chunking_strategy::{SourceFile, SourceMetadata};

        let path = Path::new(source_path);
        if !path.exists() {
            return Err(CorviaError::Ingestion(format!(
                "Path does not exist: {source_path}"
            )));
        }

        let source_version = get_head_sha(path).unwrap_or_else(|| "unknown".to_string());
        let scope_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!(
            "Ingesting sources from {} (version: {}, scope: {})",
            source_path, source_version, scope_id
        );

        let mut files = Vec::new();

        for entry in WalkDir::new(path).into_iter().filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.iter().any(|skip| name == *skip)
        }) {
            let entry =
                entry.map_err(|e| CorviaError::Ingestion(format!("Walk error: {e}")))?;
            if !entry.file_type().is_file() {
                continue;
            }

            let file_path = entry.path();
            let extension = file_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if !SUPPORTED_EXTENSIONS.contains(&extension) {
                continue;
            }

            let Ok(content) = std::fs::read_to_string(file_path) else {
                debug!(
                    "Skipping binary or unreadable file: {}",
                    file_path.display()
                );
                continue;
            };

            if content.len() > 100_000 {
                warn!(
                    "Skipping large file ({}KB): {}",
                    content.len() / 1024,
                    file_path.display()
                );
                continue;
            }

            let relative_path = file_path
                .strip_prefix(path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            files.push(SourceFile {
                content,
                metadata: SourceMetadata {
                    file_path: relative_path,
                    extension: extension.to_string(),
                    language: lang_for_ext(extension),
                    scope_id: scope_id.clone(),
                    source_version: source_version.clone(),
                    workstream: None,
                    content_role: None,
                    source_origin: None,
                    parent_session_id: None,
                },
            });
        }

        info!("Collected {} source files from {}", files.len(), source_path);
        Ok(files)
    }
}

impl GitAdapter {
    /// Ingest a source directory and return both knowledge entries and structural relations.
    ///
    /// Unlike `ingest()` (from the IngestionAdapter trait), this method also extracts
    /// structural relations (imports, implements, contains) from the AST. The
    /// `from_chunk_index` in each CodeRelation is offset to be globally unique across
    /// all files (matching the index into the entries vec).
    pub async fn ingest_with_relations(&self, source_path: &str) -> Result<IngestionResult> {
        let path = Path::new(source_path);
        if !path.exists() {
            return Err(CorviaError::Ingestion(format!(
                "Path does not exist: {source_path}"
            )));
        }

        let source_version = get_head_sha(path).unwrap_or_else(|| "unknown".to_string());
        let scope_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!(
            "Ingesting with relations {} (version: {}, scope: {})",
            source_path, source_version, scope_id
        );

        let mut entries = Vec::new();
        let mut all_relations = Vec::new();

        for entry in WalkDir::new(path).into_iter().filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.iter().any(|skip| name == *skip)
        }) {
            let entry = entry.map_err(|e| CorviaError::Ingestion(format!("Walk error: {e}")))?;
            if !entry.file_type().is_file() {
                continue;
            }

            let file_path = entry.path();
            let extension = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

            if !SUPPORTED_EXTENSIONS.contains(&extension) {
                continue;
            }

            let Ok(source) = std::fs::read_to_string(file_path) else {
                debug!(
                    "Skipping binary or unreadable file: {}",
                    file_path.display()
                );
                continue;
            };

            if source.len() > 100_000 {
                warn!(
                    "Skipping large file ({}KB): {}",
                    source.len() / 1024,
                    file_path.display()
                );
                continue;
            }

            let relative_path = file_path
                .strip_prefix(path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            let chunk_offset = entries.len();
            let result =
                treesitter::chunk_file_with_relations(&relative_path, &source, extension);

            for chunk in &result.chunks {
                entries.push(chunk.to_knowledge_entry(&scope_id, &source_version));
            }

            // Offset relation indices by the number of entries collected before this file
            for mut relation in result.relations {
                relation.from_chunk_index += chunk_offset;
                all_relations.push(relation);
            }
        }

        info!(
            "Ingested {} chunks and {} relations from {}",
            entries.len(),
            all_relations.len(),
            source_path
        );
        Ok(IngestionResult {
            entries,
            relations: all_relations,
        })
    }
}

/// Map file extensions to language names for SourceMetadata.
fn lang_for_ext(ext: &str) -> Option<String> {
    match ext {
        "rs" => Some("rust".into()),
        "js" | "jsx" => Some("javascript".into()),
        "ts" | "tsx" => Some("typescript".into()),
        "py" => Some("python".into()),
        "md" => Some("markdown".into()),
        "toml" => Some("toml".into()),
        "yaml" | "yml" => Some("yaml".into()),
        "json" => Some("json".into()),
        _ => None,
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
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_ingest_sources_nonexistent_path() {
        let adapter = GitAdapter::new();
        let result = adapter.ingest_sources("/nonexistent/path").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ingest_with_relations_rust_files() {
        let dir = TempDir::new().unwrap();
        let src_dir = dir.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();

        // Write a Rust file with use statements and an impl
        fs::write(
            src_dir.join("lib.rs"),
            r#"
use std::collections::HashMap;
use crate::foo::Bar;

pub trait MyTrait {
    fn do_thing(&self);
}

pub struct MyStruct {
    field: i32,
}

impl MyTrait for MyStruct {
    fn do_thing(&self) {
        println!("hello");
    }
}
"#,
        )
        .unwrap();

        // Write a second Rust file
        fs::write(
            src_dir.join("foo.rs"),
            r#"
use super::MyTrait;

pub struct Bar {
    val: String,
}
"#,
        )
        .unwrap();

        let adapter = GitAdapter::new();
        let result = adapter
            .ingest_with_relations(dir.path().to_str().unwrap())
            .await
            .unwrap();

        // Should have entries from both files
        assert!(
            result.entries.len() >= 4,
            "Expected at least 4 entries, got {}",
            result.entries.len()
        );

        // Should have import relations
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(
            imports.len() >= 3,
            "Expected at least 3 import relations, got {}",
            imports.len()
        );

        // Should have an implements relation
        let implements: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "implements")
            .collect();
        assert_eq!(
            implements.len(),
            1,
            "Expected 1 implements relation, got {}",
            implements.len()
        );
        assert_eq!(implements[0].to_name.as_deref(), Some("MyTrait"));

        // Verify from_chunk_index offsets are valid (within entries range)
        for rel in &result.relations {
            assert!(
                rel.from_chunk_index < result.entries.len(),
                "from_chunk_index {} out of bounds (entries len: {})",
                rel.from_chunk_index,
                result.entries.len()
            );
        }
    }

    #[tokio::test]
    async fn test_ingest_with_relations_nonexistent_path() {
        let adapter = GitAdapter::new();
        let result = adapter.ingest_with_relations("/nonexistent/path").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ingest_with_relations_empty_dir() {
        let dir = TempDir::new().unwrap();
        let adapter = GitAdapter::new();
        let result = adapter
            .ingest_with_relations(dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert!(result.entries.is_empty());
        assert!(result.relations.is_empty());
    }

    #[tokio::test]
    async fn test_ingest_sources_returns_source_files() {
        let dir = TempDir::new().unwrap();

        // Create a .rs file
        fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();

        // Create a .md file
        fs::write(dir.path().join("README.md"), "# Hello\n\nWorld.\n").unwrap();

        // Create a .txt file (not in SUPPORTED_EXTENSIONS, should be ignored)
        fs::write(dir.path().join("notes.txt"), "some notes").unwrap();

        let adapter = GitAdapter::new();
        let files = adapter
            .ingest_sources(dir.path().to_str().unwrap())
            .await
            .unwrap();

        // Should include .rs and .md but not .txt
        assert_eq!(files.len(), 2, "expected 2 source files (.rs + .md), got {}", files.len());

        let rs_file = files.iter().find(|f| f.metadata.extension == "rs");
        assert!(rs_file.is_some(), "expected a .rs SourceFile");
        let rs = rs_file.unwrap();
        assert_eq!(rs.metadata.language.as_deref(), Some("rust"));
        assert!(rs.content.contains("fn main()"));
        assert!(!rs.metadata.source_version.is_empty());

        let md_file = files.iter().find(|f| f.metadata.extension == "md");
        assert!(md_file.is_some(), "expected a .md SourceFile");
        let md = md_file.unwrap();
        assert_eq!(md.metadata.language.as_deref(), Some("markdown"));
        assert!(md.content.contains("# Hello"));
    }

    #[test]
    fn test_register_chunking_adds_ast_strategy() {
        use corvia_kernel::chunking_pipeline::FormatRegistry;
        use corvia_kernel::chunking_strategy::ChunkingStrategy;
        use std::sync::Arc;

        // Create a registry with a dummy fallback
        struct DummyFallback;
        impl ChunkingStrategy for DummyFallback {
            fn name(&self) -> &str { "dummy" }
            fn supported_extensions(&self) -> &[&str] { &[] }
            fn chunk(
                &self,
                _source: &str,
                _meta: &corvia_kernel::chunking_strategy::SourceMetadata,
            ) -> corvia_common::errors::Result<corvia_kernel::chunking_strategy::ChunkResult> {
                Ok(corvia_kernel::chunking_strategy::ChunkResult {
                    chunks: vec![],
                    relations: vec![],
                })
            }
        }

        let mut registry = FormatRegistry::new(Arc::new(DummyFallback));

        let adapter = GitAdapter::new();
        adapter.register_chunking(&mut registry);

        // After registration, .rs should resolve to "ast"
        let resolved = registry.resolve("rs");
        assert_eq!(resolved.name(), "ast", ".rs should resolve to ast strategy");

        // .py should also resolve to "ast"
        let resolved_py = registry.resolve("py");
        assert_eq!(resolved_py.name(), "ast", ".py should resolve to ast strategy");

        // .md should still fall back to dummy (not registered by AstChunker)
        let resolved_md = registry.resolve("md");
        assert_eq!(resolved_md.name(), "dummy", ".md should fall back to dummy");
    }
}
