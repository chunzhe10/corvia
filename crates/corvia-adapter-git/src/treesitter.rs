use corvia_common::types::{EntryMetadata, KnowledgeEntry};
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};
use tree_sitter_language::LanguageFn;
use tracing::debug;

/// Supported languages and their tree-sitter grammars + queries.
struct LangConfig {
    language: LanguageFn,
    /// Query to match top-level constructs (functions, classes, structs, etc.)
    query: &'static str,
}

fn lang_config_for(extension: &str) -> Option<LangConfig> {
    match extension {
        "rs" => Some(LangConfig {
            language: tree_sitter_rust::LANGUAGE,
            query: "(function_item) @chunk
                    (struct_item) @chunk
                    (enum_item) @chunk
                    (impl_item) @chunk
                    (trait_item) @chunk
                    (mod_item) @chunk",
        }),
        "js" | "jsx" => Some(LangConfig {
            language: tree_sitter_javascript::LANGUAGE,
            query: "(function_declaration) @chunk
                    (class_declaration) @chunk
                    (export_statement) @chunk
                    (lexical_declaration) @chunk",
        }),
        "ts" => Some(LangConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
            query: "(function_declaration) @chunk
                    (class_declaration) @chunk
                    (export_statement) @chunk
                    (interface_declaration) @chunk
                    (type_alias_declaration) @chunk
                    (lexical_declaration) @chunk",
        }),
        "tsx" => Some(LangConfig {
            language: tree_sitter_typescript::LANGUAGE_TSX,
            query: "(function_declaration) @chunk
                    (class_declaration) @chunk
                    (export_statement) @chunk
                    (interface_declaration) @chunk
                    (type_alias_declaration) @chunk
                    (lexical_declaration) @chunk",
        }),
        "py" => Some(LangConfig {
            language: tree_sitter_python::LANGUAGE,
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

/// A structural relation extracted from tree-sitter AST.
/// References chunks by index into the chunks vec returned by chunk_file().
#[derive(Debug, Clone)]
pub struct CodeRelation {
    /// Index into the chunks vec for the chunk that "owns" this relation.
    pub from_chunk_index: usize,
    /// Relation type: "imports", "implements", or "contains".
    pub relation: String,
    /// Best-effort target file or module path (e.g., "crate::foo" for Rust, "./utils" for JS).
    pub to_file: String,
    /// Symbol name if identifiable from the AST (e.g., "Bar" for `use crate::foo::Bar`).
    pub to_name: Option<String>,
}

/// Result of chunk_file_with_relations(): both chunks and extracted relations.
pub struct ChunkResult {
    pub chunks: Vec<CodeChunk>,
    pub relations: Vec<CodeRelation>,
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

    let ts_language: tree_sitter::Language = config.language.into();

    let mut parser = Parser::new();
    if parser.set_language(&ts_language).is_err() {
        return vec![];
    }

    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let Ok(query) = Query::new(&ts_language, config.query) else {
        return vec![];
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

    let mut chunks = Vec::new();
    while let Some(m) = matches.next() {
        for capture in m.captures {
            let node = capture.node;
            let content = &source[node.byte_range()];
            // Skip very small chunks (one-liners that are trivial)
            if content.lines().count() < 2 {
                continue;
            }
            debug!(
                file = file_path,
                kind = node.kind(),
                start = node.start_position().row + 1,
                end = node.end_position().row + 1,
                "extracted chunk"
            );
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

/// Parse a source file and extract both AST-aware chunks and structural relations.
/// This is the preferred entry point for relation-aware ingestion.
pub fn chunk_file_with_relations(file_path: &str, source: &str, extension: &str) -> ChunkResult {
    let chunks = chunk_file(file_path, source, extension);
    let relations = extract_relations(file_path, source, extension, &chunks);
    ChunkResult { chunks, relations }
}

/// Extract structural relations (imports, implements, contains) from the AST.
///
/// Relations are best-effort: cross-file resolution is deferred to the wiring step.
/// `from_chunk_index` references the chunk that owns the relation. For top-of-file
/// imports, this is chunk index 0 (the first chunk in the file).
fn extract_relations(
    file_path: &str,
    source: &str,
    extension: &str,
    chunks: &[CodeChunk],
) -> Vec<CodeRelation> {
    if chunks.is_empty() {
        return vec![];
    }

    match extension {
        "rs" => extract_rust_relations(file_path, source, chunks),
        "js" | "jsx" | "ts" | "tsx" => extract_js_ts_relations(file_path, source, extension, chunks),
        "py" => extract_python_relations(file_path, source, chunks),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Rust relation extraction
// ---------------------------------------------------------------------------

fn extract_rust_relations(
    file_path: &str,
    source: &str,
    chunks: &[CodeChunk],
) -> Vec<CodeRelation> {
    let ts_language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    let mut parser = Parser::new();
    if parser.set_language(&ts_language).is_err() {
        return vec![];
    }
    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let mut relations = Vec::new();
    let root = tree.root_node();
    let mut cursor = root.walk();

    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();
            if node.is_named() {
                match node.kind() {
                    "use_declaration" => {
                        let owner_idx = find_owning_chunk(chunks, node.start_position().row as u32 + 1);
                        extract_rust_use(&mut relations, source, &node, file_path, owner_idx);
                    }
                    "impl_item" => {
                        extract_rust_impl(&mut relations, source, &node, file_path, chunks);
                    }
                    "mod_item" => {
                        extract_rust_mod_contains(&mut relations, source, &node, file_path, chunks);
                    }
                    _ => {}
                }
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    relations
}

/// Extract imports from a Rust `use_declaration` node.
fn extract_rust_use(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    file_path: &str,
    owner_idx: usize,
) {
    let Some(argument) = node.child_by_field_name("argument") else {
        return;
    };

    match argument.kind() {
        "scoped_identifier" => {
            // e.g., `use crate::foo::Bar`
            let path_text = argument
                .child_by_field_name("path")
                .map(|p| source[p.byte_range()].to_string())
                .unwrap_or_default();
            let name_text = argument
                .child_by_field_name("name")
                .map(|n| source[n.byte_range()].to_string());
            relations.push(CodeRelation {
                from_chunk_index: owner_idx,
                relation: "imports".to_string(),
                to_file: resolve_rust_module_path(file_path, &path_text),
                to_name: name_text,
            });
        }
        "scoped_use_list" => {
            // e.g., `use std::collections::{HashMap, HashSet}`
            let path_text = argument
                .child_by_field_name("path")
                .map(|p| source[p.byte_range()].to_string())
                .unwrap_or_default();
            let resolved_path = resolve_rust_module_path(file_path, &path_text);
            if let Some(list) = argument.child_by_field_name("list") {
                for i in 0..list.child_count() {
                    let Some(child) = list.child(i as u32) else {
                        continue;
                    };
                    if !child.is_named() {
                        continue;
                    }
                    let name = match child.kind() {
                        "identifier" => source[child.byte_range()].to_string(),
                        "use_as_clause" => {
                            // Extract the original name (before `as`)
                            child
                                .child_by_field_name("path")
                                .map(|p| source[p.byte_range()].to_string())
                                .unwrap_or_else(|| source[child.byte_range()].to_string())
                        }
                        _ => continue,
                    };
                    relations.push(CodeRelation {
                        from_chunk_index: owner_idx,
                        relation: "imports".to_string(),
                        to_file: resolved_path.clone(),
                        to_name: Some(name),
                    });
                }
            }
        }
        "use_wildcard" => {
            // e.g., `use super::*`
            let full_text = source[argument.byte_range()].to_string();
            let module_path = full_text.trim_end_matches("::*");
            relations.push(CodeRelation {
                from_chunk_index: owner_idx,
                relation: "imports".to_string(),
                to_file: resolve_rust_module_path(file_path, module_path),
                to_name: Some("*".to_string()),
            });
        }
        "identifier" => {
            // e.g., `use foo` (bare crate import, rare)
            let name = source[argument.byte_range()].to_string();
            relations.push(CodeRelation {
                from_chunk_index: owner_idx,
                relation: "imports".to_string(),
                to_file: name.clone(),
                to_name: Some(name),
            });
        }
        _ => {}
    }
}

/// Extract "implements" relation from a Rust `impl_item`.
/// If `impl Trait for Type`, emit relation from the impl chunk to the trait.
fn extract_rust_impl(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    file_path: &str,
    chunks: &[CodeChunk],
) {
    let trait_node = node.child_by_field_name("trait");
    let type_node = node.child_by_field_name("type");

    // Only emit "implements" for `impl Trait for Type`
    if let (Some(trait_n), Some(_type_n)) = (trait_node, type_node) {
        let trait_name = source[trait_n.byte_range()].to_string();
        let impl_line = node.start_position().row as u32 + 1;
        let owner_idx = find_chunk_by_line(chunks, impl_line);
        relations.push(CodeRelation {
            from_chunk_index: owner_idx,
            relation: "implements".to_string(),
            to_file: file_path.to_string(),
            to_name: Some(trait_name),
        });
    }
}

/// Extract "contains" relations from a Rust `mod_item` that has an inline body.
fn extract_rust_mod_contains(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    file_path: &str,
    chunks: &[CodeChunk],
) {
    let Some(body) = node.child_by_field_name("body") else {
        return; // `mod foo;` without body — no containment to extract
    };
    let mod_line = node.start_position().row as u32 + 1;
    let mod_idx = find_chunk_by_line(chunks, mod_line);

    // Walk the body's named children for functions, structs, etc.
    for i in 0..body.child_count() {
        let Some(child) = body.child(i as u32) else {
            continue;
        };
        if !child.is_named() {
            continue;
        }
        let child_kind = child.kind();
        // Only track containment for substantial definitions
        if !matches!(
            child_kind,
            "function_item" | "struct_item" | "enum_item" | "impl_item" | "trait_item" | "mod_item"
        ) {
            continue;
        }
        let child_name = child
            .child_by_field_name("name")
            .map(|n| source[n.byte_range()].to_string());

        // Try to find a chunk for the contained item
        let child_line = child.start_position().row as u32 + 1;
        let child_idx = find_chunk_by_line(chunks, child_line);

        // Only emit if the contained item is a different chunk from the mod chunk itself
        if child_idx != mod_idx || child_name.is_some() {
            relations.push(CodeRelation {
                from_chunk_index: mod_idx,
                relation: "contains".to_string(),
                to_file: file_path.to_string(),
                to_name: child_name,
            });
        }
    }
}

/// Best-effort resolution for Rust module paths.
/// `crate::foo::bar` → record as-is (file_path for crate-internal).
/// `super::foo` → record as-is.
/// `std::*` / external → record as-is.
fn resolve_rust_module_path(file_path: &str, module_path: &str) -> String {
    if module_path.starts_with("crate::") || module_path.starts_with("super::") || module_path == "crate" || module_path == "super" {
        // Crate-internal: record the file_path as the reference file
        file_path.to_string()
    } else {
        // External crate or std — record the module path itself
        module_path.to_string()
    }
}

// ---------------------------------------------------------------------------
// JavaScript/TypeScript relation extraction
// ---------------------------------------------------------------------------

fn extract_js_ts_relations(
    _file_path: &str,
    source: &str,
    extension: &str,
    chunks: &[CodeChunk],
) -> Vec<CodeRelation> {
    let lang_fn: LanguageFn = match extension {
        "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX,
        _ => tree_sitter_javascript::LANGUAGE,
    };
    let ts_language: tree_sitter::Language = lang_fn.into();
    let mut parser = Parser::new();
    if parser.set_language(&ts_language).is_err() {
        return vec![];
    }
    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let mut relations = Vec::new();
    let root = tree.root_node();
    let mut cursor = root.walk();

    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();
            if node.is_named() && node.kind() == "import_statement" {
                let owner_idx = find_owning_chunk(chunks, node.start_position().row as u32 + 1);
                if let Some(source_node) = node.child_by_field_name("source") {
                    let raw = source[source_node.byte_range()].to_string();
                    // Strip quotes from the import source string
                    let import_path = raw.trim_matches(|c| c == '\'' || c == '"').to_string();

                    // Try to extract named imports from import_clause
                    let mut names = Vec::new();
                    for i in 0..node.child_count() {
                        let Some(child) = node.child(i as u32) else {
                            continue;
                        };
                        if child.kind() == "import_clause" {
                            collect_js_import_names(&mut names, source, &child);
                        }
                    }

                    if names.is_empty() {
                        // Bare import or couldn't parse names
                        relations.push(CodeRelation {
                            from_chunk_index: owner_idx,
                            relation: "imports".to_string(),
                            to_file: import_path,
                            to_name: None,
                        });
                    } else {
                        for name in names {
                            relations.push(CodeRelation {
                                from_chunk_index: owner_idx,
                                relation: "imports".to_string(),
                                to_file: import_path.clone(),
                                to_name: Some(name),
                            });
                        }
                    }
                }
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    relations
}

/// Collect named import symbols from a JS/TS import_clause node.
fn collect_js_import_names(names: &mut Vec<String>, source: &str, node: &Node) {
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        match child.kind() {
            "identifier" => {
                // Default import: `import foo from '...'`
                names.push(source[child.byte_range()].to_string());
            }
            "named_imports" => {
                // `{ foo, bar }` — extract each import_specifier
                for j in 0..child.child_count() {
                    let Some(spec) = child.child(j as u32) else {
                        continue;
                    };
                    if spec.kind() == "import_specifier" {
                        // The "name" field is the imported name
                        if let Some(name_node) = spec.child_by_field_name("name") {
                            names.push(source[name_node.byte_range()].to_string());
                        }
                    }
                }
            }
            "namespace_import" => {
                // `* as name`
                names.push("*".to_string());
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Python relation extraction
// ---------------------------------------------------------------------------

fn extract_python_relations(
    _file_path: &str,
    source: &str,
    chunks: &[CodeChunk],
) -> Vec<CodeRelation> {
    let ts_language: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    let mut parser = Parser::new();
    if parser.set_language(&ts_language).is_err() {
        return vec![];
    }
    let Some(tree) = parser.parse(source, None) else {
        return vec![];
    };

    let mut relations = Vec::new();
    let root = tree.root_node();
    let mut cursor = root.walk();

    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();
            if !node.is_named() {
                if !cursor.goto_next_sibling() {
                    break;
                }
                continue;
            }

            let owner_idx = find_owning_chunk(chunks, node.start_position().row as u32 + 1);

            match node.kind() {
                "import_statement" => {
                    // `import os` / `import sys`
                    // name field(s) are dotted_name children
                    for i in 0..node.child_count() {
                        let Some(child) = node.child(i as u32) else {
                            continue;
                        };
                        if child.is_named() && (child.kind() == "dotted_name" || child.kind() == "aliased_import") {
                            let module_text = source[child.byte_range()].to_string();
                            // For aliased_import, extract the module part
                            let module_name = if child.kind() == "aliased_import" {
                                child
                                    .child_by_field_name("name")
                                    .map(|n| source[n.byte_range()].to_string())
                                    .unwrap_or(module_text)
                            } else {
                                module_text
                            };
                            relations.push(CodeRelation {
                                from_chunk_index: owner_idx,
                                relation: "imports".to_string(),
                                to_file: module_name.clone(),
                                to_name: Some(module_name),
                            });
                        }
                    }
                }
                "import_from_statement" => {
                    // `from pathlib import Path`
                    let module_path = node
                        .child_by_field_name("module_name")
                        .map(|m| source[m.byte_range()].to_string())
                        .unwrap_or_default();

                    // Collect imported names
                    let mut imported_names = Vec::new();
                    for i in 0..node.child_count() {
                        let Some(child) = node.child(i as u32) else {
                            continue;
                        };
                        let field = node.field_name_for_child(i as u32);
                        if field == Some("name") && child.is_named() {
                            match child.kind() {
                                "dotted_name" | "identifier" => {
                                    imported_names.push(source[child.byte_range()].to_string());
                                }
                                "aliased_import" => {
                                    if let Some(n) = child.child_by_field_name("name") {
                                        imported_names.push(source[n.byte_range()].to_string());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    if imported_names.is_empty() {
                        relations.push(CodeRelation {
                            from_chunk_index: owner_idx,
                            relation: "imports".to_string(),
                            to_file: module_path,
                            to_name: None,
                        });
                    } else {
                        for name in imported_names {
                            relations.push(CodeRelation {
                                from_chunk_index: owner_idx,
                                relation: "imports".to_string(),
                                to_file: module_path.clone(),
                                to_name: Some(name),
                            });
                        }
                    }
                }
                _ => {}
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    relations
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Find the chunk that "owns" a given line number.
/// For top-of-file imports that appear before any chunk, returns index 0.
fn find_owning_chunk(chunks: &[CodeChunk], line: u32) -> usize {
    // Find the chunk whose range contains this line
    for (i, chunk) in chunks.iter().enumerate() {
        if line >= chunk.start_line && line <= chunk.end_line {
            return i;
        }
    }
    // Default: first chunk (top-of-file imports before any chunk)
    0
}

/// Find the chunk that starts at (or closest to) a given line.
fn find_chunk_by_line(chunks: &[CodeChunk], line: u32) -> usize {
    // Exact match first
    for (i, chunk) in chunks.iter().enumerate() {
        if line >= chunk.start_line && line <= chunk.end_line {
            return i;
        }
    }
    // Fallback: find nearest chunk by start_line
    chunks
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| (c.start_line as i64 - line as i64).unsigned_abs())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

impl CodeChunk {
    /// Convert to a KnowledgeEntry (without embedding -- kernel adds that).
    pub fn to_knowledge_entry(&self, scope_id: &str, source_version: &str) -> KnowledgeEntry {
        KnowledgeEntry::new(
            self.content.clone(),
            scope_id.to_string(),
            source_version.to_string(),
        )
        .with_metadata(EntryMetadata {
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
            content: "fn test() {\n    42\n}".into(),
            file_path: "src/lib.rs".into(),
            language: "rs".into(),
            chunk_type: "function_item".into(),
            start_line: 1,
            end_line: 3,
        };
        let entry = chunk.to_knowledge_entry("my-repo", "abc123");
        assert_eq!(entry.scope_id, "my-repo");
        assert_eq!(entry.metadata.source_file.unwrap(), "src/lib.rs");
        assert_eq!(entry.metadata.language.unwrap(), "rs");
    }

    // -----------------------------------------------------------------------
    // Relation extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_use_imports() {
        let source = r#"
use crate::foo::Bar;
use std::collections::HashMap;
use super::baz;

fn do_stuff() {
    println!("hello");
}
"#;
        let result = chunk_file_with_relations("src/main.rs", source, "rs");
        assert!(!result.chunks.is_empty());

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

        // crate::foo::Bar → to_file should be the file itself (crate-internal), to_name = "Bar"
        let bar_import = imports.iter().find(|r| r.to_name.as_deref() == Some("Bar"));
        assert!(bar_import.is_some(), "Expected import of Bar");
        assert_eq!(bar_import.unwrap().to_file, "src/main.rs");

        // std::collections::HashMap → to_file = "std::collections", to_name = "HashMap"
        let hashmap_import = imports
            .iter()
            .find(|r| r.to_name.as_deref() == Some("HashMap"));
        assert!(hashmap_import.is_some(), "Expected import of HashMap");
        assert_eq!(hashmap_import.unwrap().to_file, "std::collections");

        // super::baz → to_file = file itself, to_name = "baz"
        let baz_import = imports.iter().find(|r| r.to_name.as_deref() == Some("baz"));
        assert!(baz_import.is_some(), "Expected import of baz");
        assert_eq!(baz_import.unwrap().to_file, "src/main.rs");
    }

    #[test]
    fn test_rust_use_list_imports() {
        let source = r#"
use std::collections::{HashMap, HashSet};

fn do_stuff() {
    println!("hello");
}
"#;
        let result = chunk_file_with_relations("src/lib.rs", source, "rs");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert_eq!(imports.len(), 2, "Expected 2 import relations for {{HashMap, HashSet}}");
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("HashMap")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("HashSet")));
        for imp in &imports {
            assert_eq!(imp.to_file, "std::collections");
        }
    }

    #[test]
    fn test_rust_wildcard_import() {
        let source = r#"
use super::*;

fn do_stuff() {
    println!("hello");
}
"#;
        let result = chunk_file_with_relations("src/lib.rs", source, "rs");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].to_name.as_deref(), Some("*"));
        // super is crate-internal
        assert_eq!(imports[0].to_file, "src/lib.rs");
    }

    #[test]
    fn test_rust_impl_trait_implements() {
        let source = r#"
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

impl MyStruct {
    fn new() -> Self {
        Self { field: 0 }
    }
}
"#;
        let result = chunk_file_with_relations("src/lib.rs", source, "rs");
        let implements: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "implements")
            .collect();
        assert_eq!(
            implements.len(),
            1,
            "Expected exactly 1 implements relation (impl Trait for Type), got {}",
            implements.len()
        );
        assert_eq!(implements[0].to_name.as_deref(), Some("MyTrait"));
        assert_eq!(implements[0].to_file, "src/lib.rs");

        // The plain `impl MyStruct` should NOT produce an implements relation
    }

    #[test]
    fn test_rust_mod_contains() {
        let source = r#"
mod inner {
    fn inner_fn() {
        let x = 1;
    }

    struct InnerStruct {
        val: u32,
    }
}
"#;
        let result = chunk_file_with_relations("src/lib.rs", source, "rs");
        let contains: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "contains")
            .collect();
        assert!(
            contains.len() >= 2,
            "Expected at least 2 contains relations (inner_fn, InnerStruct), got {}",
            contains.len()
        );
        let names: Vec<Option<&str>> = contains.iter().map(|r| r.to_name.as_deref()).collect();
        assert!(names.contains(&Some("inner_fn")));
        assert!(names.contains(&Some("InnerStruct")));
    }

    #[test]
    fn test_js_import_extraction() {
        let source = r#"
import { foo, bar } from './utils';
import defaultExport from 'module-name';
import * as name from 'module-name';

function doStuff() {
    return 42;
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        // { foo, bar } → 2 imports from './utils'
        // defaultExport → 1 import from 'module-name'
        // * as name → 1 import from 'module-name'
        assert!(
            imports.len() >= 4,
            "Expected at least 4 import relations, got {}",
            imports.len()
        );
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("foo")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("bar")));
        assert!(imports.iter().any(|r| r.to_file == "./utils"));
        assert!(imports.iter().any(|r| r.to_file == "module-name"));
    }

    #[test]
    fn test_ts_import_extraction() {
        let source = r#"
import { foo } from './utils';
import type { Bar } from './types';

function doStuff(): number {
    return 42;
}
"#;
        let result = chunk_file_with_relations("app.ts", source, "ts");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(
            imports.len() >= 2,
            "Expected at least 2 import relations, got {}",
            imports.len()
        );
        assert!(imports.iter().any(|r| r.to_file == "./utils"));
        assert!(imports.iter().any(|r| r.to_file == "./types"));
    }

    #[test]
    fn test_python_import_extraction() {
        let source = r#"
import os
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict

class MyClass:
    def method(self):
        pass
"#;
        let result = chunk_file_with_relations("app.py", source, "py");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        // import os, import sys → 2
        // from pathlib import Path → 1
        // from collections import defaultdict, OrderedDict → 2
        assert!(
            imports.len() >= 5,
            "Expected at least 5 import relations, got {}",
            imports.len()
        );
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("os")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("sys")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("Path")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("defaultdict")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("OrderedDict")));
    }

    #[test]
    fn test_python_relative_import() {
        let source = r#"
from . import utils
from ..core import engine

class MyClass:
    def method(self):
        pass
"#;
        let result = chunk_file_with_relations("pkg/module.py", source, "py");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(
            imports.len() >= 2,
            "Expected at least 2 import relations, got {}",
            imports.len()
        );
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("utils")));
        assert!(imports.iter().any(|r| r.to_name.as_deref() == Some("engine")));
    }

    #[test]
    fn test_empty_file_no_relations() {
        let source = "";
        let result = chunk_file_with_relations("empty.rs", source, "rs");
        assert!(result.relations.is_empty());
    }

    #[test]
    fn test_unsupported_language_no_relations() {
        let source = "some data\nmore data\n";
        let result = chunk_file_with_relations("data.txt", source, "txt");
        assert!(result.relations.is_empty());
        assert_eq!(result.chunks.len(), 1);
    }

    #[test]
    fn test_chunk_file_with_relations_backward_compat() {
        // Verify that chunk_file_with_relations produces the same chunks as chunk_file
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}
"#;
        let chunks_only = chunk_file("src/main.rs", source, "rs");
        let result = chunk_file_with_relations("src/main.rs", source, "rs");
        assert_eq!(chunks_only.len(), result.chunks.len());
        for (a, b) in chunks_only.iter().zip(result.chunks.iter()) {
            assert_eq!(a.content, b.content);
            assert_eq!(a.chunk_type, b.chunk_type);
            assert_eq!(a.start_line, b.start_line);
            assert_eq!(a.end_line, b.end_line);
        }
    }
}
