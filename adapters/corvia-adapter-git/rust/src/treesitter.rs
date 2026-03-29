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
                    "function_item" => {
                        extract_rust_calls(&mut relations, source, &node, file_path, chunks);
                    }
                    "impl_item" => {
                        extract_rust_impl(&mut relations, source, &node, file_path, chunks);
                        extract_rust_calls(&mut relations, source, &node, file_path, chunks);
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

/// Extract "calls" relations from function/method call sites in Rust code.
///
/// Walks AST nodes inside function/impl bodies looking for `call_expression` nodes.
/// Emits "calls" relations for non-trivial function calls (skips std/common trait methods).
fn extract_rust_calls(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    file_path: &str,
    chunks: &[CodeChunk],
) {
    // Deny list: common trait methods that would create edge explosion
    const DENY_LIST: &[&str] = &[
        "new", "default", "from", "into", "clone", "fmt", "drop",
        "to_string", "to_owned", "as_ref", "as_mut", "deref", "borrow",
        "unwrap", "expect", "ok", "err", "map", "and_then", "or_else",
        "iter", "collect", "push", "len", "is_empty",
    ];

    let owner_line = node.start_position().row as u32 + 1;
    let owner_idx = find_chunk_by_line(chunks, owner_line);

    // Recursively collect call expressions
    let mut call_nodes = Vec::new();
    collect_call_expressions(node, &mut call_nodes, 0);

    for call_node in &call_nodes {
        // Extract the function part of the call expression
        let Some(func_node) = call_node.child_by_field_name("function") else {
            continue;
        };

        let func_text = source[func_node.byte_range()].to_string();

        // Skip standard library calls
        if func_text.starts_with("std::") || func_text.starts_with("core::") || func_text.starts_with("alloc::") {
            continue;
        }

        // Extract the final function/method name
        let fn_name = func_text.rsplit("::").next().unwrap_or(&func_text);
        // For method calls like `self.foo()`, the function node is a field_expression
        let fn_name = fn_name.rsplit('.').next().unwrap_or(fn_name);

        // Skip deny-listed common methods
        if DENY_LIST.contains(&fn_name) {
            continue;
        }
        // Skip single-char or empty names (noise)
        if fn_name.len() <= 1 {
            continue;
        }

        // Determine target file
        let to_file = if func_text.contains("crate::") || func_text.contains("super::") {
            resolve_rust_module_path(file_path, &func_text.rsplit("::").skip(1).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("::"))
        } else if func_text.contains("::") {
            // Qualified path — use the module part
            let parts: Vec<&str> = func_text.rsplitn(2, "::").collect();
            if parts.len() == 2 {
                resolve_rust_module_path(file_path, parts[1])
            } else {
                file_path.to_string()
            }
        } else {
            // Unqualified call — likely in the same file or imported
            file_path.to_string()
        };

        relations.push(CodeRelation {
            from_chunk_index: owner_idx,
            relation: "calls".to_string(),
            to_file,
            to_name: Some(fn_name.to_string()),
        });
    }
}

/// Recursively collect `call_expression` nodes from the AST, with depth limit.
fn collect_call_expressions<'a>(node: &Node<'a>, results: &mut Vec<Node<'a>>, depth: usize) {
    if depth > 20 {
        return;
    }
    if node.kind() == "call_expression" {
        results.push(*node);
    }
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_call_expressions(&cursor.node(), results, depth + 1);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

/// Infer the crate `src/` root from a file path.
///
/// Given `crates/corvia-kernel/src/chunking_strategy.rs`, returns
/// `Some("crates/corvia-kernel/src")`.
/// Given `src/main.rs`, returns `Some("src")`.
fn infer_crate_src_root(file_path: &str) -> Option<&str> {
    // Find the last `/src/` segment — everything up to and including `src`
    if let Some(idx) = file_path.rfind("/src/") {
        Some(&file_path[..idx + 4]) // include "/src"
    } else if file_path.starts_with("src/") {
        Some("src")
    } else {
        None
    }
}

/// Best-effort resolution for Rust module paths.
///
/// - `crate::foo::bar` → `CRATE_REF:<src_root>:foo::bar` so the wiring step
///   can resolve `foo::bar` to `<src_root>/foo/bar.rs` or `<src_root>/foo.rs`.
/// - `super::foo` → resolve relative to the source file's parent directory.
/// - `std::*` / external → record as-is (no match expected).
fn resolve_rust_module_path(file_path: &str, module_path: &str) -> String {
    if module_path.starts_with("crate::") || module_path == "crate" {
        let mod_suffix = module_path.strip_prefix("crate::").unwrap_or("");
        if let Some(root) = infer_crate_src_root(file_path) {
            format!("CRATE_REF:{root}:{mod_suffix}")
        } else {
            // Can't infer root — fall back to source file
            file_path.to_string()
        }
    } else if module_path.starts_with("super::") || module_path == "super" {
        let mod_suffix = module_path.strip_prefix("super::").unwrap_or("");
        // Resolve relative to parent directory of the source file
        let parent = file_path
            .rsplit_once('/')
            .map(|(dir, _)| dir)
            .unwrap_or("");
        // Go up one more directory for `super`
        let grandparent = parent
            .rsplit_once('/')
            .map(|(dir, _)| dir)
            .unwrap_or("");
        if mod_suffix.is_empty() {
            // `use super::*` — point at parent module
            if grandparent.is_empty() {
                file_path.to_string()
            } else {
                format!("CRATE_REF:{grandparent}:")
            }
        } else {
            // `use super::foo` → try `<grandparent>/foo.rs`
            if grandparent.is_empty() {
                format!("CRATE_REF::{mod_suffix}")
            } else {
                format!("CRATE_REF:{grandparent}:{mod_suffix}")
            }
        }
    } else {
        // External crate or std — record the module path itself
        module_path.to_string()
    }
}

// ---------------------------------------------------------------------------
// JavaScript/TypeScript relation extraction
// ---------------------------------------------------------------------------

/// JS/TS deny-list: common methods that would create edge explosion.
const JS_TS_CALL_DENY_LIST: &[&str] = &[
    "log", "warn", "error", "info", "debug",                   // console
    "then", "catch", "finally",                                 // promises
    "map", "filter", "reduce", "forEach", "find", "some",      // array
    "push", "pop", "shift", "unshift", "splice", "slice",      // array mutation
    "toString", "valueOf", "hasOwnProperty",                    // object
    "stringify", "parse",                                       // JSON
    "createElement",                                            // React DOM
    "resolve", "reject",                                        // Promise
    "require",                                                  // CJS
];

fn extract_js_ts_relations(
    file_path: &str,
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

    // Pass 1: Walk top-level children for imports and class declarations
    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();
            if node.is_named() {
                match node.kind() {
                    "import_statement" => {
                        extract_js_ts_import(&mut relations, source, &node, chunks);
                    }
                    "class_declaration" => {
                        extract_js_ts_class_heritage(
                            &mut relations, source, &node, file_path, chunks,
                        );
                    }
                    "export_statement" => {
                        // Check for exported class: `export class Foo extends Bar {}`
                        for i in 0..node.child_count() {
                            if let Some(child) = node.child(i as u32) {
                                if child.kind() == "class_declaration" || child.kind() == "class" {
                                    extract_js_ts_class_heritage(
                                        &mut relations, source, &child, file_path, chunks,
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    // Pass 2: Recursive walk for call expressions (catches nested/arrow/method calls)
    extract_js_ts_calls(&mut relations, source, &root, file_path, chunks);

    relations
}

/// Extract import relations from a JS/TS `import_statement` node.
fn extract_js_ts_import(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    chunks: &[CodeChunk],
) {
    let owner_idx = find_owning_chunk(chunks, node.start_position().row as u32 + 1);
    if let Some(source_node) = node.child_by_field_name("source") {
        let raw = source[source_node.byte_range()].to_string();
        let import_path = raw.trim_matches(|c| c == '\'' || c == '"').to_string();

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

/// Extract `extends` and `implements` relations from a JS/TS class declaration.
///
/// Handles both JavaScript (`class_heritage` with direct expression) and TypeScript
/// (`class_heritage` with `extends_clause`/`implements_clause` children).
/// Strips generic type arguments: `Bar<T>` -> `"Bar"`.
fn extract_js_ts_class_heritage(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    node: &Node,
    file_path: &str,
    chunks: &[CodeChunk],
) {
    let owner_idx = find_owning_chunk(chunks, node.start_position().row as u32 + 1);

    // Walk children to find `class_heritage` node (it's a node kind, not a field)
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        if child.kind() != "class_heritage" {
            continue;
        }

        // Check for TypeScript extends_clause / implements_clause children
        let mut found_ts_clauses = false;
        for j in 0..child.child_count() {
            let Some(clause) = child.child(j as u32) else {
                continue;
            };
            match clause.kind() {
                "extends_clause" => {
                    found_ts_clauses = true;
                    // Extract the type expression (first named child that is a type/identifier)
                    if let Some(name) = extract_extends_type_name(source, &clause) {
                        relations.push(CodeRelation {
                            from_chunk_index: owner_idx,
                            relation: "extends".to_string(),
                            to_file: file_path.to_string(),
                            to_name: Some(name),
                        });
                    }
                }
                "implements_clause" => {
                    found_ts_clauses = true;
                    // Each type in the implements list is a separate relation
                    for k in 0..clause.child_count() {
                        if let Some(type_node) = clause.child(k as u32) {
                            if !type_node.is_named() {
                                continue;
                            }
                            if let Some(name) = extract_type_identifier(source, &type_node) {
                                relations.push(CodeRelation {
                                    from_chunk_index: owner_idx,
                                    relation: "implements".to_string(),
                                    to_file: file_path.to_string(),
                                    to_name: Some(name),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // JavaScript: class_heritage directly contains the base class expression
        // (no extends_clause/implements_clause wrapper)
        if !found_ts_clauses {
            // The heritage node's children include the base class expression
            for j in 0..child.child_count() {
                if let Some(expr) = child.child(j as u32) {
                    if !expr.is_named() {
                        continue;
                    }
                    if let Some(name) = extract_type_identifier(source, &expr) {
                        relations.push(CodeRelation {
                            from_chunk_index: owner_idx,
                            relation: "extends".to_string(),
                            to_file: file_path.to_string(),
                            to_name: Some(name),
                        });
                        break; // JS only has single inheritance
                    }
                }
            }
        }
    }
}

/// Extract the base class name from a TypeScript `extends_clause`.
/// Handles generics: `extends Bar<T>` -> "Bar".
fn extract_extends_type_name(source: &str, clause: &Node) -> Option<String> {
    for i in 0..clause.child_count() {
        let child = clause.child(i as u32)?;
        if !child.is_named() {
            continue;
        }
        return extract_type_identifier(source, &child);
    }
    None
}

/// Extract a type identifier name, stripping generic parameters.
/// `Bar` -> "Bar", `Bar<T>` -> "Bar" (via generic_type's name field),
/// `member_expression` -> last segment.
fn extract_type_identifier(source: &str, node: &Node) -> Option<String> {
    match node.kind() {
        "identifier" | "type_identifier" => {
            Some(source[node.byte_range()].to_string())
        }
        "generic_type" => {
            // Extract the name part before type arguments
            node.child_by_field_name("name")
                .map(|n| source[n.byte_range()].to_string())
        }
        "member_expression" => {
            // e.g., `module.ClassName` -> extract "ClassName"
            node.child_by_field_name("property")
                .map(|p| source[p.byte_range()].to_string())
        }
        _ => {
            // Fallback: try raw text, strip anything after '<'
            let text = source[node.byte_range()].to_string();
            let name = text.split('<').next().unwrap_or(&text).trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        }
    }
}

/// Extract "calls" relations from JS/TS code via recursive AST walk.
///
/// Walks the entire AST tree to catch calls in arrow functions, nested functions,
/// class methods, and other nested scopes. Handles dynamic `import()` as an
/// "imports" relation.
fn extract_js_ts_calls(
    relations: &mut Vec<CodeRelation>,
    source: &str,
    root: &Node,
    file_path: &str,
    chunks: &[CodeChunk],
) {
    let mut call_nodes = Vec::new();
    collect_call_expressions(root, &mut call_nodes, 0);

    for call_node in &call_nodes {
        let Some(func_node) = call_node.child_by_field_name("function") else {
            continue;
        };

        let func_text = source[func_node.byte_range()].to_string();

        // Dynamic import(): `import('./module')` -> emit "imports" not "calls"
        if func_text == "import" {
            if let Some(args) = call_node.child_by_field_name("arguments") {
                for i in 0..args.child_count() {
                    if let Some(arg) = args.child(i as u32) {
                        if arg.kind() == "string" || arg.kind() == "template_string" {
                            let raw = source[arg.byte_range()].to_string();
                            let path = raw.trim_matches(|c| c == '\'' || c == '"' || c == '`');
                            let owner_idx = find_owning_chunk(
                                chunks,
                                call_node.start_position().row as u32 + 1,
                            );
                            relations.push(CodeRelation {
                                from_chunk_index: owner_idx,
                                relation: "imports".to_string(),
                                to_file: path.to_string(),
                                to_name: None,
                            });
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // Extract the callable name
        let fn_name = extract_js_call_name(source, &func_node);

        // Skip deny-listed methods
        if JS_TS_CALL_DENY_LIST.contains(&fn_name.as_str()) {
            continue;
        }
        // Skip single-char or empty names
        if fn_name.len() <= 1 {
            continue;
        }

        let owner_idx = find_owning_chunk(chunks, call_node.start_position().row as u32 + 1);

        relations.push(CodeRelation {
            from_chunk_index: owner_idx,
            relation: "calls".to_string(),
            to_file: file_path.to_string(),
            to_name: Some(fn_name),
        });
    }
}

/// Extract the function/method name from a JS/TS call expression's function node.
/// Handles: simple identifiers, member expressions (`this.foo`, `obj.bar.baz`),
/// and optional chaining (`obj?.method`).
fn extract_js_call_name(source: &str, node: &Node) -> String {
    match node.kind() {
        "identifier" => source[node.byte_range()].to_string(),
        "member_expression" | "optional_chain_expression" => {
            // Extract the last property: `this.foo` -> "foo", `a.b.c` -> "c"
            node.child_by_field_name("property")
                .map(|p| source[p.byte_range()].to_string())
                .unwrap_or_else(|| {
                    // Fallback: last segment after '.'
                    let text = source[node.byte_range()].to_string();
                    text.rsplit('.').next().unwrap_or(&text).to_string()
                })
        }
        _ => {
            // Fallback: try last segment
            let text = source[node.byte_range()].to_string();
            text.rsplit('.').next().unwrap_or(&text).to_string()
        }
    }
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
            ..Default::default()
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

        // crate::foo::Bar → to_file should be CRATE_REF with module path
        let bar_import = imports.iter().find(|r| r.to_name.as_deref() == Some("Bar"));
        assert!(bar_import.is_some(), "Expected import of Bar");
        assert_eq!(bar_import.unwrap().to_file, "CRATE_REF:src:foo");

        // std::collections::HashMap → to_file = "std::collections", to_name = "HashMap"
        let hashmap_import = imports
            .iter()
            .find(|r| r.to_name.as_deref() == Some("HashMap"));
        assert!(hashmap_import.is_some(), "Expected import of HashMap");
        assert_eq!(hashmap_import.unwrap().to_file, "std::collections");

        // super::baz from src/main.rs — super can't resolve above crate root, fallback
        let baz_import = imports.iter().find(|r| r.to_name.as_deref() == Some("baz"));
        assert!(baz_import.is_some(), "Expected import of baz");
        // Falls back to file_path since there's no grandparent above src/
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
        // super from src/lib.rs can't resolve above crate root, fallback
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
    fn test_rust_crate_ref_deep_path() {
        let source = r#"
use crate::kernel::graph::GraphStore;
use super::utils::Helper;

fn handler() {
    println!("hi");
}
"#;
        let result = chunk_file_with_relations(
            "crates/corvia-server/src/routes/api.rs",
            source,
            "rs",
        );
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(imports.len() >= 2, "Expected at least 2 import relations");

        // crate::kernel::graph → CRATE_REF with inferred src root
        let graph_import = imports
            .iter()
            .find(|r| r.to_name.as_deref() == Some("GraphStore"));
        assert!(graph_import.is_some());
        assert_eq!(
            graph_import.unwrap().to_file,
            "CRATE_REF:crates/corvia-server/src:kernel::graph"
        );

        // super::utils from routes/api.rs → CRATE_REF with grandparent
        let helper_import = imports
            .iter()
            .find(|r| r.to_name.as_deref() == Some("Helper"));
        assert!(helper_import.is_some());
        assert_eq!(
            helper_import.unwrap().to_file,
            "CRATE_REF:crates/corvia-server/src:utils"
        );
    }

    #[test]
    fn test_infer_crate_src_root() {
        assert_eq!(
            infer_crate_src_root("crates/corvia-kernel/src/chunking_strategy.rs"),
            Some("crates/corvia-kernel/src")
        );
        assert_eq!(infer_crate_src_root("src/main.rs"), Some("src"));
        assert_eq!(infer_crate_src_root("lib.rs"), None);
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

    #[test]
    fn test_rust_call_extraction() {
        let source = r#"
fn main() {
    helper();
    module::other(42);
    let x = compute(1, 2);
}
"#;
        let result = chunk_file_with_relations("src/main.rs", source, "rs");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        assert!(
            calls.len() >= 2,
            "Expected at least 2 'calls' relations (helper, other, compute), got {}",
            calls.len()
        );
        let names: Vec<Option<&str>> = calls.iter().map(|r| r.to_name.as_deref()).collect();
        assert!(names.contains(&Some("helper")));
        assert!(names.contains(&Some("other")));
    }

    #[test]
    fn test_rust_call_skips_std_and_common() {
        let source = r#"
fn process() {
    let v = Vec::new();
    let s = String::from("hello");
    let c = s.clone();
    let d = Default::default();
    std::io::stdout();
    core::mem::drop(c);
}
"#;
        let result = chunk_file_with_relations("src/lib.rs", source, "rs");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        // All of these should be filtered: new, from, clone, default, std::*, core::*
        for call in &calls {
            let name = call.to_name.as_deref().unwrap_or("");
            assert!(
                !["new", "from", "clone", "default", "drop"].contains(&name),
                "should have filtered common method '{name}'"
            );
        }
    }

    #[test]
    fn test_rust_call_resolves_crate_path() {
        let source = r#"
fn handler() {
    crate::foo::process();
}
"#;
        let result = chunk_file_with_relations(
            "crates/corvia-server/src/routes/api.rs",
            source,
            "rs",
        );
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        assert!(
            !calls.is_empty(),
            "Expected at least 1 'calls' relation for crate::foo::process()"
        );
        let process_call = calls.iter().find(|r| r.to_name.as_deref() == Some("process"));
        assert!(process_call.is_some(), "Expected call to 'process'");
        assert!(
            process_call.unwrap().to_file.starts_with("CRATE_REF:"),
            "crate:: call should resolve to CRATE_REF, got: {}",
            process_call.unwrap().to_file
        );
    }

    // -----------------------------------------------------------------------
    // P1: JS/TS class inheritance and function call tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_js_class_extends() {
        let source = r#"
class Animal {
    constructor(name) {
        this.name = name;
    }
}

class Dog extends Animal {
    bark() {
        return "woof";
    }
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let extends: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "extends")
            .collect();
        assert_eq!(extends.len(), 1, "Expected 1 extends relation, got {}", extends.len());
        assert_eq!(extends[0].to_name.as_deref(), Some("Animal"));
    }

    #[test]
    fn test_ts_class_extends_with_generics() {
        let source = r#"
class Base<T> {
    value: T;
    constructor(val: T) {
        this.value = val;
    }
}

class Derived extends Base<string> {
    greet(): string {
        return this.value;
    }
}
"#;
        let result = chunk_file_with_relations("app.ts", source, "ts");
        let extends: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "extends")
            .collect();
        assert_eq!(extends.len(), 1, "Expected 1 extends relation");
        assert_eq!(
            extends[0].to_name.as_deref(),
            Some("Base"),
            "Should extract 'Base' not 'Base<string>'"
        );
    }

    #[test]
    fn test_ts_class_implements() {
        let source = r#"
interface ISerializable {
    serialize(): string;
}

interface ICloneable {
    clone(): ICloneable;
}

class MyClass implements ISerializable, ICloneable {
    serialize(): string {
        return "{}";
    }
    clone(): ICloneable {
        return new MyClass();
    }
}
"#;
        let result = chunk_file_with_relations("app.ts", source, "ts");
        let implements: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "implements")
            .collect();
        assert!(
            implements.len() >= 2,
            "Expected at least 2 implements relations (ISerializable, ICloneable), got {}",
            implements.len()
        );
        let names: Vec<&str> = implements.iter().filter_map(|r| r.to_name.as_deref()).collect();
        assert!(names.contains(&"ISerializable"), "Missing ISerializable, got: {:?}", names);
        assert!(names.contains(&"ICloneable"), "Missing ICloneable, got: {:?}", names);
    }

    #[test]
    fn test_js_function_call_in_method() {
        let source = r#"
function helper() {
    return 42;
}

class Service {
    process() {
        const result = helper();
        return result;
    }
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        let names: Vec<&str> = calls.iter().filter_map(|r| r.to_name.as_deref()).collect();
        assert!(
            names.contains(&"helper"),
            "Expected call to 'helper' inside method body, got: {:?}",
            names
        );
    }

    #[test]
    fn test_js_call_in_arrow_function() {
        let source = r#"
const handler = () => {
    processData();
    return true;
};
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        let names: Vec<&str> = calls.iter().filter_map(|r| r.to_name.as_deref()).collect();
        assert!(
            names.contains(&"processData"),
            "Expected call to 'processData' in arrow function, got: {:?}",
            names
        );
    }

    #[test]
    fn test_js_deny_listed_calls_skipped() {
        let source = r#"
function doStuff() {
    console.log("hello");
    arr.map(x => x + 1);
    promise.then(r => r);
    arr.push(1);
    JSON.stringify({});
    return true;
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        for call in &calls {
            let name = call.to_name.as_deref().unwrap_or("");
            assert!(
                !["log", "map", "then", "push", "stringify"].contains(&name),
                "Deny-listed method '{name}' should have been filtered"
            );
        }
    }

    #[test]
    fn test_js_this_method_call() {
        let source = r#"
class MyClass {
    helperMethod() {
        return 42;
    }

    process() {
        this.helperMethod();
        return true;
    }
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        let names: Vec<&str> = calls.iter().filter_map(|r| r.to_name.as_deref()).collect();
        assert!(
            names.contains(&"helperMethod"),
            "Should extract 'helperMethod' from this.helperMethod(), got: {:?}",
            names
        );
        assert!(
            !names.contains(&"this"),
            "'this' should not appear as a call target"
        );
    }

    #[test]
    fn test_js_dynamic_import() {
        let source = r#"
async function loadModule() {
    const mod = await import('./lazy-module');
    return mod;
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(
            imports.iter().any(|r| r.to_file == "./lazy-module"),
            "Dynamic import('./lazy-module') should produce an imports relation, got: {:?}",
            imports.iter().map(|r| &r.to_file).collect::<Vec<_>>()
        );

        // Should NOT produce a "calls" relation for import()
        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls" && r.to_name.as_deref() == Some("import"))
            .collect();
        assert!(calls.is_empty(), "import() should not produce a 'calls' relation");
    }

    #[test]
    fn test_js_syntax_error_partial_extraction() {
        let source = r#"
class Good extends Base {
    method() {
        return 1;
    }
}

// Syntax error below
const broken = {{{;

class AlsoGood extends Other {
    run() {
        return 2;
    }
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        // tree-sitter is error-tolerant; should still extract from valid parts
        let extends: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "extends")
            .collect();
        assert!(
            !extends.is_empty(),
            "Should extract at least some extends relations despite syntax error"
        );
    }

    #[test]
    fn test_js_empty_class_no_crash() {
        let source = r#"
class EmptyClass {
}
"#;
        let result = chunk_file_with_relations("app.js", source, "js");
        // Should not crash, may or may not produce chunks (class has no body methods)
        // Key assertion: no panic
        let _ = result.relations;
    }

    #[test]
    fn test_js_minified_single_line() {
        let source = "class A extends B{constructor(){super();this.init()}}class C extends D{run(){return compute()}}";
        let result = chunk_file_with_relations("app.min.js", source, "js");
        let extends: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "extends")
            .collect();
        assert!(
            extends.len() >= 2,
            "Minified JS should still extract extends relations, got {}",
            extends.len()
        );
    }

    #[test]
    fn test_ts_exported_class_extends() {
        let source = r#"
export class Controller extends BaseController {
    handle() {
        return "ok";
    }
}
"#;
        let result = chunk_file_with_relations("controller.ts", source, "ts");
        let extends: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "extends")
            .collect();
        assert_eq!(extends.len(), 1, "Exported class should have extends relation");
        assert_eq!(extends[0].to_name.as_deref(), Some("BaseController"));
    }

    #[test]
    fn test_existing_rust_extraction_unchanged() {
        // Regression: existing Rust extraction must still work
        let source = r#"
use crate::foo::Bar;

pub trait MyTrait {
    fn do_thing(&self);
}

impl MyTrait for MyStruct {
    fn do_thing(&self) {
        println!("hello");
    }
}

fn main() {
    helper();
}
"#;
        let result = chunk_file_with_relations("src/main.rs", source, "rs");
        let imports: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "imports")
            .collect();
        assert!(!imports.is_empty(), "Rust imports should still work");

        let implements: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "implements")
            .collect();
        assert!(!implements.is_empty(), "Rust implements should still work");

        let calls: Vec<&CodeRelation> = result
            .relations
            .iter()
            .filter(|r| r.relation == "calls")
            .collect();
        assert!(!calls.is_empty(), "Rust calls should still work");
    }
}
