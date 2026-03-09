//! Git repository and source code ingestion adapter for Corvia.
//!
//! This crate implements the [`IngestionAdapter`](corvia_kernel::traits::IngestionAdapter)
//! trait from `corvia-kernel`, providing structured code ingestion powered by
//! [tree-sitter](https://tree-sitter.github.io/).
//!
//! # Supported Languages
//!
//! | Language | Extensions | Constructs |
//! |----------|-----------|------------|
//! | Rust | `.rs` | functions, structs, enums, impls, traits, modules |
//! | JavaScript | `.js`, `.jsx` | functions, classes, arrow functions, exports |
//! | TypeScript | `.ts`, `.tsx` | functions, classes, interfaces, type aliases |
//! | Python | `.py` | functions, classes, decorators, imports |
//! | Markdown | `.md` | headings, code blocks, sections |
//! | Config | `.toml`, `.yaml`, `.json` | key-value structures |
//!
//! # Structural Relations
//!
//! Beyond chunking, the adapter extracts [`CodeRelation`]s that capture
//! structural relationships (imports, implements, contains) for the kernel's
//! knowledge graph.
//!
//! # Usage
//!
//! ```rust,no_run
//! use corvia_adapter_git::GitAdapter;
//! use corvia_kernel::traits::IngestionAdapter;
//!
//! # async fn example() -> corvia_common::errors::Result<()> {
//! let adapter = GitAdapter::new();
//! let files = adapter.ingest_sources("/path/to/repo").await?;
//! // files are SourceFile values (content + metadata) —
//! // the kernel's ChunkingPipeline handles chunking, embedding, and storage.
//! # Ok(())
//! # }
//! ```
//!
//! See the main [Corvia repository](https://github.com/corvia/corvia) for the
//! full system architecture.

pub mod ast_chunker;
pub mod treesitter;
pub mod git;

pub use ast_chunker::AstChunker;
pub use git::{GitAdapter, IngestionResult};
pub use treesitter::{CodeRelation, ChunkResult};
