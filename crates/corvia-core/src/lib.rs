//! corvia-core: Knowledge storage, retrieval, and embedding engine.
//!
//! Provides the core pipeline: ingest -> chunk -> embed -> index -> search -> rerank.
//! Uses tantivy (BM25), redb (vector storage), and fastembed (embedding + reranking).

pub mod discover;
pub mod types;
pub mod config;
pub mod entry;
pub mod chunk;
pub mod embed;
pub mod index;
pub mod tantivy_index;
pub mod ingest;
pub mod search;
pub mod trace;
pub mod write;
pub mod init;
