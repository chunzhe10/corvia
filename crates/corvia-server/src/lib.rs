//! HTTP server for the Corvia knowledge system.
//!
//! `corvia-server` exposes the kernel's capabilities over two protocols:
//!
//! - **REST** ([`rest`]) — Axum-based HTTP API on port `8020` with endpoints for
//!   knowledge ingestion, semantic search, graph queries, temporal history,
//!   reasoning, and agent coordination.
//! - **MCP** ([`mcp`]) — JSON-RPC 2.0 Model Context Protocol via Streamable HTTP
//!   transport (POST/GET/DELETE `/mcp`), providing tool-use access for AI agents.
//!
//! Both protocol handlers share a common `AppState` that holds trait objects
//! ([`QueryableStore`](corvia_kernel::traits::QueryableStore),
//! [`GraphStore`](corvia_kernel::traits::GraphStore), etc.), so the server
//! works identically with either storage tier.
//!
//! This crate is typically started via `corvia serve` rather than used as a
//! library directly. See the [README](https://github.com/corvia/corvia) and
//! [ARCHITECTURE.md](https://github.com/corvia/corvia/blob/master/ARCHITECTURE.md)
//! for details.

pub mod rest;
pub mod mcp;
pub mod dashboard;
