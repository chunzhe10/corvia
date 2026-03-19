//! Shared types for the Corvia knowledge system.
//!
//! `corvia-common` provides the foundational types that every crate in the Corvia
//! workspace depends on:
//!
//! - [`types`] — [`KnowledgeEntry`](types::KnowledgeEntry), [`SearchResult`](types::SearchResult),
//!   [`GraphEdge`](types::GraphEdge), and other core data structures
//! - [`config`] — [`CorviaConfig`](config::CorviaConfig) loaded from `corvia.toml`
//! - [`namespace`] — Five-segment namespace model (`org:scope:workstream:source:version`)
//! - [`errors`] — Unified error types and [`Result`](errors::Result) alias
//! - [`events`] — Domain events for inter-subsystem communication
//! - [`agent_types`] — Agent identity, session state, and coordination types
//!
//! This crate is intentionally lightweight — it carries no runtime dependencies on
//! storage engines or inference providers, so external adapters can depend on it
//! without pulling in the full kernel.
//!
//! See the [project architecture](https://github.com/corvia/corvia/blob/master/ARCHITECTURE.md)
//! for how `corvia-common` fits into the layered kernel design.

pub mod types;
pub mod namespace;
pub mod config;
pub mod constants;
pub mod errors;
pub mod events;
pub mod agent_types;
pub mod dashboard;
