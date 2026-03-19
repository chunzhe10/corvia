//! Well-known string constants shared across the Corvia workspace.
//!
//! Centralising magic strings here eliminates typo risk and makes
//! rename-refactors a single-site change.

/// Scope ID for user-history entries (Claude session ingestion).
pub const USER_HISTORY_SCOPE: &str = "user-history";

/// Adapter domain identifier for the Claude-sessions adapter.
pub const CLAUDE_SESSIONS_ADAPTER: &str = "claude-sessions";

/// Fallback scope ID used when no explicit scope is configured.
pub const DEFAULT_SCOPE_ID: &str = "corvia";
