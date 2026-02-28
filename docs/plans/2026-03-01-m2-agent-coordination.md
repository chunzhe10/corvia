# M2: Agent Coordination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multiple AI agents to concurrently write knowledge through isolated staging, merge into a shared knowledge base, and survive crashes — all exposed via REST and MCP.

**Architecture:** Agent coordination uses a separate Redb database (`.corvia/coordination.redb`) for agent/session/queue metadata, independent of the knowledge store (LiteStore or SurrealStore). Agents write to staging directories with git branches for audit trail, sharing the HNSW index for instant cross-agent search. A single-threaded merge worker processes a Redb queue, using semantic similarity for conflict detection and LLM-assisted resolution via Ollama's chat API.

**Tech Stack:** Rust (2024 edition), Redb 2 (coordination), tokio (async), axum (REST), git CLI (branch ops), Ollama chat API (LLM merge), serde (JSON-RPC for MCP)

**Open questions resolved for this plan:**
- Conflict detection: cosine similarity > 0.85 within same scope = conflict
- McpClient with `_meta`: auto-registered as lightweight agent records
- LLM merge: uses Ollama `/api/chat` with a configurable chat model (separate from embedding model)

---

## Task 1: Agent & Session Types

**Files:**
- Create: `crates/corvia-common/src/agent_types.rs`
- Modify: `crates/corvia-common/src/lib.rs`
- Modify: `crates/corvia-common/src/types.rs`
- Modify: `crates/corvia-common/src/config.rs`
- Modify: `crates/corvia-common/src/errors.rs`

### Step 1: Write failing tests for agent types

Add to end of a new test module in `crates/corvia-common/src/agent_types.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registered_agent_id_format() {
        let id = AgentIdentity::Registered {
            agent_id: "myproject::code-indexer".into(),
            api_key: None,
        };
        assert_eq!(id.effective_agent_id(), "myproject::code-indexer");
    }

    #[test]
    fn test_mcp_client_with_hint() {
        let id = AgentIdentity::McpClient {
            client_name: "crewai".into(),
            client_version: "0.3".into(),
            agent_hint: Some("financial-advisor".into()),
        };
        assert_eq!(id.effective_agent_id(), "crewai::financial-advisor");
        assert!(id.can_write());
    }

    #[test]
    fn test_mcp_client_without_hint_is_read_only() {
        let id = AgentIdentity::McpClient {
            client_name: "langgraph".into(),
            client_version: "0.2".into(),
            agent_hint: None,
        };
        assert_eq!(id.effective_agent_id(), "langgraph");
        assert!(!id.can_write());
    }

    #[test]
    fn test_anonymous_is_read_only() {
        let id = AgentIdentity::Anonymous;
        assert!(!id.can_write());
    }

    #[test]
    fn test_session_state_transitions() {
        assert!(SessionState::Created.can_transition_to(SessionState::Active));
        assert!(SessionState::Active.can_transition_to(SessionState::Committing));
        assert!(SessionState::Active.can_transition_to(SessionState::Stale));
        assert!(!SessionState::Closed.can_transition_to(SessionState::Active));
        assert!(!SessionState::Created.can_transition_to(SessionState::Merging));
    }

    #[test]
    fn test_entry_status_default_is_merged() {
        let status = EntryStatus::default();
        assert_eq!(status, EntryStatus::Merged);
    }

    #[test]
    fn test_agent_record_serialization() {
        let record = AgentRecord {
            agent_id: "test::agent".into(),
            display_name: "Test Agent".into(),
            identity_type: IdentityType::Registered,
            registered_at: chrono::Utc::now(),
            permissions: AgentPermission::ReadWrite {
                scopes: vec!["test".into()],
            },
            last_seen: chrono::Utc::now(),
            status: AgentStatus::Active,
        };
        let json = serde_json::to_string(&record).unwrap();
        let deser: AgentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.agent_id, deser.agent_id);
    }

    #[test]
    fn test_visibility_mode_default() {
        let mode = VisibilityMode::default();
        assert_eq!(mode, VisibilityMode::Own);
    }

    #[test]
    fn test_session_record_serialization() {
        let record = SessionRecord {
            session_id: "test::agent/sess-12345678".into(),
            agent_id: "test::agent".into(),
            created_at: chrono::Utc::now(),
            last_heartbeat: chrono::Utc::now(),
            state: SessionState::Active,
            git_branch: Some("test-agent/sess-12345678".into()),
            staging_dir: Some(".corvia/staging/test-agent".into()),
            entries_written: 0,
            entries_merged: 0,
        };
        let json = serde_json::to_string(&record).unwrap();
        let deser: SessionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.session_id, deser.session_id);
    }
}
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p corvia-common -- agent_types`
Expected: Compilation error — module and types don't exist yet.

### Step 3: Implement agent types

Create `crates/corvia-common/src/agent_types.rs`:

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Agent identity — multi-layer per D45 Part 1.
/// Corvia accepts identity at multiple layers for ecosystem compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentIdentity {
    /// Internal: registered Corvia agent with persistent ID.
    /// Format: "{scope}::{agent-name}", e.g., "myproject::code-indexer"
    Registered {
        agent_id: String,
        api_key: Option<String>,
    },

    /// External via MCP: identified by client info + optional _meta.
    /// Used by CrewAI, LangGraph, OpenAI Agents SDK, Claude Code, etc.
    McpClient {
        client_name: String,
        client_version: String,
        agent_hint: Option<String>,
    },

    /// Anonymous: no identity, read-only access.
    Anonymous,
}

impl AgentIdentity {
    /// Returns the effective agent ID used for knowledge ownership.
    /// Registered: "{scope}::{agent-name}"
    /// McpClient with hint: "{client_name}::{agent_hint}"
    /// McpClient without hint: "{client_name}"
    /// Anonymous: "anonymous"
    pub fn effective_agent_id(&self) -> String {
        match self {
            Self::Registered { agent_id, .. } => agent_id.clone(),
            Self::McpClient {
                client_name,
                agent_hint: Some(hint),
                ..
            } => format!("{client_name}::{hint}"),
            Self::McpClient { client_name, .. } => client_name.clone(),
            Self::Anonymous => "anonymous".into(),
        }
    }

    /// Whether this identity type has write access.
    /// Registered: yes. McpClient with _meta.agent_id: yes. Others: no.
    pub fn can_write(&self) -> bool {
        match self {
            Self::Registered { .. } => true,
            Self::McpClient {
                agent_hint: Some(_),
                ..
            } => true,
            _ => false,
        }
    }

    /// Whether this identity gets full D43 staging hybrid (staging dir + git branch).
    /// Only Registered agents get full staging.
    pub fn has_staging(&self) -> bool {
        matches!(self, Self::Registered { .. })
    }
}

/// Which identity type an agent record was created from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IdentityType {
    Registered,
    McpClient,
}

/// Permission model per D45 Part 1.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentPermission {
    ReadOnly,
    ReadWrite { scopes: Vec<String> },
    Admin,
}

impl AgentPermission {
    pub fn can_write_scope(&self, scope_id: &str) -> bool {
        match self {
            Self::ReadOnly => false,
            Self::ReadWrite { scopes } => scopes.iter().any(|s| s == scope_id || s == "*"),
            Self::Admin => true,
        }
    }
}

/// Agent lifecycle status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Suspended,
    Deregistered,
}

/// Persistent agent record stored in Redb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub agent_id: String,
    pub display_name: String,
    pub identity_type: IdentityType,
    pub registered_at: DateTime<Utc>,
    pub permissions: AgentPermission,
    pub last_seen: DateTime<Utc>,
    pub status: AgentStatus,
}

/// Session state machine per D45 Part 5.
/// Created → Active → Committing → Merging → Closed (happy path)
/// Active → Stale → Orphaned → Recoverable
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SessionState {
    Created,
    Active,
    Committing,
    Merging,
    Closed,
    Stale,
    Orphaned,
}

impl SessionState {
    pub fn can_transition_to(self, next: Self) -> bool {
        use SessionState::*;
        matches!(
            (self, next),
            (Created, Active)
                | (Active, Committing)
                | (Active, Stale)
                | (Active, Closed) // explicit close without commit
                | (Committing, Merging)
                | (Committing, Active) // commit failed, retry
                | (Merging, Closed)
                | (Stale, Active) // heartbeat resumed
                | (Stale, Orphaned)
                | (Orphaned, Active) // recovery: resume
                | (Orphaned, Committing) // recovery: commit
                | (Orphaned, Closed) // recovery: rollback
        )
    }
}

/// Session record stored in Redb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub agent_id: String,
    pub created_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub state: SessionState,
    pub git_branch: Option<String>,
    pub staging_dir: Option<String>,
    pub entries_written: u64,
    pub entries_merged: u64,
}

/// Entry lifecycle status per D45 Part 3.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum EntryStatus {
    Pending,
    Committed,
    #[default]
    Merged,
    Rejected,
}

/// Search visibility mode per D43.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum VisibilityMode {
    /// Main knowledge + agent's own pending entries.
    #[default]
    Own,
    /// Main knowledge + all agents' pending entries.
    All,
    /// Main knowledge + named agents' pending entries.
    Explicit(Vec<String>),
}

/// Merge queue entry stored in Redb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeQueueEntry {
    pub entry_id: uuid::Uuid,
    pub agent_id: String,
    pub session_id: String,
    pub scope_id: String,
    pub enqueued_at: DateTime<Utc>,
    pub retry_count: u32,
    pub last_error: Option<String>,
}

/// Recovery action choices for orphaned sessions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RecoveryAction {
    Resume,
    Commit,
    Rollback,
}
```

### Step 4: Wire up module and extend KnowledgeEntry

Add to `crates/corvia-common/src/lib.rs`:
```rust
pub mod agent_types;
```

Add fields to `KnowledgeEntry` in `crates/corvia-common/src/types.rs` (after `metadata`):
```rust
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub entry_status: crate::agent_types::EntryStatus,
```

Update the `KnowledgeEntry::new()` constructor to initialize:
```rust
    agent_id: None,
    session_id: None,
    entry_status: crate::agent_types::EntryStatus::default(),
```

Add builder method:
```rust
    pub fn with_agent(mut self, agent_id: String, session_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self.session_id = Some(session_id);
        self.entry_status = crate::agent_types::EntryStatus::Pending;
        self
    }
```

### Step 5: Extend config with agent_lifecycle and merge sections

Add to `crates/corvia-common/src/config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLifecycleConfig {
    #[serde(default = "default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,
    #[serde(default = "default_stale_timeout_secs")]
    pub stale_timeout_secs: u64,
    #[serde(default = "default_orphan_grace_secs")]
    pub orphan_grace_secs: u64,
    #[serde(default = "default_gc_orphan_after_secs")]
    pub gc_orphan_after_secs: u64,
    #[serde(default = "default_gc_closed_session_after_secs")]
    pub gc_closed_session_after_secs: u64,
    #[serde(default = "default_gc_inactive_agent_after_secs")]
    pub gc_inactive_agent_after_secs: u64,
}

fn default_heartbeat_interval_secs() -> u64 { 30 }
fn default_stale_timeout_secs() -> u64 { 300 }      // 5 min
fn default_orphan_grace_secs() -> u64 { 1200 }      // 20 min
fn default_gc_orphan_after_secs() -> u64 { 86400 }  // 24 hr
fn default_gc_closed_session_after_secs() -> u64 { 604800 } // 7 days
fn default_gc_inactive_agent_after_secs() -> u64 { 2592000 } // 30 days

impl Default for AgentLifecycleConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: default_heartbeat_interval_secs(),
            stale_timeout_secs: default_stale_timeout_secs(),
            orphan_grace_secs: default_orphan_grace_secs(),
            gc_orphan_after_secs: default_gc_orphan_after_secs(),
            gc_closed_session_after_secs: default_gc_closed_session_after_secs(),
            gc_inactive_agent_after_secs: default_gc_inactive_agent_after_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Chat model for LLM-assisted merge (Ollama).
    #[serde(default = "default_merge_model")]
    pub model: String,
    /// Cosine similarity threshold — above this means conflict.
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Max retries for failed LLM merges.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_merge_model() -> String { "llama3.2".into() }
fn default_similarity_threshold() -> f32 { 0.85 }
fn default_max_retries() -> u32 { 3 }

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            model: default_merge_model(),
            similarity_threshold: default_similarity_threshold(),
            max_retries: default_max_retries(),
        }
    }
}
```

Add to `CorviaConfig`:
```rust
    #[serde(default)]
    pub agent_lifecycle: AgentLifecycleConfig,
    #[serde(default)]
    pub merge: MergeConfig,
```

### Step 6: Add Agent error variant

Add to `CorviaError` in `crates/corvia-common/src/errors.rs`:
```rust
    #[error("Agent error: {0}")]
    Agent(String),
```

### Step 7: Run tests and verify they pass

Run: `cargo test -p corvia-common`
Expected: All new and existing tests pass.

### Step 8: Verify existing tests still pass

Run: `cargo test --workspace`
Expected: All 55 existing tests pass (backward-compatible serde defaults).

### Step 9: Commit

```bash
git add crates/corvia-common/src/agent_types.rs crates/corvia-common/src/
git commit -m "feat(m2): add agent identity, session, and entry status types (D45)"
```

---

## Task 2: Agent Registry

**Files:**
- Create: `crates/corvia-kernel/src/agent_registry.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests for agent registry

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::agent_types::*;

    fn test_registry() -> AgentRegistry {
        let dir = tempfile::tempdir().unwrap();
        AgentRegistry::open(dir.path()).unwrap()
    }

    #[test]
    fn test_register_and_get_agent() {
        let reg = test_registry();
        let record = reg
            .register("test::indexer", "Code Indexer", IdentityType::Registered,
                AgentPermission::ReadWrite { scopes: vec!["test".into()] })
            .unwrap();
        assert_eq!(record.agent_id, "test::indexer");
        assert_eq!(record.status, AgentStatus::Active);

        let fetched = reg.get("test::indexer").unwrap().unwrap();
        assert_eq!(fetched.display_name, "Code Indexer");
    }

    #[test]
    fn test_duplicate_register_returns_existing() {
        let reg = test_registry();
        reg.register("test::agent", "Agent", IdentityType::Registered,
            AgentPermission::ReadOnly).unwrap();
        let second = reg.register("test::agent", "Agent v2", IdentityType::Registered,
            AgentPermission::ReadOnly).unwrap();
        // Returns existing, doesn't overwrite
        assert_eq!(second.display_name, "Agent");
    }

    #[test]
    fn test_update_last_seen() {
        let reg = test_registry();
        let original = reg.register("test::agent", "Agent", IdentityType::Registered,
            AgentPermission::ReadOnly).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        reg.touch("test::agent").unwrap();
        let updated = reg.get("test::agent").unwrap().unwrap();
        assert!(updated.last_seen > original.last_seen);
    }

    #[test]
    fn test_list_active_agents() {
        let reg = test_registry();
        reg.register("test::a", "A", IdentityType::Registered, AgentPermission::ReadOnly).unwrap();
        reg.register("test::b", "B", IdentityType::Registered, AgentPermission::ReadOnly).unwrap();
        let agents = reg.list_active().unwrap();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_suspend_agent() {
        let reg = test_registry();
        reg.register("test::agent", "Agent", IdentityType::Registered,
            AgentPermission::ReadOnly).unwrap();
        reg.set_status("test::agent", AgentStatus::Suspended).unwrap();
        let agent = reg.get("test::agent").unwrap().unwrap();
        assert_eq!(agent.status, AgentStatus::Suspended);
        // Suspended agents not in active list
        assert_eq!(reg.list_active().unwrap().len(), 0);
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let reg = test_registry();
        assert!(reg.get("nobody").unwrap().is_none());
    }
}
```

### Step 2: Run test to verify it fails

Run: `cargo test -p corvia-kernel -- agent_registry`
Expected: Compilation error.

### Step 3: Implement AgentRegistry

Create `crates/corvia-kernel/src/agent_registry.rs`:

```rust
use corvia_common::agent_types::*;
use corvia_common::errors::{CorviaError, Result};
use redb::{Database, TableDefinition};
use std::path::Path;
use tracing::info;

const AGENTS: TableDefinition<&str, &[u8]> = TableDefinition::new("agents");

/// Manages agent registration and lifecycle in a Redb database.
/// Uses `.corvia/coordination.redb` — shared with SessionManager and MergeQueue.
pub struct AgentRegistry {
    db: std::sync::Arc<Database>,
}

impl AgentRegistry {
    /// Open or create the coordination database.
    pub fn open(data_dir: &Path) -> Result<Self> {
        let db_path = data_dir.join("coordination.redb");
        let db = Database::create(&db_path)
            .map_err(|e| CorviaError::Agent(format!("Failed to open coordination db: {e}")))?;

        // Ensure table exists
        let write_txn = db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        { let _ = write_txn.open_table(AGENTS); }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to init agents table: {e}")))?;

        Ok(Self { db: std::sync::Arc::new(db) })
    }

    /// Create from an existing shared database handle.
    pub fn from_db(db: std::sync::Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        { let _ = write_txn.open_table(AGENTS); }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to init agents table: {e}")))?;
        Ok(Self { db })
    }

    pub fn db(&self) -> &std::sync::Arc<Database> {
        &self.db
    }

    /// Register a new agent. If already exists, returns existing record.
    pub fn register(
        &self,
        agent_id: &str,
        display_name: &str,
        identity_type: IdentityType,
        permissions: AgentPermission,
    ) -> Result<AgentRecord> {
        // Check if already exists
        if let Some(existing) = self.get(agent_id)? {
            return Ok(existing);
        }

        let now = chrono::Utc::now();
        let record = AgentRecord {
            agent_id: agent_id.into(),
            display_name: display_name.into(),
            identity_type,
            registered_at: now,
            permissions,
            last_seen: now,
            status: AgentStatus::Active,
        };

        let bytes = serde_json::to_vec(&record)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize agent: {e}")))?;

        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(AGENTS)
                .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;
            table.insert(agent_id, bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to insert agent: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit agent: {e}")))?;

        info!(agent_id, "agent_registered");
        Ok(record)
    }

    /// Get an agent record by ID.
    pub fn get(&self, agent_id: &str) -> Result<Option<AgentRecord>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(AGENTS)
            .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;

        match table.get(agent_id) {
            Ok(Some(val)) => {
                let record: AgentRecord = serde_json::from_slice(val.value())
                    .map_err(|e| CorviaError::Agent(format!("Failed to deserialize agent: {e}")))?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(CorviaError::Agent(format!("Failed to get agent: {e}"))),
        }
    }

    /// Update last_seen timestamp.
    pub fn touch(&self, agent_id: &str) -> Result<()> {
        let mut record = self.get(agent_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Agent {agent_id} not found")))?;
        record.last_seen = chrono::Utc::now();
        self.put(&record)
    }

    /// Update agent status.
    pub fn set_status(&self, agent_id: &str, status: AgentStatus) -> Result<()> {
        let mut record = self.get(agent_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Agent {agent_id} not found")))?;
        record.status = status;
        self.put(&record)
    }

    /// List all agents with Active status.
    pub fn list_active(&self) -> Result<Vec<AgentRecord>> {
        self.list_all()?
            .into_iter()
            .filter(|a| a.status == AgentStatus::Active)
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// List all agent records.
    pub fn list_all(&self) -> Result<Vec<AgentRecord>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(AGENTS)
            .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;

        let mut agents = Vec::new();
        let iter = table.iter()
            .map_err(|e| CorviaError::Agent(format!("Failed to iterate agents: {e}")))?;
        for item in iter {
            let (_, val) = item
                .map_err(|e| CorviaError::Agent(format!("Failed to read agent: {e}")))?;
            let record: AgentRecord = serde_json::from_slice(val.value())
                .map_err(|e| CorviaError::Agent(format!("Failed to deserialize agent: {e}")))?;
            agents.push(record);
        }
        Ok(agents)
    }

    /// Internal: write an agent record.
    fn put(&self, record: &AgentRecord) -> Result<()> {
        let bytes = serde_json::to_vec(record)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize agent: {e}")))?;
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(AGENTS)
                .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;
            table.insert(record.agent_id.as_str(), bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to update agent: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit agent update: {e}")))?;
        Ok(())
    }
}

// Helper trait for piping values (avoids separate variable)
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R where F: FnOnce(Self) -> R { f(self) }
}
impl<T> Pipe for T {}
```

### Step 4: Wire up module

Add to `crates/corvia-kernel/src/lib.rs`:
```rust
pub mod agent_registry;
```

### Step 5: Run tests and verify they pass

Run: `cargo test -p corvia-kernel -- agent_registry`
Expected: All 6 tests pass.

### Step 6: Commit

```bash
git add crates/corvia-kernel/src/agent_registry.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): agent registry with Redb coordination store (D45 Part 2)"
```

---

## Task 3: Session Manager

**Files:**
- Create: `crates/corvia-kernel/src/session_manager.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::agent_types::*;

    fn test_manager() -> SessionManager {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        SessionManager::from_db(db).unwrap()
    }

    #[test]
    fn test_create_session() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        assert!(session.session_id.starts_with("test::agent/sess-"));
        assert_eq!(session.state, SessionState::Created);
        assert!(session.git_branch.is_some());
        assert!(session.staging_dir.is_some());
    }

    #[test]
    fn test_create_session_without_staging() {
        let mgr = test_manager();
        let session = mgr.create("mcp::agent", false).unwrap();
        assert!(session.git_branch.is_none());
        assert!(session.staging_dir.is_none());
    }

    #[test]
    fn test_activate_session() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        mgr.transition(&session.session_id, SessionState::Active).unwrap();
        let updated = mgr.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.state, SessionState::Active);
    }

    #[test]
    fn test_invalid_transition_rejected() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        // Created cannot go directly to Merging
        let result = mgr.transition(&session.session_id, SessionState::Merging);
        assert!(result.is_err());
    }

    #[test]
    fn test_heartbeat_updates_timestamp() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        mgr.transition(&session.session_id, SessionState::Active).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.heartbeat(&session.session_id).unwrap();
        let updated = mgr.get(&session.session_id).unwrap().unwrap();
        assert!(updated.last_heartbeat > session.last_heartbeat);
    }

    #[test]
    fn test_list_agent_sessions() {
        let mgr = test_manager();
        mgr.create("test::a", true).unwrap();
        mgr.create("test::a", true).unwrap();
        mgr.create("test::b", true).unwrap();
        let sessions = mgr.list_by_agent("test::a").unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn test_increment_entries_written() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        mgr.increment_written(&session.session_id).unwrap();
        mgr.increment_written(&session.session_id).unwrap();
        let updated = mgr.get(&session.session_id).unwrap().unwrap();
        assert_eq!(updated.entries_written, 2);
    }

    #[test]
    fn test_find_stale_sessions() {
        let mgr = test_manager();
        let session = mgr.create("test::agent", true).unwrap();
        mgr.transition(&session.session_id, SessionState::Active).unwrap();
        // With a 0-second timeout, session is immediately stale
        let stale = mgr.find_stale(std::time::Duration::ZERO).unwrap();
        assert_eq!(stale.len(), 1);
    }
}
```

### Step 2: Run test to verify it fails

Run: `cargo test -p corvia-kernel -- session_manager`
Expected: Compilation error.

### Step 3: Implement SessionManager

Create `crates/corvia-kernel/src/session_manager.rs`. Key methods:

- `from_db(Arc<Database>) -> Result<Self>` — shares coordination Redb
- `create(agent_id, with_staging) -> Result<SessionRecord>` — generates `{agent_id}/sess-{uuid_short}`
- `get(session_id) -> Result<Option<SessionRecord>>`
- `transition(session_id, new_state) -> Result<()>` — validates state machine
- `heartbeat(session_id) -> Result<()>` — updates last_heartbeat, resets Stale→Active
- `increment_written(session_id) -> Result<()>`
- `increment_merged(session_id) -> Result<()>`
- `list_by_agent(agent_id) -> Result<Vec<SessionRecord>>`
- `find_stale(timeout: Duration) -> Result<Vec<SessionRecord>>` — Active sessions past heartbeat timeout
- `find_orphaned(grace: Duration) -> Result<Vec<SessionRecord>>` — Stale sessions past grace period
- `list_open() -> Result<Vec<SessionRecord>>` — all non-Closed sessions

Uses Redb table: `sessions` (key: session_id string, value: JSON bytes).
Session ID format: `{agent_id}/sess-{first_8_chars_of_uuid}`.
Staging dir: `.corvia/staging/{agent_id_sanitized}/{session_short}/`.
Git branch: `agents/{agent_id_sanitized}/sess-{uuid_short}`.

### Step 4: Wire up module and run tests

Add to `crates/corvia-kernel/src/lib.rs`:
```rust
pub mod session_manager;
```

Run: `cargo test -p corvia-kernel -- session_manager`
Expected: All 8 tests pass.

### Step 5: Commit

```bash
git add crates/corvia-kernel/src/session_manager.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): session manager with state machine and heartbeat (D45 Part 5)"
```

---

## Task 4: Staging Manager

**Files:**
- Create: `crates/corvia-kernel/src/staging.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_staging_in(dir: &Path) -> StagingManager {
        StagingManager::new(dir)
    }

    #[test]
    fn test_create_staging_dir() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let path = staging.create_staging_dir("test::agent", "sess-abc123").unwrap();
        assert!(path.exists());
        assert!(path.is_dir());
    }

    #[test]
    fn test_write_and_read_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let staging_dir = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        staging.write_staging_file(&staging_dir, &entry).unwrap();
        let files = staging.list_staging_files(&staging_dir).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_cleanup_staging_dir() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let path = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        assert!(path.exists());
        staging.cleanup_staging_dir(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_move_to_knowledge() {
        let dir = tempfile::tempdir().unwrap();
        let staging = test_staging_in(dir.path());
        let staging_dir = staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        let entry_id = entry.id;
        staging.write_staging_file(&staging_dir, &entry).unwrap();
        staging.move_to_knowledge(&staging_dir, &entry_id, "scope").unwrap();

        // Staging file gone, knowledge file exists
        assert_eq!(staging.list_staging_files(&staging_dir).unwrap().len(), 0);
        let knowledge_path = dir.path()
            .join("knowledge").join("scope").join(format!("{entry_id}.json"));
        assert!(knowledge_path.exists());
    }
}
```

### Step 2: Run test to verify it fails

Run: `cargo test -p corvia-kernel -- staging`
Expected: Compilation error.

### Step 3: Implement StagingManager

Create `crates/corvia-kernel/src/staging.rs`. Key methods:

- `new(data_dir: &Path) -> Self`
- `create_staging_dir(agent_id, session_short) -> Result<PathBuf>` — creates `.corvia/staging/{sanitized_agent}/{session}/`
- `write_staging_file(staging_dir, entry) -> Result<()>` — writes `{entry_id}.json`
- `read_staging_file(staging_dir, entry_id) -> Result<KnowledgeEntry>`
- `list_staging_files(staging_dir) -> Result<Vec<Uuid>>` — lists entry IDs from filenames
- `move_to_knowledge(staging_dir, entry_id, scope_id) -> Result<()>` — moves file to `.corvia/knowledge/{scope}/{id}.json`
- `cleanup_staging_dir(staging_dir) -> Result<()>` — removes directory

Git operations (shell out to `git`):
- `create_branch(branch_name) -> Result<()>` — `git branch {name}` (from HEAD)
- `commit_on_branch(branch_name, message, files) -> Result<()>` — `git checkout {branch} && git add {files} && git commit -m {msg} && git checkout -`
- `merge_branch(branch_name) -> Result<()>` — `git merge {branch} --no-ff -m "..."`
- `delete_branch(branch_name) -> Result<()>` — `git branch -d {name}`

Note: git operations check for `.git` directory. If not in a git repo, they are no-ops (logged as warnings). This allows tests to run without git init.

### Step 4: Wire up and run tests

Add to `crates/corvia-kernel/src/lib.rs`:
```rust
pub mod staging;
```

Run: `cargo test -p corvia-kernel -- staging`
Expected: All 4 tests pass.

### Step 5: Write git operation tests (require temp git repo)

```rust
    #[test]
    fn test_git_branch_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        // Initialize a git repo in the temp dir
        std::process::Command::new("git").args(["init"]).current_dir(dir.path()).output().unwrap();
        std::process::Command::new("git").args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path()).output().unwrap();

        let staging = test_staging_in(dir.path());
        staging.create_branch("agents/test/sess-abc").unwrap();
        staging.delete_branch("agents/test/sess-abc").unwrap();
    }
```

### Step 6: Run all staging tests

Run: `cargo test -p corvia-kernel -- staging`
Expected: All pass.

### Step 7: Commit

```bash
git add crates/corvia-kernel/src/staging.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): staging manager with dirs and git branch ops (D43)"
```

---

## Task 5: Agent Writer (Atomic Write Path)

**Files:**
- Create: `crates/corvia-kernel/src/agent_writer.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::traits::QueryableStore;

    // Uses a mock embedding that just returns a fixed vector
    struct MockEngine;
    #[async_trait::async_trait]
    impl crate::traits::InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize { 3 }
    }

    #[tokio::test]
    async fn test_write_entry_for_registered_agent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let engine = std::sync::Arc::new(MockEngine);
        let staging = StagingManager::new(dir.path());
        let writer = AgentWriter::new(
            std::sync::Arc::new(store),
            engine,
            staging,
        );

        let staging_dir = writer.staging.create_staging_dir("test::agent", "sess-abc").unwrap();
        let entry = writer.write(
            "test knowledge",
            "test-scope",
            "v1",
            "test::agent",
            "test::agent/sess-abc",
            Some(&staging_dir),
        ).await.unwrap();

        assert_eq!(entry.agent_id, Some("test::agent".into()));
        assert_eq!(entry.entry_status, EntryStatus::Pending);
        // Entry is in HNSW (searchable immediately)
        let results = writer.store.search(&[1.0, 0.0, 0.0], "test-scope", 5).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_write_entry_for_mcp_agent() {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        let engine = std::sync::Arc::new(MockEngine);
        let staging = StagingManager::new(dir.path());
        let writer = AgentWriter::new(
            std::sync::Arc::new(store),
            engine,
            staging,
        );

        // MCP agents have no staging dir
        let entry = writer.write(
            "mcp knowledge",
            "test-scope",
            "v1",
            "crewai::advisor",
            "crewai::advisor/sess-xyz",
            None,
        ).await.unwrap();

        assert_eq!(entry.agent_id, Some("crewai::advisor".into()));
    }
}
```

### Step 2: Run test to verify it fails

Run: `cargo test -p corvia-kernel -- agent_writer`
Expected: Compilation error.

### Step 3: Implement AgentWriter

Create `crates/corvia-kernel/src/agent_writer.rs`:

The atomic write path per D45 Part 4:
1. Write JSON to staging dir (if registered agent) — filesystem, recoverable
2. Embed content via engine — idempotent
3. Atomic Redb transaction: entry metadata + HNSW mapping + vector insert

Key struct:
```rust
pub struct AgentWriter {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub staging: StagingManager,
}
```

Key method:
```rust
pub async fn write(
    &self,
    content: &str,
    scope_id: &str,
    source_version: &str,
    agent_id: &str,
    session_id: &str,
    staging_dir: Option<&Path>,
) -> Result<KnowledgeEntry>
```

The method creates a `KnowledgeEntry` with `.with_agent()`, embeds it, writes staging file (if staging_dir provided), then inserts into the store.

### Step 4: Wire up and run tests

Run: `cargo test -p corvia-kernel -- agent_writer`
Expected: Both tests pass.

### Step 5: Commit

```bash
git add crates/corvia-kernel/src/agent_writer.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): atomic write path with staging + embed + insert (D45 Part 4)"
```

---

## Task 6: Merge Queue + Commit Pipeline

**Files:**
- Create: `crates/corvia-kernel/src/merge_queue.rs`
- Create: `crates/corvia-kernel/src/commit_pipeline.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests for merge queue

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_queue() -> MergeQueue {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            redb::Database::create(dir.path().join("coordination.redb")).unwrap()
        );
        MergeQueue::from_db(db).unwrap()
    }

    #[test]
    fn test_enqueue_and_dequeue() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "test::agent/sess-abc", "scope").unwrap();
        let entries = queue.dequeue_batch(10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_id, id);
    }

    #[test]
    fn test_dequeue_empty() {
        let queue = test_queue();
        let entries = queue.dequeue_batch(10).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_mark_complete() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "sess", "scope").unwrap();
        queue.mark_complete(&id).unwrap();
        assert!(queue.dequeue_batch(10).unwrap().is_empty());
    }

    #[test]
    fn test_mark_failed_with_retry() {
        let queue = test_queue();
        let id = uuid::Uuid::now_v7();
        queue.enqueue(id, "test::agent", "sess", "scope").unwrap();
        queue.mark_failed(&id, "Ollama down").unwrap();
        let entries = queue.dequeue_batch(10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].retry_count, 1);
        assert_eq!(entries[0].last_error, Some("Ollama down".into()));
    }

    #[test]
    fn test_queue_depth() {
        let queue = test_queue();
        for _ in 0..5 {
            queue.enqueue(uuid::Uuid::now_v7(), "a", "s", "scope").unwrap();
        }
        assert_eq!(queue.depth().unwrap(), 5);
    }
}
```

### Step 2: Implement MergeQueue

Redb table: `merge_queue` (key: UUID string, value: JSON bytes of `MergeQueueEntry`).

Methods:
- `from_db(Arc<Database>) -> Result<Self>`
- `enqueue(entry_id, agent_id, session_id, scope_id) -> Result<()>`
- `dequeue_batch(limit) -> Result<Vec<MergeQueueEntry>>` — returns oldest entries by enqueued_at
- `mark_complete(entry_id) -> Result<()>` — removes from queue
- `mark_failed(entry_id, error) -> Result<()>` — increments retry_count, sets last_error
- `depth() -> Result<u64>`

### Step 3: Write failing tests for commit pipeline

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_commit_flow_moves_entries_to_queue() {
        // Setup: staging dir with 2 entry files, session in Active state
        // Action: commit_session()
        // Assert: session state → Committing → entries in merge queue, state → Merging
    }

    #[tokio::test]
    async fn test_commit_is_idempotent() {
        // Setup: session already in Committing state with entries in queue
        // Action: commit_session() again
        // Assert: no duplicate queue entries, no error
    }
}
```

### Step 4: Implement CommitPipeline

The 5-step idempotent commit flow from D45 Part 5:
```
Step 1: Session status → Committing (Redb)
Step 2: git add staging files + git commit on agent branch
Step 3: All entry statuses → Committed (Redb entry metadata)
Step 4: Entries enter merge queue (Redb)
Step 5: Session status → Merging (Redb)
```

Each step is idempotent — re-running after crash skips completed steps.

Key struct:
```rust
pub struct CommitPipeline {
    session_mgr: Arc<SessionManager>,
    merge_queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    store: Arc<dyn QueryableStore>,
}
```

### Step 5: Run tests, verify pass, commit

Run: `cargo test -p corvia-kernel -- merge_queue commit_pipeline`

```bash
git add crates/corvia-kernel/src/merge_queue.rs crates/corvia-kernel/src/commit_pipeline.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): merge queue and idempotent commit pipeline (D45 Part 5)"
```

---

## Task 7: Merge Worker

**Files:**
- Create: `crates/corvia-kernel/src/merge_worker.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_no_conflict_auto_merges() {
        // Insert entry A (embedding [1,0,0]) as Merged in store
        // Queue entry B (embedding [0,1,0]) — very different
        // Run merge_worker.process_one()
        // Assert: B status → Merged, staging file moved to knowledge/
    }

    #[tokio::test]
    async fn test_conflict_detected_above_threshold() {
        // Insert entry A (embedding [1,0,0]) as Merged
        // Queue entry B (embedding [0.95,0.05,0]) — very similar, same scope
        // Run merge_worker.detect_conflict()
        // Assert: conflict detected (similarity > 0.85)
    }

    #[tokio::test]
    async fn test_failed_merge_retries() {
        // Queue entry with unreachable LLM
        // Run merge_worker.process_one()
        // Assert: entry still in queue with retry_count=1
    }

    #[tokio::test]
    async fn test_max_retries_exhausted() {
        // Queue entry with retry_count = max_retries
        // Run merge_worker.process_one()
        // Assert: entry marked as failed, logged, not retried
    }
}
```

### Step 2: Implement MergeWorker

Key struct:
```rust
pub struct MergeWorker {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
    queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    session_mgr: Arc<SessionManager>,
    merge_config: MergeConfig,
    ollama_url: String,
}
```

Key methods:
- `async fn run(&self)` — loop: dequeue batch, process each, sleep if empty
- `async fn process_one(&self, queue_entry: &MergeQueueEntry) -> Result<()>`
- `async fn detect_conflict(&self, entry: &KnowledgeEntry) -> Result<Option<KnowledgeEntry>>` — search for similar entries in same scope, check threshold
- `async fn auto_merge(&self, entry: &KnowledgeEntry, staging_dir: &Path) -> Result<()>` — move file, retag status, git merge
- `async fn llm_merge(&self, entry: &KnowledgeEntry, conflict: &KnowledgeEntry) -> Result<KnowledgeEntry>` — call Ollama `/api/chat` with merge prompt, re-embed merged content

LLM merge prompt template:
```
You are merging two knowledge entries that conflict. Produce a single merged entry that preserves all important information from both.

Entry A (existing):
{content_a}

Entry B (new):
{content_b}

Merged entry:
```

The merge worker calls `POST {ollama_url}/api/chat` with the configured `merge.model`.

### Step 3: Run tests, verify pass, commit

```bash
git add crates/corvia-kernel/src/merge_worker.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): merge worker with conflict detection and LLM merge (D45 Part 5)"
```

---

## Task 8: Context Builder + Visibility

**Files:**
- Create: `crates/corvia-kernel/src/context_builder.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_own_visibility_includes_agent_pending() {
        // Store: 2 Merged entries, 1 Pending entry by agent-A, 1 Pending by agent-B
        // Search with VisibilityMode::Own for agent-A
        // Assert: returns Merged + agent-A's Pending, NOT agent-B's Pending
    }

    #[tokio::test]
    async fn test_all_visibility_includes_all_pending() {
        // Same setup
        // Search with VisibilityMode::All
        // Assert: returns Merged + all Pending entries
    }

    #[tokio::test]
    async fn test_explicit_visibility() {
        // Search with VisibilityMode::Explicit(vec!["agent-A"])
        // Assert: returns Merged + agent-A's Pending only
    }

    #[tokio::test]
    async fn test_rbac_scope_filtering() {
        // Agent with ReadWrite { scopes: ["project-a"] }
        // Entries in project-a and project-b
        // Assert: only project-a entries returned
    }
}
```

### Step 2: Implement ContextBuilder

```rust
pub struct ContextBuilder {
    store: Arc<dyn QueryableStore>,
    engine: Arc<dyn InferenceEngine>,
}
```

Key method:
```rust
pub async fn search(
    &self,
    query: &str,
    scope_id: &str,
    limit: usize,
    visibility: &VisibilityMode,
    agent_id: Option<&str>,
    permissions: Option<&AgentPermission>,
) -> Result<Vec<SearchResult>>
```

The search method:
1. Embeds the query
2. Calls `store.search()` with a higher limit (2x) to allow for post-filtering
3. Post-filters results based on visibility mode and entry_status:
   - `Merged` entries always included
   - `Pending` entries filtered by visibility + agent_id
4. Post-filters by RBAC permissions (scope access)
5. Returns up to `limit` results

### Step 3: Run tests, verify pass, commit

```bash
git add crates/corvia-kernel/src/context_builder.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): context builder with visibility modes and RBAC (D39, D43)"
```

---

## Task 9: Agent Coordinator

**Files:**
- Create: `crates/corvia-kernel/src/agent_coordinator.rs`
- Modify: `crates/corvia-kernel/src/lib.rs`

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_full_agent_lifecycle() {
        // 1. Register agent
        // 2. Create session
        // 3. Write 2 entries
        // 4. Commit session
        // 5. Verify entries in merge queue
    }

    #[tokio::test]
    async fn test_connect_with_recoverable_sessions() {
        // 1. Register agent, create session, write entries
        // 2. Simulate crash (mark session Orphaned)
        // 3. Call connect() again
        // 4. Assert: returns recoverable_sessions with the orphaned session
    }

    #[tokio::test]
    async fn test_recover_resume() {
        // 1. Setup orphaned session with staging files
        // 2. Call recover(session_id, Resume)
        // 3. Assert: session back to Active, staging files re-indexed
    }

    #[tokio::test]
    async fn test_recover_rollback() {
        // 1. Setup orphaned session
        // 2. Call recover(session_id, Rollback)
        // 3. Assert: staging dir deleted, session Closed
    }

    #[tokio::test]
    async fn test_gc_cleans_old_orphans() {
        // 1. Create orphaned session with old heartbeat
        // 2. Run gc()
        // 3. Assert: session rolled back and closed
    }
}
```

### Step 2: Implement AgentCoordinator

The orchestrator that ties all components together:

```rust
pub struct AgentCoordinator {
    registry: Arc<AgentRegistry>,
    sessions: Arc<SessionManager>,
    writer: Arc<AgentWriter>,
    commit_pipeline: Arc<CommitPipeline>,
    merge_worker: Arc<MergeWorker>,
    context: Arc<ContextBuilder>,
    merge_queue: Arc<MergeQueue>,
    staging: Arc<StagingManager>,
    config: AgentLifecycleConfig,
}
```

Key methods:
- `async fn new(config, store, engine, data_dir) -> Result<Self>` — constructs all sub-components from shared coordination Redb
- `async fn register_agent(identity: &AgentIdentity, ...) -> Result<AgentRecord>`
- `async fn connect(agent_id) -> Result<ConnectResponse>` — returns active + recoverable sessions
- `async fn create_session(agent_id) -> Result<SessionRecord>`
- `async fn write_entry(session_id, content, scope_id, ...) -> Result<KnowledgeEntry>`
- `async fn commit_session(session_id) -> Result<()>`
- `async fn rollback_session(session_id) -> Result<()>`
- `async fn heartbeat(session_id) -> Result<()>`
- `async fn recover(session_id, action: RecoveryAction) -> Result<()>`
- `async fn search(query, scope_id, limit, visibility, agent_id) -> Result<Vec<SearchResult>>`
- `async fn gc(&self) -> Result<GcReport>` — garbage collection sweep
- `async fn start_background_tasks(&self)` — spawns heartbeat monitor + GC timer + merge worker

`ConnectResponse`:
```rust
pub struct ConnectResponse {
    pub agent_id: String,
    pub recoverable_sessions: Vec<SessionRecord>,
    pub active_sessions: Vec<SessionRecord>,
}
```

### Step 3: Run tests, verify pass, commit

```bash
git add crates/corvia-kernel/src/agent_coordinator.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m2): agent coordinator with lifecycle, recovery, and GC (D45)"
```

---

## Task 10: REST API — Agent & Session Endpoints

**Files:**
- Modify: `crates/corvia-server/src/rest.rs`
- Modify: `crates/corvia-server/Cargo.toml` (add `chrono` if needed for request types)

### Step 1: Write failing tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum_test::TestServer; // or use tower::ServiceExt for test requests

    #[tokio::test]
    async fn test_register_agent() {
        // POST /v1/agents { name: "indexer", scope: "project" }
        // Assert: 201, body contains agent_id
    }

    #[tokio::test]
    async fn test_create_session() {
        // POST /v1/agents/{agent_id}/sessions
        // Assert: 201, body contains session_id + recoverable_sessions
    }

    #[tokio::test]
    async fn test_heartbeat() {
        // POST /v1/sessions/{session_id}/heartbeat
        // Assert: 200
    }

    #[tokio::test]
    async fn test_write_via_session() {
        // POST /v1/sessions/{session_id}/write { content, scope_id }
        // Assert: 201, body contains entry_id
    }

    #[tokio::test]
    async fn test_commit_session() {
        // POST /v1/sessions/{session_id}/commit
        // Assert: 200
    }

    #[tokio::test]
    async fn test_session_state() {
        // GET /v1/sessions/{session_id}/state
        // Assert: 200, body contains state + entries
    }
}
```

### Step 2: Implement REST endpoints

Extend `AppState` to include `AgentCoordinator`:
```rust
pub struct AppState {
    pub store: Arc<dyn QueryableStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub coordinator: Arc<AgentCoordinator>,
}
```

New routes (nested under `/v1`):
```rust
.route("/v1/agents", post(register_agent))
.route("/v1/agents/:agent_id/sessions", post(create_session))
.route("/v1/agents/:agent_id/knowledge", get(agent_knowledge))
.route("/v1/sessions/:session_id/heartbeat", post(heartbeat))
.route("/v1/sessions/:session_id/write", post(write_entry))
.route("/v1/sessions/:session_id/commit", post(commit_session))
.route("/v1/sessions/:session_id/rollback", post(rollback_session))
.route("/v1/sessions/:session_id/recover", post(recover_session))
.route("/v1/sessions/:session_id/state", get(session_state))
```

### Step 3: Run tests, verify pass, commit

```bash
git add crates/corvia-server/src/rest.rs crates/corvia-server/Cargo.toml
git commit -m "feat(m2): REST API for agent registration, sessions, and writes (D45 Part 7)"
```

---

## Task 11: MCP Server

**Files:**
- Create: `crates/corvia-server/src/mcp.rs`
- Modify: `crates/corvia-server/src/lib.rs`
- Modify: `crates/corvia-server/Cargo.toml`

### Step 1: Research available Rust MCP SDK

Check crates.io for `mcp-server`, `rmcp`, or similar. If no suitable crate exists, implement minimal JSON-RPC 2.0 handler on axum.

The MCP protocol requires:
- JSON-RPC 2.0 over HTTP+SSE (for HTTP transport) or stdio
- Methods: `initialize`, `tools/list`, `tools/call`
- Tool definitions: `corvia_search`, `corvia_write`, `corvia_history`, `corvia_agent_status`

### Step 2: Write failing tests for MCP tool handlers

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tools_list() {
        // Call tools/list handler
        // Assert: returns 4 tools with correct schemas
    }

    #[tokio::test]
    async fn test_corvia_search() {
        // Call corvia_search with query + scope
        // Assert: returns search results
    }

    #[tokio::test]
    async fn test_corvia_write_with_meta() {
        // Call corvia_write with _meta.agent_id
        // Assert: entry created with correct agent_id
    }

    #[tokio::test]
    async fn test_corvia_write_without_meta_rejected() {
        // Call corvia_write without _meta.agent_id
        // Assert: rejected with error (read-only without identity)
    }
}
```

### Step 3: Implement MCP server

Four tools:
- `corvia_search` — `{ query: string, scope_id: string, limit?: number }`
- `corvia_write` — `{ content: string, scope_id: string, source_version?: string }` (requires `_meta.agent_id`)
- `corvia_history` — `{ entry_id: string }` (placeholder for M3 temporal)
- `corvia_agent_status` — `{}` (returns agent's contribution summary, requires `_meta.agent_id`)

Identity extraction from MCP:
- `clientInfo.name` + `clientInfo.version` from `initialize`
- `_meta.agent_id`, `_meta.agent_name`, `_meta.agent_role` from each `tools/call`

### Step 4: Add `--mcp` flag to CLI serve command

In `crates/corvia-cli/src/main.rs`, extend `Commands::Serve` with `mcp: bool` flag. When `--mcp`, start MCP server on stdio (or configurable transport).

### Step 5: Run tests, verify pass, commit

```bash
git add crates/corvia-server/src/mcp.rs crates/corvia-server/src/lib.rs crates/corvia-server/Cargo.toml
git commit -m "feat(m2): MCP server with search, write, history, agent_status tools (D45 Part 7)"
```

---

## Task 12: CLI Updates

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

### Step 1: Update `corvia status` to show agent info

When agent coordination is active, `corvia status` should display:
- Active agents count
- Active sessions count
- Merge queue depth
- Recent merge activity

### Step 2: Update `corvia serve` with `--mcp` flag

Add `--mcp` boolean flag that starts the MCP server alongside (or instead of) the REST server.

### Step 3: Add `corvia agent` subcommand

```
corvia agent list           — list registered agents
corvia agent sessions <id>  — list sessions for an agent
```

### Step 4: Run tests, verify pass, commit

```bash
git add crates/corvia-cli/src/main.rs
git commit -m "feat(m2): CLI updates for agent status and MCP serve mode"
```

---

## Task 13: Integration Tests

**Files:**
- Create: `tests/integration/agent_e2e_test.rs`

### Step 1: Write full lifecycle integration test

```rust
#[tokio::test]
async fn test_full_agent_lifecycle_e2e() {
    // 1. Create AgentCoordinator with LiteStore + MockEngine in tempdir
    // 2. Register agent "test::indexer"
    // 3. Create session
    // 4. Write 3 knowledge entries
    // 5. Search — verify entries visible with Own visibility
    // 6. Commit session
    // 7. Process merge queue (no conflicts expected)
    // 8. Verify entries are Merged status
    // 9. Search — verify entries still visible
    // 10. Session should be Closed
}
```

### Step 2: Write multi-agent concurrent test

```rust
#[tokio::test]
async fn test_two_agents_concurrent_writes() {
    // 1. Register agent-A and agent-B
    // 2. Both create sessions
    // 3. Both write entries to same scope
    // 4. agent-A commits, agent-B commits
    // 5. Process merge queue
    // 6. Verify all entries merged (no conflict — different content)
    // 7. Verify git log shows both agents' branches
}
```

### Step 3: Write crash recovery test

```rust
#[tokio::test]
async fn test_crash_recovery_resume() {
    // 1. Register agent, create session, write entries
    // 2. Mark session Orphaned (simulating crash)
    // 3. Connect again
    // 4. Resume orphaned session
    // 5. Verify entries re-indexed from staging files
    // 6. Commit and merge
    // 7. Verify all entries in main knowledge base
}
```

### Step 4: Write conflict merge test

```rust
#[tokio::test]
async fn test_conflicting_entries_trigger_llm_merge() {
    // 1. Insert entry A about "auth" with embedding [1,0,0]
    // 2. Agent writes entry B about "auth" with very similar embedding
    // 3. Commit and process merge
    // 4. Merge worker detects conflict (similarity > 0.85)
    // 5. If Ollama available: LLM merge produces combined entry
    // 6. If not: entry stays in queue with retry_count
}
```

### Step 5: Run all tests

Run: `cargo test --workspace`
Expected: All tests pass (new M2 tests + existing M1 tests).

### Step 6: Commit

```bash
git add tests/integration/agent_e2e_test.rs
git commit -m "test(m2): integration tests for agent lifecycle, concurrency, and recovery"
```

---

## Dependency Summary

### New crate dependencies:
- **corvia-kernel**: none (uses existing `redb`, `tokio`, `reqwest`, `serde_json`)
- **corvia-server**: potentially `rmcp` or manual JSON-RPC (research in Task 11 Step 1)
- **corvia-common**: `chrono` already present, `uuid` already present

### New files:
```
crates/corvia-common/src/agent_types.rs          (Task 1)
crates/corvia-kernel/src/agent_registry.rs        (Task 2)
crates/corvia-kernel/src/session_manager.rs       (Task 3)
crates/corvia-kernel/src/staging.rs               (Task 4)
crates/corvia-kernel/src/agent_writer.rs          (Task 5)
crates/corvia-kernel/src/merge_queue.rs           (Task 6)
crates/corvia-kernel/src/commit_pipeline.rs       (Task 6)
crates/corvia-kernel/src/merge_worker.rs          (Task 7)
crates/corvia-kernel/src/context_builder.rs       (Task 8)
crates/corvia-kernel/src/agent_coordinator.rs     (Task 9)
crates/corvia-server/src/mcp.rs                   (Task 11)
tests/integration/agent_e2e_test.rs               (Task 13)
```

### Modified files:
```
crates/corvia-common/src/lib.rs                   (Task 1)
crates/corvia-common/src/types.rs                 (Task 1)
crates/corvia-common/src/config.rs                (Task 1)
crates/corvia-common/src/errors.rs                (Task 1)
crates/corvia-kernel/src/lib.rs                   (Tasks 2-9)
crates/corvia-server/src/rest.rs                  (Task 10)
crates/corvia-server/src/lib.rs                   (Task 11)
crates/corvia-server/Cargo.toml                   (Task 11)
crates/corvia-cli/src/main.rs                     (Task 12)
```

### Runtime directory structure (post-M2):
```
.corvia/
├── coordination.redb              ← NEW: agent/session/queue metadata
├── lite_store.redb                ← existing: knowledge entries + HNSW mappings
├── knowledge/                     ← existing: merged knowledge files
│   └── {scope_id}/
│       └── {entry_id}.json
├── staging/                       ← NEW: per-agent staging directories
│   └── {agent_id_sanitized}/
│       └── {session_short}/
│           └── {entry_id}.json
└── .gitignore
```
