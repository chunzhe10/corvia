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

/// Sanitize an agent ID for use in file paths and git branch names.
/// Replaces `::` with `-`, removes other unsafe characters.
pub fn sanitize_agent_id(agent_id: &str) -> String {
    agent_id.replace("::", "-").replace(['/', '\\', ' '], "-")
}

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
