use corvia_common::agent_types::*;
use corvia_common::agent_types::sanitize_agent_id;
use corvia_common::errors::{CorviaError, Result};
use redb::{Database, ReadableTable, TableDefinition};
use tracing::info;

const SESSIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("sessions");

/// Manages agent sessions in the shared coordination Redb database.
/// Session ID format: `{agent_id}/sess-{uuid_short}` (first 8 chars of UUID).
pub struct SessionManager {
    db: std::sync::Arc<Database>,
}

impl SessionManager {
    /// Create from an existing shared database handle.
    pub fn from_db(db: std::sync::Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        { let _ = write_txn.open_table(SESSIONS); }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to init sessions table: {e}")))?;
        Ok(Self { db })
    }

    /// Create a new session for an agent.
    /// `with_staging`: if true, generates git branch name and staging dir path (for Registered agents).
    pub fn create(&self, agent_id: &str, with_staging: bool) -> Result<SessionRecord> {
        let full_uuid = uuid::Uuid::now_v7().to_string();
        // Use last 8 chars (random portion) to avoid collisions from same-ms timestamps
        let uuid_short = &full_uuid[full_uuid.len() - 8..];
        let session_id = format!("{agent_id}/sess-{uuid_short}");
        let now = chrono::Utc::now();

        let sanitized = sanitize_agent_id(agent_id);

        let (git_branch, staging_dir) = if with_staging {
            (
                Some(format!("agents/{sanitized}/sess-{uuid_short}")),
                Some(format!(".corvia/staging/{sanitized}/sess-{uuid_short}")),
            )
        } else {
            (None, None)
        };

        let record = SessionRecord {
            session_id: session_id.clone(),
            agent_id: agent_id.into(),
            created_at: now,
            last_heartbeat: now,
            state: SessionState::Created,
            git_branch,
            staging_dir,
            entries_written: 0,
            entries_merged: 0,
        };

        self.put(&record)?;
        info!(session_id = %session_id, agent_id, "session_created");
        Ok(record)
    }

    /// Get a session record by ID.
    pub fn get(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(SESSIONS)
            .map_err(|e| CorviaError::Agent(format!("Failed to open sessions table: {e}")))?;

        match table.get(session_id) {
            Ok(Some(val)) => {
                let bytes: &[u8] = val.value();
                let record: SessionRecord = serde_json::from_slice(bytes)
                    .map_err(|e| CorviaError::Agent(format!("Failed to deserialize session: {e}")))?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(CorviaError::Agent(format!("Failed to get session: {e}"))),
        }
    }

    /// Transition session to a new state. Validates the state machine.
    pub fn transition(&self, session_id: &str, new_state: SessionState) -> Result<()> {
        let mut record = self.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        if !record.state.can_transition_to(new_state) {
            return Err(CorviaError::Agent(format!(
                "Invalid state transition: {:?} → {:?} for session {session_id}",
                record.state, new_state
            )));
        }

        record.state = new_state;
        self.put(&record)?;
        info!(session_id, ?new_state, "session_state_changed");
        Ok(())
    }

    /// Update last_heartbeat timestamp. If session is Stale, transitions back to Active.
    pub fn heartbeat(&self, session_id: &str) -> Result<()> {
        let mut record = self.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;

        record.last_heartbeat = chrono::Utc::now();

        // Auto-recover from Stale if heartbeat resumes
        if record.state == SessionState::Stale {
            record.state = SessionState::Active;
        }

        self.put(&record)
    }

    /// Increment the entries_written counter.
    pub fn increment_written(&self, session_id: &str) -> Result<()> {
        let mut record = self.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;
        record.entries_written += 1;
        self.put(&record)
    }

    /// Increment the entries_merged counter.
    pub fn increment_merged(&self, session_id: &str) -> Result<()> {
        let mut record = self.get(session_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Session {session_id} not found")))?;
        record.entries_merged += 1;
        self.put(&record)
    }

    /// List all sessions for a specific agent.
    pub fn list_by_agent(&self, agent_id: &str) -> Result<Vec<SessionRecord>> {
        Ok(self.list_all()?
            .into_iter()
            .filter(|s| s.agent_id == agent_id)
            .collect())
    }

    /// Find Active sessions whose last_heartbeat is older than `timeout`.
    pub fn find_stale(&self, timeout: std::time::Duration) -> Result<Vec<SessionRecord>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::from_std(timeout)
            .unwrap_or(chrono::Duration::zero());
        Ok(self.list_all()?
            .into_iter()
            .filter(|s| s.state == SessionState::Active && s.last_heartbeat < cutoff)
            .collect())
    }

    /// Find Stale sessions that have been stale longer than `grace` duration.
    pub fn find_orphaned(&self, grace: std::time::Duration) -> Result<Vec<SessionRecord>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::from_std(grace)
            .unwrap_or(chrono::Duration::zero());
        Ok(self.list_all()?
            .into_iter()
            .filter(|s| s.state == SessionState::Stale && s.last_heartbeat < cutoff)
            .collect())
    }

    /// List all non-Closed sessions.
    pub fn list_open(&self) -> Result<Vec<SessionRecord>> {
        Ok(self.list_all()?
            .into_iter()
            .filter(|s| s.state != SessionState::Closed)
            .collect())
    }

    /// List all session records.
    fn list_all(&self) -> Result<Vec<SessionRecord>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(SESSIONS)
            .map_err(|e| CorviaError::Agent(format!("Failed to open sessions table: {e}")))?;

        let mut sessions = Vec::new();
        for item in table.iter()
            .map_err(|e| CorviaError::Agent(format!("Failed to iterate sessions: {e}")))?
        {
            let (_key, val) = item
                .map_err(|e| CorviaError::Agent(format!("Failed to read session: {e}")))?;
            let bytes: &[u8] = val.value();
            let record: SessionRecord = serde_json::from_slice(bytes)
                .map_err(|e| CorviaError::Agent(format!("Failed to deserialize session: {e}")))?;
            sessions.push(record);
        }
        Ok(sessions)
    }

    /// Internal: write a session record.
    fn put(&self, record: &SessionRecord) -> Result<()> {
        let bytes = serde_json::to_vec(record)
            .map_err(|e| CorviaError::Agent(format!("Failed to serialize session: {e}")))?;
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(SESSIONS)
                .map_err(|e| CorviaError::Agent(format!("Failed to open sessions table: {e}")))?;
            table.insert(record.session_id.as_str(), bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to write session: {e}")))?;
        }
        write_txn.commit()
            .map_err(|e| CorviaError::Agent(format!("Failed to commit session: {e}")))?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
