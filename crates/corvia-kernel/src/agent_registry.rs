use corvia_common::agent_types::*;
use corvia_common::errors::{CorviaError, Result};
use redb::{Database, ReadableTable, TableDefinition};
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
    /// Uses a single write transaction to atomically check-and-insert,
    /// preventing TOCTOU races from concurrent registrations.
    pub fn register(
        &self,
        agent_id: &str,
        display_name: &str,
        identity_type: IdentityType,
        permissions: AgentPermission,
    ) -> Result<AgentRecord> {
        let write_txn = self.db.begin_write()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin write txn: {e}")))?;
        {
            let mut table = write_txn.open_table(AGENTS)
                .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;

            // Check existence within the write transaction (holds exclusive lock)
            if let Some(existing_val) = table.get(agent_id)
                .map_err(|e| CorviaError::Agent(format!("Failed to check agent: {e}")))?
            {
                let existing: AgentRecord = serde_json::from_slice(existing_val.value())
                    .map_err(|e| CorviaError::Agent(format!("Failed to deserialize agent: {e}")))?;
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
                description: None,
                activity_summary: None,
            };

            let bytes = serde_json::to_vec(&record)
                .map_err(|e| CorviaError::Agent(format!("Failed to serialize agent: {e}")))?;
            table.insert(agent_id, bytes.as_slice())
                .map_err(|e| CorviaError::Agent(format!("Failed to insert agent: {e}")))?;

            // Commit before returning to ensure the write is durable
            drop(table);
            write_txn.commit()
                .map_err(|e| CorviaError::Agent(format!("Failed to commit agent: {e}")))?;

            info!(agent_id, "agent_registered");
            Ok(record)
        }
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

    /// Update agent description.
    pub fn set_description(&self, agent_id: &str, description: &str) -> Result<()> {
        let mut record = self.get(agent_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Agent {agent_id} not found")))?;
        record.description = Some(description.to_string());
        self.put(&record)
    }

    /// Update agent activity summary.
    pub fn set_activity_summary(&self, agent_id: &str, summary: &ActivitySummary) -> Result<()> {
        let mut record = self.get(agent_id)?
            .ok_or_else(|| CorviaError::NotFound(format!("Agent {agent_id} not found")))?;
        record.activity_summary = Some(summary.clone());
        self.put(&record)
    }

    /// List all agents with Active status.
    pub fn list_active(&self) -> Result<Vec<AgentRecord>> {
        Ok(self.list_all()?
            .into_iter()
            .filter(|a| a.status == AgentStatus::Active)
            .collect())
    }

    /// List all agent records.
    pub fn list_all(&self) -> Result<Vec<AgentRecord>> {
        let read_txn = self.db.begin_read()
            .map_err(|e| CorviaError::Agent(format!("Failed to begin read txn: {e}")))?;
        let table = read_txn.open_table(AGENTS)
            .map_err(|e| CorviaError::Agent(format!("Failed to open agents table: {e}")))?;

        let mut agents = Vec::new();
        for item in table.iter()
            .map_err(|e| CorviaError::Agent(format!("Failed to iterate agents: {e}")))?
        {
            let (_key, val) = item
                .map_err(|e| CorviaError::Agent(format!("Failed to read agent: {e}")))?;
            let bytes: &[u8] = val.value();
            let record: AgentRecord = serde_json::from_slice(bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_agent_description_and_summary() {
        let reg = test_registry();
        reg.register("test::agent", "Test Agent", IdentityType::Registered,
            AgentPermission::ReadOnly).unwrap();

        // Set description
        reg.set_description("test::agent", "working on graph refactor").unwrap();
        let agent = reg.get("test::agent").unwrap().unwrap();
        assert_eq!(agent.description.as_deref(), Some("working on graph refactor"));

        // Set activity summary
        let summary = ActivitySummary {
            entry_count: 12,
            topic_tags: vec!["graph store".into(), "edge handling".into()],
            last_topics: vec!["merge pipeline".into()],
            last_active: chrono::Utc::now(),
            session_count: 3,
            drifted: false,
        };
        reg.set_activity_summary("test::agent", &summary).unwrap();
        let agent = reg.get("test::agent").unwrap().unwrap();
        let summary = agent.activity_summary.as_ref().expect("activity_summary should be set");
        assert_eq!(summary.entry_count, 12);
        assert_eq!(summary.topic_tags.len(), 2);
    }
}
