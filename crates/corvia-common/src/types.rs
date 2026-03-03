use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single unit of knowledge stored in Corvia.
/// Bi-temporal: tracks both when the knowledge was true (valid_from/valid_to)
/// and when it was recorded (recorded_at). See design doc D14.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeEntry {
    pub id: Uuid,
    pub content: String,
    pub source_version: String,
    pub scope_id: String,
    pub workstream: String,
    pub recorded_at: DateTime<Utc>,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub superseded_by: Option<Uuid>,
    pub embedding: Option<Vec<f32>>,
    pub metadata: EntryMetadata,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub entry_status: crate::agent_types::EntryStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EntryMetadata {
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

impl KnowledgeEntry {
    pub fn new(content: String, scope_id: String, source_version: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            content,
            source_version,
            scope_id,
            workstream: "main".to_string(),
            recorded_at: now,
            valid_from: now,
            valid_to: None,
            superseded_by: None,
            embedding: None,
            metadata: EntryMetadata::default(),
            agent_id: None,
            session_id: None,
            entry_status: crate::agent_types::EntryStatus::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: EntryMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_agent(mut self, agent_id: String, session_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self.session_id = Some(session_id);
        self.entry_status = crate::agent_types::EntryStatus::Pending;
        self
    }

    /// Returns true if this entry is currently valid (not superseded).
    pub fn is_current(&self) -> bool {
        self.valid_to.is_none() && self.superseded_by.is_none()
    }
}

/// Result from a semantic search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub entry: KnowledgeEntry,
    pub score: f32,
}

/// A directed edge in the knowledge graph (D37).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub relation: String,
    pub metadata: Option<serde_json::Value>,
}

/// Direction for graph edge queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// A chat message for LLM inference (used by GenerationEngine implementations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_entry_new_defaults() {
        let entry = KnowledgeEntry::new(
            "fn hello() {}".to_string(),
            "my-repo".to_string(),
            "abc123".to_string(),
        );
        assert_eq!(entry.content, "fn hello() {}");
        assert_eq!(entry.scope_id, "my-repo");
        assert_eq!(entry.workstream, "main");
        assert!(entry.is_current());
        assert!(entry.embedding.is_none());
    }

    #[test]
    fn test_knowledge_entry_with_embedding() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        assert_eq!(entry.embedding.unwrap().len(), 3);
    }

    #[test]
    fn test_knowledge_entry_is_current() {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        assert!(entry.is_current());

        entry.valid_to = Some(Utc::now());
        assert!(!entry.is_current());
    }

    #[test]
    fn test_knowledge_entry_serialization() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: KnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry.id, deserialized.id);
        assert_eq!(entry.content, deserialized.content);
    }
}
