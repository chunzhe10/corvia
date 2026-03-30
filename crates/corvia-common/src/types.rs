use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Classification of knowledge by cognitive function.
///
/// Each type has a distinct decay profile (alpha values for power-law decay):
/// - Structural (α=0): Code signatures, API shapes — no decay, refreshed on re-ingestion
/// - Decisional (α=0.15): Design decisions, architectural choices — very slow decay
/// - Episodic (α=0.60): Session discoveries, agent activity — fast decay unless reinforced
/// - Analytical (α=0.30): Synthesized findings, health checks — medium decay
/// - Procedural (α=0.10): Instructions, workflows — slowest decay, reinforced by access
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Structural,
    Decisional,
    #[default]
    Episodic,
    Analytical,
    Procedural,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural => write!(f, "structural"),
            Self::Decisional => write!(f, "decisional"),
            Self::Episodic => write!(f, "episodic"),
            Self::Analytical => write!(f, "analytical"),
            Self::Procedural => write!(f, "procedural"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "structural" => Ok(Self::Structural),
            "decisional" => Ok(Self::Decisional),
            "episodic" => Ok(Self::Episodic),
            "analytical" => Ok(Self::Analytical),
            "procedural" => Ok(Self::Procedural),
            _ => Err(format!(
                "Unknown memory type: {s}. Valid: structural, decisional, episodic, analytical, procedural"
            )),
        }
    }
}

/// Lifecycle tier for knowledge entries.
///
/// - Hot: HNSW indexed, full retrieval participation
/// - Warm: HNSW indexed, retrieval deprioritized
/// - Cold: NOT in HNSW, embedding preserved, searchable via brute-force fallback
/// - Forgotten: Compacted into summary entry, original archived
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Compacted into summary entry, original archived. Only irreversible step.
    Forgotten,
    /// NOT in HNSW, embedding preserved, searchable via brute-force fallback.
    Cold,
    /// HNSW indexed, retrieval deprioritized.
    Warm,
    /// HNSW indexed, full retrieval participation.
    #[default]
    Hot,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cold => write!(f, "cold"),
            Self::Forgotten => write!(f, "forgotten"),
        }
    }
}

impl std::str::FromStr for Tier {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "hot" => Ok(Self::Hot),
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            "forgotten" => Ok(Self::Forgotten),
            _ => Err(format!(
                "Unknown tier: {s}. Valid: hot, warm, cold, forgotten"
            )),
        }
    }
}

impl MemoryType {
    /// Try to map a content_role string to a MemoryType.
    ///
    /// Returns `Some(MemoryType)` for known roles, `None` for unrecognized values.
    fn try_from_role(role: &str) -> Option<Self> {
        match role {
            "code" => Some(Self::Structural),
            "design" | "decision" | "plan" => Some(Self::Decisional),
            "memory" | "learning" => Some(Self::Episodic),
            "finding" => Some(Self::Analytical),
            "instruction" | "task" => Some(Self::Procedural),
            _ => None,
        }
    }

    /// Classify a content_role string into a MemoryType.
    ///
    /// Returns the mapped MemoryType, or `Episodic` (default) for `None` or unrecognized values.
    /// Logs a warning for unrecognized non-None values.
    pub fn from_content_role(content_role: Option<&str>) -> Self {
        match content_role {
            None => Self::default(),
            Some(role) => Self::try_from_role(role).unwrap_or_else(|| {
                tracing::warn!(content_role = role, "unmapped content_role, defaulting to Episodic");
                Self::default()
            }),
        }
    }

    /// Returns true if the given content_role maps to a known MemoryType.
    pub fn is_known_content_role(role: &str) -> bool {
        Self::try_from_role(role).is_some()
    }
}

/// Classify a content_role string into a MemoryType.
///
/// Convenience wrapper for [`MemoryType::from_content_role`].
pub fn classify_memory_type(content_role: Option<&str>) -> MemoryType {
    MemoryType::from_content_role(content_role)
}

/// Returns true if the given content_role is in the known mapping table.
///
/// Convenience wrapper for [`MemoryType::is_known_content_role`].
pub fn is_mapped_content_role(role: &str) -> bool {
    MemoryType::is_known_content_role(role)
}

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
    #[serde(skip_serializing)]
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub metadata: EntryMetadata,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub entry_status: crate::agent_types::EntryStatus,
    // Tiered knowledge fields
    #[serde(default)]
    pub memory_type: MemoryType,
    /// Confidence score for decisional/analytical entries. Value must be in `[0.0, 1.0]`.
    /// Used in decay math: `effective_age = actual_age / confidence`.
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub last_accessed: Option<DateTime<Utc>>,
    #[serde(default)]
    pub access_count: u32,
    #[serde(default)]
    pub tier: Tier,
    #[serde(default)]
    pub tier_changed_at: Option<DateTime<Utc>>,
    /// Composite retention score (0.0-1.0). Higher = more likely to stay in current tier.
    /// Computed by GC workers: `0.35×decay + 0.30×access + 0.20×graph + 0.15×confidence`.
    #[serde(default)]
    pub retention_score: Option<f32>,
    /// Pin state. `Some(info)` = pinned (exempt from GC demotion), `None` = not pinned.
    /// Pinned entries cannot be demoted below their current tier.
    #[serde(default)]
    pub pin: Option<PinInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EntryMetadata {
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    #[serde(default)]
    pub content_role: Option<String>,
    #[serde(default)]
    pub source_origin: Option<String>,
}

/// Information about a pinned entry. Presence indicates the entry is pinned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinInfo {
    /// Agent or user identifier that pinned this entry.
    pub by: String,
    /// When the entry was pinned.
    pub at: DateTime<Utc>,
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
            memory_type: MemoryType::default(),
            confidence: None,
            last_accessed: None,
            access_count: 0,
            tier: Tier::default(),
            tier_changed_at: None,
            retention_score: None,
            pin: None,
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

    pub fn with_memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type;
        self
    }

    /// Set confidence score. Value is clamped to `[0.0, 1.0]`; NaN is rejected (set to None).
    pub fn with_confidence(mut self, value: f32) -> Self {
        if value.is_nan() {
            self.confidence = None;
        } else {
            self.confidence = Some(value.clamp(0.0, 1.0));
        }
        self
    }

    /// Auto-classify memory_type from the entry's content_role (mutates in place).
    pub fn auto_classify(&mut self) {
        self.memory_type = MemoryType::from_content_role(self.metadata.content_role.as_deref());
    }

    /// Auto-classify memory_type from the entry's content_role (builder chain).
    pub fn with_auto_classify(mut self) -> Self {
        self.auto_classify();
        self
    }

    /// Record an access to this entry. Uses saturating increment to prevent overflow.
    pub fn record_access(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_accessed = Some(Utc::now());
    }

    /// Pin this entry, exempting it from GC demotion.
    ///
    /// # Panics
    /// Panics in debug mode if `by` is empty.
    pub fn pin(&mut self, by: impl Into<String>) {
        let by = by.into();
        debug_assert!(!by.is_empty(), "pin `by` must not be empty");
        self.pin = Some(PinInfo {
            by,
            at: Utc::now(),
        });
    }

    /// Unpin this entry, allowing GC demotion.
    pub fn unpin(&mut self) {
        self.pin = None;
    }

    /// Returns true if this entry is pinned.
    pub fn is_pinned(&self) -> bool {
        self.pin.is_some()
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
    /// Knowledge tier at retrieval time.
    #[serde(default)]
    pub tier: Tier,
    /// Composite retention score at retrieval time (0.0–1.0).
    #[serde(default)]
    pub retention_score: Option<f32>,
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

    #[test]
    fn test_entry_metadata_new_fields_default() {
        // Existing JSON without new fields should deserialize with None defaults
        let json = r#"{"source_file":"lib.rs","language":"rust","chunk_type":"function","start_line":1,"end_line":10}"#;
        let meta: EntryMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.source_file, Some("lib.rs".into()));
        assert!(meta.content_role.is_none());
        assert!(meta.source_origin.is_none());
    }

    #[test]
    fn test_entry_metadata_new_fields_roundtrip() {
        let meta = EntryMetadata {
            source_file: Some("design.md".into()),
            language: Some("markdown".into()),
            chunk_type: Some("section".into()),
            start_line: Some(1),
            end_line: Some(50),
            content_role: Some("design".into()),
            source_origin: Some("repo:corvia".into()),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let roundtrip: EntryMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.content_role, Some("design".into()));
        assert_eq!(roundtrip.source_origin, Some("repo:corvia".into()));
    }

    #[test]
    fn test_memory_type_default() {
        assert_eq!(MemoryType::default(), MemoryType::Episodic);
    }

    #[test]
    fn test_tier_default() {
        assert_eq!(Tier::default(), Tier::Hot);
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(MemoryType::Structural.to_string(), "structural");
        assert_eq!(MemoryType::Decisional.to_string(), "decisional");
        assert_eq!(MemoryType::Episodic.to_string(), "episodic");
        assert_eq!(MemoryType::Analytical.to_string(), "analytical");
        assert_eq!(MemoryType::Procedural.to_string(), "procedural");
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(Tier::Hot.to_string(), "hot");
        assert_eq!(Tier::Warm.to_string(), "warm");
        assert_eq!(Tier::Cold.to_string(), "cold");
        assert_eq!(Tier::Forgotten.to_string(), "forgotten");
    }

    #[test]
    fn test_memory_type_serde_roundtrip() {
        let mt = MemoryType::Decisional;
        let json = serde_json::to_string(&mt).unwrap();
        assert_eq!(json, "\"decisional\"");
        let back: MemoryType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, MemoryType::Decisional);
    }

    #[test]
    fn test_tier_serde_roundtrip() {
        let t = Tier::Cold;
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"cold\"");
        let back: Tier = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Tier::Cold);
    }

    #[test]
    fn test_classify_memory_type_mappings() {
        assert_eq!(classify_memory_type(Some("code")), MemoryType::Structural);
        assert_eq!(classify_memory_type(Some("design")), MemoryType::Decisional);
        assert_eq!(classify_memory_type(Some("decision")), MemoryType::Decisional);
        assert_eq!(classify_memory_type(Some("plan")), MemoryType::Decisional);
        assert_eq!(classify_memory_type(Some("memory")), MemoryType::Episodic);
        assert_eq!(classify_memory_type(Some("learning")), MemoryType::Episodic);
        assert_eq!(classify_memory_type(Some("finding")), MemoryType::Analytical);
        assert_eq!(classify_memory_type(Some("instruction")), MemoryType::Procedural);
    }

    #[test]
    fn test_classify_memory_type_defaults() {
        assert_eq!(classify_memory_type(None), MemoryType::Episodic);
        assert_eq!(classify_memory_type(Some("unknown_role")), MemoryType::Episodic);
    }

    #[test]
    fn test_is_mapped_content_role() {
        assert!(is_mapped_content_role("code"));
        assert!(is_mapped_content_role("design"));
        assert!(is_mapped_content_role("task"));
        assert!(!is_mapped_content_role("unknown"));
        assert!(!is_mapped_content_role(""));
    }

    #[test]
    fn test_knowledge_entry_new_tiered_defaults() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        assert_eq!(entry.memory_type, MemoryType::Episodic);
        assert_eq!(entry.tier, Tier::Hot);
        assert_eq!(entry.access_count, 0);
        assert!(entry.confidence.is_none());
        assert!(entry.last_accessed.is_none());
        assert!(entry.tier_changed_at.is_none());
        assert!(entry.retention_score.is_none());
        assert!(!entry.is_pinned());
        assert!(entry.pin.is_none());
    }

    #[test]
    fn test_knowledge_entry_backward_compat_deserialization() {
        // JSON from before tiered knowledge fields existed — must deserialize without error
        let json = r#"{
            "id": "01960000-0000-7000-8000-000000000001",
            "content": "old entry",
            "source_version": "v1",
            "scope_id": "test",
            "workstream": "main",
            "recorded_at": "2026-01-01T00:00:00Z",
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": null,
            "superseded_by": null,
            "embedding": null,
            "metadata": {}
        }"#;
        let entry: KnowledgeEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.content, "old entry");
        assert_eq!(entry.memory_type, MemoryType::Episodic);
        assert_eq!(entry.tier, Tier::Hot);
        assert_eq!(entry.access_count, 0);
        assert!(!entry.is_pinned());
    }

    #[test]
    fn test_knowledge_entry_auto_classify() {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        entry.metadata.content_role = Some("design".into());
        entry.auto_classify();
        assert_eq!(entry.memory_type, MemoryType::Decisional);
    }

    #[test]
    fn test_knowledge_entry_with_memory_type() {
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_memory_type(MemoryType::Procedural);
        assert_eq!(entry.memory_type, MemoryType::Procedural);
    }

    #[test]
    fn test_memory_type_from_str() {
        assert_eq!("structural".parse::<MemoryType>().unwrap(), MemoryType::Structural);
        assert_eq!("decisional".parse::<MemoryType>().unwrap(), MemoryType::Decisional);
        assert_eq!("episodic".parse::<MemoryType>().unwrap(), MemoryType::Episodic);
        assert_eq!("analytical".parse::<MemoryType>().unwrap(), MemoryType::Analytical);
        assert_eq!("procedural".parse::<MemoryType>().unwrap(), MemoryType::Procedural);
        assert!("unknown".parse::<MemoryType>().is_err());
    }

    #[test]
    fn test_tier_from_str() {
        assert_eq!("hot".parse::<Tier>().unwrap(), Tier::Hot);
        assert_eq!("warm".parse::<Tier>().unwrap(), Tier::Warm);
        assert_eq!("cold".parse::<Tier>().unwrap(), Tier::Cold);
        assert_eq!("forgotten".parse::<Tier>().unwrap(), Tier::Forgotten);
        assert!("unknown".parse::<Tier>().is_err());
    }

    #[test]
    fn test_tier_ordering() {
        assert!(Tier::Hot > Tier::Warm);
        assert!(Tier::Warm > Tier::Cold);
        assert!(Tier::Cold > Tier::Forgotten);
    }

    #[test]
    fn test_memory_type_from_content_role() {
        assert_eq!(MemoryType::from_content_role(Some("code")), MemoryType::Structural);
        assert_eq!(MemoryType::from_content_role(Some("design")), MemoryType::Decisional);
        assert_eq!(MemoryType::from_content_role(Some("task")), MemoryType::Procedural);
        assert_eq!(MemoryType::from_content_role(None), MemoryType::Episodic);
        assert!(MemoryType::is_known_content_role("code"));
        assert!(MemoryType::is_known_content_role("task"));
        assert!(!MemoryType::is_known_content_role("unknown"));
    }

    #[test]
    fn test_knowledge_entry_pin_unpin() {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        assert!(!entry.is_pinned());

        entry.pin("agent-1");
        assert!(entry.is_pinned());
        assert_eq!(entry.pin.as_ref().unwrap().by, "agent-1");

        entry.unpin();
        assert!(!entry.is_pinned());
        assert!(entry.pin.is_none());
    }

    #[test]
    fn test_knowledge_entry_with_auto_classify() {
        let mut entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into());
        entry.metadata.content_role = Some("instruction".into());
        let entry = entry.with_auto_classify();
        assert_eq!(entry.memory_type, MemoryType::Procedural);
    }

    #[test]
    fn test_pin_info_serde_roundtrip() {
        let pin = PinInfo { by: "agent-1".into(), at: Utc::now() };
        let json = serde_json::to_string(&pin).unwrap();
        let back: PinInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.by, "agent-1");
    }

    #[test]
    fn test_memory_type_display_fromstr_roundtrip() {
        for mt in &[MemoryType::Structural, MemoryType::Decisional, MemoryType::Episodic,
                     MemoryType::Analytical, MemoryType::Procedural] {
            let s = mt.to_string();
            let back: MemoryType = s.parse().unwrap();
            assert_eq!(*mt, back);
        }
    }

    #[test]
    fn test_tier_display_fromstr_roundtrip() {
        for t in &[Tier::Hot, Tier::Warm, Tier::Cold, Tier::Forgotten] {
            let s = t.to_string();
            let back: Tier = s.parse().unwrap();
            assert_eq!(*t, back);
        }
    }

    #[test]
    fn test_with_confidence_clamps_range() {
        let entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into())
            .with_confidence(0.5);
        assert_eq!(entry.confidence, Some(0.5));

        let entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into())
            .with_confidence(5.0);
        assert_eq!(entry.confidence, Some(1.0));

        let entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into())
            .with_confidence(-1.0);
        assert_eq!(entry.confidence, Some(0.0));

        let entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into())
            .with_confidence(f32::NAN);
        assert!(entry.confidence.is_none());
    }

    #[test]
    fn test_record_access_increments_and_timestamps() {
        let mut entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into());
        assert_eq!(entry.access_count, 0);
        assert!(entry.last_accessed.is_none());

        entry.record_access();
        assert_eq!(entry.access_count, 1);
        assert!(entry.last_accessed.is_some());
    }

    #[test]
    fn test_record_access_saturates_at_max() {
        let mut entry = KnowledgeEntry::new("t".into(), "s".into(), "v".into());
        entry.access_count = u32::MAX;
        entry.record_access();
        assert_eq!(entry.access_count, u32::MAX);
    }

    #[test]
    fn test_from_str_case_sensitive() {
        // FromStr is strictly lowercase — mixed case is rejected
        assert!("Structural".parse::<MemoryType>().is_err());
        assert!("HOT".parse::<Tier>().is_err());
        assert!("Warm".parse::<Tier>().is_err());
    }

    #[test]
    fn test_embedding_skip_serializing() {
        // Embedding is skip_serializing: set in memory but NOT written to JSON
        let entry = KnowledgeEntry::new("test".into(), "scope".into(), "v1".into())
            .with_embedding(vec![0.1, 0.2, 0.3]);
        let json = serde_json::to_string(&entry).unwrap();
        assert!(!json.contains("embedding"), "embedding should not appear in serialized JSON");
        let deserialized: KnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.embedding, None);
    }

    #[test]
    fn test_old_json_with_embedding_still_deserializes() {
        // Backward compat: old JSON files that contain embedding should still parse
        let json = r#"{
            "id": "01960000-0000-7000-8000-000000000001",
            "content": "old entry with embedding",
            "source_version": "v1",
            "scope_id": "test",
            "workstream": "main",
            "recorded_at": "2026-01-01T00:00:00Z",
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": null,
            "superseded_by": null,
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        }"#;
        let entry: KnowledgeEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.embedding, Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_forward_compat_deserialization_with_all_tiered_fields() {
        // JSON with all tiered fields explicitly set — tests forward compatibility
        let json = r#"{
            "id": "01960000-0000-7000-8000-000000000001",
            "content": "rich entry",
            "source_version": "v2",
            "scope_id": "test",
            "workstream": "main",
            "recorded_at": "2026-03-01T00:00:00Z",
            "valid_from": "2026-03-01T00:00:00Z",
            "valid_to": null,
            "superseded_by": null,
            "embedding": null,
            "metadata": {"content_role": "design"},
            "memory_type": "decisional",
            "confidence": 0.85,
            "last_accessed": "2026-03-15T12:00:00Z",
            "access_count": 42,
            "tier": "warm",
            "tier_changed_at": "2026-03-10T00:00:00Z",
            "retention_score": 0.72,
            "pin": {"by": "agent-1", "at": "2026-03-05T00:00:00Z"}
        }"#;
        let entry: KnowledgeEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.memory_type, MemoryType::Decisional);
        assert_eq!(entry.confidence, Some(0.85));
        assert_eq!(entry.access_count, 42);
        assert_eq!(entry.tier, Tier::Warm);
        assert_eq!(entry.retention_score, Some(0.72));
        assert!(entry.is_pinned());
        assert_eq!(entry.pin.unwrap().by, "agent-1");
    }
}
