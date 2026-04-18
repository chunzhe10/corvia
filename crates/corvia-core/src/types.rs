//! Core types for the corvia knowledge system.
//!
//! Defines the foundational data structures used throughout the pipeline:
//! entries, chunks, search results, quality signals, and API responses.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Kind
// ---------------------------------------------------------------------------

/// Classification of a knowledge entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Decision,
    Learning,
    Instruction,
    Reference,
}

impl Default for Kind {
    fn default() -> Self {
        Self::Learning
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Decision => "decision",
            Self::Learning => "learning",
            Self::Instruction => "instruction",
            Self::Reference => "reference",
        };
        f.write_str(s)
    }
}

impl FromStr for Kind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "decision" => Ok(Self::Decision),
            "learning" => Ok(Self::Learning),
            "instruction" => Ok(Self::Instruction),
            "reference" => Ok(Self::Reference),
            other => Err(format!("unknown kind: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// EntryMeta / Entry
// ---------------------------------------------------------------------------

/// Metadata for a knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMeta {
    pub id: String,
    pub created_at: String,
    #[serde(default)]
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// A complete knowledge entry: metadata plus body text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub meta: EntryMeta,
    pub body: String,
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A text chunk derived from an entry, ready for embedding and indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub source_entry_id: String,
    pub text: String,
    pub chunk_index: u32,
    pub kind: Kind,
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// Search types
// ---------------------------------------------------------------------------

/// A single search result returned by the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Entry ID (UUIDv7) of the source entry.
    pub id: String,
    /// Chunk ID of the specific chunk retrieved from the entry. Format `<entry_id>:<chunk_index>`.
    pub chunk_id: String,
    pub kind: Kind,
    pub score: f32,
    pub content: String,
}

/// Confidence level for a search response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
    None,
}

/// Quality signal attached to a search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignal {
    pub confidence: Confidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Full search response: results plus quality signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub quality: QualitySignal,
}

// ---------------------------------------------------------------------------
// Write response
// ---------------------------------------------------------------------------

/// Response returned after writing a knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteResponse {
    pub id: String,
    pub action: String,
    pub superseded: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

// ---------------------------------------------------------------------------
// Status / health
// ---------------------------------------------------------------------------

/// Health metrics for the search index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    pub bm25_docs: u64,
    pub vector_count: u64,
    pub last_ingest: Option<String>,
    pub stale: bool,
}

/// A single trace entry from the OTLP JSON trace file, serialized for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub name: String,
    pub elapsed_ms: u64,
    pub timestamp_ns: u64,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
}

/// System status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub entry_count: u64,
    pub superseded_count: u64,
    pub index_health: IndexHealth,
    pub storage_path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_traces: Vec<TraceEntry>,
}

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

/// Generate a new entry ID using UUIDv7 (time-ordered).
pub fn new_entry_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_default_is_learning() {
        assert_eq!(Kind::default(), Kind::Learning);
    }

    #[test]
    fn kind_round_trip() {
        for kind in [Kind::Decision, Kind::Learning, Kind::Instruction, Kind::Reference] {
            let s = kind.to_string();
            let parsed: Kind = s.parse().unwrap();
            assert_eq!(parsed, kind, "round-trip failed for {s}");
        }
    }

    #[test]
    fn kind_invalid_parse() {
        let result = "bogus".parse::<Kind>();
        assert!(result.is_err());
    }

    #[test]
    fn new_entry_id_is_lowercase_uuid() {
        let id = new_entry_id();
        assert_eq!(id.len(), 36, "UUID should be 36 characters");
        assert_eq!(id, id.to_lowercase(), "UUID should be lowercase");
    }

    #[test]
    fn new_entry_id_is_unique() {
        let a = new_entry_id();
        let b = new_entry_id();
        assert_ne!(a, b);
    }

    #[test]
    fn kind_serde_json_roundtrip() {
        let meta = EntryMeta {
            id: new_entry_id(),
            created_at: "2026-04-15T00:00:00Z".to_string(),
            kind: Kind::Decision,
            supersedes: vec!["old-id-1".to_string()],
            tags: vec!["architecture".to_string(), "v2".to_string()],
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: EntryMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, meta.id);
        assert_eq!(deserialized.kind, Kind::Decision);
        assert_eq!(deserialized.supersedes, vec!["old-id-1"]);
        assert_eq!(deserialized.tags, vec!["architecture", "v2"]);

        // Verify lowercase serde rename
        assert!(json.contains("\"decision\""), "kind should serialize as lowercase");

        // Verify skip_serializing_if works for empty vecs
        let meta_empty = EntryMeta {
            id: new_entry_id(),
            created_at: "2026-04-15T00:00:00Z".to_string(),
            kind: Kind::default(),
            supersedes: vec![],
            tags: vec![],
        };
        let json_empty = serde_json::to_string(&meta_empty).unwrap();
        assert!(
            !json_empty.contains("supersedes"),
            "empty supersedes should be skipped"
        );
        assert!(!json_empty.contains("tags"), "empty tags should be skipped");
    }

    #[test]
    fn search_result_carries_chunk_id() {
        let r = SearchResult {
            id: "entry-1".to_string(),
            chunk_id: "entry-1:3".to_string(),
            kind: Kind::Learning,
            score: 0.5,
            content: "body".to_string(),
        };
        assert_eq!(r.chunk_id, "entry-1:3");
        assert_eq!(r.id, "entry-1");
    }
}
