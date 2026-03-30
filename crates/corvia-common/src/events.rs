use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use crate::errors::{CorviaError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub ts: DateTime<Utc>,
    #[serde(rename = "type")]
    pub event_type: EventType,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    ChunksIndexed,
    SearchPerformed,
    IngestionStarted,
    IngestionCompleted,
    MergeCompleted,
    MergeFailed,
    EntryCommitted,
    SessionOpened,
    SessionClosed,
    GcCompleted,
}

impl Event {
    pub fn new(event_type: EventType, data: serde_json::Value) -> Self {
        Self {
            ts: Utc::now(),
            event_type,
            data,
        }
    }
}

/// Append an event to events.jsonl
pub fn append_event(path: &Path, event: &Event) -> Result<()> {
    let line = serde_json::to_string(event)
        .map_err(|e| CorviaError::Storage(format!("Failed to serialize event: {e}")))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| CorviaError::Storage(format!("Failed to open events file: {e}")))?;
    writeln!(file, "{line}")
        .map_err(|e| CorviaError::Storage(format!("Failed to write event: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = Event::new(EventType::ChunksIndexed, serde_json::json!({"count": 42}));
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("ChunksIndexed"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_append_event() {
        let path = std::path::Path::new("/tmp/corvia-test-events.jsonl");
        let _ = std::fs::remove_file(path); // clean up from previous runs
        let event = Event::new(EventType::SearchPerformed, serde_json::json!({"query": "test"}));
        append_event(path, &event).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("SearchPerformed"));
    }
}
