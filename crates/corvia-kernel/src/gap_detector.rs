//! Knowledge gap detection from search query analysis (Issue #47, Design Area 3).
//!
//! Captures low-confidence search queries as gap signals and aggregates them
//! into a ranked list of frequently unanswered topics. Gap signals are operational
//! telemetry, not knowledge entries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

/// Maximum gap signals to retain in the ring buffer.
const MAX_SIGNALS: usize = 1000;

/// A single low-confidence search event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapSignal {
    pub query: String,
    pub top_score: f32,
    pub result_count: usize,
    pub timestamp: DateTime<Utc>,
    pub scope_id: String,
}

/// Aggregated gap topic with frequency and recency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapTopic {
    /// Representative query for this topic cluster.
    pub query: String,
    /// Number of times this topic was searched with low confidence.
    pub frequency: usize,
    /// Most recent occurrence.
    pub last_seen: DateTime<Utc>,
    /// Average top_score across occurrences.
    pub avg_score: f32,
    pub scope_id: String,
}

/// Thread-safe gap signal accumulator with bounded ring buffer.
///
/// Records low-confidence search queries and aggregates them into ranked
/// gap topics for the dashboard.
pub struct GapDetector {
    signals: Mutex<VecDeque<GapSignal>>,
}

impl GapDetector {
    pub fn new() -> Self {
        Self {
            signals: Mutex::new(VecDeque::with_capacity(MAX_SIGNALS)),
        }
    }

    /// Record a gap signal (low-confidence search).
    pub fn record(&self, signal: GapSignal) {
        let mut signals = self.signals.lock().unwrap();
        if signals.len() >= MAX_SIGNALS {
            signals.pop_front();
        }
        signals.push_back(signal);
    }

    /// Get the number of recorded gap signals.
    pub fn len(&self) -> usize {
        self.signals.lock().unwrap().len()
    }

    /// Check if the detector has any recorded signals.
    pub fn is_empty(&self) -> bool {
        self.signals.lock().unwrap().is_empty()
    }

    /// Aggregate gap signals into ranked topics.
    ///
    /// Groups queries by normalized prefix (first 3 significant words) and
    /// returns the top-N most frequent gap topics.
    pub fn top_gaps(&self, limit: usize) -> Vec<GapTopic> {
        let signals = self.signals.lock().unwrap();

        // Group by normalized query key.
        let mut groups: std::collections::HashMap<String, Vec<&GapSignal>> =
            std::collections::HashMap::new();

        for signal in signals.iter() {
            let key = normalize_query(&signal.query);
            groups.entry(key).or_default().push(signal);
        }

        // Build GapTopic entries.
        let mut topics: Vec<GapTopic> = groups
            .into_iter()
            .map(|(_key, signals)| {
                let frequency = signals.len();
                let last_seen = signals.iter().map(|s| s.timestamp).max().unwrap();
                let avg_score = signals.iter().map(|s| s.top_score).sum::<f32>() / frequency as f32;
                let representative = signals.last().unwrap();
                GapTopic {
                    query: representative.query.clone(),
                    frequency,
                    last_seen,
                    avg_score,
                    scope_id: representative.scope_id.clone(),
                }
            })
            .collect();

        // Sort by frequency (descending), then by recency.
        topics.sort_by(|a, b| {
            b.frequency.cmp(&a.frequency)
                .then_with(|| b.last_seen.cmp(&a.last_seen))
        });
        topics.truncate(limit);
        topics
    }

    /// Get all raw gap signals (for debugging/export).
    pub fn all_signals(&self) -> Vec<GapSignal> {
        self.signals.lock().unwrap().iter().cloned().collect()
    }
}

impl Default for GapDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize a query for grouping: lowercase, take first 3 significant words.
/// Using 3 words balances specificity with grouping (4 words fragments too much).
fn normalize_query(query: &str) -> String {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "how", "what", "does",
        "do", "did", "to", "of", "in", "for", "on", "with", "and", "or",
    ];
    let words: Vec<&str> = query
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 1)
        .filter(|w| !STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .take(3)
        .collect();
    words.join(" ").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(query: &str, score: f32) -> GapSignal {
        GapSignal {
            query: query.to_string(),
            top_score: score,
            result_count: 1,
            timestamp: Utc::now(),
            scope_id: "test".to_string(),
        }
    }

    #[test]
    fn test_record_and_len() {
        let detector = GapDetector::new();
        assert!(detector.is_empty());
        detector.record(make_signal("test query", 0.3));
        assert_eq!(detector.len(), 1);
    }

    #[test]
    fn test_ring_buffer_bounds() {
        let detector = GapDetector::new();
        for i in 0..MAX_SIGNALS + 100 {
            detector.record(make_signal(&format!("query {i}"), 0.2));
        }
        assert_eq!(detector.len(), MAX_SIGNALS);
    }

    #[test]
    fn test_top_gaps_aggregation() {
        let detector = GapDetector::new();
        // Same normalized topic ("hnsw configuration parameters"), 3 times
        detector.record(make_signal("HNSW configuration parameters tuning", 0.2));
        detector.record(make_signal("HNSW configuration parameters setup", 0.3));
        detector.record(make_signal("HNSW configuration parameters options", 0.25));
        // Different topic, 1 time
        detector.record(make_signal("agent session lifecycle management", 0.1));

        let gaps = detector.top_gaps(10);
        assert!(!gaps.is_empty());
        // HNSW topic should be first (highest frequency = 3)
        assert_eq!(gaps[0].frequency, 3, "expected HNSW group to have frequency 3, got {:?}", gaps);
        assert!(gaps[0].query.contains("HNSW"));
    }

    #[test]
    fn test_top_gaps_empty() {
        let detector = GapDetector::new();
        let gaps = detector.top_gaps(10);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_normalize_query() {
        assert_eq!(normalize_query("How does the HNSW index work with embeddings?"), "hnsw index work");
        assert_eq!(normalize_query("a"), "");
        // Same prefix normalizes to same key
        assert_eq!(
            normalize_query("HNSW configuration parameters tuning"),
            normalize_query("HNSW configuration parameters setup")
        );
    }
}
