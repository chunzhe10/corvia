//! Activity feed endpoint for the dashboard.
//!
//! Provides recent entries with semantic grouping and content deltas.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use corvia_common::constants::DEFAULT_SCOPE_ID;

use crate::rest::AppState;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ActivityItem {
    pub entry_id: String,
    pub action: String,       // "wrote", "superseded", "merged"
    pub title: String,        // source_file or content preview (80 chars)
    pub agent_id: Option<String>,
    pub agent_name: Option<String>,
    pub topic_tags: Vec<String>,
    pub delta_bytes: Option<i64>, // positive = addition, negative = deletion
    pub recorded_at: String,
    pub superseded_id: Option<String>,
    pub group_id: Option<String>,    // semantic group identifier
    pub group_count: Option<usize>,  // how many items in this group
}

#[derive(Debug, Serialize)]
pub struct ActivityFeedResponse {
    pub items: Vec<ActivityItem>,
    pub total: usize,
    pub topics: Vec<String>, // available topic filters
}

// ---------------------------------------------------------------------------
// Query params
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ActivityFeedParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub agent: Option<String>,
    pub topic: Option<String>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// GET /api/dashboard/activity
///
/// Returns a reverse-chronological activity feed with semantic grouping,
/// content deltas, and topic tags from the ClusterStore.
pub async fn activity_feed_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ActivityFeedParams>,
) -> impl IntoResponse {
    let scope_id = state.default_scope_id.as_deref().unwrap_or(DEFAULT_SCOPE_ID);
    let data_dir = &state.data_dir;

    let mut entries =
        corvia_kernel::knowledge_files::read_scope(data_dir, scope_id).unwrap_or_default();

    // Sort by recorded_at descending (newest first)
    entries.sort_by(|a, b| b.recorded_at.cmp(&a.recorded_at));

    // Apply agent filter before building items
    if let Some(ref agent) = params.agent {
        entries.retain(|e| e.agent_id.as_deref() == Some(agent.as_str()));
    }

    // Build a lookup from entry_id -> content length for delta computation.
    // A predecessor is an entry whose superseded_by field == this entry's ID.
    let predecessor_content_len: std::collections::HashMap<String, usize> = entries
        .iter()
        .filter_map(|e| {
            e.superseded_by
                .as_ref()
                .map(|succ| (succ.to_string(), e.content.len()))
        })
        .collect();

    // Ensure cluster hierarchy is computed with human-readable labels
    if state.cluster_store.is_degraded() {
        let pairs: Vec<(String, Vec<f32>)> = entries
            .iter()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.to_string(), emb.clone())))
            .collect();
        let label_map: std::collections::HashMap<String, String> = entries
            .iter()
            .map(|e| {
                let label = e.metadata.source_file.clone()
                    .unwrap_or_else(|| e.content.chars().take(60).collect());
                (e.id.to_string(), label)
            })
            .collect();
        state.cluster_store.maybe_recompute_with_labels(&pairs, &label_map);
    }
    let hierarchy = state.cluster_store.current();

    // Build entry label map for resolving UUID cluster labels to human-readable names
    let entry_labels: std::collections::HashMap<String, String> = entries
        .iter()
        .map(|e| {
            let label = e.metadata.source_file.clone()
                .unwrap_or_else(|| e.content.chars().take(60).collect());
            (e.id.to_string(), label)
        })
        .collect();

    let mut items: Vec<ActivityItem> = entries
        .iter()
        .skip(params.offset.unwrap_or(0))
        .take(params.limit.unwrap_or(50))
        .map(|entry| {
            let entry_id_str = entry.id.to_string();

            // Topic tags from ClusterStore — resolve UUID labels via entry_labels map
            let topic_tags = hierarchy
                .as_ref()
                .and_then(|h| {
                    h.cluster_for_entry(&entry_id_str)
                        .map(|sc| {
                            let label = entry_labels
                                .get(&sc.label)
                                .cloned()
                                .unwrap_or_else(|| sc.label.clone());
                            vec![label]
                        })
                })
                .unwrap_or_default();

            // Action verb
            let action = if entry.superseded_by.is_some() {
                "superseded"
            } else {
                "wrote"
            };

            // Title: prefer source_file, fall back to content preview
            let title = entry
                .metadata
                .source_file
                .clone()
                .unwrap_or_else(|| entry.content.chars().take(80).collect());

            // Content delta: UTF-8 byte diff vs predecessor
            let current_bytes = entry.content.len() as i64;
            let delta_bytes = match predecessor_content_len.get(&entry_id_str) {
                Some(&prev) => current_bytes - prev as i64,
                None => current_bytes, // New entry — full content is the delta
            };

            ActivityItem {
                entry_id: entry_id_str,
                action: action.to_string(),
                title,
                agent_id: entry.agent_id.clone(),
                agent_name: None, // Could be enriched from registry in the future
                topic_tags,
                delta_bytes: Some(delta_bytes),
                recorded_at: entry.recorded_at.to_rfc3339(),
                superseded_id: entry.superseded_by.map(|id| id.to_string()),
                group_id: None,
                group_count: None,
            }
        })
        .collect();

    // Apply semantic grouping
    group_activity_items(&mut items);

    // Collect available topics from all super-clusters — resolve labels
    let topics: Vec<String> = hierarchy
        .map(|h| {
            h.super_clusters
                .iter()
                .map(|sc| {
                    entry_labels
                        .get(&sc.label)
                        .cloned()
                        .unwrap_or_else(|| sc.label.clone())
                })
                .collect()
        })
        .unwrap_or_default();

    // Apply topic filter (after grouping so group metadata is intact)
    let items = if let Some(ref topic) = params.topic {
        items
            .into_iter()
            .filter(|i| i.topic_tags.contains(topic))
            .collect()
    } else {
        items
    };

    let total = items.len();
    Json(ActivityFeedResponse {
        items,
        total,
        topics,
    })
}

// ---------------------------------------------------------------------------
// Semantic grouping
// ---------------------------------------------------------------------------

/// Groups adjacent activity items that share the same agent + topic within a
/// 5-minute window. Items in a group share a `group_id` and `group_count`.
pub fn group_activity_items(items: &mut Vec<ActivityItem>) {
    let mut i = 0;
    while i < items.len() {
        let mut group_size = 1usize;
        let mut j = i + 1;
        while j < items.len() {
            let same_agent = items[i].agent_id == items[j].agent_id;
            let same_topic = !items[i].topic_tags.is_empty()
                && items[i].topic_tags.first()
                    == items[j].topic_tags.first();

            let time_i: DateTime<Utc> = items[i]
                .recorded_at
                .parse()
                .unwrap_or_default();
            let time_j: DateTime<Utc> = items[j]
                .recorded_at
                .parse()
                .unwrap_or_default();
            let within_5min = (time_i - time_j).num_seconds().abs() < 300;

            if same_agent && (same_topic || within_5min) {
                group_size += 1;
                j += 1;
            } else {
                break;
            }
        }

        if group_size > 1 {
            let group_id = format!("group-{i}");
            for k in i..j {
                items[k].group_id = Some(group_id.clone());
                items[k].group_count = Some(group_size);
            }
        }
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(
        entry_id: &str,
        agent_id: Option<&str>,
        topic: Option<&str>,
        recorded_at: DateTime<Utc>,
    ) -> ActivityItem {
        ActivityItem {
            entry_id: entry_id.into(),
            action: "wrote".into(),
            title: "test".into(),
            agent_id: agent_id.map(String::from),
            agent_name: None,
            topic_tags: topic.map(|t| vec![t.to_string()]).unwrap_or_default(),
            delta_bytes: Some(100),
            recorded_at: recorded_at.to_rfc3339(),
            superseded_id: None,
            group_id: None,
            group_count: None,
        }
    }

    #[test]
    fn test_group_activity_items_same_agent_same_topic() {
        let now = chrono::Utc::now();
        let mut items = vec![
            make_item("a", Some("agent1"), Some("graph"), now),
            make_item(
                "b",
                Some("agent1"),
                Some("graph"),
                now - chrono::Duration::seconds(60),
            ),
        ];
        group_activity_items(&mut items);
        assert!(items[0].group_id.is_some());
        assert_eq!(items[0].group_count, Some(2));
        assert_eq!(items[0].group_id, items[1].group_id);
    }

    #[test]
    fn test_no_group_different_agents() {
        let now = chrono::Utc::now();
        let mut items = vec![
            make_item("a", Some("agent1"), Some("graph"), now),
            make_item(
                "b",
                Some("agent2"),
                Some("graph"),
                now - chrono::Duration::seconds(60),
            ),
        ];
        group_activity_items(&mut items);
        assert!(items[0].group_id.is_none());
        assert!(items[1].group_id.is_none());
    }

    #[test]
    fn test_group_same_agent_within_5min_different_topic() {
        let now = chrono::Utc::now();
        let mut items = vec![
            make_item("a", Some("agent1"), Some("graph"), now),
            make_item(
                "b",
                Some("agent1"),
                Some("merge"),
                now - chrono::Duration::seconds(120),
            ),
        ];
        group_activity_items(&mut items);
        // Same agent, within 5 min => should group
        assert!(items[0].group_id.is_some());
        assert_eq!(items[0].group_count, Some(2));
    }

    #[test]
    fn test_no_group_same_agent_outside_5min_different_topic() {
        let now = chrono::Utc::now();
        let mut items = vec![
            make_item("a", Some("agent1"), Some("graph"), now),
            make_item(
                "b",
                Some("agent1"),
                Some("merge"),
                now - chrono::Duration::seconds(600),
            ),
        ];
        group_activity_items(&mut items);
        // Different topics AND outside 5-minute window => no group
        assert!(items[0].group_id.is_none());
    }

    #[test]
    fn test_group_no_topics_but_same_agent_within_5min() {
        let now = chrono::Utc::now();
        let mut items = vec![
            make_item("a", Some("agent1"), None, now),
            make_item(
                "b",
                Some("agent1"),
                None,
                now - chrono::Duration::seconds(60),
            ),
        ];
        group_activity_items(&mut items);
        // Same agent, within 5 min => should group
        assert!(items[0].group_id.is_some());
        assert_eq!(items[0].group_count, Some(2));
    }

    #[test]
    fn test_multiple_groups() {
        let now = chrono::Utc::now();
        let mut items = vec![
            // Group 1: agent1, graph, close together
            make_item("a", Some("agent1"), Some("graph"), now),
            make_item(
                "b",
                Some("agent1"),
                Some("graph"),
                now - chrono::Duration::seconds(30),
            ),
            // Group 2: agent2, merge, close together
            make_item(
                "c",
                Some("agent2"),
                Some("merge"),
                now - chrono::Duration::seconds(60),
            ),
            make_item(
                "d",
                Some("agent2"),
                Some("merge"),
                now - chrono::Duration::seconds(90),
            ),
        ];
        group_activity_items(&mut items);

        assert_eq!(items[0].group_id, items[1].group_id);
        assert_eq!(items[0].group_count, Some(2));
        assert_eq!(items[2].group_id, items[3].group_id);
        assert_eq!(items[2].group_count, Some(2));
        assert_ne!(items[0].group_id, items[2].group_id);
    }

    #[test]
    fn test_single_item_no_group() {
        let now = chrono::Utc::now();
        let mut items = vec![make_item("a", Some("agent1"), Some("graph"), now)];
        group_activity_items(&mut items);
        assert!(items[0].group_id.is_none());
        assert!(items[0].group_count.is_none());
    }

    #[test]
    fn test_empty_items() {
        let mut items: Vec<ActivityItem> = vec![];
        group_activity_items(&mut items);
        assert!(items.is_empty());
    }
}
