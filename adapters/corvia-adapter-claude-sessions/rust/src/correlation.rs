//! Subagent-to-teammate identity correlation for Agent Teams.
//!
//! Maps teammate names to their subagent session IDs by matching Agent tool
//! call descriptions in the lead's transcript against `<teammate-message>`
//! summary tags in subagent transcripts.
//!
//! Design: RFC Section 6 (Teammate Identity Correlation).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Result of correlating teammates to their subagent sessions.
/// Maps teammate_name -> subagent_session_id.
pub type CorrelationMap = HashMap<String, String>;

/// Correlate teammate names to subagent session IDs.
///
/// Reads the lead session's transcript and its subagent transcripts from the
/// Claude sessions directory. Returns a map of teammate_name -> session_id.
///
/// # Correlation strategy (RFC Section 6)
///
/// 1. **Primary**: Match Agent tool call `description` in lead transcript to
///    `<teammate-message summary="...">` in subagent transcript first lines.
/// 2. **Fallback**: Temporal correlation using `joined_at` from config vs
///    first message timestamp from subagent transcript.
pub fn correlate_teammates(
    lead_session_id: &str,
    sessions_dir: &Path,
    members: &[(String, String)], // (name, joined_at)
) -> CorrelationMap {
    let mut result = CorrelationMap::new();

    if lead_session_id.is_empty() {
        return result;
    }

    // Step 1: Find lead's session log and extract Agent tool calls with teammate names
    let agent_calls = extract_agent_tool_calls(lead_session_id, sessions_dir);

    // Step 2: Enumerate subagent transcripts
    let subagent_dir = sessions_dir.join(lead_session_id).join("subagents");
    let subagent_info = enumerate_subagent_transcripts(&subagent_dir);

    // Step 3: Primary correlation - description matching
    for (session_id, summary) in &subagent_info {
        if let Some(teammate_name) = match_by_description(summary, &agent_calls) {
            result.insert(teammate_name, session_id.clone());
        }
    }

    // Step 4: Fallback - temporal correlation for unmatched members
    let unmatched: Vec<&(String, String)> = members
        .iter()
        .filter(|(name, _)| !result.contains_key(name))
        .collect();

    if !unmatched.is_empty() {
        let unmatched_transcripts: Vec<&(String, String)> = subagent_info
            .iter()
            .filter(|(sid, _)| !result.values().any(|v| v == sid))
            .collect();

        for (name, joined_at) in unmatched {
            if let Some(session_id) =
                match_by_temporal(joined_at, &unmatched_transcripts, sessions_dir)
            {
                result.insert(name.clone(), session_id);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Agent tool call extraction from lead transcript
// ---------------------------------------------------------------------------

/// Extracted Agent tool call info from the lead's transcript.
struct AgentCall {
    /// The teammate name from the Agent tool's `name` parameter.
    name: String,
    /// The description from the Agent tool's `description` parameter.
    description: String,
}

/// Extract Agent tool calls from the lead session's transcript.
/// Returns a list of (teammate_name, description) pairs.
fn extract_agent_tool_calls(lead_session_id: &str, sessions_dir: &Path) -> Vec<AgentCall> {
    let mut calls = Vec::new();

    // Try both .jsonl.gz and .jsonl, also check archive/
    let candidates = [sessions_dir.join(format!("{lead_session_id}.jsonl.gz")),
        sessions_dir.join(format!("{lead_session_id}.jsonl")),
        sessions_dir
            .join("archive")
            .join(format!("{lead_session_id}.jsonl.gz"))];

    let transcript_path = match candidates.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => return calls,
    };

    let content = read_transcript(&transcript_path);

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Look for tool_end events with tool="Agent"
        let event_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let tool = val.get("tool").and_then(|v| v.as_str()).unwrap_or("");

        if (event_type == "tool_start" || event_type == "tool_end") && tool == "Agent"
            && let Some(input) = val.get("input") {
                let name = input
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let description = input
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if !name.is_empty() && !description.is_empty() {
                    // Avoid duplicate entries (tool_start + tool_end)
                    if !calls.iter().any(|c: &AgentCall| c.name == name && c.description == description) {
                        calls.push(AgentCall { name, description });
                    }
                }
            }
    }

    calls
}

// ---------------------------------------------------------------------------
// Subagent transcript enumeration
// ---------------------------------------------------------------------------

/// Enumerate subagent transcripts and extract their session_id + first message summary.
/// Returns Vec<(session_id, summary_text)>.
fn enumerate_subagent_transcripts(subagent_dir: &Path) -> Vec<(String, String)> {
    let mut results = Vec::new();

    if !subagent_dir.is_dir() {
        return results;
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(subagent_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("agent-") && (name.ends_with(".jsonl") || name.ends_with(".jsonl.gz"))
        })
        .collect();
    entries.sort();

    for path in entries {
        let content = read_transcript(&path);
        let mut session_id = String::new();
        let mut summary = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Extract session_id from session_start event
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                let event_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");

                if event_type == "session_start" {
                    session_id = val
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                }

                // Look for teammate-message summary in user_prompt content
                if event_type == "user_prompt"
                    && let Some(content) = val.get("content").and_then(|v| v.as_str())
                        && let Some(s) = extract_teammate_message_summary(content) {
                            summary = s;
                        }
            }

            // We have what we need after session_start + first user_prompt
            if !session_id.is_empty() && !summary.is_empty() {
                break;
            }
        }

        if !session_id.is_empty() {
            results.push((session_id, summary));
        }
    }

    results
}

/// Extract the `summary` attribute from a `<teammate-message summary="...">` tag.
fn extract_teammate_message_summary(content: &str) -> Option<String> {
    // Look for <teammate-message summary="..."> or <teammate-message ... summary="...">
    let tag_start = content.find("<teammate-message")?;
    let tag_content = &content[tag_start..];
    let tag_end = tag_content.find('>')?;
    let tag = &tag_content[..tag_end];

    // Extract summary="..." attribute
    let summary_start = tag.find("summary=\"")?;
    let value_start = summary_start + "summary=\"".len();
    let rest = &tag[value_start..];
    let value_end = rest.find('"')?;
    Some(rest[..value_end].to_string())
}

// ---------------------------------------------------------------------------
// Correlation matching
// ---------------------------------------------------------------------------

/// Match a subagent's summary to a teammate name via Agent tool call descriptions.
fn match_by_description(summary: &str, agent_calls: &[AgentCall]) -> Option<String> {
    if summary.is_empty() {
        return None;
    }

    // Exact match first
    for call in agent_calls {
        if call.description == summary {
            return Some(call.name.clone());
        }
    }

    // Prefix match (summary may be truncated)
    for call in agent_calls {
        if call.description.starts_with(summary) || summary.starts_with(&call.description) {
            return Some(call.name.clone());
        }
    }

    None
}

/// Fallback: match by temporal correlation (joined_at vs first message timestamp).
fn match_by_temporal(
    joined_at: &str,
    unmatched_transcripts: &[&(String, String)],
    sessions_dir: &Path,
) -> Option<String> {
    if joined_at.is_empty() || unmatched_transcripts.is_empty() {
        return None;
    }

    // Find the transcript whose first message timestamp is closest to joined_at
    let mut best_match: Option<(String, i64)> = None;

    for (session_id, _summary) in unmatched_transcripts {
        if let Some(first_ts) = get_first_message_timestamp(session_id, sessions_dir) {
            let diff = timestamp_diff_ms(joined_at, &first_ts).unwrap_or(i64::MAX);
            let abs_diff = diff.unsigned_abs() as i64;

            if abs_diff < best_match.as_ref().map(|(_, d)| *d).unwrap_or(i64::MAX) {
                best_match = Some((session_id.to_string(), abs_diff));
            }
        }
    }

    // Only accept temporal matches within 30 seconds
    best_match
        .filter(|(_, diff)| *diff < 30_000)
        .map(|(sid, _)| sid)
}

/// Get the timestamp of the first message in a session transcript.
fn get_first_message_timestamp(session_id: &str, sessions_dir: &Path) -> Option<String> {
    let candidates = [sessions_dir.join(format!("{session_id}.jsonl.gz")),
        sessions_dir.join(format!("{session_id}.jsonl")),
        sessions_dir
            .join("archive")
            .join(format!("{session_id}.jsonl.gz"))];

    let path = candidates.iter().find(|p| p.exists())?;
    let content = read_transcript(path);

    for line in content.lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line.trim())
            && let Some(ts) = val.get("timestamp").and_then(|v| v.as_str())
                && !ts.is_empty() {
                    return Some(ts.to_string());
                }
    }
    None
}

/// Compute approximate millisecond difference between two ISO 8601 timestamps.
/// Returns None if parsing fails.
fn timestamp_diff_ms(a: &str, b: &str) -> Option<i64> {
    // Simple lexicographic comparison for ISO 8601 strings.
    // For precise ms diff, we'd need chrono, but for correlation within 30s
    // window, lexicographic ordering of ISO timestamps is sufficient.
    // We compare the first 23 chars (YYYY-MM-DDTHH:MM:SS.mmm) as a proxy.
    let a_norm = normalize_timestamp(a);
    let b_norm = normalize_timestamp(b);

    if a_norm.len() < 19 || b_norm.len() < 19 {
        return None;
    }

    // Parse hours, minutes, seconds from the timestamp
    let a_secs = parse_total_seconds(&a_norm)?;
    let b_secs = parse_total_seconds(&b_norm)?;

    Some((a_secs - b_secs) * 1000)
}

fn normalize_timestamp(ts: &str) -> String {
    ts.replace('Z', "+00:00")
}

fn parse_total_seconds(ts: &str) -> Option<i64> {
    // Assumes ISO 8601: YYYY-MM-DDTHH:MM:SS...
    let t_pos = ts.find('T')?;
    let time_part = &ts[t_pos + 1..];
    if time_part.len() < 8 {
        return None;
    }
    let hours: i64 = time_part[0..2].parse().ok()?;
    let minutes: i64 = time_part[3..5].parse().ok()?;
    let seconds: i64 = time_part[6..8].parse().ok()?;
    Some(hours * 3600 + minutes * 60 + seconds)
}

// ---------------------------------------------------------------------------
// File reading utilities
// ---------------------------------------------------------------------------

/// Read a transcript file (supports both .jsonl and .jsonl.gz).
fn read_transcript(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    if path_str.ends_with(".jsonl.gz") {
        read_gzip_file(path)
    } else {
        std::fs::read_to_string(path).unwrap_or_default()
    }
}

fn read_gzip_file(path: &Path) -> String {
    use std::io::Read;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut content = String::new();
    decoder.read_to_string(&mut content).ok();
    content
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_extract_teammate_message_summary() {
        let content = r#"<teammate-message teammate_id="team-lead" summary="Research auth patterns">Full prompt here</teammate-message>"#;
        assert_eq!(
            extract_teammate_message_summary(content),
            Some("Research auth patterns".into())
        );
    }

    #[test]
    fn test_extract_teammate_message_summary_missing() {
        assert_eq!(extract_teammate_message_summary("no tag here"), None);
    }

    #[test]
    fn test_extract_teammate_message_summary_no_summary_attr() {
        let content = r#"<teammate-message teammate_id="lead">text</teammate-message>"#;
        assert_eq!(extract_teammate_message_summary(content), None);
    }

    #[test]
    fn test_match_by_description_exact() {
        let calls = vec![
            AgentCall {
                name: "researcher".into(),
                description: "Research auth patterns".into(),
            },
            AgentCall {
                name: "reviewer".into(),
                description: "Review code quality".into(),
            },
        ];
        assert_eq!(
            match_by_description("Research auth patterns", &calls),
            Some("researcher".into())
        );
        assert_eq!(
            match_by_description("Review code quality", &calls),
            Some("reviewer".into())
        );
    }

    #[test]
    fn test_match_by_description_prefix() {
        let calls = vec![AgentCall {
            name: "worker".into(),
            description: "Implement the feature as described in the spec".into(),
        }];
        // Summary may be truncated
        assert_eq!(
            match_by_description("Implement the feature as described", &calls),
            Some("worker".into())
        );
    }

    #[test]
    fn test_match_by_description_no_match() {
        let calls = vec![AgentCall {
            name: "worker".into(),
            description: "specific task".into(),
        }];
        assert_eq!(match_by_description("totally different", &calls), None);
    }

    #[test]
    fn test_timestamp_diff_ms() {
        let a = "2026-03-28T10:00:00Z";
        let b = "2026-03-28T10:00:05Z";
        assert_eq!(timestamp_diff_ms(a, b), Some(-5000));
        assert_eq!(timestamp_diff_ms(b, a), Some(5000));
    }

    #[test]
    fn test_timestamp_diff_ms_same() {
        let a = "2026-03-28T10:00:00Z";
        assert_eq!(timestamp_diff_ms(a, a), Some(0));
    }

    // T8: Description-based subagent correlation finds correct matches
    #[test]
    fn t8_description_based_correlation() {
        let dir = tempfile::tempdir().unwrap();
        let sessions_dir = dir.path();

        let lead_session_id = "ses-lead-001";

        // Create lead transcript with Agent tool calls
        let lead_jsonl = format!(
            concat!(
                r#"{{"type":"session_start","session_id":"{}","timestamp":"2026-03-28T10:00:00Z","workspace":"/tmp","git_branch":"main","agent_type":"main","parent_session_id":null}}"#, "\n",
                r#"{{"type":"tool_start","session_id":"{}","turn":1,"timestamp":"2026-03-28T10:00:01Z","tool":"Agent","input":{{"name":"researcher","description":"Research auth patterns","prompt":"..."}}}}"#, "\n",
                r#"{{"type":"tool_start","session_id":"{}","turn":1,"timestamp":"2026-03-28T10:00:02Z","tool":"Agent","input":{{"name":"reviewer","description":"Review code quality","prompt":"..."}}}}"#, "\n",
                r#"{{"type":"session_end","session_id":"{}","timestamp":"2026-03-28T10:30:00Z","total_turns":1}}"#
            ),
            lead_session_id, lead_session_id, lead_session_id, lead_session_id
        );

        let lead_gz = sessions_dir.join(format!("{lead_session_id}.jsonl.gz"));
        let file = std::fs::File::create(&lead_gz).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(lead_jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();

        // Create subagent directory with two transcripts
        let subagent_dir = sessions_dir.join(lead_session_id).join("subagents");
        std::fs::create_dir_all(&subagent_dir).unwrap();

        // Subagent 1 (researcher)
        let sub1_jsonl = concat!(
            r#"{"type":"session_start","session_id":"ses-sub-researcher","timestamp":"2026-03-28T10:00:01Z","workspace":"/tmp","git_branch":"main","agent_type":"subagent","parent_session_id":"ses-lead-001"}"#, "\n",
            r#"{"type":"user_prompt","session_id":"ses-sub-researcher","turn":1,"timestamp":"2026-03-28T10:00:01Z","content":"<teammate-message teammate_id=\"team-lead\" summary=\"Research auth patterns\">Full research prompt</teammate-message>"}"#, "\n"
        );
        std::fs::write(subagent_dir.join("agent-abc123.jsonl"), sub1_jsonl).unwrap();

        // Subagent 2 (reviewer)
        let sub2_jsonl = concat!(
            r#"{"type":"session_start","session_id":"ses-sub-reviewer","timestamp":"2026-03-28T10:00:02Z","workspace":"/tmp","git_branch":"main","agent_type":"subagent","parent_session_id":"ses-lead-001"}"#, "\n",
            r#"{"type":"user_prompt","session_id":"ses-sub-reviewer","turn":1,"timestamp":"2026-03-28T10:00:02Z","content":"<teammate-message teammate_id=\"team-lead\" summary=\"Review code quality\">Full review prompt</teammate-message>"}"#, "\n"
        );
        std::fs::write(subagent_dir.join("agent-def456.jsonl"), sub2_jsonl).unwrap();

        let members = vec![
            ("researcher".to_string(), "2026-03-28T10:00:01Z".to_string()),
            ("reviewer".to_string(), "2026-03-28T10:00:02Z".to_string()),
        ];

        let result = correlate_teammates(lead_session_id, sessions_dir, &members);

        assert_eq!(result.len(), 2, "expected 2 correlations, got {:?}", result);
        assert_eq!(result.get("researcher").unwrap(), "ses-sub-researcher");
        assert_eq!(result.get("reviewer").unwrap(), "ses-sub-reviewer");
    }

    // T8b: Empty lead session returns empty map
    #[test]
    fn test_empty_lead_session() {
        let dir = tempfile::tempdir().unwrap();
        let result = correlate_teammates("", dir.path(), &[]);
        assert!(result.is_empty());
    }

    // T8c: Missing transcript files returns empty map
    #[test]
    fn test_missing_transcripts() {
        let dir = tempfile::tempdir().unwrap();
        let members = vec![("worker".to_string(), "2026-03-28T10:00:00Z".to_string())];
        let result = correlate_teammates("ses-nonexistent", dir.path(), &members);
        assert!(result.is_empty());
    }
}
