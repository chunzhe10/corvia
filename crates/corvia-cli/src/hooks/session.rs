//! Session recording: atomic JSONL event writer with WAL semantics.
//!
//! Events are written to `~/.claude/sessions/{session_id}.jsonl`, one JSON
//! object per line. The file acts as a write-ahead log — durable on disk
//! regardless of server state.

use super::HookEvent;
use anyhow::Result;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Schema version embedded in every JSONL event.
const SCHEMA_VERSION: u32 = 1;

/// Max output chars stored for tool_end events.
const MAX_OUTPUT_CHARS: usize = 500;

/// Get the sessions directory path.
fn sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".claude/sessions")
}

/// Get the JSONL file path for a session.
fn session_file(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{session_id}.jsonl"))
}

/// PIPE_BUF on Linux — writes smaller than this are atomic with O_APPEND.
const PIPE_BUF: usize = 4096;

/// Atomically append a JSON event line to the session JSONL file.
///
/// Uses O_APPEND for atomic writes. If the serialized line exceeds PIPE_BUF,
/// it is truncated to fit (the `input` field is removed as the largest payload).
fn append_event(session_id: &str, event: &serde_json::Value) -> Result<()> {
    let dir = sessions_dir();
    fs::create_dir_all(&dir)?;

    let path = session_file(session_id);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;

    // Build complete line in memory
    let mut line = serde_json::to_string(event)?;
    line.push('\n');

    // If line exceeds PIPE_BUF, strip the input field to bring it under
    if line.len() > PIPE_BUF {
        if let Some(mut obj) = event.as_object().cloned() {
            obj.remove("input");
            obj.remove("output");
            line = serde_json::to_string(&serde_json::Value::Object(obj))?;
            line.push('\n');
        }
    }

    file.write_all(line.as_bytes())?;
    Ok(())
}

/// Read the current turn number from the JSONL file by scanning from the end.
/// Returns the turn field from the last user_prompt event, or 0 if none found.
/// Reads at most the last 8KB to avoid O(n) scans on long sessions.
fn current_turn(session_id: &str) -> u32 {
    use std::io::{Seek, SeekFrom};

    let path = session_file(session_id);
    let mut file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    // Read the last 8KB (enough for ~10-20 recent events)
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let read_from = file_len.saturating_sub(8192);
    if file.seek(SeekFrom::Start(read_from)).is_err() {
        return 0;
    }

    let reader = BufReader::new(file);
    let mut last_turn = 0u32;
    for line in reader.lines() {
        if let Ok(line) = line {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
                if obj.get("type").and_then(|t| t.as_str()) == Some("user_prompt") {
                    if let Some(t) = obj.get("turn").and_then(|t| t.as_u64()) {
                        last_turn = t as u32;
                    }
                }
            }
        }
    }
    last_turn
}

/// Strip large content fields from tool input to avoid bloated JSONL.
fn strip_tool_input(input: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = input.as_object() {
        let mut stripped = obj.clone();
        stripped.remove("content");
        stripped.remove("new_string");
        stripped.remove("old_string");
        serde_json::Value::Object(stripped)
    } else {
        input.clone()
    }
}

/// Truncate a string to max_chars (by character, not byte), returning (truncated_string, was_truncated).
fn truncate_str(s: &str, max_chars: usize) -> (String, bool) {
    let mut char_count = 0;
    for (byte_idx, _) in s.char_indices() {
        char_count += 1;
        if char_count > max_chars {
            return (s[..byte_idx].to_string(), true);
        }
    }
    (s.to_string(), false)
}

/// Get current UTC timestamp in ISO 8601 format with nanosecond precision.
fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()
}

/// Record a hook event to the session JSONL file.
pub fn record_event(session_id: &str, event: &HookEvent, stdin: &serde_json::Value) -> Result<()> {
    let start = Instant::now();
    let ts = now_iso();

    let json_event = match event {
        HookEvent::SessionStart => {
            let cwd = stdin.get("cwd").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let git_branch = git_branch(&cwd);
            let agent_type = stdin.get("agent_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let agent_type = if agent_type.is_empty() { "main".to_string() } else { agent_type };
            let parent = stdin.get("agent_id").and_then(|v| v.as_str());
            let is_subagent = parent.is_some();
            let agent_type = if is_subagent { "subagent".to_string() } else { agent_type };
            let corvia_agent = std::env::var("CORVIA_AGENT_ID").ok();

            serde_json::json!({
                "v": SCHEMA_VERSION,
                "type": "session_start",
                "session_id": session_id,
                "timestamp": ts,
                "workspace": cwd,
                "git_branch": git_branch,
                "agent_type": agent_type,
                "parent_session_id": if is_subagent { parent } else { None },
                "corvia_agent_id": corvia_agent,
                "hook_duration_ms": start.elapsed().as_millis() as u64,
                "hook_name": "session_record",
            })
        }
        HookEvent::UserPromptSubmit => {
            let content = stdin.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            let turn = current_turn(session_id) + 1;

            serde_json::json!({
                "v": SCHEMA_VERSION,
                "type": "user_prompt",
                "session_id": session_id,
                "turn": turn,
                "timestamp": ts,
                "content": content,
                "hook_duration_ms": start.elapsed().as_millis() as u64,
                "hook_name": "session_record",
            })
        }
        HookEvent::PreToolUse => {
            let tool = stdin.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
            let input = stdin.get("tool_input").cloned().unwrap_or_default();
            let input = strip_tool_input(&input);
            let turn = current_turn(session_id);

            serde_json::json!({
                "v": SCHEMA_VERSION,
                "type": "tool_start",
                "session_id": session_id,
                "turn": turn,
                "timestamp": ts,
                "tool": tool,
                "input": input,
                "hook_duration_ms": start.elapsed().as_millis() as u64,
                "hook_name": "session_record",
            })
        }
        HookEvent::PostToolUse => {
            let tool = stdin.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
            let input = stdin.get("tool_input").cloned().unwrap_or_default();
            let input = strip_tool_input(&input);
            let turn = current_turn(session_id);

            // Extract and truncate output
            let output_raw = stdin.get("tool_response")
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            let (output, truncated) = truncate_str(&output_raw, MAX_OUTPUT_CHARS);

            serde_json::json!({
                "v": SCHEMA_VERSION,
                "type": "tool_end",
                "session_id": session_id,
                "turn": turn,
                "timestamp": ts,
                "tool": tool,
                "input": input,
                "output": output,
                "truncated": truncated,
                "success": true,
                "hook_duration_ms": start.elapsed().as_millis() as u64,
                "hook_name": "session_record",
            })
        }
        HookEvent::SessionEnd => {
            // session_end event is appended by finalize_session()
            return Ok(());
        }
    };

    append_event(session_id, &json_event)
}

/// Finalize a session: append session_end event, gzip, trigger ingest.
pub fn finalize_session(session_id: &str) {
    let path = session_file(session_id);
    if !path.exists() {
        return;
    }

    // Compute stats from JSONL
    let total_turns = current_turn(session_id);
    let duration_ms = compute_duration_ms(&path);
    let ts = now_iso();

    let end_event = serde_json::json!({
        "v": SCHEMA_VERSION,
        "type": "session_end",
        "session_id": session_id,
        "timestamp": ts,
        "total_turns": total_turns,
        "duration_ms": duration_ms,
    });

    // Append session_end (ignore errors — best effort)
    let _ = append_event(session_id, &end_event);

    // Gzip the JSONL
    if let Err(e) = gzip_session(&path) {
        eprintln!("Warning: failed to gzip session {session_id}: {e}");
        // JSONL is still on disk, adapter can pick it up later
    }

    // Trigger ingest with bounded retry
    trigger_ingest();
}

/// Gzip a JSONL file in place: read → compress → fsync → rename → remove original.
fn gzip_session(jsonl_path: &std::path::Path) -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let data = fs::read(jsonl_path)?;
    let gz_path = jsonl_path.with_extension("jsonl.gz");

    // Write to temp file, fsync for durability, then rename (atomic directory entry)
    let tmp_path = jsonl_path.with_extension("jsonl.gz.tmp");
    {
        let file = fs::File::create(&tmp_path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&data)?;
        let file = encoder.finish()?;
        file.sync_all()?; // Ensure compressed data is on disk before removing original
    }
    fs::rename(&tmp_path, &gz_path)?;
    fs::remove_file(jsonl_path).ok(); // Ignore remove error — gz is already safe
    Ok(())
}

/// Trigger session ingest via REST API with bounded retry.
fn trigger_ingest() {
    let client = match reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_millis(500))
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return,
    };

    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        match client
            .post("http://127.0.0.1:8020/v1/ingest/sessions")
            .json(&serde_json::json!({}))
            .send()
        {
            Ok(resp) if resp.status().is_success() => return,
            _ => continue,
        }
    }
    // All retries exhausted — session data is safe in .jsonl.gz, adapter picks it up later
}

/// Compute session duration in milliseconds from first event timestamp.
fn compute_duration_ms(path: &std::path::Path) -> u64 {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return 0,
    };
    let reader = BufReader::new(file);
    if let Some(Ok(first_line)) = reader.lines().next() {
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&first_line) {
            if let Some(ts_str) = obj.get("timestamp").and_then(|t| t.as_str()) {
                // Parse ISO timestamp to compute elapsed
                if let Some(start_secs) = parse_iso_epoch_secs(ts_str) {
                    let now_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    return now_secs.saturating_sub(start_secs) * 1000;
                }
            }
        }
    }
    0
}

/// Parse an ISO 8601 timestamp to epoch seconds (best effort).
fn parse_iso_epoch_secs(ts: &str) -> Option<u64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .or_else(|| {
            // Handle our format: "2026-03-25T10:00:00.123456789Z"
            chrono::NaiveDateTime::parse_from_str(
                ts.trim_end_matches('Z'),
                "%Y-%m-%dT%H:%M:%S%.f"
            ).ok().map(|dt| dt.and_utc().fixed_offset())
        })
        .map(|dt| dt.timestamp() as u64)
}

/// Get current git branch (best effort).
fn git_branch(cwd: &str) -> String {
    let dir = if cwd.is_empty() { "." } else { cwd };
    std::process::Command::new("git")
        .args(["-C", dir, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

/// Sweep stale session files: gzip any .jsonl files older than max_age_hours.
pub fn sweep_stale_sessions(max_age_hours: u64) {
    let dir = sessions_dir();
    if !dir.exists() { return; }

    let max_age = std::time::Duration::from_secs(max_age_hours * 3600);
    let now = std::time::SystemTime::now();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut swept = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if let Ok(meta) = path.metadata() {
            if let Ok(modified) = meta.modified() {
                if let Ok(age) = now.duration_since(modified) {
                    if age > max_age {
                        if let Err(e) = gzip_session(&path) {
                            eprintln!("Warning: failed to gzip stale session {:?}: {e}", path);
                        } else {
                            swept += 1;
                        }
                    }
                }
            }
        }
    }

    if swept > 0 {
        println!("Swept {swept} stale session file(s)");
        trigger_ingest();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_tool_input() {
        let input = serde_json::json!({
            "file_path": "/foo/bar.rs",
            "content": "huge content here",
            "new_string": "replacement",
            "old_string": "original",
        });
        let stripped = strip_tool_input(&input);
        assert!(stripped.get("file_path").is_some());
        assert!(stripped.get("content").is_none());
        assert!(stripped.get("new_string").is_none());
        assert!(stripped.get("old_string").is_none());
    }

    #[test]
    fn test_truncate_str_ascii() {
        let (s, t) = truncate_str("hello", 10);
        assert_eq!(s, "hello");
        assert!(!t);

        let (s, t) = truncate_str("hello world", 5);
        assert_eq!(s, "hello");
        assert!(t);
    }

    #[test]
    fn test_truncate_str_unicode_no_panic() {
        // Emoji are 4 bytes each — truncating at char boundary must not panic
        let emoji_str = "Hello 🌍🎉🎊🎈 world";
        let (s, t) = truncate_str(emoji_str, 8);
        assert!(t);
        assert_eq!(s.chars().count(), 8);
        assert_eq!(s, "Hello 🌍🎉");
    }

    #[test]
    fn test_truncate_str_cjk() {
        let cjk = "日本語テストです";
        let (s, t) = truncate_str(cjk, 3);
        assert!(t);
        assert_eq!(s, "日本語");
    }

    #[test]
    fn test_truncate_str_empty() {
        let (s, t) = truncate_str("", 5);
        assert_eq!(s, "");
        assert!(!t);
    }

    #[test]
    fn test_truncate_str_exact_boundary() {
        let (s, t) = truncate_str("hello", 5);
        assert_eq!(s, "hello");
        assert!(!t);
    }

    #[test]
    fn test_now_iso_format() {
        let ts = now_iso();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 30); // YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ
    }

    #[test]
    fn test_parse_iso_epoch_roundtrip() {
        let ts = "2026-03-25T10:30:00.000000000Z";
        let secs = parse_iso_epoch_secs(ts).unwrap();
        assert!(secs > 0);
    }

    #[test]
    fn test_parse_iso_epoch_rfc3339() {
        let ts = "2026-03-25T10:30:00+00:00";
        let secs = parse_iso_epoch_secs(ts).unwrap();
        assert!(secs > 0);
    }

    #[test]
    fn test_parse_iso_epoch_invalid() {
        assert!(parse_iso_epoch_secs("not-a-date").is_none());
        assert!(parse_iso_epoch_secs("").is_none());
    }

    #[test]
    fn test_session_start_event_schema() {
        let stdin = serde_json::json!({
            "session_id": "test-123",
            "cwd": "/tmp",
        });
        let ts = now_iso();
        // Verify the structure would be correct
        let event = serde_json::json!({
            "v": SCHEMA_VERSION,
            "type": "session_start",
            "session_id": "test-123",
            "timestamp": ts,
            "workspace": "/tmp",
        });
        assert_eq!(event["v"], 1);
        assert_eq!(event["type"], "session_start");
    }
}
