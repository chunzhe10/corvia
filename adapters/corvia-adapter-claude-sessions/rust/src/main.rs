//! corvia-adapter-claude-sessions — Claude Code session history ingestion adapter.
//!
//! Reads gzipped JSONL session logs from `~/.claude/sessions/`, groups events by
//! turn, and produces structured text chunks (one KnowledgeEntry per turn).
//!
//! Protocol: D75 (JSONL over stdin/stdout)
//! Design: Session History RFC (2026-03-14)

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: u32 = 1;
const MAX_OUTPUT_LEN: usize = 500;

/// Adapter domain identifier (matches `corvia_common::constants::CLAUDE_SESSIONS_ADAPTER`).
const ADAPTER_DOMAIN: &str = "claude-sessions";

/// Default scope for session ingestion (matches `corvia_common::constants::USER_HISTORY_SCOPE`).
const DEFAULT_SCOPE: &str = "user-history";

// ---------------------------------------------------------------------------
// Protocol types (matches corvia-kernel/src/adapter_protocol.rs)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Metadata {
    name: &'static str,
    version: String,
    domain: &'static str,
    protocol_version: u32,
    description: &'static str,
    supported_extensions: Vec<String>,
    chunking_extensions: Vec<String>,
}

#[derive(Serialize)]
struct SourceFileMsg {
    source_file: SourceFilePayload,
}

#[derive(Serialize)]
struct SourceFilePayload {
    content: String,
    metadata: SourceMetadata,
}

#[derive(Serialize)]
struct SourceMetadata {
    file_path: String,
    extension: String,
    language: Option<String>,
    scope_id: String,
    source_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    workstream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_origin: Option<String>,
}

#[derive(Serialize)]
struct DoneMsg {
    done: bool,
    total_files: usize,
}

#[derive(Serialize)]
struct ErrorMsg {
    error: ErrorPayload,
}

#[derive(Serialize)]
struct ErrorPayload {
    code: String,
    message: String,
}

#[derive(Deserialize)]
#[serde(tag = "method", content = "params")]
enum Request {
    #[serde(rename = "ingest")]
    Ingest {
        #[allow(dead_code)]
        source_path: String,
        scope_id: String,
    },
    #[serde(rename = "chunk")]
    Chunk {
        #[allow(dead_code)]
        content: String,
        #[allow(dead_code)]
        metadata: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Session log event types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SessionEvent {
    #[serde(rename = "session_start")]
    SessionStart {
        session_id: String,
        #[allow(dead_code)]
        #[serde(default)]
        workspace: String,
        #[serde(default)]
        git_branch: String,
        #[serde(default)]
        agent_type: String,
        #[serde(default)]
        parent_session_id: Option<String>,
        #[serde(default)]
        timestamp: String,
    },
    #[serde(rename = "user_prompt")]
    UserPrompt {
        turn: u32,
        content: String,
        #[serde(default)]
        timestamp: String,
    },
    #[serde(rename = "tool_start")]
    #[allow(dead_code)]
    ToolStart {
        turn: u32,
        tool: String,
        #[serde(default)]
        input: serde_json::Value,
        #[serde(default)]
        timestamp: String,
    },
    #[serde(rename = "tool_end")]
    ToolEnd {
        turn: u32,
        tool: String,
        #[serde(default)]
        input: serde_json::Value,
        #[serde(default)]
        output: String,
        #[allow(dead_code)]
        #[serde(default)]
        truncated: bool,
        #[serde(default)]
        success: bool,
        #[serde(default)]
        timestamp: String,
    },
    #[serde(rename = "agent_response")]
    AgentResponse {
        turn: u32,
        content: String,
        #[serde(default)]
        timestamp: String,
    },
    #[serde(rename = "session_end")]
    #[allow(dead_code)]
    SessionEnd {
        #[serde(default)]
        total_turns: u32,
        #[serde(default)]
        timestamp: String,
    },
}

// ---------------------------------------------------------------------------
// Parsed session data
// ---------------------------------------------------------------------------

struct SessionInfo {
    session_id: String,
    git_branch: String,
    agent_type: String,
    parent_session_id: Option<String>,
    timestamp: String,
}

struct TurnData {
    prompt: Option<String>,
    tools: Vec<ToolCall>,
    response: Option<String>,
    timestamp: String,
    has_repo_paths: bool,
}

struct ToolCall {
    tool: String,
    input_summary: String,
    output_summary: String,
    success: bool,
    duration_hint: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        session_mode();
        return;
    }

    match args[1].as_str() {
        "--corvia-metadata" => {
            let meta = metadata();
            println!("{}", serde_json::to_string(&meta).unwrap());
        }
        "ingest" => {
            let stdout = io::stdout();
            let mut out = io::BufWriter::new(stdout.lock());
            let scope_id = if args.len() > 3 { &args[3] } else { DEFAULT_SCOPE };
            ingest_sessions(scope_id, &mut out);
        }
        _ => {
            eprintln!("Usage: corvia-adapter-claude-sessions [--corvia-metadata | ingest]");
            std::process::exit(1);
        }
    }
}

fn metadata() -> Metadata {
    Metadata {
        name: ADAPTER_DOMAIN,
        version: VERSION.to_string(),
        domain: ADAPTER_DOMAIN,
        protocol_version: PROTOCOL_VERSION,
        description: "Claude Code session history adapter — ingests ~/.claude/sessions/*.jsonl.gz",
        supported_extensions: vec!["jsonl.gz".into()],
        chunking_extensions: vec![],
    }
}

fn session_mode() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        match serde_json::from_str::<Request>(&line) {
            Ok(Request::Ingest { scope_id, .. }) => {
                ingest_sessions(&scope_id, &mut out);
            }
            Ok(Request::Chunk { .. }) => {
                let err = ErrorMsg {
                    error: ErrorPayload {
                        code: "NOT_SUPPORTED".into(),
                        message: "claude-sessions adapter does not provide chunking".into(),
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&err).unwrap()).ok();
                out.flush().ok();
            }
            Err(e) => {
                let err = ErrorMsg {
                    error: ErrorPayload {
                        code: "PARSE_ERROR".into(),
                        message: format!("Invalid request: {e}"),
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&err).unwrap()).ok();
                out.flush().ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core ingestion logic
// ---------------------------------------------------------------------------

fn default_sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".claude").join("sessions")
}

fn ingest_sessions<W: Write>(scope_id: &str, out: &mut W) {
    ingest_sessions_from(scope_id, &default_sessions_dir(), out);
}

fn ingest_sessions_from<W: Write>(scope_id: &str, dir: &Path, out: &mut W) {
    if !dir.is_dir() {
        let done = DoneMsg { done: true, total_files: 0 };
        writeln!(out, "{}", serde_json::to_string(&done).unwrap()).ok();
        out.flush().ok();
        return;
    }

    // Read already-ingested session IDs
    let ingested_path = dir.join(".ingested");
    let ingested: HashSet<String> = read_lines_to_set(&ingested_path);

    // Scan for new .jsonl.gz files
    let mut total_files = 0;
    let mut newly_ingested: Vec<String> = Vec::new();
    let mut classify_entries: Vec<String> = Vec::new();

    // Stale session threshold: raw .jsonl files older than this are treated as
    // sessions that never received SessionEnd (user did /clear or auto-compact).
    let stale_threshold = std::time::Duration::from_secs(4 * 3600); // 4 hours
    let now = std::time::SystemTime::now();

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let path_str = e.path().to_string_lossy().to_string();
            if path_str.ends_with(".jsonl.gz") {
                return true;
            }
            // Include raw .jsonl files only if they are stale (no recent writes)
            if path_str.ends_with(".jsonl") {
                if let Ok(meta) = e.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if let Ok(age) = now.duration_since(modified) {
                            return age > stale_threshold;
                        }
                    }
                }
            }
            false
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        // Extract session_id from filename: "ses-abc123.jsonl.gz" or "ses-abc123.jsonl"
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let session_id = filename
            .strip_suffix(".jsonl.gz")
            .or_else(|| filename.strip_suffix(".jsonl"))
            .unwrap_or("")
            .to_string();

        if session_id.is_empty() || ingested.contains(&session_id) {
            continue;
        }

        // Parse session
        match parse_session(&path) {
            Ok((info, turns)) => {
                let turn_count = turns.len();
                for (turn_num, turn) in &turns {
                    let content = format_turn(&info, *turn_num, turn);
                    let source_version = format!("{}:turn-{}", info.session_id, turn_num);

                    let content_role = infer_turn_content_role(&info, turn);
                    let source_origin = if info.agent_type == "subagent"
                        || (info.agent_type != "main" && !info.agent_type.is_empty())
                    {
                        "claude:subagent".to_string()
                    } else {
                        "claude:main".to_string()
                    };

                    let msg = SourceFileMsg {
                        source_file: SourceFilePayload {
                            content,
                            metadata: SourceMetadata {
                                file_path: info.session_id.clone(),
                                extension: "jsonl.gz".into(),
                                language: None,
                                scope_id: scope_id.to_string(),
                                source_version,
                                workstream: if info.git_branch.is_empty() {
                                    None
                                } else {
                                    Some(info.git_branch.clone())
                                },
                                content_role: Some(content_role),
                                source_origin: Some(source_origin),
                            },
                        },
                    };

                    writeln!(out, "{}", serde_json::to_string(&msg).unwrap()).ok();

                    // Queue entries that reference repo paths for classification
                    if turn.has_repo_paths {
                        classify_entries.push(format!(
                            "{}:turn-{}",
                            info.session_id, turn_num
                        ));
                    }
                }

                if turn_count > 0 {
                    total_files += 1;
                    newly_ingested.push(session_id.clone());
                }

                // Move to archive
                let archive_dir = dir.join("archive");
                std::fs::create_dir_all(&archive_dir).ok();
                let archive_path = archive_dir.join(entry.file_name());
                std::fs::rename(&path, &archive_path).ok();
            }
            Err(_) => {
                // Skip unparseable files silently
                continue;
            }
        }
    }

    // Update .ingested state
    if !newly_ingested.is_empty() {
        append_lines(&ingested_path, &newly_ingested);
    }

    // Append to .classify-queue
    if !classify_entries.is_empty() {
        let queue_path = dir.join(".classify-queue");
        append_lines(&queue_path, &classify_entries);
    }

    let done = DoneMsg { done: true, total_files };
    writeln!(out, "{}", serde_json::to_string(&done).unwrap()).ok();
    out.flush().ok();
}

// ---------------------------------------------------------------------------
// Session parsing
// ---------------------------------------------------------------------------

fn parse_session(path: &Path) -> Result<(SessionInfo, BTreeMap<u32, TurnData>), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let is_gzip = path.to_string_lossy().ends_with(".jsonl.gz");
    let reader: Box<dyn BufRead> = if is_gzip {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut info = SessionInfo {
        session_id: String::new(),
        git_branch: String::new(),
        agent_type: "main".into(),
        parent_session_id: None,
        timestamp: String::new(),
    };

    let mut turns: BTreeMap<u32, TurnData> = BTreeMap::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }

        let event: SessionEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        match event {
            SessionEvent::SessionStart {
                session_id,
                git_branch,
                agent_type,
                parent_session_id,
                timestamp,
                ..
            } => {
                info.session_id = session_id;
                info.git_branch = git_branch;
                info.agent_type = if agent_type.is_empty() {
                    "main".into()
                } else {
                    agent_type
                };
                info.parent_session_id = parent_session_id.filter(|s| !s.is_empty());
                info.timestamp = timestamp;
            }
            SessionEvent::UserPrompt {
                turn,
                content,
                timestamp,
                ..
            } => {
                let td = turns.entry(turn).or_insert_with(|| TurnData {
                    prompt: None,
                    tools: Vec::new(),
                    response: None,
                    timestamp: timestamp.clone(),
                    has_repo_paths: false,
                });
                td.prompt = Some(content);
                if td.timestamp.is_empty() {
                    td.timestamp = timestamp;
                }
            }
            SessionEvent::ToolEnd {
                turn,
                tool,
                input,
                output,
                success,
                timestamp,
                ..
            } => {
                let td = turns.entry(turn).or_insert_with(|| TurnData {
                    prompt: None,
                    tools: Vec::new(),
                    response: None,
                    timestamp: timestamp.clone(),
                    has_repo_paths: false,
                });

                // Check for repo paths in input
                let input_str = input.to_string();
                if input_str.contains("repos/") || input_str.contains("/workspaces/") {
                    td.has_repo_paths = true;
                }

                let input_summary = summarize_tool_input(&tool, &input);
                let output_summary = truncate_str(&output, MAX_OUTPUT_LEN).to_string();

                td.tools.push(ToolCall {
                    tool,
                    input_summary,
                    output_summary,
                    success,
                    duration_hint: String::new(),
                });
            }
            SessionEvent::AgentResponse {
                turn,
                content,
                timestamp,
                ..
            } => {
                let td = turns.entry(turn).or_insert_with(|| TurnData {
                    prompt: None,
                    tools: Vec::new(),
                    response: None,
                    timestamp: timestamp.clone(),
                    has_repo_paths: false,
                });
                td.response = Some(truncate_str(&content, 1000).to_string());
            }
            SessionEvent::ToolStart { .. } | SessionEvent::SessionEnd { .. } => {
                // tool_start is informational; session_end has no turn data
            }
        }
    }

    if info.session_id.is_empty() {
        return Err("no session_start found".into());
    }

    Ok((info, turns))
}

// ---------------------------------------------------------------------------
// Turn formatting (one chunk per turn)
// ---------------------------------------------------------------------------

fn format_turn(info: &SessionInfo, turn_num: u32, turn: &TurnData) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "[Turn {} | {} | {} | {}]\n",
        turn_num,
        if info.git_branch.is_empty() {
            "(no branch)"
        } else {
            &info.git_branch
        },
        info.session_id,
        turn.timestamp
    ));

    // User prompt
    if let Some(prompt) = &turn.prompt {
        out.push_str(&format!("USER: {}\n", prompt));
    }

    // Tool calls
    if !turn.tools.is_empty() {
        out.push_str("TOOLS:\n");
        for tc in &turn.tools {
            let status = if tc.success { "✓" } else { "ERROR" };
            out.push_str(&format!(
                "  - {}({}) → {} {}\n",
                tc.tool, tc.input_summary, tc.duration_hint, status
            ));
            if !tc.success && !tc.output_summary.is_empty() {
                out.push_str(&format!("    ERROR: {}\n", tc.output_summary));
            }
        }
    }

    // Response
    if let Some(response) = &turn.response {
        out.push_str(&format!("RESPONSE: {}\n", response));
    }

    out
}

fn infer_turn_content_role(info: &SessionInfo, turn: &TurnData) -> String {
    // Subagent turns with search+read+synthesize patterns are "research"
    if info.agent_type != "main" && !turn.tools.is_empty() {
        let has_search = turn.tools.iter().any(|t| {
            matches!(t.tool.as_str(), "Grep" | "Glob" | "WebSearch" | "mcp__corvia__corvia_search")
        });
        let has_read = turn.tools.iter().any(|t| {
            matches!(t.tool.as_str(), "Read" | "WebFetch" | "mcp__corvia__corvia_ask")
        });
        if has_search && has_read {
            return "research".into();
        }
    }
    "session-turn".into()
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn summarize_tool_input(tool: &str, input: &serde_json::Value) -> String {
    match tool {
        "Grep" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if path.is_empty() {
                format!("\"{}\"", truncate_str(pattern, 50))
            } else {
                format!("\"{}\" in {}", truncate_str(pattern, 40), truncate_str(path, 40))
            }
        }
        "Glob" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            format!("\"{}\"", truncate_str(pattern, 60))
        }
        "Read" => {
            let path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
            truncate_str(path, 80).to_string()
        }
        "Edit" | "Write" => {
            let path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("?");
            truncate_str(path, 80).to_string()
        }
        "Bash" => {
            let cmd = input.get("command").and_then(|v| v.as_str()).unwrap_or("?");
            truncate_str(cmd, 80).to_string()
        }
        "Agent" => {
            let desc = input.get("description").and_then(|v| v.as_str())
                .or_else(|| input.get("prompt").and_then(|v| v.as_str()))
                .unwrap_or("?");
            truncate_str(desc, 60).to_string()
        }
        _ => {
            // Generic: show first key-value pair
            if let Some(obj) = input.as_object()
                && let Some((k, v)) = obj.iter().next() {
                    let val_str = match v.as_str() {
                        Some(s) => truncate_str(s, 50).to_string(),
                        None => truncate_str(&v.to_string(), 50).to_string(),
                    };
                    return format!("{}={}", k, val_str);
                }
            String::new()
        }
    }
}

fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        // Find a char boundary
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

fn read_lines_to_set(path: &Path) -> HashSet<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect()
}

fn append_lines(path: &Path, lines: &[String]) {
    use std::fs::OpenOptions;
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        for line in lines {
            writeln!(f, "{}", line).ok();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    fn make_test_session_gz(dir: &Path, session_id: &str) -> PathBuf {
        let jsonl = format!(
            r#"{{"type":"session_start","session_id":"{}","timestamp":"2026-03-14T05:00:00Z","workspace":"/workspaces/corvia-workspace","git_branch":"feat/auth","agent_type":"main","parent_session_id":null,"corvia_agent_id":null}}
{{"type":"user_prompt","session_id":"{}","turn":1,"timestamp":"2026-03-14T05:01:00Z","content":"can we record what tools was used?"}}
{{"type":"tool_start","session_id":"{}","turn":1,"timestamp":"2026-03-14T05:01:01.123Z","tool":"Grep","input":{{"pattern":"scope_id","path":"repos/corvia/crates"}}}}
{{"type":"tool_end","session_id":"{}","turn":1,"timestamp":"2026-03-14T05:01:01.987Z","tool":"Grep","input":{{"pattern":"scope_id","path":"repos/corvia/crates"}},"output":"repos/corvia/crates/corvia-kernel/src/lite_store.rs:42","truncated":false,"success":true}}
{{"type":"tool_end","session_id":"{}","turn":1,"timestamp":"2026-03-14T05:01:02.100Z","tool":"Read","input":{{"file_path":"repos/corvia/crates/corvia-kernel/src/lite_store.rs"}},"output":"pub struct LiteStore ...","truncated":true,"success":true}}
{{"type":"agent_response","session_id":"{}","turn":1,"timestamp":"2026-03-14T05:01:10Z","content":"Good news — it's straightforward."}}
{{"type":"user_prompt","session_id":"{}","turn":2,"timestamp":"2026-03-14T05:02:00Z","content":"show me the search function"}}
{{"type":"tool_end","session_id":"{}","turn":2,"timestamp":"2026-03-14T05:02:01Z","tool":"Read","input":{{"file_path":"src/retriever.rs"}},"output":"fn search(...)","truncated":false,"success":true}}
{{"type":"session_end","session_id":"{}","timestamp":"2026-03-14T05:30:00Z","total_turns":2,"duration_ms":1800000}}"#,
            session_id, session_id, session_id, session_id,
            session_id, session_id, session_id, session_id, session_id
        );

        let gz_path = dir.join(format!("{}.jsonl.gz", session_id));
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();
        gz_path
    }

    #[test]
    fn test_metadata_output() {
        let meta = metadata();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"name\":\"claude-sessions\""));
        assert!(json.contains("\"domain\":\"claude-sessions\""));
    }

    #[test]
    fn test_parse_session() {
        let dir = tempfile::tempdir().unwrap();
        let gz_path = make_test_session_gz(dir.path(), "ses-test-001");

        let (info, turns) = parse_session(&gz_path).unwrap();
        assert_eq!(info.session_id, "ses-test-001");
        assert_eq!(info.git_branch, "feat/auth");
        assert_eq!(info.agent_type, "main");
        assert!(info.parent_session_id.is_none());

        assert_eq!(turns.len(), 2);

        // Turn 1
        let t1 = &turns[&1];
        assert_eq!(t1.prompt.as_deref(), Some("can we record what tools was used?"));
        assert_eq!(t1.tools.len(), 2); // Grep + Read
        assert!(t1.response.is_some());
        assert!(t1.has_repo_paths); // input contains "repos/"

        // Turn 2
        let t2 = &turns[&2];
        assert_eq!(t2.prompt.as_deref(), Some("show me the search function"));
        assert_eq!(t2.tools.len(), 1);
    }

    #[test]
    fn test_format_turn() {
        let dir = tempfile::tempdir().unwrap();
        let gz_path = make_test_session_gz(dir.path(), "ses-fmt-001");

        let (info, turns) = parse_session(&gz_path).unwrap();
        let formatted = format_turn(&info, 1, &turns[&1]);

        assert!(formatted.contains("[Turn 1 | feat/auth | ses-fmt-001"));
        assert!(formatted.contains("USER: can we record what tools was used?"));
        assert!(formatted.contains("TOOLS:"));
        assert!(formatted.contains("Grep("));
        assert!(formatted.contains("RESPONSE:"));
    }

    #[test]
    fn test_ingest_produces_source_files() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path();

        // Create a test session
        let jsonl = r#"{"type":"session_start","session_id":"ses-ingest-001","timestamp":"2026-03-14T05:00:00Z","workspace":"/tmp","git_branch":"main","agent_type":"main","parent_session_id":null,"corvia_agent_id":null}
{"type":"user_prompt","session_id":"ses-ingest-001","turn":1,"timestamp":"2026-03-14T05:01:00Z","content":"hello"}
{"type":"session_end","session_id":"ses-ingest-001","timestamp":"2026-03-14T05:02:00Z","total_turns":1,"duration_ms":60000}"#;

        let gz_path = sessions.join("ses-ingest-001.jsonl.gz");
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();

        let mut output = Vec::new();
        ingest_sessions_from(DEFAULT_SCOPE, sessions, &mut output);

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();

        // 1 source_file (1 turn) + 1 done
        assert_eq!(lines.len(), 2, "got: {:?}", lines);
        assert!(lines[0].contains("\"source_file\""));
        assert!(lines[0].contains("ses-ingest-001"));
        assert!(lines[0].contains("\"scope_id\":\"user-history\""));
        assert!(lines[0].contains("\"workstream\":\"main\""));
        assert!(lines[0].contains("\"content_role\":\"session-turn\""));
        assert!(lines[0].contains("\"source_origin\":\"claude:main\""));
        assert!(lines[1].contains("\"done\":true"));

        // Verify .ingested was updated
        let ingested = std::fs::read_to_string(sessions.join(".ingested")).unwrap();
        assert!(ingested.contains("ses-ingest-001"));

        // Verify archive
        assert!(sessions.join("archive").join("ses-ingest-001.jsonl.gz").exists());
        assert!(!gz_path.exists()); // moved to archive
    }

    #[test]
    fn test_ingest_skips_already_ingested() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path();

        // Mark as ingested
        std::fs::write(sessions.join(".ingested"), "ses-skip-001\n").unwrap();

        // Create the gz file
        let jsonl = r#"{"type":"session_start","session_id":"ses-skip-001","timestamp":"2026-03-14T05:00:00Z","workspace":"/tmp","git_branch":"main","agent_type":"main","parent_session_id":null,"corvia_agent_id":null}
{"type":"session_end","session_id":"ses-skip-001","timestamp":"2026-03-14T05:01:00Z","total_turns":0,"duration_ms":0}"#;

        let gz_path = sessions.join("ses-skip-001.jsonl.gz");
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        enc.write_all(jsonl.as_bytes()).unwrap();
        enc.finish().unwrap();

        let mut output = Vec::new();
        ingest_sessions_from(DEFAULT_SCOPE, sessions, &mut output);

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"total_files\":0"));
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 5), "hello");
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn test_summarize_tool_input() {
        let input = serde_json::json!({"pattern": "scope_id", "path": "repos/corvia"});
        let summary = summarize_tool_input("Grep", &input);
        assert!(summary.contains("scope_id"));
        assert!(summary.contains("repos/corvia"));

        let input = serde_json::json!({"file_path": "src/main.rs"});
        let summary = summarize_tool_input("Read", &input);
        assert_eq!(summary, "src/main.rs");
    }

    #[test]
    fn test_infer_content_role_research() {
        let info = SessionInfo {
            session_id: "ses-1".into(),
            git_branch: "feat/x".into(),
            agent_type: "subagent".into(),
            parent_session_id: Some("ses-parent".into()),
            timestamp: String::new(),
        };
        let turn = TurnData {
            prompt: Some("find auth patterns".into()),
            tools: vec![
                ToolCall {
                    tool: "Grep".into(),
                    input_summary: String::new(),
                    output_summary: String::new(),
                    success: true,
                    duration_hint: String::new(),
                },
                ToolCall {
                    tool: "Read".into(),
                    input_summary: String::new(),
                    output_summary: String::new(),
                    success: true,
                    duration_hint: String::new(),
                },
            ],
            response: None,
            timestamp: String::new(),
            has_repo_paths: false,
        };
        assert_eq!(infer_turn_content_role(&info, &turn), "research");
    }

    #[test]
    fn test_infer_content_role_session_turn() {
        let info = SessionInfo {
            session_id: "ses-1".into(),
            git_branch: "main".into(),
            agent_type: "main".into(),
            parent_session_id: None,
            timestamp: String::new(),
        };
        let turn = TurnData {
            prompt: Some("hello".into()),
            tools: vec![],
            response: None,
            timestamp: String::new(),
            has_repo_paths: false,
        };
        assert_eq!(infer_turn_content_role(&info, &turn), "session-turn");
    }
}
