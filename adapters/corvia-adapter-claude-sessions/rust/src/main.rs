//! corvia-adapter-claude-sessions — Claude Code session history ingestion adapter.
//!
//! Two domain modes:
//! 1. Sessions: Reads gzipped JSONL session logs from `~/.claude/sessions/`,
//!    groups events by turn, produces structured text chunks (one entry per turn).
//! 2. Agent Teams: Reads staging data from `~/.corvia/staging/agent-teams/`,
//!    produces team structure, task, and message entries.
//!
//! Domain dispatch: `source_path` containing "agent-teams" triggers teams path.
//!
//! Protocol: D75 (JSONL over stdin/stdout)
//! Design: Session History RFC (2026-03-14), Agent Teams RFC (2026-03-28)

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_session_id: Option<String>,
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
            Ok(Request::Ingest { source_path, scope_id }) => {
                if source_path.contains("agent-teams") {
                    ingest_agent_teams(&scope_id, &mut out);
                } else {
                    ingest_sessions(&scope_id, &mut out);
                }
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
            if path_str.ends_with(".jsonl")
                && let Ok(meta) = e.metadata()
                && let Ok(modified) = meta.modified()
                && let Ok(age) = now.duration_since(modified)
            {
                return age > stale_threshold;
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

                    // Only attach parent_session_id on turn 1 — the CLI
                    // ingest layer uses it to create a single spawned_by edge.
                    let parent_for_edge = if *turn_num == 1 {
                        info.parent_session_id.clone()
                    } else {
                        None
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
                                parent_session_id: parent_for_edge,
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
// Agent Teams ingestion (RFC Section 4)
// ---------------------------------------------------------------------------

/// Default staging directory for agent teams data.
fn default_staging_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".corvia").join("staging").join("agent-teams")
}

/// Retention period for cleaned-up staging directories (7 days).
const STAGING_RETENTION_SECS: u64 = 7 * 24 * 3600;

fn ingest_agent_teams<W: Write>(scope_id: &str, out: &mut W) {
    ingest_agent_teams_from(scope_id, &default_staging_dir(), out);
}

fn ingest_agent_teams_from<W: Write>(scope_id: &str, staging_root: &Path, out: &mut W) {
    if !staging_root.is_dir() {
        let done = DoneMsg { done: true, total_files: 0 };
        writeln!(out, "{}", serde_json::to_string(&done).unwrap()).ok();
        out.flush().ok();
        return;
    }

    // Read already-ingested team names
    let ingested_path = staging_root.join(".ingested");
    let ingested: HashSet<String> = read_lines_to_set(&ingested_path);

    let mut total_files = 0;
    let mut newly_ingested: Vec<String> = Vec::new();

    // Scan for team directories
    let mut team_dirs: Vec<_> = std::fs::read_dir(staging_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                && e.file_name().to_string_lossy() != "."
                && e.file_name().to_string_lossy() != ".."
        })
        .collect();
    team_dirs.sort_by_key(|e| e.file_name());

    for entry in &team_dirs {
        let team_name = entry.file_name().to_string_lossy().to_string();
        if ingested.contains(&team_name) {
            continue;
        }

        let team_dir = entry.path();
        let mut team_entries = 0;

        // A3: Parse config.json -> team structure entry
        let config = parse_team_config(&team_dir);
        if let Some(ref cfg) = config {
            // A4: Parse tasks
            let tasks = parse_tasks_jsonl(&team_dir);

            // Emit team structure entry (A3)
            let team_content = format_team_structure(&team_name, cfg, &tasks, &team_dir);
            let msg = SourceFileMsg {
                source_file: SourceFilePayload {
                    content: team_content,
                    metadata: SourceMetadata {
                        file_path: team_name.clone(),
                        extension: "json".into(),
                        language: None,
                        scope_id: scope_id.to_string(),
                        source_version: format!("{team_name}:config"),
                        workstream: None,
                        content_role: Some("memory".into()),
                        source_origin: Some(format!("claude:team:{team_name}")),
                        parent_session_id: None,
                    },
                },
            };
            writeln!(out, "{}", serde_json::to_string(&msg).unwrap()).ok();
            team_entries += 1;

            // A5: Emit task entries
            let grouped_tasks = group_tasks(&tasks);
            for (task_id, task_events) in &grouped_tasks {
                let task_content = format_task_entry(&team_name, task_id, task_events);
                let msg = SourceFileMsg {
                    source_file: SourceFilePayload {
                        content: task_content,
                        metadata: SourceMetadata {
                            file_path: team_name.clone(),
                            extension: "json".into(),
                            language: None,
                            scope_id: scope_id.to_string(),
                            source_version: format!("{team_name}:task:{task_id}"),
                            workstream: None,
                            content_role: Some("task".into()),
                            source_origin: Some(format!("claude:team:{team_name}")),
                            parent_session_id: None,
                        },
                    },
                };
                writeln!(out, "{}", serde_json::to_string(&msg).unwrap()).ok();
                team_entries += 1;
            }

            // A6: Parse and emit message entries
            let messages = parse_inbox_messages(&team_dir);
            if !messages.is_empty() {
                let deduped = dedup_broadcasts(messages);
                let grouped = group_messages_by_task(deduped);
                for (group_key, msgs) in &grouped {
                    let msg_content = format_message_entry(&team_name, group_key, msgs);
                    let source_version = format!(
                        "{team_name}:messages:{}",
                        group_key.replace(' ', "-").to_lowercase()
                    );
                    let msg = SourceFileMsg {
                        source_file: SourceFilePayload {
                            content: msg_content,
                            metadata: SourceMetadata {
                                file_path: team_name.clone(),
                                extension: "json".into(),
                                language: None,
                                scope_id: scope_id.to_string(),
                                source_version,
                                workstream: None,
                                content_role: Some("finding".into()),
                                source_origin: Some(format!("claude:team:{team_name}")),
                                parent_session_id: None,
                            },
                        },
                    };
                    writeln!(out, "{}", serde_json::to_string(&msg).unwrap()).ok();
                    team_entries += 1;
                }
            }
        } else {
            // Empty team: config missing or unparseable, still mark as ingested
            eprintln!("Warning: team {team_name} has no valid config.json, skipping entries");
        }

        if team_entries > 0 {
            total_files += 1;
        }
        newly_ingested.push(team_name);
    }

    // A7: Update .ingested state
    if !newly_ingested.is_empty() {
        append_lines(&ingested_path, &newly_ingested);
    }

    // A7: Clean up old staging directories
    cleanup_old_staging(staging_root, &read_lines_to_set(&ingested_path));

    let done = DoneMsg { done: true, total_files };
    writeln!(out, "{}", serde_json::to_string(&done).unwrap()).ok();
    out.flush().ok();
}

// ---------------------------------------------------------------------------
// Agent Teams: Config parsing
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TeamConfig {
    lead_session_id: String,
    members: Vec<TeamMember>,
    created_at: String,
}

#[derive(Debug)]
struct TeamMember {
    name: String,
    model: String,
    joined_at: String,
}

fn parse_team_config(team_dir: &Path) -> Option<TeamConfig> {
    let config_path = team_dir.join("config.json");
    let content = std::fs::read_to_string(&config_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;

    let lead_session_id = val.get("leadSessionId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let created_at = val.get("createdAt")
        .or_else(|| val.get("created_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut members = Vec::new();
    if let Some(members_val) = val.get("members").and_then(|v| v.as_array()) {
        for m in members_val {
            let name = m.get("name")
                .or_else(|| m.get("agentName"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let model = m.get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let joined_at = m.get("joinedAt")
                .or_else(|| m.get("joined_at"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            members.push(TeamMember { name, model, joined_at });
        }
    }

    Some(TeamConfig { lead_session_id, members, created_at })
}

// ---------------------------------------------------------------------------
// Agent Teams: Tasks parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TaskEvent {
    event: String,
    task_id: String,
    subject: String,
    description: String,
    owner: String,
    timestamp: String,
    full_task: Option<serde_json::Value>,
}

fn parse_tasks_jsonl(team_dir: &Path) -> Vec<TaskEvent> {
    let path = team_dir.join("tasks.jsonl");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut events = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // T13: Skip truncated/invalid last lines gracefully
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let event = val.get("event").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let task_id = val.get("task_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let subject = val.get("subject").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let description = val.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let owner = val.get("owner").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let timestamp = val.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let full_task = val.get("full_task").cloned();

        if !task_id.is_empty() {
            events.push(TaskEvent {
                event, task_id, subject, description, owner, timestamp, full_task,
            });
        }
    }
    events
}

fn group_tasks(events: &[TaskEvent]) -> BTreeMap<String, Vec<&TaskEvent>> {
    let mut grouped: BTreeMap<String, Vec<&TaskEvent>> = BTreeMap::new();
    for event in events {
        grouped.entry(event.task_id.clone()).or_default().push(event);
    }
    grouped
}

// ---------------------------------------------------------------------------
// Agent Teams: Message parsing and dedup
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct InboxMessage {
    sender: String,
    recipient: String,
    content: String,
    timestamp: String,
    message_type: String,
    task_ref: Option<String>,
}

fn parse_inbox_messages(team_dir: &Path) -> Vec<InboxMessage> {
    let inboxes_dir = team_dir.join("inboxes");
    if !inboxes_dir.is_dir() {
        return Vec::new();
    }

    let mut messages = Vec::new();

    let entries: Vec<_> = std::fs::read_dir(&inboxes_dir)
        .into_iter()
        .flatten()
        .flatten()
        .collect();

    for entry in entries {
        let path = entry.path();
        if !path.to_string_lossy().ends_with(".json") {
            continue;
        }
        let inbox_owner = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let val: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if let Some(arr) = val.as_array() {
            for msg in arr {
                let sender = msg.get("from")
                    .or_else(|| msg.get("sender"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let body = msg.get("body")
                    .or_else(|| msg.get("content"))
                    .or_else(|| msg.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let ts = msg.get("timestamp")
                    .or_else(|| msg.get("ts"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let msg_type = msg.get("type")
                    .or_else(|| msg.get("message_type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("message")
                    .to_string();
                let task_ref = msg.get("task_assignment")
                    .or_else(|| msg.get("task_id"))
                    .or_else(|| msg.get("taskId"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                if !body.is_empty() {
                    messages.push(InboxMessage {
                        sender: sender.clone(),
                        recipient: inbox_owner.clone(),
                        content: body,
                        timestamp: ts,
                        message_type: msg_type,
                        task_ref,
                    });
                }
            }
        }
    }

    messages
}

/// Deduplicate broadcast messages by hash(sender + content + timestamp).
fn dedup_broadcasts(messages: Vec<InboxMessage>) -> Vec<InboxMessage> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut result = Vec::new();

    for msg in messages {
        let mut hasher = Sha256::new();
        hasher.update(msg.sender.as_bytes());
        hasher.update(msg.content.as_bytes());
        hasher.update(msg.timestamp.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        if seen.insert(hash) {
            result.push(msg);
        }
    }

    result
}

/// Group messages by task reference. System messages go to "coordination",
/// unreferenced messages go to "general".
fn group_messages_by_task(messages: Vec<InboxMessage>) -> BTreeMap<String, Vec<InboxMessage>> {
    let mut grouped: BTreeMap<String, Vec<InboxMessage>> = BTreeMap::new();

    for msg in messages {
        let is_system = matches!(
            msg.message_type.as_str(),
            "idle_notification" | "shutdown_request" | "shutdown_response" | "system"
        );

        let group_key = if is_system {
            "coordination".to_string()
        } else if let Some(ref task_id) = msg.task_ref {
            format!("task-{task_id}")
        } else {
            "general".to_string()
        };

        grouped.entry(group_key).or_default().push(msg);
    }

    // Sort messages within each group by timestamp
    for msgs in grouped.values_mut() {
        msgs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    grouped
}

// ---------------------------------------------------------------------------
// Agent Teams: Entry formatting
// ---------------------------------------------------------------------------

fn format_team_structure(
    team_name: &str,
    config: &TeamConfig,
    tasks: &[TaskEvent],
    team_dir: &Path,
) -> String {
    let mut out = String::new();

    // Determine ended timestamp and status
    let ended = infer_team_ended(tasks, team_dir);
    let status = infer_team_status(tasks);

    let created = if config.created_at.is_empty() {
        "unknown".to_string()
    } else {
        config.created_at.clone()
    };

    out.push_str(&format!(
        "[Team: {team_name} | Created: {created} | Ended: {ended} | Status: {status}]\n"
    ));

    // Lead
    if !config.lead_session_id.is_empty() {
        out.push_str(&format!("LEAD SESSION: {}\n", config.lead_session_id));
    }

    // Members
    if !config.members.is_empty() {
        out.push_str("MEMBERS:\n");
        for member in &config.members {
            let joined = if member.joined_at.is_empty() {
                String::new()
            } else {
                format!(", joined {}", member.joined_at)
            };
            out.push_str(&format!("  - {} ({}{})\n", member.name, member.model, joined));
        }
    }

    // Task summary
    let total_tasks = count_unique_tasks(tasks);
    let completed_tasks = count_completed_tasks(tasks);
    if total_tasks > 0 {
        out.push_str(&format!(
            "TASKS: {} total ({} completed, {} pending)\n",
            total_tasks,
            completed_tasks,
            total_tasks - completed_tasks,
        ));
    }

    out
}

fn format_task_entry(team_name: &str, task_id: &str, events: &[&TaskEvent]) -> String {
    let mut out = String::new();

    // Find the most informative event (prefer completed with full_task, then sweep, then created)
    let best = events.iter()
        .max_by_key(|e| match e.event.as_str() {
            "completed" => 3,
            "sweep" => 2,
            "created" => 1,
            _ => 0,
        })
        .unwrap();

    let subject = if !best.subject.is_empty() {
        best.subject.clone()
    } else {
        // Try to extract from full_task
        best.full_task.as_ref()
            .and_then(|ft| ft.get("subject").or_else(|| ft.get("title")))
            .and_then(|v| v.as_str())
            .unwrap_or("untitled")
            .to_string()
    };

    let status = infer_task_status(events);
    out.push_str(&format!(
        "[Task #{task_id}: {subject} | Team: {team_name} | Status: {status}]\n"
    ));

    // Description
    let description = events.iter()
        .find(|e| !e.description.is_empty())
        .map(|e| e.description.clone())
        .or_else(|| {
            best.full_task.as_ref()
                .and_then(|ft| ft.get("description"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    if !description.is_empty() {
        out.push_str(&format!("DESCRIPTION: {}\n", truncate_str(&description, 2000)));
    }

    // Owner
    let owner = events.iter()
        .find(|e| !e.owner.is_empty())
        .map(|e| e.owner.clone())
        .or_else(|| {
            best.full_task.as_ref()
                .and_then(|ft| ft.get("owner"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    if !owner.is_empty() {
        out.push_str(&format!("ASSIGNED TO: {owner}\n"));
    }

    // Dependencies
    if let Some(ref ft) = best.full_task
        && let Some(blocked_by) = ft.get("blockedBy").and_then(|v| v.as_array())
    {
        let deps: Vec<String> = blocked_by.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        if !deps.is_empty() {
            out.push_str(&format!("DEPENDS ON: {}\n", deps.join(", ")));
        }
    }

    // Timestamps
    let created_ts = events.iter()
        .find(|e| e.event == "created")
        .map(|e| e.timestamp.clone())
        .unwrap_or_default();
    let completed_ts = events.iter()
        .find(|e| e.event == "completed")
        .map(|e| e.timestamp.clone())
        .unwrap_or_default();

    if !created_ts.is_empty() || !completed_ts.is_empty() {
        let ts_parts: Vec<String> = [
            if !created_ts.is_empty() { Some(format!("CREATED: {created_ts}")) } else { None },
            if !completed_ts.is_empty() { Some(format!("COMPLETED: {completed_ts}")) } else { None },
        ].into_iter().flatten().collect();
        out.push_str(&format!("{}\n", ts_parts.join(" -> ")));
    }

    out
}

fn format_message_entry(
    team_name: &str,
    group_key: &str,
    messages: &[InboxMessage],
) -> String {
    let mut out = String::new();

    out.push_str(&format!("[Messages: {group_key} | Team: {team_name}]\n"));

    for msg in messages {
        let ts = if msg.timestamp.is_empty() {
            String::new()
        } else {
            // Try to extract just the time portion for compactness
            extract_time(&msg.timestamp)
        };

        let direction = if msg.recipient == "broadcast" || msg.message_type == "broadcast" {
            format!("{} -> broadcast", msg.sender)
        } else {
            format!("{} -> {}", msg.sender, msg.recipient)
        };

        if ts.is_empty() {
            out.push_str(&format!("[--:--] {direction}:\n"));
        } else {
            out.push_str(&format!("[{ts}] {direction}:\n"));
        }
        out.push_str(&format!("  \"{}\"\n", truncate_str(&msg.content, 2000)));
    }

    out
}

// ---------------------------------------------------------------------------
// Agent Teams: Inference helpers
// ---------------------------------------------------------------------------

fn infer_team_ended(tasks: &[TaskEvent], team_dir: &Path) -> String {
    // Try .capture-log for latest event timestamp
    let capture_log = team_dir.join(".capture-log");
    let mut latest_ts = String::new();

    if let Ok(content) = std::fs::read_to_string(&capture_log) {
        for line in content.lines().rev() {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line)
                && let Some(ts) = val.get("timestamp").and_then(|v| v.as_str())
                && ts > latest_ts.as_str()
            {
                latest_ts = ts.to_string();
            }
        }
    }

    // Also check latest task completion time
    for task in tasks {
        if task.event == "completed" && task.timestamp > latest_ts {
            latest_ts = task.timestamp.clone();
        }
    }

    if latest_ts.is_empty() {
        "unknown".to_string()
    } else {
        latest_ts
    }
}

fn infer_team_status(tasks: &[TaskEvent]) -> String {
    if tasks.is_empty() {
        return "unknown".to_string();
    }

    // Build map of task_id -> latest event
    let mut task_status: HashMap<&str, &str> = HashMap::new();
    for event in tasks {
        task_status.insert(&event.task_id, &event.event);
    }

    let all_done = task_status.values().all(|status| {
        matches!(*status, "completed" | "deleted")
    });

    if all_done {
        "completed".to_string()
    } else {
        // Check if there are any completed events at all (mixed state = abandoned)
        let any_completed = task_status.values().any(|s| *s == "completed");
        if any_completed {
            "abandoned".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

fn infer_task_status(events: &[&TaskEvent]) -> String {
    // Latest event type determines status
    if events.iter().any(|e| e.event == "completed") {
        return "completed".to_string();
    }
    if let Some(last) = events.iter().max_by_key(|e| &e.timestamp)
        && last.event == "sweep"
        && let Some(ref ft) = last.full_task
        && let Some(status) = ft.get("status").and_then(|v| v.as_str())
    {
        return status.to_string();
    }
    "pending".to_string()
}

fn count_unique_tasks(tasks: &[TaskEvent]) -> usize {
    let ids: HashSet<&str> = tasks.iter().map(|t| t.task_id.as_str()).collect();
    ids.len()
}

fn count_completed_tasks(tasks: &[TaskEvent]) -> usize {
    let mut completed: HashSet<&str> = HashSet::new();
    for event in tasks {
        if event.event == "completed" {
            completed.insert(&event.task_id);
        }
    }
    completed.len()
}

fn extract_time(timestamp: &str) -> String {
    // Try to extract HH:MM from ISO 8601 timestamp
    if let Some(t_pos) = timestamp.find('T') {
        let time_part = &timestamp[t_pos + 1..];
        if time_part.len() >= 5 {
            return time_part[..5].to_string();
        }
    }
    timestamp.to_string()
}

/// Clean up staging directories that are in .ingested and older than retention period.
/// Also cleans up stale .lock files.
fn cleanup_old_staging(staging_root: &Path, ingested: &HashSet<String>) {
    let now = std::time::SystemTime::now();

    let entries = match std::fs::read_dir(staging_root) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();

        // Clean up .lock files in team directories
        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            cleanup_lock_files(&entry.path());
        }

        // Only clean up ingested team directories
        if !ingested.contains(&name) {
            continue;
        }
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }

        // Check age
        if let Ok(meta) = entry.metadata()
            && let Ok(modified) = meta.modified()
            && let Ok(age) = now.duration_since(modified)
            && age.as_secs() > STAGING_RETENTION_SECS
            && let Err(e) = std::fs::remove_dir_all(entry.path())
        {
            eprintln!("Warning: failed to clean up staging dir {name}: {e}");
        }
    }
}

fn cleanup_lock_files(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".lock") {
                let _ = std::fs::remove_file(entry.path());
            }
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

    #[test]
    fn test_subagent_parent_session_id_on_turn_1_only() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path();

        // Create a subagent session with parent_session_id
        let jsonl = r#"{"type":"session_start","session_id":"ses-sub-001","timestamp":"2026-03-14T05:00:00Z","workspace":"/tmp","git_branch":"feat/auth","agent_type":"subagent","parent_session_id":"ses-parent-001","corvia_agent_id":null}
{"type":"user_prompt","session_id":"ses-sub-001","turn":1,"timestamp":"2026-03-14T05:01:00Z","content":"find auth patterns"}
{"type":"tool_end","session_id":"ses-sub-001","turn":1,"timestamp":"2026-03-14T05:01:01Z","tool":"Grep","input":{"pattern":"auth"},"output":"found","truncated":false,"success":true}
{"type":"tool_end","session_id":"ses-sub-001","turn":1,"timestamp":"2026-03-14T05:01:02Z","tool":"Read","input":{"file_path":"src/auth.rs"},"output":"impl Auth","truncated":false,"success":true}
{"type":"user_prompt","session_id":"ses-sub-001","turn":2,"timestamp":"2026-03-14T05:02:00Z","content":"summarize findings"}
{"type":"session_end","session_id":"ses-sub-001","timestamp":"2026-03-14T05:03:00Z","total_turns":2,"duration_ms":180000}"#;

        let gz_path = sessions.join("ses-sub-001.jsonl.gz");
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();

        let mut output = Vec::new();
        ingest_sessions_from(DEFAULT_SCOPE, sessions, &mut output);

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();

        // 2 source_file messages (turn 1 + turn 2) + 1 done
        assert_eq!(lines.len(), 3, "got: {:?}", lines);

        // Turn 1 should have parent_session_id
        let turn1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(
            turn1["source_file"]["metadata"]["parent_session_id"].as_str(),
            Some("ses-parent-001"),
            "turn 1 should carry parent_session_id"
        );
        assert_eq!(
            turn1["source_file"]["metadata"]["source_origin"].as_str(),
            Some("claude:subagent")
        );
        assert_eq!(
            turn1["source_file"]["metadata"]["content_role"].as_str(),
            Some("research"),
            "subagent with Grep+Read should be 'research'"
        );

        // Turn 2 should NOT have parent_session_id (only turn 1 gets it)
        let turn2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert!(
            turn2["source_file"]["metadata"]["parent_session_id"].is_null(),
            "turn 2 should not carry parent_session_id"
        );
    }

    #[test]
    fn test_main_session_no_parent_id() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path();

        let jsonl = r#"{"type":"session_start","session_id":"ses-main-001","timestamp":"2026-03-14T05:00:00Z","workspace":"/tmp","git_branch":"main","agent_type":"main","parent_session_id":null,"corvia_agent_id":null}
{"type":"user_prompt","session_id":"ses-main-001","turn":1,"timestamp":"2026-03-14T05:01:00Z","content":"hello"}
{"type":"session_end","session_id":"ses-main-001","timestamp":"2026-03-14T05:02:00Z","total_turns":1,"duration_ms":60000}"#;

        let gz_path = sessions.join("ses-main-001.jsonl.gz");
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();

        let mut output = Vec::new();
        ingest_sessions_from(DEFAULT_SCOPE, sessions, &mut output);

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();
        let turn1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();

        // Main session should never have parent_session_id
        assert!(
            turn1["source_file"]["metadata"]["parent_session_id"].is_null(),
            "main session should not carry parent_session_id"
        );
    }

    // ===================================================================
    // Agent Teams tests (T4, T5, T6, T10, T13, T14 from RFC)
    // ===================================================================

    /// Create a minimal team staging directory with config, tasks, and inboxes.
    fn setup_team_staging(
        staging_root: &Path,
        team_name: &str,
        config: &serde_json::Value,
        tasks_jsonl: &str,
        inboxes: &[(&str, &str)],
    ) {
        let team_dir = staging_root.join(team_name);
        std::fs::create_dir_all(&team_dir).unwrap();

        // config.json
        std::fs::write(
            team_dir.join("config.json"),
            serde_json::to_string(config).unwrap(),
        ).unwrap();

        // tasks.jsonl
        if !tasks_jsonl.is_empty() {
            std::fs::write(team_dir.join("tasks.jsonl"), tasks_jsonl).unwrap();
        }

        // inboxes
        if !inboxes.is_empty() {
            let inbox_dir = team_dir.join("inboxes");
            std::fs::create_dir_all(&inbox_dir).unwrap();
            for (name, content) in inboxes {
                std::fs::write(inbox_dir.join(format!("{name}.json")), content).unwrap();
            }
        }
    }

    fn parse_output_entries(output: &[u8]) -> (Vec<serde_json::Value>, serde_json::Value) {
        let text = String::from_utf8_lossy(output);
        let lines: Vec<&str> = text.trim().lines().collect();
        assert!(!lines.is_empty(), "expected at least a done message");

        let done: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        let entries: Vec<serde_json::Value> = lines[..lines.len() - 1]
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        (entries, done)
    }

    // T4: Adapter parses staging files into correct SourceFile entries
    #[test]
    fn t4_adapter_parses_staging_into_source_files() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-lead-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": [
                {"name": "researcher", "model": "haiku", "joinedAt": "2026-03-28T10:00:01Z"},
                {"name": "reviewer", "model": "sonnet", "joinedAt": "2026-03-28T10:00:02Z"}
            ]
        });

        let tasks = concat!(
            r#"{"v":1,"event":"created","task_id":"1","subject":"Review auth","description":"Check auth module","timestamp":"2026-03-28T10:01:00Z"}"#, "\n",
            r#"{"v":1,"event":"completed","task_id":"1","subject":"Review auth","owner":"researcher","timestamp":"2026-03-28T10:15:00Z","full_task":{"status":"completed","owner":"researcher","blockedBy":[]}}"#, "\n"
        );

        let inboxes: Vec<(&str, &str)> = vec![
            ("researcher", r#"[{"from":"reviewer","body":"Found an issue in token handling","timestamp":"2026-03-28T10:05:00Z","task_assignment":"1"}]"#),
            ("reviewer", r#"[{"from":"researcher","body":"Acknowledged, fixing now","timestamp":"2026-03-28T10:06:00Z","task_assignment":"1"}]"#),
        ];

        setup_team_staging(staging, "security-review", &config, tasks, &inboxes);

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let (entries, done) = parse_output_entries(&output);

        // Should produce: 1 team structure + 1 task + 1 message group = 3 entries
        assert!(done["done"].as_bool().unwrap());
        assert_eq!(done["total_files"].as_u64().unwrap(), 1); // 1 team
        assert_eq!(entries.len(), 3, "expected 3 entries (team + task + messages), got {}", entries.len());

        // Team structure entry
        let team = &entries[0];
        assert!(team["source_file"]["content"].as_str().unwrap().contains("[Team: security-review"));
        assert_eq!(team["source_file"]["metadata"]["content_role"].as_str().unwrap(), "memory");
        assert_eq!(team["source_file"]["metadata"]["source_origin"].as_str().unwrap(), "claude:team:security-review");
        assert_eq!(team["source_file"]["metadata"]["source_version"].as_str().unwrap(), "security-review:config");
        assert_eq!(team["source_file"]["metadata"]["scope_id"].as_str().unwrap(), "user-history");

        // Task entry
        let task = &entries[1];
        assert!(task["source_file"]["content"].as_str().unwrap().contains("[Task #1: Review auth"));
        assert_eq!(task["source_file"]["metadata"]["content_role"].as_str().unwrap(), "task");
        assert_eq!(task["source_file"]["metadata"]["source_version"].as_str().unwrap(), "security-review:task:1");

        // Message entry
        let msgs = &entries[2];
        assert!(msgs["source_file"]["content"].as_str().unwrap().contains("Found an issue"));
        assert_eq!(msgs["source_file"]["metadata"]["content_role"].as_str().unwrap(), "finding");
        assert!(msgs["source_file"]["metadata"]["source_version"].as_str().unwrap().contains("messages:task-1"));
    }

    // T5: Task-grouped message chunking produces expected entry count
    #[test]
    fn t5_message_grouping_by_task() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": [{"name": "a", "model": "haiku", "joinedAt": ""}]
        });

        // Two tasks
        let tasks = concat!(
            r#"{"v":1,"event":"created","task_id":"1","subject":"Task A","description":"","timestamp":"2026-03-28T10:01:00Z"}"#, "\n",
            r#"{"v":1,"event":"created","task_id":"2","subject":"Task B","description":"","timestamp":"2026-03-28T10:02:00Z"}"#, "\n"
        );

        // Messages referencing different tasks + one general + one system
        let inbox = serde_json::json!([
            {"from":"a","body":"working on task 1","timestamp":"2026-03-28T10:05:00Z","task_assignment":"1"},
            {"from":"a","body":"working on task 2","timestamp":"2026-03-28T10:06:00Z","task_assignment":"2"},
            {"from":"a","body":"general update","timestamp":"2026-03-28T10:07:00Z"},
            {"from":"system","body":"teammate idle","timestamp":"2026-03-28T10:08:00Z","type":"idle_notification"}
        ]);

        setup_team_staging(staging, "multi-task", &config, tasks, &[
            ("team-lead", &serde_json::to_string(&inbox).unwrap()),
        ]);

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let (entries, _) = parse_output_entries(&output);

        // 1 team + 2 tasks + 4 message groups (task-1, task-2, general, coordination)
        assert_eq!(entries.len(), 7, "expected 7 entries, got {}: {:?}",
            entries.len(),
            entries.iter().map(|e| e["source_file"]["metadata"]["source_version"].as_str().unwrap_or("?")).collect::<Vec<_>>()
        );

        // Verify message groups exist
        let versions: Vec<&str> = entries.iter()
            .map(|e| e["source_file"]["metadata"]["source_version"].as_str().unwrap_or(""))
            .collect();
        assert!(versions.iter().any(|v| v.contains("messages:task-1")));
        assert!(versions.iter().any(|v| v.contains("messages:task-2")));
        assert!(versions.iter().any(|v| v.contains("messages:general")));
        assert!(versions.iter().any(|v| v.contains("messages:coordination")));
    }

    // T6: Broadcast dedup removes duplicate messages
    #[test]
    fn t6_broadcast_dedup() {
        // Same message in 5 inboxes (broadcast to all teammates)
        let broadcast_msg = serde_json::json!({
            "from": "lead",
            "body": "Consensus: refresh token handling needs a rewrite.",
            "timestamp": "2026-03-28T10:08:00Z"
        });

        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": [
                {"name": "a", "model": "haiku", "joinedAt": ""},
                {"name": "b", "model": "haiku", "joinedAt": ""},
                {"name": "c", "model": "haiku", "joinedAt": ""},
                {"name": "d", "model": "haiku", "joinedAt": ""},
                {"name": "e", "model": "haiku", "joinedAt": ""}
            ]
        });

        let inbox_content = serde_json::to_string(&serde_json::json!([broadcast_msg])).unwrap();
        setup_team_staging(staging, "bcast-team", &config, "", &[
            ("a", &inbox_content),
            ("b", &inbox_content),
            ("c", &inbox_content),
            ("d", &inbox_content),
            ("e", &inbox_content),
        ]);

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let (entries, _) = parse_output_entries(&output);

        // 1 team + 1 message group (deduped broadcast)
        assert_eq!(entries.len(), 2, "expected 2 entries (team + 1 message group), got {}", entries.len());

        // The message entry should contain the broadcast content exactly once
        let msg_entry = &entries[1];
        let content = msg_entry["source_file"]["content"].as_str().unwrap();
        let count = content.matches("Consensus: refresh token handling").count();
        assert_eq!(count, 1, "broadcast should appear exactly once after dedup, found {count}");
    }

    // T10: Idempotency - re-running adapter on same staging produces no duplicates
    #[test]
    fn t10_idempotency() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": [{"name": "a", "model": "haiku", "joinedAt": ""}]
        });
        let tasks = r#"{"v":1,"event":"created","task_id":"1","subject":"Task A","description":"do stuff","timestamp":"2026-03-28T10:01:00Z"}"#;
        setup_team_staging(staging, "idem-team", &config, tasks, &[]);

        // First run
        let mut output1 = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output1);
        let (entries1, done1) = parse_output_entries(&output1);
        assert_eq!(done1["total_files"].as_u64().unwrap(), 1);
        assert_eq!(entries1.len(), 2); // team + task

        // Second run (same staging)
        let mut output2 = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output2);
        let (entries2, done2) = parse_output_entries(&output2);
        assert_eq!(done2["total_files"].as_u64().unwrap(), 0, "second run should produce 0 files");
        assert_eq!(entries2.len(), 0, "second run should produce 0 entries");

        // Verify .ingested contains the team
        let ingested = std::fs::read_to_string(staging.join(".ingested")).unwrap();
        assert!(ingested.contains("idem-team"));
    }

    // T13: Partial staging - truncated last line in tasks.jsonl is skipped
    #[test]
    fn t13_partial_jsonl_handling() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": []
        });

        // Valid line + truncated line
        let tasks = concat!(
            r#"{"v":1,"event":"created","task_id":"1","subject":"Valid task","description":"works","timestamp":"2026-03-28T10:01:00Z"}"#, "\n",
            r#"{"v":1,"event":"completed","task_id":"2","subject":"Trunc"#  // truncated!
        );

        setup_team_staging(staging, "partial-team", &config, tasks, &[]);

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let (entries, done) = parse_output_entries(&output);
        assert_eq!(done["total_files"].as_u64().unwrap(), 1);
        // Should produce team + 1 valid task (truncated line skipped)
        assert_eq!(entries.len(), 2, "expected 2 entries (team + 1 valid task), got {}", entries.len());

        let task = &entries[1];
        assert!(task["source_file"]["content"].as_str().unwrap().contains("Valid task"));
    }

    // T14: Empty team - config.json with members but no tasks produces valid team entry
    #[test]
    fn t14_empty_team() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        let config = serde_json::json!({
            "leadSessionId": "sess-001",
            "createdAt": "2026-03-28T10:00:00Z",
            "members": [{"name": "solo", "model": "opus", "joinedAt": "2026-03-28T10:00:01Z"}]
        });

        // No tasks, no inboxes
        setup_team_staging(staging, "empty-team", &config, "", &[]);

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let (entries, done) = parse_output_entries(&output);
        assert_eq!(done["total_files"].as_u64().unwrap(), 1);
        assert_eq!(entries.len(), 1, "empty team should produce exactly 1 entry (team structure)");

        let team = &entries[0];
        let content = team["source_file"]["content"].as_str().unwrap();
        assert!(content.contains("[Team: empty-team"));
        assert!(content.contains("solo (opus"));
        assert_eq!(team["source_file"]["metadata"]["content_role"].as_str().unwrap(), "memory");
    }

    // Additional: Domain dispatch routes correctly
    #[test]
    fn test_domain_dispatch_agent_teams() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path();

        // Create empty staging dir (no teams)
        std::fs::create_dir_all(staging).unwrap();

        let mut output = Vec::new();
        ingest_agent_teams_from(DEFAULT_SCOPE, staging, &mut output);

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"done\":true"));
        assert!(text.contains("\"total_files\":0"));
    }

    // Additional: Session ingestion still works (regression check)
    #[test]
    fn test_sessions_ingestion_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path();

        let jsonl = r#"{"type":"session_start","session_id":"ses-regression-001","timestamp":"2026-03-28T05:00:00Z","workspace":"/tmp","git_branch":"main","agent_type":"main","parent_session_id":null,"corvia_agent_id":null}
{"type":"user_prompt","session_id":"ses-regression-001","turn":1,"timestamp":"2026-03-28T05:01:00Z","content":"regression test"}
{"type":"session_end","session_id":"ses-regression-001","timestamp":"2026-03-28T05:02:00Z","total_turns":1,"duration_ms":60000}"#;

        let gz_path = sessions.join("ses-regression-001.jsonl.gz");
        let file = std::fs::File::create(&gz_path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(jsonl.as_bytes()).unwrap();
        encoder.finish().unwrap();

        let mut output = Vec::new();
        ingest_sessions_from(DEFAULT_SCOPE, sessions, &mut output);

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"source_file\""));
        assert!(text.contains("ses-regression-001"));
        assert!(text.contains("\"content_role\":\"session-turn\""));
    }

    // Additional: Team status inference
    #[test]
    fn test_team_status_completed() {
        let tasks = vec![
            TaskEvent { event: "created".into(), task_id: "1".into(), subject: "".into(), description: "".into(), owner: "".into(), timestamp: "".into(), full_task: None },
            TaskEvent { event: "completed".into(), task_id: "1".into(), subject: "".into(), description: "".into(), owner: "".into(), timestamp: "".into(), full_task: None },
        ];
        assert_eq!(infer_team_status(&tasks), "completed");
    }

    #[test]
    fn test_team_status_abandoned() {
        let tasks = vec![
            TaskEvent { event: "created".into(), task_id: "1".into(), subject: "".into(), description: "".into(), owner: "".into(), timestamp: "".into(), full_task: None },
            TaskEvent { event: "completed".into(), task_id: "1".into(), subject: "".into(), description: "".into(), owner: "".into(), timestamp: "".into(), full_task: None },
            TaskEvent { event: "created".into(), task_id: "2".into(), subject: "".into(), description: "".into(), owner: "".into(), timestamp: "".into(), full_task: None },
        ];
        assert_eq!(infer_team_status(&tasks), "abandoned");
    }

    #[test]
    fn test_team_status_unknown_no_tasks() {
        let tasks: Vec<TaskEvent> = vec![];
        assert_eq!(infer_team_status(&tasks), "unknown");
    }

    #[test]
    fn test_extract_time() {
        assert_eq!(extract_time("2026-03-28T10:05:00Z"), "10:05");
        assert_eq!(extract_time("2026-03-28T10:05:00.123Z"), "10:05");
        assert_eq!(extract_time("no-t-here"), "no-t-here");
    }

    #[test]
    fn test_broadcast_dedup_different_senders() {
        // Same content from different senders should NOT be deduped
        let messages = vec![
            InboxMessage {
                sender: "alice".into(), recipient: "bob".into(),
                content: "same message".into(), timestamp: "2026-03-28T10:00:00Z".into(),
                message_type: "message".into(), task_ref: None,
            },
            InboxMessage {
                sender: "carol".into(), recipient: "bob".into(),
                content: "same message".into(), timestamp: "2026-03-28T10:00:00Z".into(),
                message_type: "message".into(), task_ref: None,
            },
        ];
        let deduped = dedup_broadcasts(messages);
        assert_eq!(deduped.len(), 2, "different senders should not be deduped");
    }
}
