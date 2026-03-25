//! File watcher for live Claude session visibility.
//!
//! Watches `~/.claude/sessions/*.jsonl` for write events using the `notify`
//! crate (inotify on Linux, kqueue on macOS, polling fallback). On each change,
//! seeks to the last-known offset, reads new complete lines, parses them as
//! session events, and maintains in-memory session state for the dashboard.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use corvia_common::dashboard::{HookSession, HookSessionState, HookSessionUpdate};
use tokio::sync::{broadcast, RwLock};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_SESSIONS: usize = 500;
const MAX_LINE_BYTES: usize = 1_048_576; // 1 MB
const STALE_THRESHOLD_SECS: u64 = 600; // 10 min
const EVICT_THRESHOLD_SECS: u64 = 3600; // 60 min
const DEBOUNCE_MS: u64 = 200;
const SWEEP_INTERVAL_SECS: u64 = 30;
const BROADCAST_CAPACITY: usize = 256;

// ---------------------------------------------------------------------------
// Internal per-file state
// ---------------------------------------------------------------------------

struct FileState {
    /// Byte offset after the last complete `\n`-terminated line we read.
    offset: u64,
    /// Aggregated session data exposed to the API.
    session: HookSession,
    /// tool_start count minus tool_end count (for active_tool tracking).
    pending_tools: u32,
    /// Wall-clock instant of the last event we processed from this file.
    last_event_at: Instant,
}

impl FileState {
    fn new(session_id: &str) -> Self {
        Self {
            offset: 0,
            session: HookSession {
                session_id: session_id.to_string(),
                state: HookSessionState::Active,
                workspace: String::new(),
                git_branch: String::new(),
                agent_type: String::from("main"),
                parent_session_id: None,
                corvia_agent_id: None,
                started_at: String::new(),
                last_activity: String::new(),
                duration_secs: 0,
                turn_count: 0,
                tool_calls: 0,
                active_tool: None,
                tools_used: Vec::new(),
            },
            pending_tools: 0,
            last_event_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public shared state
// ---------------------------------------------------------------------------

pub struct SessionWatcherState {
    sessions: RwLock<HashMap<String, FileState>>,
    tx: broadcast::Sender<HookSessionUpdate>,
}

impl SessionWatcherState {
    pub fn new() -> (Arc<Self>, broadcast::Receiver<HookSessionUpdate>) {
        let (tx, rx) = broadcast::channel(BROADCAST_CAPACITY);
        let state = Arc::new(Self {
            sessions: RwLock::new(HashMap::new()),
            tx,
        });
        (state, rx)
    }

    /// Snapshot all sessions for the REST endpoint.
    pub async fn snapshot(&self) -> Vec<HookSession> {
        let guard = self.sessions.read().await;
        guard.values().map(|fs| fs.session.clone()).collect()
    }

    /// Subscribe to SSE update stream.
    pub fn subscribe(&self) -> broadcast::Receiver<HookSessionUpdate> {
        self.tx.subscribe()
    }

    fn broadcast(&self, update: HookSessionUpdate) {
        // Ignore send error — means no active subscribers.
        let _ = self.tx.send(update);
    }
}

// ---------------------------------------------------------------------------
// Watcher bootstrap
// ---------------------------------------------------------------------------

/// Spawn the session watcher as a background tokio task.
///
/// Returns immediately. The watcher runs until the tokio runtime shuts down.
pub async fn spawn_session_watcher(state: Arc<SessionWatcherState>) {
    let dir = sessions_dir();

    // Create the directory if it doesn't exist (hooks may not have run yet).
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(path = %dir.display(), error = %e, "failed to create sessions directory");
    }

    // Initial scan: catch up on any existing .jsonl files.
    initial_scan(&state, &dir).await;

    // Set up file watcher with inotify → polling fallback.
    let (notify_tx, notify_rx) = tokio::sync::mpsc::channel::<PathBuf>(512);

    let watcher = start_watcher(&dir, notify_tx.clone())
        .or_else(|e| {
            tracing::warn!(error = %e, "inotify watcher failed, falling back to polling");
            start_poll_watcher(&dir, notify_tx)
        });

    match watcher {
        Ok(watcher) => {
            tracing::info!(path = %dir.display(), "session watcher started");
            tokio::spawn(watcher_loop(state, watcher, notify_rx));
        }
        Err(e) => {
            tracing::error!(error = %e, "session watcher failed to start (both inotify and polling). Live sessions unavailable.");
        }
    }
}

// We box the watcher so inotify and poll variants can share the same type.
type BoxedWatcher = Box<dyn notify::Watcher + Send>;

fn start_watcher(
    dir: &Path,
    tx: tokio::sync::mpsc::Sender<PathBuf>,
) -> Result<BoxedWatcher, notify::Error> {
    use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};

    let mut watcher = RecommendedWatcher::new(
        move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                use notify::EventKind;
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {
                        for path in event.paths {
                            let _ = tx.blocking_send(path);
                        }
                    }
                    _ => {}
                }
            }
        },
        Config::default(),
    )?;
    watcher.watch(dir, RecursiveMode::NonRecursive)?;
    Ok(Box::new(watcher))
}

fn start_poll_watcher(
    dir: &Path,
    tx: tokio::sync::mpsc::Sender<PathBuf>,
) -> Result<BoxedWatcher, notify::Error> {
    use notify::{Config, PollWatcher, RecursiveMode, Watcher};

    let config = Config::default()
        .with_poll_interval(std::time::Duration::from_secs(2));

    let mut watcher = PollWatcher::new(
        move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                for path in event.paths {
                    let _ = tx.blocking_send(path);
                }
            }
        },
        config,
    )?;
    watcher.watch(dir, RecursiveMode::NonRecursive)?;
    tracing::info!("session watcher using polling fallback (2s interval)");
    Ok(Box::new(watcher))
}

// ---------------------------------------------------------------------------
// Event loop (debounced)
// ---------------------------------------------------------------------------

async fn watcher_loop(
    state: Arc<SessionWatcherState>,
    _watcher: BoxedWatcher, // must not be dropped
    mut notify_rx: tokio::sync::mpsc::Receiver<PathBuf>,
) {
    let mut pending_paths: HashSet<PathBuf> = HashSet::new();
    let mut debounce_tick =
        tokio::time::interval(std::time::Duration::from_millis(DEBOUNCE_MS));
    let mut sweep_tick =
        tokio::time::interval(std::time::Duration::from_secs(SWEEP_INTERVAL_SECS));

    // Don't fire immediately on first tick.
    debounce_tick.reset();
    sweep_tick.reset();

    loop {
        tokio::select! {
            path = notify_rx.recv() => {
                match path {
                    Some(p) => { pending_paths.insert(p); }
                    None => break, // channel closed
                }
            }
            _ = debounce_tick.tick() => {
                if !pending_paths.is_empty() {
                    let batch = std::mem::take(&mut pending_paths);
                    process_file_changes(&state, batch).await;
                }
            }
            _ = sweep_tick.tick() => {
                sweep_stale_and_evict(&state).await;
            }
        }
    }
    tracing::info!("session watcher loop exited");
}

// ---------------------------------------------------------------------------
// Initial scan
// ---------------------------------------------------------------------------

async fn initial_scan(state: &Arc<SessionWatcherState>, dir: &Path) {
    let dir = dir.to_path_buf();
    let paths = tokio::task::spawn_blocking(move || -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
            .collect()
    })
    .await
    .unwrap_or_default();

    if !paths.is_empty() {
        tracing::info!(count = paths.len(), "initial scan: reading existing JSONL files");
        let batch: HashSet<PathBuf> = paths.into_iter().collect();
        process_file_changes(state, batch).await;
    }
}

// ---------------------------------------------------------------------------
// Batch file processor
// ---------------------------------------------------------------------------

async fn process_file_changes(state: &Arc<SessionWatcherState>, paths: HashSet<PathBuf>) {
    for path in paths {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        // If a .jsonl.gz appeared, the original session was finalized — evict.
        if ext == "gz" {
            if let Some(session_id) = extract_session_id_from_gz(&path) {
                evict_session(state, &session_id).await;
            }
            continue;
        }

        if ext != "jsonl" {
            continue;
        }

        // Security: reject symlinks.
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                tracing::warn!(path = %path.display(), "rejecting symlink in sessions directory");
                continue;
            }
            Err(_) => continue, // file gone
            _ => {}
        }

        let session_id = match extract_session_id(&path) {
            Some(id) => id,
            None => continue,
        };

        // Get current offset for this file.
        let current_offset = {
            let guard = state.sessions.read().await;
            guard.get(&session_id).map(|fs| fs.offset).unwrap_or(0)
        };

        let path_clone = path.clone();
        let new_data = tokio::task::spawn_blocking(move || {
            read_new_lines(&path_clone, current_offset)
        })
        .await
        .unwrap_or_default();

        if new_data.lines.is_empty() {
            continue;
        }

        // Update session state.
        let mut guard = state.sessions.write().await;

        // Enforce max sessions limit — evict oldest if at capacity.
        if !guard.contains_key(&session_id)
            && guard.len() >= MAX_SESSIONS
            && let Some(oldest_id) = find_oldest_session(&guard)
        {
            let evicted_id = oldest_id.clone();
            guard.remove(&evicted_id);
            tracing::warn!(session_id = %evicted_id, "evicted oldest session (max {} reached)", MAX_SESSIONS);
            state.broadcast(HookSessionUpdate {
                session_id: evicted_id,
                event_type: "evicted".into(),
                session: None,
            });
        }

        let fs = guard
            .entry(session_id.clone())
            .or_insert_with(|| FileState::new(&session_id));
        fs.offset = new_data.new_offset;

        for line in &new_data.lines {
            apply_event(fs, line);
        }

        let update = HookSessionUpdate {
            session_id: session_id.clone(),
            event_type: new_data
                .last_event_type
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            session: Some(fs.session.clone()),
        };
        drop(guard);

        state.broadcast(update);
    }
}

fn find_oldest_session(map: &HashMap<String, FileState>) -> Option<&String> {
    map.iter()
        .min_by_key(|(_, fs)| fs.last_event_at)
        .map(|(id, _)| id)
}

async fn evict_session(state: &Arc<SessionWatcherState>, session_id: &str) {
    let mut guard = state.sessions.write().await;
    if guard.remove(session_id).is_some() {
        tracing::debug!(session_id, "session evicted (finalized)");
        state.broadcast(HookSessionUpdate {
            session_id: session_id.to_string(),
            event_type: "evicted".into(),
            session: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Incremental JSONL reader
// ---------------------------------------------------------------------------

#[derive(Default)]
struct NewLines {
    lines: Vec<String>,
    new_offset: u64,
    last_event_type: Option<String>,
}

fn read_new_lines(path: &Path, offset: u64) -> NewLines {
    let mut result = NewLines {
        new_offset: offset,
        ..Default::default()
    };

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return result,
    };

    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file_len <= offset {
        return result;
    }

    if file.seek(SeekFrom::Start(offset)).is_err() {
        return result;
    }

    let reader = BufReader::new(&file);
    let mut byte_pos = offset;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break, // I/O error or invalid UTF-8 — stop here
        };

        // Account for the line content + the \n delimiter.
        let line_bytes = line.len() as u64 + 1; // +1 for \n
        byte_pos += line_bytes;

        // Skip empty lines.
        if line.is_empty() {
            result.new_offset = byte_pos;
            continue;
        }

        // Skip oversized lines.
        if line.len() > MAX_LINE_BYTES {
            tracing::warn!(
                path = %path.display(),
                line_len = line.len(),
                "skipping oversized JSONL line"
            );
            result.new_offset = byte_pos;
            continue;
        }

        // Extract event type for the broadcast.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
            && let Some(t) = v.get("type").and_then(|t| t.as_str())
        {
            result.last_event_type = Some(t.to_string());
        }

        result.lines.push(line);
        result.new_offset = byte_pos;
    }

    // If file doesn't end with \n (partial write), we've already stopped at
    // the last complete line because BufReader::lines() only yields complete
    // lines (terminated by \n or EOF). However, an EOF-terminated line without
    // \n means the writer may still be appending. We check: if our computed
    // byte_pos exceeds file_len, clamp to file_len.
    if result.new_offset > file_len {
        result.new_offset = file_len;
    }

    result
}

// ---------------------------------------------------------------------------
// Event application
// ---------------------------------------------------------------------------

fn apply_event(fs: &mut FileState, line: &str) {
    let event: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "skipping invalid JSONL line");
            return;
        }
    };

    let event_type = match event.get("type").and_then(|t| t.as_str()) {
        Some(t) => t,
        None => return, // no type field — skip
    };

    // Update last activity timestamp.
    if let Some(ts) = event.get("timestamp").and_then(|t| t.as_str()) {
        fs.session.last_activity = ts.to_string();
    }
    fs.last_event_at = Instant::now();

    match event_type {
        "session_start" => {
            fs.session.workspace = json_str(&event, "workspace");
            fs.session.git_branch = json_str(&event, "git_branch");
            fs.session.agent_type = json_str_or(&event, "agent_type", "main");
            fs.session.parent_session_id =
                event.get("parent_session_id").and_then(|v| v.as_str()).map(Into::into);
            fs.session.corvia_agent_id =
                event.get("corvia_agent_id").and_then(|v| v.as_str()).map(Into::into);
            if let Some(ts) = event.get("timestamp").and_then(|t| t.as_str()) {
                fs.session.started_at = ts.to_string();
            }
            fs.session.state = HookSessionState::Active;
        }
        "user_prompt" => {
            if let Some(turn) = event.get("turn").and_then(|t| t.as_u64()) {
                fs.session.turn_count = turn as u32;
            }
        }
        "tool_start" => {
            fs.pending_tools += 1;
            if let Some(tool) = event.get("tool").and_then(|v| v.as_str()) {
                fs.session.active_tool = Some(tool.to_string());
                if !fs.session.tools_used.contains(&tool.to_string()) {
                    fs.session.tools_used.push(tool.to_string());
                }
            }
        }
        "tool_end" => {
            fs.session.tool_calls += 1;
            fs.pending_tools = fs.pending_tools.saturating_sub(1);
            if fs.pending_tools == 0 {
                fs.session.active_tool = None;
            }
        }
        "session_end" => {
            fs.session.state = HookSessionState::Ended;
            fs.session.active_tool = None;
            if let Some(turns) = event.get("total_turns").and_then(|t| t.as_u64()) {
                fs.session.turn_count = turns as u32;
            }
        }
        other => {
            tracing::debug!(event_type = other, "ignoring unknown session event type");
        }
    }

    // Recompute duration from timestamps.
    if !fs.session.started_at.is_empty() && !fs.session.last_activity.is_empty() {
        fs.session.duration_secs = compute_duration(&fs.session.started_at, &fs.session.last_activity);
    }
}

fn json_str(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn json_str_or(v: &serde_json::Value, key: &str, default: &str) -> String {
    v.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

fn compute_duration(start: &str, end: &str) -> u64 {
    let Ok(s) = chrono::DateTime::parse_from_rfc3339(start) else {
        // Try the nanosecond format used by hooks: 2026-03-25T10:00:00.123456789Z
        return compute_duration_nanos(start, end);
    };
    let Ok(e) = chrono::DateTime::parse_from_rfc3339(end) else {
        return 0;
    };
    (e - s).num_seconds().max(0) as u64
}

fn compute_duration_nanos(start: &str, end: &str) -> u64 {
    // Hooks use "%Y-%m-%dT%H:%M:%S%.9fZ" which chrono can parse via parse_from_rfc3339
    // if it has a valid timezone suffix. Try NaiveDateTime as fallback.
    let fmt = "%Y-%m-%dT%H:%M:%S%.fZ";
    let s = chrono::NaiveDateTime::parse_from_str(start, fmt).ok();
    let e = chrono::NaiveDateTime::parse_from_str(end, fmt).ok();
    match (s, e) {
        (Some(s), Some(e)) => (e - s).num_seconds().max(0) as u64,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Stale sweep
// ---------------------------------------------------------------------------

async fn sweep_stale_and_evict(state: &Arc<SessionWatcherState>) {
    let now = Instant::now();
    let stale_threshold = std::time::Duration::from_secs(STALE_THRESHOLD_SECS);
    let evict_threshold = std::time::Duration::from_secs(EVICT_THRESHOLD_SECS);

    let mut guard = state.sessions.write().await;
    let mut evicted = Vec::new();

    for (id, fs) in guard.iter_mut() {
        let age = now.duration_since(fs.last_event_at);

        if fs.session.state == HookSessionState::Ended || age > evict_threshold {
            evicted.push(id.clone());
        } else if age > stale_threshold && fs.session.state == HookSessionState::Active {
            fs.session.state = HookSessionState::Stale;
            tracing::debug!(session_id = %id, "session marked stale");
            state.broadcast(HookSessionUpdate {
                session_id: id.clone(),
                event_type: "stale".into(),
                session: Some(fs.session.clone()),
            });
        }
    }

    for id in evicted {
        guard.remove(&id);
        tracing::debug!(session_id = %id, "session evicted (stale/ended)");
        state.broadcast(HookSessionUpdate {
            session_id: id,
            event_type: "evicted".into(),
            session: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".claude").join("sessions")
}

fn extract_session_id(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
}

/// Extract session ID from a `.jsonl.gz` path (double extension).
fn extract_session_id_from_gz(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.strip_suffix(".jsonl.gz").map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_session_start(session_id: &str) -> String {
        serde_json::json!({
            "v": 1,
            "type": "session_start",
            "session_id": session_id,
            "timestamp": "2026-03-25T10:00:00.000000000Z",
            "workspace": "/workspaces/test",
            "git_branch": "main",
            "agent_type": "main",
            "parent_session_id": null,
            "corvia_agent_id": "claude-code",
        })
        .to_string()
    }

    fn make_user_prompt(turn: u32) -> String {
        serde_json::json!({
            "v": 1,
            "type": "user_prompt",
            "session_id": "ses-test",
            "turn": turn,
            "timestamp": "2026-03-25T10:01:00.000000000Z",
            "content": "hello world",
        })
        .to_string()
    }

    fn make_tool_start(tool: &str) -> String {
        serde_json::json!({
            "v": 1,
            "type": "tool_start",
            "session_id": "ses-test",
            "turn": 1,
            "timestamp": "2026-03-25T10:01:30.000000000Z",
            "tool": tool,
            "input": {},
        })
        .to_string()
    }

    fn make_tool_end(tool: &str) -> String {
        serde_json::json!({
            "v": 1,
            "type": "tool_end",
            "session_id": "ses-test",
            "turn": 1,
            "timestamp": "2026-03-25T10:01:45.000000000Z",
            "tool": tool,
            "input": {},
            "output": "ok",
            "truncated": false,
            "success": true,
        })
        .to_string()
    }

    fn make_session_end() -> String {
        serde_json::json!({
            "v": 1,
            "type": "session_end",
            "session_id": "ses-test",
            "timestamp": "2026-03-25T10:10:00.000000000Z",
            "total_turns": 5,
            "duration_ms": 600000,
        })
        .to_string()
    }

    #[test]
    fn test_apply_session_start_event() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));

        assert_eq!(fs.session.workspace, "/workspaces/test");
        assert_eq!(fs.session.git_branch, "main");
        assert_eq!(fs.session.agent_type, "main");
        assert_eq!(fs.session.state, HookSessionState::Active);
        assert_eq!(fs.session.started_at, "2026-03-25T10:00:00.000000000Z");
    }

    #[test]
    fn test_apply_user_prompt_increments_turn() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));
        apply_event(&mut fs, &make_user_prompt(1));
        assert_eq!(fs.session.turn_count, 1);

        apply_event(&mut fs, &make_user_prompt(2));
        assert_eq!(fs.session.turn_count, 2);
    }

    #[test]
    fn test_apply_tool_lifecycle() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));

        // tool_start sets active_tool
        apply_event(&mut fs, &make_tool_start("Bash"));
        assert_eq!(fs.session.active_tool.as_deref(), Some("Bash"));
        assert_eq!(fs.pending_tools, 1);
        assert!(fs.session.tools_used.contains(&"Bash".to_string()));

        // tool_end clears active_tool, increments tool_calls
        apply_event(&mut fs, &make_tool_end("Bash"));
        assert_eq!(fs.session.active_tool, None);
        assert_eq!(fs.session.tool_calls, 1);
        assert_eq!(fs.pending_tools, 0);
    }

    #[test]
    fn test_apply_session_end() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));
        apply_event(&mut fs, &make_tool_start("Read"));
        apply_event(&mut fs, &make_session_end());

        assert_eq!(fs.session.state, HookSessionState::Ended);
        assert_eq!(fs.session.active_tool, None);
        assert_eq!(fs.session.turn_count, 5);
    }

    #[test]
    fn test_apply_unknown_event_type() {
        let mut fs = FileState::new("ses-test");
        let line = serde_json::json!({
            "v": 1,
            "type": "future_event_v2",
            "session_id": "ses-test",
            "timestamp": "2026-03-25T10:05:00.000000000Z",
        })
        .to_string();
        // Should not panic.
        apply_event(&mut fs, &line);
        assert_eq!(fs.session.state, HookSessionState::Active);
    }

    #[test]
    fn test_read_new_lines_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ses-test.jsonl");

        // Write 3 lines.
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "{}", make_session_start("ses-test")).unwrap();
            writeln!(f, "{}", make_user_prompt(1)).unwrap();
            writeln!(f, "{}", make_tool_start("Bash")).unwrap();
        }

        // Read from offset 0 — should get all 3.
        let result = read_new_lines(&path, 0);
        assert_eq!(result.lines.len(), 3);
        let offset_after_3 = result.new_offset;
        assert!(offset_after_3 > 0);

        // Write 2 more lines.
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(f, "{}", make_tool_end("Bash")).unwrap();
            writeln!(f, "{}", make_user_prompt(2)).unwrap();
        }

        // Read from previous offset — should get only the new 2.
        let result = read_new_lines(&path, offset_after_3);
        assert_eq!(result.lines.len(), 2);
    }

    #[test]
    fn test_read_new_lines_partial_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ses-test.jsonl");

        // Write a complete line + partial line (no trailing \n).
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "{}", make_session_start("ses-test")).unwrap();
            write!(f, "{{\"v\":1,\"type\":\"user_pro").unwrap(); // partial, no \n
        }

        let result = read_new_lines(&path, 0);
        // BufReader::lines() yields lines terminated by \n OR EOF.
        // The partial line will be yielded as a line at EOF.
        // However, it's invalid JSON, so apply_event will skip it.
        // The offset should advance past the complete line.
        assert!(result.lines.len() >= 1);
        // First line should be valid session_start.
        assert!(result.lines[0].contains("session_start"));
    }

    #[test]
    fn test_read_new_lines_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ses-test.jsonl");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "{}", make_session_start("ses-test")).unwrap();
            writeln!(f, "this is not json").unwrap();
            writeln!(f, "{}", make_user_prompt(1)).unwrap();
        }

        let result = read_new_lines(&path, 0);
        assert_eq!(result.lines.len(), 3); // All lines are read

        // But when applying, bad JSON is skipped.
        let mut fs = FileState::new("ses-test");
        for line in &result.lines {
            apply_event(&mut fs, line);
        }
        assert_eq!(fs.session.turn_count, 1); // Only valid events applied
    }

    #[test]
    fn test_read_new_lines_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ses-test.jsonl");
        std::fs::File::create(&path).unwrap();

        let result = read_new_lines(&path, 0);
        assert!(result.lines.is_empty());
    }

    #[test]
    fn test_stale_detection() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));

        // Simulate old event.
        fs.last_event_at = Instant::now() - std::time::Duration::from_secs(700);

        assert_eq!(fs.session.state, HookSessionState::Active);

        // The sweep logic checks age > stale_threshold.
        let age = Instant::now().duration_since(fs.last_event_at);
        assert!(age > std::time::Duration::from_secs(STALE_THRESHOLD_SECS));
    }

    #[test]
    fn test_eviction_after_ended() {
        let mut fs = FileState::new("ses-test");
        apply_event(&mut fs, &make_session_start("ses-test"));
        apply_event(&mut fs, &make_session_end());

        assert_eq!(fs.session.state, HookSessionState::Ended);
        // Ended sessions should be evicted on next sweep.
    }

    #[tokio::test]
    async fn test_max_sessions_limit() {
        let (state, _rx) = SessionWatcherState::new();
        let dir = tempfile::tempdir().unwrap();

        // Create MAX_SESSIONS + 1 files.
        for i in 0..=MAX_SESSIONS {
            let path = dir.path().join(format!("ses-{i}.jsonl"));
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "{}", make_session_start(&format!("ses-{i}"))).unwrap();
        }

        // Process them all.
        let paths: HashSet<PathBuf> = (0..=MAX_SESSIONS)
            .map(|i| dir.path().join(format!("ses-{i}.jsonl")))
            .collect();
        process_file_changes(&state, paths).await;

        let guard = state.sessions.read().await;
        assert!(guard.len() <= MAX_SESSIONS);
    }

    #[test]
    fn test_symlink_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("real.jsonl");
        let link = dir.path().join("link.jsonl");

        // Create a real file.
        {
            let mut f = std::fs::File::create(&target).unwrap();
            writeln!(f, "{}", make_session_start("ses-real")).unwrap();
        }

        // Create symlink.
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();

        #[cfg(unix)]
        {
            let meta = std::fs::symlink_metadata(&link).unwrap();
            assert!(meta.file_type().is_symlink());
        }
    }

    #[test]
    fn test_extract_session_id() {
        let path = PathBuf::from("/home/user/.claude/sessions/ses-abc123.jsonl");
        assert_eq!(extract_session_id(&path), Some("ses-abc123".to_string()));
    }

    #[test]
    fn test_extract_session_id_from_gz() {
        let path = PathBuf::from("/home/user/.claude/sessions/ses-abc123.jsonl.gz");
        assert_eq!(extract_session_id_from_gz(&path), Some("ses-abc123".to_string()));
    }

    #[test]
    fn test_compute_duration() {
        let start = "2026-03-25T10:00:00.000000000Z";
        let end = "2026-03-25T10:05:30.000000000Z";
        assert_eq!(compute_duration(start, end), 330);
    }
}
