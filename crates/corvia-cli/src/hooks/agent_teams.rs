//! Agent Teams capture hooks: incremental staging of team coordination artifacts.
//!
//! Captures task events, teammate state, and team config to
//! `~/.corvia/staging/agent-teams/{team-name}/` before Claude Code's team cleanup
//! deletes the ephemeral files under `~/.claude/`.
//!
//! Security baseline (RFC Section 3.0):
//! - Name validation: `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`
//! - Directory permissions: 0700 via umask + mkdir
//! - Atomic writes: temp-then-rename for single files, flock for JSONL appends
//! - Truncation: 64 KB max for task descriptions
//!
//! Staging format: see `docs/rfcs/2026-03-28-agent-teams-adapter-design.md` Section 8.

use anyhow::Result;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

/// Schema version for staging events. Increment when the JSONL format changes.
const SCHEMA_VERSION: u32 = 1;

/// Max event line size (64 KB). Lines exceeding this trigger fallback serialization.
const MAX_EVENT_LINE_BYTES: usize = 64 * 1024;

/// Directory layout for Agent Teams hook operations.
/// Production code uses `Dirs::from_home()`; tests inject custom paths.
struct Dirs {
    staging_root: PathBuf,
    claude_tasks: PathBuf,
    claude_teams: PathBuf,
}

impl Dirs {
    fn from_home() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        let home = PathBuf::from(home);
        Self {
            staging_root: home.join(".corvia/staging/agent-teams"),
            claude_tasks: home.join(".claude/tasks"),
            claude_teams: home.join(".claude/teams"),
        }
    }

    fn ensure_staging_dir(&self, team_name: &str) -> Result<PathBuf> {
        let dir = self.staging_root.join(team_name);
        // Set restrictive umask before creating to avoid TOCTOU permission gap
        let old_umask = unsafe { libc::umask(0o077) };
        let result = fs::create_dir_all(&dir);
        unsafe { libc::umask(old_umask) };
        result?;
        // Ensure 0700 even if directory pre-existed with different permissions
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        Ok(dir)
    }
}

/// Validate team/teammate names against `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`.
#[must_use]
fn is_valid_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    let bytes = name.as_bytes();
    if !bytes[0].is_ascii_alphanumeric() {
        return false;
    }
    bytes[1..].iter().all(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
}

/// Append a JSON line to a file using flock for serialization.
///
/// The lock file (`{path}.lock`) serializes concurrent appends from multiple hook
/// processes. `lock_file` is declared before `file` so it is dropped (unlocked) AFTER
/// `file` is flushed and closed (Rust drops in reverse declaration order).
fn flock_append(path: &Path, line: &str) -> Result<()> {
    let lock_path = path.with_file_name(format!(
        "{}.lock",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));

    // lock_file must be declared before data file for correct drop ordering
    let lock_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)?;

    let fd = lock_file.as_raw_fd();
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if ret != 0 {
        anyhow::bail!("flock failed: {}", std::io::Error::last_os_error());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(line.as_bytes())?;
    if !line.ends_with('\n') {
        file.write_all(b"\n")?;
    }
    file.flush()?;
    // file is dropped here (closed), then lock_file is dropped (flock released)

    Ok(())
}

/// Atomic file copy: write to temp (with PID suffix), then rename.
///
/// Caller must ensure source and target are on the same filesystem for atomicity.
/// On failure, best-effort cleanup of the temp file is attempted.
fn atomic_copy(source: &Path, target: &Path) -> Result<()> {
    let tmp = target.with_file_name(format!(
        "{}.{}.tmp",
        target.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    if let Err(e) = fs::copy(source, &tmp).and_then(|_| fs::rename(&tmp, target)) {
        let _ = fs::remove_file(&tmp); // best-effort cleanup
        return Err(e.into());
    }
    Ok(())
}

/// Truncate a string to at most `max_bytes` bytes (on a char boundary).
fn truncate_to_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Get current ISO 8601 timestamp with millisecond precision.
fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// Check if source file is newer than target (or target doesn't exist).
/// Returns false if source metadata cannot be read (file may have been deleted).
fn should_copy(source: &Path, target: &Path) -> bool {
    let src_mtime = match source.metadata().and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false, // Can't read source, don't attempt copy
    };
    let dst_mtime = match target.metadata().and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true, // Target doesn't exist, should copy
    };
    src_mtime > dst_mtime
}

/// Infer team name by scanning tasks directory for the directory containing `task_id`.
///
/// If multiple teams have the same task_id (rare, since Claude Code limits to one team
/// per session), the most recently modified directory wins.
fn infer_team_from_task(tasks_dir: &Path, task_id: &str) -> Option<String> {
    if !tasks_dir.exists() {
        return None;
    }
    let entries = fs::read_dir(tasks_dir).ok()?;
    let mut candidates: Vec<(String, std::time::SystemTime)> = Vec::new();

    for entry in entries.flatten() {
        let team_name = entry.file_name().to_string_lossy().to_string();
        // Skip entries with unreadable metadata instead of aborting the whole scan
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if !ft.is_dir() {
            continue;
        }
        let task_file = entry.path().join(format!("{task_id}.json"));
        if task_file.exists() {
            let mtime = match entry.metadata().and_then(|m| m.modified()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            candidates.push((team_name, mtime));
        }
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.into_iter().next().map(|(name, _)| name)
}

// ─── Inner handlers (accept Dirs for testability) ──────────────────────────

fn handle_task_created_inner(dirs: &Dirs, stdin: &serde_json::Value) -> Result<()> {
    let task_id = stdin.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
    if task_id.is_empty() {
        eprintln!("Warning: TaskCreated hook received empty task_id");
        return Ok(());
    }

    let subject = stdin.get("task_subject").and_then(|v| v.as_str()).unwrap_or("");
    let description_raw = stdin.get("task_description").and_then(|v| v.as_str()).unwrap_or("");
    let description = truncate_to_bytes(description_raw, MAX_EVENT_LINE_BYTES);

    let team_name = match infer_team_from_task(&dirs.claude_tasks, task_id) {
        Some(name) => name,
        None => {
            eprintln!("Warning: TaskCreated could not infer team for task {task_id}");
            return Ok(());
        }
    };

    if !is_valid_name(&team_name) {
        eprintln!("Warning: TaskCreated rejected invalid team name: {team_name}");
        return Ok(());
    }

    let staging_dir = dirs.ensure_staging_dir(&team_name)?;
    let jsonl_path = staging_dir.join("tasks.jsonl");

    let event = serde_json::json!({
        "v": SCHEMA_VERSION,
        "event": "created",
        "task_id": task_id,
        "subject": subject,
        "description": description,
        "timestamp": now_iso(),
    });

    flock_append(&jsonl_path, &serde_json::to_string(&event)?)
}

fn handle_task_completed_inner(dirs: &Dirs, stdin: &serde_json::Value) -> Result<()> {
    let task_id = stdin.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
    if task_id.is_empty() {
        eprintln!("Warning: TaskCompleted hook received empty task_id");
        return Ok(());
    }

    let subject = stdin.get("task_subject").and_then(|v| v.as_str()).unwrap_or("");

    let team_name = match infer_team_from_task(&dirs.claude_tasks, task_id) {
        Some(name) => name,
        None => {
            eprintln!("Warning: TaskCompleted could not infer team for task {task_id}");
            return Ok(());
        }
    };

    if !is_valid_name(&team_name) {
        eprintln!("Warning: TaskCompleted rejected invalid team name: {team_name}");
        return Ok(());
    }

    let task_file = dirs.claude_tasks.join(&team_name).join(format!("{task_id}.json"));
    let full_task: serde_json::Value = if task_file.exists() {
        let content = fs::read_to_string(&task_file).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };

    let owner = full_task.get("owner").and_then(|v| v.as_str()).unwrap_or("");

    let staging_dir = dirs.ensure_staging_dir(&team_name)?;
    let jsonl_path = staging_dir.join("tasks.jsonl");

    let event = serde_json::json!({
        "v": SCHEMA_VERSION,
        "event": "completed",
        "task_id": task_id,
        "subject": subject,
        "owner": owner,
        "timestamp": now_iso(),
        "full_task": full_task,
    });

    let mut line = serde_json::to_string(&event)?;
    if line.len() > MAX_EVENT_LINE_BYTES {
        let fallback = serde_json::json!({
            "v": SCHEMA_VERSION,
            "event": "completed",
            "task_id": task_id,
            "subject": subject,
            "owner": owner,
            "timestamp": now_iso(),
            "full_task_truncated": true,
        });
        line = serde_json::to_string(&fallback)?;
    }

    flock_append(&jsonl_path, &line)
}

fn handle_teammate_idle_inner(dirs: &Dirs, stdin: &serde_json::Value) -> Result<()> {
    let team_name = stdin.get("team_name").and_then(|v| v.as_str()).unwrap_or("");
    let teammate_name = stdin.get("teammate_name").and_then(|v| v.as_str()).unwrap_or("");

    if team_name.is_empty() || teammate_name.is_empty() {
        eprintln!("Warning: TeammateIdle missing team_name or teammate_name");
        return Ok(());
    }
    if !is_valid_name(team_name) {
        eprintln!("Warning: TeammateIdle rejected invalid team name: {team_name}");
        return Ok(());
    }
    if !is_valid_name(teammate_name) {
        eprintln!("Warning: TeammateIdle rejected invalid teammate name: {teammate_name}");
        return Ok(());
    }

    let staging_dir = dirs.ensure_staging_dir(team_name)?;
    let team_dir = dirs.claude_teams.join(team_name);

    // 1. Copy config.json (if source newer than staging)
    let src_config = team_dir.join("config.json");
    let dst_config = staging_dir.join("config.json");
    if src_config.exists() && should_copy(&src_config, &dst_config) {
        if let Err(e) = atomic_copy(&src_config, &dst_config) {
            eprintln!("Warning: TeammateIdle failed to copy config.json: {e}");
        }
    }

    // 2. Copy teammate's inbox
    let inboxes_staging = staging_dir.join("inboxes");
    fs::create_dir_all(&inboxes_staging)?;

    let src_inbox = team_dir.join("inboxes").join(format!("{teammate_name}.json"));
    let dst_inbox = inboxes_staging.join(format!("{teammate_name}.json"));
    if src_inbox.exists() && should_copy(&src_inbox, &dst_inbox) {
        if let Err(e) = atomic_copy(&src_inbox, &dst_inbox) {
            eprintln!("Warning: TeammateIdle failed to copy inbox for {teammate_name}: {e}");
        }
    }

    // 3. Copy lead's inbox (last writer wins via atomic rename)
    let src_lead = team_dir.join("inboxes").join("team-lead.json");
    let dst_lead = inboxes_staging.join("team-lead.json");
    if src_lead.exists() && should_copy(&src_lead, &dst_lead) {
        if let Err(e) = atomic_copy(&src_lead, &dst_lead) {
            eprintln!("Warning: TeammateIdle failed to copy lead inbox: {e}");
        }
    }

    // 4. Append to .capture-log
    let log_path = staging_dir.join(".capture-log");
    let log_entry = serde_json::json!({
        "v": SCHEMA_VERSION,
        "event": "TeammateIdle",
        "teammate": teammate_name,
        "team": team_name,
        "timestamp": now_iso(),
    });
    if let Err(e) = flock_append(&log_path, &serde_json::to_string(&log_entry)?) {
        eprintln!("Warning: TeammateIdle failed to append capture-log: {e}");
    }

    Ok(())
}

fn team_sweep_inner(dirs: &Dirs, session_id: &str) -> Result<()> {
    if !dirs.claude_teams.exists() {
        return Ok(());
    }

    let entries = fs::read_dir(&dirs.claude_teams)?;
    for entry in entries.flatten() {
        let team_name = entry.file_name().to_string_lossy().to_string();
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if !ft.is_dir() {
            continue;
        }
        if !is_valid_name(&team_name) {
            continue;
        }

        // Scope to teams matching this session's leadSessionId
        let config_path = entry.path().join("config.json");
        if config_path.exists() {
            let config_content = fs::read_to_string(&config_path).unwrap_or_default();
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                let lead_session = config.get("leadSessionId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !lead_session.is_empty() && lead_session != session_id {
                    continue;
                }
            }
        }

        let staging_dir = match dirs.ensure_staging_dir(&team_name) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: team sweep failed to create staging for {team_name}: {e}");
                continue;
            }
        };

        // Copy config.json
        let src_config = entry.path().join("config.json");
        if src_config.exists() {
            if let Err(e) = atomic_copy(&src_config, &staging_dir.join("config.json")) {
                eprintln!("Warning: team sweep failed to copy config.json for {team_name}: {e}");
            }
        }

        // Copy all task files as sweep events
        let tasks_dir = dirs.claude_tasks.join(&team_name);
        if tasks_dir.exists() {
            if let Ok(task_entries) = fs::read_dir(&tasks_dir) {
                let jsonl_path = staging_dir.join("tasks.jsonl");
                for task_entry in task_entries.flatten() {
                    let fname = task_entry.file_name().to_string_lossy().to_string();
                    if fname.ends_with(".json") {
                        let content = fs::read_to_string(task_entry.path()).unwrap_or_default();
                        if let Ok(task) = serde_json::from_str::<serde_json::Value>(&content) {
                            let task_id = fname.trim_end_matches(".json");
                            let event = serde_json::json!({
                                "v": SCHEMA_VERSION,
                                "event": "sweep",
                                "task_id": task_id,
                                "timestamp": now_iso(),
                                "full_task": task,
                            });
                            let line = serde_json::to_string(&event).unwrap_or_default();
                            if let Err(e) = flock_append(&jsonl_path, &line) {
                                eprintln!("Warning: team sweep failed to write task {task_id} for {team_name}: {e}");
                            }
                        }
                    }
                }
            }
        }

        // Copy all inbox files
        let inboxes_dir = entry.path().join("inboxes");
        if inboxes_dir.exists() {
            let staging_inboxes = staging_dir.join("inboxes");
            let _ = fs::create_dir_all(&staging_inboxes);
            if let Ok(inbox_entries) = fs::read_dir(&inboxes_dir) {
                for inbox_entry in inbox_entries.flatten() {
                    let fname = inbox_entry.file_name();
                    let dst = staging_inboxes.join(&fname);
                    if let Err(e) = atomic_copy(&inbox_entry.path(), &dst) {
                        eprintln!("Warning: team sweep failed to copy inbox {:?} for {team_name}: {e}", fname);
                    }
                }
            }
        }
    }

    Ok(())
}

// ─── Public API (uses default Dirs) ────────────────────────────────────────

/// Handle TaskCreated hook event (RFC Section 3.1).
pub fn handle_task_created(stdin: &serde_json::Value) -> Result<()> {
    handle_task_created_inner(&Dirs::from_home(), stdin)
}

/// Handle TaskCompleted hook event (RFC Section 3.2).
pub fn handle_task_completed(stdin: &serde_json::Value) -> Result<()> {
    handle_task_completed_inner(&Dirs::from_home(), stdin)
}

/// Handle TeammateIdle hook event (RFC Section 3.3).
pub fn handle_teammate_idle(stdin: &serde_json::Value) -> Result<()> {
    handle_teammate_idle_inner(&Dirs::from_home(), stdin)
}

/// Final sweep at SessionEnd (RFC Section 3.4). Errors are non-fatal.
pub fn team_sweep(session_id: &str) {
    if let Err(e) = team_sweep_inner(&Dirs::from_home(), session_id) {
        eprintln!("Warning: Agent Teams sweep failed (non-fatal): {e}");
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
// Test IDs (T1-T3) reference the test plan in
// docs/rfcs/2026-03-28-agent-teams-adapter-design.md Section 7.

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    /// Create isolated Dirs backed by a temp directory.
    fn test_dirs(base: &Path) -> Dirs {
        Dirs {
            staging_root: base.join("staging/agent-teams"),
            claude_tasks: base.join("claude/tasks"),
            claude_teams: base.join("claude/teams"),
        }
    }

    /// Set up a mock task file so infer_team_from_task works.
    fn setup_task(dirs: &Dirs, team: &str, task_id: &str, task_json: &serde_json::Value) {
        let task_dir = dirs.claude_tasks.join(team);
        fs::create_dir_all(&task_dir).unwrap();
        let path = task_dir.join(format!("{task_id}.json"));
        fs::write(&path, serde_json::to_string(task_json).unwrap()).unwrap();
    }

    /// Set up mock team files (config + inboxes).
    fn setup_team(dirs: &Dirs, team: &str, config: &serde_json::Value, inboxes: &[(&str, &str)]) {
        let team_dir = dirs.claude_teams.join(team);
        let inbox_dir = team_dir.join("inboxes");
        fs::create_dir_all(&inbox_dir).unwrap();
        fs::write(team_dir.join("config.json"), serde_json::to_string(config).unwrap()).unwrap();
        for (name, content) in inboxes {
            fs::write(inbox_dir.join(format!("{name}.json")), content).unwrap();
        }
    }

    // ─── T2: Name validation ───────────────────────────────────────────────

    #[test]
    fn test_valid_names() {
        assert!(is_valid_name("my-team"));
        assert!(is_valid_name("team_1"));
        assert!(is_valid_name("a"));
        assert!(is_valid_name("A123-test_name"));
        assert!(is_valid_name(&"x".repeat(64)));
    }

    #[test]
    fn test_invalid_names_path_traversal() {
        assert!(!is_valid_name("../etc"));
        assert!(!is_valid_name("../../etc/cron.d/malicious"));
        assert!(!is_valid_name("foo/../bar"));
    }

    #[test]
    fn test_invalid_names_special_chars() {
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("-starts-with-dash"));
        assert!(!is_valid_name("_starts-with-underscore"));
        assert!(!is_valid_name("has space"));
        assert!(!is_valid_name("has/slash"));
        assert!(!is_valid_name("has.dot"));
        assert!(!is_valid_name(&"x".repeat(65)));
    }

    // ─── T1: Hook scripts write correct staging files ──────────────────────

    #[test]
    fn t1_task_created_writes_valid_jsonl() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        setup_task(&dirs, "my-team", "1", &serde_json::json!({"status": "open"}));

        let stdin = serde_json::json!({
            "session_id": "sess-123",
            "hook_event_name": "TaskCreated",
            "task_id": "1",
            "task_subject": "Review auth module",
            "task_description": "Check the auth module for security issues"
        });

        handle_task_created_inner(&dirs, &stdin).unwrap();

        let jsonl_path = dirs.staging_root.join("my-team/tasks.jsonl");
        assert!(jsonl_path.exists());

        let content = fs::read_to_string(&jsonl_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["v"], SCHEMA_VERSION);
        assert_eq!(parsed["event"], "created");
        assert_eq!(parsed["task_id"], "1");
        assert_eq!(parsed["subject"], "Review auth module");
        assert_eq!(parsed["description"], "Check the auth module for security issues");
        assert!(parsed["timestamp"].as_str().unwrap().ends_with('Z'));
    }

    #[test]
    fn t1_task_completed_writes_jsonl_with_full_task() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let task_data = serde_json::json!({
            "status": "completed",
            "owner": "researcher",
            "blocks": [],
            "blockedBy": ["2"]
        });
        setup_task(&dirs, "my-team", "1", &task_data);

        let stdin = serde_json::json!({
            "session_id": "sess-123",
            "hook_event_name": "TaskCompleted",
            "task_id": "1",
            "task_subject": "Review auth module"
        });

        handle_task_completed_inner(&dirs, &stdin).unwrap();

        let jsonl_path = dirs.staging_root.join("my-team/tasks.jsonl");
        let content = fs::read_to_string(&jsonl_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["v"], SCHEMA_VERSION);
        assert_eq!(parsed["event"], "completed");
        assert_eq!(parsed["task_id"], "1");
        assert_eq!(parsed["owner"], "researcher");
        assert_eq!(parsed["full_task"]["blockedBy"][0], "2");
    }

    #[test]
    fn t1_teammate_idle_copies_config_and_inboxes() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let config = serde_json::json!({"leadSessionId": "sess-123", "members": ["researcher"]});
        setup_team(&dirs, "my-team", &config, &[
            ("researcher", r#"[{"from":"lead","body":"hello"}]"#),
            ("team-lead", r#"[{"from":"researcher","body":"done"}]"#),
        ]);

        let stdin = serde_json::json!({
            "session_id": "sess-123",
            "hook_event_name": "TeammateIdle",
            "teammate_name": "researcher",
            "team_name": "my-team"
        });

        handle_teammate_idle_inner(&dirs, &stdin).unwrap();

        let staging = dirs.staging_root.join("my-team");
        assert!(staging.join("config.json").exists());
        let config_content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(staging.join("config.json")).unwrap()).unwrap();
        assert_eq!(config_content["leadSessionId"], "sess-123");

        assert!(staging.join("inboxes/researcher.json").exists());
        assert!(staging.join("inboxes/team-lead.json").exists());

        let log = fs::read_to_string(staging.join(".capture-log")).unwrap();
        let log_entry: serde_json::Value = serde_json::from_str(log.trim()).unwrap();
        assert_eq!(log_entry["v"], SCHEMA_VERSION);
        assert_eq!(log_entry["event"], "TeammateIdle");
        assert_eq!(log_entry["teammate"], "researcher");
    }

    // ─── T2: Rejects invalid names ─────────────────────────────────────────

    #[test]
    fn t2_task_created_rejects_invalid_team_name() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let bad_dir = dirs.claude_tasks.join("../etc");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(bad_dir.join("1.json"), "{}").unwrap();

        let stdin = serde_json::json!({
            "task_id": "1",
            "task_subject": "malicious",
            "task_description": "pwned"
        });

        handle_task_created_inner(&dirs, &stdin).unwrap();
        assert!(!dirs.staging_root.exists() || fs::read_dir(&dirs.staging_root).unwrap().count() == 0);
    }

    #[test]
    fn t2_teammate_idle_rejects_path_traversal_team() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let stdin = serde_json::json!({
            "team_name": "../etc",
            "teammate_name": "researcher"
        });

        handle_teammate_idle_inner(&dirs, &stdin).unwrap();
        assert!(!dirs.staging_root.join("../etc").exists());
    }

    #[test]
    fn t2_teammate_idle_rejects_path_traversal_teammate() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let config = serde_json::json!({"leadSessionId": "sess-123"});
        setup_team(&dirs, "my-team", &config, &[]);

        let stdin = serde_json::json!({
            "team_name": "my-team",
            "teammate_name": "../exploit"
        });

        handle_teammate_idle_inner(&dirs, &stdin).unwrap();
        // No inbox file should be created for invalid teammate name
        let inboxes = dirs.staging_root.join("my-team/inboxes");
        assert!(!inboxes.exists() || fs::read_dir(&inboxes).unwrap().count() == 0);
    }

    // ─── T3: Concurrent flock writes produce valid JSONL ───────────────────

    #[test]
    fn t3_concurrent_flock_writes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tasks.jsonl");

        let threads: Vec<_> = (0..10)
            .map(|thread_id| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..10 {
                        let line = serde_json::json!({
                            "thread": thread_id,
                            "seq": i,
                            "data": "x".repeat(100),
                        });
                        flock_append(&path, &serde_json::to_string(&line).unwrap()).unwrap();
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 100, "expected 100 lines, got {}", lines.len());

        for (i, line) in lines.iter().enumerate() {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(parsed.is_ok(), "line {i} is not valid JSON: {line}");
        }
    }

    #[test]
    fn t3_concurrent_flock_writes_large_payloads() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tasks.jsonl");

        // Lines > PIPE_BUF (4096 bytes) to verify flock serialization
        let threads: Vec<_> = (0..4)
            .map(|thread_id| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..5 {
                        let line = serde_json::json!({
                            "thread": thread_id,
                            "seq": i,
                            "data": "x".repeat(8000), // > PIPE_BUF
                        });
                        flock_append(&path, &serde_json::to_string(&line).unwrap()).unwrap();
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 20);
        for (i, line) in lines.iter().enumerate() {
            assert!(serde_json::from_str::<serde_json::Value>(line).is_ok(),
                "line {i} is not valid JSON (len={})", line.len());
        }
    }

    // ─── SessionEnd sweep ──────────────────────────────────────────────────

    #[test]
    fn test_team_sweep_copies_matching_team() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let config = serde_json::json!({"leadSessionId": "sess-abc"});
        setup_team(&dirs, "my-team", &config, &[
            ("researcher", r#"[{"msg":"hi"}]"#),
        ]);
        setup_task(&dirs, "my-team", "1", &serde_json::json!({"status": "done"}));

        team_sweep_inner(&dirs, "sess-abc").unwrap();

        let staging = dirs.staging_root.join("my-team");
        assert!(staging.join("config.json").exists());
        assert!(staging.join("tasks.jsonl").exists());
        assert!(staging.join("inboxes/researcher.json").exists());

        // Verify sweep events have schema version
        let content = fs::read_to_string(staging.join("tasks.jsonl")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["v"], SCHEMA_VERSION);
        assert_eq!(parsed["event"], "sweep");
    }

    #[test]
    fn test_team_sweep_skips_non_matching_team() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        // Set up matching team AND non-matching team
        let matching_config = serde_json::json!({"leadSessionId": "sess-abc"});
        setup_team(&dirs, "my-team", &matching_config, &[]);
        setup_task(&dirs, "my-team", "1", &serde_json::json!({"status": "done"}));

        let other_config = serde_json::json!({"leadSessionId": "sess-other"});
        setup_team(&dirs, "other-team", &other_config, &[]);

        team_sweep_inner(&dirs, "sess-abc").unwrap();

        // Matching team staged, non-matching team NOT staged
        assert!(dirs.staging_root.join("my-team/config.json").exists());
        assert!(!dirs.staging_root.join("other-team").exists());
    }

    #[test]
    fn test_team_sweep_no_teams_dir() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());
        team_sweep_inner(&dirs, "sess-abc").unwrap();
    }

    // ─── Missing file / edge case tests (from QA review) ──────────────────

    #[test]
    fn test_task_created_missing_task_file_is_noop() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());
        // task_id "99" has no file on disk
        let stdin = serde_json::json!({
            "task_id": "99",
            "task_subject": "ghost task",
            "task_description": "does not exist"
        });
        handle_task_created_inner(&dirs, &stdin).unwrap();
        assert!(!dirs.staging_root.exists());
    }

    #[test]
    fn test_task_completed_oversized_full_task_truncates() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        // Create a task file with a very large description field
        let big_task = serde_json::json!({
            "status": "completed",
            "owner": "researcher",
            "description": "x".repeat(MAX_EVENT_LINE_BYTES + 1000),
        });
        setup_task(&dirs, "my-team", "1", &big_task);

        let stdin = serde_json::json!({
            "task_id": "1",
            "task_subject": "big task"
        });

        handle_task_completed_inner(&dirs, &stdin).unwrap();

        let content = fs::read_to_string(dirs.staging_root.join("my-team/tasks.jsonl")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["full_task_truncated"], true);
        assert!(parsed.get("full_task").is_none());
    }

    #[test]
    fn test_teammate_idle_missing_team_dir_writes_log_only() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());
        // Team exists in name but no files on disk under claude_teams
        let stdin = serde_json::json!({
            "team_name": "ghost-team",
            "teammate_name": "researcher"
        });

        handle_teammate_idle_inner(&dirs, &stdin).unwrap();

        // Staging dir created, capture-log written, but no config/inbox copies
        let staging = dirs.staging_root.join("ghost-team");
        assert!(staging.join(".capture-log").exists());
        assert!(!staging.join("config.json").exists());
    }

    // ─── Utility tests ─────────────────────────────────────────────────────

    #[test]
    fn test_truncate_to_bytes() {
        assert_eq!(truncate_to_bytes("hello", 10), "hello");
        assert_eq!(truncate_to_bytes("hello", 3), "hel");
        assert_eq!(truncate_to_bytes("", 5), "");
        assert_eq!(truncate_to_bytes("café", 4), "caf");
        assert_eq!(truncate_to_bytes("café", 5), "café");
    }

    #[test]
    fn test_should_copy_target_missing() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src.json");
        fs::write(&src, "{}").unwrap();
        let dst = dir.path().join("dst.json");
        assert!(should_copy(&src, &dst));
    }

    #[test]
    fn test_should_copy_source_missing_returns_false() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("nonexistent.json");
        let dst = dir.path().join("dst.json");
        fs::write(&dst, "{}").unwrap();
        assert!(!should_copy(&src, &dst));
    }

    #[test]
    fn test_atomic_copy() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src.json");
        fs::write(&src, r#"{"key":"value"}"#).unwrap();
        let dst = dir.path().join("dst.json");
        atomic_copy(&src, &dst).unwrap();
        assert_eq!(fs::read_to_string(&dst).unwrap(), r#"{"key":"value"}"#);
    }

    #[test]
    fn test_staging_dir_permissions() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());
        let staging = dirs.ensure_staging_dir("test-team").unwrap();
        let perms = staging.metadata().unwrap().permissions().mode();
        assert_eq!(perms & 0o777, 0o700);
    }

    #[test]
    fn test_now_iso_format() {
        let ts = now_iso();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
    }

    #[test]
    fn test_infer_team_from_task() {
        let dir = TempDir::new().unwrap();
        let tasks_dir = dir.path().join("tasks");
        let team_dir = tasks_dir.join("my-team");
        fs::create_dir_all(&team_dir).unwrap();
        fs::write(team_dir.join("42.json"), "{}").unwrap();

        assert_eq!(infer_team_from_task(&tasks_dir, "42"), Some("my-team".into()));
        assert_eq!(infer_team_from_task(&tasks_dir, "999"), None);
    }

    #[test]
    fn test_empty_task_id_is_noop() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());
        let stdin = serde_json::json!({"task_id": ""});
        handle_task_created_inner(&dirs, &stdin).unwrap();
        assert!(!dirs.staging_root.exists());
    }

    #[test]
    fn test_description_truncation() {
        let dir = TempDir::new().unwrap();
        let dirs = test_dirs(dir.path());

        let big_desc = "x".repeat(MAX_EVENT_LINE_BYTES + 1000);
        setup_task(&dirs, "my-team", "1", &serde_json::json!({}));

        let stdin = serde_json::json!({
            "task_id": "1",
            "task_subject": "big task",
            "task_description": big_desc,
        });

        handle_task_created_inner(&dirs, &stdin).unwrap();

        let content = fs::read_to_string(dirs.staging_root.join("my-team/tasks.jsonl")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        let desc = parsed["description"].as_str().unwrap();
        assert!(desc.len() <= MAX_EVENT_LINE_BYTES);
    }
}
