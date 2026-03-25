//! Claude Code lifecycle hook handlers.
//!
//! Replaces bash scripts in `.claude/hooks/` with typed Rust handlers.
//! Entry point: `corvia hooks run --event <type> [--handler <name>]`
//!
//! Each invocation reads JSON from stdin (Claude Code hook protocol),
//! dispatches to the appropriate handler, and exits with:
//! - 0: success (action proceeds)
//! - 2: block (action prevented, stderr = reason)

pub mod agent_check;
pub mod cleanup;
pub mod doc_placement;
#[allow(dead_code)]
pub mod legacy;
pub mod reminders;
pub mod session;
pub mod settings;
pub mod status;

use anyhow::Result;
use corvia_common::config::{CorviaConfig, HooksConfig};
use std::io::Read;
use std::str::FromStr;

/// Check if debug mode is enabled via CORVIA_HOOKS_DEBUG=1 env var.
fn debug_enabled() -> bool {
    std::env::var("CORVIA_HOOKS_DEBUG").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Print a debug message to stderr if CORVIA_HOOKS_DEBUG=1.
macro_rules! hook_debug {
    ($($arg:tt)*) => {
        if debug_enabled() {
            eprintln!("[hooks debug] {}", format!($($arg)*));
        }
    };
}

/// Claude Code hook event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    SessionEnd,
}

impl FromStr for HookEvent {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "SessionStart" => Ok(Self::SessionStart),
            "UserPromptSubmit" => Ok(Self::UserPromptSubmit),
            "PreToolUse" => Ok(Self::PreToolUse),
            "PostToolUse" => Ok(Self::PostToolUse),
            "SessionEnd" => Ok(Self::SessionEnd),
            _ => anyhow::bail!("Unknown hook event: {s}"),
        }
    }
}

/// Specific handler to run (when --handler is specified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookHandler {
    DocPlacement,
    CorviaFirstReminder,
    WriteReminder,
}

impl FromStr for HookHandler {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "doc-placement" => Ok(Self::DocPlacement),
            "corvia-first-reminder" => Ok(Self::CorviaFirstReminder),
            "write-reminder" => Ok(Self::WriteReminder),
            _ => anyhow::bail!("Unknown hook handler: {s}"),
        }
    }
}

/// Main entry point for `corvia hooks run --event <type> [--handler <name>]`.
///
/// This function runs synchronously (no tokio, no telemetry init) for minimal
/// cold-start overhead (~14ms).
pub fn run_hook_from_args(event_str: &str, handler_str: Option<&str>) -> Result<()> {
    let start = std::time::Instant::now();
    let event: HookEvent = event_str.parse()?;
    let handler: Option<HookHandler> = handler_str.map(|s| s.parse()).transpose()?;

    hook_debug!("event={event_str} handler={:?}", handler_str);

    // Read stdin JSON (Claude Code hook protocol), limited to 10MB
    let mut input = String::new();
    std::io::stdin().take(10 * 1024 * 1024).read_to_string(&mut input)?;
    let stdin: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            if !input.trim().is_empty() {
                eprintln!("Warning: hook received invalid JSON on stdin: {e}");
            }
            serde_json::Value::default()
        }
    };

    // Extract session_id from stdin (NOT from shared file)
    let session_id = stdin.get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Load config (best-effort — use defaults if corvia.toml missing)
    let hooks_config = load_hooks_config();

    // Master switch — if hooks are disabled globally, exit immediately
    if !hooks_config.enabled {
        hook_debug!("master switch disabled, skipping all hooks");
        return Ok(());
    }

    match handler {
        // Handler-specific invocations (matcher-filtered by Claude Code)
        Some(HookHandler::DocPlacement) => {
            if hooks_config.doc_placement {
                hook_debug!("doc-placement: checking");
                match doc_placement::check(&stdin)? {
                    doc_placement::HookResult::Allow => {
                        hook_debug!("doc-placement: allowed");
                    }
                    doc_placement::HookResult::Block(msg) => {
                        hook_debug!("doc-placement: BLOCKED");
                        eprintln!("{msg}");
                        std::process::exit(2);
                    }
                }
            } else {
                hook_debug!("doc-placement: skipped (disabled in config)");
            }
        }
        Some(HookHandler::CorviaFirstReminder) => {
            if hooks_config.corvia_first_reminder {
                println!("{}", reminders::corvia_first_reminder());
            } else {
                hook_debug!("corvia-first-reminder: skipped (disabled in config)");
            }
        }
        Some(HookHandler::WriteReminder) => {
            if hooks_config.write_reminder {
                if let Some(msg) = reminders::write_reminder(&stdin) {
                    println!("{msg}");
                }
            } else {
                hook_debug!("write-reminder: skipped (disabled in config)");
            }
        }
        // No handler specified: run session recording + event-specific handlers
        None => {
            // Session recording (all events) — errors are non-fatal
            if hooks_config.session_recording && !session_id.is_empty() {
                hook_debug!("session-record: writing event");
                if let Err(e) = session::record_event(&session_id, &event, &stdin) {
                    eprintln!("Warning: failed to record session event: {e}");
                }
            }

            // Event-specific built-in handlers
            match event {
                HookEvent::SessionStart => {
                    if hooks_config.agent_check {
                        hook_debug!("agent-check: registering");
                        let msg = agent_check::agent_check();
                        println!("{msg}");
                    }
                }
                HookEvent::SessionEnd => {
                    if hooks_config.orphan_cleanup {
                        hook_debug!("orphan-cleanup: scanning");
                        cleanup::orphan_cleanup(true);
                    }
                    // Finalize runs even if record_event failed — gzip + ingest what we have
                    if hooks_config.session_recording && !session_id.is_empty() {
                        hook_debug!("session-finalize: gzip + ingest");
                        session::finalize_session(&session_id);
                    }
                }
                _ => {}
            }
        }
    }

    hook_debug!("completed in {}ms", start.elapsed().as_millis());
    Ok(())
}

/// Load hooks config from corvia.toml, falling back to all-enabled defaults.
fn load_hooks_config() -> HooksConfig {
    let config_path = CorviaConfig::config_path();
    if config_path.exists() {
        match CorviaConfig::load(&config_path) {
            Ok(config) => return config.hooks.unwrap_or_default(),
            Err(e) => eprintln!("Warning: failed to load corvia.toml hooks config: {e} (using defaults)"),
        }
    }
    HooksConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hook_event() {
        assert_eq!("SessionStart".parse::<HookEvent>().unwrap(), HookEvent::SessionStart);
        assert_eq!("PreToolUse".parse::<HookEvent>().unwrap(), HookEvent::PreToolUse);
        assert_eq!("PostToolUse".parse::<HookEvent>().unwrap(), HookEvent::PostToolUse);
        assert_eq!("UserPromptSubmit".parse::<HookEvent>().unwrap(), HookEvent::UserPromptSubmit);
        assert_eq!("SessionEnd".parse::<HookEvent>().unwrap(), HookEvent::SessionEnd);
        assert!("Invalid".parse::<HookEvent>().is_err());
    }

    #[test]
    fn test_parse_hook_handler() {
        assert_eq!("doc-placement".parse::<HookHandler>().unwrap(), HookHandler::DocPlacement);
        assert_eq!("write-reminder".parse::<HookHandler>().unwrap(), HookHandler::WriteReminder);
        assert_eq!("corvia-first-reminder".parse::<HookHandler>().unwrap(), HookHandler::CorviaFirstReminder);
        assert!("invalid".parse::<HookHandler>().is_err());
    }

    #[test]
    fn test_default_hooks_config_all_enabled() {
        let config = HooksConfig::default();
        assert!(config.enabled);
        assert!(config.session_recording);
        assert!(config.doc_placement);
        assert!(config.agent_check);
        assert!(config.write_reminder);
        assert!(config.orphan_cleanup);
        assert!(config.corvia_first_reminder);
    }

    #[test]
    fn test_master_switch_disables_all() {
        let mut config = HooksConfig::default();
        config.enabled = false;
        assert!(!config.is_active("session_recording"));
        assert!(!config.is_active("doc_placement"));
        assert!(!config.is_active("agent_check"));
    }

    #[test]
    fn test_individual_hook_disable() {
        let mut config = HooksConfig::default();
        config.doc_placement = false;
        assert!(config.is_active("session_recording"));
        assert!(!config.is_active("doc_placement"));
    }
}
