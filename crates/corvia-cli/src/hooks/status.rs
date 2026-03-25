//! `corvia hooks status` — show which hooks are enabled and registered.

use anyhow::Result;
use corvia_common::config::{CorviaConfig, HooksConfig};

/// Available hook handlers with descriptions.
const HANDLERS: &[(&str, &str)] = &[
    ("session_recording", "Record session events to JSONL (WAL)"),
    ("doc_placement", "Block writes to disallowed doc paths (PreToolUse Write|Edit)"),
    ("agent_check", "Register agent identity on SessionStart"),
    ("write_reminder", "Remind to persist decisions after git commits (PostToolUse Bash)"),
    ("orphan_cleanup", "Kill orphaned processes on SessionEnd (WSL workaround)"),
    ("corvia_first_reminder", "Remind to call corvia_search before Grep/Glob"),
];

/// Print hook status: config + settings.json registration.
pub fn print_status() -> Result<()> {
    let config_path = CorviaConfig::config_path();
    let hooks_config = if config_path.exists() {
        match CorviaConfig::load(&config_path) {
            Ok(config) => {
                println!("Config: {}", config_path.display());
                config.hooks.unwrap_or_default()
            }
            Err(e) => {
                println!("Config: {} (parse error: {e}, using defaults)", config_path.display());
                HooksConfig::default()
            }
        }
    } else {
        println!("Config: not found (using defaults)");
        HooksConfig::default()
    };

    // Master switch
    if !hooks_config.enabled {
        println!("\nMaster switch: DISABLED (all hooks inactive)");
    } else {
        println!("\nMaster switch: enabled");
    }

    // Per-hook status
    println!("\nHooks:");
    for &(name, desc) in HANDLERS {
        let active = hooks_config.is_active(name);
        let status = if active { "ON " } else { "OFF" };
        println!("  [{status}] {name:<25} {desc}");
    }

    // Check settings.json registration
    let settings_path = std::env::current_dir()
        .unwrap_or_default()
        .join(".claude/settings.json");

    println!();
    if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path).unwrap_or_default();
        let corvia_hooks = content.matches("corvia hooks run").count();
        if corvia_hooks > 0 {
            println!("Settings: {} ({} hook entries registered)", settings_path.display(), corvia_hooks);
        } else {
            println!("Settings: {} (no corvia hooks registered — run 'corvia hooks init')", settings_path.display());
        }
    } else {
        println!("Settings: .claude/settings.json not found — run 'corvia hooks init'");
    }

    // Check for stale sessions
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let sessions_dir = std::path::PathBuf::from(&home).join(".claude/sessions");
    if sessions_dir.exists() {
        let jsonl_count = std::fs::read_dir(&sessions_dir)
            .map(|entries| {
                entries.flatten()
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
                    .count()
            })
            .unwrap_or(0);
        let gz_count = std::fs::read_dir(&sessions_dir)
            .map(|entries| {
                entries.flatten()
                    .filter(|e| {
                        e.path().to_string_lossy().ends_with(".jsonl.gz")
                    })
                    .count()
            })
            .unwrap_or(0);

        if jsonl_count > 0 || gz_count > 0 {
            println!("\nSessions: {jsonl_count} active, {gz_count} pending ingest");
            if jsonl_count > 5 {
                println!("  Hint: {jsonl_count} stale sessions detected — run 'corvia hooks sweep'");
            }
        }
    }

    Ok(())
}
