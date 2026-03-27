//! Settings.json generator for Claude Code hook registration.
//!
//! `corvia hooks init` generates `.claude/settings.json` entries that point
//! to `corvia hooks run` instead of individual bash scripts.

use corvia_common::config::{CorviaConfig, HooksConfig};
use corvia_common::errors::Result;
use serde_json::json;
use std::path::Path;

/// Patterns that identify corvia-managed hook entries (old bash + new Rust).
const PURGE_PATTERNS: &[&str] = &[
    "bash .claude/hooks/record-",
    "bash .claude/hooks/agent-check",
    "bash .claude/hooks/corvia-write-reminder",
    "bash .corvia/hooks/doc-placement",
    "corvia hooks run",
    // Also catch the inline echo reminder
    "REMINDER: Have you called corvia_search",
    // Cleanup orphans bash script (now handled by Rust cleanup handler)
    "cleanup-orphans.sh",
];

/// Generate `.claude/settings.json` with Rust hook entries.
///
/// Purges old bash hook entries, preserves non-corvia hooks, and generates
/// new entries based on `[hooks]` config.
pub fn init_hooks(root: &Path, config: &CorviaConfig) -> Result<()> {
    let settings_dir = root.join(".claude");
    std::fs::create_dir_all(&settings_dir)
        .map_err(|e| corvia_common::errors::CorviaError::Config(format!("Failed to create .claude dir: {e}")))?;

    let settings_path = settings_dir.join("settings.json");
    let mut settings: serde_json::Value = if settings_path.exists() {
        let content = std::fs::read_to_string(&settings_path)
            .map_err(|e| corvia_common::errors::CorviaError::Config(format!("Failed to read settings: {e}")))?;
        serde_json::from_str(&content)
            .map_err(|e| corvia_common::errors::CorviaError::Config(format!("Failed to parse settings: {e}")))?
    } else {
        json!({})
    };

    let root_obj = settings.as_object_mut()
        .ok_or_else(|| corvia_common::errors::CorviaError::Config("settings.json root is not a JSON object".into()))?;

    let hooks = root_obj.entry("hooks").or_insert(json!({}));
    let hooks_obj = hooks.as_object_mut()
        .ok_or_else(|| corvia_common::errors::CorviaError::Config("settings.json 'hooks' is not a JSON object".into()))?;

    let hooks_config = config.hooks.clone().unwrap_or_default();

    // Purge old corvia-managed entries from all event arrays
    for (_event_name, entries) in hooks_obj.iter_mut() {
        if let Some(arr) = entries.as_array_mut() {
            arr.retain(|entry| !is_corvia_hook(entry));
        }
    }

    // Remove empty event arrays left after purge
    hooks_obj.retain(|_, v| !v.as_array().map(|a| a.is_empty()).unwrap_or(false));

    // Generate new entries (added in deterministic order)
    add_hooks(hooks_obj, &hooks_config);

    // Write back
    let content = serde_json::to_string_pretty(&settings)
        .map_err(|e| corvia_common::errors::CorviaError::Config(format!("Failed to serialize: {e}")))?;
    std::fs::write(&settings_path, format!("{content}\n"))
        .map_err(|e| corvia_common::errors::CorviaError::Config(format!("Failed to write: {e}")))?;

    println!("  Updated: {}", settings_path.display());

    // Also generate post-commit hooks for git repos
    if let Some(ws) = config.workspace.as_ref() {
        super::legacy::generate_post_commit_hooks(root, ws)?;
    }

    println!("Generated hook entries for corvia hooks run");
    Ok(())
}

/// Check if a hook entry is corvia-managed (should be purged on regeneration).
fn is_corvia_hook(entry: &serde_json::Value) -> bool {
    // Check the "hooks" array inside the entry
    if let Some(hooks_arr) = entry.get("hooks").and_then(|h| h.as_array()) {
        for hook in hooks_arr {
            if let Some(cmd) = hook.get("command").and_then(|c| c.as_str()) && PURGE_PATTERNS.iter().any(|p| cmd.contains(p)) {
                return true;
            }
        }
    }
    false
}

/// Add all hook entries based on config.
fn add_hooks(hooks_obj: &mut serde_json::Map<String, serde_json::Value>, config: &HooksConfig) {
    // SessionStart
    {
        let arr = ensure_array(hooks_obj, "SessionStart");
        arr.push(hook_entry(None, "corvia hooks run --event SessionStart", 5000));
    }

    // UserPromptSubmit
    if config.session_recording {
        let arr = ensure_array(hooks_obj, "UserPromptSubmit");
        arr.push(hook_entry(None, "corvia hooks run --event UserPromptSubmit", 5000));
    }

    // PreToolUse
    {
        let arr = ensure_array(hooks_obj, "PreToolUse");
        if config.session_recording {
            arr.push(hook_entry(None, "corvia hooks run --event PreToolUse", 5000));
        }
        if config.doc_placement {
            arr.push(hook_entry(Some("Write|Edit"), "corvia hooks run --event PreToolUse --handler doc-placement", 5000));
        }
        if config.corvia_first_reminder {
            arr.push(hook_entry(Some("Grep|Glob"), "corvia hooks run --event PreToolUse --handler corvia-first-reminder", 5000));
        }
    }

    // PostToolUse
    {
        let arr = ensure_array(hooks_obj, "PostToolUse");
        if config.session_recording {
            arr.push(hook_entry(None, "corvia hooks run --event PostToolUse", 5000));
        }
        if config.write_reminder {
            arr.push(hook_entry(Some("Bash"), "corvia hooks run --event PostToolUse --handler write-reminder", 5000));
        }
    }

    // SessionEnd
    {
        let arr = ensure_array(hooks_obj, "SessionEnd");
        arr.push(hook_entry(None, "corvia hooks run --event SessionEnd", 30000));
    }
}

fn ensure_array<'a>(obj: &'a mut serde_json::Map<String, serde_json::Value>, key: &str) -> &'a mut Vec<serde_json::Value> {
    // If the existing value is not an array, replace it with an empty array
    if obj.get(key).map(|v| !v.is_array()).unwrap_or(false) {
        obj.insert(key.to_string(), json!([]));
    }
    obj.entry(key).or_insert(json!([])).as_array_mut().unwrap()
}

fn hook_entry(matcher: Option<&str>, command: &str, timeout: u64) -> serde_json::Value {
    // Wrap command with:
    // 1. CORVIA_HOOKS_DISABLED=1 env var bypass (emergency escape hatch)
    // 2. Graceful fallback if `corvia hooks` subcommand doesn't exist (binary mismatch)
    //
    // The fallback checks if the exit was due to a missing subcommand (exit code 2 from
    // clap) vs a legitimate hook block (also exit 2 from doc-placement). We distinguish
    // by checking stderr for "unrecognized subcommand".
    let wrapped = format!(
        r#"[ "$CORVIA_HOOKS_DISABLED" = "1" ] && exit 0; {command} 2>/tmp/.corvia-hook-err || {{ rc=$?; grep -q "unrecognized subcommand" /tmp/.corvia-hook-err 2>/dev/null && exit 0; cat /tmp/.corvia-hook-err >&2; exit $rc; }}"#
    );
    let mut entry = json!({
        "hooks": [{
            "type": "command",
            "command": wrapped,
            "timeout": timeout,
        }]
    });
    if let Some(m) = matcher {
        entry.as_object_mut().unwrap().insert("matcher".into(), json!(m));
    }
    entry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_corvia_hook_bash() {
        let entry = json!({
            "hooks": [{"command": "bash .claude/hooks/record-session-start.sh"}]
        });
        assert!(is_corvia_hook(&entry));
    }

    #[test]
    fn test_is_corvia_hook_rust() {
        let entry = json!({
            "hooks": [{"command": "corvia hooks run --event SessionStart"}]
        });
        assert!(is_corvia_hook(&entry));
    }

    #[test]
    fn test_is_not_corvia_hook() {
        let entry = json!({
            "hooks": [{"command": "bash my-custom-hook.sh"}]
        });
        assert!(!is_corvia_hook(&entry));
    }

    #[test]
    fn test_hook_entry_with_matcher() {
        let entry = hook_entry(Some("Write|Edit"), "corvia hooks run --event PreToolUse --handler doc-placement", 5000);
        assert_eq!(entry["matcher"], "Write|Edit");
        let cmd = entry["hooks"][0]["command"].as_str().unwrap();
        assert!(cmd.contains("corvia hooks run --event PreToolUse --handler doc-placement"), "command should contain the corvia hooks call");
        assert!(cmd.contains("CORVIA_HOOKS_DISABLED"), "command should have bypass check");
        assert!(cmd.contains("unrecognized subcommand"), "command should have fallback for missing subcommand");
        assert_eq!(entry["hooks"][0]["timeout"], 5000);
    }

    #[test]
    fn test_hook_entry_without_matcher() {
        let entry = hook_entry(None, "corvia hooks run --event SessionStart", 5000);
        assert!(entry.get("matcher").is_none());
        let cmd = entry["hooks"][0]["command"].as_str().unwrap();
        assert!(cmd.contains("corvia hooks run --event SessionStart"));
    }

    #[test]
    fn test_add_hooks_with_disabled() {
        let mut hooks_obj = serde_json::Map::new();
        let mut config = HooksConfig::default();
        config.doc_placement = false;
        config.write_reminder = false;
        add_hooks(&mut hooks_obj, &config);

        // PreToolUse should have session-recording and corvia-first, but NOT doc-placement
        let pre = hooks_obj.get("PreToolUse").unwrap().as_array().unwrap();
        assert_eq!(pre.len(), 2); // session-recording + corvia-first
        let has_doc_placement = pre.iter().any(|e| {
            e.get("hooks").and_then(|h| h.as_array())
                .and_then(|a| a.first())
                .and_then(|h| h.get("command"))
                .and_then(|c| c.as_str())
                .map(|c| c.contains("doc-placement"))
                .unwrap_or(false)
        });
        assert!(!has_doc_placement);

        // PostToolUse should have session-recording but NOT write-reminder
        let post = hooks_obj.get("PostToolUse").unwrap().as_array().unwrap();
        assert_eq!(post.len(), 1); // session-recording only
    }

    #[test]
    fn test_purge_preserves_custom_hooks() {
        let custom = json!({
            "hooks": [{"command": "bash my-custom-hook.sh", "type": "command"}]
        });
        let corvia = json!({
            "hooks": [{"command": "corvia hooks run --event SessionStart", "type": "command"}]
        });
        assert!(!is_corvia_hook(&custom));
        assert!(is_corvia_hook(&corvia));
    }
}
