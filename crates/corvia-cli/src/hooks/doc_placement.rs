//! Doc placement validation for Write/Edit tool calls.
//!
//! Checks file paths against workspace doc placement rules from corvia.toml.
//! Returns Allow or Block(reason) to control whether Claude Code proceeds.

use anyhow::Result;
use corvia_common::config::CorviaConfig;

/// Result of a doc placement check.
#[derive(Debug, PartialEq)]
pub enum HookResult {
    Allow,
    Block(String),
}

/// Common root-level markdown files that are always allowed.
const ALLOWED_ROOT_FILES: &[&str] = &[
    "README.md", "CLAUDE.md", "AGENTS.md", "CHANGELOG.md",
    "CONTRIBUTING.md", "LICENSE.md", "CLAUDE-AUTONOMOUS.md",
];

/// Check whether a file path is allowed by workspace doc placement rules.
pub fn check(stdin: &serde_json::Value) -> Result<HookResult> {
    // Extract file_path from stdin (Claude Code provides it in tool_input for Write/Edit)
    let file_path = stdin.get("tool_input")
        .and_then(|ti| ti.get("file_path"))
        .or_else(|| stdin.get("file_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if file_path.is_empty() {
        return Ok(HookResult::Allow);
    }

    // Strip workspace prefix to get relative path
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let rel_path = file_path
        .strip_prefix(&format!("{cwd}/"))
        .unwrap_or(file_path);

    // Only check markdown files
    if !is_markdown(rel_path) {
        return Ok(HookResult::Allow);
    }

    // Common root-level files — always allowed
    if ALLOWED_ROOT_FILES.contains(&rel_path) {
        return Ok(HookResult::Allow);
    }

    // Agent skills and config — always allowed
    if rel_path.starts_with(".agents/") {
        return Ok(HookResult::Allow);
    }

    // Load rules from corvia.toml (best effort)
    let config_path = CorviaConfig::config_path();
    let config = if config_path.exists() {
        CorviaConfig::load(&config_path).ok()
    } else {
        None
    };

    let ws = config.as_ref().and_then(|c| c.workspace.as_ref());
    let docs = ws.and_then(|w| w.docs.as_ref());
    let rules = docs.and_then(|d| d.rules.as_ref());

    // Check blocked paths
    if let Some(rules) = rules {
        for pattern in &rules.blocked_paths {
            if glob_match(rel_path, pattern) {
                return Ok(HookResult::Block(format!(
                    "BLOCKED: '{rel_path}' matches blocked path '{pattern}'. Save product docs to repos/<repo>/docs/ instead."
                )));
            }
        }
    }

    // Check allowed paths: repos/*/docs/* and configured workspace subdirs
    if rel_path.starts_with("repos/") && rel_path.contains("/docs/") {
        return Ok(HookResult::Allow);
    }

    let allowed_subdirs = docs
        .map(|d| d.allowed_workspace_subdirs.as_slice())
        .unwrap_or_default();
    for subdir in allowed_subdirs {
        if rel_path.starts_with(&format!("docs/{subdir}/")) {
            return Ok(HookResult::Allow);
        }
    }

    // Memory files — always allowed
    if rel_path.starts_with(".claude/") || rel_path.starts_with(".corvia/") {
        return Ok(HookResult::Allow);
    }

    // Unknown location — warn but allow
    eprintln!("NOTE: '{rel_path}' is in an unusual location for docs.");
    Ok(HookResult::Allow)
}

fn is_markdown(path: &str) -> bool {
    path.ends_with(".md") || path.ends_with(".mdx") || path.ends_with(".rst")
}

/// Simple glob matching: supports trailing `*` wildcard (e.g., `docs/superpowers/*`).
fn glob_match(path: &str, pattern: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        path.starts_with(prefix)
    } else {
        path == pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stdin(file_path: &str) -> serde_json::Value {
        serde_json::json!({
            "tool_input": { "file_path": file_path }
        })
    }

    #[test]
    fn test_non_markdown_allowed() {
        let result = check(&make_stdin("src/main.rs")).unwrap();
        assert_eq!(result, HookResult::Allow);
    }

    #[test]
    fn test_root_files_allowed() {
        for f in ALLOWED_ROOT_FILES {
            let result = check(&make_stdin(f)).unwrap();
            assert_eq!(result, HookResult::Allow, "Expected Allow for {f}");
        }
    }

    #[test]
    fn test_agents_dir_allowed() {
        let result = check(&make_stdin(".agents/skills/test.md")).unwrap();
        assert_eq!(result, HookResult::Allow);
    }

    #[test]
    fn test_empty_path_allowed() {
        let result = check(&serde_json::json!({})).unwrap();
        assert_eq!(result, HookResult::Allow);
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("docs/superpowers/test.md", "docs/superpowers/*"));
        assert!(!glob_match("docs/decisions/test.md", "docs/superpowers/*"));
        assert!(glob_match("exact.md", "exact.md"));
    }
}
