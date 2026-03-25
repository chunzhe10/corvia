//! Reminder hooks: corvia-first and write-reminder.

/// Return the corvia-first reminder text.
pub fn corvia_first_reminder() -> String {
    "REMINDER: Have you called corvia_search or corvia_ask first? Per CLAUDE.md, \
     you MUST query corvia MCP tools before using Grep/Glob for any new task or \
     question. If you already called corvia this session, proceed."
        .to_string()
}

/// Check if a PostToolUse event was a successful git commit and return reminder.
pub fn write_reminder(stdin: &serde_json::Value) -> Option<String> {
    let command = stdin.get("tool_input")
        .and_then(|ti| ti.get("command"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if !command.contains("git commit ") {
        return None;
    }

    // Extract exit code — tool_response may be string or object
    let exit_code = stdin.get("tool_response")
        .and_then(|resp| {
            // Try object with exitCode or exit_code field
            resp.get("exitCode").or_else(|| resp.get("exit_code"))
                .and_then(|c| c.as_i64())
                .or_else(|| {
                    // If response is a string, it succeeded if we got here
                    if resp.is_string() { Some(0) } else { None }
                })
        })
        .unwrap_or(0);

    if exit_code != 0 {
        return None;
    }

    Some(
        "REMINDER: You just committed code. If this commit contains a design decision, \
         architectural change, or notable learning, persist it with corvia_write \
         (scope_id: corvia, agent_id: claude-code). Skip if the commit is trivial \
         (typo fix, formatting, etc.)."
            .to_string()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_reminder_git_commit() {
        let stdin = serde_json::json!({
            "tool_input": { "command": "git commit -m \"test\"" },
            "tool_response": { "exitCode": 0 }
        });
        assert!(write_reminder(&stdin).is_some());
    }

    #[test]
    fn test_write_reminder_failed_commit() {
        let stdin = serde_json::json!({
            "tool_input": { "command": "git commit -m \"test\"" },
            "tool_response": { "exitCode": 1 }
        });
        assert!(write_reminder(&stdin).is_none());
    }

    #[test]
    fn test_write_reminder_non_commit() {
        let stdin = serde_json::json!({
            "tool_input": { "command": "cargo build" },
            "tool_response": { "exitCode": 0 }
        });
        assert!(write_reminder(&stdin).is_none());
    }

    #[test]
    fn test_write_reminder_string_response() {
        // Claude Code sometimes sends tool_response as a plain string
        let stdin = serde_json::json!({
            "tool_input": { "command": "git commit -m \"test\"" },
            "tool_response": "commit abc123\n 1 file changed"
        });
        assert!(write_reminder(&stdin).is_some());
    }

    #[test]
    fn test_write_reminder_exit_code_field() {
        // Some versions use exit_code instead of exitCode
        let stdin = serde_json::json!({
            "tool_input": { "command": "git commit -m \"test\"" },
            "tool_response": { "exit_code": 0 }
        });
        assert!(write_reminder(&stdin).is_some());
    }

    #[test]
    fn test_corvia_first_reminder() {
        let msg = corvia_first_reminder();
        assert!(msg.contains("corvia_search"));
    }
}
