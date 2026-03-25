//! Agent identity registration on SessionStart.
//!
//! Calls the corvia server REST API to register the agent session.
//! Gracefully falls back if server is unavailable.

/// Register the agent with the corvia server and return a status message.
pub fn agent_check() -> String {
    let api = std::env::var("CORVIA_API").unwrap_or_else(|_| "http://localhost:8020".into());
    let agent_id = std::env::var("CORVIA_AGENT_ID").unwrap_or_else(|_| "claude-code".into());

    let client = match reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_millis(500))
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return format!("Agent auto-registration deferred (HTTP client init failed). Will register on first MCP write."),
    };

    let url = format!("{api}/api/dashboard/agents/{agent_id}/connect");
    match client.post(&url).json(&serde_json::json!({})).send() {
        Ok(resp) if resp.status().is_success() => {
            let sessions = resp.json::<serde_json::Value>()
                .ok()
                .and_then(|v| v.get("active_sessions").and_then(|s| s.as_u64()))
                .unwrap_or(0);
            format!("Connected as: {agent_id} (active sessions: {sessions})")
        }
        _ => {
            format!("Agent auto-registration deferred (server not ready). Will register on first MCP write.")
        }
    }
}
