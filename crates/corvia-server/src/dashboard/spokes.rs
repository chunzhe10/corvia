//! Spoke container dashboard endpoints.
//!
//! Queries Docker directly via Unix socket to avoid importing bollard
//! which would pull axum 0.7 types into the server crate and break handler bounds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata for a spoke container as seen from the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSpokeInfo {
    pub name: String,
    pub agent_id: String,
    pub repo: String,
    pub branch: String,
    pub issue: String,
    pub container_state: String,
    pub container_status: String,
    pub created: i64,
    pub health: String,
    pub repo_url: Option<String>,
}

/// Response for GET /api/dashboard/spokes.
#[derive(Debug, Clone, Serialize)]
pub struct SpokesResponse {
    pub spokes: Vec<DashboardSpokeInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// Raw Docker API container response (subset of fields we need).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct DockerContainer {
    #[serde(default)]
    names: Vec<String>,
    #[serde(default)]
    state: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    created: i64,
    #[serde(default)]
    labels: HashMap<String, String>,
}

fn extract_health(status: &str) -> String {
    if status.contains("healthy") && !status.contains("unhealthy") {
        "healthy".to_string()
    } else if status.contains("unhealthy") {
        "unhealthy".to_string()
    } else if status.contains("starting") {
        "starting".to_string()
    } else {
        "none".to_string()
    }
}

fn container_to_spoke(c: &DockerContainer) -> DashboardSpokeInfo {
    DashboardSpokeInfo {
        name: c.names.first()
            .map(|n| n.trim_start_matches('/').to_string())
            .unwrap_or_default(),
        agent_id: c.labels.get("corvia.agent_id").cloned().unwrap_or_default(),
        repo: c.labels.get("corvia.repo").cloned().unwrap_or_default(),
        branch: c.labels.get("corvia.branch").cloned().unwrap_or_default(),
        issue: c.labels.get("corvia.issue").cloned().unwrap_or_default(),
        container_state: c.state.to_lowercase(),
        container_status: c.status.clone(),
        created: c.created,
        health: extract_health(&c.status),
        repo_url: c.labels.get("corvia.repo_url").cloned(),
    }
}

/// URL encoding for the Docker filter parameter.
fn url_encode_filter(s: &str) -> String {
    s.replace('{', "%7B")
     .replace('}', "%7D")
     .replace('"', "%22")
     .replace('[', "%5B")
     .replace(']', "%5D")
     .replace(':', "%3A")
     .replace('=', "%3D")
     .replace(' ', "%20")
     .replace('&', "%26")
}

/// Query Docker for spoke containers via Unix socket.
async fn docker_list_spokes() -> Result<Vec<DockerContainer>, String> {
    use tokio::net::UnixStream;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let socket_path = "/var/run/docker.sock";
    if !std::path::Path::new(socket_path).exists() {
        return Err("Docker socket not found".into());
    }

    let mut stream = UnixStream::connect(socket_path)
        .await
        .map_err(|e| format!("Cannot connect to Docker socket: {e}"))?;

    let filter = r#"{"label":["corvia.spoke=true"]}"#;
    let encoded = url_encode_filter(filter);
    let request = format!(
        "GET /v1.43/containers/json?all=true&filters={encoded} HTTP/1.0\r\nHost: localhost\r\n\r\n"
    );

    stream.write_all(request.as_bytes()).await
        .map_err(|e| format!("Docker write failed: {e}"))?;

    // Bounded read: cap at 1MB to prevent unbounded memory consumption
    let mut buf = Vec::new();
    stream.take(1_048_576).read_to_end(&mut buf).await
        .map_err(|e| format!("Docker read failed: {e}"))?;

    let response = String::from_utf8_lossy(&buf);

    // Verify HTTP status before parsing body
    let status_line = response.lines().next().unwrap_or("");
    if !status_line.contains("200") {
        return Err(format!("Docker returned: {status_line}"));
    }

    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .ok_or_else(|| "Invalid Docker response".to_string())?;

    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(vec![]);
    }

    serde_json::from_str(trimmed)
        .map_err(|e| format!("Docker JSON parse failed: {e}"))
}

/// Query spokes via tokio::spawn to isolate from handler futures.
pub async fn query_spokes() -> SpokesResponse {
    let handle = tokio::spawn(async move {
        match docker_list_spokes().await {
            Ok(containers) => SpokesResponse {
                spokes: containers.iter().map(container_to_spoke).collect(),
                warning: None,
            },
            Err(e) => SpokesResponse {
                spokes: vec![],
                warning: Some(e),
            },
        }
    });
    handle.await.unwrap_or(SpokesResponse {
        spokes: vec![],
        warning: Some("Spoke query task failed".into()),
    })
}

/// Query spoke counts. Returns (total, running).
pub fn counts_from_response(resp: &SpokesResponse) -> (usize, usize) {
    let total = resp.spokes.len();
    let running = resp.spokes.iter().filter(|s| s.container_state == "running").count();
    (total, running)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_health_healthy() {
        assert_eq!(extract_health("Up 2 hours (healthy)"), "healthy");
    }

    #[test]
    fn extract_health_unhealthy() {
        assert_eq!(extract_health("Up 1 hour (unhealthy)"), "unhealthy");
    }

    #[test]
    fn extract_health_starting() {
        assert_eq!(extract_health("Up 30 seconds (health: starting)"), "starting");
    }

    #[test]
    fn extract_health_exited() {
        assert_eq!(extract_health("Exited (0) 5 minutes ago"), "none");
    }

    #[test]
    fn extract_health_empty() {
        assert_eq!(extract_health(""), "none");
    }

    #[test]
    fn container_to_spoke_full_labels() {
        let c = DockerContainer {
            names: vec!["/spoke-42".to_string()],
            state: "running".to_string(),
            status: "Up 2 hours (healthy)".to_string(),
            created: 1711600000,
            labels: HashMap::from([
                ("corvia.agent_id".into(), "spoke-42".into()),
                ("corvia.repo".into(), "corvia".into()),
                ("corvia.branch".into(), "feat/42-bm25".into()),
                ("corvia.issue".into(), "42".into()),
                ("corvia.repo_url".into(), "https://github.com/org/repo".into()),
            ]),
        };
        let spoke = container_to_spoke(&c);
        assert_eq!(spoke.name, "spoke-42");
        assert_eq!(spoke.agent_id, "spoke-42");
        assert_eq!(spoke.repo, "corvia");
        assert_eq!(spoke.branch, "feat/42-bm25");
        assert_eq!(spoke.issue, "42");
        assert_eq!(spoke.container_state, "running");
        assert_eq!(spoke.health, "healthy");
        assert_eq!(spoke.repo_url.as_deref(), Some("https://github.com/org/repo"));
    }

    #[test]
    fn container_to_spoke_missing_labels() {
        let c = DockerContainer {
            names: vec![],
            state: "exited".to_string(),
            status: "Exited (0)".to_string(),
            created: 0,
            labels: HashMap::new(),
        };
        let spoke = container_to_spoke(&c);
        assert_eq!(spoke.name, "");
        assert_eq!(spoke.agent_id, "");
        assert_eq!(spoke.health, "none");
        assert!(spoke.repo_url.is_none());
    }

    #[test]
    fn url_encode_filter_encodes_special_chars() {
        let input = r#"{"label":["corvia.spoke=true"]}"#;
        let encoded = url_encode_filter(input);
        assert!(!encoded.contains('{'));
        assert!(!encoded.contains('}'));
        assert!(!encoded.contains('"'));
        assert!(!encoded.contains('['));
        assert!(!encoded.contains(']'));
        assert!(!encoded.contains('='));
        assert!(encoded.contains("%7B"));
        assert!(encoded.contains("%3D"));
    }

    #[test]
    fn counts_from_response_mixed() {
        let resp = SpokesResponse {
            spokes: vec![
                DashboardSpokeInfo {
                    name: "s1".into(), agent_id: "a1".into(), repo: "r".into(),
                    branch: "b".into(), issue: "1".into(), container_state: "running".into(),
                    container_status: "Up".into(), created: 0, health: "healthy".into(),
                    repo_url: None,
                },
                DashboardSpokeInfo {
                    name: "s2".into(), agent_id: "a2".into(), repo: "r".into(),
                    branch: "b".into(), issue: "2".into(), container_state: "exited".into(),
                    container_status: "Exited".into(), created: 0, health: "none".into(),
                    repo_url: None,
                },
            ],
            warning: None,
        };
        let (total, running) = counts_from_response(&resp);
        assert_eq!(total, 2);
        assert_eq!(running, 1);
    }
}
