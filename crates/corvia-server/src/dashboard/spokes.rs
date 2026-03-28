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
    /// CPU usage percentage (0.0-100.0+). None if stats unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_percent: Option<f64>,
    /// Memory usage in bytes. None if stats unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    /// Memory limit in bytes. None if stats unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<u64>,
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
        cpu_percent: None,
        memory_bytes: None,
        memory_limit: None,
    }
}

/// Docker container stats response (subset of fields).
#[derive(Debug, Deserialize)]
struct DockerStats {
    #[serde(default)]
    cpu_stats: CpuStats,
    #[serde(default)]
    precpu_stats: CpuStats,
    #[serde(default)]
    memory_stats: MemoryStats,
}

#[derive(Debug, Default, Deserialize)]
struct CpuStats {
    #[serde(default)]
    cpu_usage: CpuUsage,
    #[serde(default)]
    system_cpu_usage: Option<u64>,
    #[serde(default)]
    online_cpus: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct CpuUsage {
    #[serde(default)]
    total_usage: u64,
}

#[derive(Debug, Default, Deserialize)]
struct MemoryStats {
    #[serde(default)]
    usage: Option<u64>,
    #[serde(default)]
    limit: Option<u64>,
}

/// Calculate CPU percentage from Docker stats (same formula as `docker stats`).
/// Includes the online_cpus multiplier so 100% = one full core.
fn calc_cpu_percent(stats: &DockerStats) -> f64 {
    let cpu_delta = stats.cpu_stats.cpu_usage.total_usage
        .saturating_sub(stats.precpu_stats.cpu_usage.total_usage) as f64;
    let sys_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0)
        .saturating_sub(stats.precpu_stats.system_cpu_usage.unwrap_or(0)) as f64;
    let num_cpus = stats.cpu_stats.online_cpus.unwrap_or(1).max(1) as f64;
    if sys_delta > 0.0 {
        (cpu_delta / sys_delta) * num_cpus * 100.0
    } else {
        0.0
    }
}

/// Query stats for a single container by name (one-shot, no stream).
async fn docker_container_stats(container_name: &str) -> Result<DockerStats, String> {
    use tokio::net::UnixStream;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Validate container name to prevent HTTP request smuggling via Docker socket.
    // Docker container names contain only [a-zA-Z0-9_.-].
    if !container_name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
        return Err(format!("Invalid container name: {container_name}"));
    }

    let socket_path = "/var/run/docker.sock";
    let mut stream = UnixStream::connect(socket_path)
        .await
        .map_err(|e| format!("Docker socket: {e}"))?;

    // stream=false gives a single JSON snapshot instead of streaming
    let request = format!(
        "GET /v1.43/containers/{container_name}/stats?stream=false HTTP/1.0\r\nHost: localhost\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).await
        .map_err(|e| format!("Docker write: {e}"))?;

    let mut buf = Vec::new();
    stream.take(1_048_576).read_to_end(&mut buf).await
        .map_err(|e| format!("Docker read: {e}"))?;

    let response = String::from_utf8_lossy(&buf);
    let status_line = response.lines().next().unwrap_or("");
    if !status_line.contains("200") {
        return Err(format!("Docker stats: {status_line}"));
    }

    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .ok_or_else(|| "Invalid stats response".to_string())?;

    serde_json::from_str(body.trim())
        .map_err(|e| format!("Stats parse: {e}"))
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
/// Fetches container list and stats for running containers.
pub async fn query_spokes() -> SpokesResponse {
    let handle = tokio::spawn(async move {
        match docker_list_spokes().await {
            Ok(containers) => {
                let mut spokes: Vec<DashboardSpokeInfo> = containers
                    .iter()
                    .map(container_to_spoke)
                    .collect();

                // Fetch stats for running containers (best-effort, parallel, 5s timeout each)
                let mut handles = Vec::new();
                for (i, spoke) in spokes.iter().enumerate() {
                    if spoke.container_state == "running" {
                        let name = spoke.name.clone();
                        handles.push((i, tokio::spawn(async move {
                            tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                docker_container_stats(&name),
                            ).await.unwrap_or(Err("stats timeout".into()))
                        })));
                    }
                }
                for (i, handle) in handles {
                    if let Ok(Ok(stats)) = handle.await {
                        spokes[i].cpu_percent = Some(calc_cpu_percent(&stats));
                        spokes[i].memory_bytes = stats.memory_stats.usage;
                        spokes[i].memory_limit = stats.memory_stats.limit;
                    }
                }

                SpokesResponse {
                    spokes,
                    warning: None,
                }
            }
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
        assert!(spoke.cpu_percent.is_none());
        assert!(spoke.memory_bytes.is_none());
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
    fn calc_cpu_percent_basic() {
        let stats = DockerStats {
            cpu_stats: CpuStats {
                cpu_usage: CpuUsage { total_usage: 200 },
                system_cpu_usage: Some(10000),
                online_cpus: Some(4),
            },
            precpu_stats: CpuStats {
                cpu_usage: CpuUsage { total_usage: 100 },
                system_cpu_usage: Some(9000),
                online_cpus: None,
            },
            memory_stats: MemoryStats { usage: Some(512), limit: Some(1024) },
        };
        let pct = calc_cpu_percent(&stats);
        // (100/1000) * 4 * 100 = 40.0
        assert!((pct - 40.0).abs() < 0.01);
    }

    #[test]
    fn calc_cpu_percent_zero_delta() {
        let stats = DockerStats {
            cpu_stats: CpuStats {
                cpu_usage: CpuUsage { total_usage: 100 },
                system_cpu_usage: Some(9000),
                online_cpus: Some(2),
            },
            precpu_stats: CpuStats {
                cpu_usage: CpuUsage { total_usage: 100 },
                system_cpu_usage: Some(9000),
                online_cpus: None,
            },
            memory_stats: MemoryStats::default(),
        };
        assert_eq!(calc_cpu_percent(&stats), 0.0);
    }

    #[test]
    fn counts_from_response_mixed() {
        let resp = SpokesResponse {
            spokes: vec![
                DashboardSpokeInfo {
                    name: "s1".into(), agent_id: "a1".into(), repo: "r".into(),
                    branch: "b".into(), issue: "1".into(), container_state: "running".into(),
                    container_status: "Up".into(), created: 0, health: "healthy".into(),
                    repo_url: None, cpu_percent: Some(5.2), memory_bytes: Some(1024),
                    memory_limit: Some(4096),
                },
                DashboardSpokeInfo {
                    name: "s2".into(), agent_id: "a2".into(), repo: "r".into(),
                    branch: "b".into(), issue: "2".into(), container_state: "exited".into(),
                    container_status: "Exited".into(), created: 0, health: "none".into(),
                    repo_url: None, cpu_percent: None, memory_bytes: None, memory_limit: None,
                },
            ],
            warning: None,
        };
        let (total, running) = counts_from_response(&resp);
        assert_eq!(total, 2);
        assert_eq!(running, 1);
    }
}
