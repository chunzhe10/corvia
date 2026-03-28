//! Spoke container management for multi-spoke workspaces.
//!
//! A spoke is a Docker container running Claude Code on one issue/branch,
//! connected to the hub's corvia server via MCP. This module provides:
//!
//! - [`HubContext`]: Container detection and host path resolution
//! - [`SpokeProvisioner`]: Create, list, destroy, restart spoke containers
//! - [`select_network`]: Deterministic Docker network selection

use bollard::models::{ContainerCreateBody, ContainerSummary, HostConfig};
use bollard::query_parameters::{
    CreateContainerOptionsBuilder, InspectContainerOptions, ListContainersOptionsBuilder,
    LogsOptionsBuilder, RemoveContainerOptions, StartContainerOptions, StopContainerOptions,
};
use bollard::Docker;
use corvia_common::config::{CorviaConfig, SpokeAuthMode, SpokeConfig};
use corvia_common::errors::{CorviaError, Result};
use std::collections::HashMap;
use tracing::info;

/// Resolved identity and mount table for the hub container.
///
/// When the CLI runs inside a devcontainer, it must translate its own
/// container paths to host paths for bind mounts into spoke containers
/// (Docker-from-Docker pattern).
pub struct HubContext {
    pub container_name: String,
    pub networks: HashMap<String, NetworkInfo>,
    /// Maps container_path -> host_path from the hub's mount table.
    pub host_mounts: HashMap<String, String>,
}

/// Minimal network info extracted from Docker inspect.
pub struct NetworkInfo {
    pub network_id: String,
}

impl HubContext {
    /// Detect the hub container by inspecting the current hostname.
    /// Returns an error if not running inside a Docker container.
    pub async fn detect(docker: &Docker) -> Result<Self> {
        let in_container = std::path::Path::new("/.dockerenv").exists()
            || std::fs::read_to_string("/proc/1/cgroup")
                .map(|s| s.contains("docker") || s.contains("containerd"))
                .unwrap_or(false);

        if !in_container {
            return Err(CorviaError::Config(
                "Spoke management requires running inside a Docker container \
                 with the host Docker socket mounted. \
                 Run `corvia workspace spoke check` to diagnose."
                    .into(),
            ));
        }

        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| {
                std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string())
            })
            .map_err(|_| CorviaError::Config("Cannot determine container hostname".into()))?;

        let inspect = docker
            .inspect_container(&hostname, None::<InspectContainerOptions>)
            .await
            .map_err(|e| {
                CorviaError::Docker(format!("Cannot inspect hub container '{hostname}': {e}"))
            })?;

        let mut host_mounts = HashMap::new();
        if let Some(mounts) = inspect.mounts {
            for m in mounts {
                if let (Some(src), Some(dst)) = (m.source, m.destination) {
                    host_mounts.insert(dst, src);
                }
            }
        }

        let mut networks = HashMap::new();
        if let Some(ns) = inspect.network_settings.and_then(|ns| ns.networks) {
            for (name, endpoint) in ns {
                networks.insert(
                    name,
                    NetworkInfo {
                        network_id: endpoint.network_id.unwrap_or_default(),
                    },
                );
            }
        }

        Ok(Self {
            container_name: hostname,
            networks,
            host_mounts,
        })
    }

    /// Translate a container path to a host path using the mount table.
    /// Returns an error with diagnostic info if the path cannot be resolved.
    pub fn host_path_or_err(&self, container_path: &str) -> Result<String> {
        self.host_path(container_path).ok_or_else(|| {
            let mounts: Vec<_> = self.host_mounts.keys().collect();
            CorviaError::Config(format!(
                "Cannot resolve host path for '{}'. \
                 Hub mount table: {:?}. \
                 Ensure the devcontainer mounts cover this path.",
                container_path, mounts
            ))
        })
    }

    /// Translate a container path to a host path. Returns None if no mount covers this path.
    pub fn host_path(&self, container_path: &str) -> Option<String> {
        let mut best_match: Option<(&str, &str)> = None;
        for (cpath, hpath) in &self.host_mounts {
            if container_path.starts_with(cpath.as_str())
                && best_match.is_none_or(|(bp, _)| cpath.len() > bp.len())
            {
                best_match = Some((cpath.as_str(), hpath.as_str()));
            }
        }
        best_match.map(|(cpath, hpath)| {
            let suffix = &container_path[cpath.len()..];
            format!("{hpath}{suffix}")
        })
    }
}

/// Select the correct Docker network, deterministically.
///
/// Strategy:
/// 1. If config override is set, use it (error if not found).
/// 2. Filter out bridge/host/none.
/// 3. If exactly one candidate, use it.
/// 4. If multiple, prefer one containing "devcontainer".
/// 5. Otherwise error with candidates list.
pub fn select_network(
    networks: &HashMap<String, NetworkInfo>,
    config_override: Option<&str>,
) -> Result<String> {
    if let Some(net) = config_override {
        if networks.contains_key(net) {
            return Ok(net.to_string());
        }
        return Err(CorviaError::Config(format!(
            "Configured network '{}' not found. Available: {:?}",
            net,
            networks.keys().collect::<Vec<_>>()
        )));
    }

    let candidates: Vec<_> = networks
        .keys()
        .filter(|n| !matches!(n.as_str(), "bridge" | "host" | "none"))
        .collect();

    match candidates.len() {
        0 => Err(CorviaError::Config(
            "No user-defined Docker network found. \
             Set [workspace.spokes] network in corvia.toml."
                .into(),
        )),
        1 => Ok(candidates[0].clone()),
        _ => {
            if let Some(net) = candidates.iter().find(|n| n.contains("devcontainer")) {
                return Ok((*net).clone());
            }
            Err(CorviaError::Config(format!(
                "Multiple networks found: {:?}. Set [workspace.spokes] network in corvia.toml.",
                candidates
            )))
        }
    }
}

/// Parse a memory limit string (e.g., "4g", "512m") to bytes.
pub fn parse_memory_limit(limit: &str) -> i64 {
    let limit = limit.trim().to_lowercase();
    if let Some(val) = limit.strip_suffix('g') {
        val.parse::<i64>().unwrap_or(4) * 1024 * 1024 * 1024
    } else if let Some(val) = limit.strip_suffix('m') {
        val.parse::<i64>().unwrap_or(4096) * 1024 * 1024
    } else {
        // Default to 4GB
        4 * 1024 * 1024 * 1024
    }
}

/// Metadata for a running or exited spoke.
pub struct SpokeInfo {
    pub name: String,
    pub repo: String,
    pub branch: String,
    pub issue: String,
    pub agent_id: String,
    pub state: String,
    pub status: String,
}

impl SpokeInfo {
    /// Extract spoke info from a Docker container summary.
    fn from_container(container: &ContainerSummary) -> Option<Self> {
        let labels = container.labels.as_ref()?;
        let name = container
            .names
            .as_ref()
            .and_then(|n| n.first())
            .map(|n| n.trim_start_matches('/').to_string())
            .unwrap_or_default();
        Some(Self {
            name,
            repo: labels.get("corvia.repo").cloned().unwrap_or_default(),
            branch: labels.get("corvia.branch").cloned().unwrap_or_default(),
            issue: labels.get("corvia.issue").cloned().unwrap_or_default(),
            agent_id: labels.get("corvia.agent_id").cloned().unwrap_or_default(),
            state: container.state.as_ref().map(|s| s.to_string()).unwrap_or_default(),
            status: container.status.clone().unwrap_or_default(),
        })
    }
}

/// Manages spoke container lifecycle via the Docker API.
pub struct SpokeProvisioner {
    docker: Docker,
}

impl SpokeProvisioner {
    pub fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| CorviaError::Docker(format!("Failed to connect to Docker: {e}")))?;
        Ok(Self { docker })
    }

    /// Expose the Docker client for HubContext detection.
    pub fn docker(&self) -> &Docker {
        &self.docker
    }

    /// List spoke containers (optionally including stopped ones).
    pub async fn list(&self, include_stopped: bool) -> Result<Vec<SpokeInfo>> {
        let mut filters: HashMap<String, Vec<String>> = HashMap::new();
        filters.insert("label".into(), vec!["corvia.spoke=true".into()]);
        if !include_stopped {
            filters.insert("status".into(), vec!["running".into()]);
        }

        let options = ListContainersOptionsBuilder::new()
            .all(include_stopped)
            .filters(&filters)
            .build();

        let containers = self
            .docker
            .list_containers(Some(options))
            .await
            .map_err(|e| CorviaError::Docker(format!("Failed to list spoke containers: {e}")))?;

        Ok(containers
            .iter()
            .filter_map(SpokeInfo::from_container)
            .collect())
    }

    /// Find a spoke by name prefix.
    pub async fn find(&self, name_prefix: &str) -> Result<Option<SpokeInfo>> {
        let all = self.list(true).await?;
        Ok(all.into_iter().find(|s| s.name.starts_with(name_prefix)))
    }

    /// Create and start a spoke container.
    #[allow(clippy::too_many_arguments)]
    pub async fn create(
        &self,
        spoke_name: &str,
        agent_id: &str,
        hub: &HubContext,
        config: &CorviaConfig,
        spoke_config: &SpokeConfig,
        network: &str,
        repo_name: &str,
        repo_url: &str,
        issue: Option<u32>,
        branch_name: &str,
        workspace_root: &str,
    ) -> Result<()> {
        // Read the MCP token from data dir
        let data_dir = std::path::Path::new(&config.storage.data_dir);
        let mcp_token_path = data_dir.join("mcp-token");
        let mcp_token = std::fs::read_to_string(&mcp_token_path).unwrap_or_default();
        let mcp_token = mcp_token.trim().to_string();

        // Generate per-spoke auth token
        let spoke_token = uuid::Uuid::new_v4().to_string();

        // Resolve GitHub token from env
        let github_token = std::env::var("GITHUB_TOKEN").unwrap_or_default();

        // Build bind mounts for credentials and workspace instruction files
        let mut binds = Vec::new();

        // Credentials mount
        let creds_path = format!("{workspace_root}/.claude/.credentials.json");
        if let Some(host_path) = hub.host_path(&creds_path) {
            binds.push(format!("{host_path}:/spoke-config/.credentials.json:ro"));
        }

        // Workspace instruction files
        let agents_md = format!("{workspace_root}/AGENTS.md");
        if let Some(host_path) = hub.host_path(&agents_md) {
            binds.push(format!("{host_path}:/spoke-config/AGENTS.md:ro"));
        }

        let claude_md = format!("{workspace_root}/CLAUDE.md");
        if let Some(host_path) = hub.host_path(&claude_md) {
            binds.push(format!("{host_path}:/spoke-config/CLAUDE.md:ro"));
        }

        let skills_dir = format!("{workspace_root}/.agents/skills");
        if let Some(host_path) = hub.host_path(&skills_dir) {
            binds.push(format!("{host_path}:/spoke-config/skills:ro"));
        }

        // Build environment variables
        let mut env = vec![
            format!("CORVIA_AGENT_ID={agent_id}"),
            format!("CORVIA_SPOKE_TOKEN={spoke_token}"),
            format!("CORVIA_MCP_URL=http://app:8020/mcp"),
            format!("CORVIA_MCP_TOKEN={mcp_token}"),
            format!("CORVIA_REPO_URL={repo_url}"),
            format!("CORVIA_ISSUE={}", issue.unwrap_or(0)),
            format!("CORVIA_BRANCH={branch_name}"),
        ];
        if !github_token.is_empty() {
            env.push(format!("GITHUB_TOKEN={github_token}"));
        }

        // API key auth mode
        if spoke_config.auth_mode == SpokeAuthMode::ApiKey {
            let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
            if !api_key.is_empty() {
                env.push(format!("ANTHROPIC_API_KEY={api_key}"));
            }
        }

        // Labels for spoke identification
        let labels: HashMap<String, String> = HashMap::from([
            ("corvia.spoke".into(), "true".into()),
            (
                "corvia.workspace".into(),
                config.project.name.clone(),
            ),
            ("corvia.repo".into(), repo_name.into()),
            (
                "corvia.issue".into(),
                issue.map(|i| i.to_string()).unwrap_or_default(),
            ),
            ("corvia.branch".into(), branch_name.into()),
            ("corvia.agent_id".into(), agent_id.into()),
        ]);

        // Log config
        let log_config = bollard::models::HostConfigLogConfig {
            typ: Some("json-file".into()),
            config: Some(HashMap::from([
                ("max-size".into(), "50m".into()),
                ("max-file".into(), "3".into()),
            ])),
        };

        let host_config = HostConfig {
            binds: if binds.is_empty() {
                None
            } else {
                Some(binds)
            },
            network_mode: Some(network.into()),
            memory: Some(parse_memory_limit(&spoke_config.memory_limit)),
            cpu_shares: Some(spoke_config.cpu_shares as i64),
            log_config: Some(log_config),
            ..Default::default()
        };

        let container_config = ContainerCreateBody {
            image: Some(spoke_config.image.clone()),
            host_config: Some(host_config),
            env: Some(env),
            labels: Some(labels),
            working_dir: Some("/workspace".into()),
            ..Default::default()
        };

        // Remove existing container if present
        let _ = self
            .docker
            .remove_container(spoke_name, Some(RemoveContainerOptions { force: true, ..Default::default() }))
            .await;

        let create_options = CreateContainerOptionsBuilder::new()
            .name(spoke_name)
            .build();

        self.docker
            .create_container(Some(create_options), container_config)
            .await
            .map_err(|e| CorviaError::Docker(format!("Failed to create spoke '{spoke_name}': {e}")))?;

        self.docker
            .start_container(spoke_name, None::<StartContainerOptions>)
            .await
            .map_err(|e| CorviaError::Docker(format!("Failed to start spoke '{spoke_name}': {e}")))?;

        info!(spoke_name, repo = repo_name, ?issue, "Spoke started");
        Ok(())
    }

    /// Stream logs from a spoke container.
    pub async fn logs(
        &self,
        spoke_name: &str,
        follow: bool,
        tail: u32,
    ) -> Result<impl futures_util::Stream<Item = std::result::Result<bollard::container::LogOutput, bollard::errors::Error>>>
    {
        let options = LogsOptionsBuilder::new()
            .follow(follow)
            .stdout(true)
            .stderr(true)
            .tail(&tail.to_string())
            .build();

        Ok(self.docker.logs(spoke_name, Some(options)))
    }

    /// Stop and remove a spoke container.
    pub async fn destroy(&self, spoke_name: &str, force: bool) -> Result<()> {
        if !force {
            // Graceful: send SIGTERM, wait 10s, then SIGKILL
            self.docker
                .stop_container(spoke_name, Some(StopContainerOptions { t: Some(10), signal: None }))
                .await
                .map_err(|e| {
                    CorviaError::Docker(format!("Failed to stop spoke '{spoke_name}': {e}"))
                })?;
        }

        self.docker
            .remove_container(
                spoke_name,
                Some(RemoveContainerOptions {
                    force,
                    ..Default::default()
                }),
            )
            .await
            .map_err(|e| {
                CorviaError::Docker(format!("Failed to remove spoke '{spoke_name}': {e}"))
            })?;

        info!(spoke_name, "Spoke destroyed");
        Ok(())
    }

    /// Destroy all spoke containers.
    pub async fn destroy_all(&self) -> Result<u32> {
        let spokes = self.list(true).await?;
        let count = spokes.len() as u32;
        for spoke in &spokes {
            let _ = self.destroy(&spoke.name, true).await;
        }
        Ok(count)
    }

    /// Restart a spoke container, preserving its filesystem.
    pub async fn restart(&self, spoke_name: &str) -> Result<()> {
        self.docker
            .stop_container(spoke_name, Some(StopContainerOptions { t: Some(10), signal: None }))
            .await
            .map_err(|e| {
                CorviaError::Docker(format!("Failed to stop spoke '{spoke_name}': {e}"))
            })?;

        self.docker
            .start_container(spoke_name, None::<StartContainerOptions>)
            .await
            .map_err(|e| {
                CorviaError::Docker(format!("Failed to restart spoke '{spoke_name}': {e}"))
            })?;

        info!(spoke_name, "Spoke restarted");
        Ok(())
    }

    /// Run pre-flight checks for spoke capability.
    pub async fn check(&self) -> Vec<CheckResult> {
        let mut results = Vec::new();

        // 1. Docker socket
        match self.docker.ping().await {
            Ok(_) => results.push(CheckResult::pass("Docker socket accessible")),
            Err(e) => results.push(CheckResult::fail(
                "Docker socket",
                &format!("Cannot connect: {e}"),
            )),
        }

        // 2. Container detection
        let in_container = std::path::Path::new("/.dockerenv").exists()
            || std::fs::read_to_string("/proc/1/cgroup")
                .map(|s| s.contains("docker") || s.contains("containerd"))
                .unwrap_or(false);
        if in_container {
            results.push(CheckResult::pass("Running inside a container"));
        } else {
            results.push(CheckResult::fail(
                "Container detection",
                "Not running inside a Docker container. Spoke management requires Docker-from-Docker.",
            ));
        }

        // 3. GITHUB_TOKEN
        if std::env::var("GITHUB_TOKEN")
            .unwrap_or_default()
            .is_empty()
        {
            results.push(CheckResult::warn(
                "GITHUB_TOKEN",
                "Not set. Spokes won't be able to push to GitHub.",
            ));
        } else {
            results.push(CheckResult::pass("GITHUB_TOKEN is set"));
        }

        // 4. Credentials file
        let creds = std::path::Path::new("/root/.claude/.credentials.json");
        if creds.exists() {
            results.push(CheckResult::pass("Claude credentials found"));
        } else {
            results.push(CheckResult::warn(
                "Claude credentials",
                &format!("{} not found", creds.display()),
            ));
        }

        // 5. MCP token
        // Try to find data_dir from config
        let mcp_token_exists = std::path::Path::new(".corvia/mcp-token").exists();
        if mcp_token_exists {
            results.push(CheckResult::pass("MCP token found"));
        } else {
            results.push(CheckResult::warn(
                "MCP token",
                ".corvia/mcp-token not found. Server may need to be started with 0.0.0.0 binding.",
            ));
        }

        results
    }
}

/// Result of a pre-flight check.
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
}

#[derive(PartialEq)]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckResult {
    fn pass(name: &str) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Pass,
            message: String::new(),
        }
    }
    fn warn(name: &str, message: &str) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Warn,
            message: message.into(),
        }
    }
    fn fail(name: &str, message: &str) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Fail,
            message: message.into(),
        }
    }
}

/// Generate a spoke name from issue number or branch name.
pub fn generate_spoke_name(workspace_name: &str, issue: Option<u32>, branch: Option<&str>) -> String {
    let slug = workspace_name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>();
    let timestamp = chrono::Utc::now().format("%m%d%H%M");
    let suffix = if let Some(n) = issue {
        n.to_string()
    } else if let Some(b) = branch {
        b.chars()
            .take(20)
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect()
    } else {
        "manual".into()
    };
    format!("corvia-{slug}-spoke-{suffix}-{timestamp}")
}

/// Generate the agent ID for a spoke.
pub fn generate_agent_id(issue: Option<u32>, branch: Option<&str>) -> String {
    if let Some(n) = issue {
        format!("spoke-{n}")
    } else if let Some(b) = branch {
        let slug: String = b
            .chars()
            .take(30)
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        format!("spoke-{slug}")
    } else {
        format!("spoke-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("x"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_path_resolution_exact_mount() {
        let hub = HubContext {
            container_name: "test".into(),
            networks: HashMap::new(),
            host_mounts: HashMap::from([
                ("/workspaces".into(), "/home/user/workspaces".into()),
            ]),
        };
        assert_eq!(
            hub.host_path("/workspaces/corvia-workspace/AGENTS.md"),
            Some("/home/user/workspaces/corvia-workspace/AGENTS.md".into())
        );
    }

    #[test]
    fn test_host_path_resolution_nested_mount_wins() {
        let hub = HubContext {
            container_name: "test".into(),
            networks: HashMap::new(),
            host_mounts: HashMap::from([
                ("/workspaces".into(), "/host/ws".into()),
                ("/workspaces/corvia-workspace".into(), "/host/cw".into()),
            ]),
        };
        // More specific mount should win
        assert_eq!(
            hub.host_path("/workspaces/corvia-workspace/file.txt"),
            Some("/host/cw/file.txt".into())
        );
    }

    #[test]
    fn test_host_path_resolution_no_match() {
        let hub = HubContext {
            container_name: "test".into(),
            networks: HashMap::new(),
            host_mounts: HashMap::from([("/data".into(), "/host/data".into())]),
        };
        assert_eq!(hub.host_path("/workspaces/file.txt"), None);
    }

    #[test]
    fn test_host_path_or_err_returns_diagnostic() {
        let hub = HubContext {
            container_name: "test".into(),
            networks: HashMap::new(),
            host_mounts: HashMap::from([("/data".into(), "/host/data".into())]),
        };
        let err = hub.host_path_or_err("/workspaces/file.txt").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("/workspaces/file.txt"));
        assert!(msg.contains("/data"));
    }

    #[test]
    fn test_select_network_single_user_network() {
        let networks = HashMap::from([
            ("bridge".into(), NetworkInfo { network_id: "1".into() }),
            ("my-net".into(), NetworkInfo { network_id: "2".into() }),
        ]);
        assert_eq!(select_network(&networks, None).unwrap(), "my-net");
    }

    #[test]
    fn test_select_network_prefers_devcontainer() {
        let networks = HashMap::from([
            ("bridge".into(), NetworkInfo { network_id: "1".into() }),
            ("custom-net".into(), NetworkInfo { network_id: "2".into() }),
            (
                "devcontainer-abc_default".into(),
                NetworkInfo { network_id: "3".into() },
            ),
        ]);
        assert_eq!(
            select_network(&networks, None).unwrap(),
            "devcontainer-abc_default"
        );
    }

    #[test]
    fn test_select_network_config_override() {
        let networks = HashMap::from([
            ("bridge".into(), NetworkInfo { network_id: "1".into() }),
            ("my-net".into(), NetworkInfo { network_id: "2".into() }),
        ]);
        assert_eq!(
            select_network(&networks, Some("my-net")).unwrap(),
            "my-net"
        );
    }

    #[test]
    fn test_select_network_config_override_not_found() {
        let networks = HashMap::from([
            ("bridge".into(), NetworkInfo { network_id: "1".into() }),
        ]);
        assert!(select_network(&networks, Some("nonexistent")).is_err());
    }

    #[test]
    fn test_select_network_no_user_networks() {
        let networks = HashMap::from([
            ("bridge".into(), NetworkInfo { network_id: "1".into() }),
            ("host".into(), NetworkInfo { network_id: "2".into() }),
            ("none".into(), NetworkInfo { network_id: "3".into() }),
        ]);
        assert!(select_network(&networks, None).is_err());
    }

    #[test]
    fn test_select_network_ambiguous() {
        let networks = HashMap::from([
            ("net-a".into(), NetworkInfo { network_id: "1".into() }),
            ("net-b".into(), NetworkInfo { network_id: "2".into() }),
        ]);
        assert!(select_network(&networks, None).is_err());
    }

    #[test]
    fn test_parse_memory_limit_gigabytes() {
        assert_eq!(parse_memory_limit("4g"), 4 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("2G"), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_memory_limit_megabytes() {
        assert_eq!(parse_memory_limit("512m"), 512 * 1024 * 1024);
        assert_eq!(parse_memory_limit("1024M"), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_memory_limit_invalid_defaults() {
        // Invalid number defaults to 4GB
        assert_eq!(parse_memory_limit("xyzg"), 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_generate_spoke_name_with_issue() {
        let name = generate_spoke_name("corvia-workspace", Some(42), None);
        assert!(name.starts_with("corvia-corvia-workspace-spoke-42-"));
    }

    #[test]
    fn test_generate_spoke_name_with_branch() {
        let name = generate_spoke_name("corvia", None, Some("feat/my-branch"));
        assert!(name.contains("spoke-feat-my-branch"));
    }

    #[test]
    fn test_generate_agent_id_with_issue() {
        assert_eq!(generate_agent_id(Some(42), None), "spoke-42");
    }

    #[test]
    fn test_generate_agent_id_with_branch() {
        let id = generate_agent_id(None, Some("feat/my-branch"));
        assert!(id.starts_with("spoke-feat-my-branch"));
    }

    #[test]
    fn test_spoke_config_defaults() {
        let config = SpokeConfig::default();
        assert_eq!(config.image, "corvia-spoke:latest");
        assert_eq!(config.memory_limit, "4g");
        assert_eq!(config.cpu_shares, 512);
        assert!(config.network.is_none());
        assert_eq!(config.auth_mode, SpokeAuthMode::Credentials);
    }
}
