//! Shared kernel operations callable from both CLI and MCP server.
//!
//! This module provides a unified API for system introspection and control,
//! eliminating code duplication between `corvia-cli` and `corvia-server`.

use crate::adapter_discovery::{discover_adapters, DiscoveredAdapter};
use crate::agent_coordinator::AgentCoordinator;
use crate::agent_coordinator::GcReport;
use crate::lite_store::LiteStore;
use crate::traits::QueryableStore;
use corvia_common::agent_types::{AgentRecord, AgentStatus, MergeQueueEntry, SessionRecord};
use corvia_common::config::CorviaConfig;
use corvia_common::errors::{CorviaError, Result};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// System status
// ---------------------------------------------------------------------------

/// System status snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemStatus {
    pub entry_count: u64,
    pub active_agents: usize,
    pub open_sessions: usize,
    pub merge_queue_depth: u64,
    pub scope_id: String,
}

/// Gather a point-in-time system status snapshot.
pub async fn system_status(
    store: Arc<dyn QueryableStore>,
    coordinator: &AgentCoordinator,
    scope_id: &str,
) -> Result<SystemStatus> {
    let entry_count = store.count(scope_id).await?;
    let active_agents = coordinator.registry.list_active()?.len();
    let open_sessions = coordinator.sessions.list_open()?.len();
    let merge_queue_depth = coordinator.merge_queue.depth()?;

    Ok(SystemStatus {
        entry_count,
        active_agents,
        open_sessions,
        merge_queue_depth,
        scope_id: scope_id.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Agent operations
// ---------------------------------------------------------------------------

/// List all registered agents.
pub fn agents_list(coordinator: &AgentCoordinator) -> Result<Vec<AgentRecord>> {
    coordinator.registry.list_all()
}

/// Suspend an agent by setting its status to `Suspended`.
pub fn agent_suspend(coordinator: &AgentCoordinator, agent_id: &str) -> Result<()> {
    coordinator.registry.set_status(agent_id, AgentStatus::Suspended)
}

// ---------------------------------------------------------------------------
// Session operations
// ---------------------------------------------------------------------------

/// List all sessions for a given agent.
pub fn sessions_list(coordinator: &AgentCoordinator, agent_id: &str) -> Result<Vec<SessionRecord>> {
    coordinator.sessions.list_by_agent(agent_id)
}

// ---------------------------------------------------------------------------
// Merge queue
// ---------------------------------------------------------------------------

/// Merge queue status snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MergeQueueStatus {
    pub depth: u64,
    pub entries: Vec<MergeQueueEntry>,
}

/// Return current merge queue depth and entries (up to `limit`).
pub fn merge_queue_status(
    coordinator: &AgentCoordinator,
    limit: usize,
) -> Result<MergeQueueStatus> {
    let depth = coordinator.merge_queue.depth()?;
    let entries = coordinator.merge_queue.list(limit)?;
    Ok(MergeQueueStatus { depth, entries })
}

/// Retry failed merge queue entries by resetting their retry count and
/// clearing the last error, so the merge worker picks them up again.
/// Returns the number of entries successfully reset.
pub fn merge_retry(
    coordinator: &AgentCoordinator,
    entry_ids: &[uuid::Uuid],
) -> Result<usize> {
    let mut retried = 0usize;
    for id in entry_ids {
        if coordinator.merge_queue.reset_retry(id)? {
            retried += 1;
        }
    }
    Ok(retried)
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/// Discover available adapter binaries.
pub fn adapters_list(extra_dirs: &[String]) -> Vec<DiscoveredAdapter> {
    discover_adapters(extra_dirs)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Config sections that require a server restart (not hot-reloadable).
const RESTART_REQUIRED_SECTIONS: &[&str] = &["storage", "server", "embedding", "project", "telemetry"];

/// Config sections that are hot-reloadable at runtime.
const HOT_RELOADABLE_SECTIONS: &[&str] = &["agent_lifecycle", "merge", "rag", "chunking", "reasoning", "adapters", "inference", "dashboard"];

/// All valid config sections.
fn all_sections() -> Vec<&'static str> {
    RESTART_REQUIRED_SECTIONS.iter().chain(HOT_RELOADABLE_SECTIONS.iter()).copied().collect()
}

/// Return the config as JSON, optionally filtered to a single section.
pub fn config_get(config: &CorviaConfig, section: Option<&str>) -> Result<serde_json::Value> {
    let full = serde_json::to_value(config)
        .map_err(|e| CorviaError::Config(format!("Failed to serialize config: {e}")))?;

    match section {
        None => Ok(full),
        Some(s) => {
            let all = all_sections();
            if !all.contains(&s) {
                return Err(CorviaError::Config(format!(
                    "Unknown config section '{s}'. Valid sections: {}",
                    all.join(", ")
                )));
            }
            full.get(s)
                .cloned()
                .ok_or_else(|| CorviaError::Config(format!("Config section '{s}' not found")))
        }
    }
}

/// Set a config value and persist to disk.
///
/// Only hot-reloadable sections can be updated at runtime. Non-hot-reloadable
/// sections (storage, server, embedding, project, telemetry) are rejected
/// with an error requiring a server restart.
pub fn config_set(
    config_path: &std::path::Path,
    config: &mut CorviaConfig,
    section: &str,
    key: &str,
    value: serde_json::Value,
) -> Result<CorviaConfig> {
    if RESTART_REQUIRED_SECTIONS.contains(&section) {
        return Err(CorviaError::Config(format!(
            "Section '{section}' is not hot-reloadable; requires server restart"
        )));
    }
    let all = all_sections();
    if !all.contains(&section) {
        return Err(CorviaError::Config(format!(
            "Unknown config section '{section}'. Valid sections: {}",
            all.join(", ")
        )));
    }

    // Read the existing TOML file (or start with a serialized version of current config)
    let toml_str = std::fs::read_to_string(config_path)
        .map_err(|e| CorviaError::Config(format!("Failed to read config file: {e}")))?;

    let mut toml_value: toml::Value = toml::from_str(&toml_str)
        .map_err(|e| CorviaError::Config(format!("Failed to parse config TOML: {e}")))?;

    // Navigate to section.key and update
    let section_table = toml_value
        .as_table_mut()
        .ok_or_else(|| CorviaError::Config("Config root is not a table".into()))?
        .entry(section)
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));

    let section_table = section_table
        .as_table_mut()
        .ok_or_else(|| CorviaError::Config(format!("Config section '{section}' is not a table")))?;

    // Convert JSON value to TOML value
    let toml_val = json_to_toml(&value)?;
    section_table.insert(key.to_string(), toml_val);

    // Write back
    let updated_toml = toml::to_string_pretty(&toml_value)
        .map_err(|e| CorviaError::Config(format!("Failed to serialize config: {e}")))?;
    std::fs::write(config_path, &updated_toml)
        .map_err(|e| CorviaError::Config(format!("Failed to write config file: {e}")))?;

    // Re-parse to get the updated CorviaConfig
    let updated_config: CorviaConfig = toml::from_str(&updated_toml)
        .map_err(|e| CorviaError::Config(format!("Failed to re-parse updated config: {e}")))?;

    *config = updated_config.clone();
    Ok(updated_config)
}

/// Convert a serde_json::Value to a toml::Value.
fn json_to_toml(json: &serde_json::Value) -> Result<toml::Value> {
    match json {
        serde_json::Value::Null => Err(CorviaError::Config("Cannot set null value in TOML config".into())),
        serde_json::Value::Bool(b) => Ok(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(toml::Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(toml::Value::Float(f))
            } else {
                Err(CorviaError::Config("Unsupported number type".into()))
            }
        }
        serde_json::Value::String(s) => Ok(toml::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: std::result::Result<Vec<toml::Value>, _> =
                arr.iter().map(json_to_toml).collect();
            Ok(toml::Value::Array(items?))
        }
        serde_json::Value::Object(map) => {
            let mut table = toml::map::Map::new();
            for (k, v) in map {
                table.insert(k.clone(), json_to_toml(v)?);
            }
            Ok(toml::Value::Table(table))
        }
    }
}

// ---------------------------------------------------------------------------
// GC
// ---------------------------------------------------------------------------

/// Run garbage collection sweep with timing.
pub async fn gc_run(coordinator: &AgentCoordinator) -> Result<GcReport> {
    let start = std::time::Instant::now();
    let mut report = coordinator.gc().await?;
    report.duration_ms = start.elapsed().as_millis() as u64;
    if report.started_at.is_empty() {
        report.started_at = chrono::Utc::now().to_rfc3339();
    }
    Ok(report)
}

/// In-memory ring buffer of recent GC reports.
pub struct GcHistory {
    reports: std::sync::Mutex<std::collections::VecDeque<GcReport>>,
    max_size: usize,
}

impl GcHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            reports: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn push(&self, report: GcReport) {
        let mut reports = self.reports.lock().unwrap();
        if reports.len() >= self.max_size {
            reports.pop_front();
        }
        reports.push_back(report);
    }

    pub fn last(&self) -> Option<GcReport> {
        self.reports.lock().unwrap().back().cloned()
    }

    pub fn all(&self) -> Vec<GcReport> {
        self.reports.lock().unwrap().iter().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Index rebuild
// ---------------------------------------------------------------------------

/// Rebuild the HNSW index from knowledge JSON files on disk.
/// Returns the number of entries re-indexed.
///
/// Accepts an existing `&LiteStore` reference to avoid opening a second
/// Redb database (which would fail with an exclusive-lock error when the
/// server already holds the file lock).
pub fn rebuild_index(store: &LiteStore) -> Result<usize> {
    store.rebuild_from_files()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::traits::{GenerationEngine, InferenceEngine};
    use corvia_common::config::{AgentLifecycleConfig, MergeConfig};

    struct MockEngine;
    #[async_trait::async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize {
            3
        }
    }

    struct MockGenEngine;
    #[async_trait::async_trait]
    impl GenerationEngine for MockGenEngine {
        fn name(&self) -> &str {
            "mock"
        }
        async fn generate(
            &self,
            _system_prompt: &str,
            user_message: &str,
        ) -> Result<crate::traits::GenerationResult> {
            Ok(crate::traits::GenerationResult {
                text: format!("merged: {user_message}"),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize {
            4096
        }
    }

    async fn setup_coordinator(
        dir: &std::path::Path,
    ) -> (Arc<dyn QueryableStore>, AgentCoordinator) {
        let store = Arc::new(LiteStore::open(dir, 3).unwrap()) as Arc<dyn QueryableStore>;
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        store.init_schema().await.unwrap();
        let gen_engine = Arc::new(MockGenEngine) as Arc<dyn GenerationEngine>;

        let coord = AgentCoordinator::new(
            store.clone(),
            engine,
            dir,
            AgentLifecycleConfig::default(),
            MergeConfig::default(),
            gen_engine,
        )
        .unwrap();

        (store, coord)
    }

    #[tokio::test]
    async fn test_system_status_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (store, coord) = setup_coordinator(dir.path()).await;

        let status = system_status(store, &coord, "test").await.unwrap();
        assert_eq!(status.entry_count, 0);
        assert_eq!(status.active_agents, 0);
        assert_eq!(status.open_sessions, 0);
        assert_eq!(status.merge_queue_depth, 0);
        assert_eq!(status.scope_id, "test");
    }

    #[tokio::test]
    async fn test_agents_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (_store, coord) = setup_coordinator(dir.path()).await;

        let agents = agents_list(&coord).unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_merge_queue_status_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (_store, coord) = setup_coordinator(dir.path()).await;

        let status = merge_queue_status(&coord, 10).unwrap();
        assert_eq!(status.depth, 0);
        assert!(status.entries.is_empty());
    }

    #[test]
    fn test_config_get_full() {
        let config = CorviaConfig::default();
        let result = config_get(&config, None).unwrap();
        assert!(result.is_object());
        assert!(result.get("storage").is_some());
        assert!(result.get("embedding").is_some());
    }

    #[test]
    fn test_config_get_section() {
        let config = CorviaConfig::default();
        let result = config_get(&config, Some("storage")).unwrap();
        assert!(result.is_object());
    }

    #[test]
    fn test_config_get_invalid_section() {
        let config = CorviaConfig::default();
        let result = config_get(&config, Some("nonexistent"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown config section"));
    }

    #[test]
    fn test_config_get_hot_reloadable_section() {
        let config = CorviaConfig::default();
        // "rag" is a hot-reloadable section — should be queryable
        let result = config_get(&config, Some("rag"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_set_rejects_restart_required() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("corvia.toml");
        let mut config = CorviaConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        std::fs::write(&config_path, &toml_str).unwrap();

        let result = config_set(
            &config_path, &mut config, "storage", "data_dir",
            serde_json::json!("/new/path"),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not hot-reloadable"));
    }

    #[test]
    fn test_config_set_rejects_null_value() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("corvia.toml");
        let mut config = CorviaConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        std::fs::write(&config_path, &toml_str).unwrap();

        let result = config_set(
            &config_path, &mut config, "rag", "default_limit",
            serde_json::Value::Null,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("null"));
    }

    #[tokio::test]
    async fn test_merge_retry_resets_failed_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (_store, coord) = setup_coordinator(dir.path()).await;

        // Enqueue an entry and mark it failed
        let id = uuid::Uuid::now_v7();
        coord.merge_queue.enqueue(id, "test::agent", "sess", "scope").unwrap();
        coord.merge_queue.mark_failed(&id, "Ollama down").unwrap();

        // Verify it's failed
        let entries = coord.merge_queue.list(10).unwrap();
        assert_eq!(entries[0].retry_count, 1);
        assert!(entries[0].last_error.is_some());

        // Retry it
        let count = merge_retry(&coord, &[id]).unwrap();
        assert_eq!(count, 1);

        // Verify reset
        let entries = coord.merge_queue.list(10).unwrap();
        assert_eq!(entries[0].retry_count, 0);
        assert!(entries[0].last_error.is_none());
    }

    #[tokio::test]
    async fn test_merge_retry_nonexistent_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (_store, coord) = setup_coordinator(dir.path()).await;

        let id = uuid::Uuid::now_v7();
        let count = merge_retry(&coord, &[id]).unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_gc_run_populates_timing() {
        let dir = tempfile::tempdir().unwrap();
        let (_store, coord) = setup_coordinator(dir.path()).await;

        let report = gc_run(&coord).await.unwrap();
        assert!(!report.started_at.is_empty(), "gc_run should populate started_at");
        // duration_ms may be 0 for sub-millisecond GC on empty coordinator,
        // but it must not be absurdly large
        assert!(report.duration_ms < 10_000, "gc_run duration should be reasonable");
    }

    #[test]
    fn test_gc_history_ring_buffer() {
        let history = GcHistory::new(3);
        assert!(history.last().is_none());
        assert!(history.all().is_empty());

        for i in 0..5 {
            history.push(GcReport {
                orphans_rolled_back: i,
                duration_ms: i as u64 * 10,
                ..Default::default()
            });
        }

        let all = history.all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].orphans_rolled_back, 2);
        assert_eq!(all[2].orphans_rolled_back, 4);
        assert_eq!(history.last().unwrap().orphans_rolled_back, 4);
    }
}
