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

/// Per-scope tier distribution counts.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TierDistribution {
    pub hot: u64,
    pub warm: u64,
    pub cold: u64,
    pub forgotten: u64,
}

/// System status snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemStatus {
    pub entry_count: u64,
    pub active_agents: usize,
    pub open_sessions: usize,
    pub merge_queue_depth: u64,
    pub scope_id: String,
    /// Tier distribution for the scope.
    pub tier_distribution: TierDistribution,
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

    // Compute tier distribution (lightweight — avoids full entry deserialization)
    let tier_distribution = if let Some(lite_store) = store.as_any().downcast_ref::<LiteStore>() {
        lite_store.count_tiers_by_scope(scope_id).unwrap_or_default()
    } else {
        TierDistribution::default()
    };

    Ok(SystemStatus {
        entry_count,
        active_agents,
        open_sessions,
        merge_queue_depth,
        scope_id: scope_id.to_string(),
        tier_distribution,
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

    // Handle dotted keys (e.g., "pipeline.searchers") by navigating/creating
    // nested tables instead of inserting a flat key with dots.
    let key_parts: Vec<&str> = key.split('.').collect();
    if key_parts.len() == 1 {
        section_table.insert(key.to_string(), toml_val);
    } else {
        let mut current = section_table;
        for part in &key_parts[..key_parts.len() - 1] {
            current = current
                .entry(part.to_string())
                .or_insert_with(|| toml::Value::Table(toml::map::Map::new()))
                .as_table_mut()
                .ok_or_else(|| {
                    CorviaError::Config(format!("Config key '{part}' is not a table"))
                })?;
        }
        let leaf_key = key_parts[key_parts.len() - 1];
        current.insert(leaf_key.to_string(), toml_val);
    }

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

/// Run garbage collection sweep with timing (session-level GC).
pub async fn gc_run(coordinator: &AgentCoordinator) -> Result<GcReport> {
    let start = std::time::Instant::now();
    let mut report = coordinator.gc().await?;
    report.duration_ms = start.elapsed().as_millis() as u64;
    if report.started_at.is_empty() {
        report.started_at = chrono::Utc::now().to_rfc3339();
    }
    Ok(report)
}

/// Run knowledge-level GC: retention scoring + tier transitions.
///
/// Triggers a single GC cycle immediately (outside the periodic worker loop).
/// Used by the MCP `corvia_gc_run` tool for on-demand GC.
pub async fn gc_knowledge_run(
    store: &std::sync::Arc<dyn QueryableStore>,
    graph: &std::sync::Arc<dyn crate::traits::GraphStore>,
    data_dir: &std::path::Path,
    config: &corvia_common::config::CorviaConfig,
) -> Result<crate::gc_worker::GcCycleReport> {
    let forgetting = config.forgetting.as_ref();
    let scope_configs: std::collections::HashMap<String, Option<corvia_common::config::ScopeForgettingOverride>> =
        config
            .scope
            .as_ref()
            .map(|scopes| {
                scopes
                    .iter()
                    .map(|s| (s.id.clone(), s.forgetting.clone()))
                    .collect()
            })
            .unwrap_or_default();

    crate::gc_worker::run_gc_cycle(store, graph, data_dir, forgetting, &scope_configs, None).await
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
// Pin / Unpin / Inspect
// ---------------------------------------------------------------------------

/// Pin result returned to callers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PinResult {
    pub entry_id: String,
    pub pinned_by: String,
    pub pinned_at: String,
}

/// Pin an entry by ID, making it exempt from GC demotion.
pub async fn pin_entry(
    store: &Arc<dyn QueryableStore>,
    entry_id: &uuid::Uuid,
    agent_id: &str,
) -> Result<PinResult> {
    let lite_store = store
        .as_any()
        .downcast_ref::<LiteStore>()
        .ok_or_else(|| CorviaError::Storage("Pin is only supported for LiteStore".into()))?;

    let mut entry = lite_store
        .get(entry_id)
        .await?
        .ok_or_else(|| CorviaError::Storage(format!("Entry {entry_id} not found")))?;

    let now = chrono::Utc::now();
    entry.pin = Some(corvia_common::types::PinInfo {
        by: agent_id.to_string(),
        at: now,
    });

    lite_store.update_entry_metadata(&entry)?;

    Ok(PinResult {
        entry_id: entry_id.to_string(),
        pinned_by: agent_id.to_string(),
        pinned_at: now.to_rfc3339(),
    })
}

/// Unpin result returned to callers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UnpinResult {
    pub entry_id: String,
    pub was_pinned: bool,
}

/// Unpin an entry by ID, making it eligible for GC demotion again.
pub async fn unpin_entry(
    store: &Arc<dyn QueryableStore>,
    entry_id: &uuid::Uuid,
) -> Result<UnpinResult> {
    let lite_store = store
        .as_any()
        .downcast_ref::<LiteStore>()
        .ok_or_else(|| CorviaError::Storage("Unpin is only supported for LiteStore".into()))?;

    let mut entry = lite_store
        .get(entry_id)
        .await?
        .ok_or_else(|| CorviaError::Storage(format!("Entry {entry_id} not found")))?;

    let was_pinned = entry.pin.is_some();
    entry.pin = None;

    lite_store.update_entry_metadata(&entry)?;

    Ok(UnpinResult {
        entry_id: entry_id.to_string(),
        was_pinned,
    })
}

/// Full lifecycle metadata for an entry, used by the `inspect` command.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EntryInspection {
    pub id: String,
    pub scope_id: String,
    pub content_preview: String,
    pub tier: String,
    pub tier_changed_at: Option<String>,
    pub retention_score: Option<f32>,
    pub memory_type: String,
    pub confidence: Option<f32>,
    pub access_count: u32,
    pub last_accessed: Option<String>,
    pub pin: Option<PinInspection>,
    pub recorded_at: String,
    pub valid_from: String,
    pub valid_to: Option<String>,
    pub superseded_by: Option<String>,
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
    pub source_file: Option<String>,
    pub inbound_edges: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PinInspection {
    pub by: String,
    pub at: String,
}

/// Inspect an entry: return full lifecycle metadata with score breakdown.
pub async fn inspect_entry(
    store: &Arc<dyn QueryableStore>,
    graph: &Arc<dyn crate::traits::GraphStore>,
    entry_id: &uuid::Uuid,
) -> Result<EntryInspection> {
    let entry = store
        .get(entry_id)
        .await?
        .ok_or_else(|| CorviaError::Storage(format!("Entry {entry_id} not found")))?;

    let inbound_edges = graph
        .edges(entry_id, corvia_common::types::EdgeDirection::Incoming)
        .await
        .unwrap_or_default()
        .len();

    let mut chars = entry.content.chars();
    let content_preview: String = (&mut chars).take(200).collect();
    let ellipsis = if chars.next().is_some() { "..." } else { "" };

    Ok(EntryInspection {
        id: entry.id.to_string(),
        scope_id: entry.scope_id.clone(),
        content_preview: format!("{content_preview}{ellipsis}"),
        tier: entry.tier.to_string(),
        tier_changed_at: entry.tier_changed_at.map(|t| t.to_rfc3339()),
        retention_score: entry.retention_score,
        memory_type: entry.memory_type.to_string(),
        confidence: entry.confidence,
        access_count: entry.access_count,
        last_accessed: entry.last_accessed.map(|t| t.to_rfc3339()),
        pin: entry.pin.as_ref().map(|p| PinInspection {
            by: p.by.clone(),
            at: p.at.to_rfc3339(),
        }),
        recorded_at: entry.recorded_at.to_rfc3339(),
        valid_from: entry.valid_from.to_rfc3339(),
        valid_to: entry.valid_to.map(|t| t.to_rfc3339()),
        superseded_by: entry.superseded_by.map(|u| u.to_string()),
        content_role: entry.metadata.content_role.clone(),
        source_origin: entry.metadata.source_origin.clone(),
        source_file: entry.metadata.source_file.clone(),
        inbound_edges,
    })
}

/// GC knowledge cycle history: wraps the in-memory GcCycleReport ring buffer.
pub struct GcKnowledgeHistory {
    reports: std::sync::Mutex<std::collections::VecDeque<crate::gc_worker::GcCycleReport>>,
    max_size: usize,
}

impl GcKnowledgeHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            reports: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn push(&self, report: crate::gc_worker::GcCycleReport) {
        let mut reports = self.reports.lock().unwrap();
        if reports.len() >= self.max_size {
            reports.pop_front();
        }
        reports.push_back(report);
    }

    pub fn all(&self) -> Vec<crate::gc_worker::GcCycleReport> {
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
        assert_eq!(status.tier_distribution.hot, 0);
        assert_eq!(status.tier_distribution.warm, 0);
        assert_eq!(status.tier_distribution.cold, 0);
        assert_eq!(status.tier_distribution.forgotten, 0);
    }

    #[tokio::test]
    async fn test_system_status_tier_distribution_with_entries() {
        let dir = tempfile::tempdir().unwrap();
        let (store, coord) = setup_coordinator(dir.path()).await;

        // Insert entries with different tiers
        let mut hot_entry = corvia_common::types::KnowledgeEntry::new(
            "hot entry".into(), "test-scope".into(), "v1".into(),
        );
        hot_entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        hot_entry.tier = corvia_common::types::Tier::Hot;
        store.insert(&hot_entry).await.unwrap();

        let mut warm_entry = corvia_common::types::KnowledgeEntry::new(
            "warm entry".into(), "test-scope".into(), "v1".into(),
        );
        warm_entry.embedding = Some(vec![0.0, 1.0, 0.0]);
        warm_entry.tier = corvia_common::types::Tier::Warm;
        store.insert(&warm_entry).await.unwrap();

        let mut cold_entry = corvia_common::types::KnowledgeEntry::new(
            "cold entry".into(), "test-scope".into(), "v1".into(),
        );
        cold_entry.embedding = Some(vec![0.0, 0.0, 1.0]);
        cold_entry.tier = corvia_common::types::Tier::Cold;
        store.insert(&cold_entry).await.unwrap();

        // Different scope — should not be counted
        let mut other_scope = corvia_common::types::KnowledgeEntry::new(
            "other scope".into(), "other-scope".into(), "v1".into(),
        );
        other_scope.embedding = Some(vec![1.0, 1.0, 0.0]);
        other_scope.tier = corvia_common::types::Tier::Hot;
        store.insert(&other_scope).await.unwrap();

        let status = system_status(store, &coord, "test-scope").await.unwrap();
        assert_eq!(status.tier_distribution.hot, 1);
        assert_eq!(status.tier_distribution.warm, 1);
        assert_eq!(status.tier_distribution.cold, 1);
        assert_eq!(status.tier_distribution.forgotten, 0);
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

    #[tokio::test]
    async fn test_pin_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _coord) = setup_coordinator(dir.path()).await;

        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "test pin".into(), "test-scope".into(), "v1".into(),
        );
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        let entry_id = entry.id;
        store.insert(&entry).await.unwrap();

        let result = pin_entry(&store, &entry_id, "claude-code").await.unwrap();
        assert_eq!(result.pinned_by, "claude-code");

        // Verify entry is now pinned
        let updated = store.get(&entry_id).await.unwrap().unwrap();
        assert!(updated.pin.is_some());
        assert_eq!(updated.pin.as_ref().unwrap().by, "claude-code");

        // Re-pin with different agent (idempotent, overwrites)
        let result2 = pin_entry(&store, &entry_id, "other-agent").await.unwrap();
        assert_eq!(result2.pinned_by, "other-agent");
        let updated2 = store.get(&entry_id).await.unwrap().unwrap();
        assert_eq!(updated2.pin.as_ref().unwrap().by, "other-agent");
    }

    #[tokio::test]
    async fn test_unpin_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _coord) = setup_coordinator(dir.path()).await;

        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "test unpin".into(), "test-scope".into(), "v1".into(),
        );
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        entry.pin = Some(corvia_common::types::PinInfo {
            by: "agent".into(),
            at: chrono::Utc::now(),
        });
        let entry_id = entry.id;
        store.insert(&entry).await.unwrap();

        let result = unpin_entry(&store, &entry_id).await.unwrap();
        assert!(result.was_pinned);

        // Verify entry is unpinned
        let updated = store.get(&entry_id).await.unwrap().unwrap();
        assert!(updated.pin.is_none());
    }

    #[tokio::test]
    async fn test_unpin_entry_not_pinned() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _coord) = setup_coordinator(dir.path()).await;

        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "not pinned".into(), "test-scope".into(), "v1".into(),
        );
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        let entry_id = entry.id;
        store.insert(&entry).await.unwrap();

        let result = unpin_entry(&store, &entry_id).await.unwrap();
        assert!(!result.was_pinned);
    }

    struct MockGraphStore;
    #[async_trait::async_trait]
    impl crate::traits::GraphStore for MockGraphStore {
        async fn relate(&self, _from: &uuid::Uuid, _relation: &str, _to: &uuid::Uuid, _metadata: Option<serde_json::Value>) -> Result<()> { Ok(()) }
        async fn edges(&self, _id: &uuid::Uuid, _direction: corvia_common::types::EdgeDirection) -> Result<Vec<corvia_common::types::GraphEdge>> { Ok(vec![]) }
        async fn traverse(&self, _start: &uuid::Uuid, _relation: Option<&str>, _direction: corvia_common::types::EdgeDirection, _max_depth: usize) -> Result<Vec<corvia_common::types::KnowledgeEntry>> { Ok(vec![]) }
        async fn shortest_path(&self, _from: &uuid::Uuid, _to: &uuid::Uuid) -> Result<Option<Vec<corvia_common::types::KnowledgeEntry>>> { Ok(None) }
        async fn remove_edges(&self, _id: &uuid::Uuid) -> Result<()> { Ok(()) }
    }

    #[tokio::test]
    async fn test_inspect_entry() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap()) as Arc<dyn QueryableStore>;
        store.init_schema().await.unwrap();
        let graph = Arc::new(MockGraphStore) as Arc<dyn crate::traits::GraphStore>;

        let mut entry = corvia_common::types::KnowledgeEntry::new(
            "test inspect content".into(), "test-scope".into(), "v1".into(),
        );
        entry.embedding = Some(vec![1.0, 0.0, 0.0]);
        entry.tier = corvia_common::types::Tier::Warm;
        entry.retention_score = Some(0.42);
        entry.access_count = 5;
        let entry_id = entry.id;
        store.insert(&entry).await.unwrap();

        let inspection = inspect_entry(&store, &graph, &entry_id).await.unwrap();
        assert_eq!(inspection.tier, "warm");
        assert_eq!(inspection.retention_score, Some(0.42));
        assert_eq!(inspection.access_count, 5);
        assert_eq!(inspection.memory_type, "episodic"); // default
        assert!(inspection.content_preview.starts_with("test inspect content"));
        assert!(inspection.pin.is_none());
    }

    #[tokio::test]
    async fn test_pin_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _coord) = setup_coordinator(dir.path()).await;

        let fake_id = uuid::Uuid::now_v7();
        let result = pin_entry(&store, &fake_id, "agent").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_gc_knowledge_history_ring_buffer() {
        let history = GcKnowledgeHistory::new(3);
        assert!(history.all().is_empty());

        for _ in 0..5 {
            history.push(crate::gc_worker::GcCycleReport::default());
        }

        assert_eq!(history.all().len(), 3);
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
