//! Shared response types for the dashboard REST API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Service health state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceState {
    Healthy,
    Unhealthy,
    Starting,
    Stopped,
}

/// Individual service status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub state: ServiceState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

/// Span timing statistics (mirrors Python SpanStats)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SpanStats {
    pub count: u64,
    pub count_1h: u64,
    pub avg_ms: f64,
    pub last_ms: f64,
    pub errors: u64,
    #[serde(default)]
    pub p50_ms: f64,
    #[serde(default)]
    pub p95_ms: f64,
    #[serde(default)]
    pub p99_ms: f64,
}

/// A structured trace event (mirrors Python TraceEvent)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEvent {
    pub ts: String,
    pub level: String,
    pub module: String,
    pub msg: String,
}

/// Aggregated trace data (mirrors Python TracesData)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TracesData {
    pub spans: HashMap<String, SpanStats>,
    pub recent_events: Vec<TraceEvent>,
}

/// Dashboard config summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub embedding_provider: String,
    pub merge_provider: String,
    pub storage: String,
    pub workspace: String,
}

/// GET /api/dashboard/status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatusResponse {
    pub services: Vec<ServiceStatus>,
    pub entry_count: u64,
    pub agent_count: usize,
    pub merge_queue_depth: u64,
    pub session_count: usize,
    pub config: DashboardConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traces: Option<TracesData>,
    /// Coverage ratio: HNSW entries / knowledge files on disk (0.0-1.0).
    /// null when disk_count == 0 (fresh workspace).
    pub index_coverage: Option<f64>,
    /// true when coverage < threshold. null when coverage is null.
    pub index_stale: Option<bool>,
    /// Knowledge JSON files on disk for the default scope.
    pub index_disk_count: u64,
    /// Entries in Redb SCOPE_INDEX for the default scope.
    pub index_store_count: u64,
    /// Entries in Redb HNSW_TO_UUID table.
    pub index_hnsw_count: u64,
    /// Configured staleness threshold (0.0-1.0).
    pub index_stale_threshold: f64,
    /// ISO 8601 timestamp of last coverage computation.
    pub index_coverage_checked_at: Option<String>,
    /// Total spoke containers (running + exited).
    #[serde(default)]
    pub spoke_count: usize,
    /// Currently running spoke containers.
    #[serde(default)]
    pub spokes_running: usize,
}

/// Coverage snapshot returned by both GET /api/dashboard/status (embedded)
/// and POST /api/dashboard/status/refresh-coverage (standalone).
/// Shared struct ensures both endpoints stay in sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageResponse {
    /// Coverage ratio: HNSW entries / knowledge files on disk (0.0-1.0).
    /// null when no knowledge files exist on disk.
    pub index_coverage: Option<f64>,
    /// true when coverage < threshold. null when coverage is null.
    pub index_stale: Option<bool>,
    /// Knowledge JSON files on disk for the default scope.
    pub index_disk_count: u64,
    /// Entries in Redb SCOPE_INDEX for the default scope.
    pub index_store_count: u64,
    /// Entries in Redb HNSW_TO_UUID table.
    pub index_hnsw_count: u64,
    /// Configured staleness threshold (0.0-1.0).
    pub index_stale_threshold: f64,
    /// ISO 8601 timestamp of last coverage computation.
    pub index_coverage_checked_at: Option<String>,
}

/// A single structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub module: String,
    pub message: String,
}

/// GET /api/dashboard/logs response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub entries: Vec<LogEntry>,
    pub total: usize,
}

/// Pre-aggregated module-level statistics (computed server-side to reduce client work)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleStats {
    pub count: u64,
    pub count_1h: u64,
    pub avg_ms: f64,
    pub errors: u64,
    pub span_count: u32,
}

/// GET /api/dashboard/traces response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracesResponse {
    pub spans: HashMap<String, SpanStats>,
    pub recent_events: Vec<TraceEvent>,
    /// Pre-aggregated per-module stats (agent, entry, merge, storage, rag, inference, gc)
    #[serde(default)]
    pub modules: HashMap<String, ModuleStats>,
}

/// GC report DTO (mirrors corvia-kernel GcReport for API responses)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcReportDto {
    pub orphans_rolled_back: usize,
    pub duration_ms: u64,
    pub stale_transitioned: usize,
    pub closed_sessions_cleaned: usize,
    pub agents_suspended: usize,
    pub entries_deduplicated: usize,
    pub started_at: String,
}

/// GET /api/dashboard/gc response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcStatusResponse {
    pub last_run: Option<GcReportDto>,
    pub history: Vec<GcReportDto>,
    pub scheduled: bool,
}

// ---------------------------------------------------------------------------
// Tiered knowledge dashboard types
// ---------------------------------------------------------------------------

/// GET /api/dashboard/tiers response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiersResponse {
    pub hot: u64,
    pub warm: u64,
    pub cold: u64,
    pub forgotten: u64,
    pub total: u64,
    pub forgetting_enabled: bool,
}

/// A single tier transition event (extracted from structured trace logs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransitionDto {
    pub entry_id: String,
    pub scope_id: String,
    pub from_tier: String,
    pub to_tier: String,
    pub retention_score: f64,
    pub reason: String,
    pub timestamp: String,
}

/// GET /api/dashboard/tiers/transitions response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransitionsResponse {
    pub transitions: Vec<TierTransitionDto>,
    pub count: usize,
}

/// A pinned entry summary for dashboard display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedEntryDto {
    pub entry_id: String,
    pub content_preview: String,
    pub pinned_by: String,
    pub pinned_at: String,
    pub scope_id: String,
}

/// GET /api/dashboard/tiers/pinned response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedEntriesResponse {
    pub entries: Vec<PinnedEntryDto>,
    pub count: usize,
}

/// Knowledge GC cycle report DTO (distinct from session-level GcReportDto)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcCycleReportDto {
    pub entries_scanned: usize,
    pub entries_scored: usize,
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub cold_to_forgotten: usize,
    pub warm_to_hot: usize,
    pub cold_to_warm: usize,
    pub hnsw_rebuild_triggered: bool,
    pub rebuild_duration_ms: u64,
    pub cycle_duration_ms: u64,
    pub scopes_processed: usize,
    pub chain_protected: usize,
    pub auto_protected: usize,
    pub budget_demoted: usize,
    pub circuit_breaker_tripped: bool,
}

/// GET /api/dashboard/tiers/gc-history response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcCycleHistoryResponse {
    pub cycles: Vec<GcCycleReportDto>,
    pub count: usize,
}

/// Live session entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSession {
    pub session_id: String,
    pub agent_id: String,
    pub agent_name: String,
    pub state: String,
    pub started_at: String,
    pub duration_secs: u64,
    pub entries_written: u64,
    pub entries_merged: u64,
    pub pending_entries: u64,
    pub git_branch: Option<String>,
    pub has_staging_dir: bool,
}

/// Live sessions summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionsSummary {
    pub total_active: usize,
    pub total_stale: usize,
    pub total_entries_pending: u64,
}

/// GET /api/dashboard/sessions/live response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionsResponse {
    pub sessions: Vec<LiveSession>,
    pub summary: LiveSessionsSummary,
}

// ---------------------------------------------------------------------------
// Hook-observed Claude sessions (file watcher)
// ---------------------------------------------------------------------------

/// State of a Claude session observed via JSONL file watching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HookSessionState {
    Active,
    Stale,
    Ended,
}

/// A Claude session observed via JSONL file watching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSession {
    pub session_id: String,
    pub state: HookSessionState,
    pub workspace: String,
    pub git_branch: String,
    pub agent_type: String,
    pub parent_session_id: Option<String>,
    pub corvia_agent_id: Option<String>,
    pub started_at: String,
    pub last_activity: String,
    pub duration_secs: u64,
    pub turn_count: u32,
    pub tool_calls: u32,
    pub active_tool: Option<String>,
    pub tools_used: Vec<String>,
}

/// SSE update delta for a hook session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSessionUpdate {
    pub session_id: String,
    pub event_type: String,
    pub session: Option<HookSession>,
}

/// GET /api/dashboard/sessions/hook response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookSessionsResponse {
    pub sessions: Vec<HookSession>,
    pub total_active: usize,
    pub total_stale: usize,
}

// ---------------------------------------------------------------------------
// Trace spans
// ---------------------------------------------------------------------------

/// A node in a span trace tree.
/// `module` is derived via span_to_module() for waterfall UI color-coding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanNode {
    pub span_id: String,
    pub parent_span_id: String,
    pub trace_id: String,
    pub span_name: String,
    pub elapsed_ms: f64,
    pub start_offset_ms: f64,
    pub depth: usize,
    pub module: String,
    pub fields: serde_json::Value,
    pub children: Vec<SpanNode>,
}

/// A complete trace tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceTree {
    pub trace_id: String,
    pub root_span: String,
    pub total_ms: f64,
    pub span_count: usize,
    pub started_at: String,
    pub spans: Vec<SpanNode>,
}

/// GET /api/dashboard/traces/recent response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentTracesResponse {
    pub traces: Vec<TraceTree>,
}

/// A single process using the GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub memory_used_mb: u64,
}

/// A single detected GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vendor: String,
    /// GPU core utilization percentage (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utilization_pct: Option<u32>,
    /// GPU memory currently used in MB (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_used_mb: Option<u64>,
    /// GPU total memory in MB (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_total_mb: Option<u64>,
    /// GPU temperature in Celsius (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature_c: Option<u32>,
    /// GPU power draw in watts (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_draw_w: Option<f64>,
    /// GPU power limit in watts (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_limit_w: Option<f64>,
    /// Processes using the GPU (NVIDIA only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processes: Option<Vec<GpuProcess>>,
    /// Render engine busy percentage (Intel only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_busy_pct: Option<u32>,
    /// Video engine busy percentage (Intel only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_busy_pct: Option<u32>,
    /// Current GPU frequency in MHz (Intel only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_mhz: Option<u64>,
    /// Maximum GPU frequency in MHz (Intel only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_max_mhz: Option<u64>,
}

/// Inference backend information for a single model type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceBackendInfo {
    pub model: String,
    pub device: String,
    pub backend: String,
}

/// Inference probe health status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceProbeStatus {
    Pending,
    Healthy,
    Degraded,
    Failed,
    Unreachable,
}

/// Inference health status from the canary probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceHealthStatus {
    pub status: InferenceProbeStatus,
    pub ep_name: String,
    pub ep_requested: String,
    pub fallback_used: bool,
    pub baseline_us: u64,
    pub last_probe_us: u64,
    pub drift_pct: f64,
    pub models_loaded: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_probe_at: Option<String>,
}

/// GET /api/dashboard/gpu response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatusResponse {
    pub gpus: Vec<GpuInfo>,
    pub inference_backend: HashMap<String, InferenceBackendInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_health: Option<InferenceHealthStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_state_serializes_lowercase() {
        let status = ServiceStatus {
            name: "corvia-server".to_string(),
            state: ServiceState::Healthy,
            port: Some(8020),
            latency_ms: Some(1.5),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"healthy\""));
        assert!(json.contains("\"corvia-server\""));

        let parsed: ServiceStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.state, ServiceState::Healthy);
    }

    #[test]
    fn span_stats_default_is_zeroed() {
        let stats = SpanStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_ms, 0.0);
    }

    #[test]
    fn span_stats_default_includes_percentiles() {
        let stats = SpanStats::default();
        assert_eq!(stats.p50_ms, 0.0);
        assert_eq!(stats.p95_ms, 0.0);
        assert_eq!(stats.p99_ms, 0.0);
    }

    #[test]
    fn gc_status_response_serializes() {
        let resp = GcStatusResponse {
            last_run: None,
            history: vec![],
            scheduled: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"scheduled\":false"));
        assert!(json.contains("\"last_run\":null"));
    }

    #[test]
    fn status_response_omits_none_traces() {
        let resp = DashboardStatusResponse {
            services: vec![],
            entry_count: 0,
            agent_count: 0,
            merge_queue_depth: 0,
            session_count: 0,
            config: DashboardConfig {
                embedding_provider: "corvia".to_string(),
                merge_provider: "corvia".to_string(),
                storage: "lite".to_string(),
                workspace: "test".to_string(),
            },
            traces: None,
            index_coverage: None,
            index_stale: None,
            index_disk_count: 0,
            index_store_count: 0,
            index_hnsw_count: 0,
            index_stale_threshold: 0.9,
            index_coverage_checked_at: None,
            spoke_count: 0,
            spokes_running: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("traces"));
        // Coverage fields should be present as null
        assert!(json.contains("\"index_coverage\":null"));
        assert!(json.contains("\"index_stale\":null"));
    }

    #[test]
    fn gpu_status_response_serializes() {
        let resp = GpuStatusResponse {
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "NVIDIA GeForce RTX 4090".to_string(),
                    vendor: "nvidia".to_string(),
                    utilization_pct: Some(45),
                    memory_used_mb: Some(2048),
                    memory_total_mb: Some(24576),
                    temperature_c: Some(65),
                    power_draw_w: None,
                    power_limit_w: None,
                    processes: None,
                    render_busy_pct: None,
                    video_busy_pct: None,
                    frequency_mhz: None,
                    frequency_max_mhz: None,
                },
                GpuInfo {
                    index: 1,
                    name: "Intel UHD 770".to_string(),
                    vendor: "intel".to_string(),
                    utilization_pct: None,
                    memory_used_mb: None,
                    memory_total_mb: None,
                    temperature_c: None,
                    power_draw_w: None,
                    power_limit_w: None,
                    processes: None,
                    render_busy_pct: None,
                    video_busy_pct: None,
                    frequency_mhz: Some(1200),
                    frequency_max_mhz: Some(1550),
                },
            ],
            inference_backend: HashMap::from([
                ("embedding".to_string(), InferenceBackendInfo {
                    model: "nomic-embed-text-v1.5".to_string(),
                    device: "cpu".to_string(),
                    backend: "cpu".to_string(),
                }),
            ]),
            inference_health: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"vendor\":\"nvidia\""));
        assert!(json.contains("\"vendor\":\"intel\""));
        assert!(json.contains("\"utilization_pct\":45"));
        assert!(json.contains("\"frequency_mhz\":1200"));
        // Intel GPU should not have NVIDIA-only fields
        assert!(!json.contains("\"utilization_pct\":null"));

        let roundtrip: GpuStatusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.gpus.len(), 2);
        assert_eq!(roundtrip.gpus[0].vendor, "nvidia");
        assert_eq!(roundtrip.gpus[1].vendor, "intel");
    }

    #[test]
    fn gpu_status_empty_gpus() {
        let resp = GpuStatusResponse {
            gpus: vec![],
            inference_backend: HashMap::new(),
            inference_health: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"gpus\":[]"));
    }

    #[test]
    fn test_gpu_status_response_without_inference_health() {
        let resp = GpuStatusResponse {
            gpus: vec![],
            inference_backend: Default::default(),
            inference_health: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("inference_health"), "inference_health should be absent when None");
    }

    #[test]
    fn test_gpu_status_response_with_inference_health() {
        let resp = GpuStatusResponse {
            gpus: vec![],
            inference_backend: Default::default(),
            inference_health: Some(InferenceHealthStatus {
                status: InferenceProbeStatus::Healthy,
                ep_name: "cuda".to_string(),
                ep_requested: "gpu".to_string(),
                fallback_used: false,
                baseline_us: 12000,
                last_probe_us: 13000,
                drift_pct: 8.3,
                models_loaded: 1,
                last_probe_at: Some("2026-03-26T12:00:00Z".to_string()),
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"inference_health\""));
        assert!(json.contains("\"healthy\""));
    }

    #[test]
    fn test_probe_status_serialization() {
        assert_eq!(serde_json::to_string(&InferenceProbeStatus::Pending).unwrap(), "\"pending\"");
        assert_eq!(serde_json::to_string(&InferenceProbeStatus::Healthy).unwrap(), "\"healthy\"");
        assert_eq!(serde_json::to_string(&InferenceProbeStatus::Degraded).unwrap(), "\"degraded\"");
        assert_eq!(serde_json::to_string(&InferenceProbeStatus::Failed).unwrap(), "\"failed\"");
        assert_eq!(serde_json::to_string(&InferenceProbeStatus::Unreachable).unwrap(), "\"unreachable\"");
    }

    #[test]
    fn test_gpu_info_new_fields_absent_when_none() {
        let gpu = GpuInfo {
            index: 0,
            name: "Test GPU".to_string(),
            vendor: "nvidia".to_string(),
            utilization_pct: Some(50),
            memory_used_mb: None,
            memory_total_mb: None,
            temperature_c: None,
            power_draw_w: None,
            power_limit_w: None,
            processes: None,
            render_busy_pct: None,
            video_busy_pct: None,
            frequency_mhz: None,
            frequency_max_mhz: None,
        };
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(!json.contains("power_draw_w"));
        assert!(!json.contains("processes"));
        assert!(!json.contains("render_busy_pct"));
        assert!(!json.contains("video_busy_pct"));
        assert!(json.contains("utilization_pct")); // present because it's Some
    }
}
