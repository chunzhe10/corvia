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
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("traces"));
    }
}
