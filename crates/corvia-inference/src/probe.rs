use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::embedding_service::EmbeddingServiceImpl;

#[derive(Debug, Clone, PartialEq)]
pub enum ProbeStatus {
    Pending,
    Healthy,
    Degraded,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ProbeState {
    pub baseline_us: u64,
    pub last_probe_us: u64,
    pub last_probe_at: Option<DateTime<Utc>>,
    pub drift_pct: f64,
    pub ep_name: String,
    pub ep_requested: String,
    pub fallback_used: bool,
    pub status: ProbeStatus,
}

impl Default for ProbeState {
    fn default() -> Self {
        Self {
            baseline_us: 0,
            last_probe_us: 0,
            last_probe_at: None,
            drift_pct: 0.0,
            ep_name: String::new(),
            ep_requested: String::new(),
            fallback_used: false,
            status: ProbeStatus::Pending,
        }
    }
}

impl ProbeState {
    pub fn calc_drift(baseline_us: u64, last_us: u64) -> f64 {
        if baseline_us == 0 {
            return 0.0;
        }
        (last_us as f64 - baseline_us as f64) / baseline_us as f64 * 100.0
    }

    pub fn status_from_drift(drift_pct: f64, threshold_pct: f64) -> ProbeStatus {
        if drift_pct >= threshold_pct {
            ProbeStatus::Degraded
        } else {
            ProbeStatus::Healthy
        }
    }
}

pub type SharedProbeState = Arc<RwLock<ProbeState>>;

pub fn new_shared_probe_state() -> SharedProbeState {
    Arc::new(RwLock::new(ProbeState::default()))
}

pub async fn run_canary(
    embed_svc: &EmbeddingServiceImpl,
    ep_name: &str,
    ep_requested: &str,
    fallback_used: bool,
    state: &SharedProbeState,
) {
    let start = std::time::Instant::now();
    match embed_svc.embed_canary("corvia health probe").await {
        Ok(dims) => {
            let elapsed_us = start.elapsed().as_micros() as u64;
            let mut s = state.write().await;
            s.baseline_us = elapsed_us;
            s.last_probe_us = elapsed_us;
            s.last_probe_at = Some(Utc::now());
            s.ep_name = ep_name.to_string();
            s.ep_requested = ep_requested.to_string();
            s.fallback_used = fallback_used;
            s.drift_pct = 0.0;
            s.status = ProbeStatus::Healthy;
            info!(ep = %ep_name, latency_us = elapsed_us, dims = dims, "inference self-test passed");
            if fallback_used {
                warn!(
                    requested = %ep_requested,
                    actual = %ep_name,
                    "GPU was requested but CPU is being used — check GPU drivers and device availability"
                );
            }
        }
        Err(e) => {
            let mut s = state.write().await;
            s.status = ProbeStatus::Failed;
            s.ep_name = ep_name.to_string();
            s.ep_requested = ep_requested.to_string();
            warn!(error = %e, "inference self-test failed");
        }
    }
}

pub fn spawn_probe_loop(
    embed_svc: EmbeddingServiceImpl,
    state: SharedProbeState,
    interval_secs: u64,
    drift_threshold_pct: f64,
) {
    if interval_secs == 0 {
        info!("periodic health probe disabled (interval = 0)");
        return;
    }
    tokio::spawn(async move {
        let interval = tokio::time::Duration::from_secs(interval_secs);
        loop {
            tokio::time::sleep(interval).await;
            let start = std::time::Instant::now();
            match embed_svc.embed_canary("corvia health probe").await {
                Ok(_) => {
                    let elapsed_us = start.elapsed().as_micros() as u64;
                    let mut s = state.write().await;
                    let drift = ProbeState::calc_drift(s.baseline_us, elapsed_us);
                    s.last_probe_us = elapsed_us;
                    s.last_probe_at = Some(Utc::now());
                    s.drift_pct = drift;
                    s.status = ProbeState::status_from_drift(drift, drift_threshold_pct);
                    if s.status == ProbeStatus::Degraded {
                        warn!(
                            last_us = elapsed_us,
                            baseline_us = s.baseline_us,
                            drift_pct = format!("{:.1}", drift),
                            "inference latency drift detected"
                        );
                    }
                }
                Err(e) => {
                    let mut s = state.write().await;
                    s.status = ProbeStatus::Failed;
                    s.last_probe_at = Some(Utc::now());
                    warn!(error = %e, "periodic health probe failed");
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_drift_normal() {
        let drift = ProbeState::calc_drift(1000, 1500);
        assert!((drift - 50.0).abs() < f64::EPSILON, "expected ~50.0, got {drift}");
    }

    #[test]
    fn test_calc_drift_zero_baseline() {
        assert_eq!(ProbeState::calc_drift(0, 500), 0.0);
    }

    #[test]
    fn test_calc_drift_same_value() {
        assert_eq!(ProbeState::calc_drift(1000, 1000), 0.0);
    }

    #[test]
    fn test_calc_drift_faster_than_baseline() {
        let drift = ProbeState::calc_drift(1000, 500);
        assert!(drift < 0.0, "expected negative drift, got {drift}");
    }

    #[test]
    fn test_status_healthy() {
        assert_eq!(
            ProbeState::status_from_drift(50.0, 100.0),
            ProbeStatus::Healthy
        );
    }

    #[test]
    fn test_status_degraded() {
        assert_eq!(
            ProbeState::status_from_drift(100.0, 100.0),
            ProbeStatus::Degraded
        );
    }

    #[test]
    fn test_status_degraded_above() {
        assert_eq!(
            ProbeState::status_from_drift(150.0, 100.0),
            ProbeStatus::Degraded
        );
    }

    #[test]
    fn test_default_state_is_pending() {
        let state = ProbeState::default();
        assert_eq!(state.status, ProbeStatus::Pending);
        assert_eq!(state.baseline_us, 0);
    }
}
