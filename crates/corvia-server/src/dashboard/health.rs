//! Service health probing — HTTP, gRPC, TCP.

use corvia_common::dashboard::{ServiceState, ServiceStatus};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Health check result
pub struct HealthResult {
    pub healthy: Option<bool>, // None = no port to check
    pub latency_ms: f64,       // -1.0 if unhealthy or indeterminate
}

/// Health check protocol
pub enum HealthProto {
    Http,
    Grpc,
    Tcp,
    None,
}

/// Service definition for health probing
pub struct ServiceDef {
    pub name: &'static str,
    pub port: Option<u16>,
    pub health_proto: HealthProto,
    pub health_path: &'static str,
}

const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);

/// Known services to probe
pub fn service_definitions() -> Vec<ServiceDef> {
    vec![
        ServiceDef {
            name: "corvia-server",
            port: Some(8020),
            health_proto: HealthProto::Http,
            health_path: "/health",
        },
        ServiceDef {
            name: "corvia-inference",
            port: Some(8030),
            health_proto: HealthProto::Grpc,
            health_path: "",
        },
    ]
}

/// HTTP health check via raw TCP (avoids reqwest dependency)
pub async fn check_http(host: &str, port: u16, path: &str) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    let stream = match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(s)) => s,
        _ => return HealthResult { healthy: Some(false), latency_ms: -1.0 },
    };
    let mut stream = stream;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).await.is_err() {
        return HealthResult { healthy: Some(false), latency_ms: -1.0 };
    }
    let mut buf = [0u8; 64];
    match timeout(HEALTH_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 12 => {
            let response = String::from_utf8_lossy(&buf[..n]);
            if response.contains("200") || response.contains("204") {
                HealthResult {
                    healthy: Some(true),
                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                }
            } else {
                HealthResult { healthy: Some(false), latency_ms: -1.0 }
            }
        }
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// gRPC health check — TCP connect + HTTP/2 preface handshake
pub async fn check_grpc(host: &str, port: u16) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    let stream = match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(s)) => s,
        _ => return HealthResult { healthy: Some(false), latency_ms: -1.0 },
    };
    let mut stream = stream;
    let preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    if stream.write_all(preface).await.is_err() {
        return HealthResult { healthy: Some(false), latency_ms: -1.0 };
    }
    let mut buf = [0u8; 9]; // HTTP/2 frame header
    match timeout(HEALTH_TIMEOUT, stream.read_exact(&mut buf)).await {
        Ok(Ok(_)) if buf[3] == 0x04 => HealthResult {
            healthy: Some(true),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        },
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// TCP-only health check — just connect, no payload
pub async fn check_tcp(host: &str, port: u16) -> HealthResult {
    let start = Instant::now();
    let addr = format!("{host}:{port}");
    match timeout(HEALTH_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => HealthResult {
            healthy: Some(true),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        },
        _ => HealthResult { healthy: Some(false), latency_ms: -1.0 },
    }
}

/// Convert a HealthResult to a ServiceStatus
pub fn result_to_status(name: &str, port: Option<u16>, result: &HealthResult) -> ServiceStatus {
    let state = match result.healthy {
        Some(true) => ServiceState::Healthy,
        Some(false) => ServiceState::Unhealthy,
        None => ServiceState::Stopped,
    };
    let latency_ms = if result.latency_ms > 0.0 {
        Some(result.latency_ms)
    } else {
        None
    };
    ServiceStatus {
        name: name.to_string(),
        state,
        port,
        latency_ms,
    }
}

/// Check a service's health based on its definition
pub async fn check_service(svc: &ServiceDef) -> HealthResult {
    let host = "127.0.0.1";
    match (svc.port, &svc.health_proto) {
        (None, _) | (_, HealthProto::None) => HealthResult {
            healthy: None,
            latency_ms: -1.0,
        },
        (Some(port), HealthProto::Http) => check_http(host, port, svc.health_path).await,
        (Some(port), HealthProto::Grpc) => check_grpc(host, port).await,
        (Some(port), HealthProto::Tcp) => check_tcp(host, port).await,
    }
}

/// Check all known services and return their statuses
pub async fn check_all_services() -> Vec<ServiceStatus> {
    let defs = service_definitions();
    let mut statuses = Vec::with_capacity(defs.len());
    for def in &defs {
        let result = check_service(def).await;
        statuses.push(result_to_status(def.name, def.port, &result));
    }
    statuses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_definitions_include_core_services() {
        let defs = service_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"corvia-server"));
        assert!(names.contains(&"corvia-inference"));
    }

    #[test]
    fn corvia_server_uses_http_health() {
        let defs = service_definitions();
        let server = defs.iter().find(|d| d.name == "corvia-server").unwrap();
        assert_eq!(server.port, Some(8020));
        assert!(matches!(server.health_proto, HealthProto::Http));
        assert_eq!(server.health_path, "/health");
    }

    #[test]
    fn corvia_inference_uses_grpc_health() {
        let defs = service_definitions();
        let inference = defs.iter().find(|d| d.name == "corvia-inference").unwrap();
        assert_eq!(inference.port, Some(8030));
        assert!(matches!(inference.health_proto, HealthProto::Grpc));
    }

    #[test]
    fn health_result_to_service_status_healthy() {
        let result = HealthResult { healthy: Some(true), latency_ms: 2.5 };
        let status = result_to_status("corvia-server", Some(8020), &result);
        assert_eq!(status.state, ServiceState::Healthy);
        assert_eq!(status.latency_ms, Some(2.5));
    }

    #[test]
    fn health_result_to_service_status_unhealthy() {
        let result = HealthResult { healthy: Some(false), latency_ms: -1.0 };
        let status = result_to_status("corvia-server", Some(8020), &result);
        assert_eq!(status.state, ServiceState::Unhealthy);
        assert_eq!(status.latency_ms, None);
    }

    #[test]
    fn health_result_to_service_status_no_port() {
        let result = HealthResult { healthy: None, latency_ms: -1.0 };
        let status = result_to_status("vllm", None, &result);
        assert_eq!(status.state, ServiceState::Stopped);
    }
}
