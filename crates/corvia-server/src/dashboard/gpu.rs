//! GPU utilization detection — NVIDIA via `nvidia-smi`, Intel via sysfs.

use corvia_common::dashboard::{GpuInfo, GpuStatusResponse, InferenceBackendInfo};
use std::collections::HashMap;
use std::path::Path;

/// Detect all NVIDIA GPUs by shelling out to `nvidia-smi`.
///
/// Returns an empty vec if `nvidia-smi` is not available or fails.
/// Uses synchronous `std::process::Command` to avoid complex async futures
/// that conflict with the dual axum 0.7/0.8 dependency graph.
pub fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 6 {
            continue;
        }

        let index = fields[0].parse::<u32>().unwrap_or(0);
        let name = fields[1].to_string();
        let utilization_pct = fields[2].parse::<u32>().ok();
        let memory_used_mb = fields[3].parse::<u64>().ok();
        let memory_total_mb = fields[4].parse::<u64>().ok();
        let temperature_c = fields[5].parse::<u32>().ok();

        gpus.push(GpuInfo {
            index,
            name,
            vendor: "nvidia".to_string(),
            utilization_pct,
            memory_used_mb,
            memory_total_mb,
            temperature_c,
            power_draw_w: None,
            power_limit_w: None,
            processes: None,
            render_busy_pct: None,
            video_busy_pct: None,
            frequency_mhz: None,
            frequency_max_mhz: None,
        });
    }

    gpus
}

/// Detect Intel integrated GPUs via sysfs (`/sys/class/drm/card*/device/vendor`).
///
/// Intel vendor ID is `0x8086`. Reads `gt_act_freq_mhz` and `gt_max_freq_mhz`
/// when available. Returns an empty vec on non-Linux or when no Intel GPU is found.
pub fn detect_intel_gpus() -> Vec<GpuInfo> {
    let drm_dir = Path::new("/sys/class/drm");
    if !drm_dir.exists() {
        return Vec::new();
    }

    let entries = match std::fs::read_dir(drm_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut gpus = Vec::new();
    let mut intel_index: u32 = 0;

    for entry in entries.filter_map(|e| e.ok()) {
        let card_path = entry.path();
        let name = match card_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Only look at card* entries (skip renderD* etc.)
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let vendor_path = card_path.join("device/vendor");
        let vendor = match std::fs::read_to_string(&vendor_path) {
            Ok(v) => v.trim().to_lowercase(),
            Err(_) => continue,
        };

        // Intel vendor ID
        if vendor != "0x8086" {
            continue;
        }

        // Read the device name from sysfs (may not always be available)
        let device_name = read_sysfs_string(&card_path.join("device/label"))
            .or_else(|| {
                // Fallback: construct a name from the PCI device ID
                let device_id = read_sysfs_string(&card_path.join("device/device"))
                    .unwrap_or_else(|| "unknown".to_string());
                Some(format!("Intel GPU [{}]", device_id))
            })
            .unwrap_or_else(|| "Intel GPU".to_string());

        let freq_mhz = read_sysfs_u64(&card_path.join("gt_act_freq_mhz"));
        let freq_max_mhz = read_sysfs_u64(&card_path.join("gt_max_freq_mhz"));

        gpus.push(GpuInfo {
            index: intel_index,
            name: device_name,
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
            frequency_mhz: freq_mhz,
            frequency_max_mhz: freq_max_mhz,
        });

        intel_index += 1;
    }

    gpus
}

/// Build inference backend info from the live config.
pub fn inference_backend_from_config(
    cfg: &corvia_common::config::CorviaConfig,
) -> HashMap<String, InferenceBackendInfo> {
    let mut backends = HashMap::new();

    // Embedding backend
    let emb_device = if !cfg.inference.embedding_backend.is_empty()
        || !cfg.inference.backend.is_empty()
    {
        "gpu"
    } else {
        "cpu"
    };

    let emb_backend = if !cfg.inference.embedding_backend.is_empty() {
        cfg.inference.embedding_backend.clone()
    } else if !cfg.inference.backend.is_empty() {
        cfg.inference.backend.clone()
    } else {
        "cpu".to_string()
    };

    backends.insert(
        "embedding".to_string(),
        InferenceBackendInfo {
            model: cfg.embedding.model.clone(),
            device: emb_device.to_string(),
            backend: emb_backend,
        },
    );

    // Chat backend (from inference config + first chat model)
    let chat_device = if cfg.inference.device == "gpu"
        || (!cfg.inference.backend.is_empty() && cfg.inference.backend != "cpu")
    {
        "gpu"
    } else if cfg.inference.device == "cpu" {
        "cpu"
    } else {
        // "auto" — check if there's a backend hint
        if !cfg.inference.backend.is_empty() {
            "gpu"
        } else {
            "cpu"
        }
    };

    let chat_backend = if !cfg.inference.backend.is_empty() {
        cfg.inference.backend.clone()
    } else {
        "cpu".to_string()
    };

    let chat_model = cfg
        .inference
        .chat_models
        .keys()
        .next()
        .cloned()
        .unwrap_or_else(|| "none".to_string());

    backends.insert(
        "chat".to_string(),
        InferenceBackendInfo {
            model: chat_model,
            device: chat_device.to_string(),
            backend: chat_backend,
        },
    );

    backends
}

/// Collect full GPU status: detected hardware + inference backend config.
///
/// All detection is synchronous (bounded by single `nvidia-smi` call + sysfs reads)
/// to avoid complex async futures that conflict with the dual axum 0.7/0.8 dep graph.
pub fn collect_gpu_status(
    cfg: &corvia_common::config::CorviaConfig,
) -> GpuStatusResponse {
    let mut gpus = detect_nvidia_gpus();
    gpus.extend(detect_intel_gpus());

    // Re-index so NVIDIA and Intel GPUs have contiguous indices
    for (i, gpu) in gpus.iter_mut().enumerate() {
        gpu.index = i as u32;
    }

    let inference_backend = inference_backend_from_config(cfg);

    GpuStatusResponse {
        gpus,
        inference_backend,
        inference_health: None,
    }
}

// ---------------------------------------------------------------------------
// sysfs helpers
// ---------------------------------------------------------------------------

fn read_sysfs_string(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_sysfs_u64(path: &Path) -> Option<u64> {
    read_sysfs_string(path).and_then(|s| s.parse::<u64>().ok())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_intel_gpus_no_sysfs() {
        // On CI/containers without /sys/class/drm, should return empty
        let gpus = detect_intel_gpus();
        // We can't assert count because the test might run on a machine with
        // Intel GPU. Just assert it doesn't panic and returns a Vec.
        assert!(gpus.iter().all(|g| g.vendor == "intel"));
    }

    #[test]
    fn test_inference_backend_defaults() {
        let cfg = corvia_common::config::CorviaConfig::default();
        let backends = inference_backend_from_config(&cfg);

        assert!(backends.contains_key("embedding"));
        assert!(backends.contains_key("chat"));

        let emb = &backends["embedding"];
        assert_eq!(emb.backend, "cpu");
        assert_eq!(emb.device, "cpu");

        let chat = &backends["chat"];
        assert_eq!(chat.backend, "cpu");
        assert_eq!(chat.device, "cpu");
    }

    #[test]
    fn test_inference_backend_with_cuda() {
        let mut cfg = corvia_common::config::CorviaConfig::default();
        cfg.inference.backend = "cuda".to_string();
        cfg.inference.device = "gpu".to_string();

        let backends = inference_backend_from_config(&cfg);

        let emb = &backends["embedding"];
        assert_eq!(emb.backend, "cuda");
        assert_eq!(emb.device, "gpu");

        let chat = &backends["chat"];
        assert_eq!(chat.backend, "cuda");
        assert_eq!(chat.device, "gpu");
    }

    #[test]
    fn test_inference_backend_split_backends() {
        let mut cfg = corvia_common::config::CorviaConfig::default();
        cfg.inference.backend = "cuda".to_string();
        cfg.inference.embedding_backend = "openvino".to_string();
        cfg.inference.device = "gpu".to_string();

        let backends = inference_backend_from_config(&cfg);

        let emb = &backends["embedding"];
        assert_eq!(emb.backend, "openvino");
        assert_eq!(emb.device, "gpu");

        let chat = &backends["chat"];
        assert_eq!(chat.backend, "cuda");
        assert_eq!(chat.device, "gpu");
    }

    #[test]
    fn test_detect_nvidia_no_binary() {
        // On systems without nvidia-smi, should return empty, not error
        let gpus = detect_nvidia_gpus();
        // Can't assert count — might actually have nvidia-smi
        assert!(gpus.iter().all(|g| g.vendor == "nvidia"));
    }

    #[test]
    fn test_collect_gpu_status() {
        let cfg = corvia_common::config::CorviaConfig::default();
        let status = collect_gpu_status(&cfg);

        // Should always have inference_backend entries
        assert!(status.inference_backend.contains_key("embedding"));
        assert!(status.inference_backend.contains_key("chat"));
        // GPUs may or may not be present depending on hardware
        for gpu in &status.gpus {
            assert!(!gpu.name.is_empty());
            assert!(gpu.vendor == "nvidia" || gpu.vendor == "intel");
        }
    }

    #[test]
    fn test_read_sysfs_helpers_missing_path() {
        assert!(read_sysfs_string(Path::new("/nonexistent/path")).is_none());
        assert!(read_sysfs_u64(Path::new("/nonexistent/path")).is_none());
    }
}
