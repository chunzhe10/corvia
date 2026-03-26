//! GPU utilization detection — NVIDIA via `nvidia-smi`, Intel via sysfs.

use corvia_common::dashboard::{GpuInfo, GpuProcess, GpuStatusResponse, InferenceBackendInfo};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// GPU metrics cache with stampede protection
// ---------------------------------------------------------------------------

/// Cached GPU status with a 5-second TTL and stampede protection.
///
/// Multiple concurrent dashboard requests will not each shell out to nvidia-smi;
/// only the first request after the TTL expires will refresh while others return
/// the stale cached result.
pub struct GpuMetricsCache {
    pub last_result: GpuStatusResponse,
    pub last_fetched: Instant,
    pub refreshing: bool,
    pub refresh_started: Instant,
}

impl GpuMetricsCache {
    pub fn new() -> Self {
        Self {
            last_result: GpuStatusResponse {
                gpus: vec![],
                inference_backend: HashMap::new(),
                inference_health: None,
            },
            last_fetched: Instant::now() - Duration::from_secs(60), // start stale
            refreshing: false,
            refresh_started: Instant::now(),
        }
    }

    pub fn is_stale(&self) -> bool {
        self.last_fetched.elapsed() > Duration::from_secs(5)
    }
}

/// Timeout for nvidia-smi commands.
const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_secs(3);

/// Timeout for intel_gpu_top commands.
const INTEL_GPU_TOP_TIMEOUT: Duration = Duration::from_secs(2);

/// Run a command with a timeout. Returns None if the command fails or times out.
fn run_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    let output = child.wait_with_output().ok()?;
                    return Some(String::from_utf8_lossy(&output.stdout).to_string());
                }
                return None;
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
}

/// Parse nvidia-smi CSV output into `Vec<GpuInfo>`.
///
/// Accepts 6-field rows (without power) or 8-field rows (with power.draw and power.limit).
pub fn parse_nvidia_csv(output: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
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

        let (power_draw_w, power_limit_w) = if fields.len() >= 8 {
            (fields[6].parse::<f64>().ok(), fields[7].parse::<f64>().ok())
        } else {
            (None, None)
        };

        gpus.push(GpuInfo {
            index,
            name,
            vendor: "nvidia".to_string(),
            utilization_pct,
            memory_used_mb,
            memory_total_mb,
            temperature_c,
            power_draw_w,
            power_limit_w,
            processes: None,
            render_busy_pct: None,
            video_busy_pct: None,
            frequency_mhz: None,
            frequency_max_mhz: None,
        });
    }

    gpus
}

/// Parse nvidia-smi compute-apps CSV output into `Vec<GpuProcess>`.
pub fn parse_nvidia_processes(output: &str) -> Vec<GpuProcess> {
    let mut procs = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 3 {
            continue;
        }

        let pid = match fields[0].parse::<u32>() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let name = fields[1].to_string();
        let memory_used_mb = fields[2].parse::<u64>().unwrap_or(0);

        procs.push(GpuProcess {
            pid,
            name,
            memory_used_mb,
        });
    }

    procs
}

/// Detect all NVIDIA GPUs by shelling out to `nvidia-smi`.
///
/// Returns an empty vec if `nvidia-smi` is not available, fails, or times out (3s).
/// Includes power draw/limit fields when available. Also queries running GPU
/// processes and attaches them to the detected GPUs.
///
/// Uses synchronous `std::process::Command` to avoid complex async futures
/// that conflict with the dual axum 0.7/0.8 dependency graph.
pub fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let stdout = match run_with_timeout(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        NVIDIA_SMI_TIMEOUT,
    ) {
        Some(s) => s,
        None => return Vec::new(),
    };

    let mut gpus = parse_nvidia_csv(&stdout);

    // Query GPU processes and attach to GPUs
    let procs = detect_nvidia_processes();
    if !procs.is_empty() && !gpus.is_empty() {
        // Attach all processes to the first GPU for simplicity.
        // A more precise mapping would use --query-compute-apps=gpu_uuid.
        gpus[0].processes = Some(procs);
    }

    gpus
}

/// Query running GPU compute processes via `nvidia-smi`.
///
/// Returns an empty vec if the command fails or times out (3s).
pub fn detect_nvidia_processes() -> Vec<GpuProcess> {
    let stdout = match run_with_timeout(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        NVIDIA_SMI_TIMEOUT,
    ) {
        Some(s) => s,
        None => return Vec::new(),
    };

    parse_nvidia_processes(&stdout)
}

/// Parse JSON output from `intel_gpu_top -J`.
///
/// `intel_gpu_top -J` outputs newline-delimited JSON objects (one per sample).
/// Extracts `Render/3D/0` busy % and `Video/0` busy % from the first JSON frame.
fn parse_intel_gpu_top(output: &str) -> (Option<u32>, Option<u32>) {
    let first_line = output.lines().find(|l| l.starts_with('{')).unwrap_or("");
    let obj: serde_json::Value = match serde_json::from_str(first_line) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let engines = match obj.get("engines") {
        Some(e) => e,
        None => return (None, None),
    };
    let render = engines
        .get("Render/3D/0")
        .and_then(|e| e.get("busy"))
        .and_then(|v| v.as_f64())
        .map(|v| v as u32);
    let video = engines
        .get("Video/0")
        .and_then(|e| e.get("busy"))
        .and_then(|v| v.as_f64())
        .map(|v| v as u32);
    (render, video)
}

/// Detect Intel GPU utilization by spawning `intel_gpu_top -J -s 500`.
///
/// This command runs forever, so we read the first JSON frame from stdout
/// and then SIGKILL the child process. Returns `(render_busy_pct, video_busy_pct)`.
/// Falls back to `(None, None)` on ENOENT, EPERM, timeout, or parse error.
fn detect_intel_gpu_utilization() -> (Option<u32>, Option<u32>) {
    // Check if intel_gpu_top is available
    let which = std::process::Command::new("which")
        .arg("intel_gpu_top")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match which {
        Ok(s) if s.success() => {}
        _ => return (None, None),
    }

    // Spawn intel_gpu_top — it runs forever, outputting JSON frames
    let mut child = match std::process::Command::new("intel_gpu_top")
        .args(["-J", "-s", "500"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("failed to spawn intel_gpu_top: {e}");
            return (None, None);
        }
    };

    // Read stdout line-by-line until we get a JSON frame or timeout
    let stdout = child.stdout.take().expect("stdout was piped");
    let reader = std::io::BufReader::new(stdout);
    let start = std::time::Instant::now();
    let mut collected = String::new();

    use std::io::BufRead;
    for line in reader.lines() {
        if start.elapsed() >= INTEL_GPU_TOP_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return (None, None);
        }
        match line {
            Ok(l) => {
                collected.push_str(&l);
                collected.push('\n');
                // Once we have a line starting with '{', we have our first frame
                if l.starts_with('{') {
                    let _ = child.kill();
                    let _ = child.wait();
                    return parse_intel_gpu_top(&collected);
                }
            }
            Err(e) => {
                tracing::debug!("error reading intel_gpu_top stdout: {e}");
                let _ = child.kill();
                let _ = child.wait();
                return (None, None);
            }
        }
    }

    // Shouldn't reach here (intel_gpu_top runs forever), but just in case
    let _ = child.kill();
    let _ = child.wait();
    (None, None)
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

    // Enrich the first Intel GPU with utilization data from intel_gpu_top
    if !gpus.is_empty() {
        let (render, video) = detect_intel_gpu_utilization();
        gpus[0].render_busy_pct = render;
        gpus[0].video_busy_pct = video;
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

    #[test]
    fn test_parse_nvidia_csv_with_power() {
        let csv = "0, NVIDIA GeForce RTX 3090, 45, 2048, 24576, 65, 120.50, 350.00\n";
        let gpus = parse_nvidia_csv(csv);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].power_draw_w, Some(120.5));
        assert_eq!(gpus[0].power_limit_w, Some(350.0));
        assert_eq!(gpus[0].utilization_pct, Some(45));
    }

    #[test]
    fn test_parse_nvidia_csv_missing_power() {
        let csv = "0, NVIDIA GTX 1080, 30, 1024, 8192, 55\n";
        let gpus = parse_nvidia_csv(csv);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].power_draw_w, None);
    }

    #[test]
    fn test_parse_nvidia_processes() {
        let csv = "12345, python3, 1024\n67890, corvia-inference, 2048\n";
        let procs = parse_nvidia_processes(csv);
        assert_eq!(procs.len(), 2);
        assert_eq!(procs[0].pid, 12345);
        assert_eq!(procs[0].name, "python3");
        assert_eq!(procs[0].memory_used_mb, 1024);
    }

    #[test]
    fn test_parse_nvidia_processes_empty() {
        assert!(parse_nvidia_processes("").is_empty());
    }

    #[test]
    fn test_parse_intel_gpu_top_json() {
        let json = r#"{"period":{"duration":0.500},"engines":{"Render/3D/0":{"busy":45.2},"Video/0":{"busy":12.8}}}"#;
        let (render, video) = parse_intel_gpu_top(json);
        assert_eq!(render, Some(45));
        assert_eq!(video, Some(12));
    }

    #[test]
    fn test_parse_intel_gpu_top_invalid_json() {
        let (render, video) = parse_intel_gpu_top("not json");
        assert_eq!(render, None);
        assert_eq!(video, None);
    }

    #[test]
    fn test_parse_intel_gpu_top_missing_engines() {
        let json = r#"{"period":{"duration":0.500}}"#;
        let (render, video) = parse_intel_gpu_top(json);
        assert_eq!(render, None);
        assert_eq!(video, None);
    }

    #[test]
    fn test_parse_intel_gpu_top_multiline() {
        let output = "some header\n{\"period\":{\"duration\":0.500},\"engines\":{\"Render/3D/0\":{\"busy\":30.0},\"Video/0\":{\"busy\":5.0}}}\n{\"period\":{\"duration\":0.500},\"engines\":{\"Render/3D/0\":{\"busy\":40.0}}}";
        let (render, video) = parse_intel_gpu_top(output);
        assert_eq!(render, Some(30)); // first JSON frame only
        assert_eq!(video, Some(5));
    }

    #[test]
    fn test_parse_nvidia_csv_multi_gpu() {
        let csv = "0, GPU A, 45, 2048, 24576, 65, 120.50, 350.00\n1, GPU B, 80, 16384, 40960, 72, 250.00, 400.00\n";
        let gpus = parse_nvidia_csv(csv);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[1].name, "GPU B");
        assert_eq!(gpus[1].power_draw_w, Some(250.0));
    }
}
