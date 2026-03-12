/// GPU/CPU backend resolution for corvia-inference.
///
/// Probes hardware once at startup and caches availability.
/// `resolve_backend()` maps (device, backend, model_type) → ResolvedBackend.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    OpenVino,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Embedding,
    Chat,
}

#[derive(Debug, Clone)]
pub struct ResolvedBackend {
    pub device: Device,
    pub backend: BackendKind,
    pub fallback_used: bool,
}

#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub cuda_available: bool,
    pub openvino_available: bool,
}

impl GpuCapabilities {
    /// Probe actual hardware. Call once at startup.
    pub fn probe() -> Self {
        let cuda_available = std::process::Command::new("nvidia-smi")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        let openvino_available = std::path::Path::new("/dev/dri").exists()
            && (std::path::Path::new("/usr/lib/x86_64-linux-gnu/libopenvino.so").exists()
                || std::env::var("INTEL_OPENVINO_DIR").is_ok());

        tracing::info!(cuda = cuda_available, openvino = openvino_available, "GPU capabilities probed");

        Self {
            cuda_available,
            openvino_available,
        }
    }

    /// Create with explicit values (for testing).
    pub fn new(cuda: bool, openvino: bool) -> Self {
        Self {
            cuda_available: cuda,
            openvino_available: openvino,
        }
    }
}

pub fn resolve_backend(
    device: &str,
    backend: &str,
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    // 1. Explicit backend override
    if !backend.is_empty() {
        return resolve_explicit_backend(backend, model_type, gpu);
    }

    // 2. Device-based resolution
    match device {
        "cpu" => Ok(ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        }),
        "gpu" | "auto" | "" => resolve_gpu_preferred(model_type, gpu),
        other => Err(format!("Unknown device: '{other}'. Expected 'auto', 'gpu', or 'cpu'.")),
    }
}

fn resolve_explicit_backend(
    backend: &str,
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    match backend {
        "cuda" => {
            if !gpu.cuda_available {
                return Err("CUDA requested but not available (nvidia-smi not found)".into());
            }
            Ok(ResolvedBackend {
                device: Device::Gpu,
                backend: BackendKind::Cuda,
                fallback_used: false,
            })
        }
        "openvino" => {
            if model_type == ModelType::Chat {
                return Err("OpenVINO does not support chat models (llama.cpp)".into());
            }
            if !gpu.openvino_available {
                return Err("OpenVINO requested but not available (no Intel GPU or libs)".into());
            }
            Ok(ResolvedBackend {
                device: Device::Gpu,
                backend: BackendKind::OpenVino,
                fallback_used: false,
            })
        }
        "cpu" => Ok(ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        }),
        other => Err(format!(
            "Unknown backend: '{other}'. Expected 'cuda', 'openvino', or 'cpu'."
        )),
    }
}

fn resolve_gpu_preferred(
    model_type: ModelType,
    gpu: &GpuCapabilities,
) -> Result<ResolvedBackend, String> {
    // Prefer CUDA
    if gpu.cuda_available {
        return Ok(ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::Cuda,
            fallback_used: false,
        });
    }

    // Then OpenVINO (embedding only)
    if gpu.openvino_available && model_type == ModelType::Embedding {
        return Ok(ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::OpenVino,
            fallback_used: false,
        });
    }

    // Fallback to CPU
    Ok(ResolvedBackend {
        device: Device::Cpu,
        backend: BackendKind::Cpu,
        fallback_used: true,
    })
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Gpu => write!(f, "gpu"),
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "cuda"),
            BackendKind::OpenVino => write!(f, "openvino"),
            BackendKind::Cpu => write!(f, "cpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Explicit backend ---

    #[test]
    fn explicit_cuda_available() {
        let gpu = GpuCapabilities::new(true, false);
        let r = resolve_backend("", "cuda", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn explicit_cuda_unavailable() {
        let gpu = GpuCapabilities::new(false, false);
        let r = resolve_backend("", "cuda", ModelType::Embedding, &gpu);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("CUDA"));
    }

    #[test]
    fn explicit_openvino_embedding() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("", "openvino", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::OpenVino);
        assert_eq!(r.device, Device::Gpu);
    }

    #[test]
    fn explicit_openvino_chat_rejected() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("", "openvino", ModelType::Chat, &gpu);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("not support"));
    }

    #[test]
    fn explicit_cpu() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("", "cpu", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn explicit_unknown_backend() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("", "rocm", ModelType::Embedding, &gpu);
        assert!(r.is_err());
    }

    // --- Device-based resolution ---

    #[test]
    fn device_cpu() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("cpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
    }

    #[test]
    fn device_gpu_prefers_cuda() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn device_gpu_falls_back_to_openvino_for_embedding() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::OpenVino);
        assert_eq!(r.device, Device::Gpu);
        assert!(!r.fallback_used);
    }

    #[test]
    fn device_gpu_no_gpu_falls_back_to_cpu() {
        let gpu = GpuCapabilities::new(false, false);
        let r = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(r.fallback_used);
    }

    #[test]
    fn device_gpu_chat_skips_openvino() {
        let gpu = GpuCapabilities::new(false, true);
        let r = resolve_backend("gpu", "", ModelType::Chat, &gpu).unwrap();
        // OpenVINO not valid for chat, so falls back to CPU
        assert_eq!(r.backend, BackendKind::Cpu);
        assert_eq!(r.device, Device::Cpu);
        assert!(r.fallback_used);
    }

    #[test]
    fn device_auto_same_as_gpu() {
        let gpu = GpuCapabilities::new(true, false);
        let auto = resolve_backend("auto", "", ModelType::Embedding, &gpu).unwrap();
        let explicit = resolve_backend("gpu", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(auto.backend, explicit.backend);
        assert_eq!(auto.device, explicit.device);
    }

    #[test]
    fn device_empty_same_as_auto() {
        let gpu = GpuCapabilities::new(true, false);
        let empty = resolve_backend("", "", ModelType::Embedding, &gpu).unwrap();
        let auto = resolve_backend("auto", "", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(empty.backend, auto.backend);
    }

    #[test]
    fn explicit_backend_overrides_device_field() {
        let gpu = GpuCapabilities::new(true, false);
        // Even though device says "cpu", explicit backend="cuda" takes priority
        let r = resolve_backend("cpu", "cuda", ModelType::Embedding, &gpu).unwrap();
        assert_eq!(r.backend, BackendKind::Cuda);
        assert_eq!(r.device, Device::Gpu);
    }

    #[test]
    fn unknown_device() {
        let gpu = GpuCapabilities::new(true, true);
        let r = resolve_backend("tpu", "", ModelType::Embedding, &gpu);
        assert!(r.is_err());
    }
}
