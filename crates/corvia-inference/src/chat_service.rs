use crate::backend::{BackendKind, ResolvedBackend};
use llama_cpp_2::context::params::KvCacheType;
use llama_cpp_sys_2::{LLAMA_FLASH_ATTN_TYPE_ENABLED, LLAMA_FLASH_ATTN_TYPE_DISABLED};
use corvia_proto::chat_service_server::ChatService;
use corvia_proto::*;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

/// Resolved model coordinates on Hugging Face.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub repo: String,
    pub filename: String,
}

/// Map a short model name (e.g. "llama3.2", "llama3.2:1b") to a HF repo + GGUF filename.
/// This is the hardcoded fallback — keep in sync with `default_chat_models()` in
/// `corvia-common/src/config.rs`. Config-driven resolution is preferred when available.
pub fn resolve_model(name: &str) -> Result<ResolvedModel, Status> {
    match name {
        "llama3.2" | "llama3.2:3b" => Ok(ResolvedModel {
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
        }),
        "llama3.2:1b" => Ok(ResolvedModel {
            repo: "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
            filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
        }),
        "qwen3" | "qwen3:8b" => Ok(ResolvedModel {
            repo: "bartowski/Qwen_Qwen3-8B-GGUF".to_string(),
            filename: "Qwen3-8B-Q4_K_M.gguf".to_string(),
        }),
        other => Err(Status::invalid_argument(format!(
            "Unknown model: '{other}'. Supported: qwen3, qwen3:8b, llama3.2, llama3.2:3b, llama3.2:1b"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Backend initialization (global, once)
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Configure llama.cpp model params based on resolved backend.
fn build_model_params(backend: &ResolvedBackend) -> LlamaModelParams {
    match backend.backend {
        BackendKind::Cuda => LlamaModelParams::default().with_n_gpu_layers(999),
        _ => LlamaModelParams::default(), // CPU: n_gpu_layers = 0
    }
}

/// Get a static reference to the initialized llama.cpp backend.
fn llama_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        let backend = LlamaBackend::init().expect("Failed to initialize llama.cpp backend");
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        backend
    })
}

// ---------------------------------------------------------------------------
// Model entry
// ---------------------------------------------------------------------------

struct ChatModelEntry {
    model: Arc<LlamaModel>,
    backend: ResolvedBackend,
    kv_cache_type: KvCacheType,
    flash_attention: bool,
}

// ---------------------------------------------------------------------------
// ChatServiceImpl
// ---------------------------------------------------------------------------

/// Real ChatService implementation backed by llama.cpp via llama-cpp-2.
#[derive(Clone)]
pub struct ChatServiceImpl {
    models: Arc<RwLock<HashMap<String, ChatModelEntry>>>,
}

impl ChatServiceImpl {
    pub fn new() -> Self {
        let _ = llama_backend(); // Ensure backend is initialized
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Download (if needed) and load a GGUF model into memory.
    /// When `hf_repo` and `hf_filename` are provided (non-empty), they override
    /// the internal `resolve_model` lookup — enabling config-driven model selection.
    pub async fn load_model(&self, name: &str, backend: ResolvedBackend, kv_cache_type: KvCacheType, flash_attention: bool, hf_repo: &str, hf_filename: &str) -> Result<(), Status> {
        let resolved = if !hf_repo.is_empty() && !hf_filename.is_empty() {
            ResolvedModel { repo: hf_repo.to_string(), filename: hf_filename.to_string() }
        } else {
            resolve_model(name)?
        };
        let name_owned = name.to_string();
        tracing::info!(model = %name_owned, repo = %resolved.repo, file = %resolved.filename,
            device = %backend.device, backend_kind = %backend.backend, "Loading chat model...");

        let backend_clone = backend.clone();
        // Download + load on a blocking thread (both are CPU/IO heavy).
        let model = tokio::task::spawn_blocking(move || -> Result<Arc<LlamaModel>, Status> {
            // Download via hf-hub
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| Status::internal(format!("hf-hub API init failed: {e}")))?;
            let repo = api.model(resolved.repo.clone());
            let model_path: PathBuf = repo
                .get(&resolved.filename)
                .map_err(|e| Status::internal(format!("Model download failed: {e}")))?;

            tracing::info!(path = %model_path.display(), "GGUF file ready, loading into llama.cpp...");

            let model_params = build_model_params(&backend_clone);
            let model = LlamaModel::load_from_file(llama_backend(), &model_path, &model_params)
                .map_err(|e| Status::internal(format!("Model load failed: {e}")))?;

            Ok(Arc::new(model))
        })
        .await
        .map_err(|e| Status::internal(format!("spawn_blocking failed: {e}")))??;

        let mut models = self.models.write().await;
        models.insert(
            name.to_string(),
            ChatModelEntry { model, backend, kv_cache_type, flash_attention },
        );
        tracing::info!(model = %name, "Chat model loaded successfully");
        Ok(())
    }

    /// Get the resolved backend for a loaded model.
    pub async fn get_backend(&self, name: &str) -> Option<ResolvedBackend> {
        let models = self.models.read().await;
        models.get(name).map(|e| e.backend.clone())
    }

    /// Get a loaded model with its KV cache settings.
    async fn get_model_with_settings(&self, name: &str) -> Result<(Arc<LlamaModel>, KvCacheType, bool), Status> {
        let models = self.models.read().await;
        models
            .get(name)
            .map(|e| (Arc::clone(&e.model), e.kv_cache_type, e.flash_attention))
            .ok_or_else(|| Status::not_found(format!("Chat model '{}' not loaded", name)))
    }
}

// ---------------------------------------------------------------------------
// Shared inference helpers
// ---------------------------------------------------------------------------

/// Parameters for a generation request, extracted from the proto.
struct GenerateParams {
    model: Arc<LlamaModel>,
    prompt: String,
    temperature: f32,
    max_tokens: u32,
    kv_cache_type: KvCacheType,
    flash_attention: bool,
}

/// Result of running inference.
struct GenerateResult {
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Build the prompt string from ChatRequest messages using the model's chat template.
fn build_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, Status> {
    let template = model
        .chat_template(None)
        .map_err(|e| Status::internal(format!("Failed to get chat template: {e}")))?;

    let llama_messages: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            LlamaChatMessage::new(m.role.clone(), m.content.clone())
                .map_err(|e| Status::internal(format!("Invalid chat message: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    model
        .apply_chat_template(&template, &llama_messages, true)
        .map_err(|e| Status::internal(format!("Failed to apply chat template: {e}")))
}

/// Build LlamaContextParams with KV cache quantization and flash attention.
fn build_context_params(ctx_size: u32, kv_cache_type: KvCacheType, flash_attention: bool) -> LlamaContextParams {
    let flash_policy = if flash_attention && kv_cache_type == KvCacheType::Q4_0 {
        tracing::warn!("Flash attention not compatible with Q4 KV cache, disabling");
        LLAMA_FLASH_ATTN_TYPE_DISABLED
    } else if flash_attention {
        LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        LLAMA_FLASH_ATTN_TYPE_DISABLED
    };

    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size.max(512)))
        .with_n_batch(512)
        .with_type_k(kv_cache_type)
        .with_type_v(kv_cache_type)
        .with_flash_attention_policy(flash_policy)
}

/// Run full generation (blocking). Returns the generated text and token counts.
fn generate_blocking(params: &GenerateParams) -> Result<GenerateResult, Status> {
    let model = &params.model;

    // Tokenize the prompt
    let prompt_tokens = model
        .str_to_token(&params.prompt, AddBos::Never)
        .map_err(|e| Status::internal(format!("Tokenization failed: {e}")))?;

    let n_prompt = prompt_tokens.len();
    if n_prompt == 0 {
        return Err(Status::invalid_argument("Prompt is empty after tokenization"));
    }
    let max_tokens = if params.max_tokens == 0 {
        2048
    } else {
        params.max_tokens as usize
    };

    // Create context
    let ctx_size = (n_prompt + max_tokens + 64) as u32;
    let ctx_params = build_context_params(ctx_size, params.kv_cache_type, params.flash_attention);

    let mut ctx = model
        .new_context(llama_backend(), ctx_params)
        .map_err(|e| Status::internal(format!("Context creation failed: {e}")))?;

    // Build sampler chain
    let temperature = if params.temperature <= 0.0 {
        0.0 // greedy
    } else {
        params.temperature
    };

    let mut sampler = if temperature == 0.0 {
        LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::greedy(),
        ])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ])
    };

    // Feed the prompt tokens in chunks of n_batch to avoid exceeding batch size
    let n_batch = 512;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut i = 0;
    while i < n_prompt {
        batch.clear();
        let end = (i + n_batch).min(n_prompt);
        for j in i..end {
            let is_last = j == n_prompt - 1;
            batch
                .add(prompt_tokens[j], j as i32, &[0], is_last)
                .map_err(|e| Status::internal(format!("Batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Status::internal(format!("Prompt decode failed: {e}")))?;
        i = end;
    }

    // Generation loop
    let mut output = String::new();
    let mut n_decoded: u32 = 0;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    loop {
        if n_decoded as usize >= max_tokens {
            break;
        }

        let new_token = sampler.sample(&ctx, -1);
        sampler.accept(new_token);

        // Check for end of generation
        if model.is_eog_token(new_token) {
            break;
        }

        // Decode token to string
        let piece = model
            .token_to_piece(new_token, &mut decoder, true, None)
            .map_err(|e| Status::internal(format!("Token decode failed: {e}")))?;
        output.push_str(&piece);
        n_decoded += 1;

        // Prepare next batch
        batch.clear();
        batch
            .add(new_token, (n_prompt + n_decoded as usize - 1) as i32, &[0], true)
            .map_err(|e| Status::internal(format!("Batch add failed: {e}")))?;

        ctx.decode(&mut batch)
            .map_err(|e| Status::internal(format!("Decode failed: {e}")))?;
    }

    Ok(GenerateResult {
        text: output,
        prompt_tokens: n_prompt as u32,
        completion_tokens: n_decoded,
    })
}

/// Run generation token by token, sending each piece through a channel (blocking).
fn generate_streaming_blocking(
    params: &GenerateParams,
    tx: &tokio::sync::mpsc::Sender<Result<ChatChunk, Status>>,
) -> Result<(), Status> {
    let model = &params.model;

    let prompt_tokens = model
        .str_to_token(&params.prompt, AddBos::Never)
        .map_err(|e| Status::internal(format!("Tokenization failed: {e}")))?;

    let n_prompt = prompt_tokens.len();
    let max_tokens = if params.max_tokens == 0 {
        2048
    } else {
        params.max_tokens as usize
    };

    let ctx_size = (n_prompt + max_tokens + 64) as u32;
    let ctx_params = build_context_params(ctx_size, params.kv_cache_type, params.flash_attention);

    let mut ctx = model
        .new_context(llama_backend(), ctx_params)
        .map_err(|e| Status::internal(format!("Context creation failed: {e}")))?;

    let temperature = if params.temperature <= 0.0 {
        0.0
    } else {
        params.temperature
    };

    let mut sampler = if temperature == 0.0 {
        LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::greedy(),
        ])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ])
    };

    // Feed prompt in chunks of n_batch to avoid exceeding batch size
    let n_batch = 512;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut i = 0;
    while i < n_prompt {
        batch.clear();
        let end = (i + n_batch).min(n_prompt);
        for j in i..end {
            let is_last = j == n_prompt - 1;
            batch
                .add(prompt_tokens[j], j as i32, &[0], is_last)
                .map_err(|e| Status::internal(format!("Batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Status::internal(format!("Prompt decode failed: {e}")))?;
        i = end;
    }

    let mut n_decoded: u32 = 0;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    loop {
        if n_decoded as usize >= max_tokens {
            // Send final chunk
            let _ = tx.blocking_send(Ok(ChatChunk {
                delta: String::new(),
                done: true,
                prompt_tokens: n_prompt as u32,
                completion_tokens: n_decoded,
            }));
            break;
        }

        let new_token = sampler.sample(&ctx, -1);
        sampler.accept(new_token);

        if model.is_eog_token(new_token) {
            let _ = tx.blocking_send(Ok(ChatChunk {
                delta: String::new(),
                done: true,
                prompt_tokens: n_prompt as u32,
                completion_tokens: n_decoded,
            }));
            break;
        }

        let piece = model
            .token_to_piece(new_token, &mut decoder, true, None)
            .map_err(|e| Status::internal(format!("Token decode failed: {e}")))?;

        n_decoded += 1;

        // Send chunk — if receiver dropped, stop generating
        if tx
            .blocking_send(Ok(ChatChunk {
                delta: piece,
                done: false,
                prompt_tokens: 0,
                completion_tokens: 0,
            }))
            .is_err()
        {
            break;
        }

        // Prepare next batch
        batch.clear();
        batch
            .add(new_token, (n_prompt + n_decoded as usize - 1) as i32, &[0], true)
            .map_err(|e| Status::internal(format!("Batch add failed: {e}")))?;

        ctx.decode(&mut batch)
            .map_err(|e| Status::internal(format!("Decode failed: {e}")))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// gRPC trait impl
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl ChatService for ChatServiceImpl {
    async fn chat(&self, req: Request<ChatRequest>) -> Result<Response<ChatResponse>, Status> {
        let req = req.into_inner();
        let (model, kv_cache_type, flash_attention) = self.get_model_with_settings(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let params = GenerateParams {
            model,
            prompt,
            temperature,
            max_tokens,
            kv_cache_type,
            flash_attention,
        };

        let result = tokio::task::spawn_blocking(move || generate_blocking(&params))
            .await
            .map_err(|e| Status::internal(format!("spawn_blocking failed: {e}")))??;

        Ok(Response::new(ChatResponse {
            message: Some(ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            }),
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
        }))
    }

    type ChatStreamStream = tokio_stream::wrappers::ReceiverStream<Result<ChatChunk, Status>>;

    async fn chat_stream(
        &self,
        req: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        let req = req.into_inner();
        let (model, kv_cache_type, flash_attention) = self.get_model_with_settings(&req.model).await?;

        let prompt = build_prompt(&model, &req.messages)?;
        let temperature = req.temperature;
        let max_tokens = req.max_tokens;

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let params = GenerateParams {
                model,
                prompt,
                temperature,
                max_tokens,
                kv_cache_type,
                flash_attention,
            };

            if let Err(e) = generate_streaming_blocking(&params, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, Device, ResolvedBackend};

    #[test]
    fn test_build_model_params_cpu() {
        let backend = ResolvedBackend {
            device: Device::Cpu,
            backend: BackendKind::Cpu,
            fallback_used: false,
        };
        let params = build_model_params(&backend);
        let _ = params;
    }

    #[test]
    fn test_build_model_params_cuda() {
        let backend = ResolvedBackend {
            device: Device::Gpu,
            backend: BackendKind::Cuda,
            fallback_used: false,
        };
        let params = build_model_params(&backend);
        let _ = params;
    }

    #[test]
    fn test_resolve_model_llama32_default() {
        let resolved = resolve_model("llama3.2").unwrap();
        assert_eq!(resolved.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(resolved.filename, "Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    }

    #[test]
    fn test_resolve_model_llama32_3b_explicit() {
        let resolved = resolve_model("llama3.2:3b").unwrap();
        assert_eq!(resolved.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(resolved.filename, "Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    }

    #[test]
    fn test_resolve_model_llama32_1b() {
        let resolved = resolve_model("llama3.2:1b").unwrap();
        assert_eq!(resolved.repo, "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(resolved.filename, "Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    }

    #[test]
    fn test_resolve_model_default_equals_3b() {
        let default = resolve_model("llama3.2").unwrap();
        let explicit = resolve_model("llama3.2:3b").unwrap();
        assert_eq!(default, explicit);
    }

    #[test]
    fn test_resolve_model_unknown() {
        let result = resolve_model("gpt-4");
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("gpt-4"));
    }

    #[test]
    fn test_resolve_model_empty() {
        let result = resolve_model("");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_model_qwen3_default() {
        let resolved = resolve_model("qwen3").unwrap();
        assert_eq!(resolved.repo, "bartowski/Qwen_Qwen3-8B-GGUF");
        assert_eq!(resolved.filename, "Qwen3-8B-Q4_K_M.gguf");
    }

    #[test]
    fn test_resolve_model_qwen3_8b_explicit() {
        let resolved = resolve_model("qwen3:8b").unwrap();
        assert_eq!(resolved, resolve_model("qwen3").unwrap());
    }
}
