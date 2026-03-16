# Chat Inference Implementation Plan

> **Status:** Shipped (v0.3.6)

**Goal:** Replace the stub ChatService in corvia-inference with real LLM text generation via llama.cpp.

**Architecture:** Dual runtime — fastembed (ONNX) for embeddings (unchanged), llama-cpp-2 for chat generation (new). The gRPC proto and kernel-side client are unchanged; only the server-side `ChatServiceImpl` is rewritten.

**Tech Stack:** Rust, llama-cpp-2 (llama.cpp bindings), hf-hub (HuggingFace model downloads), tonic (gRPC), tokio (async runtime)

**Design doc:** `2026-03-09-chat-inference-design.md`

---

### Task 1: Add dependencies

**Files:**
- Modify: `crates/corvia-inference/Cargo.toml`

**Step 1: Add llama-cpp-2 and hf-hub to dependencies**

```toml
[dependencies]
corvia-proto.workspace = true
tonic.workspace = true
prost = "0.13"
tokio.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
clap = { version = "4", features = ["derive"] }
serde.workspace = true
serde_json.workspace = true
fastembed = "5"
tokio-stream = "0.1"
llama-cpp-2 = "0.1"
hf-hub = "0.4"
```

**Step 2: Verify it compiles**

Run: `cargo check --package corvia-inference`
Expected: compiles (warnings OK, no errors). First build will take a few minutes as llama.cpp compiles from source.

**Step 3: Commit**

```bash
git add crates/corvia-inference/Cargo.toml Cargo.lock
git commit -m "chore(corvia-inference): add llama-cpp-2 and hf-hub dependencies"
```

---

### Task 2: Model resolution

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

**Step 1: Write model resolution tests**

Add at the bottom of `chat_service.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_known_model() {
        let (repo, file) = ChatServiceImpl::resolve_model("llama3.2").unwrap();
        assert_eq!(repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert!(file.contains("Q4_K_M"));
    }

    #[test]
    fn test_resolve_1b_variant() {
        let (repo, file) = ChatServiceImpl::resolve_model("llama3.2:1b").unwrap();
        assert_eq!(repo, "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert!(file.contains("Q4_K_M"));
    }

    #[test]
    fn test_resolve_unknown_model() {
        let result = ChatServiceImpl::resolve_model("unknown-model");
        assert!(result.is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --package corvia-inference -- tests::test_resolve`
Expected: FAIL — `resolve_model` method doesn't exist yet.

**Step 3: Implement resolve_model**

Replace the `ChatModelEntry` struct and add `resolve_model` to `ChatServiceImpl`. The full updated top section of `chat_service.rs`:

```rust
use corvia_proto::chat_service_server::ChatService;
use corvia_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use llama_cpp_2::model::LlamaModel;

/// A loaded chat model backed by llama.cpp.
struct ChatModelEntry {
    model: Arc<LlamaModel>,
}

#[derive(Clone)]
pub struct ChatServiceImpl {
    models: Arc<RwLock<HashMap<String, ChatModelEntry>>>,
    backend: Arc<llama_cpp_2::llama_backend::LlamaBackend>,
}

impl ChatServiceImpl {
    pub fn new() -> Self {
        let backend = llama_cpp_2::llama_backend::LlamaBackend::init()
            .expect("Failed to initialize llama.cpp backend");
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            backend: Arc::new(backend),
        }
    }

    /// Resolve a short model name to (hf_repo, gguf_filename).
    pub fn resolve_model(name: &str) -> Result<(String, String), Status> {
        match name {
            "llama3.2" | "llama3.2:3b" => Ok((
                "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
                "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
            )),
            "llama3.2:1b" => Ok((
                "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
                "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
            )),
            other => Err(Status::not_found(format!(
                "Unknown chat model: '{other}'. Available: llama3.2, llama3.2:1b"
            ))),
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --package corvia-inference -- tests::test_resolve`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(corvia-inference): add chat model name resolution"
```

---

### Task 3: Model loading with hf-hub

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

**Step 1: Implement load_model with real GGUF loading**

Replace the existing `load_model` method:

```rust
    pub async fn load_model(&self, name: &str) -> Result<(), Status> {
        let (repo_id, filename) = Self::resolve_model(name)?;

        tracing::info!(model = %name, repo = %repo_id, file = %filename, "Downloading chat model...");

        // Download GGUF file from HuggingFace (blocks on I/O, run on blocking thread)
        let path = {
            let repo_id = repo_id.clone();
            let filename = filename.clone();
            tokio::task::spawn_blocking(move || {
                let api = hf_hub::api::sync::ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .map_err(|e| Status::internal(format!("HF API init failed: {e}")))?;
                api.model(repo_id)
                    .get(&filename)
                    .map_err(|e| Status::internal(format!("Model download failed: {e}")))
            })
            .await
            .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
            ?
        };

        tracing::info!(model = %name, path = %path.display(), "Loading GGUF into llama.cpp...");

        // Load model on blocking thread (CPU-intensive)
        let backend = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            let params = llama_cpp_2::model::params::LlamaModelParams::default();
            LlamaModel::load_from_file(&backend, &path, &params)
                .map_err(|e| Status::internal(format!("GGUF load failed: {e}")))
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        ?;

        let mut models = self.models.write().await;
        models.insert(
            name.to_string(),
            ChatModelEntry {
                model: Arc::new(model),
            },
        );
        tracing::info!(model = %name, "Chat model loaded and ready");
        Ok(())
    }
```

**Step 2: Verify it compiles**

Run: `cargo check --package corvia-inference`
Expected: compiles (the existing `ChatService` trait impl will still have stubs, that's OK).

**Step 3: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(corvia-inference): implement GGUF model download and loading"
```

---

### Task 4: Implement chat() with real inference

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

**Step 1: Replace the stub chat() implementation**

Replace the `ChatService` trait impl's `chat` method. Key steps: apply chat template, tokenize, decode loop, sample tokens, return response.

```rust
    async fn chat(&self, req: Request<ChatRequest>) -> Result<Response<ChatResponse>, Status> {
        let req = req.into_inner();
        let model = {
            let models = self.models.read().await;
            models
                .get(&req.model)
                .map(|e| e.model.clone())
                .ok_or_else(|| Status::not_found(format!("Chat model '{}' not loaded", req.model)))?
        };

        let temperature = if req.temperature > 0.0 { req.temperature } else { 0.7 };
        let max_tokens = if req.max_tokens > 0 { req.max_tokens } else { 2048 };

        // Build chat messages for template application
        let messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        let result = tokio::task::spawn_blocking(move || {
            Self::generate_sync(&model, &messages, temperature, max_tokens)
        })
        .await
        .map_err(|e| Status::internal(format!("Spawn failed: {e}")))?
        ?;

        Ok(Response::new(ChatResponse {
            message: Some(ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            }),
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
        }))
    }
```

**Step 2: Implement the synchronous generation helper**

Add this to `ChatServiceImpl`:

```rust
    /// Synchronous generation — runs on a blocking thread.
    fn generate_sync(
        model: &LlamaModel,
        messages: &[(String, String)],
        temperature: f32,
        max_tokens: u32,
    ) -> Result<GenerateResult, Status> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;
        use llama_cpp_2::token::data::LlamaTokenData;

        // Apply chat template from the GGUF metadata
        let chat_messages: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
            .iter()
            .map(|(role, content)| {
                llama_cpp_2::model::LlamaChatMessage::new(role.clone(), content.clone())
                    .map_err(|e| Status::internal(format!("Invalid chat message: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let prompt = model
            .apply_chat_template(None, &chat_messages, true)
            .map_err(|e| Status::internal(format!("Chat template failed: {e}")))?;

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(4096));
        let mut ctx = model
            .new_context(&model.backend(), &ctx_params)
            .map_err(|e| Status::internal(format!("Context creation failed: {e}")))?;

        // Tokenize
        let tokens = model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Status::internal(format!("Tokenize failed: {e}")))?;

        let prompt_tokens = tokens.len() as u32;

        // Feed prompt tokens
        let mut batch = LlamaBatch::new(4096, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|_| Status::internal("Batch add failed"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Status::internal(format!("Prompt decode failed: {e}")))?;

        // Sampling setup
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(0),
        ]);

        // Generation loop
        let mut output = String::new();
        let mut completion_tokens = 0u32;
        let mut n_cur = tokens.len() as i32;

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            // Check for end of generation
            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_str(token, llama_cpp_2::token::LlamaTokenAttr::UNDEFINED)
                .map_err(|e| Status::internal(format!("Token decode failed: {e}")))?;
            output.push_str(&piece);
            completion_tokens += 1;

            // Prepare next token
            batch.clear();
            batch.add(token, n_cur, &[0], true)
                .map_err(|_| Status::internal("Batch add failed"))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| Status::internal(format!("Decode failed: {e}")))?;
        }

        Ok(GenerateResult {
            text: output,
            prompt_tokens,
            completion_tokens,
        })
    }
```

And add the result struct at the top of the file:

```rust
struct GenerateResult {
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
}
```

**Step 3: Fix compilation — adjust API calls to match actual llama-cpp-2 API**

The above code is based on research. The exact method names and signatures may differ slightly in the actual `llama-cpp-2` crate. After the initial write:

Run: `cargo check --package corvia-inference`

If there are API mismatches, check `llama-cpp-2` docs and examples to fix them. Key areas likely to need adjustment:
- `LlamaSampler` construction — check exact builder pattern
- `model.backend()` — may need to pass backend separately
- `model.apply_chat_template()` — check exact signature
- `model.is_eog_token()` — may be named differently
- `model.token_to_str()` — check exact signature and attr type

**Step 4: Verify it compiles**

Run: `cargo check --package corvia-inference`
Expected: compiles.

**Step 5: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(corvia-inference): implement real chat inference via llama.cpp"
```

---

### Task 5: Implement chat_stream() with real streaming

**Files:**
- Modify: `crates/corvia-inference/src/chat_service.rs`

**Step 1: Replace the stub chat_stream() implementation**

The streaming version uses the same generation logic but sends tokens through an mpsc channel:

```rust
    async fn chat_stream(
        &self,
        req: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        let req = req.into_inner();
        let model = {
            let models = self.models.read().await;
            models
                .get(&req.model)
                .map(|e| e.model.clone())
                .ok_or_else(|| Status::not_found(format!("Chat model '{}' not loaded", req.model)))?
        };

        let temperature = if req.temperature > 0.0 { req.temperature } else { 0.7 };
        let max_tokens = if req.max_tokens > 0 { req.max_tokens } else { 2048 };

        let messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let result = Self::generate_streaming_sync(&model, &messages, temperature, max_tokens, &tx);
            if let Err(e) = result {
                let _ = tx.blocking_send(Ok(ChatChunk {
                    delta: String::new(),
                    done: true,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                }));
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
```

**Step 2: Implement generate_streaming_sync**

This is similar to `generate_sync` but sends each token through the channel:

```rust
    fn generate_streaming_sync(
        model: &LlamaModel,
        messages: &[(String, String)],
        temperature: f32,
        max_tokens: u32,
        tx: &tokio::sync::mpsc::Sender<Result<ChatChunk, Status>>,
    ) -> Result<(), Status> {
        // Same setup as generate_sync: apply template, tokenize, create context, decode prompt
        // ... (reuse the same code up to the generation loop)

        // Generation loop — send each token as a ChatChunk
        let mut completion_tokens = 0u32;

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_str(token, llama_cpp_2::token::LlamaTokenAttr::UNDEFINED)
                .map_err(|e| Status::internal(format!("Token decode failed: {e}")))?;
            completion_tokens += 1;

            let _ = tx.blocking_send(Ok(ChatChunk {
                delta: piece,
                done: false,
                prompt_tokens,
                completion_tokens,
            }));

            // Prepare next token (same as generate_sync)
            batch.clear();
            batch.add(token, n_cur, &[0], true)
                .map_err(|_| Status::internal("Batch add failed"))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| Status::internal(format!("Decode failed: {e}")))?;
        }

        // Final chunk
        let _ = tx.blocking_send(Ok(ChatChunk {
            delta: String::new(),
            done: true,
            prompt_tokens,
            completion_tokens,
        }));

        Ok(())
    }
```

**Step 3: Refactor — extract shared setup into a helper**

Both `generate_sync` and `generate_streaming_sync` share prompt formatting, tokenization, and context setup. Extract that into a `prepare_context` helper to avoid duplication.

**Step 4: Verify it compiles**

Run: `cargo check --package corvia-inference`
Expected: compiles.

**Step 5: Commit**

```bash
git add crates/corvia-inference/src/chat_service.rs
git commit -m "feat(corvia-inference): implement streaming chat inference"
```

---

### Task 6: Build and smoke test

**Files:**
- No new files

**Step 1: Build the full workspace**

Run: `cargo build --package corvia-inference`
Expected: builds successfully.

**Step 2: Run existing tests**

Run: `cargo test --package corvia-inference`
Expected: all tests pass (model resolution tests + any existing tests).

**Step 3: Run all workspace tests to check for regressions**

Run: `cargo test --workspace`
Expected: all Tier 1 tests pass (385+).

**Step 4: Manual smoke test (if model is available)**

```bash
# Stop existing services
kill $(pgrep -f "corvia-inference serve") 2>/dev/null
kill $(pgrep -f "corvia serve") 2>/dev/null

# Install updated binaries
cp target/debug/corvia-inference /usr/local/bin/corvia-inference
cp target/debug/corvia /usr/local/bin/corvia

# Start services
corvia-inference serve --port 8030 &
sleep 2
corvia serve &
sleep 2

# Test via MCP (corvia_ask)
# Reconnect MCP client, then ask a question
```

Expected: `corvia_ask` returns a real generated answer instead of `[stub]`.

**Step 5: Commit any fixes**

If smoke test revealed issues, fix and commit.

---

### Task 7: Integration test (Tier 3)

**Files:**
- Create: `crates/corvia-inference/tests/chat_e2e_test.rs`

**Step 1: Write the integration test**

```rust
//! E2E test for chat inference. Requires GGUF model download (~800MB).
//! Auto-skips when model download would be too slow or fails.

use corvia_proto::chat_service_client::ChatServiceClient;
use corvia_proto::*;

/// Helper: skip test if we can't download the model.
fn skip_if_no_model() -> bool {
    std::env::var("CORVIA_CHAT_TEST").is_err()
}

#[tokio::test]
async fn test_chat_e2e_generates_response() {
    if skip_if_no_model() {
        eprintln!("Skipping: set CORVIA_CHAT_TEST=1 to enable (downloads ~800MB model)");
        return;
    }
    // Start server, load model, send chat request, verify non-empty response
    // ...
}

#[tokio::test]
async fn test_chat_stream_e2e() {
    if skip_if_no_model() {
        eprintln!("Skipping: set CORVIA_CHAT_TEST=1 to enable");
        return;
    }
    // Start server, load model, send streaming request, collect chunks,
    // verify final chunk has done=true and concatenated text is non-empty
    // ...
}
```

**Step 2: Run tests (skipped by default)**

Run: `cargo test --package corvia-inference -- chat_e2e`
Expected: 2 tests skipped (no CORVIA_CHAT_TEST env var).

**Step 3: Commit**

```bash
git add crates/corvia-inference/tests/chat_e2e_test.rs
git commit -m "test(corvia-inference): add Tier 3 chat inference e2e tests"
```
