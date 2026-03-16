# M4: Observability + Control Plane — Implementation Plan

> **Status:** Shipped (v0.4.0)

**Goal:** Add structured tracing, telemetry configuration, shared kernel operations, 10 new MCP admin tools with a tiered safety model, and CLI metrics to corvia.

**Architecture:** A new `corvia-telemetry` crate provides initialization and span constants. Shared kernel operations (`ops.rs`) eliminate code duplication between CLI and MCP. The existing stateless MCP server gets 10 new tools across 3 safety tiers. Config hot-reload enables runtime tuning via MCP.

**Tech Stack:** Rust, tracing, tracing-subscriber, tracing-appender, OpenTelemetry (declared, OTLP deferred to M5), Redb, Axum, JSON-RPC/MCP

**Spec:** `2026-03-10-m4-observability-control-plane-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `crates/corvia-telemetry/Cargo.toml` | Crate manifest — depends on `corvia-common`, `tracing`, `tracing-subscriber`, `tracing-appender` |
| `crates/corvia-telemetry/src/lib.rs` | `init_telemetry()` function + `pub mod spans` with D45 span name constants |
| `crates/corvia-kernel/src/ops.rs` | Shared kernel operations callable from CLI and MCP |

### Modified files

| File | Changes |
|------|---------|
| `Cargo.toml` (workspace) | Add `corvia-telemetry` member + `tracing-appender` workspace dep |
| `crates/corvia-common/src/config.rs` | Add `TelemetryConfig` struct + `telemetry` field to `CorviaConfig` |
| `crates/corvia-kernel/src/lib.rs` | Add `pub mod ops;` |
| `crates/corvia-kernel/src/merge_queue.rs` | Rename `dequeue_batch` → `list`, update callers |
| `crates/corvia-kernel/Cargo.toml` | Add `corvia-telemetry` dep |
| `crates/corvia-kernel/src/agent_coordinator.rs` | Add `#[tracing::instrument]` spans |
| `crates/corvia-kernel/src/merge_worker.rs` | Add spans |
| `crates/corvia-kernel/src/lite_store.rs` | Add spans |
| `crates/corvia-kernel/src/ollama_engine.rs` | Add span to `embed` |
| `crates/corvia-kernel/src/grpc_engine.rs` | Add span to `embed` |
| `crates/corvia-kernel/src/rag_pipeline.rs` | Add spans to `context`, `ask` |
| `crates/corvia-server/src/rest.rs` | Add `config`, `config_path` to `AppState` |
| `crates/corvia-server/src/mcp.rs` | Add 10 tools + safety tier dispatch + handlers + update test helpers |
| `crates/corvia-server/Cargo.toml` | Add `corvia-telemetry` dep |
| `crates/corvia-cli/src/main.rs` | Refactor to `ops::*`, add `--metrics`, wire `init_telemetry()` |
| `crates/corvia-cli/Cargo.toml` | Add `corvia-telemetry` dep |

---

## Chunk 1: Foundation (D80 + D82)

### Task 1: Create `corvia-telemetry` crate with `TelemetryConfig`

**Files:**
- Create: `crates/corvia-telemetry/Cargo.toml`
- Create: `crates/corvia-telemetry/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/corvia-common/src/config.rs`
- Modify: `crates/corvia-common/Cargo.toml`

-**Step 1: Add `TelemetryConfig` to `corvia-common/src/config.rs`**

Add the struct and its Default impl alongside the other config types. Add the field to `CorviaConfig`.

```rust
// In corvia-common/src/config.rs, add after ChunkingConfig:

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_telemetry_exporter")]
    pub exporter: String,
    #[serde(default)]
    pub otlp_endpoint: String,
    #[serde(default = "default_telemetry_log_format")]
    pub log_format: String,
    #[serde(default = "default_telemetry_metrics_enabled")]
    pub metrics_enabled: bool,
}

fn default_telemetry_exporter() -> String { "stdout".into() }
fn default_telemetry_log_format() -> String { "text".into() }
fn default_telemetry_metrics_enabled() -> bool { true }

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            exporter: default_telemetry_exporter(),
            otlp_endpoint: String::new(),
            log_format: default_telemetry_log_format(),
            metrics_enabled: default_telemetry_metrics_enabled(),
        }
    }
}
```

Add to `CorviaConfig` struct:
```rust
    #[serde(default)]
    pub telemetry: TelemetryConfig,
```

Add to `CorviaConfig::default()`:
```rust
    telemetry: TelemetryConfig::default(),
```

-**Step 2: Add workspace deps and member for telemetry crate**

In workspace `Cargo.toml`, add to `[workspace.members]`:
```toml
"crates/corvia-telemetry"
```

Add to `[workspace.dependencies]`:
```toml
corvia-telemetry = { path = "crates/corvia-telemetry" }
tracing-appender = "0.2"
```

Also update the existing `tracing-subscriber` workspace dep to include required features:
```toml
tracing-subscriber = { version = "0.3", features = ["env-filter", "json", "fmt"] }
```
(It currently only has `env-filter`; add `json` and `fmt`.)

-**Step 3: Create `crates/corvia-telemetry/Cargo.toml`**

```toml
[package]
name = "corvia-telemetry"
version = "0.1.0"
edition = "2021"

[dependencies]
corvia-common = { workspace = true }
tracing.workspace = true
tracing-subscriber = { workspace = true, features = ["env-filter", "json", "fmt"] }
tracing-appender = { workspace = true }
anyhow.workspace = true
```

-**Step 4: Create `crates/corvia-telemetry/src/lib.rs`**

```rust
use corvia_common::config::TelemetryConfig;

/// Span name constants matching the D45 observability contract.
/// Leaf crates import these; they use `tracing` directly for `#[instrument]`.
pub mod spans {
    pub const AGENT_REGISTER: &str = "corvia.agent.register";
    pub const SESSION_CREATE: &str = "corvia.session.create";
    pub const ENTRY_WRITE: &str = "corvia.entry.write";
    pub const ENTRY_EMBED: &str = "corvia.entry.embed";
    pub const ENTRY_INSERT: &str = "corvia.entry.insert";
    pub const SESSION_COMMIT: &str = "corvia.session.commit";
    pub const MERGE_PROCESS: &str = "corvia.merge.process";
    pub const MERGE_CONFLICT: &str = "corvia.merge.conflict";
    pub const MERGE_LLM_RESOLVE: &str = "corvia.merge.llm_resolve";
    pub const GC_RUN: &str = "corvia.gc.run";
    pub const SEARCH: &str = "corvia.search";
    pub const STORE_INSERT: &str = "corvia.store.insert";
    pub const STORE_SEARCH: &str = "corvia.store.search";
    pub const STORE_GET: &str = "corvia.store.get";
    pub const RAG_CONTEXT: &str = "corvia.rag.context";
    pub const RAG_ASK: &str = "corvia.rag.ask";
}

/// Initialize the tracing subscriber pipeline based on config.
/// Call once at startup from `corvia serve` or CLI entry point.
pub fn init_telemetry(config: &TelemetryConfig) -> anyhow::Result<()> {
    use tracing_subscriber::{fmt, EnvFilter, prelude::*};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    match config.exporter.as_str() {
        "file" => {
            let file_appender = tracing_appender::rolling::daily("logs", "corvia.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            // Leak the guard so it lives for the process lifetime.
            // This is standard practice for tracing file appenders.
            std::mem::forget(_guard);

            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json().with_writer(non_blocking))
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().with_writer(non_blocking))
                    .init();
            }
        }
        "otlp" => {
            // OTLP exporter deferred to M5. For now, fall through to stdout
            // with a warning that OTLP is not yet wired.
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer())
                .init();
            tracing::warn!("OTLP exporter configured but not yet implemented; falling back to stdout");
        }
        _ => {
            // "stdout" or any unrecognized value
            if config.log_format == "json" {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer().json())
                    .init();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt::layer())
                    .init();
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_constants_are_dotted() {
        // All span names must use dotted notation
        let all = [
            spans::AGENT_REGISTER, spans::SESSION_CREATE, spans::ENTRY_WRITE,
            spans::ENTRY_EMBED, spans::ENTRY_INSERT, spans::SESSION_COMMIT,
            spans::MERGE_PROCESS, spans::MERGE_CONFLICT, spans::MERGE_LLM_RESOLVE,
            spans::GC_RUN, spans::SEARCH, spans::STORE_INSERT, spans::STORE_SEARCH,
            spans::STORE_GET, spans::RAG_CONTEXT, spans::RAG_ASK,
        ];
        for name in &all {
            assert!(name.starts_with("corvia."), "{name} must start with 'corvia.'");
            assert!(name.contains('.'), "{name} must use dotted notation");
        }
    }

    #[test]
    fn test_default_telemetry_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.exporter, "stdout");
        assert_eq!(config.log_format, "text");
        assert!(config.metrics_enabled);
        assert!(config.otlp_endpoint.is_empty());
    }
}
```

-**Step 5: Verify telemetry crate compiles and tests pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-telemetry`
Expected: 2 tests pass

-**Step 6: Verify full workspace still compiles**

Run: `cargo build --workspace`
Expected: Compiles with no errors. `TelemetryConfig` added to `CorviaConfig` with `#[serde(default)]` so existing configs still deserialize.

-**Step 7: Commit**

```bash
git add crates/corvia-telemetry/ Cargo.toml crates/corvia-common/src/config.rs crates/corvia-common/Cargo.toml
git commit -m "feat(m4): add corvia-telemetry crate with TelemetryConfig, span constants, and init_telemetry"
```

---

### Task 2: Rename `MergeQueue::dequeue_batch` → `list`

**Files:**
- Modify: `crates/corvia-kernel/src/merge_queue.rs`
- Modify: all callers of `dequeue_batch` (search for usage)

-**Step 1: Find all callers of `dequeue_batch`**

Run: `cd /workspaces/corvia-workspace/repos/corvia && grep -rn "dequeue_batch" crates/`
Record all call sites.

-**Step 2: Rename the method**

In `crates/corvia-kernel/src/merge_queue.rs`, rename `pub fn dequeue_batch` to `pub fn list`. The implementation is already read-only — this is a naming fix only.

-**Step 3: Update all callers**

Replace `dequeue_batch` with `list` at every call site found in step 1.

-**Step 4: Run tests**

Run: `cargo test --workspace`
Expected: All tests pass (pure rename, no behavior change).

-**Step 5: Commit**

```bash
git add -u
git commit -m "refactor(kernel): rename MergeQueue::dequeue_batch to list for clarity"
```

---

### Task 3: Create `ops.rs` with `system_status`

**Files:**
- Create: `crates/corvia-kernel/src/ops.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add `pub mod ops;`)

-**Step 1: Write the test for `system_status`**

Add at the bottom of the new `ops.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::traits::{QueryableStore, InferenceEngine, GenerationEngine};
    use crate::agent_coordinator::AgentCoordinator;
    use corvia_common::config::{AgentLifecycleConfig, MergeConfig};
    use std::sync::Arc;

    // Reuse the MockEngine pattern from agent_coordinator.rs tests
    struct MockEngine;
    #[async_trait::async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> corvia_common::errors::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> corvia_common::errors::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dimensions(&self) -> usize { 3 }
    }
    struct MockGenEngine;
    #[async_trait::async_trait]
    impl GenerationEngine for MockGenEngine {
        fn name(&self) -> &str { "mock" }
        async fn generate(&self, _system_prompt: &str, user_message: &str) -> corvia_common::errors::Result<crate::traits::GenerationResult> {
            Ok(crate::traits::GenerationResult {
                text: format!("merged: {user_message}"),
                model: "mock".into(),
                input_tokens: 0,
                output_tokens: 0,
            })
        }
        fn context_window(&self) -> usize { 4096 }
    }

    #[tokio::test]
    async fn test_system_status_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen = Arc::new(MockGenEngine) as Arc<dyn GenerationEngine>;
        let coord = Arc::new(AgentCoordinator::new(
            store.clone() as Arc<dyn QueryableStore>,
            engine, dir.path(),
            AgentLifecycleConfig::default(),
            MergeConfig::default(),
            gen,
        ).unwrap());

        let status = system_status(
            store as Arc<dyn QueryableStore>,
            &coord,
            "test-scope",
        ).await.unwrap();

        assert_eq!(status.entry_count, 0);
        assert_eq!(status.active_agents, 0);
        assert_eq!(status.open_sessions, 0);
        assert_eq!(status.merge_queue_depth, 0);
    }
}
```

-**Step 2: Run the test — verify it fails**

Run: `cargo test -p corvia-kernel -- ops::tests::test_system_status_empty_store`
Expected: FAIL — module `ops` doesn't exist yet.

-**Step 3: Add `pub mod ops;` to `lib.rs`**

In `crates/corvia-kernel/src/lib.rs`, add after the last `pub mod` line:
```rust
pub mod ops;
```

-**Step 4: Implement `system_status` in `ops.rs`**

```rust
use crate::agent_coordinator::AgentCoordinator;
use crate::adapter_discovery::discover_adapters;
use crate::traits::QueryableStore;
use corvia_common::errors::Result;
use std::sync::Arc;

/// System status snapshot — returned by both CLI and MCP.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemStatus {
    pub entry_count: u64,
    pub active_agents: usize,
    pub open_sessions: usize,
    pub merge_queue_depth: u64,
    pub scope_id: String,
}

/// Get a snapshot of system status.
pub async fn system_status(
    store: Arc<dyn QueryableStore>,
    coordinator: &AgentCoordinator,
    scope_id: &str,
) -> Result<SystemStatus> {
    let entry_count = store.count(scope_id).await.unwrap_or(0);
    let active_agents = coordinator.registry.list_active()?.len();
    let open_sessions = coordinator.sessions.list_open()?.len();
    let merge_queue_depth = coordinator.merge_queue.depth()?;

    Ok(SystemStatus {
        entry_count,
        active_agents,
        open_sessions,
        merge_queue_depth,
        scope_id: scope_id.to_string(),
    })
}
```

-**Step 5: Run the test — verify it passes**

Run: `cargo test -p corvia-kernel -- ops::tests::test_system_status_empty_store`
Expected: PASS

-**Step 6: Commit**

```bash
git add crates/corvia-kernel/src/ops.rs crates/corvia-kernel/src/lib.rs
git commit -m "feat(m4): add ops::system_status — shared kernel operation for CLI and MCP"
```

---

### Task 4: Add remaining ops functions

**Files:**
- Modify: `crates/corvia-kernel/src/ops.rs`
- Modify: `crates/corvia-kernel/Cargo.toml` (add `toml` dep if not present for config_set)
- Modify: `crates/corvia-kernel/src/adapter_discovery.rs` (add `Serialize` derive to `DiscoveredAdapter`)

-**Step 0: Add `Serialize` derive to `DiscoveredAdapter`**

In `crates/corvia-kernel/src/adapter_discovery.rs`, change:
```rust
#[derive(Debug, Clone)]
pub struct DiscoveredAdapter {
```
to:
```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscoveredAdapter {
```

This is needed so MCP tools can serialize adapter info to JSON responses.

-**Step 0b: Verify `toml` is already a dependency of `corvia-kernel`**

Check `crates/corvia-kernel/Cargo.toml` for `toml = "0.8"`. It should already be present. If not, add it.

-**Step 1: Write tests for `agents_list`, `merge_queue_status`, `config_get`**

Add to `ops.rs` tests module:

```rust
    #[tokio::test]
    async fn test_agents_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen = Arc::new(MockGenEngine) as Arc<dyn GenerationEngine>;
        let coord = Arc::new(AgentCoordinator::new(
            store.clone() as Arc<dyn QueryableStore>,
            engine, dir.path(),
            AgentLifecycleConfig::default(),
            MergeConfig::default(),
            gen,
        ).unwrap());

        let agents = agents_list(&coord).unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_merge_queue_status_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LiteStore::open(dir.path(), 3).unwrap());
        store.init_schema().await.unwrap();
        let engine = Arc::new(MockEngine) as Arc<dyn InferenceEngine>;
        let gen = Arc::new(MockGenEngine) as Arc<dyn GenerationEngine>;
        let coord = Arc::new(AgentCoordinator::new(
            store.clone() as Arc<dyn QueryableStore>,
            engine, dir.path(),
            AgentLifecycleConfig::default(),
            MergeConfig::default(),
            gen,
        ).unwrap());

        let status = merge_queue_status(&coord, 10).unwrap();
        assert_eq!(status.depth, 0);
        assert!(status.entries.is_empty());
    }

    #[test]
    fn test_config_get_full() {
        let config = corvia_common::config::CorviaConfig::default();
        let result = config_get(&config, None).unwrap();
        assert!(result.is_object());
        assert!(result.get("storage").is_some());
    }

    #[test]
    fn test_config_get_section() {
        let config = corvia_common::config::CorviaConfig::default();
        let result = config_get(&config, Some("server")).unwrap();
        assert!(result.is_object());
    }

    #[test]
    fn test_config_get_invalid_section() {
        let config = corvia_common::config::CorviaConfig::default();
        let result = config_get(&config, Some("nonexistent"));
        assert!(result.is_err());
    }
```

-**Step 2: Run tests — verify they fail**

Run: `cargo test -p corvia-kernel -- ops::tests`
Expected: FAIL — functions not defined yet.

-**Step 3: Implement remaining ops functions**

Add to `ops.rs`:

```rust
use crate::adapter_discovery::{discover_adapters, DiscoveredAdapter};
use crate::merge_queue::MergeQueue;
use corvia_common::agent_types::{AgentRecord, MergeQueueEntry, AgentStatus};
use corvia_common::config::CorviaConfig;
use corvia_common::errors::CorviaError;

/// Agent list from the registry.
pub fn agents_list(coordinator: &AgentCoordinator) -> Result<Vec<AgentRecord>> {
    coordinator.registry.list_all()
}

/// Session list for a specific agent.
pub fn sessions_list(
    coordinator: &AgentCoordinator,
    agent_id: &str,
) -> Result<Vec<corvia_common::agent_types::SessionRecord>> {
    coordinator.sessions.list_by_agent(agent_id)
}

/// Merge queue status snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MergeQueueStatus {
    pub depth: u64,
    pub entries: Vec<MergeQueueEntry>,
}

pub fn merge_queue_status(
    coordinator: &AgentCoordinator,
    limit: usize,
) -> Result<MergeQueueStatus> {
    let depth = coordinator.merge_queue.depth()?;
    let entries = coordinator.merge_queue.list(limit)?;
    Ok(MergeQueueStatus { depth, entries })
}

/// Discover available adapters.
pub fn adapters_list(extra_dirs: &[String]) -> Vec<DiscoveredAdapter> {
    discover_adapters(extra_dirs)
}

/// Read config as JSON, optionally a specific section.
pub fn config_get(
    config: &CorviaConfig,
    section: Option<&str>,
) -> Result<serde_json::Value> {
    let full = serde_json::to_value(config)
        .map_err(|e| CorviaError::Config(format!("Failed to serialize config: {e}")))?;

    match section {
        None => Ok(full),
        Some(s) => full.get(s).cloned()
            .ok_or_else(|| CorviaError::Config(format!("Unknown config section: {s}"))),
    }
}

/// Non-hot-reloadable config sections.
const STRUCTURAL_SECTIONS: &[&str] = &["storage", "server", "embedding", "project", "telemetry"];

/// Write a config value. Returns the updated config.
pub fn config_set(
    config_path: &std::path::Path,
    config: &mut CorviaConfig,
    section: &str,
    key: &str,
    value: serde_json::Value,
) -> Result<CorviaConfig> {
    if STRUCTURAL_SECTIONS.contains(&section) {
        return Err(CorviaError::Config(
            format!("Section '{section}' is not hot-reloadable; requires server restart")
        ));
    }

    // Serialize to Value, mutate, deserialize back
    let mut config_value = serde_json::to_value(&*config)
        .map_err(|e| CorviaError::Config(format!("Serialize error: {e}")))?;

    let section_obj = config_value.get_mut(section)
        .ok_or_else(|| CorviaError::Config(format!("Unknown section: {section}")))?;

    let obj = section_obj.as_object_mut()
        .ok_or_else(|| CorviaError::Config(format!("Section '{section}' is not an object")))?;

    obj.insert(key.to_string(), value);

    let updated: CorviaConfig = serde_json::from_value(config_value)
        .map_err(|e| CorviaError::Config(format!("Invalid config after update: {e}")))?;

    // Write TOML to disk
    let toml_str = toml::to_string_pretty(&updated)
        .map_err(|e| CorviaError::Config(format!("TOML serialize error: {e}")))?;
    std::fs::write(config_path, toml_str)
        .map_err(|e| CorviaError::Config(format!("Failed to write config: {e}")))?;

    *config = updated.clone();
    Ok(updated)
}

/// Suspend an agent.
pub fn agent_suspend(
    coordinator: &AgentCoordinator,
    agent_id: &str,
) -> Result<()> {
    coordinator.registry.set_status(agent_id, AgentStatus::Suspended)
}

/// Run garbage collection.
pub async fn gc_run(coordinator: &AgentCoordinator) -> Result<crate::agent_coordinator::GcReport> {
    coordinator.gc().await
}

/// Rebuild HNSW index from knowledge files.
pub fn rebuild_index(data_dir: &std::path::Path, dimensions: usize) -> Result<usize> {
    let store = crate::lite_store::LiteStore::open(data_dir, dimensions)?;
    store.rebuild_from_files()
}

/// Retry failed merge entries by resetting retry_count and re-enqueuing.
pub fn merge_retry(
    coordinator: &AgentCoordinator,
    entry_ids: &[uuid::Uuid],
) -> Result<usize> {
    let mut retried = 0;
    for entry_id in entry_ids {
        // Get the current entry from queue
        let entries = coordinator.merge_queue.list(1000)?;
        if let Some(entry) = entries.iter().find(|e| e.entry_id == *entry_id) {
            if entry.last_error.is_some() {
                // Re-enqueue with reset retry count
                coordinator.merge_queue.mark_complete(entry_id)?;
                coordinator.merge_queue.enqueue(
                    entry.entry_id,
                    &entry.agent_id,
                    &entry.session_id,
                    &entry.scope_id,
                )?;
                retried += 1;
            }
        }
    }
    Ok(retried)
}
```

-**Step 4: Run tests — verify they pass**

Run: `cargo test -p corvia-kernel -- ops::tests`
Expected: All ops tests pass.

-**Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All 385+ tests pass.

-**Step 6: Commit**

```bash
git add crates/corvia-kernel/src/ops.rs
git commit -m "feat(m4): add full ops module — agents_list, merge_queue_status, config_get/set, gc_run, rebuild, merge_retry"
```

---

## Chunk 2: Instrumentation + Config Infrastructure (D81 + D87 + D89)

### Task 5: Add `#[tracing::instrument]` spans to kernel

**Files:**
- Modify: `crates/corvia-kernel/Cargo.toml`
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs`
- Modify: `crates/corvia-kernel/src/merge_worker.rs`
- Modify: `crates/corvia-kernel/src/lite_store.rs`
- Modify: `crates/corvia-kernel/src/ollama_engine.rs`
- Modify: `crates/corvia-kernel/src/grpc_engine.rs`
- Modify: `crates/corvia-kernel/src/rag_pipeline.rs`

-**Step 1: Add `corvia-telemetry` dependency to kernel**

In `crates/corvia-kernel/Cargo.toml`, add:
```toml
corvia-telemetry = { workspace = true }
```

-**Step 2: Add spans to `agent_coordinator.rs`**

Add `use corvia_telemetry::spans;` at the top. Add `#[tracing::instrument]` to key methods. Use actual signatures from the codebase:

```rust
// register_agent is sync (fn, not async fn) — takes AgentIdentity, not agent_id string
#[tracing::instrument(name = "corvia.agent.register", skip(self, identity, permissions), fields(display_name = %display_name))]
pub fn register_agent(&self, identity: &AgentIdentity, display_name: &str, permissions: AgentPermission) -> Result<AgentRecord> { ... }

#[tracing::instrument(name = "corvia.session.create", skip(self), fields(agent_id = %agent_id, with_staging = %with_staging))]
pub fn create_session(&self, agent_id: &str, with_staging: bool) -> Result<SessionRecord> { ... }

#[tracing::instrument(name = "corvia.entry.write", skip(self, content, scope_id, source_version), fields(session_id = %session_id))]
pub async fn write_entry(&self, session_id: &str, content: &str, scope_id: &str, source_version: &str) -> Result<KnowledgeEntry> { ... }

#[tracing::instrument(name = "corvia.session.commit", skip(self), fields(session_id = %session_id))]
pub async fn commit_session(&self, session_id: &str) -> Result<()> { ... }

#[tracing::instrument(name = "corvia.gc.run", skip(self))]
pub async fn gc(&self) -> Result<GcReport> { ... }

#[tracing::instrument(name = "corvia.merge.process", skip(self))]
pub async fn process_merge_queue(&self) -> Result<()> { ... }
```

Note: Use `skip(self)` to avoid Debug-printing the entire coordinator. Use `fields(...)` for key identifiers only. `register_agent` and `create_session` are sync (`fn`), the rest are `async fn`.

-**Step 3: Add spans to `merge_worker.rs`**

```rust
#[tracing::instrument(name = "corvia.merge.conflict", skip(self, entry), fields(entry_id = %entry.id))]
pub async fn detect_conflict(&self, entry: &KnowledgeEntry) -> Result<...> { ... }
```

-**Step 4: Add spans to `lite_store.rs`**

```rust
#[tracing::instrument(name = "corvia.store.insert", skip(self, entry, embedding), fields(entry_id = %entry.id, scope_id = %entry.scope_id))]
async fn insert(&self, entry: &KnowledgeEntry, embedding: &[f32]) -> Result<()> { ... }

#[tracing::instrument(name = "corvia.store.search", skip(self, embedding), fields(scope_id = %scope_id))]
async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<...> { ... }
```

-**Step 5: Add spans to `ollama_engine.rs` and `grpc_engine.rs`**

```rust
#[tracing::instrument(name = "corvia.entry.embed", skip(self, text))]
async fn embed(&self, text: &str) -> Result<Vec<f32>> { ... }
```

-**Step 6: Add spans to `rag_pipeline.rs`**

```rust
#[tracing::instrument(name = "corvia.rag.context", skip(self), fields(scope_id = %scope_id))]
pub async fn context(&self, query: &str, scope_id: &str, ...) -> Result<...> { ... }

#[tracing::instrument(name = "corvia.rag.ask", skip(self), fields(scope_id = %scope_id))]
pub async fn ask(&self, query: &str, scope_id: &str, ...) -> Result<...> { ... }
```

-**Step 7: Verify compilation and tests**

Run: `cargo test --workspace`
Expected: All tests pass. Spans are no-ops without a subscriber, so behavior is unchanged.

-**Step 8: Commit**

```bash
git add -u
git commit -m "feat(m4): add tracing instrument spans across kernel subsystems (D81)"
```

---

### Task 6: Add `config` and `config_path` to `AppState` + wire telemetry init

**Files:**
- Modify: `crates/corvia-server/src/rest.rs`
- Modify: `crates/corvia-server/src/mcp.rs` (test_state helper)
- Modify: `crates/corvia-server/Cargo.toml`
- Modify: `crates/corvia-cli/src/main.rs`
- Modify: `crates/corvia-cli/Cargo.toml`

-**Step 1: Add deps to server and CLI**

In `crates/corvia-server/Cargo.toml`, add:
```toml
corvia-telemetry = { workspace = true }
```

In `crates/corvia-cli/Cargo.toml`, add:
```toml
corvia-telemetry = { workspace = true }
```

-**Step 2: Add fields to `AppState`**

In `crates/corvia-server/src/rest.rs`, add to the `AppState` struct:

```rust
pub config: Arc<std::sync::RwLock<corvia_common::config::CorviaConfig>>,
pub config_path: std::path::PathBuf,
```

-**Step 3: Update `cmd_serve` in CLI to pass config**

In `crates/corvia-cli/src/main.rs`, in the `cmd_serve` function where `AppState` is constructed (~line 441), add:

```rust
config: Arc::new(std::sync::RwLock::new(config.clone())),
config_path: config_path.clone(),
```

Where `config_path` is the path to `corvia.toml` (already resolved earlier in the function).

-**Step 4: Wire `init_telemetry()` into `cmd_serve`**

In `cmd_serve`, before the server starts listening, add:

```rust
if let Err(e) = corvia_telemetry::init_telemetry(&config.telemetry) {
    eprintln!("Warning: telemetry init failed: {e}");
}
```

-**Step 5: Update `test_state` in `mcp.rs`**

In `crates/corvia-server/src/mcp.rs`, update the `test_state` helper to include new fields:

```rust
config: Arc::new(std::sync::RwLock::new(corvia_common::config::CorviaConfig::default())),
config_path: dir.join("corvia.toml"),
```

-**Step 6: Run tests**

Run: `cargo test --workspace`
Expected: All tests pass. The new fields are populated in both production and test code.

-**Step 7: Commit**

```bash
git add -u
git commit -m "feat(m4): add config hot-reload to AppState, wire init_telemetry into serve (D87, D89)"
```

---

## Chunk 3: MCP Control Plane (D83 + D84 + D85 + D86)

### Task 7: Add tiered safety model (D83)

**Files:**
- Modify: `crates/corvia-server/src/mcp.rs`

-**Step 1: Add `ToolTier` enum and helper**

Add near the top of `mcp.rs` (after the error constants):

```rust
/// Safety tier for MCP control-plane tools.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ToolTier {
    ReadOnly,
    LowRisk,
    MediumRisk,
}

/// Check if a tool call has confirmation via _meta.confirmed.
fn is_confirmed(meta: Option<&Value>) -> bool {
    meta.and_then(|m| m.get("confirmed"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Check if dry_run is requested in arguments.
fn is_dry_run(args: &Value) -> bool {
    args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false)
}

/// Return a confirmation-required response for unconfirmed Tier 2+ tools.
fn confirmation_response(preview: Value, message: &str) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&json!({
                "confirmation_required": true,
                "preview": preview,
                "message": message,
            })).unwrap()
        }]
    })
}
```

-**Step 2: Run tests — verify nothing breaks**

Run: `cargo test -p corvia-server -- mcp`
Expected: All existing MCP tests pass (new code is just definitions, not wired yet).

-**Step 3: Commit**

```bash
git add crates/corvia-server/src/mcp.rs
git commit -m "feat(m4): add ToolTier enum and confirmation helpers for MCP safety model (D83)"
```

---

### Task 8: Add Tier 1 read-only MCP tools (D84)

**Files:**
- Modify: `crates/corvia-server/src/mcp.rs`
- Modify: `crates/corvia-server/Cargo.toml` (add corvia-kernel dep if needed for ops)

-**Step 1: Write tests for Tier 1 tools**

Add to the test module in `mcp.rs`:

```rust
    #[tokio::test]
    async fn test_corvia_system_status() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_system_status",
            "arguments": { "scope_id": "test" }
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["entry_count"], 0);
    }

    #[tokio::test]
    async fn test_corvia_agents_list() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_agents_list",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(parsed["agents"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_corvia_merge_queue() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_merge_queue",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["depth"], 0);
    }

    #[tokio::test]
    async fn test_corvia_config_get() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_config_get",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(parsed.get("storage").is_some());
    }

    #[tokio::test]
    async fn test_tools_list_count_18() {
        let result = handle_tools_list();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 18);
    }
```

-**Step 2: Run tests — verify they fail**

Run: `cargo test -p corvia-server -- mcp::tests::test_corvia_system_status`
Expected: FAIL — tool not registered.

-**Step 3: Add 5 tool definitions to `tool_definitions()`**

Add to the `tool_definitions()` function return vector:

```rust
        json!({
            "name": "corvia_system_status",
            "description": "Get system status: entry counts, active agents, sessions, merge queue depth.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope_id": { "type": "string", "description": "Scope to check entry count for" }
                }
            },
            "annotations": { "tier": "ReadOnly" }
        }),
        json!({
            "name": "corvia_config_get",
            "description": "Read current configuration. Optionally specify a section name to get just that section.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Config section name (e.g., 'rag', 'merge'). Omit for full config." }
                }
            },
            "annotations": { "tier": "ReadOnly" }
        }),
        json!({
            "name": "corvia_adapters_list",
            "description": "List all discovered ingestion adapters.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "tier": "ReadOnly" }
        }),
        json!({
            "name": "corvia_agents_list",
            "description": "List all registered agents with their status.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "tier": "ReadOnly" }
        }),
        json!({
            "name": "corvia_merge_queue",
            "description": "Inspect the merge queue: depth and top entries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Max entries to return (default 10)" }
                }
            },
            "annotations": { "tier": "ReadOnly" }
        }),
```

-**Step 4: Add tool handler functions**

```rust
async fn tool_corvia_system_status(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let scope_id = resolve_scope_id(args, state)?;
    let status = corvia_kernel::ops::system_status(
        state.store.clone(),
        &state.coordinator,
        &scope_id,
    ).await.map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;

    let text = serde_json::to_string_pretty(&status)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

fn tool_corvia_config_get(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let section = args.get("section").and_then(|v| v.as_str());
    let config = state.config.read()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock: {e}")))?;
    let result = corvia_kernel::ops::config_get(&config, section)
        .map_err(|e| (INVALID_PARAMS, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&result)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

fn tool_corvia_adapters_list(
    state: &AppState,
) -> Result<Value, (i32, String)> {
    let config = state.config.read()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock: {e}")))?;
    let extra_dirs = config.adapters.as_ref()
        .map(|a| a.search_dirs.clone())
        .unwrap_or_default();
    drop(config); // release lock before potentially slow disk scan
    let adapters = corvia_kernel::ops::adapters_list(&extra_dirs);
    let text = serde_json::to_string_pretty(&json!({ "adapters": adapters }))
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

fn tool_corvia_agents_list(
    state: &AppState,
) -> Result<Value, (i32, String)> {
    let agents = corvia_kernel::ops::agents_list(&state.coordinator)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&json!({ "agents": agents }))
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

fn tool_corvia_merge_queue(
    state: &AppState,
    args: &Value,
) -> Result<Value, (i32, String)> {
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let status = corvia_kernel::ops::merge_queue_status(&state.coordinator, limit)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&status)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}
```

-**Step 5: Add dispatch arms to `handle_tools_call`**

In the `match tool_name` block, add:
```rust
        "corvia_system_status" => tool_corvia_system_status(state, &arguments).await,
        "corvia_config_get"    => tool_corvia_config_get(state, &arguments),
        "corvia_adapters_list" => tool_corvia_adapters_list(state),
        "corvia_agents_list"   => tool_corvia_agents_list(state),
        "corvia_merge_queue"   => tool_corvia_merge_queue(state, &arguments),
```

-**Step 6: Update existing tools/list test assertion**

Change `assert_eq!(tools.len(), 8)` to account for the 5 new tools (update as tools are added; final count will be 18 after all chunks).

-**Step 7: Run tests**

Run: `cargo test -p corvia-server -- mcp`
Expected: All existing + new Tier 1 tests pass.

-**Step 8: Commit**

```bash
git add -u
git commit -m "feat(m4): add 5 Tier 1 read-only MCP tools — system_status, config_get, adapters_list, agents_list, merge_queue (D84)"
```

---

### Task 9: Add Tier 2 low-risk mutation tools (D85)

**Files:**
- Modify: `crates/corvia-server/src/mcp.rs`

-**Step 1: Write tests for Tier 2 tools**

```rust
    #[tokio::test]
    async fn test_config_set_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        // Write a corvia.toml so config_set can write back
        let config = corvia_common::config::CorviaConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        std::fs::write(dir.path().join("corvia.toml"), &toml_str).unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_config_set",
            "arguments": {
                "section": "rag",
                "key": "default_limit",
                "value": 20
            }
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
    }

    #[tokio::test]
    async fn test_config_set_with_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let config = corvia_common::config::CorviaConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        std::fs::write(dir.path().join("corvia.toml"), &toml_str).unwrap();
        let state = test_state(dir.path()).await;

        let params = json!({
            "name": "corvia_config_set",
            "arguments": {
                "section": "rag",
                "key": "default_limit",
                "value": 20
            },
            "_meta": { "confirmed": true }
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        // Should have succeeded, not asked for confirmation
        assert!(parsed.get("confirmation_required").is_none());
    }

    #[tokio::test]
    async fn test_gc_run_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_gc_run",
            "arguments": {}
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
    }
```

-**Step 2: Run tests — verify they fail**

Run: `cargo test -p corvia-server -- mcp::tests::test_config_set`
Expected: FAIL — tools not registered.

-**Step 3: Add 3 tool definitions**

```rust
        json!({
            "name": "corvia_config_set",
            "description": "Update a configuration value. Requires confirmation via _meta.confirmed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Config section (e.g., 'rag', 'merge')" },
                    "key": { "type": "string", "description": "Key within the section" },
                    "value": { "description": "New value to set" }
                },
                "required": ["section", "key", "value"]
            },
            "annotations": { "tier": "LowRisk" }
        }),
        json!({
            "name": "corvia_gc_run",
            "description": "Run garbage collection to clean orphaned sessions and entries. Requires confirmation.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "tier": "LowRisk" }
        }),
        json!({
            "name": "corvia_rebuild_index",
            "description": "Rebuild the HNSW search index from knowledge files. Requires confirmation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dimensions": { "type": "integer", "description": "Embedding dimensions (default: from config)" }
                }
            },
            "annotations": { "tier": "LowRisk" }
        }),
```

-**Step 4: Add handler functions with confirmation logic**

```rust
async fn tool_corvia_config_set(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let section = args.get("section").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'section'".into()))?;
    let key = args.get("key").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'key'".into()))?;
    let value = args.get("value")
        .ok_or((INVALID_PARAMS, "Missing 'value'".into()))?;

    if !is_confirmed(meta) {
        let preview = json!({
            "section": section,
            "key": key,
            "new_value": value,
        });
        return Ok(confirmation_response(preview, &format!("Set {section}.{key}? Send again with _meta.confirmed: true")));
    }

    let mut config = state.config.write()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock: {e}")))?;
    let updated = corvia_kernel::ops::config_set(
        &state.config_path, &mut config, section, key, value.clone(),
    ).map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;

    let text = serde_json::to_string_pretty(&json!({
        "status": "updated",
        "section": section,
        "key": key,
    })).map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

async fn tool_corvia_gc_run(
    state: &AppState,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    if !is_confirmed(meta) {
        let preview = json!({ "action": "gc_run", "description": "Clean orphaned sessions and inactive agents" });
        return Ok(confirmation_response(preview, "Run GC? Send again with _meta.confirmed: true"));
    }

    let report = corvia_kernel::ops::gc_run(&state.coordinator).await
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&json!({
        "orphans_rolled_back": report.orphans_rolled_back,
        "closed_sessions_cleaned": report.closed_sessions_cleaned,
        "inactive_agents_cleaned": report.inactive_agents_cleaned,
    })).map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

async fn tool_corvia_rebuild_index(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let config = state.config.read()
        .map_err(|e| (INTERNAL_ERROR, format!("Config lock: {e}")))?;
    let dimensions = args.get("dimensions").and_then(|v| v.as_u64())
        .unwrap_or(config.embedding.dimensions as u64) as usize;

    if !is_confirmed(meta) {
        let preview = json!({ "action": "rebuild_index", "data_dir": state.data_dir.display().to_string(), "dimensions": dimensions });
        return Ok(confirmation_response(preview, "Rebuild HNSW index? Send again with _meta.confirmed: true"));
    }

    let count = corvia_kernel::ops::rebuild_index(&state.data_dir, dimensions)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&json!({ "status": "rebuilt", "entries_indexed": count }))
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}
```

-**Step 5: Update `handle_tools_call` dispatch**

Tier 2 tools need `meta` passed through. Update the dispatch to pass `meta`:

```rust
        "corvia_config_set"    => tool_corvia_config_set(state, &arguments, meta).await,
        "corvia_gc_run"        => tool_corvia_gc_run(state, meta).await,
        "corvia_rebuild_index" => tool_corvia_rebuild_index(state, &arguments, meta).await,
```

-**Step 6: Run tests**

Run: `cargo test -p corvia-server -- mcp`
Expected: All tests pass including new confirmation tests.

-**Step 7: Commit**

```bash
git add -u
git commit -m "feat(m4): add 3 Tier 2 MCP tools with confirmation — config_set, gc_run, rebuild_index (D85)"
```

---

### Task 10: Add Tier 3 medium-risk mutation tools (D86)

**Files:**
- Modify: `crates/corvia-server/src/mcp.rs`

-**Step 1: Write tests for Tier 3 tools**

```rust
    #[tokio::test]
    async fn test_agent_suspend_dry_run() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_agent_suspend",
            "arguments": { "agent_id": "test::agent", "dry_run": true },
            "_meta": { "confirmed": true }
        });
        let result = handle_tools_call(&state, &params).await;
        // Agent doesn't exist, so it should error — but it shouldn't panic
        assert!(result.is_err() || {
            let text = result.unwrap()["content"][0]["text"].as_str().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            parsed.get("dry_run").is_some()
        });
    }

    #[tokio::test]
    async fn test_agent_suspend_requires_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state(dir.path()).await;
        let params = json!({
            "name": "corvia_agent_suspend",
            "arguments": { "agent_id": "test::agent" }
        });
        let result = handle_tools_call(&state, &params).await.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["confirmation_required"], true);
    }
```

-**Step 2: Add 2 tool definitions**

```rust
        json!({
            "name": "corvia_agent_suspend",
            "description": "Suspend an agent. Supports dry_run. Requires confirmation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "Agent ID to suspend" },
                    "dry_run": { "type": "boolean", "description": "Preview without executing" }
                },
                "required": ["agent_id"]
            },
            "annotations": { "tier": "MediumRisk" }
        }),
        json!({
            "name": "corvia_merge_retry",
            "description": "Retry failed merge entries. Supports dry_run. Requires confirmation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entry_ids": { "type": "array", "items": { "type": "string" }, "description": "UUIDs of failed entries to retry" },
                    "dry_run": { "type": "boolean", "description": "Preview without executing" }
                },
                "required": ["entry_ids"]
            },
            "annotations": { "tier": "MediumRisk" }
        }),
```

-**Step 3: Add handler functions**

```rust
async fn tool_corvia_agent_suspend(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let agent_id = args.get("agent_id").and_then(|v| v.as_str())
        .ok_or((INVALID_PARAMS, "Missing 'agent_id'".into()))?;

    if !is_confirmed(meta) {
        let preview = json!({ "action": "agent_suspend", "agent_id": agent_id });
        return Ok(confirmation_response(preview, &format!("Suspend agent '{agent_id}'? Send with _meta.confirmed: true")));
    }

    if is_dry_run(args) {
        let text = serde_json::to_string_pretty(&json!({ "dry_run": true, "would_suspend": agent_id }))
            .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
        return Ok(json!({ "content": [{ "type": "text", "text": text }] }));
    }

    corvia_kernel::ops::agent_suspend(&state.coordinator, agent_id)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&json!({ "status": "suspended", "agent_id": agent_id }))
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

async fn tool_corvia_merge_retry(
    state: &AppState,
    args: &Value,
    meta: Option<&Value>,
) -> Result<Value, (i32, String)> {
    let entry_ids_raw = args.get("entry_ids").and_then(|v| v.as_array())
        .ok_or((INVALID_PARAMS, "Missing 'entry_ids' array".into()))?;
    let entry_ids: Vec<uuid::Uuid> = entry_ids_raw.iter()
        .filter_map(|v| v.as_str().and_then(|s| uuid::Uuid::parse_str(s).ok()))
        .collect();

    if entry_ids.is_empty() {
        return Err((INVALID_PARAMS, "No valid UUIDs in entry_ids".into()));
    }

    if !is_confirmed(meta) {
        let preview = json!({ "action": "merge_retry", "entry_ids": entry_ids.iter().map(|u| u.to_string()).collect::<Vec<_>>() });
        return Ok(confirmation_response(preview, "Retry these failed merges? Send with _meta.confirmed: true"));
    }

    if is_dry_run(args) {
        let text = serde_json::to_string_pretty(&json!({ "dry_run": true, "would_retry": entry_ids.len() }))
            .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
        return Ok(json!({ "content": [{ "type": "text", "text": text }] }));
    }

    let retried = corvia_kernel::ops::merge_retry(&state.coordinator, &entry_ids)
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    let text = serde_json::to_string_pretty(&json!({ "status": "retried", "count": retried }))
        .map_err(|e| (INTERNAL_ERROR, format!("{e}")))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}
```

-**Step 4: Add dispatch arms**

```rust
        "corvia_agent_suspend" => tool_corvia_agent_suspend(state, &arguments, meta).await,
        "corvia_merge_retry"   => tool_corvia_merge_retry(state, &arguments, meta).await,
```

-**Step 5: Update tools/list test to assert 18 tools**

Change the tools count assertion to `assert_eq!(tools.len(), 18);`.

-**Step 6: Run tests**

Run: `cargo test -p corvia-server -- mcp`
Expected: All 18-tool tests pass.

-**Step 7: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass.

-**Step 8: Commit**

```bash
git add -u
git commit -m "feat(m4): add 2 Tier 3 MCP tools — agent_suspend, merge_retry with dry_run support (D86)"
```

---

## Chunk 4: CLI Observability + Refactor (D88)

### Task 11: Refactor CLI to use `ops::*` and add `--metrics`

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

-**Step 1: Refactor `cmd_status` to use `ops::system_status`**

Replace the inline store queries in `cmd_status` with a call to `ops::system_status()`. The function needs a coordinator, so for the CLI path (where no server is running), construct a minimal coordinator from the data dir.

Keep the existing output format unchanged — just replace the data source.

-**Step 2: Refactor `cmd_agent list` to use `ops::agents_list`**

Replace the inline `AgentRegistry::open()` + `list_all()` in `cmd_agent` with constructing a coordinator and calling `ops::agents_list()`. Or, since `agents_list` takes `&AgentCoordinator` and the CLI may not have a full coordinator for simple list operations, keep the direct registry call and have `ops::agents_list` be the shared path for when a coordinator IS available (MCP + CLI serve mode). The CLI's direct-open path stays for the non-serve case.

Note: This is a pragmatic choice. The CLI opens Redb directly for `agent list` because it doesn't need a full coordinator. The MCP server uses `ops::agents_list(&coordinator)`. Both produce the same data.

-**Step 3: Add `--metrics` flag to status command**

In the CLI's `StatusArgs` (or wherever status subcommand args are defined), add:
```rust
#[clap(long, help = "Show extended metrics")]
metrics: bool,
```

When `--metrics` is passed, print additional info after the standard status:
```rust
if args.metrics {
    println!("\n--- Metrics ---");
    println!("Store type: {}", config.storage.store_type);
    println!("Inference: {} ({})", config.embedding.provider, config.embedding.url);
    println!("Telemetry exporter: {}", config.telemetry.exporter);
    // Agent lifecycle stats from coordinator
    if let Ok(agents) = registry.list_all() {
        println!("Registered agents: {}", agents.len());
        let active = agents.iter().filter(|a| a.status == AgentStatus::Active).count();
        println!("Active agents: {}", active);
    }
    // Adapter discovery
    let adapters = corvia_kernel::ops::adapters_list(&extra_dirs);
    println!("Discovered adapters: {}", adapters.len());
    for a in &adapters {
        println!("  - {} ({})", a.metadata.name, a.binary_path.display());
    }
}
```

-**Step 4: Verify `corvia status` output unchanged without `--metrics`**

Run: `cargo run -p corvia-cli -- status`
Expected: Same output as before the refactor.

-**Step 5: Verify `corvia status --metrics` shows extended info**

Run: `cargo run -p corvia-cli -- status --metrics`
Expected: Standard output plus metrics section.

-**Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass.

-**Step 7: Commit**

```bash
git add -u
git commit -m "feat(m4): refactor CLI to use ops module, add corvia status --metrics (D88)"
```

---

### Task 12: Final verification

-**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: All 385+ existing tests plus ~20 new tests pass.

-**Step 2: Verify MCP tools/list returns 18 tools**

Run: `cargo test -p corvia-server -- mcp::tests::test_tools_list`
Expected: PASS with 18 tools.

-**Step 3: Verify telemetry crate**

Run: `cargo test -p corvia-telemetry`
Expected: All telemetry tests pass.

-**Step 4: Verify ops module**

Run: `cargo test -p corvia-kernel -- ops`
Expected: All ops tests pass.

-**Step 5: Check for compilation warnings**

Run: `cargo build --workspace 2>&1 | grep warning`
Expected: No new warnings introduced.

-**Step 6: Final commit with version bump if desired**

```bash
git add -u
git commit -m "chore(m4): final verification — all M4 deliverables complete"
```

---

## Summary

| Task | Deliverable | Tests Added |
|------|-------------|-------------|
| 1 | D80: corvia-telemetry crate + TelemetryConfig | 2 |
| 2 | D82 prereq: MergeQueue rename | 0 (rename only) |
| 3 | D82: ops::system_status | 1 |
| 4 | D82: remaining ops functions | 5 |
| 5 | D81: kernel instrumentation | 0 (spans are passive) |
| 6 | D87+D89: AppState config + telemetry wiring | 0 (verified via existing tests) |
| 7 | D83: ToolTier enum + helpers | 0 (definitions only) |
| 8 | D84: 5 Tier 1 MCP tools | 5 |
| 9 | D85: 3 Tier 2 MCP tools | 3 |
| 10 | D86: 2 Tier 3 MCP tools | 2 |
| 11 | D88: CLI --metrics | 0 (manual verification) |
| 12 | Final verification | 0 |
| **Total** | **10 deliverables** | **~18 new tests** |
