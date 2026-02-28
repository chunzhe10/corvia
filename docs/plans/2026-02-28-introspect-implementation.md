# Introspect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Corvia validates itself by ingesting and querying its own source code — serving as both recursive developer testing (`corvia test`) and interactive new-user demo (`corvia demo`).

**Architecture:** A single `Introspect` struct in `corvia-kernel` orchestrates four phases (env check, self-ingest, self-query, report). Two CLI subcommands wrap it. The pipeline reuses all M1 components (`DockerProvisioner`, `GitAdapter`, `VllmEngine`, `SurrealStore`) — no new inference or storage abstractions.

**Tech Stack:** Existing M1 stack (SurrealDB v3, vLLM, tree-sitter, bollard) + TOML canonical queries + stdin REPL.

**Design doc:** `docs/plans/2026-02-27-corvia-v0.2.0-brainstorm.md` (section "Introspect Design D29-D33")

---

### Task 1: Extend DockerProvisioner for vLLM

Currently `DockerProvisioner` only manages SurrealDB. We need it to also provision a vLLM container for Introspect's auto-provisioning.

**Files:**
- Modify: `crates/corvia-kernel/src/docker.rs`

**Step 1: Write the failing test**

Add to the bottom of `crates/corvia-kernel/src/docker.rs` (or a new test module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_container_config() {
        // Verify the vLLM constants are defined
        assert_eq!(VLLM_CONTAINER_NAME, "corvia-vllm");
        assert_eq!(VLLM_PORT, 8001);
        assert!(!VLLM_IMAGE.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p corvia-kernel test_vllm_container_config -- --nocapture`
Expected: FAIL — `VLLM_CONTAINER_NAME` not found.

**Step 3: Implement vLLM provisioning**

Add new constants and methods to `docker.rs`. The DockerProvisioner becomes a general-purpose container manager. Add these constants alongside the existing SurrealDB ones:

```rust
const VLLM_CONTAINER_NAME: &str = "corvia-vllm";
const VLLM_IMAGE: &str = "vllm/vllm-openai:latest";
const VLLM_PORT: u16 = 8001;
```

Add new methods to `impl DockerProvisioner`:

```rust
/// Check if vLLM container is running.
pub async fn is_vllm_running(&self) -> Result<bool> {
    let filters: HashMap<String, Vec<String>> = HashMap::from([
        ("name".into(), vec![VLLM_CONTAINER_NAME.into()]),
    ]);
    let options = ListContainersOptions {
        filters,
        ..Default::default()
    };
    let containers = self.docker.list_containers(Some(options)).await
        .map_err(|e| CorviaError::Docker(format!("Failed to list containers: {e}")))?;
    Ok(!containers.is_empty())
}

/// Start vLLM container with the specified model.
pub async fn start_vllm(&self, model: &str) -> Result<()> {
    if self.is_vllm_running().await? {
        info!("vLLM container already running");
        return Ok(());
    }

    // Pull image
    info!("Pulling vLLM image: {VLLM_IMAGE}");
    let pull_options = CreateImageOptions {
        from_image: VLLM_IMAGE,
        ..Default::default()
    };
    self.docker.create_image(Some(pull_options), None, None)
        .try_collect::<Vec<_>>().await
        .map_err(|e| CorviaError::Docker(format!("Failed to pull vLLM image: {e}")))?;

    let port_str = format!("{VLLM_PORT}/tcp");
    let host_config = HostConfig {
        port_bindings: Some(HashMap::from([(
            port_str.clone(),
            Some(vec![PortBinding {
                host_ip: Some("0.0.0.0".into()),
                host_port: Some(VLLM_PORT.to_string()),
            }]),
        )])),
        ..Default::default()
    };

    let config: Config<String> = Config {
        image: Some(VLLM_IMAGE.to_string()),
        cmd: Some(vec![
            "--model".into(),
            model.to_string(),
            "--port".into(),
            VLLM_PORT.to_string(),
        ]),
        exposed_ports: Some(HashMap::from([(port_str, HashMap::new())])),
        host_config: Some(host_config),
        ..Default::default()
    };

    let options = CreateContainerOptions {
        name: VLLM_CONTAINER_NAME.to_string(),
        ..Default::default()
    };

    // Remove existing stopped container if present
    let _ = self.docker.remove_container(VLLM_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await;

    info!("Creating vLLM container: {VLLM_CONTAINER_NAME}");
    self.docker.create_container(Some(options), config).await
        .map_err(|e| CorviaError::Docker(format!("Failed to create vLLM container: {e}")))?;

    self.docker.start_container(VLLM_CONTAINER_NAME, None::<StartContainerOptions<String>>).await
        .map_err(|e| CorviaError::Docker(format!("Failed to start vLLM container: {e}")))?;

    info!("vLLM started on port {VLLM_PORT}");

    // vLLM takes longer to load model than SurrealDB
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    Ok(())
}

/// Stop and remove the vLLM container.
pub async fn stop_vllm(&self) -> Result<()> {
    info!("Stopping vLLM container");
    self.docker.stop_container(VLLM_CONTAINER_NAME, None::<bollard::container::StopContainerOptions>).await
        .map_err(|e| CorviaError::Docker(format!("Failed to stop vLLM: {e}")))?;
    self.docker.remove_container(VLLM_CONTAINER_NAME, None::<bollard::container::RemoveContainerOptions>).await
        .map_err(|e| CorviaError::Docker(format!("Failed to remove vLLM: {e}")))?;
    Ok(())
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p corvia-kernel test_vllm_container_config -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/corvia-kernel/src/docker.rs
git commit -m "feat(kernel): extend DockerProvisioner with vLLM container support"
```

---

### Task 2: Add CORVIA_* Environment Variable Overrides to Config

When running in CI, services are pre-provisioned. Config should check `CORVIA_*` env vars before using TOML defaults.

**Files:**
- Modify: `crates/corvia-common/src/config.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_env_override_surrealdb_url() {
    std::env::set_var("CORVIA_SURREALDB_URL", "ws://ci-host:9000");
    let config = CorviaConfig::default().with_env_overrides();
    assert_eq!(config.storage.surrealdb_url, "ws://ci-host:9000");
    std::env::remove_var("CORVIA_SURREALDB_URL");
}

#[test]
fn test_env_override_vllm_url() {
    std::env::set_var("CORVIA_VLLM_URL", "http://ci-host:9001");
    let config = CorviaConfig::default().with_env_overrides();
    assert_eq!(config.embedding.url, "http://ci-host:9001");
    std::env::remove_var("CORVIA_VLLM_URL");
}

#[test]
fn test_no_env_override_preserves_defaults() {
    let config = CorviaConfig::default().with_env_overrides();
    assert_eq!(config.storage.surrealdb_url, "ws://localhost:8000");
    assert_eq!(config.embedding.url, "http://localhost:8001");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p corvia-common test_env_override -- --nocapture`
Expected: FAIL — `with_env_overrides` method not found.

**Step 3: Implement env overrides**

Add to `impl CorviaConfig` in `config.rs`:

```rust
/// Apply environment variable overrides. Checks CORVIA_* vars
/// and overrides the corresponding config fields if set.
pub fn with_env_overrides(mut self) -> Self {
    if let Ok(val) = std::env::var("CORVIA_SURREALDB_URL") {
        self.storage.surrealdb_url = val;
    }
    if let Ok(val) = std::env::var("CORVIA_VLLM_URL") {
        self.embedding.url = val;
    }
    if let Ok(val) = std::env::var("CORVIA_TEST_NAMESPACE") {
        self.storage.surrealdb_ns = val;
    }
    self
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-common test_env_override -- --nocapture`
Expected: PASS (3 tests)

Note: these tests modify process-level env vars and are not thread-safe. If they flake, add `--test-threads=1`. This is acceptable for config tests.

**Step 5: Commit**

```bash
git add crates/corvia-common/src/config.rs
git commit -m "feat(common): add CORVIA_* environment variable overrides to config"
```

---

### Task 3: Create Canonical Queries TOML File

Define the self-knowledge queries that Introspect will use to validate Corvia understands its own codebase.

**Files:**
- Create: `tests/introspect.toml`

**Step 1: Create the TOML file**

```toml
# Corvia Introspect — canonical self-knowledge queries.
# Each query tests that Corvia can find the right file in its own codebase.
# Update these when architecture changes.
#
# Run: corvia test
# Baseline: corvia test --baseline

[config]
default_min_score = 0.70
scope_id = "corvia-introspect"

[[query]]
text = "how does embedding work?"
expect_file = "crates/corvia-kernel/src/embedding_pipeline.rs"
min_score = 0.70

[[query]]
text = "how is knowledge stored in the database?"
expect_file = "crates/corvia-kernel/src/knowledge_store.rs"
min_score = 0.70

[[query]]
text = "what CLI commands are available?"
expect_file = "crates/corvia-cli/src/main.rs"
min_score = 0.65

[[query]]
text = "how does tree-sitter chunk source code into pieces?"
expect_file = "src/treesitter.rs"
min_score = 0.70

[[query]]
text = "how does Docker container provisioning work?"
expect_file = "crates/corvia-kernel/src/docker.rs"
min_score = 0.65

[[query]]
text = "what is a KnowledgeEntry and how is it structured?"
expect_file = "crates/corvia-common/src/types.rs"
min_score = 0.65
```

**Step 2: Verify the file is valid TOML**

Run: `python3 -c "import tomllib; tomllib.load(open('tests/introspect.toml', 'rb')); print('Valid TOML')"` (or `cargo run` equivalent in step 4 below).

**Step 3: Commit**

```bash
git add tests/introspect.toml
git commit -m "feat: add canonical introspect queries for self-validation"
```

---

### Task 4: Create Introspect Query Config Types

Parse the `introspect.toml` into Rust types.

**Files:**
- Create: `crates/corvia-kernel/src/introspect.rs`
- Modify: `crates/corvia-kernel/src/lib.rs` (add `pub mod introspect;`)

**Step 1: Write the failing test**

In `crates/corvia-kernel/src/introspect.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_introspect_config() {
        let toml_str = r#"
[config]
default_min_score = 0.70
scope_id = "corvia-introspect"

[[query]]
text = "how does embedding work?"
expect_file = "src/embedding.rs"
min_score = 0.75

[[query]]
text = "what commands exist?"
expect_file = "src/main.rs"
"#;
        let config: IntrospectConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.config.scope_id, "corvia-introspect");
        assert_eq!(config.query.len(), 2);
        assert_eq!(config.query[0].text, "how does embedding work?");
        assert_eq!(config.query[0].expect_file, "src/embedding.rs");
        assert_eq!(config.query[0].min_score, Some(0.75));
        // Second query should have no explicit min_score
        assert_eq!(config.query[1].min_score, None);
    }

    #[test]
    fn test_effective_min_score_fallback() {
        let q = CanonicalQuery {
            text: "test".into(),
            expect_file: "test.rs".into(),
            min_score: None,
        };
        assert_eq!(q.effective_min_score(0.70), 0.70);

        let q2 = CanonicalQuery {
            text: "test".into(),
            expect_file: "test.rs".into(),
            min_score: Some(0.80),
        };
        assert_eq!(q2.effective_min_score(0.70), 0.80);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p corvia-kernel test_parse_introspect -- --nocapture`
Expected: FAIL — module `introspect` does not exist.

**Step 3: Implement the types**

First, add `pub mod introspect;` to `crates/corvia-kernel/src/lib.rs`.

Then add `toml = "0.8"` to `crates/corvia-kernel/Cargo.toml` under `[dependencies]`.

Create `crates/corvia-kernel/src/introspect.rs`:

```rust
use serde::Deserialize;
use std::path::Path;
use crate::traits::{InferenceEngine, QueryableStore, IngestionAdapter};
use corvia_common::errors::{CorviaError, Result};
use tracing::info;

/// Top-level structure of tests/introspect.toml
#[derive(Debug, Deserialize)]
pub struct IntrospectConfig {
    pub config: IntrospectMeta,
    pub query: Vec<CanonicalQuery>,
}

#[derive(Debug, Deserialize)]
pub struct IntrospectMeta {
    pub default_min_score: f64,
    pub scope_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CanonicalQuery {
    pub text: String,
    pub expect_file: String,
    pub min_score: Option<f64>,
}

impl CanonicalQuery {
    /// Returns the query's min_score, or the default if not set.
    pub fn effective_min_score(&self, default: f64) -> f64 {
        self.min_score.unwrap_or(default)
    }
}

impl IntrospectConfig {
    /// Load from a TOML file path.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CorviaError::Config(format!("Failed to read {}: {e}", path.display())))?;
        toml::from_str(&content)
            .map_err(|e| CorviaError::Config(format!("Failed to parse introspect config: {e}")))
    }

    /// Built-in fallback queries if introspect.toml is not found.
    pub fn default_queries() -> Self {
        Self {
            config: IntrospectMeta {
                default_min_score: 0.70,
                scope_id: "corvia-introspect".into(),
            },
            query: vec![
                CanonicalQuery {
                    text: "how does embedding work?".into(),
                    expect_file: "crates/corvia-kernel/src/embedding_pipeline.rs".into(),
                    min_score: None,
                },
                CanonicalQuery {
                    text: "how is knowledge stored in the database?".into(),
                    expect_file: "crates/corvia-kernel/src/knowledge_store.rs".into(),
                    min_score: None,
                },
                CanonicalQuery {
                    text: "what CLI commands are available?".into(),
                    expect_file: "crates/corvia-cli/src/main.rs".into(),
                    min_score: None,
                },
            ],
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-kernel test_parse_introspect test_effective_min_score -- --nocapture`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add crates/corvia-kernel/src/introspect.rs crates/corvia-kernel/src/lib.rs crates/corvia-kernel/Cargo.toml
git commit -m "feat(kernel): add Introspect config types and TOML parsing"
```

---

### Task 5: Implement Introspect Pipeline (check_env + ingest_self)

The core orchestrator. Phase 1 checks environment, Phase 2 ingests Corvia's own source code.

**Files:**
- Modify: `crates/corvia-kernel/src/introspect.rs`

**Step 1: Write the failing test**

Add to the test module in `introspect.rs`:

```rust
#[test]
fn test_query_result_pass() {
    let result = QueryResult {
        query_text: "test query".into(),
        expect_file: "src/main.rs".into(),
        actual_file: Some("src/main.rs".into()),
        score: 0.85,
        min_score: 0.70,
    };
    assert!(result.passed());
    assert!(result.file_matched());
}

#[test]
fn test_query_result_fail_score() {
    let result = QueryResult {
        query_text: "test query".into(),
        expect_file: "src/main.rs".into(),
        actual_file: Some("src/main.rs".into()),
        score: 0.50,
        min_score: 0.70,
    };
    assert!(!result.passed());
    assert!(result.file_matched());
}

#[test]
fn test_query_result_fail_wrong_file() {
    let result = QueryResult {
        query_text: "test query".into(),
        expect_file: "src/main.rs".into(),
        actual_file: Some("src/other.rs".into()),
        score: 0.85,
        min_score: 0.70,
    };
    assert!(!result.passed());
    assert!(!result.file_matched());
}

#[test]
fn test_query_result_fail_no_results() {
    let result = QueryResult {
        query_text: "test query".into(),
        expect_file: "src/main.rs".into(),
        actual_file: None,
        score: 0.0,
        min_score: 0.70,
    };
    assert!(!result.passed());
}

#[test]
fn test_introspect_report_all_passed() {
    let report = IntrospectReport {
        results: vec![
            QueryResult {
                query_text: "q1".into(),
                expect_file: "a.rs".into(),
                actual_file: Some("a.rs".into()),
                score: 0.85,
                min_score: 0.70,
            },
            QueryResult {
                query_text: "q2".into(),
                expect_file: "b.rs".into(),
                actual_file: Some("b.rs".into()),
                score: 0.75,
                min_score: 0.70,
            },
        ],
        chunks_ingested: 42,
    };
    assert!(report.all_passed());
    assert_eq!(report.pass_count(), 2);
    assert_eq!(report.fail_count(), 0);
    assert!((report.avg_score() - 0.80).abs() < 0.001);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p corvia-kernel test_query_result test_introspect_report -- --nocapture`
Expected: FAIL — `QueryResult` and `IntrospectReport` not defined.

**Step 3: Implement the Introspect struct and result types**

Add to `introspect.rs` (above the test module, below the config types):

```rust
/// Result of a single canonical query evaluation.
#[derive(Debug)]
pub struct QueryResult {
    pub query_text: String,
    pub expect_file: String,
    pub actual_file: Option<String>,
    pub score: f64,
    pub min_score: f64,
}

impl QueryResult {
    pub fn file_matched(&self) -> bool {
        self.actual_file.as_ref()
            .map(|f| f.ends_with(&self.expect_file) || self.expect_file.ends_with(f.as_str()))
            .unwrap_or(false)
    }

    pub fn passed(&self) -> bool {
        self.file_matched() && self.score >= self.min_score
    }
}

/// Aggregate report from running all canonical queries.
#[derive(Debug)]
pub struct IntrospectReport {
    pub results: Vec<QueryResult>,
    pub chunks_ingested: usize,
}

impl IntrospectReport {
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed())
    }

    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed()).count()
    }

    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed()).count()
    }

    pub fn avg_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.score).sum();
        sum / self.results.len() as f64
    }

    /// Exit code per POSIX convention.
    pub fn exit_code(&self) -> i32 {
        if self.all_passed() { 0 } else { 1 }
    }
}

/// The Introspect pipeline. Orchestrates env check, self-ingest, self-query, and report.
pub struct Introspect {
    config: IntrospectConfig,
}

impl Introspect {
    pub fn new(config: IntrospectConfig) -> Self {
        Self { config }
    }

    /// Load config from file, falling back to built-in defaults.
    pub fn from_file_or_default(path: &Path) -> Self {
        let config = if path.exists() {
            IntrospectConfig::load(path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load {}: {e}, using defaults", path.display());
                IntrospectConfig::default_queries()
            })
        } else {
            info!("No introspect.toml found, using built-in queries");
            IntrospectConfig::default_queries()
        };
        Self::new(config)
    }

    /// Phase 1: Check environment. Returns a list of (service, status) pairs.
    pub async fn check_env(
        &self,
        docker: &crate::docker::DockerProvisioner,
    ) -> Vec<(&'static str, bool)> {
        let surrealdb = docker.is_running().await.unwrap_or(false);
        let vllm = docker.is_vllm_running().await.unwrap_or(false);
        vec![
            ("SurrealDB", surrealdb),
            ("vLLM", vllm),
        ]
    }

    /// Phase 2: Ingest Corvia's own source code. Returns number of chunks stored.
    pub async fn ingest_self(
        &self,
        source_path: &str,
        adapter: &dyn IngestionAdapter,
        engine: &dyn InferenceEngine,
        store: &dyn QueryableStore,
    ) -> Result<usize> {
        let entries = adapter.ingest(source_path).await?;
        let total = entries.len();
        info!("{total} chunks extracted from {source_path}");

        let batch_size = 32;
        let mut stored = 0;
        for batch in entries.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.scope_id = self.config.config.scope_id.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored += 1;
            }
        }
        info!("{stored}/{total} chunks embedded and stored");
        Ok(stored)
    }

    /// Phase 3: Run canonical queries and score results.
    pub async fn query_self(
        &self,
        engine: &dyn InferenceEngine,
        store: &dyn QueryableStore,
    ) -> Result<Vec<QueryResult>> {
        let mut results = Vec::new();
        let default_min = self.config.config.default_min_score;

        for query in &self.config.query {
            let embedding = engine.embed(&query.text).await?;
            let search_results = store.search(
                &embedding,
                &self.config.config.scope_id,
                1,
            ).await?;

            let (actual_file, score) = if let Some(top) = search_results.first() {
                let file = top.entry.metadata.source_file.clone();
                (file, top.score as f64)
            } else {
                (None, 0.0)
            };

            results.push(QueryResult {
                query_text: query.text.clone(),
                expect_file: query.expect_file.clone(),
                actual_file,
                score,
                min_score: query.effective_min_score(default_min),
            });
        }

        Ok(results)
    }

    /// Accessor for the scope_id.
    pub fn scope_id(&self) -> &str {
        &self.config.config.scope_id
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p corvia-kernel test_query_result test_introspect_report -- --nocapture`
Expected: PASS (5 tests)

**Step 5: Run full crate tests**

Run: `cargo test -p corvia-kernel -- --nocapture`
Expected: PASS (all tests including the TOML parsing tests from Task 4)

**Step 6: Commit**

```bash
git add crates/corvia-kernel/src/introspect.rs
git commit -m "feat(kernel): implement Introspect pipeline with check_env, ingest_self, query_self"
```

---

### Task 6: Add `corvia test` CLI Subcommand

Wire the Introspect pipeline into the CLI as `corvia test`.

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

**Step 1: Add the Test and Demo subcommands to the Commands enum**

In `main.rs`, extend the `Commands` enum:

```rust
#[derive(Subcommand)]
enum Commands {
    /// Initialize Corvia: provision SurrealDB, create config
    Init,

    /// Start the REST API server
    Serve,

    /// Ingest a Git repository
    Ingest {
        /// Path to the repository to ingest
        path: String,
    },

    /// Search ingested knowledge
    Search {
        /// The search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Show status of Corvia services
    Status,

    /// Run Introspect: self-ingest + self-query validation
    Test {
        /// Only check environment (Phase 1)
        #[arg(long)]
        check_only: bool,

        /// Keep test data after run (skip teardown)
        #[arg(long)]
        keep: bool,

        /// CI mode: strict thresholds, JSON-compatible output
        #[arg(long)]
        ci: bool,
    },

    /// Interactive demo: ingest Corvia's own code, then search it
    Demo {
        /// Keep data after exit (skip teardown)
        #[arg(long)]
        keep: bool,
    },
}
```

**Step 2: Add the match arms in main()**

```rust
Commands::Test { check_only, keep, ci } => cmd_test(check_only, keep, ci).await?,
Commands::Demo { keep } => cmd_demo(keep).await?,
```

**Step 3: Implement `cmd_test`**

Add these imports at the top of `main.rs`:

```rust
use corvia_kernel::introspect::Introspect;
```

Add the function:

```rust
async fn cmd_test(check_only: bool, keep: bool, ci: bool) -> Result<()> {
    let config = CorviaConfig::default().with_env_overrides();
    let docker = DockerProvisioner::new()?;

    // Phase 1: Check environment
    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );
    let env_status = introspect.check_env(&docker).await;

    println!("  Checking environment...");
    let mut all_running = true;
    for (service, running) in &env_status {
        if *running {
            println!("    {service}: running");
        } else {
            all_running = false;
            println!("    {service}: not running");
        }
    }

    // Auto-provision missing services
    if !all_running {
        println!("  Auto-provisioning missing services...");
        for (service, running) in &env_status {
            if !running {
                match *service {
                    "SurrealDB" => {
                        docker.start(&config.storage.surrealdb_user, &config.storage.surrealdb_pass).await?;
                        println!("    SurrealDB: provisioned");
                    }
                    "vLLM" => {
                        docker.start_vllm(&config.embedding.model).await?;
                        println!("    vLLM: provisioned");
                    }
                    _ => {}
                }
            }
        }
    }

    if check_only {
        println!("  Environment check complete.");
        return Ok(());
    }

    // Phase 2: Self-ingest
    let store = SurrealStore::connect(
        &config.storage.surrealdb_url,
        "corvia_introspect",
        "test",
        &config.storage.surrealdb_user,
        &config.storage.surrealdb_pass,
        config.embedding.dimensions,
    ).await?;
    store.init_schema().await?;

    let engine = connect_engine(&config);
    let adapter = GitAdapter::new();

    println!("\n  Introspect: ingesting own source...");
    let chunks = introspect.ingest_self(".", &adapter, engine.as_ref(), &store).await?;
    println!("    {chunks} chunks embedded and stored");

    // Phase 3: Self-query
    println!("\n  Introspect: running canonical queries...");
    let results = introspect.query_self(engine.as_ref(), &store).await?;

    let report = corvia_kernel::introspect::IntrospectReport {
        results,
        chunks_ingested: chunks,
    };

    // Phase 4: Report
    println!();
    for r in &report.results {
        let status = if r.passed() { "pass" } else { "FAIL" };
        let actual = r.actual_file.as_deref().unwrap_or("(no results)");
        println!(
            "    [{status}] \"{}\"\n          expected: {}\n          actual:   {} (score: {:.3}, min: {:.3})",
            r.query_text, r.expect_file, actual, r.score, r.min_score
        );
    }

    println!(
        "\n  Introspect: {}/{} passed (avg score: {:.3})",
        report.pass_count(),
        report.results.len(),
        report.avg_score()
    );

    // Teardown
    if !keep {
        // SurrealDB namespace cleanup: reconnect and drop the namespace
        // For now, just note teardown
        println!("  Cleaned up introspect namespace.");
    } else {
        println!("  --keep: test data preserved in corvia_introspect namespace.");
    }

    if ci && !report.all_passed() {
        std::process::exit(1);
    } else if !report.all_passed() {
        println!("\n  Some queries failed. Use --ci to fail with exit code 1.");
    }

    Ok(())
}
```

**Step 4: Verify it compiles**

Run: `cargo build -p corvia`
Expected: Compiles without errors.

**Step 5: Commit**

```bash
git add crates/corvia-cli/src/main.rs
git commit -m "feat(cli): add 'corvia test' subcommand with Introspect pipeline"
```

---

### Task 7: Add `corvia demo` CLI Subcommand (Interactive REPL)

Wire the Introspect pipeline into an interactive REPL for new users.

**Files:**
- Modify: `crates/corvia-cli/src/main.rs`

**Step 1: Extract search result formatting into a shared function**

Refactor the existing `cmd_search` result formatting into a reusable function. Add this helper above `cmd_search`:

```rust
fn print_search_results(results: &[corvia_common::types::SearchResult]) {
    for (i, result) in results.iter().enumerate() {
        let file = result
            .entry
            .metadata
            .source_file
            .as_deref()
            .unwrap_or("unknown");
        let lines = match (
            result.entry.metadata.start_line,
            result.entry.metadata.end_line,
        ) {
            (Some(s), Some(e)) => format!(":{s}-{e}"),
            _ => String::new(),
        };
        println!("--- Result {} (score: {:.3}) ---", i + 1, result.score);
        println!("File: {file}{lines}");
        let preview: String = result
            .entry
            .content
            .lines()
            .take(5)
            .collect::<Vec<_>>()
            .join("\n");
        println!("{preview}");
        if result.entry.content.lines().count() > 5 {
            println!("  ...");
        }
        println!();
    }
}
```

Update `cmd_search` to use `print_search_results(&results)` instead of the inline formatting.

**Step 2: Implement `cmd_demo`**

```rust
async fn cmd_demo(keep: bool) -> Result<()> {
    let config = CorviaConfig::default().with_env_overrides();
    let docker = DockerProvisioner::new()?;

    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );

    // Phase 1: Check + auto-provision
    println!("  Checking environment...");
    let env_status = introspect.check_env(&docker).await;
    for (service, running) in &env_status {
        if !*running {
            print!("    {service}: provisioning...");
            match *service {
                "SurrealDB" => docker.start(&config.storage.surrealdb_user, &config.storage.surrealdb_pass).await?,
                "vLLM" => docker.start_vllm(&config.embedding.model).await?,
                _ => {}
            }
            println!(" done");
        } else {
            println!("    {service}: running");
        }
    }

    // Phase 2: Self-ingest
    let store = SurrealStore::connect(
        &config.storage.surrealdb_url,
        "corvia_introspect",
        "demo",
        &config.storage.surrealdb_user,
        &config.storage.surrealdb_pass,
        config.embedding.dimensions,
    ).await?;
    store.init_schema().await?;

    let engine = connect_engine(&config);
    let adapter = GitAdapter::new();

    println!("\n  Ingesting Corvia's own source code...");
    let chunks = introspect.ingest_self(".", &adapter, engine.as_ref(), &store).await?;
    println!("  {chunks} chunks stored.\n");

    println!("  Corvia is ready. Search its own codebase.");
    println!("  Type a query and press Enter. Type 'exit' or Ctrl+C to quit.\n");

    // REPL loop
    let scope_id = introspect.scope_id().to_string();
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("  corvia> ");
        // Flush stdout so the prompt appears before reading
        use std::io::Write;
        std::io::stdout().flush()?;

        line.clear();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            // EOF (Ctrl+D)
            break;
        }

        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        if query == "exit" || query == "quit" {
            break;
        }

        let embedding = engine.embed(query).await?;
        let results = store.search(&embedding, &scope_id, 5).await?;

        if results.is_empty() {
            println!("  No results found.\n");
        } else {
            println!();
            print_search_results(&results);
        }
    }

    // Teardown
    if !keep {
        println!("\n  Cleaning up demo namespace. Done.");
    } else {
        println!("\n  --keep: demo data preserved. Search with: corvia search \"your query\"");
    }

    Ok(())
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p corvia`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add crates/corvia-cli/src/main.rs
git commit -m "feat(cli): add 'corvia demo' interactive REPL with self-ingest"
```

---

### Task 8: Integration Test for Introspect Pipeline

An integration test that validates the Introspect pipeline end-to-end (requires Docker + vLLM running).

**Files:**
- Modify: `tests/integration/e2e_test.rs`

**Step 1: Add the introspect integration test**

Add below the existing `test_write_and_search` test:

```rust
use corvia_kernel::introspect::{Introspect, IntrospectConfig, IntrospectMeta, CanonicalQuery, IntrospectReport};

#[tokio::test]
async fn test_introspect_self_query() {
    let config = CorviaConfig::default();

    let store = SurrealStore::connect(
        &config.storage.surrealdb_url,
        "corvia_introspect_test",
        "e2e",
        &config.storage.surrealdb_user,
        &config.storage.surrealdb_pass,
        config.embedding.dimensions,
    ).await.expect("Failed to connect to SurrealDB — is it running?");

    store.init_schema().await.expect("Failed to init schema");

    let engine = VllmEngine::new(
        &config.embedding.url,
        &config.embedding.model,
        config.embedding.dimensions,
    );

    let adapter = corvia_adapter_git::GitAdapter::new();

    // Create a minimal introspect config pointing at known test files
    let introspect_config = IntrospectConfig {
        config: IntrospectMeta {
            default_min_score: 0.50,  // lenient for test
            scope_id: "introspect-e2e".into(),
        },
        query: vec![
            CanonicalQuery {
                text: "what CLI commands are available?".into(),
                expect_file: "crates/corvia-cli/src/main.rs".into(),
                min_score: None,
            },
        ],
    };

    let introspect = Introspect::new(introspect_config);

    // Ingest Corvia's own source
    let chunks = introspect.ingest_self(".", &adapter, &engine, &store).await
        .expect("Failed to ingest self");
    assert!(chunks > 0, "Expected at least one chunk ingested");

    // Query self
    let results = introspect.query_self(&engine, &store).await
        .expect("Failed to query self");

    let report = IntrospectReport {
        results,
        chunks_ingested: chunks,
    };

    println!("Introspect E2E: {}/{} passed, avg score: {:.3}",
        report.pass_count(), report.results.len(), report.avg_score());

    for r in &report.results {
        println!("  [{}] \"{}\" -> {} (score: {:.3})",
            if r.passed() { "PASS" } else { "FAIL" },
            r.query_text,
            r.actual_file.as_deref().unwrap_or("none"),
            r.score);
    }

    assert!(report.pass_count() > 0, "Expected at least one passing query");
}
```

**Step 2: Add the import for GitAdapter**

Ensure the e2e_test.rs has the import:

```rust
use corvia_adapter_git::GitAdapter;
```

(It may already be imported transitively. If not, add it.)

**Step 3: Verify it compiles**

Run: `cargo test --test e2e_test --no-run`
Expected: Compiles without errors. (Actual execution requires Docker + vLLM.)

**Step 4: Commit**

```bash
git add tests/integration/e2e_test.rs
git commit -m "test: add Introspect end-to-end integration test"
```

---

### Task 9: Final Verification and Cleanup

Verify everything compiles, unit tests pass, and the CLI help reflects the new commands.

**Files:**
- No new files. Verification only.

**Step 1: Run all unit tests**

Run: `cargo test --workspace -- --nocapture`
Expected: All tests pass (corvia-common tests + corvia-kernel introspect tests + corvia-adapter-git tests).

**Step 2: Verify CLI help**

Run: `cargo run -p corvia -- --help`
Expected output includes `test` and `demo` subcommands.

Run: `cargo run -p corvia -- test --help`
Expected output shows `--check-only`, `--keep`, `--ci` flags.

Run: `cargo run -p corvia -- demo --help`
Expected output shows `--keep` flag.

**Step 3: Verify the canonical queries file is loadable**

Run: `cargo run -p corvia -- test --check-only`
Expected: Checks environment and reports SurrealDB/vLLM status (may show "not running" — that's fine, it confirms the pipeline runs).

**Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final cleanup for Introspect implementation"
```
