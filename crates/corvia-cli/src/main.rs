//! Corvia CLI — organizational reasoning memory for AI agents.
//!
//! The `corvia` binary is the primary user interface for the Corvia knowledge
//! system. It manages workspaces, ingests repositories, runs semantic search,
//! and coordinates multi-agent sessions.
//!
//! # Commands
//!
//! | Command | Purpose |
//! |---------|---------|
//! | `init` | Initialize a new Corvia store (`--store lite\|postgres`) |
//! | `ingest` | Ingest a Git repository (or all workspace repos) |
//! | `search` | Semantic search across ingested knowledge |
//! | `serve` | Start the REST API and MCP server |
//! | `reason` | Run health checks and reasoning over a scope |
//! | `history` | Show the supersession chain for an entry |
//! | `evolution` | Show entries that changed within a time range |
//! | `graph` | Traverse the knowledge graph from a starting node |
//! | `relate` | Create a directed edge between two entries |
//! | `agent` | Multi-agent session management (start, list, commit, merge) |
//! | `workspace` | Workspace lifecycle (init, add, list, status, ingest) |
//! | `migrate` | Migrate data between storage backends (`--to lite\|postgres`) |
//! | `demo` | Run the built-in demo workspace |
//!
//! See the [README](https://github.com/corvia/corvia) and
//! [ARCHITECTURE.md](https://github.com/corvia/corvia/blob/master/ARCHITECTURE.md)
//! for the full project overview.

mod hooks;
mod server_client;
mod upgrade;
mod workspace;

use anyhow::Result;
use clap::{Parser, Subcommand};
use corvia_common::config::CorviaConfig;
use corvia_kernel::ollama_provisioner::OllamaProvisioner;
use corvia_kernel::agent_coordinator::AgentCoordinator;
use corvia_kernel::traits::{InferenceEngine, IngestionAdapter, QueryableStore, GraphStore, TemporalStore};
use corvia_adapter_git::GitAdapter;
use corvia_kernel::introspect::Introspect;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "corvia")]
#[command(about = "Code memory for AI agents — point, ingest, search")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Corvia (LiteStore by default, --store to select backend)
    Init {
        /// Storage backend: lite (default), postgres
        #[arg(long, default_value = "lite")]
        store: String,
    },

    /// Start the REST API server (includes MCP endpoint)
    Serve,

    /// Ingest a repository (or all workspace repos if no path given)
    Ingest {
        /// Path to repository (optional in workspace mode)
        path: Option<String>,
        /// Incremental mode: only re-index changed files
        #[arg(long)]
        incremental: bool,
        /// Specific files to re-index (requires --incremental)
        #[arg(long, num_args = 1.., requires = "incremental")]
        files: Vec<String>,
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
    Status {
        /// Show extended metrics (store type, inference, telemetry, agents, adapters)
        #[arg(long)]
        metrics: bool,
    },

    /// Run Introspect: self-ingest + self-query validation
    Test {
        /// Only check environment (Phase 1)
        #[arg(long)]
        check_only: bool,

        /// Keep test data after run (skip teardown)
        #[arg(long)]
        keep: bool,

        /// CI mode: strict thresholds
        #[arg(long)]
        ci: bool,
    },

    /// Interactive demo: ingest Corvia's own code, then search it
    Demo {
        /// Keep data after exit (skip teardown)
        #[arg(long)]
        keep: bool,
    },

    /// Rebuild LiteStore indexes from knowledge files
    Rebuild,

    /// Manage registered agents
    Agent {
        #[command(subcommand)]
        command: AgentCommands,
    },

    /// Workspace management for multi-repo projects
    Workspace {
        #[command(subcommand)]
        command: WorkspaceCommands,
    },

    /// Show the supersession history of an entry
    History {
        /// Entry UUID
        entry_id: String,
    },

    /// Show entries that changed within a time range
    Evolution {
        /// Scope to analyze
        #[arg(long)]
        scope: Option<String>,

        /// Time range: e.g., "7d", "1d", "30d"
        #[arg(long, default_value = "7d")]
        since: String,
    },

    /// Show graph edges for an entry or scope
    Graph {
        /// Entry UUID (show edges for this entry)
        entry_id: Option<String>,

        /// Scope to filter
        #[arg(long)]
        scope: Option<String>,

        /// Filter by relation type
        #[arg(long)]
        relation: Option<String>,
    },

    /// Create a directed edge between two entries
    Relate {
        /// Source entry UUID
        from: String,
        /// Relation type (e.g., "depends_on", "imports")
        relation: String,
        /// Target entry UUID
        to: String,
    },

    /// Run reasoning checks on the knowledge store
    Reason {
        /// Scope to analyze (uses config scope_id if not specified)
        #[arg(long)]
        scope: Option<String>,

        /// Run only a specific check type
        #[arg(long)]
        check: Option<String>,

        /// Include LLM-powered checks (requires `reasoning` config)
        #[arg(long)]
        llm: bool,
    },

    /// Migrate data between storage backends
    Migrate {
        /// Target storage backend: lite, postgres
        #[arg(long)]
        to: String,
        /// Show what would be migrated without making changes
        #[arg(long)]
        dry_run: bool,
    },

    /// Manage the corvia-inference server (reload models, show status)
    Inference {
        #[command(subcommand)]
        command: InferenceCommands,
    },

    /// Run retrieval quality benchmarks against the knowledge store
    Bench {
        #[command(subcommand)]
        command: BenchCommands,
    },

    /// Manage Claude Code lifecycle hooks
    Hooks {
        #[command(subcommand)]
        command: HooksCommands,
    },
}

#[derive(Subcommand)]
enum HooksCommands {
    /// Execute a hook handler for the given event (called by Claude Code)
    Run {
        /// Event type: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, SessionEnd
        #[arg(long)]
        event: String,
        /// Specific handler to run (e.g., doc-placement, write-reminder)
        #[arg(long)]
        handler: Option<String>,
    },
    /// Generate .claude/settings.json hook entries
    Init,
    /// Show which hooks are enabled and registered
    Status,
    /// Gzip and ingest stale session files (no SessionEnd received)
    Sweep {
        /// Maximum age in hours before a session file is considered stale
        #[arg(long, default_value = "4")]
        max_age_hours: u64,
    },
}

#[derive(Subcommand)]
enum BenchCommands {
    /// Run the eval suite and print results
    Run {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8020")]
        server: String,
        /// Number of results to retrieve per query
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Run A/B test: compare vector vs graph_expand
        #[arg(long)]
        ab: bool,
    },
    /// Show the latest benchmark results
    Report,
    /// Run A/B comparison: vector-only vs graph-expanded retrieval
    Compare {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8020")]
        server: String,
        /// Number of results to retrieve per query
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

#[derive(Subcommand)]
enum AgentCommands {
    /// List registered agents
    List,
    /// Show sessions for an agent
    Sessions {
        /// Agent ID (e.g., "myproject::indexer")
        agent_id: String,
    },
    /// Interactive agent selection for session identity
    Connect,
}

#[derive(Subcommand)]
enum InferenceCommands {
    /// Reload loaded models with a different device/backend/kv-quant
    Reload {
        /// Device: "auto", "gpu", or "cpu"
        #[arg(long)]
        device: Option<String>,
        /// Backend override: "cuda", "openvino", or "" (auto-select)
        #[arg(long)]
        backend: Option<String>,
        /// Reload only this model (omit to reload all)
        #[arg(long)]
        model: Option<String>,
        /// KV cache quantization: "q8", "q4", "none"
        #[arg(long)]
        kv_quant: Option<String>,
        /// Enable/disable flash attention
        #[arg(long)]
        flash_attention: Option<bool>,
        /// Don't persist changes to corvia.toml
        #[arg(long)]
        no_persist: bool,
    },
    /// Show loaded models and their device/backend
    Status,
}

#[derive(Subcommand)]
enum WorkspaceCommands {
    /// Create a new workspace directory
    Create {
        /// Workspace name (becomes directory name)
        name: String,
        /// Add a repo (repeatable)
        #[arg(long = "repo", num_args = 1)]
        repos: Vec<String>,
        /// Scaffold from a template workspace
        #[arg(long)]
        template: Option<String>,
        /// Set local override for a repo (name=path)
        #[arg(long = "local", num_args = 1)]
        locals: Vec<String>,
    },
    /// Initialize workspace in current directory
    Init,
    /// Show workspace status
    Status,
    /// Add a repo to the workspace
    Add {
        /// Repository URL
        url: String,
        /// Override repo name
        #[arg(long)]
        name: Option<String>,
        /// Override namespace
        #[arg(long)]
        namespace: Option<String>,
        /// Local path override
        #[arg(long)]
        local: Option<String>,
    },
    /// Remove a repo from the workspace
    Remove {
        /// Repo name to remove
        name: String,
        /// Also delete cloned repo and knowledge entries
        #[arg(long)]
        purge: bool,
    },
    /// Ingest all workspace repos
    Ingest {
        /// Only ingest this specific repo
        #[arg(long)]
        repo: Option<String>,
        /// Re-ingest from scratch
        #[arg(long)]
        fresh: bool,
    },
    /// Clean build artifacts (target/ directories) from workspace repos
    Clean,
    /// Documentation health checks
    Docs {
        #[command(subcommand)]
        command: DocsCommands,
    },
    /// Generate enforcement hooks from corvia.toml config
    InitHooks,
}

#[derive(Subcommand)]
enum DocsCommands {
    /// Run documentation placement and coverage checks
    Check {
        /// Run only a specific check type: misplaced, temporal, coverage
        #[arg(long)]
        check: Option<String>,
        /// Auto-fix misplaced docs (requires --yes to execute)
        #[arg(long)]
        fix: bool,
        /// Confirm auto-fix moves without prompting
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> Result<()> {
    // Fast path: `corvia hooks run` runs synchronously — no tokio, no telemetry.
    // Must happen before tokio runtime is created (reqwest::blocking panics inside async).
    let cli = Cli::parse();
    if let Commands::Hooks { command: HooksCommands::Run { ref event, ref handler } } = cli.command {
        return hooks::run_hook_from_args(event, handler.as_deref());
    }

    // All other commands use the async runtime.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    // Load telemetry config if available; otherwise use defaults (stdout + text).
    let telemetry_config = CorviaConfig::config_path();
    let telem_cfg = if telemetry_config.exists() {
        CorviaConfig::load(&telemetry_config)
            .map(|c| c.telemetry)
            .unwrap_or_default()
    } else {
        corvia_common::config::TelemetryConfig::default()
    };
    // Hold the guard for the process lifetime so file logs flush on exit.
    let _telemetry_guard = match corvia_telemetry::init_telemetry(&telem_cfg) {
        Ok(guard) => Some(guard),
        Err(e) => {
            eprintln!("Warning: telemetry init failed: {e}");
            None
        }
    };

    match cli.command {
        Commands::Init { store } => cmd_init(&store).await?,
        Commands::Serve => cmd_serve().await?,
        Commands::Ingest { path, incremental, files } => cmd_ingest(path.as_deref(), incremental, &files).await?,
        Commands::Search { query, limit } => cmd_search(&query, limit).await?,
        Commands::Status { metrics } => cmd_status(metrics).await?,
        Commands::Test { check_only, keep, ci } => cmd_test(check_only, keep, ci).await?,
        Commands::Demo { keep } => cmd_demo(keep).await?,
        Commands::Rebuild => cmd_rebuild().await?,
        Commands::Agent { command } => cmd_agent(command).await?,
        Commands::Workspace { command } => cmd_workspace(command).await?,
        Commands::History { entry_id } => cmd_history(&entry_id).await?,
        Commands::Evolution { scope, since } => cmd_evolution(scope.as_deref(), &since).await?,
        Commands::Graph { entry_id, scope, relation } => cmd_graph(entry_id.as_deref(), scope.as_deref(), relation.as_deref()).await?,
        Commands::Relate { from, relation, to } => cmd_relate(&from, &relation, &to).await?,
        Commands::Reason { scope, check, llm } => cmd_reason(scope.as_deref(), check.as_deref(), llm).await?,
        Commands::Migrate { to, dry_run } => upgrade::cmd_migrate(&to, dry_run).await?,
        Commands::Inference { command } => match command {
            InferenceCommands::Reload { device, backend, model, kv_quant, flash_attention, no_persist } =>
                cmd_inference_reload(device.as_deref(), backend.as_deref(), model.as_deref(), kv_quant.as_deref(), flash_attention, no_persist).await?,
            InferenceCommands::Status => cmd_inference_status().await?,
        },
        Commands::Bench { command } => match command {
            BenchCommands::Run { server, limit, ab } => cmd_bench_run(&server, limit, ab).await?,
            BenchCommands::Report => cmd_bench_report()?,
            BenchCommands::Compare { server, limit } => cmd_bench_compare(&server, limit).await?,
        },
        Commands::Hooks { command } => match command {
            HooksCommands::Run { .. } => unreachable!("handled in fast path above"),
            HooksCommands::Init => {
                let root = std::env::current_dir()?;
                let config = load_config()?;
                hooks::settings::init_hooks(&root, &config)?;
            }
            HooksCommands::Status => {
                hooks::status::print_status()?;
            }
            HooksCommands::Sweep { max_age_hours } => {
                hooks::session::sweep_stale_sessions(max_age_hours);
            }
        },
    }

    Ok(())
}

async fn cmd_init(store: &str) -> Result<()> {
    let config = match store {
        "postgres" => {
            println!("Initializing Corvia (PostgreSQL + pgvector + vLLM)...");
            CorviaConfig::postgres_default()
        }
        "lite" => {
            println!("Initializing Corvia (LiteStore: zero Docker)...");
            CorviaConfig::default()
        }
        other => {
            anyhow::bail!("Unknown store type '{other}'. Valid options: lite, postgres");
        }
    };

    let config_path = CorviaConfig::config_path();
    if !config_path.exists() {
        config.save(&config_path)?;
        println!("  Created {}", config_path.display());
    } else {
        println!("  Config already exists: {}", config_path.display());
    }

    match store {
        "postgres" => {
            // PostgreSQL must be running (user manages via docker-compose or make postgres-up)
            println!("  Connecting to PostgreSQL...");
            let s = connect_store(&config).await?;
            s.init_schema().await?;
            println!("  PostgreSQL schema initialized at {}",
                config.storage.postgres_url.as_deref().unwrap_or("postgres://127.0.0.1:5432/corvia"));
        }
        _ => {
            // LiteStore initialization
            let s = connect_store(&config).await?;
            s.init_schema().await?;

            // Create .gitignore for ephemeral files
            let data_dir = std::path::Path::new(&config.storage.data_dir);
            let gitignore_path = data_dir.join(".gitignore");
            if !gitignore_path.exists() {
                std::fs::create_dir_all(data_dir)?;
                std::fs::write(&gitignore_path,
                    "# Ephemeral indexes (rebuilt from knowledge files)\nhnsw/\nlite_store.redb\nlite_store.redb.tmp\n"
                )?;
                println!("  Created {}", gitignore_path.display());
            }

            println!("  LiteStore initialized in {}/", config.storage.data_dir);

            // Provision inference backend
            match config.embedding.provider {
                corvia_common::config::InferenceProvider::Corvia => {
                    println!("  Provisioning Corvia inference server...");
                    let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(
                        &config.embedding.url,
                    );
                    let chat_coords = resolve_chat_model_coords(&config);
                    provisioner.ensure_ready(
                        &config.embedding.model,
                        chat_coords.as_ref(),
                        &config.inference.device,
                        &config.inference.backend,
                        &config.inference.embedding_backend,
                        &config.inference.kv_quant,
                        config.inference.flash_attention,
                    ).await?;
                    if let Some(ref cc) = chat_coords {
                        println!("  Corvia inference ready (embed: {}, chat: {})",
                            config.embedding.model, cc.name);
                    } else {
                        println!("  Corvia inference ready (embed: {}, chat: disabled)",
                            config.embedding.model);
                    }
                }
                corvia_common::config::InferenceProvider::Ollama => {
                    println!("  Provisioning Ollama...");
                    let provisioner = OllamaProvisioner::new(&config.embedding.url);
                    provisioner.ensure_ready(&config.embedding.model).await?;
                    println!("  Ollama ready (model: {})", config.embedding.model);
                }
                corvia_common::config::InferenceProvider::Vllm => {
                    // vLLM is provisioned via Docker — nothing needed here
                }
            }
        }
    }

    println!("\nNext: corvia ingest <path-to-repo>");
    Ok(())
}

async fn cmd_serve() -> Result<()> {
    let config = load_config()?;

    // Bind the TCP listener FIRST so the port is occupied immediately.
    // This prevents "connection refused" errors while the store initializes.
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Corvia server bound to {addr} (initializing...)");

    println!("Checking inference engine...");
    ensure_inference_ready(&config).await?;
    println!("Opening store...");
    let (store, graph, temporal) = connect_full_store(&config).await?;
    println!("Store ready");
    let engine: Arc<dyn InferenceEngine> = connect_engine(&config);

    // Construct generation engine (optional — only when [merge] is configured)
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let gen_engine: Option<std::sync::Arc<dyn corvia_kernel::traits::GenerationEngine>> =
        config.merge.as_ref().map(|merge| -> std::sync::Arc<dyn corvia_kernel::traits::GenerationEngine> {
            match merge.provider {
                corvia_common::config::InferenceProvider::Corvia => {
                    std::sync::Arc::new(corvia_kernel::grpc_chat::GrpcChatEngine::new(&config.embedding.url, &merge.model))
                }
                corvia_common::config::InferenceProvider::Ollama => {
                    std::sync::Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url, &merge.model))
                }
                corvia_common::config::InferenceProvider::Vllm => {
                    // vLLM chat not yet implemented — fall back to Ollama
                    std::sync::Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url, &merge.model))
                }
            }
        });
    let merge_config = config.merge.clone().unwrap_or_default();
    let coordinator_gen_engine: std::sync::Arc<dyn corvia_kernel::traits::GenerationEngine> =
        gen_engine.clone().unwrap_or_else(|| {
            // Fallback no-op: coordinator needs a GenerationEngine but it won't be used for ask()
            std::sync::Arc::new(corvia_kernel::grpc_chat::GrpcChatEngine::new(&config.embedding.url, "disabled"))
        });
    let coordinator = Arc::new(AgentCoordinator::new(
        store.clone(),
        engine.clone(),
        data_dir,
        config.agent_lifecycle.clone(),
        merge_config,
        coordinator_gen_engine,
    )?);
    println!("Agent coordination: enabled");

    // Construct RAG pipeline — auto-selects retriever based on graph availability.
    let rag = Arc::new(corvia_kernel::create_rag_pipeline(
        store.clone(),
        engine.clone(),
        Some(graph.clone()),
        gen_engine,
        &config,
    ).await);
    println!("RAG pipeline: enabled (retriever: {})", rag.retriever_name());
    if config.merge.is_none() {
        println!("  ask() mode: disabled (no [merge] configured)");
    }

    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let default_scope_id = Some(config.project.scope_id.clone());
    let config_path = CorviaConfig::config_path();
    let cluster_store = Arc::new(corvia_server::dashboard::clustering::ClusterStore::new());
    let (hook_sessions, _hook_rx) = corvia_server::dashboard::session_watcher::SessionWatcherState::new();
    let state = Arc::new(corvia_server::rest::AppState {
        store, engine, coordinator, graph, temporal, data_dir,
        rag: Some(rag), ready: ready.clone(), default_scope_id,
        config: Arc::new(std::sync::RwLock::new(config.clone())),
        config_path,
        cluster_store: cluster_store.clone(),
        gc_history: Arc::new(corvia_kernel::ops::GcHistory::new(50)),
        session_ingest_lock: tokio::sync::Mutex::new(()),
        hook_sessions: hook_sessions.clone(),
    });
    // Background cluster recompute every 60s
    {
        let cluster_store_bg = state.cluster_store.clone();
        let data_dir_bg = state.data_dir.clone();
        let scope_id_bg = state.default_scope_id.clone().unwrap_or_else(|| "corvia".into());
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                // Read files on blocking thread to avoid starving the async runtime.
                let dir = data_dir_bg.clone();
                let scope = scope_id_bg.clone();
                let entries = tokio::task::spawn_blocking(move || {
                    corvia_kernel::knowledge_files::read_scope(&dir, &scope)
                }).await;
                match entries {
                    Ok(Ok(entries)) => {
                        let pairs: Vec<(String, Vec<f32>)> = entries
                            .iter()
                            .filter_map(|e| {
                                e.embedding
                                    .as_ref()
                                    .map(|emb| (e.id.to_string(), emb.clone()))
                            })
                            .collect();
                        if cluster_store_bg.maybe_recompute(&pairs) {
                            tracing::info!(
                                "Cluster hierarchy recomputed ({} entries)",
                                pairs.len()
                            );
                        }
                    }
                    Ok(Err(e)) => tracing::warn!("Cluster recompute failed: {e}"),
                    Err(e) => tracing::warn!(error = %e, "cluster_recompute_task_failed"),
                }
            }
        });
    }

    let mut app = corvia_server::rest::router(state.clone());
    app = app.merge(corvia_server::mcp::mcp_router(state.clone()));
    app = app.merge(corvia_server::dashboard::router(state));
    app = app.layer(tower_http::cors::CorsLayer::permissive());
    println!("MCP endpoint: POST/GET/DELETE /mcp (Streamable HTTP)");
    println!("Dashboard API: GET /api/dashboard/{{status,traces,logs,config,graph}}");

    // Spawn JSONL session watcher for real-time dashboard visibility.
    corvia_server::dashboard::session_watcher::spawn_session_watcher(hook_sessions).await;
    println!("Session watcher: monitoring ~/.claude/sessions/");

    ready.store(true, std::sync::atomic::Ordering::Relaxed);
    println!("Corvia server listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    println!("Corvia server shut down gracefully");
    Ok(())
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to listen for Ctrl-C");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to listen for SIGTERM")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { println!("\nReceived SIGINT, shutting down..."); }
        _ = terminate => { println!("\nReceived SIGTERM, shutting down..."); }
    }
}

async fn cmd_ingest(path: Option<&str>, incremental: bool, files: &[String]) -> Result<()> {
    // Incremental mode: re-index specific files through the chunking pipeline
    if incremental && !files.is_empty() {
        let config = load_config()?;
        ensure_inference_ready(&config).await?;
        let (store, graph, _temporal) = connect_full_store(&config).await?;
        let engine = connect_engine(&config);
        let scope_id = &config.project.scope_id;

        // Register adapter chunking strategies (tree-sitter AST for code files)
        let mut pipeline = corvia_kernel::create_chunking_pipeline(&config);
        let adapter = GitAdapter::new();
        adapter.register_chunking(pipeline.registry_mut());

        // Resolve git HEAD for source_version
        let source_version = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "incremental".into());

        // Load existing entries for supersession lookup
        let data_dir = std::path::Path::new(&config.storage.data_dir);
        let existing_entries = corvia_kernel::knowledge_files::read_scope(data_dir, scope_id)
            .unwrap_or_default();

        println!("Incremental re-indexing {} file(s)...", files.len());
        for file_path in files {
            let path = std::path::Path::new(file_path);
            if !path.exists() {
                println!("  Skipped (not found): {}", file_path);
                continue;
            }

            // Skip binary files gracefully
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(e) => {
                    println!("  Skipped (read error): {} — {}", file_path, e);
                    continue;
                }
            };

            println!("  Re-indexing: {}", file_path);

            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let meta = corvia_kernel::chunking_strategy::SourceMetadata {
                file_path: file_path.to_string(),
                extension: ext.to_string(),
                language: None,
                scope_id: scope_id.clone(),
                source_version: source_version.clone(),
                workstream: None,
                content_role: None,
                source_origin: None,
            };
            let (chunks, pipeline_relations) = pipeline.process(&content, &meta)?;
            println!("    Chunked into {} pieces", chunks.len());

            // Build entries with full metadata (matching full ingest quality)
            let entries: Vec<corvia_common::types::KnowledgeEntry> = chunks
                .iter()
                .map(|chunk| {
                    let mut entry = corvia_common::types::KnowledgeEntry::new(
                        chunk.content.clone(),
                        scope_id.clone(),
                        source_version.clone(),
                    );
                    entry.metadata = corvia_common::types::EntryMetadata {
                        source_file: Some(file_path.to_string()),
                        language: chunk.metadata.language.clone(),
                        chunk_type: Some(chunk.chunk_type.clone()),
                        start_line: Some(chunk.start_line),
                        end_line: Some(chunk.end_line),
                        content_role: workspace::infer_content_role(file_path),
                        source_origin: workspace::infer_source_origin(None, file_path),
                    };
                    entry
                })
                .collect();

            // Batch embed
            let texts: Vec<String> = entries.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            // Insert new entries, collecting IDs for graph wiring
            let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(entries.len());
            for (entry, embedding) in entries.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored_ids.push(entry.id);
            }

            // Supersede old entries for this file
            let old_entries: Vec<&corvia_common::types::KnowledgeEntry> = existing_entries
                .iter()
                .filter(|e| {
                    e.metadata.source_file.as_deref() == Some(file_path.as_str())
                        && e.is_current()
                })
                .collect();
            if !old_entries.is_empty()
                && let Some(lite) = store.as_any().downcast_ref::<corvia_kernel::lite_store::LiteStore>() {
                    let first_new_id = stored_ids[0];
                    for old in &old_entries {
                        let _ = lite.supersede(&old.id, &first_new_id).await;
                    }
                    println!("    Superseded {} old entries", old_entries.len());
                }

            // Wire graph edges from pipeline relations
            if !pipeline_relations.is_empty() {
                let edges = wire_pipeline_relations(
                    &pipeline_relations, &chunks, &stored_ids, &*graph,
                ).await;
                if edges > 0 {
                    println!("    {edges} graph relations stored");
                }
            }

            println!("    Created {} entries", stored_ids.len());
        }
        println!("Incremental re-index complete.");
        return Ok(());
    }

    if let Some(path) = path {
        // D69 pipeline flow: source files → ChunkingPipeline → embed → store
        let config = load_config()?;
        let (store, graph) = connect_store_with_graph(&config).await?;
        ensure_inference_ready(&config).await?;
        let engine = connect_engine(&config);

        let adapter = GitAdapter::new();
        println!("Ingesting {}...", path);

        // Step 1: Collect source files via D69 adapter interface
        let source_files = adapter.ingest_sources(path).await?;
        let total_files = source_files.len();

        // Step 2: Build chunking pipeline with adapter strategies
        let mut pipeline = corvia_kernel::create_chunking_pipeline(&config);
        adapter.register_chunking(pipeline.registry_mut());

        // Step 3: Process through chunking pipeline (merge, split, overlap)
        let (processed, pipeline_relations, report) = pipeline.process_batch(&source_files)?;
        println!(
            "Chunked {} files → {} chunks ({} merged, {} split)",
            report.files_processed, report.total_chunks,
            report.chunks_merged, report.chunks_split
        );

        // Step 4: Convert ProcessedChunks to KnowledgeEntries, embed, and store
        let src_meta_lookup: std::collections::HashMap<&str, &corvia_kernel::chunking_strategy::SourceMetadata> =
            source_files.iter().map(|sf| (sf.metadata.file_path.as_str(), &sf.metadata)).collect();

        let entries: Vec<corvia_common::types::KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let src_meta = src_meta_lookup.get(pc.metadata.source_file.as_str());
                let mut entry = corvia_common::types::KnowledgeEntry::new(
                    pc.content.clone(),
                    config.project.scope_id.clone(),
                    pc.metadata.source_file.clone(),
                );
                if let Some(ws) = src_meta.and_then(|m| m.workstream.clone()) {
                    entry.workstream = ws;
                }
                entry.metadata = corvia_common::types::EntryMetadata {
                    source_file: Some(pc.metadata.source_file.clone()),
                    language: pc.metadata.language.clone(),
                    chunk_type: Some(pc.chunk_type.clone()),
                    start_line: Some(pc.start_line),
                    end_line: Some(pc.end_line),
                    content_role: src_meta
                        .and_then(|m| m.content_role.clone())
                        .or_else(|| workspace::infer_content_role(&pc.metadata.source_file)),
                    source_origin: src_meta
                        .and_then(|m| m.source_origin.clone())
                        .or_else(|| workspace::infer_source_origin(None, &pc.metadata.source_file)),
                };
                entry
            })
            .collect();

        let total = entries.len();
        let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
        let mut stored = 0;
        for batch in entries.chunks(corvia_kernel::introspect::EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored_ids.push(entry.id);
                stored += 1;
            }
            println!("  {stored}/{total} chunks stored");
        }

        // Step 5: Wire relations from pipeline (now native to ChunkingStrategy)
        if !pipeline_relations.is_empty() {
            let relations_stored = wire_pipeline_relations(
                &pipeline_relations, &processed, &stored_ids, &*graph,
            ).await;
            if relations_stored > 0 {
                println!("  {relations_stored} graph relations stored");
            }
        }

        // Step 5b: Wire doc-to-code relations
        let doc_edges = wire_doc_to_code_relations(&processed, &stored_ids);
        let mut doc_edges_stored = 0;
        for (from, rel, to) in &doc_edges {
            if graph.relate(from, rel, to, None).await.is_ok() {
                doc_edges_stored += 1;
            }
        }
        if doc_edges_stored > 0 {
            println!("  {doc_edges_stored} doc-to-code graph edges stored");
        }

        println!("Done. {stored} chunks from {total_files} files ingested from {path}.");
        println!("Next: corvia search \"your query\"");
    } else {
        // Workspace mode: ingest all workspace repos
        let root = std::env::current_dir()?;
        let config = load_config()?;
        if config.is_workspace() {
            ensure_inference_ready(&config).await?;
            workspace::ingest_workspace(&root, None, false).await?;
        } else {
            anyhow::bail!("No path provided and not in a workspace. Usage: corvia ingest <path>");
        }
    }
    Ok(())
}

async fn cmd_search(query: &str, limit: usize) -> Result<()> {
    let config = load_config()?;
    let is_ws = config.is_workspace();

    // Try routing through running server
    if let Some(client) = server_client::ServerClient::detect(&config).await {
        println!("(via server at {})\n", client.url());
        let response = client.search(query, &config.project.scope_id, limit).await?;

        if response.results.is_empty() {
            println!("No results found for: {query}");
            return Ok(());
        }

        println!("Found {} results for: {query}\n", response.count);
        print_server_search_results(&response.results, is_ws);
        return Ok(());
    }

    // Fallback: direct store access
    let store = connect_store(&config).await?;
    let engine = connect_engine(&config);

    let embedding = engine.embed(query).await?;
    let results = store
        .search(&embedding, &config.project.scope_id, limit)
        .await?;

    if results.is_empty() {
        println!("No results found for: {query}");
        return Ok(());
    }

    println!("Found {} results for: {query}\n", results.len());
    print_search_results(&results, is_ws);
    Ok(())
}

async fn cmd_status(metrics: bool) -> Result<()> {
    let config = load_config()?;

    if config.is_workspace() {
        let root = std::env::current_dir()?;
        return workspace::workspace_status(&root).await;
    }

    match config.storage.store_type {
        corvia_common::config::StoreType::Lite => {
            println!("Store: LiteStore ({})", config.storage.data_dir);
            println!("Inference: {} ({})",
                match config.embedding.provider {
                    corvia_common::config::InferenceProvider::Ollama => "Ollama",
                    corvia_common::config::InferenceProvider::Vllm => "vLLM",
                    corvia_common::config::InferenceProvider::Corvia => "Corvia",
                },
                config.embedding.url);

            let store = connect_store(&config).await?;
            let count = store.count(&config.project.scope_id).await?;
            println!("Entries in scope '{}': {count}", config.project.scope_id);
        }
        corvia_common::config::StoreType::Postgres => {
            println!("Store: PostgresStore ({})",
                config.storage.postgres_url.as_deref().unwrap_or("postgres://127.0.0.1:5432/corvia"));
            match connect_store(&config).await {
                Ok(store) => {
                    let count = store.count(&config.project.scope_id).await?;
                    println!("Entries in scope '{}': {count}", config.project.scope_id);
                }
                Err(e) => println!("  PostgreSQL not reachable: {e}"),
            }
        }
    }

    // Show agent coordination status if available (open registry once, reuse for --metrics)
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let coord_db_path = data_dir.join("coordination.redb");
    let registry = if coord_db_path.exists() {
        corvia_kernel::agent_registry::AgentRegistry::open(data_dir).ok()
    } else {
        None
    };

    if let Some(ref reg) = registry {
        let active = reg.list_active().unwrap_or_default();
        let sessions = corvia_kernel::session_manager::SessionManager::from_db(reg.db().clone())
            .map(|sm| sm.list_open().unwrap_or_default().len())
            .unwrap_or(0);
        let queue_depth = corvia_kernel::merge_queue::MergeQueue::from_db(reg.db().clone())
            .map(|mq| mq.depth().unwrap_or(0))
            .unwrap_or(0);
        println!("\nAgent coordination:");
        println!("  Active agents: {}", active.len());
        println!("  Open sessions: {sessions}");
        println!("  Merge queue depth: {queue_depth}");
    }

    // Extended metrics output (only when --metrics is passed)
    if metrics {
        println!("\nExtended metrics:");

        // Store type
        let store_type = match config.storage.store_type {
            corvia_common::config::StoreType::Lite => "lite",
            corvia_common::config::StoreType::Postgres => "postgres",
        };
        println!("  Store type: {store_type}");

        // Inference provider and URL
        let provider = match config.embedding.provider {
            corvia_common::config::InferenceProvider::Ollama => "ollama",
            corvia_common::config::InferenceProvider::Vllm => "vllm",
            corvia_common::config::InferenceProvider::Corvia => "corvia",
        };
        println!("  Inference provider: {provider}");
        println!("  Inference URL: {}", config.embedding.url);

        // Telemetry exporter
        println!("  Telemetry exporter: {}", config.telemetry.exporter);

        // Agent counts (reuse already-opened registry)
        match &registry {
            Some(reg) => {
                let all_agents = reg.list_all().unwrap_or_default();
                let active_agents = reg.list_active().unwrap_or_default();
                println!("  Registered agents: {}", all_agents.len());
                println!("  Active agents: {}", active_agents.len());
            }
            None => {
                println!("  Registered agents: 0");
                println!("  Active agents: 0");
            }
        }

        // Discovered adapters
        let extra_dirs = config.adapters.as_ref()
            .map(|a| a.search_dirs.clone())
            .unwrap_or_default();
        let adapters = corvia_kernel::ops::adapters_list(&extra_dirs);
        println!("  Discovered adapters: {}", adapters.len());
        for adapter in &adapters {
            println!("    - {} ({})", adapter.metadata.name, adapter.binary_path.display());
        }
    }

    Ok(())
}

async fn cmd_rebuild() -> Result<()> {
    let config = load_config()?;

    match config.storage.store_type {
        corvia_common::config::StoreType::Lite => {
            println!("Rebuilding LiteStore indexes from knowledge files...");
            let data_dir = std::path::Path::new(&config.storage.data_dir);
            let store = corvia_kernel::lite_store::LiteStore::open(data_dir, config.embedding.dimensions)?;
            let count = corvia_kernel::ops::rebuild_index(&store)?;
            println!("Rebuilt {count} entries.");
        }
        corvia_common::config::StoreType::Postgres => {
            println!("Rebuild is only needed for LiteStore. PostgreSQL manages its own indexes.");
        }
    }

    Ok(())
}

async fn cmd_test(check_only: bool, keep: bool, ci: bool) -> Result<()> {
    let config = CorviaConfig::default().with_env_overrides();

    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );

    // Phase 1: Check environment
    println!("  Checking environment...");
    println!("    LiteStore: no Docker required");
    let provisioner = OllamaProvisioner::new(&config.embedding.url);
    if provisioner.is_running().await {
        println!("    Ollama: running");
    } else {
        println!("    Ollama: not running");
        println!("  Auto-provisioning Ollama...");
        if let Err(e) = provisioner.ensure_ready(&config.embedding.model).await {
            eprintln!("  Failed to provision Ollama: {e}");
            std::process::exit(2);
        }
        println!("    Ollama: provisioned");
    }

    if check_only {
        println!("  Environment check complete.");
        return Ok(());
    }

    // Phase 2: Create store for test
    let test_config = {
        let mut tc = config.clone();
        let test_dir = std::env::temp_dir().join("corvia-introspect-test");
        tc.storage.data_dir = test_dir.to_string_lossy().to_string();
        tc
    };

    let store = match corvia_kernel::create_store(&test_config).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Infrastructure error: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = store.init_schema().await {
        eprintln!("  Infrastructure error: {e}");
        std::process::exit(2);
    }

    let engine = connect_engine(&config);
    let adapter = GitAdapter::new();

    println!("\n  Introspect: ingesting own source...");
    let chunks = introspect.ingest_self(".", &adapter, engine.as_ref(), store.as_ref()).await?;
    println!("    {chunks} chunks embedded and stored");

    // Phase 3: Self-query
    println!("\n  Introspect: running canonical queries...");
    let results = introspect.query_self(engine.as_ref(), store.as_ref()).await?;

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
        report.pass_count(), report.results.len(), report.avg_score()
    );

    // Teardown
    if !keep {
        println!("  Cleaning up introspect data...");
        if let Err(e) = store.delete_scope(introspect.scope_id()).await {
            eprintln!("  Warning: teardown failed: {e}");
        } else {
            println!("  Teardown complete.");
        }
    } else {
        println!("  --keep: test data preserved.");
    }

    if ci && report.exit_code() != 0 {
        std::process::exit(report.exit_code());
    } else if !report.all_passed() {
        println!("\n  Some queries failed. Use --ci to fail with exit code 1.");
    }

    Ok(())
}

async fn cmd_demo(keep: bool) -> Result<()> {
    let config = CorviaConfig::default().with_env_overrides();

    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );

    // Phase 1: Check + auto-provision
    println!("  Checking environment...");
    println!("    LiteStore: no Docker required");
    let provisioner = OllamaProvisioner::new(&config.embedding.url);
    if provisioner.is_running().await {
        println!("    Ollama: running");
    } else {
        print!("    Ollama: provisioning...");
        provisioner.ensure_ready(&config.embedding.model).await?;
        println!(" done");
    }

    // Phase 2: Create store
    let demo_config = {
        let mut tc = config.clone();
        let demo_dir = std::env::temp_dir().join("corvia-demo");
        tc.storage.data_dir = demo_dir.to_string_lossy().to_string();
        tc
    };

    let store = corvia_kernel::create_store(&demo_config).await?;
    store.init_schema().await?;

    let engine = connect_engine(&config);
    let adapter = GitAdapter::new();

    println!("\n  Ingesting Corvia's own source code...");
    let chunks = introspect.ingest_self(".", &adapter, engine.as_ref(), store.as_ref()).await?;
    println!("  {chunks} chunks stored.\n");

    println!("  Corvia is ready. Search its own codebase.");
    println!("  Type a query and press Enter. Type 'exit' or Ctrl+C to quit.\n");

    // REPL loop
    let scope_id = introspect.scope_id().to_string();
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("  corvia> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        line.clear();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
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
            print_search_results(&results, false);
        }
    }

    if !keep {
        println!("\n  Cleaning up demo data...");
        if let Err(e) = store.delete_scope(introspect.scope_id()).await {
            eprintln!("  Warning: teardown failed: {e}");
        } else {
            println!("  Teardown complete.");
        }
    } else {
        println!("\n  --keep: demo data preserved. Search with: corvia search \"your query\"");
    }

    Ok(())
}

async fn cmd_agent(command: AgentCommands) -> Result<()> {
    let config = load_config()?;
    let data_dir = std::path::Path::new(&config.storage.data_dir);

    let registry = corvia_kernel::agent_registry::AgentRegistry::open(data_dir)?;
    let db = registry.db().clone();

    match command {
        AgentCommands::List => {
            let agents = registry.list_all()?;
            if agents.is_empty() {
                println!("No registered agents.");
            } else {
                println!("{:<30} {:<12} DISPLAY NAME", "AGENT ID", "STATUS");
                for a in &agents {
                    println!("{:<30} {:<12} {}", a.agent_id, format!("{:?}", a.status), a.display_name);
                }
            }
        }
        AgentCommands::Sessions { agent_id } => {
            let session_mgr = corvia_kernel::session_manager::SessionManager::from_db(db)?;
            let sessions = session_mgr.list_by_agent(&agent_id)?;
            if sessions.is_empty() {
                println!("No sessions for agent '{agent_id}'.");
            } else {
                println!("{:<40} {:<12} {:>8} {:>8}", "SESSION ID", "STATE", "WRITTEN", "MERGED");
                for s in &sessions {
                    println!(
                        "{:<40} {:<12} {:>8} {:>8}",
                        s.session_id,
                        format!("{:?}", s.state),
                        s.entries_written,
                        s.entries_merged,
                    );
                }
            }
        }
        AgentCommands::Connect => {
            use corvia_common::agent_types::SessionState;
            use std::io::Write;

            let session_mgr = corvia_kernel::session_manager::SessionManager::from_db(db)?;
            let all_agents = registry.list_active()?;

            // Find agents with stale/orphaned sessions
            let mut reconnectable = Vec::new();
            for agent in &all_agents {
                let sessions = session_mgr.list_by_agent(&agent.agent_id)?;
                let has_stale_or_orphaned = sessions.iter().any(|s|
                    matches!(s.state, SessionState::Stale | SessionState::Orphaned)
                );
                if has_stale_or_orphaned {
                    reconnectable.push(agent);
                }
            }

            if reconnectable.is_empty() {
                println!("No reconnectable agents found.");
            } else {
                println!("Reconnectable agents:");
                for (i, agent) in reconnectable.iter().enumerate() {
                    println!("  [{}] {} ({})", i + 1, agent.display_name, agent.agent_id);
                    if let Some(ref desc) = agent.description {
                        println!("      Purpose: {desc}");
                    }
                    if let Some(ref summary) = agent.activity_summary {
                        let tags = summary.topic_tags.join(", ");
                        println!("      Activity: {} entries across [{}]", summary.entry_count, tags);
                        if summary.drifted && !summary.last_topics.is_empty() {
                            let last = summary.last_topics.join(", ");
                            println!("      Last session drifted to: [{last}]");
                        }
                        let ago = humanize_duration(chrono::Utc::now() - summary.last_active);
                        println!("      Last active: {ago}");
                    }
                }
            }

            println!("  [N] Register new agent");
            print!("Pick one: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("n") {
                print!("Agent ID (e.g., myproject::refactor): ");
                std::io::stdout().flush()?;
                let mut name = String::new();
                std::io::stdin().read_line(&mut name)?;
                let name = name.trim();

                print!("Display name: ");
                std::io::stdout().flush()?;
                let mut display = String::new();
                std::io::stdin().read_line(&mut display)?;
                let display = display.trim();

                print!("Purpose (optional): ");
                std::io::stdout().flush()?;
                let mut desc = String::new();
                std::io::stdin().read_line(&mut desc)?;
                let desc = desc.trim();

                let scope_id = config.project.scope_id.clone();
                let record = registry.register(
                    name,
                    display,
                    corvia_common::agent_types::IdentityType::Registered,
                    corvia_common::agent_types::AgentPermission::ReadWrite {
                        scopes: vec![scope_id],
                    },
                )?;
                if !desc.is_empty() {
                    registry.set_description(&record.agent_id, desc)?;
                }
                println!("Connected as {}.", record.agent_id);
                println!("  export CORVIA_AGENT_ID=\"{}\"", record.agent_id);
            } else {
                let idx: usize = input.parse()
                    .map_err(|_| anyhow::anyhow!("Invalid selection: '{input}'"))?;
                let agent = reconnectable.get(idx - 1)
                    .ok_or_else(|| anyhow::anyhow!("Invalid index: {idx}"))?;
                // Touch the agent to update last_seen
                let _ = registry.touch(&agent.agent_id);
                println!("Connected as {}.", agent.agent_id);
                println!("  export CORVIA_AGENT_ID=\"{}\"", agent.agent_id);
            }
        }
    }
    Ok(())
}

/// Format a chrono Duration as a human-readable string (e.g., "3m ago", "2h ago").
fn humanize_duration(dur: chrono::Duration) -> String {
    let secs = dur.num_seconds();
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

async fn cmd_workspace(command: WorkspaceCommands) -> Result<()> {
    match command {
        WorkspaceCommands::Create { name, repos, template: _template, locals } => {
            if repos.is_empty() {
                anyhow::bail!("Provide at least one --repo URL");
            }
            let local_overrides = workspace::parse_local_overrides(&locals)?;
            let config = workspace::generate_workspace_config(&name, &repos, &local_overrides)?;
            let root = std::path::PathBuf::from(&name);
            workspace::scaffold_workspace(&root, &config)?;
            workspace::init_workspace(&root).await?;

            println!("\nWorkspace '{}' created at ./{}", name, name);
            let ws = config.workspace.as_ref().unwrap();
            for repo in &ws.repos {
                println!("  repo: {} (namespace: {})", repo.name, repo.namespace);
            }
            println!("\nNext: cd {} && corvia workspace ingest", name);
            Ok(())
        }
        WorkspaceCommands::Init => {
            let root = std::env::current_dir()?;
            workspace::init_workspace(&root).await?;
            Ok(())
        }
        WorkspaceCommands::Status => {
            let root = std::env::current_dir()?;
            workspace::workspace_status(&root).await?;
            Ok(())
        }
        WorkspaceCommands::Add { url, name, namespace, local } => {
            let root = std::env::current_dir()?;
            let config_path = root.join("corvia.toml");
            let mut config = CorviaConfig::load(&config_path)?;
            workspace::add_repo_to_config(
                &mut config,
                &url,
                name.as_deref(),
                namespace.as_deref(),
                local.as_deref(),
            )?;
            config.save(&config_path)?;

            let ws = config.workspace.as_ref().unwrap();
            let repo = ws.repos.last().unwrap();
            println!("Added repo '{}' (namespace: {})", repo.name, repo.namespace);

            // Clone if needed
            let repo_path = workspace::resolve_repo_path(&root, &ws.repos_dir, repo);
            if !repo_path.exists() {
                workspace::clone_repo(&repo.url, &repo_path)?;
            }
            println!("Next: corvia workspace ingest --repo {}", repo.name);
            Ok(())
        }
        WorkspaceCommands::Remove { name, purge } => {
            let root = std::env::current_dir()?;
            let config_path = root.join("corvia.toml");
            let mut config = CorviaConfig::load(&config_path)?;

            // Get repo info before removing (capture repos_dir before mutable borrow)
            let ws = config
                .workspace
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Not a workspace"))?;
            let repos_dir = ws.repos_dir.clone();
            let repo_path = ws
                .repos
                .iter()
                .find(|r| r.name == name)
                .map(|r| workspace::resolve_repo_path(&root, &ws.repos_dir, r));

            workspace::remove_repo_from_config(&mut config, &name)?;
            config.save(&config_path)?;
            println!("Removed repo '{}' from config", name);

            if purge
                && let Some(path) = repo_path
                    && path.exists() && path.starts_with(root.join(&repos_dir)) {
                        std::fs::remove_dir_all(&path)?;
                        println!("Deleted cloned repo at {}", path.display());
                    }
            Ok(())
        }
        WorkspaceCommands::Ingest { repo, fresh } => {
            let root = std::env::current_dir()?;
            let config = load_config()?;
            ensure_inference_ready(&config).await?;
            workspace::ingest_workspace(&root, repo.as_deref(), fresh).await?;
            Ok(())
        }
        WorkspaceCommands::Clean => {
            let root = std::env::current_dir()?;
            println!("Cleaning build artifacts...");
            let report = workspace::clean_build_artifacts(&root)?;
            if report.dirs_cleaned == 0 {
                println!("No build artifacts found.");
            } else {
                println!(
                    "\nCleaned {} target dir(s), freed {}.",
                    report.dirs_cleaned,
                    workspace::human_bytes(report.bytes_freed)
                );
            }
            Ok(())
        }
        WorkspaceCommands::Docs { command } => {
            match command {
                DocsCommands::Check { check, fix, yes } =>
                    cmd_docs_check(check.as_deref(), fix, yes).await?,
            }
            Ok(())
        }
        WorkspaceCommands::InitHooks => {
            eprintln!("Note: 'workspace init-hooks' is deprecated. Use 'corvia hooks init' instead.");
            let root = std::env::current_dir()?;
            let config = load_config()?;
            hooks::settings::init_hooks(&root, &config)?;
            Ok(())
        }
    }
}

async fn cmd_docs_check(
    check: Option<&str>,
    fix: bool,
    yes: bool,
) -> Result<()> {
    let config = load_config()?;
    let ws = config.workspace.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace — 'docs check' requires [workspace] config"))?;
    let docs_rules = ws.docs.as_ref()
        .and_then(|d| d.rules.clone())
        .unwrap_or_default();
    let scope_id = &config.project.scope_id;

    // Determine which checks to run
    let checks: Vec<corvia_kernel::reasoner::CheckType> = if let Some(c) = check {
        vec![c.parse().map_err(|e: String| anyhow::anyhow!(e))?]
    } else {
        vec![
            corvia_kernel::reasoner::CheckType::MisplacedDoc,
            corvia_kernel::reasoner::CheckType::CoverageGap,
            corvia_kernel::reasoner::CheckType::TemporalContradiction,
        ]
    };

    println!("Running documentation health checks...");

    // Try routing through running server first
    if !fix {
        if let Some(client) = server_client::ServerClient::detect(&config).await {
            println!("(via server at {})\n", client.url());
            let mut total = 0;
            for check_type in &checks {
                let response = client.reason(scope_id, Some(check_type.as_str())).await?;
                for (i, finding) in response.findings.iter().enumerate() {
                    println!("  [{}] {} (confidence: {:.0}%)", total + i + 1, finding.check_type, finding.confidence * 100.0);
                    println!("      {}", finding.rationale);
                }
                total += response.findings.len();
            }
            if total == 0 {
                println!("No issues found.");
            }
            return Ok(());
        }
    } else {
        // --fix requires local execution (needs filesystem access for moves)
        println!("(--fix mode: running locally)\n");
    }

    // Run locally
    let (store, graph, _temporal) = connect_full_store(&config).await?;
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let entries = corvia_kernel::knowledge_files::read_scope(data_dir, scope_id)?;

    if entries.is_empty() {
        println!("No entries found in scope '{}'. Run 'corvia ingest' first.", scope_id);
        return Ok(());
    }
    println!("Checking {} entries in scope '{}'...\n", entries.len(), scope_id);

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*store, &*graph)
        .with_docs_rules(docs_rules);

    let mut all_findings = Vec::new();
    for check_type in &checks {
        let findings = reasoner.run_check(&entries, scope_id, *check_type).await?;
        all_findings.extend(findings);
    }

    if all_findings.is_empty() {
        println!("No issues found.");
    } else {
        for (i, finding) in all_findings.iter().enumerate() {
            println!("  {}. [{}] {} (confidence: {:.0}%)",
                i + 1, finding.check_type.as_str(), finding.rationale, finding.confidence * 100.0);
        }
    }

    // Auto-fix mode for MisplacedDoc
    if fix {
        let misplaced: Vec<_> = all_findings.iter()
            .filter(|f| f.check_type == corvia_kernel::reasoner::CheckType::MisplacedDoc)
            .collect();
        if misplaced.is_empty() {
            println!("No misplaced docs to fix.");
            return Ok(());
        }
        let status = std::process::Command::new("git").args(["status", "--porcelain"]).output()?;
        if !status.stdout.is_empty() {
            return Err(anyhow::anyhow!("Working tree has uncommitted changes. Commit or stash first."));
        }
        if !yes {
            println!("Would move {} file(s). Re-run with --yes to execute.", misplaced.len());
            return Ok(());
        }
        println!("Auto-fix is not yet implemented. Showing planned moves:\n");
        for finding in misplaced {
            println!("  {}", finding.rationale);
        }
        println!("\nManually move these files to the correct location.");
    }

    Ok(())
}

// ---- Temporal / Graph / Reasoning commands ----

async fn cmd_history(entry_id: &str) -> Result<()> {
    let config = load_config()?;

    // Try routing through running server
    if let Some(client) = server_client::ServerClient::detect(&config).await {
        println!("(via server at {})\n", client.url());
        let chain = client.history(entry_id).await?;

        if chain.is_empty() {
            println!("No history found for entry {entry_id}");
            return Ok(());
        }

        println!("History for entry {} ({} entries):", entry_id, chain.len());
        for (i, entry) in chain.iter().enumerate() {
            let status = if entry.is_current { " (current)" } else { "" };
            println!("\n  [{i}] {}{status}", entry.id);
            println!("      recorded: {}", entry.recorded_at.format("%Y-%m-%d %H:%M:%S"));
            println!("      valid: {} → {}",
                entry.valid_from.format("%Y-%m-%d %H:%M:%S"),
                entry.valid_to.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_else(|| "now".to_string()),
            );
            if let Some(first_line) = entry.content.lines().next() {
                let truncated = truncate_str(first_line, 80);
                if first_line.len() > 80 {
                    println!("      content: {truncated}...");
                } else {
                    println!("      content: {truncated}");
                }
            }
        }
        return Ok(());
    }

    // Fallback: direct store access
    let (_store, _graph, temporal) = connect_full_store(&config).await?;

    let uuid = uuid::Uuid::parse_str(entry_id)
        .map_err(|e| anyhow::anyhow!("Invalid UUID: {e}"))?;

    let chain = temporal.history(&uuid).await?;

    if chain.is_empty() {
        println!("No history found for entry {entry_id}");
        return Ok(());
    }

    println!("History for entry {} ({} entries):", entry_id, chain.len());
    for (i, entry) in chain.iter().enumerate() {
        let status = if entry.is_current() { " (current)" } else { "" };
        println!("\n  [{i}] {}{status}", entry.id);
        println!("      recorded: {}", entry.recorded_at.format("%Y-%m-%d %H:%M:%S"));
        println!("      valid: {} → {}",
            entry.valid_from.format("%Y-%m-%d %H:%M:%S"),
            entry.valid_to.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or_else(|| "now".to_string()),
        );
        if let Some(ref file) = entry.metadata.source_file {
            println!("      file: {file}");
        }
        // Show first line of content (char-boundary safe truncation)
        if let Some(first_line) = entry.content.lines().next() {
            let truncated = truncate_str(first_line, 80);
            if first_line.len() > 80 {
                println!("      content: {truncated}...");
            } else {
                println!("      content: {truncated}");
            }
        }
    }
    Ok(())
}

async fn cmd_evolution(scope: Option<&str>, since: &str) -> Result<()> {
    let config = load_config()?;
    let scope_id = scope.unwrap_or(&config.project.scope_id);

    // Try routing through running server
    if let Some(client) = server_client::ServerClient::detect(&config).await {
        println!("(via server at {})\n", client.url());
        let entries = client.evolution(scope_id, since).await?;

        if entries.is_empty() {
            println!("No changes in scope '{}' in the last {since}", scope_id);
            return Ok(());
        }

        println!("Changes in scope '{}' (last {since}): {} entries\n", scope_id, entries.len());
        for entry in &entries {
            let status = if entry.is_current { "active" } else { "superseded" };
            println!("  {} [{}]", entry.id, status);
            println!("    valid: {} → {}",
                entry.valid_from.format("%Y-%m-%d %H:%M"),
                entry.valid_to.map(|t| t.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_else(|| "now".to_string()),
            );
        }
        return Ok(());
    }

    // Fallback: direct store access
    let (_store, _graph, temporal) = connect_full_store(&config).await?;

    // Parse duration: "7d", "1d", "30d", "1h"
    let duration = parse_duration(since)?;
    let now = chrono::Utc::now();
    let from = now - duration;

    let entries = temporal.evolution(scope_id, from, now).await?;

    if entries.is_empty() {
        println!("No changes in scope '{}' in the last {since}", scope_id);
        return Ok(());
    }

    println!("Changes in scope '{}' (last {since}): {} entries\n", scope_id, entries.len());
    for entry in &entries {
        let status = if entry.is_current() { "active" } else { "superseded" };
        let file = entry.metadata.source_file.as_deref().unwrap_or("(unknown)");
        println!("  {} [{}] {}", entry.id, status, file);
        println!("    valid: {} → {}",
            entry.valid_from.format("%Y-%m-%d %H:%M"),
            entry.valid_to.map(|t| t.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_else(|| "now".to_string()),
        );
    }
    Ok(())
}

async fn cmd_graph(entry_id: Option<&str>, scope: Option<&str>, relation: Option<&str>) -> Result<()> {
    let config = load_config()?;

    if let Some(id_str) = entry_id {
        // Try routing through running server for entry-specific queries
        if let Some(client) = server_client::ServerClient::detect(&config).await {
            println!("(via server at {})\n", client.url());
            let edges = client.edges(id_str, relation).await?;

            if edges.is_empty() {
                println!("No edges for entry {id_str}");
                return Ok(());
            }

            println!("Edges for entry {} ({} total):\n", id_str, edges.len());
            for edge in &edges {
                let direction = if edge.from == id_str { "\u{2192}" } else { "\u{2190}" };
                let other = if edge.from == id_str { &edge.to } else { &edge.from };
                println!("  {} {} {} ({})", id_str, direction, other, edge.relation);
            }
            return Ok(());
        }

        // Fallback: direct store access
        let (_store, graph, _temporal) = connect_full_store(&config).await?;
        let uuid = uuid::Uuid::parse_str(id_str)
            .map_err(|e| anyhow::anyhow!("Invalid UUID: {e}"))?;

        let edges = graph.edges(&uuid, corvia_common::types::EdgeDirection::Both).await?;

        if edges.is_empty() {
            println!("No edges for entry {id_str}");
            return Ok(());
        }

        let filtered: Vec<_> = if let Some(rel) = relation {
            edges.into_iter().filter(|e| e.relation == rel).collect()
        } else {
            edges
        };

        if filtered.is_empty() {
            println!("No edges matching filter for entry {id_str}");
            return Ok(());
        }

        println!("Edges for entry {} ({} total):\n", id_str, filtered.len());
        for edge in &filtered {
            let direction = if edge.from == uuid { "\u{2192}" } else { "\u{2190}" };
            let other = if edge.from == uuid { edge.to } else { edge.from };
            println!("  {} {} {} ({})", uuid, direction, other, edge.relation);
        }
    } else {
        // Scope-wide: list all entries in scope that have edges (always direct)
        let (_store, graph, _temporal) = connect_full_store(&config).await?;
        let scope_id = scope.unwrap_or(&config.project.scope_id);
        let data_dir = std::path::Path::new(&config.storage.data_dir);
        let entries = corvia_kernel::knowledge_files::read_scope(data_dir, scope_id)?;

        let mut edge_count = 0;
        for entry in &entries {
            let edges = graph.edges(&entry.id, corvia_common::types::EdgeDirection::Both).await?;
            let filtered: Vec<_> = if let Some(rel) = relation {
                edges.into_iter().filter(|e| e.relation == rel).collect()
            } else {
                edges
            };
            if !filtered.is_empty() {
                for edge in &filtered {
                    let direction = if edge.from == entry.id { "\u{2192}" } else { "\u{2190}" };
                    let other = if edge.from == entry.id { edge.to } else { edge.from };
                    let file = entry.metadata.source_file.as_deref().unwrap_or("?");
                    println!("  {} ({}) {} {} ({})", entry.id, file, direction, other, edge.relation);
                    edge_count += 1;
                }
            }
        }
        if edge_count == 0 {
            println!("No edges found in scope '{scope_id}'");
        } else {
            println!("\n{edge_count} edges in scope '{scope_id}'");
        }
    }
    Ok(())
}

async fn cmd_relate(from: &str, relation: &str, to: &str) -> Result<()> {
    let config = load_config()?;
    let (_store, graph, _temporal) = connect_full_store(&config).await?;

    let from_uuid = uuid::Uuid::parse_str(from)
        .map_err(|e| anyhow::anyhow!("Invalid 'from' UUID: {e}"))?;
    let to_uuid = uuid::Uuid::parse_str(to)
        .map_err(|e| anyhow::anyhow!("Invalid 'to' UUID: {e}"))?;

    graph.relate(&from_uuid, relation, &to_uuid, None).await?;
    println!("Created edge: {} --[{}]--> {}", from, relation, to);
    Ok(())
}

async fn cmd_reason(scope: Option<&str>, check: Option<&str>, llm: bool) -> Result<()> {
    let config = load_config()?;
    let scope_id = scope.unwrap_or(&config.project.scope_id);

    // Try routing through running server (deterministic checks only)
    if !llm
        && let Some(client) = server_client::ServerClient::detect(&config).await {
            println!("(via server at {})\n", client.url());
            let response = client.reason(scope_id, check).await?;

            if response.findings.is_empty() {
                println!("No issues found in scope '{scope_id}'.");
                return Ok(());
            }

            println!("Findings for scope '{}': {} issues\n", scope_id, response.findings.len());
            for (i, finding) in response.findings.iter().enumerate() {
                println!("  [{}] {} (confidence: {:.0}%)", i + 1, finding.check_type, finding.confidence * 100.0);
                println!("      {}", finding.rationale);
                if !finding.target_ids.is_empty() {
                    println!("      targets: {}", finding.target_ids.join(", "));
                }
            }

            println!();
            let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
            for f in &response.findings {
                *counts.entry(&f.check_type).or_insert(0) += 1;
            }
            for (check_type, count) in &counts {
                println!("  {check_type}: {count}");
            }
            return Ok(());
        }

    let (store, graph, _temporal) = connect_full_store(&config).await?;

    // Load entries for the target scope only
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let entries = corvia_kernel::knowledge_files::read_scope(data_dir, scope_id)?;

    // Wire DocsRulesConfig so MisplacedDoc check works via `corvia reason`
    let docs_rules = config.workspace.as_ref()
        .and_then(|ws| ws.docs.as_ref()?.rules.clone())
        .unwrap_or_default();
    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*store, &*graph)
        .with_docs_rules(docs_rules);

    let mut findings = if let Some(check_str) = check {
        // Run a single check type
        let check_type = parse_check_type(check_str)?;
        reasoner.run_check(&entries, scope_id, check_type).await?
    } else {
        // Run all algorithmic checks (includes docs workflow checks)
        reasoner.run_all(&entries, scope_id).await?
    };

    // Optionally run LLM checks
    if llm {
        if config.reasoning.is_some() {
            let engine = corvia_kernel::create_engine(&config);
            let llm_findings = reasoner.run_llm_checks(&entries, scope_id, &*engine).await?;
            findings.extend(llm_findings);
        } else {
            println!("Configure [reasoning] in corvia.toml to enable LLM-powered analysis.");
        }
    }

    if findings.is_empty() {
        println!("No issues found in scope '{scope_id}'.");
        return Ok(());
    }

    println!("Findings for scope '{}': {} issues\n", scope_id, findings.len());
    for (i, finding) in findings.iter().enumerate() {
        println!("  [{}] {} (confidence: {:.0}%)", i + 1, finding.check_type.as_str(), finding.confidence * 100.0);
        println!("      {}", finding.rationale);
        if !finding.target_ids.is_empty() {
            println!("      targets: {}", finding.target_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "));
        }
    }

    // Print summary by check type (BTreeMap for deterministic output)
    println!();
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in &findings {
        *counts.entry(f.check_type.as_str()).or_insert(0) += 1;
    }
    for (check_type, count) in &counts {
        println!("  {check_type}: {count}");
    }

    Ok(())
}

/// Parse a human-readable duration like "7d", "1d", "30d", "1h".
/// Rejects zero and negative values.
fn parse_duration(s: &str) -> Result<chrono::Duration> {
    let s = s.trim();
    if let Some(days) = s.strip_suffix('d') {
        let n: i64 = days.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {s}"))?;
        if n <= 0 { anyhow::bail!("Duration must be positive: {s}"); }
        Ok(chrono::Duration::days(n))
    } else if let Some(hours) = s.strip_suffix('h') {
        let n: i64 = hours.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {s}"))?;
        if n <= 0 { anyhow::bail!("Duration must be positive: {s}"); }
        Ok(chrono::Duration::hours(n))
    } else {
        anyhow::bail!("Invalid duration format: {s}. Use '7d' for days or '1h' for hours.")
    }
}

async fn cmd_inference_reload(
    device: Option<&str>,
    backend: Option<&str>,
    model: Option<&str>,
    kv_quant: Option<&str>,
    flash_attention: Option<bool>,
    no_persist: bool,
) -> Result<()> {
    let mut config = load_config()?;
    let grpc_url = match config.embedding.provider {
        corvia_common::config::InferenceProvider::Corvia => config.embedding.url.clone(),
        _ => anyhow::bail!("inference reload requires provider = \"corvia\" in corvia.toml"),
    };

    // Apply overrides to config
    if let Some(d) = device {
        config.inference.device = d.to_string();
    }
    if let Some(b) = backend {
        config.inference.backend = b.to_string();
    }
    if let Some(kv) = kv_quant {
        config.inference.kv_quant = kv.to_string();
    }
    if let Some(fa) = flash_attention {
        config.inference.flash_attention = fa;
    }

    // Persist to config file unless --no-persist
    if !no_persist {
        let config_path = corvia_common::config::CorviaConfig::config_path();
        config.save(&config_path)?;
        println!("Updated corvia.toml [inference] section");
    }

    // Trigger gRPC reload
    let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(&grpc_url);
    if !provisioner.is_running().await {
        anyhow::bail!("corvia-inference is not running at {grpc_url}");
    }
    provisioner.reload_models(
        &config.inference.device,
        &config.inference.backend,
        &config.inference.embedding_backend,
        &config.inference.kv_quant,
        config.inference.flash_attention,
        model,
    ).await?;
    println!("Reload complete.");
    Ok(())
}

async fn cmd_inference_status() -> Result<()> {
    let config = load_config()?;
    let grpc_url = match config.embedding.provider {
        corvia_common::config::InferenceProvider::Corvia => config.embedding.url.clone(),
        _ => anyhow::bail!("inference status requires provider = \"corvia\" in corvia.toml"),
    };
    let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(&grpc_url);
    if !provisioner.is_running().await {
        println!("corvia-inference: not running ({})", grpc_url);
        return Ok(());
    }
    let models = provisioner.list_models().await?;
    if models.is_empty() {
        println!("corvia-inference: running, no models loaded");
        return Ok(());
    }
    println!("corvia-inference: running ({} model(s) loaded)\n", models.len());
    println!("{:<30} {:<10} {:<8} {:<10} {:<8} {:<6}", "MODEL", "TYPE", "DEVICE", "BACKEND", "KV_QUANT", "FLASH");
    println!("{}", "-".repeat(72));
    for m in &models {
        println!(
            "{:<30} {:<10} {:<8} {:<10} {:<8} {:<6}",
            truncate_str(&m.name, 29),
            m.model_type,
            m.device,
            m.backend,
            if m.kv_quant.is_empty() { "f16" } else { &m.kv_quant },
            if m.flash_attention { "on" } else { "off" },
        );
    }
    Ok(())
}

fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

fn parse_check_type(s: &str) -> Result<corvia_kernel::reasoner::CheckType> {
    s.parse::<corvia_kernel::reasoner::CheckType>()
        .map_err(|e| anyhow::anyhow!(e))
}

fn load_config() -> Result<CorviaConfig> {
    let path = CorviaConfig::config_path();
    if !path.exists() {
        anyhow::bail!("No corvia.toml found. Run 'corvia init' first.");
    }
    Ok(CorviaConfig::load(&path)?)
}

fn connect_engine(config: &CorviaConfig) -> Arc<dyn InferenceEngine> {
    corvia_kernel::create_engine(config)
}

/// Resolve chat model coordinates from config registry.
/// Falls back to empty HF coords (server-side resolve) if the model name isn't in the registry.
fn resolve_chat_model_coords(config: &CorviaConfig) -> Option<corvia_kernel::inference_provisioner::ChatModelCoords> {
    let merge = config.merge.as_ref()?;
    let name = &merge.model;
    let coords = config.inference.chat_models.get(name);
    if coords.is_none() {
        eprintln!("  Warning: chat model '{name}' not in [inference.chat_models] registry, falling back to server-side resolve");
    }
    Some(corvia_kernel::inference_provisioner::ChatModelCoords {
        name: name.clone(),
        hf_repo: coords.map(|c| c.repo.clone()).unwrap_or_default(),
        hf_filename: coords.map(|c| c.filename.clone()).unwrap_or_default(),
    })
}

/// Ensure the inference backend is ready (model loaded).
/// For Ollama: checks server + model availability.
/// For Corvia: ensures gRPC server is running + model loaded.
/// For vLLM: no-op (Docker-managed).
async fn ensure_inference_ready(config: &CorviaConfig) -> Result<()> {
    match config.embedding.provider {
        corvia_common::config::InferenceProvider::Corvia => {
            let provisioner = corvia_kernel::inference_provisioner::InferenceProvisioner::new(
                &config.embedding.url,
            );
            let chat_coords = resolve_chat_model_coords(config);
            provisioner.ensure_ready(
                &config.embedding.model,
                chat_coords.as_ref(),
                &config.inference.device,
                &config.inference.backend,
                &config.inference.embedding_backend,
                &config.inference.kv_quant,
                config.inference.flash_attention,
            ).await?;
        }
        corvia_common::config::InferenceProvider::Ollama => {
            let provisioner = OllamaProvisioner::new(&config.embedding.url);
            provisioner.ensure_ready(&config.embedding.model).await?;
        }
        corvia_common::config::InferenceProvider::Vllm => {}
    }
    Ok(())
}

async fn connect_store(config: &CorviaConfig) -> Result<Box<dyn QueryableStore>> {
    Ok(corvia_kernel::create_store(config).await?)
}

async fn connect_store_with_graph(
    config: &CorviaConfig,
) -> Result<(Arc<dyn QueryableStore>, Arc<dyn GraphStore>)> {
    Ok(corvia_kernel::create_store_with_graph(config).await?)
}

async fn connect_full_store(
    config: &CorviaConfig,
) -> Result<(Arc<dyn QueryableStore>, Arc<dyn GraphStore>, Arc<dyn TemporalStore>)> {
    Ok(corvia_kernel::create_full_store(config).await?)
}

/// Scan markdown chunks for references to known code files and symbols,
/// creating "references" edges to connect documentation to the code it describes.
fn wire_doc_to_code_relations(
    processed: &[corvia_kernel::chunking_strategy::ProcessedChunk],
    stored_ids: &[uuid::Uuid],
) -> Vec<(uuid::Uuid, String, uuid::Uuid)> {
    use std::collections::{HashMap, HashSet};

    // Build code file index: basename (without extension) → Vec<(idx, &chunk)>
    let mut code_index: HashMap<&str, Vec<(usize, &corvia_kernel::chunking_strategy::ProcessedChunk)>> =
        HashMap::new();
    // Also build a full-name index for exact file references
    let mut file_index: HashMap<&str, Vec<(usize, &corvia_kernel::chunking_strategy::ProcessedChunk)>> =
        HashMap::new();

    for (idx, pc) in processed.iter().enumerate() {
        let src = pc.metadata.source_file.as_str();
        file_index.entry(src).or_default().push((idx, pc));

        // Only index code files (not markdown) for basename matching
        if !src.ends_with(".md") {
            // Extract basename without extension: "crates/.../rest.rs" → "rest"
            if let Some(filename) = src.rsplit('/').next()
                && let Some(stem) = filename.rsplit_once('.').map(|(s, _)| s) {
                    code_index.entry(stem).or_default().push((idx, pc));
                }
        }
    }

    // Regex patterns for file references and symbol references
    let file_re = regex::Regex::new(r"\b(\w[\w.-]*)\.(rs|py|ts|js|tsx|jsx)\b").unwrap();
    let symbol_re = regex::Regex::new(r"`([A-Z]\w+)`").unwrap();

    let mut edges: Vec<(uuid::Uuid, String, uuid::Uuid)> = Vec::new();

    for (idx, pc) in processed.iter().enumerate() {
        if idx >= stored_ids.len() {
            continue;
        }
        // Only process markdown chunks
        if !pc.metadata.source_file.ends_with(".md") {
            continue;
        }
        let from_id = stored_ids[idx];
        let mut seen_targets: HashSet<uuid::Uuid> = HashSet::new();

        // Scan for file references: `traits.rs`, `config.py`, etc.
        for cap in file_re.captures_iter(&pc.content) {
            let stem = cap.get(1).unwrap().as_str();
            // Look up code chunks by basename
            if let Some(code_chunks) = code_index.get(stem) {
                // Use the first code chunk from the matched file
                if let Some((target_idx, _)) = code_chunks.first()
                    && *target_idx < stored_ids.len() {
                        let to_id = stored_ids[*target_idx];
                        if to_id != from_id && seen_targets.insert(to_id) {
                            edges.push((from_id, "references".to_string(), to_id));
                        }
                    }
            }
        }

        // Scan for PascalCase symbol references in backticks: `GraphStore`, `Config`, etc.
        for cap in symbol_re.captures_iter(&pc.content) {
            let symbol = cap.get(1).unwrap().as_str();
            // Find a code chunk that contains this symbol
            let mut found = false;
            for chunks in code_index.values() {
                for (target_idx, target_pc) in chunks {
                    if *target_idx < stored_ids.len() && target_pc.content.contains(symbol) {
                        let to_id = stored_ids[*target_idx];
                        if to_id != from_id && seen_targets.insert(to_id) {
                            edges.push((from_id, "references".to_string(), to_id));
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
        }
    }

    edges
}

/// Resolve pipeline relations and store them as graph edges.
///
/// Uses `(source_file, start_line)` to match relations to stored chunks.
/// Best-effort: unresolvable cross-file references are silently skipped.
pub(crate) async fn wire_pipeline_relations(
    relations: &[corvia_kernel::chunking_strategy::ChunkRelation],
    processed: &[corvia_kernel::chunking_strategy::ProcessedChunk],
    stored_ids: &[uuid::Uuid],
    graph: &dyn GraphStore,
) -> usize {
    let mut relations_stored = 0;
    for rel in relations {
        // Find the source chunk by (source_file, start_line) match
        let from_idx = processed.iter().position(|pc| {
            pc.metadata.source_file == rel.from_source_file && pc.start_line == rel.from_start_line
        });
        let from_uuid = match from_idx {
            Some(idx) if idx < stored_ids.len() => stored_ids[idx],
            _ => continue,
        };

        // Find the target chunk by file name (and optionally symbol name)
        let to_uuid = processed.iter().zip(stored_ids.iter()).find_map(|(pc, id)| {
            if pc.metadata.source_file == rel.to_file {
                if let Some(ref name) = rel.to_name {
                    if pc.content.contains(name) {
                        return Some(*id);
                    }
                } else {
                    return Some(*id);
                }
            }
            None
        });

        if let Some(to_uuid) = to_uuid {
            // Filter self-edges for "imports" and "calls" relations
            if (rel.relation == "imports" || rel.relation == "calls") && to_uuid == from_uuid {
                continue;
            }
            if let Err(e) = graph.relate(&from_uuid, &rel.relation, &to_uuid, None).await {
                tracing::warn!("Failed to store relation: {e}");
            } else {
                relations_stored += 1;
            }
        }
    }
    relations_stored
}

fn print_server_search_results(results: &[server_client::SearchResultDto], _show_namespace: bool) {
    for (i, result) in results.iter().enumerate() {
        let file = result.source_file.as_deref().unwrap_or("unknown");
        let lines = match (result.start_line, result.end_line) {
            (Some(s), Some(e)) => format!(":{s}-{e}"),
            _ => String::new(),
        };
        println!("--- Result {} (score: {:.3}) ---", i + 1, result.score);
        println!("File: {file}{lines}");
        let preview: String = result.content.lines().take(5).collect::<Vec<_>>().join("\n");
        println!("{preview}");
        if result.content.lines().count() > 5 {
            println!("  ...");
        }
        println!();
    }
}

fn print_search_results(results: &[corvia_common::types::SearchResult], show_namespace: bool) {
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
        if show_namespace {
            println!(
                "--- Result {} [{}] (score: {:.3}) ---",
                i + 1,
                result.entry.workstream,
                result.score
            );
        } else {
            println!("--- Result {} (score: {:.3}) ---", i + 1, result.score);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_kernel::chunking_strategy::{ChunkMetadata, ProcessedChunk, ProcessingInfo};

    fn make_chunk(content: &str, source_file: &str, start_line: u32) -> ProcessedChunk {
        ProcessedChunk {
            content: content.to_string(),
            original_content: content.to_string(),
            chunk_type: "file".to_string(),
            start_line,
            end_line: start_line + 10,
            metadata: ChunkMetadata {
                source_file: source_file.to_string(),
                language: None,
                parent_chunk_id: None,
                merge_group: None,
            },
            token_estimate: content.len() / 4,
            processing: ProcessingInfo {
                strategy_name: "test".to_string(),
                was_split: false,
                was_merged: false,
                overlap_tokens: 0,
            },
        }
    }

    #[test]
    fn test_wire_doc_to_code_relations() {
        let md_chunk = make_chunk(
            "See `traits.rs` for the `GraphStore` trait.",
            "docs/architecture.md",
            1,
        );
        let code_chunk = make_chunk(
            "pub trait GraphStore {\n    fn relate(&self);\n}",
            "crates/corvia-kernel/src/traits.rs",
            1,
        );
        let other_code = make_chunk(
            "pub fn main() {\n    println!(\"hello\");\n}",
            "crates/corvia-cli/src/main.rs",
            1,
        );

        let processed = vec![md_chunk, code_chunk, other_code];
        let stored_ids: Vec<uuid::Uuid> = (0..3).map(|_| uuid::Uuid::now_v7()).collect();

        let edges = wire_doc_to_code_relations(&processed, &stored_ids);

        // Should have an edge from markdown chunk to traits.rs (file ref)
        // and potentially to traits.rs again (symbol match for GraphStore)
        assert!(
            !edges.is_empty(),
            "should find at least one doc-to-code edge"
        );

        // All edges should start from the markdown chunk
        for (from, rel, to) in &edges {
            assert_eq!(*from, stored_ids[0], "edges should start from markdown chunk");
            assert_eq!(rel, "references");
            assert_ne!(*to, stored_ids[0], "no self-edges");
        }

        // Should have an edge to traits.rs chunk
        assert!(
            edges.iter().any(|(_, _, to)| *to == stored_ids[1]),
            "should have edge to traits.rs chunk"
        );
    }

    #[test]
    fn test_wire_doc_to_code_no_self_edges() {
        // A markdown file shouldn't create edges to itself
        let md_chunk = make_chunk(
            "This is README.md file",
            "README.md",
            1,
        );
        let processed = vec![md_chunk];
        let stored_ids = vec![uuid::Uuid::now_v7()];

        let edges = wire_doc_to_code_relations(&processed, &stored_ids);
        assert!(edges.is_empty(), "should not create self-edges");
    }

    #[test]
    fn test_wire_doc_to_code_no_edges_for_code_only() {
        // No markdown chunks → no edges
        let code_chunk = make_chunk(
            "pub fn hello() {}",
            "src/lib.rs",
            1,
        );
        let processed = vec![code_chunk];
        let stored_ids = vec![uuid::Uuid::now_v7()];

        let edges = wire_doc_to_code_relations(&processed, &stored_ids);
        assert!(edges.is_empty(), "code-only should produce no doc-to-code edges");
    }
}

// ---------------------------------------------------------------------------
// Bench commands
// ---------------------------------------------------------------------------

/// Run the eval suite against a corvia server.
async fn cmd_bench_run(server: &str, limit: usize, ab: bool) -> Result<()> {
    use std::process::Command;

    println!("corvia bench — Retrieval Quality Evaluation");
    println!("Server: {server}");
    println!("Top-K:  {limit}");
    println!();

    // Check server health
    let health_url = format!("{server}/health");
    let client = reqwest::Client::new();
    match client.get(&health_url).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("Server: OK");
        }
        _ => {
            anyhow::bail!("Server at {server} is not responding. Start with 'corvia serve'.");
        }
    }

    // Determine which script to run
    let script = if ab {
        "benchmarks/rag-retrieval/ab-test.py"
    } else {
        "benchmarks/rag-retrieval/eval.py"
    };

    // Find the script relative to workspace root
    let workspace_root = std::env::var("CORVIA_WORKSPACE_ROOT")
        .unwrap_or_else(|_| ".".to_string());
    let script_path = std::path::Path::new(&workspace_root).join(script);

    if !script_path.exists() {
        // Try relative to current dir
        let alt_path = std::path::Path::new(script);
        if !alt_path.exists() {
            eprintln!("Error: Eval script not found at {script_path:?} or {alt_path:?}");
            eprintln!("Run from workspace root or set CORVIA_WORKSPACE_ROOT.");
            return Ok(());
        }
    }

    let actual_path = if script_path.exists() {
        script_path
    } else {
        std::path::Path::new(script).to_path_buf()
    };

    println!("Running: python3 {}", actual_path.display());
    println!("{}", "-".repeat(70));

    let mut args = vec![
        actual_path.to_string_lossy().to_string(),
    ];
    // Both eval and ab-test scripts accept --server and --limit
    args.push("--server".into());
    args.push(server.into());
    args.push("--limit".into());
    args.push(limit.to_string());

    let status = Command::new("python3")
        .args(&args)
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run eval: {e}"))?;

    if !status.success() {
        eprintln!("Eval exited with non-zero status");
    }

    Ok(())
}

/// Show the latest benchmark results.
fn cmd_bench_report() -> Result<()> {
    let workspace_root = std::env::var("CORVIA_WORKSPACE_ROOT")
        .unwrap_or_else(|_| ".".to_string());
    let results_dir = std::path::Path::new(&workspace_root).join("benchmarks/rag-retrieval/results");
    if !results_dir.exists() {
        eprintln!("No benchmark results found. Run 'corvia bench run' first.");
        return Ok(());
    }

    // Find most recent result file
    let mut files: Vec<_> = std::fs::read_dir(results_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));

    if files.is_empty() {
        eprintln!("No benchmark results found. Run 'corvia bench run' first.");
        return Ok(());
    }

    let latest = &files[0];
    let content = std::fs::read_to_string(latest.path())?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    println!("corvia bench report — Latest Results");
    println!("File: {}", latest.path().display());
    println!("{}", "-".repeat(70));

    if let Some(summary) = data.get("summary") {
        println!("Source Recall@5:  {:.1}%", summary["avg_source_recall_at_5"].as_f64().unwrap_or(0.0) * 100.0);
        println!("Source Recall@10: {:.1}%", summary["avg_source_recall_at_10"].as_f64().unwrap_or(0.0) * 100.0);
        println!("Keyword Recall:   {:.1}%", summary["avg_keyword_recall"].as_f64().unwrap_or(0.0) * 100.0);
        println!("MRR:              {:.3}", summary["avg_mrr"].as_f64().unwrap_or(0.0));
        println!("Relevance Score:  {:.3}", summary["avg_relevance_score"].as_f64().unwrap_or(0.0));
        println!("Avg Latency:      {:.0}ms", summary["avg_latency_ms"].as_f64().unwrap_or(0.0));
    } else if let Some(graph) = data.get("graph_expand") {
        // A/B test result
        println!("A/B Test Results:");
        println!("  Graph Expand: Recall@5={:.1}% MRR={:.3} Lat={:.0}ms",
            graph["avg_recall_5"].as_f64().unwrap_or(0.0) * 100.0,
            graph["avg_mrr"].as_f64().unwrap_or(0.0),
            graph["avg_latency"].as_f64().unwrap_or(0.0));
        if let Some(vector) = data.get("vector") {
            println!("  Vector Only:  Recall@5={:.1}% MRR={:.3} Lat={:.0}ms",
                vector["avg_recall_5"].as_f64().unwrap_or(0.0) * 100.0,
                vector["avg_mrr"].as_f64().unwrap_or(0.0),
                vector["avg_latency"].as_f64().unwrap_or(0.0));
        }
    }

    Ok(())
}

/// Run A/B comparison: vector-only vs graph-expanded retrieval.
async fn cmd_bench_compare(server: &str, limit: usize) -> Result<()> {
    use std::process::Command;

    println!("corvia bench compare — A/B Retrieval Comparison");
    println!("Server: {server}");
    println!("Top-K:  {limit}");
    println!();

    // Check server health
    let health_url = format!("{server}/health");
    let client = reqwest::Client::new();
    match client.get(&health_url).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("Server: OK");
        }
        _ => {
            anyhow::bail!("Server at {server} is not responding. Start with 'corvia serve'.");
        }
    }

    let workspace_root = std::env::var("CORVIA_WORKSPACE_ROOT")
        .unwrap_or_else(|_| ".".to_string());

    // Prefer ab-test.py; fall back to running eval.py twice
    let ab_script = "benchmarks/rag-retrieval/ab-test.py";
    let ab_path = std::path::Path::new(&workspace_root).join(ab_script);
    let ab_alt = std::path::Path::new(ab_script);
    let ab_actual = if ab_path.exists() {
        Some(ab_path)
    } else if ab_alt.exists() {
        Some(ab_alt.to_path_buf())
    } else {
        None
    };

    if let Some(script) = ab_actual {
        println!("Running: python3 {} --server {server} --limit {limit}", script.display());
        println!("{}", "-".repeat(70));

        let status = Command::new("python3")
            .arg(script.to_string_lossy().as_ref())
            .args(["--server", server, "--limit", &limit.to_string()])
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to run ab-test: {e}"))?;

        if !status.success() {
            eprintln!("ab-test exited with non-zero status");
        }
    } else {
        // Fallback: run eval.py twice with different expand_graph settings
        let eval_script = "benchmarks/rag-retrieval/eval.py";
        let eval_path = std::path::Path::new(&workspace_root).join(eval_script);
        let eval_alt = std::path::Path::new(eval_script);
        let eval_actual = if eval_path.exists() {
            eval_path
        } else if eval_alt.exists() {
            eval_alt.to_path_buf()
        } else {
            anyhow::bail!(
                "Neither ab-test.py nor eval.py found. Run from workspace root or set CORVIA_WORKSPACE_ROOT."
            );
        };

        println!("ab-test.py not found, falling back to running eval.py twice");
        println!("{}", "-".repeat(70));

        println!("\n--- Run 1: graph_expand=true ---");
        let status = Command::new("python3")
            .arg(eval_actual.to_string_lossy().as_ref())
            .args(["--server", server, "--limit", &limit.to_string()])
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to run eval (graph_expand): {e}"))?;
        if !status.success() {
            eprintln!("eval.py (graph_expand) exited with non-zero status");
        }

        println!("\n--- Run 2: vector-only ---");
        let status = Command::new("python3")
            .arg(eval_actual.to_string_lossy().as_ref())
            .args(["--server", server, "--limit", &limit.to_string()])
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to run eval (vector-only): {e}"))?;
        if !status.success() {
            eprintln!("eval.py (vector-only) exited with non-zero status");
        }

        println!("\nNote: fallback mode runs eval.py twice but cannot toggle graph expansion.");
        println!("Install ab-test.py for proper A/B comparison with graph vs vector modes.");
    }

    Ok(())
}

