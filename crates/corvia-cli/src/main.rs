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
//! | `init` | Initialize a new Corvia store (`--store lite\|surrealdb\|postgres`) |
//! | `ingest` | Ingest a Git repository (or all workspace repos) |
//! | `search` | Semantic search across ingested knowledge |
//! | `serve` | Start the REST API and optional MCP server |
//! | `reason` | Run health checks and reasoning over a scope |
//! | `history` | Show the supersession chain for an entry |
//! | `evolution` | Show entries that changed within a time range |
//! | `graph` | Traverse the knowledge graph from a starting node |
//! | `relate` | Create a directed edge between two entries |
//! | `agent` | Multi-agent session management (start, list, commit, merge) |
//! | `workspace` | Workspace lifecycle (init, add, list, status, ingest) |
//! | `migrate` | Migrate data between storage backends (`--to lite\|surrealdb\|postgres`) |
//! | `demo` | Run the built-in demo workspace |
//!
//! See the [README](https://github.com/corvia/corvia) and
//! [ARCHITECTURE.md](https://github.com/corvia/corvia/blob/master/ARCHITECTURE.md)
//! for the full project overview.

mod upgrade;
mod workspace;

use anyhow::Result;
use clap::{Parser, Subcommand};
use corvia_common::config::CorviaConfig;
use corvia_kernel::docker::DockerProvisioner;
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
        /// Storage backend: lite (default), surrealdb, postgres
        #[arg(long, default_value = "lite")]
        store: String,
    },

    /// Start the REST API server
    Serve {
        /// Also start MCP server on /mcp endpoint
        #[arg(long)]
        mcp: bool,
    },

    /// Ingest a repository (or all workspace repos if no path given)
    Ingest {
        /// Path to repository (optional in workspace mode)
        path: Option<String>,
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
        /// Target storage backend: lite, surrealdb, postgres
        #[arg(long)]
        to: String,
        /// Show what would be migrated without making changes
        #[arg(long)]
        dry_run: bool,
    },

    /// Alias for 'migrate --to surrealdb' (deprecated, use migrate instead)
    #[command(hide = true)]
    Upgrade {
        /// Dry run: show what would be migrated without making changes
        #[arg(long)]
        dry_run: bool,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corvia=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { store } => cmd_init(&store).await?,
        Commands::Serve { mcp } => cmd_serve(mcp).await?,
        Commands::Ingest { path } => cmd_ingest(path.as_deref()).await?,
        Commands::Search { query, limit } => cmd_search(&query, limit).await?,
        Commands::Status => cmd_status().await?,
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
        Commands::Upgrade { dry_run } => upgrade::cmd_migrate("surrealdb", dry_run).await?,
    }

    Ok(())
}

async fn cmd_init(store: &str) -> Result<()> {
    let config = match store {
        "surrealdb" => {
            println!("Initializing Corvia (SurrealDB + vLLM)...");
            CorviaConfig::full_default()
        }
        "postgres" => {
            println!("Initializing Corvia (PostgreSQL + pgvector + vLLM)...");
            CorviaConfig::postgres_default()
        }
        "lite" => {
            println!("Initializing Corvia (LiteStore: zero Docker)...");
            CorviaConfig::default()
        }
        other => {
            anyhow::bail!("Unknown store type '{other}'. Valid options: lite, surrealdb, postgres");
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
        "surrealdb" => {
            // Provision Docker containers for SurrealDB
            let docker = DockerProvisioner::new()?;
            docker.start(
                config.storage.surrealdb_user.as_deref().unwrap_or("root"),
                config.storage.surrealdb_pass.as_deref().unwrap_or("root"),
            ).await?;

            let s = connect_store(&config).await?;
            s.init_schema().await?;
            println!("  SurrealDB running on port 8000.");
        }
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
                    provisioner.ensure_ready(&config.embedding.model, &config.merge.model).await?;
                    println!("  Corvia inference ready (embed: {}, chat: {})",
                        config.embedding.model, config.merge.model);
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

async fn cmd_serve(mcp: bool) -> Result<()> {
    let config = load_config()?;
    ensure_inference_ready(&config).await?;
    let (store, graph, temporal) = connect_full_store(&config).await?;
    let engine: Arc<dyn InferenceEngine> = connect_engine(&config);

    // Construct AgentCoordinator
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let gen_engine: std::sync::Arc<dyn corvia_kernel::traits::GenerationEngine> = match config.merge.provider {
        corvia_common::config::InferenceProvider::Corvia => {
            std::sync::Arc::new(corvia_kernel::grpc_chat::GrpcChatEngine::new(&config.embedding.url, &config.merge.model))
        }
        corvia_common::config::InferenceProvider::Ollama => {
            std::sync::Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url, &config.merge.model))
        }
        corvia_common::config::InferenceProvider::Vllm => {
            // vLLM chat not yet implemented — fall back to Ollama
            std::sync::Arc::new(corvia_kernel::ollama_chat::OllamaChatEngine::new(&config.embedding.url, &config.merge.model))
        }
    };
    let coordinator = Arc::new(AgentCoordinator::new(
        store.clone(),
        engine.clone(),
        data_dir,
        config.agent_lifecycle.clone(),
        config.merge.clone(),
        gen_engine,
    )?);
    println!("Agent coordination: enabled");

    // Construct RAG pipeline — auto-selects retriever based on graph availability.
    // generator: None until a GenerationEngine adapter is wired (ask mode unavailable).
    let rag = Arc::new(corvia_kernel::create_rag_pipeline(
        store.clone(),
        engine.clone(),
        Some(graph.clone()),
        None, // GenerationEngine: wired when M3.1 adapter lands
        &config,
    ));
    println!("RAG pipeline: enabled (retriever: {})", rag.retriever_name());

    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let state = Arc::new(corvia_server::rest::AppState { store, engine, coordinator, graph, temporal, data_dir, rag: Some(rag) });
    let mut app = corvia_server::rest::router(state.clone());

    if mcp {
        app = app.merge(corvia_server::mcp::mcp_router(state));
        println!("MCP endpoint: POST /mcp");
    }

    let addr = format!("{}:{}", config.server.host, config.server.port);
    println!("Corvia server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn cmd_ingest(path: Option<&str>) -> Result<()> {
    if let Some(path) = path {
        // D69 pipeline flow: source files → ChunkingPipeline → embed → store
        let config = load_config()?;
        let (store, graph) = connect_store_with_graph(&config).await?;
        ensure_inference_ready(&config).await?;
        let engine = connect_engine(&config);

        let mut adapter = GitAdapter::new();
        adapter.prepare(path);
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
        let entries: Vec<corvia_common::types::KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let mut entry = corvia_common::types::KnowledgeEntry::new(
                    pc.content.clone(),
                    config.project.scope_id.clone(),
                    pc.metadata.source_file.clone(),
                );
                entry.metadata = corvia_common::types::EntryMetadata {
                    source_file: Some(pc.metadata.source_file.clone()),
                    language: pc.metadata.language.clone(),
                    chunk_type: Some(pc.chunk_type.clone()),
                    start_line: Some(pc.start_line),
                    end_line: Some(pc.end_line),
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

        println!("Done. {stored} chunks from {total_files} files ingested from {path}.");
        println!("Next: corvia search \"your query\"");
    } else {
        // Workspace mode: ingest all workspace repos
        let root = std::env::current_dir()?;
        let config = load_config()?;
        if config.is_workspace() {
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

async fn cmd_status() -> Result<()> {
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
        corvia_common::config::StoreType::Surrealdb => {
            let docker = DockerProvisioner::new()?;
            let running = docker.is_running().await?;
            println!("Store: SurrealDB ({})",
                if running { "running" } else { "stopped" });

            if running {
                let store = connect_store(&config).await?;
                let count = store.count(&config.project.scope_id).await?;
                println!("Entries in scope '{}': {count}", config.project.scope_id);
            }
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

    // Show agent coordination status if available
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let coord_db_path = data_dir.join("coordination.redb");
    if coord_db_path.exists() {
        use corvia_kernel::agent_registry::AgentRegistry;
        match AgentRegistry::open(data_dir) {
            Ok(registry) => {
                let agents = registry.list_active().unwrap_or_default();
                let sessions = corvia_kernel::session_manager::SessionManager::from_db(registry.db().clone())
                    .map(|sm| sm.list_open().unwrap_or_default().len())
                    .unwrap_or(0);
                let queue_depth = corvia_kernel::merge_queue::MergeQueue::from_db(registry.db().clone())
                    .map(|mq| mq.depth().unwrap_or(0))
                    .unwrap_or(0);
                println!("\nAgent coordination:");
                println!("  Active agents: {}", agents.len());
                println!("  Open sessions: {sessions}");
                println!("  Merge queue depth: {queue_depth}");
            }
            Err(_) => {} // coordination not initialized yet
        }
    }

    Ok(())
}

async fn cmd_rebuild() -> Result<()> {
    let config = load_config()?;

    match config.storage.store_type {
        corvia_common::config::StoreType::Lite => {
            println!("Rebuilding LiteStore indexes from knowledge files...");
            let store = corvia_kernel::lite_store::LiteStore::open(
                std::path::Path::new(&config.storage.data_dir),
                config.embedding.dimensions,
            )?;
            let count = store.rebuild_from_files()?;
            println!("Rebuilt {count} entries.");
        }
        corvia_common::config::StoreType::Surrealdb => {
            println!("Rebuild is only needed for LiteStore. SurrealDB manages its own indexes.");
        }
        corvia_common::config::StoreType::Postgres => {
            println!("Rebuild is only needed for LiteStore. PostgreSQL manages its own indexes.");
        }
    }

    Ok(())
}

async fn cmd_test(check_only: bool, keep: bool, ci: bool) -> Result<()> {
    let config = CorviaConfig::default().with_env_overrides();
    let is_lite = matches!(config.storage.store_type, corvia_common::config::StoreType::Lite);

    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );

    // Phase 1: Check environment
    println!("  Checking environment...");
    if is_lite {
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
    } else {
        let docker = match DockerProvisioner::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  Infrastructure error: {e}");
                std::process::exit(2);
            }
        };
        let env_status = introspect.check_env(&docker).await;
        let mut all_running = true;
        for (service, running) in &env_status {
            if *running {
                println!("    {service}: running");
            } else {
                all_running = false;
                println!("    {service}: not running");
            }
        }
        if !all_running {
            println!("  Auto-provisioning missing services...");
            for (service, running) in &env_status {
                if !running {
                    let result = match *service {
                        "SurrealDB" => docker.start(
                            config.storage.surrealdb_user.as_deref().unwrap_or("root"),
                            config.storage.surrealdb_pass.as_deref().unwrap_or("root"),
                        ).await,
                        "vLLM" => docker.start_vllm(&config.embedding.model).await,
                        _ => Ok(()),
                    };
                    if let Err(e) = result {
                        eprintln!("  Infrastructure error: failed to provision {service}: {e}");
                        std::process::exit(2);
                    }
                    println!("    {service}: provisioned");
                }
            }
        }
    }

    if check_only {
        println!("  Environment check complete.");
        return Ok(());
    }

    // Phase 2: Create store for test
    let test_config = if is_lite {
        let mut tc = config.clone();
        let test_dir = std::env::temp_dir().join("corvia-introspect-test");
        tc.storage.data_dir = test_dir.to_string_lossy().to_string();
        tc
    } else {
        let mut tc = config.clone();
        tc.storage.surrealdb_ns = Some("corvia_introspect".into());
        tc.storage.surrealdb_db = Some("test".into());
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
    let mut adapter = GitAdapter::new();
    adapter.prepare(".");

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
    let is_lite = matches!(config.storage.store_type, corvia_common::config::StoreType::Lite);

    let introspect = Introspect::from_file_or_default(
        std::path::Path::new("tests/introspect.toml"),
    );

    // Phase 1: Check + auto-provision
    println!("  Checking environment...");
    if is_lite {
        println!("    LiteStore: no Docker required");
        let provisioner = OllamaProvisioner::new(&config.embedding.url);
        if provisioner.is_running().await {
            println!("    Ollama: running");
        } else {
            print!("    Ollama: provisioning...");
            provisioner.ensure_ready(&config.embedding.model).await?;
            println!(" done");
        }
    } else {
        let docker = DockerProvisioner::new()?;
        let env_status = introspect.check_env(&docker).await;
        for (service, running) in &env_status {
            if !*running {
                print!("    {service}: provisioning...");
                match *service {
                    "SurrealDB" => docker.start(
                        config.storage.surrealdb_user.as_deref().unwrap_or("root"),
                        config.storage.surrealdb_pass.as_deref().unwrap_or("root"),
                    ).await?,
                    "vLLM" => docker.start_vllm(&config.embedding.model).await?,
                    _ => {}
                }
                println!(" done");
            } else {
                println!("    {service}: running");
            }
        }
    }

    // Phase 2: Create store
    let demo_config = if is_lite {
        let mut tc = config.clone();
        let demo_dir = std::env::temp_dir().join("corvia-demo");
        tc.storage.data_dir = demo_dir.to_string_lossy().to_string();
        tc
    } else {
        let mut tc = config.clone();
        tc.storage.surrealdb_ns = Some("corvia_introspect".into());
        tc.storage.surrealdb_db = Some("demo".into());
        tc
    };

    let store = corvia_kernel::create_store(&demo_config).await?;
    store.init_schema().await?;

    let engine = connect_engine(&config);
    let mut adapter = GitAdapter::new();
    adapter.prepare(".");

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
                println!("{:<30} {:<12} {}", "AGENT ID", "STATUS", "DISPLAY NAME");
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
    }
    Ok(())
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

            if purge {
                if let Some(path) = repo_path {
                    if path.exists() && path.starts_with(root.join(&repos_dir)) {
                        std::fs::remove_dir_all(&path)?;
                        println!("Deleted cloned repo at {}", path.display());
                    }
                }
            }
            Ok(())
        }
        WorkspaceCommands::Ingest { repo, fresh } => {
            let root = std::env::current_dir()?;
            workspace::ingest_workspace(&root, repo.as_deref(), fresh).await?;
            Ok(())
        }
    }
}

// ---- Temporal / Graph / Reasoning commands ----

async fn cmd_history(entry_id: &str) -> Result<()> {
    let config = load_config()?;
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
    let (_store, _graph, temporal) = connect_full_store(&config).await?;

    let scope_id = scope.unwrap_or(&config.project.scope_id);

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
    let (_store, graph, _temporal) = connect_full_store(&config).await?;

    if let Some(id_str) = entry_id {
        // Show edges for a specific entry
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
        // Scope-wide: list all entries in scope that have edges
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
    let (store, graph, _temporal) = connect_full_store(&config).await?;

    let scope_id = scope.unwrap_or(&config.project.scope_id);

    // Load entries for the target scope only
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let entries = corvia_kernel::knowledge_files::read_scope(data_dir, scope_id)?;

    let reasoner = corvia_kernel::reasoner::Reasoner::new(&*store, &*graph);

    let mut findings = if let Some(check_str) = check {
        // Run a single check type
        let check_type = parse_check_type(check_str)?;
        reasoner.run_check(&entries, scope_id, check_type).await?
    } else {
        // Run all algorithmic checks
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

/// Truncate a string to at most `max_chars` characters, respecting UTF-8 char boundaries.
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
            provisioner.ensure_ready(&config.embedding.model, &config.merge.model).await?;
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
