mod mcp;
mod telemetry;

use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand};

use corvia_core::config::Config;
use corvia_core::embed::Embedder;
use corvia_core::entry::scan_entries;
use corvia_core::index::RedbIndex;
use corvia_core::search::SearchParams;
use corvia_core::tantivy_index::TantivyIndex;
use corvia_core::types::Kind;
use corvia_core::write::WriteParams;

#[derive(Parser)]
#[command(name = "corvia", version, about = "Organizational memory for AI agents")]
struct Cli {
    /// OTLP gRPC endpoint for exporting traces (e.g. http://localhost:4317).
    /// Can also be set via OTEL_EXPORTER_OTLP_ENDPOINT env var.
    #[arg(long, global = true)]
    otlp_endpoint: Option<String>,

    /// Project root directory (default: auto-discover by walking up to find .corvia/)
    #[arg(long, global = true)]
    base_dir: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ingest documents into the knowledge store
    Ingest {
        /// Path to ingest (defaults to current directory)
        path: Option<std::path::PathBuf>,
        /// Re-index from scratch
        #[arg(long)]
        fresh: bool,
        /// Path to pre-downloaded models (for airgapped environments)
        #[arg(long)]
        model_path: Option<std::path::PathBuf>,
    },
    /// Search the knowledge store
    Search {
        /// Search query
        query: String,
        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,
        /// Filter by knowledge kind
        #[arg(short, long)]
        kind: Option<String>,
        /// Maximum total tokens across all results
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// Write a knowledge entry
    Write {
        /// Content to write (markdown)
        content: String,
        /// Knowledge kind: decision, learning, instruction, reference
        #[arg(short, long, default_value = "learning")]
        kind: String,
        /// Tags (comma-separated)
        #[arg(short, long)]
        tags: Option<String>,
        /// IDs of entries to explicitly supersede (comma-separated)
        #[arg(short, long)]
        supersedes: Option<String>,
    },
    /// Show system status
    Status,
    /// Show recent operation traces
    Traces {
        /// Number of recent traces
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Filter by span name prefix
        #[arg(short, long)]
        filter: Option<String>,
    },
    /// Start stdio MCP server
    Mcp,
    /// Initialize corvia in the current directory
    Init {
        /// Auto-accept all prompts
        #[arg(long)]
        yes: bool,
        /// Force past version checks
        #[arg(long)]
        force: bool,
        /// Path to pre-downloaded embedding models
        #[arg(long)]
        model_path: Option<std::path::PathBuf>,
        /// Output format
        #[arg(long, value_parser = ["text", "json"])]
        format: Option<String>,
    },
}

/// Resolve the project root and load config.
/// Used by all commands except `Init` (which creates .corvia/).
fn load_config(base_dir_arg: Option<&Path>) -> anyhow::Result<(PathBuf, Config)> {
    let base_dir = corvia_core::discover::resolve_base_dir(base_dir_arg)?;
    let config = Config::load_discovered(&base_dir)?;
    Ok((base_dir, config))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize telemetry (trace file + OTLP if endpoint provided).
    let trace_file = Path::new(".corvia/traces.jsonl");
    let _telemetry_guard = telemetry::init_telemetry(
        cli.otlp_endpoint.as_deref(),
        Some(trace_file),
    )
    .expect("failed to initialize telemetry");

    let result = match cli.command {
        Command::Ingest {
            path,
            fresh,
            model_path,
        } => cmd_ingest(cli.base_dir.as_deref(), path.as_deref(), fresh, model_path),
        Command::Search {
            query,
            limit,
            kind,
            max_tokens,
        } => cmd_search(cli.base_dir.as_deref(), &query, limit, kind.as_deref(), max_tokens),
        Command::Write {
            content,
            kind,
            tags,
            supersedes,
        } => cmd_write(cli.base_dir.as_deref(), &content, &kind, tags.as_deref(), supersedes.as_deref()),
        Command::Status => cmd_status(cli.base_dir.as_deref()),
        Command::Traces { limit, filter } => cmd_traces(cli.base_dir.as_deref(), limit, filter.as_deref()),
        Command::Mcp => mcp::run(cli.base_dir.as_deref()).await,
        Command::Init { yes, force, model_path, format } => {
            cmd_init(cli.base_dir, yes, force, model_path, format)
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

fn cmd_ingest(
    base_dir_arg: Option<&Path>,
    path: Option<&Path>,
    fresh: bool,
    model_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let (base_dir, mut config) = load_config(base_dir_arg)?;

    // Override model_path from CLI flag if provided.
    if let Some(mp) = model_path {
        config.embedding.model_path = Some(mp);
    }

    let ingest_path = path.unwrap_or(&base_dir);
    let result = corvia_core::ingest::ingest(&config, ingest_path, fresh)?;

    println!(
        "Ingested {} entries ({} chunks). Superseded: {}. Skipped: {}.",
        result.entries_ingested,
        result.chunks_indexed,
        result.superseded_count,
        result.entries_skipped.len(),
    );

    if !result.entries_skipped.is_empty() {
        for (file, reason) in &result.entries_skipped {
            println!("  skipped: {file} ({reason})");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn cmd_search(
    base_dir_arg: Option<&Path>,
    query: &str,
    limit: usize,
    kind: Option<&str>,
    max_tokens: Option<usize>,
) -> anyhow::Result<()> {
    let (base_dir, config) = load_config(base_dir_arg)?;

    let cache_dir = config.embedding.model_path.as_deref();
    let embedder = Embedder::new(cache_dir, &config.embedding.model, &config.embedding.reranker_model)?;

    let kind_filter = match kind {
        Some(k) => Some(
            k.parse::<Kind>()
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        ),
        None => None,
    };

    let params = SearchParams {
        query: query.to_string(),
        limit,
        max_tokens,
        min_score: None,
        kind: kind_filter,
    };

    let response = corvia_core::search::search(&config, &base_dir, &embedder, &params)?;

    for result in &response.results {
        println!("[{:.3}] ({}) {}", result.score, result.kind, result.id);
        println!("{}", result.content);
        println!();
    }

    println!("Confidence: {:?}", response.quality.confidence);
    if let Some(suggestion) = &response.quality.suggestion {
        println!("Suggestion: {suggestion}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

fn cmd_write(
    base_dir_arg: Option<&Path>,
    content: &str,
    kind: &str,
    tags: Option<&str>,
    supersedes: Option<&str>,
) -> anyhow::Result<()> {
    let (base_dir, config) = load_config(base_dir_arg)?;

    let cache_dir = config.embedding.model_path.as_deref();
    let embedder = Embedder::new(cache_dir, &config.embedding.model, &config.embedding.reranker_model)?;

    let kind = kind
        .parse::<Kind>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let tags: Vec<String> = match tags {
        Some(t) => t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        None => vec![],
    };

    let supersedes: Vec<String> = match supersedes {
        Some(s) => s.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        None => vec![],
    };

    let params = WriteParams {
        content: content.to_string(),
        kind,
        tags,
        supersedes,
    };

    let response = corvia_core::write::write(&config, &base_dir, &embedder, params)?;

    println!("Entry {} {}.", response.id, response.action);

    if !response.superseded.is_empty() {
        println!("Superseded: {:?}", response.superseded);
    }

    if let Some(warning) = &response.warning {
        println!("Warning: {warning}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

fn cmd_status(base_dir_arg: Option<&Path>) -> anyhow::Result<()> {
    let (base_dir, config) = load_config(base_dir_arg)?;

    let redb_path = base_dir.join(config.redb_path());
    let tantivy_dir = base_dir.join(config.tantivy_dir());

    // Check if the index directory exists at all.
    let index_exists = redb_path.exists();

    if !index_exists {
        println!("Status: No index found. Run 'corvia ingest' first.");
        return Ok(());
    }

    let redb = RedbIndex::open(&redb_path)?;
    let tantivy = TantivyIndex::open(&tantivy_dir)?;

    let entry_count = redb.entry_count()?;
    let superseded_ids = redb.superseded_ids()?;
    let superseded_count = superseded_ids.len() as u64;
    let vector_count = redb.vector_count()?;
    let last_ingest = redb.get_meta("last_ingest")?.unwrap_or_else(|| "never".to_string());
    let doc_count = tantivy.doc_count();

    // Drift detection: compare actual entry files vs indexed count.
    let entries_dir = base_dir.join(config.entries_dir());
    let actual_files = scan_entries(&entries_dir)?;
    let actual_count = actual_files.len() as u64;
    let stale = actual_count != entry_count;

    println!("corvia status");
    println!("  entries:     {entry_count} ({superseded_count} superseded)");
    println!("  vectors:     {vector_count}");
    println!("  bm25 docs:   {doc_count}");
    println!("  last ingest: {last_ingest}");
    println!("  storage:     {}", config.data_dir.display());

    if stale {
        println!(
            "  WARNING: index may be stale ({actual_count} entry files vs {entry_count} indexed)"
        );
    }

    // Show recent traces.
    let trace_path = base_dir.join(&config.data_dir).join("traces.jsonl");
    let recent = corvia_core::trace::read_recent_traces(&trace_path, 10);
    if !recent.is_empty() {
        println!();
        println!("Recent operations:");
        for entry in &recent {
            let mut detail_parts: Vec<String> = Vec::new();
            if let Some(rc) = entry.attributes.get("result_count").and_then(|v| v.as_u64()) {
                detail_parts.push(format!("{rc} results"));
            }
            if let Some(conf) = entry.attributes.get("confidence").and_then(|v| v.as_str()) {
                detail_parts.push(format!("{conf} confidence"));
            }
            if let Some(action) = entry.attributes.get("action").and_then(|v| v.as_str()) {
                detail_parts.push(action.to_string());
            }
            if let Some(sc) = entry.attributes.get("superseded_count").and_then(|v| v.as_u64()) {
                if sc > 0 {
                    detail_parts.push(format!("{sc} superseded"));
                }
            }
            if let Some(ei) = entry.attributes.get("entries_ingested").and_then(|v| v.as_u64()) {
                detail_parts.push(format!("{ei} entries"));
            }
            if let Some(ci) = entry.attributes.get("chunks_indexed").and_then(|v| v.as_u64()) {
                detail_parts.push(format!("{ci} chunks"));
            }

            if detail_parts.is_empty() {
                println!("  {}  {}ms", entry.name, entry.elapsed_ms);
            } else {
                println!("  {}  {}ms  ({})", entry.name, entry.elapsed_ms, detail_parts.join(", "));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

fn cmd_traces(base_dir_arg: Option<&Path>, limit: usize, filter: Option<&str>) -> anyhow::Result<()> {
    let (base_dir, config) = load_config(base_dir_arg)?;
    let trace_path = base_dir.join(&config.data_dir).join("traces.jsonl");

    let mut traces = corvia_core::trace::read_recent_traces(&trace_path, limit);

    // Apply optional span name prefix filter.
    if let Some(prefix) = filter {
        traces.retain(|t| t.name.starts_with(prefix));
    }

    if traces.is_empty() {
        println!("No traces found.");
        return Ok(());
    }

    for entry in &traces {
        let mut detail_parts: Vec<String> = Vec::new();
        for (key, value) in &entry.attributes {
            match value {
                serde_json::Value::String(s) => detail_parts.push(format!("{key}={s}")),
                serde_json::Value::Number(n) => detail_parts.push(format!("{key}={n}")),
                serde_json::Value::Bool(b) => detail_parts.push(format!("{key}={b}")),
                _ => detail_parts.push(format!("{key}={value}")),
            }
        }

        if detail_parts.is_empty() {
            println!("  {}  {}ms", entry.name, entry.elapsed_ms);
        } else {
            println!(
                "  {}  {}ms  ({})",
                entry.name,
                entry.elapsed_ms,
                detail_parts.join(", ")
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

fn cmd_init(
    base_dir: Option<std::path::PathBuf>,
    yes: bool,
    force: bool,
    model_path: Option<std::path::PathBuf>,
    format: Option<String>,
) -> anyhow::Result<()> {
    use corvia_core::init::{self, InitOptions};

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let opts = InitOptions {
        yes: yes || !is_tty,
        base_dir,
        force,
        model_path,
    };

    let result = init::run_init(&opts)?;

    if format.as_deref() == Some("json") {
        let json = serde_json::json!({
            "status": "ok",
            "created": result.created,
            "config_migrated": result.config_migrated,
            "version_updated": result.version_updated,
            "actions": result.actions,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        if result.created {
            println!("corvia initialized (.corvia/)");
        } else {
            println!("corvia health check");
        }
        for action in &result.actions {
            println!("  {action}");
        }
        if result.actions.is_empty() {
            println!("  all checks passed");
        }
        println!();
        println!("Try: corvia search \"how does X work?\"");
    }

    Ok(())
}
