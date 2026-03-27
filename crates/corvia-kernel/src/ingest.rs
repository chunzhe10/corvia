//! Shared workspace ingestion logic used by both CLI and server.
//!
//! The core function [`run_workspace_ingest`] orchestrates the full adapter pipeline:
//! discover adapters → spawn processes → stream source files → chunk → embed → store.
//!
//! # Blocking I/O Warning
//!
//! [`ProcessAdapter`](crate::process_adapter::ProcessAdapter) performs synchronous I/O
//! (stdin/stdout pipes to child processes). Callers in an async context **must** wrap
//! calls to [`run_workspace_ingest`] in `tokio::task::spawn_blocking` or run on a
//! dedicated thread to avoid stalling the async runtime.

use crate::adapter_discovery;
use crate::chunking_pipeline::register_adapter_chunking;
use crate::chunking_strategy::{ChunkRelation, ProcessedChunk, SourceMetadata};
use crate::introspect::EMBED_BATCH_SIZE;
use crate::process_adapter::ProcessAdapter;
use crate::traits::{GraphStore, InferenceEngine, QueryableStore};
use corvia_common::config::CorviaConfig;
use corvia_common::constants::{CLAUDE_SESSIONS_ADAPTER, USER_HISTORY_SCOPE};
use corvia_common::types::KnowledgeEntry;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Progress trait
// ---------------------------------------------------------------------------

/// Callback trait for reporting ingestion progress.
///
/// CLI implementations use `println!`; server implementations use `tracing::info!`.
pub trait IngestProgress: Send + Sync {
    fn log(&self, msg: &str);
}

/// Simple println-based progress reporter for CLI use.
pub struct PrintProgress;

impl IngestProgress for PrintProgress {
    fn log(&self, msg: &str) {
        println!("{msg}");
    }
}

/// Tracing-based progress reporter for server use.
pub struct TracingProgress;

impl IngestProgress for TracingProgress {
    fn log(&self, msg: &str) {
        tracing::info!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Report / Status types
// ---------------------------------------------------------------------------

/// Per-repo stats from an ingestion run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RepoIngestStats {
    pub name: String,
    pub files: usize,
    pub chunks: usize,
    pub relations: usize,
}

/// Summary report of a full workspace ingestion.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IngestReport {
    pub repos: Vec<RepoIngestStats>,
    pub docs_chunks: usize,
    pub session_turns: usize,
    pub total_chunks: usize,
}

/// Status of a server-side ingestion (for polling via GET /v1/ingest/status).
#[derive(Debug, Clone, Serialize)]
pub struct IngestStatus {
    pub state: IngestState,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
    pub error: Option<String>,
    pub report: Option<IngestReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IngestState {
    Idle,
    Running,
    Completed,
    Failed,
}

impl IngestStatus {
    pub fn idle() -> Self {
        Self {
            state: IngestState::Idle,
            started_at: None,
            finished_at: None,
            error: None,
            report: None,
        }
    }
}

/// RAII guard that resets an AtomicBool on drop (handles panics/early returns).
pub struct IngestGuard {
    flag: Arc<AtomicBool>,
}

impl IngestGuard {
    /// Attempt to acquire the ingest lock. Returns `None` if already held.
    pub fn try_acquire(flag: Arc<AtomicBool>) -> Option<Self> {
        if flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            Some(Self { flag })
        } else {
            None
        }
    }
}

impl Drop for IngestGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Shared utility functions (moved from CLI)
// ---------------------------------------------------------------------------

/// Infer `content_role` metadata from the source file path.
pub fn infer_content_role(source_file: &str) -> Option<String> {
    let lower = source_file.to_lowercase();
    let file_name = std::path::Path::new(source_file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Special files: instruction role
    if file_name == "AGENTS.md" || file_name == "CLAUDE.md" || file_name == "README.md" {
        return Some("instruction".into());
    }

    // Memory directory
    if lower.contains("/.memory/") || lower.starts_with(".memory/") || lower == ".memory" {
        return Some("memory".into());
    }

    // Directory-based inference for markdown/docs
    if lower.ends_with(".md") || lower.ends_with(".mdx") {
        if lower.contains("/rfcs/") || lower.starts_with("rfcs/") {
            return Some("design".into());
        }
        if lower.contains("/plans/") || lower.starts_with("plans/") {
            return Some("plan".into());
        }
        if lower.contains("/decisions/") || lower.starts_with("decisions/") {
            return Some("decision".into());
        }
        if lower.contains("/learnings/") || lower.starts_with("learnings/") {
            return Some("learning".into());
        }
        if lower.contains("/docs/") || lower.starts_with("docs/") {
            return Some("design".into());
        }
        return None;
    }

    // Code file extensions
    const CODE_EXTS: &[&str] = &[
        "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "rb", "php",
        "c", "cpp", "h", "hpp", "cs", "swift", "kt", "scala", "ex", "exs",
        "sh", "bash", "zsh", "toml", "yaml", "yml", "json",
    ];
    let ext = std::path::Path::new(source_file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if CODE_EXTS.contains(&ext) {
        return Some("code".into());
    }

    None
}

/// Infer `source_origin` metadata from repo context and file path.
///
/// - Repo ingestion (repo_name is Some): `"repo:<name>"`
/// - `.memory/` files: `"memory"`
/// - Workspace docs (repo_name is None): `"workspace"`
pub fn infer_source_origin(repo_name: Option<&str>, source_file: &str) -> Option<String> {
    let lower = source_file.to_lowercase();
    if lower.contains("/.memory/") || lower.starts_with(".memory/") {
        return Some("memory".into());
    }
    if let Some(name) = repo_name {
        return Some(format!("repo:{name}"));
    }
    Some("workspace".into())
}

/// Simple glob matching for blocked paths (supports trailing `*` only).
pub fn blocked_path_match(pattern: &str, path: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        path.starts_with(prefix)
    } else {
        path == pattern
    }
}

/// Wire chunk-level graph relations produced by the chunking pipeline.
///
/// Resolves `ChunkRelation` references (source_file + start_line) to stored
/// entry UUIDs, then creates graph edges. Returns the number of relations stored.
pub async fn wire_pipeline_relations(
    relations: &[ChunkRelation],
    processed: &[ProcessedChunk],
    stored_ids: &[uuid::Uuid],
    graph: &dyn GraphStore,
) -> usize {
    let mut relations_stored = 0;
    let mut source_miss = 0u64;
    let mut target_miss = 0u64;
    for rel in relations {
        let from_idx = processed.iter().position(|pc| {
            pc.metadata.source_file == rel.from_source_file && pc.start_line == rel.from_start_line
        });
        let from_uuid = match from_idx {
            Some(idx) if idx < stored_ids.len() => stored_ids[idx],
            _ => {
                source_miss += 1;
                tracing::debug!(
                    source_file = %rel.from_source_file,
                    start_line = rel.from_start_line,
                    relation = %rel.relation,
                    "wire_pipeline_relations: source chunk not found"
                );
                continue;
            }
        };

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
            if (rel.relation == "imports" || rel.relation == "calls") && to_uuid == from_uuid {
                continue;
            }
            if let Err(e) = graph.relate(&from_uuid, &rel.relation, &to_uuid, None).await {
                tracing::warn!("Failed to store relation: {e}");
            } else {
                relations_stored += 1;
            }
        } else {
            target_miss += 1;
            tracing::debug!(
                target_file = %rel.to_file,
                target_name = ?rel.to_name,
                relation = %rel.relation,
                "wire_pipeline_relations: target chunk not found"
            );
        }
    }
    if source_miss > 0 || target_miss > 0 {
        let total = relations.len();
        if relations_stored * 2 < total {
            tracing::warn!(
                total,
                stored = relations_stored,
                source_miss,
                target_miss,
                "wire_pipeline_relations: most relations failed to resolve — check adapter version"
            );
        } else {
            tracing::info!(
                total,
                stored = relations_stored,
                source_miss,
                target_miss,
                "wire_pipeline_relations: relation wiring summary"
            );
        }
    }
    relations_stored
}

// ---------------------------------------------------------------------------
// Resolve repo path (shared with CLI)
// ---------------------------------------------------------------------------

/// Resolve the filesystem path for a repo, preferring local override if valid.
fn resolve_repo_path(workspace_root: &Path, repos_dir: &str, repo: &corvia_common::config::RepoConfig) -> PathBuf {
    if let Some(local) = &repo.local {
        let local_path = PathBuf::from(local);
        if local_path.join(".git").exists() {
            return local_path;
        }
    }
    workspace_root.join(repos_dir).join(&repo.name)
}

// ---------------------------------------------------------------------------
// Core ingest orchestration
// ---------------------------------------------------------------------------

/// Parameters for [`run_workspace_ingest`], bundled to stay under clippy's
/// `too_many_arguments` threshold.
pub struct WorkspaceIngestCtx<'a> {
    pub root: &'a Path,
    pub config: &'a CorviaConfig,
    pub store: Arc<dyn QueryableStore>,
    pub graph: Arc<dyn GraphStore>,
    pub engine: Arc<dyn InferenceEngine>,
    pub repo_filter: Option<&'a str>,
    pub session_lock: Option<&'a tokio::sync::Mutex<()>>,
    pub progress: &'a dyn IngestProgress,
}

/// Run workspace ingestion using pre-existing store/graph/engine instances.
///
/// This function contains **blocking I/O** (adapter process pipes, chunking).
/// Callers in an async context must wrap in `spawn_blocking` or a dedicated thread.
///
/// If `session_lock` is provided, it will be acquired before the Claude session
/// history phase to prevent races with concurrent session ingest endpoints.
pub async fn run_workspace_ingest(ctx: WorkspaceIngestCtx<'_>) -> anyhow::Result<IngestReport> {
    let WorkspaceIngestCtx {
        root,
        config,
        store,
        graph,
        engine,
        repo_filter,
        session_lock,
        progress,
    } = ctx;
    let ws = config
        .workspace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace — no [workspace] section in corvia.toml"))?;

    // Discover adapters once for the whole workspace
    let extra_dirs = config
        .adapters
        .as_ref()
        .map(|a| a.search_dirs.clone())
        .unwrap_or_default();
    let discovered = adapter_discovery::discover_adapters(&extra_dirs);

    if discovered.is_empty() {
        anyhow::bail!(
            "No adapters found. Install corvia-adapter-basic to PATH or \
             ~/.config/corvia/adapters/. Adapter binaries must be named corvia-adapter-<name>."
        );
    }

    let default_name = config.adapters.as_ref().and_then(|a| a.default.as_deref());
    let mut report = IngestReport::default();

    // --- Phase 1: Repos ---
    let repos_to_ingest: Vec<&corvia_common::config::RepoConfig> = if let Some(name) = repo_filter {
        let repo = ws
            .repos
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| anyhow::anyhow!("Repo '{}' not found in workspace config", name))?;
        vec![repo]
    } else {
        ws.repos.iter().collect()
    };

    for repo_config in &repos_to_ingest {
        let repo_path = resolve_repo_path(root, &ws.repos_dir, repo_config);
        if !repo_path.exists() {
            progress.log(&format!(
                "  Skipping {} — not cloned (run corvia workspace init)",
                repo_config.name
            ));
            continue;
        }

        let repo_path_str = repo_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path for repo {}", repo_config.name))?;

        // Resolve adapter for this repo
        let explicit = config
            .sources
            .as_ref()
            .and_then(|sources| {
                sources.iter().find(|s| s.path == repo_path_str).and_then(|s| s.adapter.as_deref())
            });
        let adapter_info = adapter_discovery::resolve_adapter(repo_path_str, &discovered, explicit, default_name)
            .ok_or_else(|| anyhow::anyhow!("No suitable adapter found for '{}'", repo_path_str))?;

        progress.log(&format!(
            "\nIngesting {} (namespace: {}, adapter: {})...",
            repo_config.name, repo_config.namespace, adapter_info.metadata.name
        ));

        // Spawn adapter process
        let mut process = ProcessAdapter::new(
            adapter_info.binary_path.clone(),
            adapter_info.metadata.clone(),
        );
        process.spawn().map_err(|e| anyhow::anyhow!(e))?;

        // Ingest source files via IPC
        let source_files = process
            .ingest(repo_path_str, &config.project.scope_id)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Build chunking pipeline with adapter strategies
        let mut pipeline = crate::create_chunking_pipeline(config);

        // Register adapter's chunking extensions via IPC
        if !adapter_info.metadata.chunking_extensions.is_empty() {
            let adapter_arc = std::sync::Arc::new(std::sync::Mutex::new(
                ProcessAdapter::new(adapter_info.binary_path.clone(), adapter_info.metadata.clone()),
            ));
            adapter_arc.lock().unwrap().spawn().map_err(|e| anyhow::anyhow!(e))?;
            register_adapter_chunking(
                pipeline.registry_mut(),
                adapter_arc,
                &adapter_info.metadata.chunking_extensions,
            );
        }

        let (processed, pipeline_relations, pipeline_report) = pipeline.process_batch(&source_files)?;
        progress.log(&format!(
            "  {} files → {} chunks ({} merged, {} split)",
            pipeline_report.files_processed, pipeline_report.total_chunks,
            pipeline_report.chunks_merged, pipeline_report.chunks_split
        ));

        // Build lookup for adapter-provided SourceMetadata overrides
        let src_meta_lookup: HashMap<&str, &SourceMetadata> =
            source_files.iter().map(|sf| (sf.metadata.file_path.as_str(), &sf.metadata)).collect();

        // Convert ProcessedChunks to KnowledgeEntries, embed, and store
        let entries: Vec<KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let src_meta = src_meta_lookup.get(pc.metadata.source_file.as_str());
                let mut entry = KnowledgeEntry::new(
                    pc.content.clone(),
                    config.project.scope_id.clone(),
                    pc.metadata.source_file.clone(),
                );
                entry.workstream = src_meta
                    .and_then(|m| m.workstream.clone())
                    .unwrap_or_else(|| repo_config.namespace.clone());
                entry.metadata = corvia_common::types::EntryMetadata {
                    source_file: Some(pc.metadata.source_file.clone()),
                    language: pc.metadata.language.clone(),
                    chunk_type: Some(pc.chunk_type.clone()),
                    start_line: Some(pc.start_line),
                    end_line: Some(pc.end_line),
                    content_role: src_meta
                        .and_then(|m| m.content_role.clone())
                        .or_else(|| infer_content_role(&pc.metadata.source_file)),
                    source_origin: src_meta
                        .and_then(|m| m.source_origin.clone())
                        .or_else(|| infer_source_origin(
                            Some(&repo_config.name),
                            &pc.metadata.source_file,
                        )),
                };
                entry
            })
            .collect();

        let total = entries.len();
        let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
        let mut stored = 0;
        for batch in entries.chunks(EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
            let embeddings = engine.embed_batch(&texts).await?;

            for (entry, embedding) in batch.iter().zip(embeddings) {
                let mut entry = entry.clone();
                entry.embedding = Some(embedding);
                store.insert(&entry).await?;
                stored_ids.push(entry.id);
                stored += 1;
            }
            progress.log(&format!("  embedded and stored {}/{}", stored, total));
        }

        // Wire relations from pipeline
        let relations_stored = if !pipeline_relations.is_empty() {
            wire_pipeline_relations(
                &pipeline_relations, &processed, &stored_ids, &*graph,
            ).await
        } else {
            0
        };
        if relations_stored > 0 {
            progress.log(&format!("  {relations_stored} graph relations stored"));
        } else if processed.len() > 10 {
            tracing::warn!(
                chunks = processed.len(),
                input_relations = pipeline_relations.len(),
                "0 graph relations stored for {} chunks — check adapter version",
                processed.len()
            );
        }

        // Shutdown adapter
        process.shutdown().map_err(|e| anyhow::anyhow!(e))?;

        progress.log(&format!(
            "  {} chunks stored for namespace '{}'",
            stored, repo_config.namespace
        ));

        report.repos.push(RepoIngestStats {
            name: repo_config.name.clone(),
            files: pipeline_report.files_processed,
            chunks: stored,
            relations: relations_stored,
        });
        report.total_chunks += stored;
    }

    // --- Phase 2: Workspace docs (only on full ingest) ---
    if repo_filter.is_none()
        && let Some(docs_config) = ws.docs.as_ref()
        && let Some(workspace_docs_dir) = &docs_config.workspace_docs
    {
        let docs_path = root.join(workspace_docs_dir);
        if docs_path.exists() && docs_path.is_dir() {
            let docs_path_str = docs_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid workspace docs path"))?;

            let blocked = docs_config
                .rules
                .as_ref()
                .map(|r| r.blocked_paths.clone())
                .unwrap_or_default();

            let adapter_info = adapter_discovery::resolve_adapter(
                docs_path_str,
                &discovered,
                None,
                default_name,
            );

            if let Some(adapter_info) = adapter_info {
                progress.log(&format!(
                    "\nIngesting workspace docs ({})...",
                    workspace_docs_dir
                ));

                let mut process = ProcessAdapter::new(
                    adapter_info.binary_path.clone(),
                    adapter_info.metadata.clone(),
                );
                process.spawn().map_err(|e| anyhow::anyhow!(e))?;

                let source_files = process
                    .ingest(docs_path_str, &config.project.scope_id)
                    .map_err(|e| anyhow::anyhow!(e))?;

                // Filter: only allowed subdirs, exclude blocked paths
                let allowed = &docs_config.allowed_workspace_subdirs;
                let source_files: Vec<_> = source_files
                    .into_iter()
                    .filter(|sf| {
                        let path = &sf.metadata.file_path;
                        let in_allowed = allowed.is_empty()
                            || allowed.iter().any(|sub| {
                                path.starts_with(&format!("{sub}/"))
                                    || path.starts_with(sub.as_str())
                            });
                        let is_blocked = blocked
                            .iter()
                            .any(|bp| blocked_path_match(bp, path));
                        in_allowed && !is_blocked
                    })
                    .collect();

                if source_files.is_empty() {
                    progress.log("  No docs files to ingest.");
                } else {
                    let pipeline = crate::create_chunking_pipeline(config);
                    let (processed, pipeline_relations, pipeline_report) =
                        pipeline.process_batch(&source_files)?;

                    progress.log(&format!(
                        "  {} files → {} chunks ({} merged, {} split)",
                        pipeline_report.files_processed,
                        pipeline_report.total_chunks,
                        pipeline_report.chunks_merged,
                        pipeline_report.chunks_split
                    ));

                    let docs_meta_lookup: HashMap<&str, &SourceMetadata> =
                        source_files.iter().map(|sf| (sf.metadata.file_path.as_str(), &sf.metadata)).collect();

                    let entries: Vec<KnowledgeEntry> = processed
                        .iter()
                        .map(|pc| {
                            let src_meta = docs_meta_lookup.get(pc.metadata.source_file.as_str());
                            let mut entry = KnowledgeEntry::new(
                                pc.content.clone(),
                                config.project.scope_id.clone(),
                                pc.metadata.source_file.clone(),
                            );
                            entry.workstream = src_meta
                                .and_then(|m| m.workstream.clone())
                                .unwrap_or_else(|| "docs".to_string());
                            entry.metadata = corvia_common::types::EntryMetadata {
                                source_file: Some(pc.metadata.source_file.clone()),
                                language: pc.metadata.language.clone(),
                                chunk_type: Some(pc.chunk_type.clone()),
                                start_line: Some(pc.start_line),
                                end_line: Some(pc.end_line),
                                content_role: src_meta
                                    .and_then(|m| m.content_role.clone())
                                    .or_else(|| infer_content_role(&pc.metadata.source_file)),
                                source_origin: src_meta
                                    .and_then(|m| m.source_origin.clone())
                                    .or_else(|| infer_source_origin(None, &pc.metadata.source_file)),
                            };
                            entry
                        })
                        .collect();

                    let total = entries.len();
                    let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
                    let mut stored = 0;
                    for batch in entries.chunks(EMBED_BATCH_SIZE) {
                        let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
                        let embeddings = engine.embed_batch(&texts).await?;
                        for (entry, embedding) in batch.iter().zip(embeddings) {
                            let mut entry = entry.clone();
                            entry.embedding = Some(embedding);
                            store.insert(&entry).await?;
                            stored_ids.push(entry.id);
                            stored += 1;
                        }
                        progress.log(&format!("  embedded and stored {}/{}", stored, total));
                    }

                    if !pipeline_relations.is_empty() {
                        let relations_stored = wire_pipeline_relations(
                            &pipeline_relations, &processed, &stored_ids, &*graph,
                        ).await;
                        if relations_stored > 0 {
                            progress.log(&format!("  {relations_stored} graph relations stored"));
                        }
                    }

                    progress.log(&format!("  {} chunks stored for workspace docs", stored));
                    report.docs_chunks = stored;
                    report.total_chunks += stored;
                }

                process.shutdown().map_err(|e| anyhow::anyhow!(e))?;
            } else {
                progress.log(
                    "\n  Skipping workspace docs — no suitable adapter found \
                     (install corvia-adapter-basic)"
                );
            }
        }
    }

    // --- Phase 3: Claude Code session history (only on full ingest) ---
    if repo_filter.is_none() {
        let session_scope = config
            .scope
            .as_ref()
            .and_then(|scopes| scopes.iter().find(|s| s.id == USER_HISTORY_SCOPE));

        if session_scope.is_some() {
            // Acquire session lock if provided (prevents race with /v1/ingest/sessions)
            let _session_guard = if let Some(lock) = session_lock {
                Some(lock.lock().await)
            } else {
                None
            };

            let adapter_info = discovered.iter().find(|a| a.metadata.domain == CLAUDE_SESSIONS_ADAPTER);
            if let Some(adapter_info) = adapter_info {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
                let sessions_dir = PathBuf::from(home).join(".claude").join("sessions");

                if sessions_dir.is_dir() {
                    progress.log("\nIngesting Claude Code session history...");

                    let sessions_path = sessions_dir
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("Sessions directory path is not valid UTF-8"))?;

                    let mut process = ProcessAdapter::new(
                        adapter_info.binary_path.clone(),
                        adapter_info.metadata.clone(),
                    );
                    process.spawn().map_err(|e| anyhow::anyhow!(e))?;

                    let source_files = process
                        .ingest(sessions_path, USER_HISTORY_SCOPE)
                        .map_err(|e| anyhow::anyhow!(e))?;

                    if source_files.is_empty() {
                        progress.log("  No new sessions to ingest.");
                    } else {
                        let session_count = {
                            let mut seen = std::collections::HashSet::new();
                            source_files.iter().for_each(|sf| { seen.insert(&sf.metadata.file_path); });
                            seen.len()
                        };

                        progress.log(&format!(
                            "  {} sessions → {} turn entries",
                            session_count, source_files.len()
                        ));

                        let entries: Vec<KnowledgeEntry> = source_files
                            .iter()
                            .map(|sf| {
                                let mut entry = KnowledgeEntry::new(
                                    sf.content.clone(),
                                    USER_HISTORY_SCOPE.to_string(),
                                    sf.metadata.source_version.clone(),
                                );
                                entry.workstream = sf.metadata.workstream.clone().unwrap_or_default();
                                entry.metadata = corvia_common::types::EntryMetadata {
                                    source_file: Some(sf.metadata.file_path.clone()),
                                    language: sf.metadata.language.clone(),
                                    chunk_type: Some("session-turn".into()),
                                    start_line: None,
                                    end_line: None,
                                    content_role: sf.metadata.content_role.clone(),
                                    source_origin: sf.metadata.source_origin.clone(),
                                };
                                entry
                            })
                            .collect();

                        let parent_ids: Vec<Option<String>> = source_files
                            .iter()
                            .map(|sf| sf.metadata.parent_session_id.clone())
                            .collect();

                        let total = entries.len();
                        let mut stored = 0;
                        let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
                        for batch in entries.chunks(EMBED_BATCH_SIZE) {
                            let texts: Vec<String> = batch.iter().map(|e| e.content.clone()).collect();
                            let embeddings = engine.embed_batch(&texts).await?;
                            for (entry, embedding) in batch.iter().zip(embeddings) {
                                let mut entry = entry.clone();
                                entry.embedding = Some(embedding);
                                store.insert(&entry).await?;
                                stored_ids.push(entry.id);
                                stored += 1;
                            }
                            progress.log(&format!("  embedded and stored {}/{}", stored, total));
                        }

                        // Create spawned_by graph edges for subagent sessions
                        debug_assert_eq!(
                            source_files.len(),
                            stored_ids.len(),
                            "1:1 source-to-entry invariant violated in session ingest"
                        );
                        {
                            let mut session_first_entry: HashMap<String, uuid::Uuid> = HashMap::new();
                            for (i, sf) in source_files.iter().enumerate() {
                                session_first_entry
                                    .entry(sf.metadata.file_path.clone())
                                    .or_insert(stored_ids[i]);
                            }

                            let mut edges_created = 0u32;
                            for (i, parent_id) in parent_ids.iter().enumerate() {
                                if let Some(parent_sid) = parent_id {
                                    let child_entry_id = stored_ids[i];
                                    if let Some(&parent_entry_id) = session_first_entry.get(parent_sid) {
                                        if let Err(e) = graph
                                            .relate(&child_entry_id, "spawned_by", &parent_entry_id, None)
                                            .await
                                        {
                                            tracing::warn!("failed to create spawned_by edge: {e}");
                                        } else {
                                            edges_created += 1;
                                        }
                                    } else {
                                        let parent_sv = format!("{parent_sid}:turn-1");
                                        match store
                                            .get_by_source_version(USER_HISTORY_SCOPE, &parent_sv)
                                            .await
                                        {
                                            Ok(Some(parent_entry)) => {
                                                if let Err(e) = graph
                                                    .relate(&child_entry_id, "spawned_by", &parent_entry.id, None)
                                                    .await
                                                {
                                                    tracing::warn!("cross-batch spawned_by edge failed: {e}");
                                                } else {
                                                    edges_created += 1;
                                                }
                                            }
                                            Ok(None) => {
                                                tracing::debug!(
                                                    "parent session {parent_sid} not found — \
                                                     spawned_by edge skipped (not yet ingested or GC'd)"
                                                );
                                            }
                                            Err(e) => {
                                                tracing::warn!("cross-batch parent lookup failed: {e}");
                                            }
                                        }
                                    }
                                }
                            }
                            if edges_created > 0 {
                                progress.log(&format!("  {edges_created} spawned_by graph edges created"));
                            }
                        }

                        progress.log(&format!("  {} session turns stored", stored));
                        report.session_turns = stored;
                        report.total_chunks += stored;
                    }

                    process.shutdown().map_err(|e| anyhow::anyhow!(e))?;
                }
            }
        }
    }

    progress.log("\nWorkspace ingest complete.");
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_content_role() {
        assert_eq!(infer_content_role("AGENTS.md"), Some("instruction".into()));
        assert_eq!(infer_content_role("CLAUDE.md"), Some("instruction".into()));
        assert_eq!(infer_content_role("README.md"), Some("instruction".into()));
        assert_eq!(infer_content_role("src/.memory/foo.md"), Some("memory".into()));
        assert_eq!(infer_content_role("docs/rfcs/design.md"), Some("design".into()));
        assert_eq!(infer_content_role("docs/plans/plan.md"), Some("plan".into()));
        assert_eq!(infer_content_role("docs/decisions/d.md"), Some("decision".into()));
        assert_eq!(infer_content_role("docs/learnings/l.md"), Some("learning".into()));
        assert_eq!(infer_content_role("docs/other.md"), Some("design".into()));
        assert_eq!(infer_content_role("src/main.rs"), Some("code".into()));
        assert_eq!(infer_content_role("image.png"), None);
    }

    #[test]
    fn test_infer_source_origin() {
        assert_eq!(infer_source_origin(Some("corvia"), "src/main.rs"), Some("repo:corvia".into()));
        assert_eq!(infer_source_origin(None, "docs/readme.md"), Some("workspace".into()));
        assert_eq!(infer_source_origin(Some("x"), ".memory/foo"), Some("memory".into()));
    }

    #[test]
    fn test_blocked_path_match() {
        assert!(blocked_path_match("docs/superpowers/*", "docs/superpowers/foo.md"));
        assert!(blocked_path_match("docs/superpowers/*", "docs/superpowers/bar/baz.md"));
        assert!(!blocked_path_match("docs/superpowers/*", "docs/other/foo.md"));
        assert!(blocked_path_match("exact.md", "exact.md"));
        assert!(!blocked_path_match("exact.md", "other.md"));
    }

    #[test]
    fn test_ingest_guard_acquire() {
        let flag = Arc::new(AtomicBool::new(false));
        let guard = IngestGuard::try_acquire(flag.clone());
        assert!(guard.is_some());
        assert!(flag.load(Ordering::SeqCst));

        // Second acquire should fail
        let guard2 = IngestGuard::try_acquire(flag.clone());
        assert!(guard2.is_none());

        // Drop first guard, should release
        drop(guard);
        assert!(!flag.load(Ordering::SeqCst));

        // Now should succeed again
        let guard3 = IngestGuard::try_acquire(flag.clone());
        assert!(guard3.is_some());
    }

    #[test]
    fn test_ingest_status_idle() {
        let status = IngestStatus::idle();
        assert_eq!(status.state, IngestState::Idle);
        assert!(status.started_at.is_none());
        assert!(status.error.is_none());
    }
}
