use anyhow::{Context, Result};
use corvia_common::config::{CorviaConfig, InferenceProvider, RepoConfig, WorkspaceConfig};
use corvia_common::constants::{CLAUDE_SESSIONS_ADAPTER, USER_HISTORY_SCOPE};
use corvia_kernel::adapter_discovery;
use corvia_kernel::chunking_pipeline::register_adapter_chunking;
use corvia_kernel::process_adapter::ProcessAdapter;
use std::path::{Path, PathBuf};

/// Validate that a name is safe to use as a directory component.
///
/// Rejects empty names, path separators, `.`/`..`, and null bytes to prevent
/// path traversal when names are used to construct filesystem paths.
fn validate_dir_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
        || name == "."
        || name == ".."
    {
        anyhow::bail!(
            "Invalid name '{}': must not be empty, contain path separators, or be '.' or '..'",
            name
        );
    }
    Ok(())
}

/// Extract the repository name from a URL.
///
/// Handles HTTPS URLs (with or without trailing slash, with or without .git)
/// and SSH URLs (git@host:org/repo.git).
///
/// # Examples
/// ```ignore
/// assert_eq!(repo_name_from_url("https://github.com/org/my-repo.git"), "my-repo");
/// assert_eq!(repo_name_from_url("git@github.com:org/my-repo.git"), "my-repo");
/// ```
pub fn repo_name_from_url(url: &str) -> String {
    // Strip trailing slashes
    let url = url.trim_end_matches('/');

    // Take the last path segment (works for both HTTPS and SSH with colon)
    let last_segment = url
        .rsplit('/')
        .next()
        .or_else(|| url.rsplit(':').next())
        .unwrap_or(url);

    // Strip .git suffix, fall back to "unknown" if empty
    let name = last_segment.trim_end_matches(".git").to_string();
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
    }
}

/// Parse local override strings of the form "name=path".
///
/// Returns a vector of (name, path) pairs. Errors if any string
/// does not contain an `=` separator.
pub fn parse_local_overrides(locals: &[String]) -> Result<Vec<(String, String)>> {
    locals
        .iter()
        .map(|s| {
            let (name, path) = s
                .split_once('=')
                .with_context(|| format!("Invalid local override '{s}': expected name=path"))?;
            Ok((name.to_string(), path.to_string()))
        })
        .collect()
}

/// Generate a workspace config from CLI arguments.
///
/// Builds a `CorviaConfig` with a `WorkspaceConfig` containing repos
/// derived from the given URLs, with optional local path overrides applied.
pub fn generate_workspace_config(
    name: &str,
    repo_urls: &[String],
    local_overrides: &[(String, String)],
) -> Result<CorviaConfig> {
    validate_dir_name(name)?;
    let repos: Vec<RepoConfig> = repo_urls
        .iter()
        .map(|url| {
            let repo_name = repo_name_from_url(url);
            validate_dir_name(&repo_name)?;
            let local = local_overrides
                .iter()
                .find(|(n, _)| n == &repo_name)
                .map(|(_, p)| p.clone());
            Ok(RepoConfig {
                name: repo_name.clone(),
                url: url.clone(),
                local,
                namespace: repo_name,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut config = CorviaConfig::default();
    config.project.name = name.to_string();
    config.project.scope_id = name.to_string();
    config.workspace = Some(WorkspaceConfig {
        repos_dir: "repos".into(),
        repos,
        docs: None,
    });
    Ok(config)
}

/// Resolve the actual filesystem path for a repo.
///
/// If the repo has a local override and that path exists as a git repository
/// (i.e. contains a `.git` directory), returns the local path.
/// Otherwise, returns the cloned path under `{workspace_root}/{repos_dir}/{repo.name}`.
pub fn resolve_repo_path(workspace_root: &Path, repos_dir: &str, repo: &RepoConfig) -> PathBuf {
    if let Some(local) = &repo.local {
        let local_path = PathBuf::from(local);
        if local_path.join(".git").exists() {
            return local_path;
        }
    }
    workspace_root.join(repos_dir).join(&repo.name)
}

/// Clone a git repository to the target directory.
///
/// If the target directory already exists and contains a `.git` directory,
/// the clone is skipped and the existing path is returned.
pub fn clone_repo(url: &str, target_dir: &Path) -> Result<PathBuf> {
    if target_dir.join(".git").exists() {
        tracing::info!("Repository already cloned at {}", target_dir.display());
        return Ok(target_dir.to_path_buf());
    }

    tracing::info!("Cloning {} into {}", url, target_dir.display());
    let status = std::process::Command::new("git")
        .args(["clone", url])
        .arg(target_dir)
        .status()
        .with_context(|| format!("Failed to run 'git clone' for '{}'", url))?;
    if !status.success() {
        anyhow::bail!("git clone failed for '{}' into '{}'", url, target_dir.display());
    }

    Ok(target_dir.to_path_buf())
}

/// Create the workspace directory structure.
///
/// Creates:
/// - `{root}/` — workspace root
/// - `{root}/repos/` — with `.gitkeep`
/// - `{root}/.corvia/` — data directory
/// - `{root}/corvia.toml` — serialized config
/// - `{root}/.gitignore` — ignoring repos/ and ephemeral .corvia files
pub fn scaffold_workspace(root: &Path, config: &CorviaConfig) -> Result<()> {
    let repos_dir = config
        .workspace
        .as_ref()
        .map(|ws| ws.repos_dir.as_str())
        .unwrap_or("repos");

    // Create root directory
    std::fs::create_dir_all(root)
        .with_context(|| format!("Failed to create workspace root '{}'", root.display()))?;

    // Create repos directory with .gitkeep
    let repos_path = root.join(repos_dir);
    std::fs::create_dir_all(&repos_path)
        .with_context(|| format!("Failed to create repos dir '{}'", repos_path.display()))?;
    std::fs::write(repos_path.join(".gitkeep"), "")
        .with_context(|| "Failed to create repos/.gitkeep")?;

    // Create .corvia data directory
    let corvia_dir = root.join(".corvia");
    std::fs::create_dir_all(&corvia_dir)
        .with_context(|| format!("Failed to create .corvia dir '{}'", corvia_dir.display()))?;

    // Write corvia.toml
    let config_path = root.join("corvia.toml");
    config
        .save(&config_path)
        .with_context(|| format!("Failed to write '{}'", config_path.display()))?;

    // Write .gitignore (only if it doesn't already exist, to preserve manual edits)
    let gitignore_path = root.join(".gitignore");
    if !gitignore_path.exists() {
        let gitignore_content = format!(
            "# Cloned repositories\n{repos_dir}/*/\n\n# Ephemeral Corvia data\n.corvia/hnsw/\n.corvia/lite_store.redb\n.corvia/lite_store.redb.tmp\n.corvia/coordination.redb\n.corvia/coordination.redb.tmp\n"
        );
        std::fs::write(&gitignore_path, gitignore_content)
            .with_context(|| format!("Failed to write '{}'", gitignore_path.display()))?;
    }

    Ok(())
}

/// Initialize a workspace: clone missing repos, set up .corvia/, provision Ollama.
///
/// Loads `corvia.toml` from the workspace root, validates it has a `[workspace]`
/// section, clones any missing repos, initializes the knowledge store, writes a
/// `.gitignore` for ephemeral files, and provisions Ollama if configured.
///
/// Ollama provisioning failures are reported as warnings rather than errors,
/// since the user may want to set up the workspace structure first and start
/// Ollama later.
pub async fn init_workspace(root: &Path) -> Result<()> {
    let config_path = root.join("corvia.toml");
    if !config_path.exists() {
        anyhow::bail!(
            "No corvia.toml found in {}. Run 'corvia workspace create' first.",
            root.display()
        );
    }
    let config = CorviaConfig::load(&config_path)?;
    let ws = config
        .workspace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("corvia.toml has no [workspace] section"))?;

    // Clone or link repos
    println!("Initializing workspace repos...");
    for repo in &ws.repos {
        let repo_path = resolve_repo_path(root, &ws.repos_dir, repo);
        if repo_path.exists() {
            println!(
                "  {} — already present at {}",
                repo.name,
                repo_path.display()
            );
        } else {
            clone_repo(&repo.url, &repo_path)?;
        }
    }

    // Initialize .corvia store
    println!("Initializing knowledge store...");
    let data_dir = root.join(&config.storage.data_dir);
    std::fs::create_dir_all(&data_dir)?;

    let store = corvia_kernel::create_store_at(&config, &data_dir).await?;
    store.init_schema().await?;

    // Write .corvia/.gitignore for ephemeral files
    let corvia_gitignore = data_dir.join(".gitignore");
    if !corvia_gitignore.exists() {
        std::fs::write(
            &corvia_gitignore,
            "hnsw/\nlite_store.redb\nlite_store.redb.tmp\ncoordination.redb\ncoordination.redb.tmp\nstaging/\n",
        )?;
    }

    // Provision Ollama if using it
    if config.embedding.provider == InferenceProvider::Ollama {
        println!("Checking Ollama...");
        let provisioner =
            corvia_kernel::ollama_provisioner::OllamaProvisioner::new(&config.embedding.url);
        match provisioner.ensure_ready(&config.embedding.model).await {
            Ok(_) => println!("  Ollama ready with model {}", config.embedding.model),
            Err(e) => println!(
                "  Warning: Ollama not available ({}). Ingest will fail until Ollama is running.",
                e
            ),
        }
    }

    println!("Workspace initialized.");
    Ok(())
}

/// Add a repo to the workspace config
pub fn add_repo_to_config(
    config: &mut CorviaConfig,
    url: &str,
    name: Option<&str>,
    namespace: Option<&str>,
    local: Option<&str>,
) -> Result<()> {
    let ws = config
        .workspace
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace config"))?;

    let repo_name = name.map(String::from).unwrap_or_else(|| repo_name_from_url(url));
    validate_dir_name(&repo_name)?;
    if ws.repos.iter().any(|r| r.name == repo_name) {
        anyhow::bail!("Repo '{}' already exists in workspace", repo_name);
    }

    ws.repos.push(RepoConfig {
        name: repo_name.clone(),
        url: url.to_string(),
        local: local.map(String::from),
        namespace: namespace.map(String::from).unwrap_or(repo_name),
    });
    Ok(())
}

/// Remove a repo from the workspace config
pub fn remove_repo_from_config(config: &mut CorviaConfig, name: &str) -> Result<()> {
    let ws = config
        .workspace
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace config"))?;

    let before = ws.repos.len();
    ws.repos.retain(|r| r.name != name);
    if ws.repos.len() == before {
        anyhow::bail!("Repo '{}' not found in workspace", name);
    }
    Ok(())
}

/// Infer `content_role` from a source file path.
///
/// Rules (from docs-workflow design spec, Section 2 Population Rules):
/// - AGENTS.md / CLAUDE.md → "instruction"
/// - .memory/ files → "memory"
/// - Markdown in rfcs/ → "design"
/// - Markdown in plans/ → "plan"
/// - Markdown in decisions/ → "decision"
/// - Markdown in learnings/ → "learning"
/// - Markdown in docs/ (fallback) → "design"
/// - Code extensions (.rs, .py, .ts, etc.) → "code"
/// - Config/other → None (unclassified)
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
        // Check specific subdirs first (more specific wins)
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
        // Fallback: any markdown in a docs/ tree
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

/// Infer `source_origin` based on where the file came from.
///
/// - Repo ingestion (repo_name is Some): "repo:<name>"
/// - .memory/ files: "memory"
/// - Workspace docs (repo_name is None): "workspace"
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
fn blocked_path_match(pattern: &str, path: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        path.starts_with(prefix)
    } else {
        path == pattern
    }
}

/// Ingest all (or one) workspace repos with namespace isolation.
///
/// For each repo in the workspace, discovers an adapter at runtime, spawns
/// it as a process, streams source files via JSONL, runs them through the
/// chunking pipeline, embeds, and stores entries with namespace isolation.
///
/// Also ingests workspace-level docs (docs/decisions/, docs/learnings/, etc.)
/// when running a full workspace ingest (no repo filter).
///
/// The `fresh` parameter is accepted but not yet used (future: delete existing
/// entries before re-ingest).
pub async fn ingest_workspace(
    root: &Path,
    repo_filter: Option<&str>,
    _fresh: bool,
) -> Result<()> {
    let config_path = root.join("corvia.toml");
    let config = CorviaConfig::load(&config_path)?;
    let ws = config
        .workspace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace — no [workspace] section in corvia.toml"))?;

    let data_dir = root.join(&config.storage.data_dir);
    let (store, graph) = corvia_kernel::create_store_at_with_graph(&config, &data_dir).await?;
    let engine = corvia_kernel::create_engine(&config);

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

    let repos_to_ingest: Vec<&RepoConfig> = if let Some(name) = repo_filter {
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
            println!(
                "  Skipping {} — not cloned (run corvia workspace init)",
                repo_config.name
            );
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

        println!(
            "\nIngesting {} (namespace: {}, adapter: {})...",
            repo_config.name, repo_config.namespace, adapter_info.metadata.name
        );

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
        let mut pipeline = corvia_kernel::create_chunking_pipeline(&config);

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

        let (processed, pipeline_relations, report) = pipeline.process_batch(&source_files)?;
        println!(
            "  {} files → {} chunks ({} merged, {} split)",
            report.files_processed, report.total_chunks,
            report.chunks_merged, report.chunks_split
        );

        // Build lookup for adapter-provided SourceMetadata overrides
        let src_meta_lookup: std::collections::HashMap<&str, &corvia_kernel::chunking_strategy::SourceMetadata> =
            source_files.iter().map(|sf| (sf.metadata.file_path.as_str(), &sf.metadata)).collect();

        // Convert ProcessedChunks to KnowledgeEntries, embed, and store
        let entries: Vec<corvia_common::types::KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let src_meta = src_meta_lookup.get(pc.metadata.source_file.as_str());
                let mut entry = corvia_common::types::KnowledgeEntry::new(
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
            println!("  embedded and stored {}/{}", stored, total);
        }

        // Wire relations from pipeline
        if !pipeline_relations.is_empty() {
            let relations_stored = crate::wire_pipeline_relations(
                &pipeline_relations, &processed, &stored_ids, &*graph,
            ).await;
            if relations_stored > 0 {
                println!("  {relations_stored} graph relations stored");
            }
        }

        // Shutdown adapter
        process.shutdown().map_err(|e| anyhow::anyhow!(e))?;

        println!(
            "  {} chunks stored for namespace '{}'",
            stored, repo_config.namespace
        );
    }

    // --- Ingest workspace docs (only on full ingest, not single-repo) ---
    if repo_filter.is_none()
        && let Some(docs_config) = ws.docs.as_ref()
            && let Some(workspace_docs_dir) = &docs_config.workspace_docs {
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
                        println!(
                            "\nIngesting workspace docs ({})...",
                            workspace_docs_dir
                        );

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
                                // Must be in an allowed subdir (or no restriction)
                                let in_allowed = allowed.is_empty()
                                    || allowed.iter().any(|sub| {
                                        path.starts_with(&format!("{sub}/"))
                                            || path.starts_with(sub.as_str())
                                    });
                                // Must not match blocked paths
                                let is_blocked = blocked
                                    .iter()
                                    .any(|bp| blocked_path_match(bp, path));
                                in_allowed && !is_blocked
                            })
                            .collect();

                        if source_files.is_empty() {
                            println!("  No docs files to ingest.");
                        } else {
                            let pipeline =
                                corvia_kernel::create_chunking_pipeline(&config);
                            let (processed, pipeline_relations, report) =
                                pipeline.process_batch(&source_files)?;

                            println!(
                                "  {} files → {} chunks ({} merged, {} split)",
                                report.files_processed,
                                report.total_chunks,
                                report.chunks_merged,
                                report.chunks_split
                            );

                            // Build lookup for adapter-provided SourceMetadata overrides
                            let docs_meta_lookup: std::collections::HashMap<&str, &corvia_kernel::chunking_strategy::SourceMetadata> =
                                source_files.iter().map(|sf| (sf.metadata.file_path.as_str(), &sf.metadata)).collect();

                            let entries: Vec<corvia_common::types::KnowledgeEntry> =
                                processed
                                    .iter()
                                    .map(|pc| {
                                        let src_meta = docs_meta_lookup.get(pc.metadata.source_file.as_str());
                                        let mut entry =
                                            corvia_common::types::KnowledgeEntry::new(
                                                pc.content.clone(),
                                                config.project.scope_id.clone(),
                                                pc.metadata.source_file.clone(),
                                            );
                                        entry.workstream = src_meta
                                            .and_then(|m| m.workstream.clone())
                                            .unwrap_or_else(|| "docs".to_string());
                                        entry.metadata =
                                            corvia_common::types::EntryMetadata {
                                                source_file: Some(
                                                    pc.metadata.source_file.clone(),
                                                ),
                                                language: pc.metadata.language.clone(),
                                                chunk_type: Some(
                                                    pc.chunk_type.clone(),
                                                ),
                                                start_line: Some(pc.start_line),
                                                end_line: Some(pc.end_line),
                                                content_role: src_meta
                                                    .and_then(|m| m.content_role.clone())
                                                    .or_else(|| infer_content_role(
                                                        &pc.metadata.source_file,
                                                    )),
                                                source_origin: src_meta
                                                    .and_then(|m| m.source_origin.clone())
                                                    .or_else(|| infer_source_origin(
                                                        None,
                                                        &pc.metadata.source_file,
                                                    )),
                                            };
                                        entry
                                    })
                                    .collect();

                            let total = entries.len();
                            let mut stored_ids: Vec<uuid::Uuid> =
                                Vec::with_capacity(total);
                            let mut stored = 0;
                            for batch in
                                entries.chunks(corvia_kernel::introspect::EMBED_BATCH_SIZE)
                            {
                                let texts: Vec<String> =
                                    batch.iter().map(|e| e.content.clone()).collect();
                                let embeddings = engine.embed_batch(&texts).await?;
                                for (entry, embedding) in batch.iter().zip(embeddings) {
                                    let mut entry = entry.clone();
                                    entry.embedding = Some(embedding);
                                    store.insert(&entry).await?;
                                    stored_ids.push(entry.id);
                                    stored += 1;
                                }
                                println!(
                                    "  embedded and stored {}/{}",
                                    stored, total
                                );
                            }

                            if !pipeline_relations.is_empty() {
                                let relations_stored = crate::wire_pipeline_relations(
                                    &pipeline_relations,
                                    &processed,
                                    &stored_ids,
                                    &*graph,
                                )
                                .await;
                                if relations_stored > 0 {
                                    println!(
                                        "  {relations_stored} graph relations stored"
                                    );
                                }
                            }

                            println!(
                                "  {} chunks stored for workspace docs",
                                stored
                            );
                        }

                        process.shutdown().map_err(|e| anyhow::anyhow!(e))?;
                    } else {
                        println!(
                            "\n  Skipping workspace docs — no suitable adapter found \
                             (install corvia-adapter-basic)"
                        );
                    }
                }
            }

    // --- Ingest Claude Code session history (only on full ingest) ---
    if repo_filter.is_none() {
        // Check for a user-history scope in [[scope]] config
        let session_scope = config
            .scope
            .as_ref()
            .and_then(|scopes| scopes.iter().find(|s| s.id == USER_HISTORY_SCOPE));

        if session_scope.is_some() {
            let adapter_info = discovered.iter().find(|a| a.metadata.domain == CLAUDE_SESSIONS_ADAPTER);
            if let Some(adapter_info) = adapter_info {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
                let sessions_dir = PathBuf::from(home)
                    .join(".claude")
                    .join("sessions");

                if sessions_dir.is_dir() {
                    println!("\nIngesting Claude Code session history...");

                    let sessions_path = sessions_dir
                        .to_str()
                        .unwrap_or("~/.claude/sessions");

                    let mut process = ProcessAdapter::new(
                        adapter_info.binary_path.clone(),
                        adapter_info.metadata.clone(),
                    );
                    process.spawn().map_err(|e| anyhow::anyhow!(e))?;

                    let source_files = process
                        .ingest(sessions_path, USER_HISTORY_SCOPE)
                        .map_err(|e| anyhow::anyhow!(e))?;

                    if source_files.is_empty() {
                        println!("  No new sessions to ingest.");
                    } else {
                        // Build entries directly from adapter output — each SourceFile
                        // is one session turn, already chunked by the adapter. Skipping
                        // the ChunkingPipeline preserves the per-turn source_version
                        // (e.g. "ses-abc:turn-1") which the pipeline would collapse to
                        // just the file_path (session ID).
                        let session_count = {
                            let mut seen = std::collections::HashSet::new();
                            source_files.iter().for_each(|sf| { seen.insert(&sf.metadata.file_path); });
                            seen.len()
                        };

                        println!(
                            "  {} sessions → {} turn entries",
                            session_count,
                            source_files.len()
                        );

                        let entries: Vec<corvia_common::types::KnowledgeEntry> =
                            source_files
                                .iter()
                                .map(|sf| {
                                    let mut entry =
                                        corvia_common::types::KnowledgeEntry::new(
                                            sf.content.clone(),
                                            USER_HISTORY_SCOPE.to_string(),
                                            sf.metadata.source_version.clone(),
                                        );
                                    entry.workstream = sf.metadata.workstream.clone()
                                        .unwrap_or_default();
                                    entry.metadata =
                                        corvia_common::types::EntryMetadata {
                                            source_file: Some(
                                                sf.metadata.file_path.clone(),
                                            ),
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

                        // Build a parallel vec of parent_session_id for edge creation.
                        // Only turn-1 entries carry this (set by the adapter).
                        let parent_ids: Vec<Option<String>> = source_files
                            .iter()
                            .map(|sf| sf.metadata.parent_session_id.clone())
                            .collect();

                        let total = entries.len();
                        let mut stored = 0;
                        let mut stored_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
                        for batch in
                            entries.chunks(corvia_kernel::introspect::EMBED_BATCH_SIZE)
                        {
                            let texts: Vec<String> =
                                batch.iter().map(|e| e.content.clone()).collect();
                            let embeddings = engine.embed_batch(&texts).await?;
                            for (entry, embedding) in batch.iter().zip(embeddings) {
                                let mut entry = entry.clone();
                                entry.embedding = Some(embedding);
                                store.insert(&entry).await?;
                                stored_ids.push(entry.id);
                                stored += 1;
                            }
                            println!("  embedded and stored {}/{}", stored, total);
                        }

                        // Create spawned_by graph edges for subagent sessions.
                        // For each entry with a parent_session_id, find the parent
                        // session's first-turn entry (by file_path == parent session ID)
                        // and create: subagent_turn1 --spawned_by--> parent_turn1.
                        // Create spawned_by graph edges for subagent sessions.
                        // Invariant: adapter emits exactly one entry per turn, so
                        // source_files[i] maps to stored_ids[i] and parent_ids[i].
                        debug_assert_eq!(
                            source_files.len(),
                            stored_ids.len(),
                            "1:1 source-to-entry invariant violated in session ingest"
                        );
                        {
                            // Build lookup: session_id → first stored entry UUID.
                            // Relies on adapter emitting turns in order (turn 1 first),
                            // so or_insert picks the turn-1 entry for each session.
                            let mut session_first_entry: std::collections::HashMap<String, uuid::Uuid> =
                                std::collections::HashMap::new();
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
                                            .relate(
                                                &child_entry_id,
                                                "spawned_by",
                                                &parent_entry_id,
                                                None,
                                            )
                                            .await
                                        {
                                            eprintln!(
                                                "  warning: failed to create spawned_by edge: {e}"
                                            );
                                        } else {
                                            edges_created += 1;
                                        }
                                    } else {
                                        // Parent was ingested in a prior run. Cross-batch
                                        // edge resolution requires a source_version index
                                        // (not yet implemented). Skip gracefully.
                                        // TODO: implement cross-batch parent lookup via
                                        // source_version index or metadata query.
                                        eprintln!(
                                            "  note: parent session {parent_sid} not in current batch — \
                                             spawned_by edge skipped"
                                        );
                                    }
                                }
                            }
                            if edges_created > 0 {
                                println!("  {edges_created} spawned_by graph edges created");
                            }
                        }

                        println!("  {} session turns stored", stored);
                    }

                    process.shutdown().map_err(|e| anyhow::anyhow!(e))?;
                }
            }
        }
    }

    println!("\nWorkspace ingest complete.");
    Ok(())
}

/// Clean build artifacts (target/ directories) from workspace repos.
///
/// Scans each repo for a Cargo `target/` directory and removes it, reporting
/// the space freed. This is a workspace-level GC complement to the kernel-level
/// session GC (`corvia_gc_run`).
pub fn clean_build_artifacts(root: &Path) -> Result<CleanReport> {
    let config_path = root.join("corvia.toml");
    let config = CorviaConfig::load(&config_path)?;
    let ws = config
        .workspace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace — no [workspace] section in corvia.toml"))?;

    let mut report = CleanReport::default();

    for repo in &ws.repos {
        let repo_path = resolve_repo_path(root, &ws.repos_dir, repo);
        let target_dir = repo_path.join("target");
        if !target_dir.exists() {
            continue;
        }

        let size = dir_size(&target_dir);
        println!(
            "  Cleaning {}/target/ ({})...",
            repo.name,
            human_bytes(size)
        );

        match std::fs::remove_dir_all(&target_dir) {
            Ok(()) => {
                report.dirs_cleaned += 1;
                report.bytes_freed += size;
            }
            Err(e) => {
                println!("    Warning: failed to remove {}: {}", target_dir.display(), e);
            }
        }
    }

    Ok(report)
}

/// Report from build-artifact cleanup.
#[derive(Debug, Default)]
pub struct CleanReport {
    pub dirs_cleaned: usize,
    pub bytes_freed: u64,
}

/// Recursively compute the size of a directory in bytes.
fn dir_size(path: &Path) -> u64 {
    walkdir(path)
}

fn walkdir(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() || ft.is_symlink() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if ft.is_dir() {
                total += walkdir(&entry.path());
            }
        }
    }
    total
}

pub fn human_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Display workspace status including config, repo state, and entry counts.
pub async fn workspace_status(root: &Path) -> Result<()> {
    let config_path = root.join("corvia.toml");
    let config = CorviaConfig::load(&config_path)?;
    let ws = config
        .workspace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Not a workspace"))?;

    println!(
        "Workspace: {} (scope: {})",
        config.project.name, config.project.scope_id
    );
    println!(
        "Store: {:?} at {}",
        config.storage.store_type, config.storage.data_dir
    );
    println!(
        "Embedding: {:?} ({})",
        config.embedding.provider, config.embedding.model
    );
    println!();

    // Try to open store for entry counts
    let data_dir = root.join(&config.storage.data_dir);
    let store = if data_dir.exists() {
        corvia_kernel::create_store_at(&config, &data_dir).await.ok()
    } else {
        None
    };

    println!("Repos ({}):", ws.repos.len());
    for repo in &ws.repos {
        let repo_path = resolve_repo_path(root, &ws.repos_dir, repo);
        let exists = repo_path.exists();
        let is_local = repo
            .local
            .as_ref()
            .map(|l| std::path::PathBuf::from(l).exists())
            .unwrap_or(false);

        let source = if is_local {
            "local"
        } else if exists {
            "cloned"
        } else {
            "missing"
        };

        println!("  {} [{}] namespace:{}", repo.name, source, repo.namespace);
        println!("    url: {}", repo.url);
        if let Some(local) = &repo.local {
            println!("    local: {}", local);
        }
        println!("    path: {}", repo_path.display());
    }

    if let Some(ref store) = store {
        let total = store.count(&config.project.scope_id).await.unwrap_or(0);
        println!("\nTotal knowledge entries: {}", total);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_name_from_url() {
        assert_eq!(
            repo_name_from_url("https://github.com/org/my-repo"),
            "my-repo"
        );
        assert_eq!(
            repo_name_from_url("https://github.com/org/my-repo.git"),
            "my-repo"
        );
        assert_eq!(
            repo_name_from_url("https://github.com/org/my-repo/"),
            "my-repo"
        );
        assert_eq!(
            repo_name_from_url("git@github.com:org/my-repo.git"),
            "my-repo"
        );
    }

    #[test]
    fn test_repo_name_from_url_empty_fallback() {
        assert_eq!(repo_name_from_url(""), "unknown");
        assert_eq!(repo_name_from_url(".git"), "unknown");
    }

    #[test]
    fn test_parse_local_overrides() {
        let overrides = vec![
            "backend=/home/dev/backend".into(),
            "frontend=/tmp/fe".into(),
        ];
        let parsed = parse_local_overrides(&overrides).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], ("backend".into(), "/home/dev/backend".into()));
        assert_eq!(parsed[1], ("frontend".into(), "/tmp/fe".into()));
    }

    #[test]
    fn test_parse_local_overrides_invalid() {
        let overrides = vec!["no-equals-sign".into()];
        assert!(parse_local_overrides(&overrides).is_err());
    }

    #[test]
    fn test_generate_workspace_config() {
        let config = generate_workspace_config(
            "my-project",
            &[
                "https://github.com/org/backend".into(),
                "https://github.com/org/frontend".into(),
            ],
            &[("frontend".into(), "/home/dev/frontend".into())],
        )
        .unwrap();
        assert!(config.is_workspace());
        let ws = config.workspace.as_ref().unwrap();
        assert_eq!(ws.repos.len(), 2);
        assert_eq!(ws.repos[0].name, "backend");
        assert!(ws.repos[0].local.is_none());
        assert_eq!(ws.repos[1].name, "frontend");
        assert_eq!(ws.repos[1].local.as_deref(), Some("/home/dev/frontend"));
    }

    #[test]
    fn test_scaffold_workspace_creates_structure() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("my-workspace");
        let config = generate_workspace_config(
            "test-ws",
            &["https://github.com/org/repo-a".into()],
            &[],
        )
        .unwrap();
        scaffold_workspace(&root, &config).unwrap();
        assert!(root.join("corvia.toml").exists());
        assert!(root.join("repos").exists());
        assert!(root.join("repos/.gitkeep").exists());
        assert!(root.join(".corvia").exists());
        assert!(root.join(".gitignore").exists());
        // Verify config roundtrips
        let loaded = CorviaConfig::load(&root.join("corvia.toml")).unwrap();
        assert!(loaded.is_workspace());
        assert_eq!(loaded.workspace.unwrap().repos.len(), 1);
    }

    #[test]
    fn test_resolve_repo_path_no_local() {
        let repo = RepoConfig {
            name: "backend".into(),
            url: "https://github.com/org/backend".into(),
            local: None,
            namespace: "backend".into(),
        };
        let path = resolve_repo_path(Path::new("/workspace"), "repos", &repo);
        assert_eq!(path, PathBuf::from("/workspace/repos/backend"));
    }

    #[test]
    fn test_resolve_repo_path_local_exists() {
        let dir = tempfile::tempdir().unwrap();
        let local_path = dir.path().join("my-backend");
        std::fs::create_dir_all(local_path.join(".git")).unwrap();
        let repo = RepoConfig {
            name: "backend".into(),
            url: "https://github.com/org/backend".into(),
            local: Some(local_path.to_str().unwrap().into()),
            namespace: "backend".into(),
        };
        let path = resolve_repo_path(Path::new("/workspace"), "repos", &repo);
        assert_eq!(path, local_path);
    }

    #[test]
    fn test_resolve_repo_path_local_missing_falls_back() {
        let repo = RepoConfig {
            name: "backend".into(),
            url: "https://github.com/org/backend".into(),
            local: Some("/nonexistent/path".into()),
            namespace: "backend".into(),
        };
        let path = resolve_repo_path(Path::new("/workspace"), "repos", &repo);
        assert_eq!(path, PathBuf::from("/workspace/repos/backend"));
    }

    #[test]
    fn test_add_repo_to_config() {
        let mut config =
            generate_workspace_config("test", &["https://github.com/org/a".into()], &[]).unwrap();
        add_repo_to_config(
            &mut config,
            "https://github.com/org/b",
            Some("b"),
            Some("b-ns"),
            None,
        )
        .unwrap();
        let ws = config.workspace.as_ref().unwrap();
        assert_eq!(ws.repos.len(), 2);
        assert_eq!(ws.repos[1].name, "b");
        assert_eq!(ws.repos[1].namespace, "b-ns");
    }

    #[test]
    fn test_add_repo_duplicate_fails() {
        let mut config =
            generate_workspace_config("test", &["https://github.com/org/a".into()], &[]).unwrap();
        let result = add_repo_to_config(
            &mut config,
            "https://github.com/org/other",
            Some("a"),
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_repo_from_config() {
        let mut config = generate_workspace_config(
            "test",
            &[
                "https://github.com/org/a".into(),
                "https://github.com/org/b".into(),
            ],
            &[],
        )
        .unwrap();
        remove_repo_from_config(&mut config, "a").unwrap();
        let ws = config.workspace.as_ref().unwrap();
        assert_eq!(ws.repos.len(), 1);
        assert_eq!(ws.repos[0].name, "b");
    }

    #[test]
    fn test_remove_repo_not_found() {
        let mut config =
            generate_workspace_config("test", &["https://github.com/org/a".into()], &[]).unwrap();
        let result = remove_repo_from_config(&mut config, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dir_name_rejects_traversal() {
        assert!(validate_dir_name("..").is_err());
        assert!(validate_dir_name(".").is_err());
        assert!(validate_dir_name("foo/bar").is_err());
        assert!(validate_dir_name("foo\\bar").is_err());
        assert!(validate_dir_name("").is_err());
        assert!(validate_dir_name("valid-name").is_ok());
        assert!(validate_dir_name("my_repo.v2").is_ok());
    }

    #[test]
    fn test_generate_rejects_bad_workspace_name() {
        let result = generate_workspace_config(
            "../escape",
            &["https://github.com/org/repo".into()],
            &[],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_add_repo_rejects_bad_name() {
        let mut config =
            generate_workspace_config("test", &["https://github.com/org/a".into()], &[]).unwrap();
        let result = add_repo_to_config(
            &mut config,
            "https://github.com/org/b",
            Some("../escape"),
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_content_role_code() {
        assert_eq!(infer_content_role("src/main.rs"), Some("code".into()));
        assert_eq!(infer_content_role("lib/utils.py"), Some("code".into()));
        assert_eq!(infer_content_role("src/app.tsx"), Some("code".into()));
        assert_eq!(infer_content_role("build.sh"), Some("code".into()));
    }

    #[test]
    fn test_infer_content_role_instruction() {
        assert_eq!(infer_content_role("AGENTS.md"), Some("instruction".into()));
        assert_eq!(infer_content_role("CLAUDE.md"), Some("instruction".into()));
        assert_eq!(infer_content_role("README.md"), Some("instruction".into()));
        assert_eq!(
            infer_content_role("crates/foo/README.md"),
            Some("instruction".into())
        );
    }

    #[test]
    fn test_infer_content_role_markdown_dirs() {
        assert_eq!(
            infer_content_role("docs/rfcs/design-spec.md"),
            Some("design".into())
        );
        assert_eq!(
            infer_content_role("docs/plans/impl-plan.md"),
            Some("plan".into())
        );
        assert_eq!(
            infer_content_role("docs/decisions/d01.md"),
            Some("decision".into())
        );
        assert_eq!(
            infer_content_role("docs/learnings/rag-modes.md"),
            Some("learning".into())
        );
        assert_eq!(
            infer_content_role("docs/some-other.md"),
            Some("design".into())
        );
    }

    #[test]
    fn test_infer_content_role_memory() {
        assert_eq!(
            infer_content_role(".memory/user_role.md"),
            Some("memory".into())
        );
    }

    #[test]
    fn test_infer_content_role_unclassified() {
        assert_eq!(infer_content_role("notes.txt"), None);
    }

    #[test]
    fn test_infer_source_origin_repo() {
        assert_eq!(
            infer_source_origin(Some("corvia"), "src/main.rs"),
            Some("repo:corvia".into())
        );
    }

    #[test]
    fn test_infer_source_origin_workspace() {
        assert_eq!(
            infer_source_origin(None, "decisions/d01.md"),
            Some("workspace".into())
        );
    }

    #[test]
    fn test_infer_source_origin_memory() {
        assert_eq!(
            infer_source_origin(Some("corvia"), ".memory/foo.md"),
            Some("memory".into())
        );
        assert_eq!(
            infer_source_origin(None, ".memory/foo.md"),
            Some("memory".into())
        );
    }

    #[test]
    fn test_blocked_path_match() {
        assert!(blocked_path_match("docs/superpowers/*", "docs/superpowers/brainstorm.md"));
        assert!(blocked_path_match("docs/superpowers/*", "docs/superpowers/plans/foo.md"));
        assert!(!blocked_path_match("docs/superpowers/*", "docs/decisions/d01.md"));
        assert!(blocked_path_match("exact.md", "exact.md"));
        assert!(!blocked_path_match("exact.md", "other.md"));
    }
}
