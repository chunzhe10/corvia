use anyhow::{Context, Result};
use corvia_adapter_git::GitAdapter;
use corvia_common::config::{CorviaConfig, InferenceProvider, RepoConfig, WorkspaceConfig};
use corvia_kernel::traits::IngestionAdapter;
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
    git2::Repository::clone(url, target_dir)
        .with_context(|| format!("Failed to clone '{}' into '{}'", url, target_dir.display()))?;

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

/// Ingest all (or one) workspace repos with namespace isolation.
///
/// For each repo in the workspace, runs tree-sitter parsing via `GitAdapter`,
/// sets `scope_id` and `workstream` on each entry for namespace isolation,
/// batch-embeds the content, stores the entries, and wires structural
/// relations into the graph store.
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

        println!(
            "\nIngesting {} (namespace: {})...",
            repo_config.name, repo_config.namespace
        );

        let adapter = GitAdapter::new();
        let repo_path_str = repo_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path for repo {}", repo_config.name))?;

        // D69 pipeline flow: source files → ChunkingPipeline → embed → store
        let source_files = adapter.ingest_sources(repo_path_str).await?;

        let mut pipeline = corvia_kernel::create_chunking_pipeline(&config);
        adapter.register_chunking(pipeline.registry_mut());

        let (processed, report) = pipeline.process_batch(&source_files)?;
        println!(
            "  {} files → {} chunks ({} merged, {} split)",
            report.files_processed, report.total_chunks,
            report.chunks_merged, report.chunks_split
        );

        // Convert ProcessedChunks to KnowledgeEntries, embed, and store
        let entries: Vec<corvia_common::types::KnowledgeEntry> = processed
            .iter()
            .map(|pc| {
                let mut entry = corvia_common::types::KnowledgeEntry::new(
                    pc.content.clone(),
                    config.project.scope_id.clone(),
                    pc.metadata.source_file.clone(),
                );
                entry.workstream = repo_config.namespace.clone();
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
            println!("  embedded and stored {}/{}", stored, total);
        }

        // Wire relations via old path (temporary — relation extraction will be
        // integrated into ChunkingStrategy in a future milestone)
        let relation_result = adapter.ingest_with_relations(repo_path_str).await?;
        if !relation_result.relations.is_empty() {
            let relations_stored = crate::wire_relations(
                &relation_result, &stored_ids, &*graph,
            ).await;
            if relations_stored > 0 {
                println!("  {relations_stored} graph relations stored");
            }
        }

        println!(
            "  {} chunks stored for namespace '{}'",
            stored, repo_config.namespace
        );
    }

    println!("\nWorkspace ingest complete.");
    Ok(())
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
}
