//! `corvia init` — setup + health check for .corvia/ directory.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::config::Config;

/// Schema version written to `.corvia/version`.
pub const STORE_SCHEMA_VERSION: &str = "1.0.0";

/// Result of running `corvia init`.
#[derive(Debug)]
pub struct InitResult {
    pub base_dir: PathBuf,
    pub created: bool,
    pub config_migrated: bool,
    pub version_updated: bool,
    pub actions: Vec<String>,
}

/// Options for `corvia init`.
pub struct InitOptions {
    pub yes: bool,
    pub base_dir: Option<PathBuf>,
    pub force: bool,
    pub model_path: Option<PathBuf>,
}

/// Run `corvia init`. Safe to call repeatedly (idempotent).
pub fn run_init(opts: &InitOptions) -> Result<InitResult> {
    let base_dir = opts
        .base_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().expect("cannot determine cwd"));
    let corvia_dir = base_dir.join(".corvia");
    let created = !corvia_dir.exists();

    let mut result = InitResult {
        base_dir: base_dir.clone(),
        created,
        config_migrated: false,
        version_updated: false,
        actions: Vec::new(),
    };

    if created {
        fs::create_dir_all(&corvia_dir)
            .context("failed to create .corvia/ directory")?;
        result.actions.push("created .corvia/".into());
    }

    // Acquire exclusive lock.
    let lock_path = corvia_dir.join(".lock");
    let lock_file = fs::File::create(&lock_path)
        .context("failed to create .corvia/.lock")?;
    {
        use fs2::FileExt;
        lock_file
            .lock_exclusive()
            .context("failed to acquire .corvia/.lock (another corvia init running?)")?;
    }

    ensure_config(&base_dir, &corvia_dir, &mut result)?;

    let config = Config::load(&corvia_dir.join("corvia.toml"))
        .context("failed to load .corvia/corvia.toml after ensure_config")?;

    ensure_version(&corvia_dir, opts, &mut result)?;
    ensure_internal_gitignore(&corvia_dir, &mut result)?;
    ensure_mcp_json(&base_dir, &mut result)?;
    ensure_claude_settings(&base_dir, &mut result)?;

    // Model download — best-effort, don't fail init if models can't download.
    ensure_models(&corvia_dir, &config, opts, &mut result);

    Ok(result)
}

fn ensure_config(base_dir: &Path, corvia_dir: &Path, result: &mut InitResult) -> Result<()> {
    let v2_config = corvia_dir.join("corvia.toml");

    if v2_config.is_file() {
        Config::load(&v2_config).context("existing .corvia/corvia.toml is invalid")?;
        return Ok(());
    }

    let v1_config = base_dir.join("corvia.toml");
    if v1_config.is_file() {
        fs::copy(&v1_config, &v2_config)
            .context("failed to copy v1 corvia.toml to .corvia/")?;
        let backup = base_dir.join("corvia.toml.v1-backup");
        fs::rename(&v1_config, &backup)
            .context("failed to rename v1 corvia.toml to .v1-backup")?;
        result.config_migrated = true;
        result.actions.push("migrated config from ./corvia.toml".into());
        return Ok(());
    }

    let defaults = Config::default();
    let toml_str = toml::to_string_pretty(&defaults)
        .context("failed to serialize default config")?;
    fs::write(&v2_config, toml_str)
        .context("failed to write default .corvia/corvia.toml")?;
    result.actions.push("created .corvia/corvia.toml (defaults)".into());
    Ok(())
}

fn ensure_version(corvia_dir: &Path, opts: &InitOptions, result: &mut InitResult) -> Result<()> {
    let version_path = corvia_dir.join("version");

    if let Ok(existing) = fs::read_to_string(&version_path) {
        let existing = existing.trim();
        if existing == STORE_SCHEMA_VERSION {
            return Ok(());
        }
        if !opts.force && existing > STORE_SCHEMA_VERSION {
            if opts.yes {
                eprintln!(
                    "warning: store schema v{} is newer than this binary's v{}. \
                     Some features may not work. Upgrade corvia or use --force.",
                    existing, STORE_SCHEMA_VERSION
                );
                return Ok(());
            } else {
                bail!(
                    "store schema v{} is newer than this binary's v{}. \
                     Upgrade corvia or use --force.",
                    existing, STORE_SCHEMA_VERSION
                );
            }
        }
    }

    fs::write(&version_path, STORE_SCHEMA_VERSION)
        .context("failed to write .corvia/version")?;
    result.version_updated = true;
    result.actions.push(format!("set store schema v{STORE_SCHEMA_VERSION}"));
    Ok(())
}

fn ensure_internal_gitignore(corvia_dir: &Path, result: &mut InitResult) -> Result<()> {
    let gitignore_path = corvia_dir.join(".gitignore");
    let expected = "\
# Derived data (rebuilt by corvia init / corvia ingest)
index/
models/
traces.jsonl
version
*.lock

# Source-of-truth files are NOT ignored:
# - corvia.toml (config)
# - entries/ (knowledge entries)
";

    if gitignore_path.is_file() {
        let existing = fs::read_to_string(&gitignore_path).unwrap_or_default();
        if existing.contains("index/") && existing.contains("models/") {
            return Ok(());
        }
    }

    fs::write(&gitignore_path, expected)
        .context("failed to write .corvia/.gitignore")?;
    result.actions.push("created .corvia/.gitignore".into());
    Ok(())
}

fn ensure_mcp_json(base_dir: &Path, result: &mut InitResult) -> Result<()> {
    let mcp_path = base_dir.join(".mcp.json");

    let expected_entry = serde_json::json!({
        "type": "stdio",
        "command": "corvia",
        "args": ["mcp"]
    });

    if mcp_path.is_file() {
        let content = fs::read_to_string(&mcp_path)
            .context("failed to read .mcp.json")?;
        let mut doc: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("warning: .mcp.json has syntax errors ({e}), not modifying");
                return Ok(());
            }
        };

        if let Some(servers) = doc.get("mcpServers") {
            if let Some(entry) = servers.get("corvia") {
                if entry == &expected_entry {
                    return Ok(());
                }
            }
        }

        let servers = doc
            .as_object_mut()
            .unwrap()
            .entry("mcpServers")
            .or_insert_with(|| serde_json::json!({}));
        servers
            .as_object_mut()
            .unwrap()
            .insert("corvia".into(), expected_entry);

        let output = serde_json::to_string_pretty(&doc)?;
        fs::write(&mcp_path, format!("{output}\n"))
            .context("failed to update .mcp.json")?;
        result.actions.push(".mcp.json updated (stdio)".into());
    } else {
        let doc = serde_json::json!({
            "mcpServers": {
                "corvia": expected_entry
            }
        });
        let output = serde_json::to_string_pretty(&doc)?;
        fs::write(&mcp_path, format!("{output}\n"))
            .context("failed to create .mcp.json")?;
        result.actions.push(".mcp.json created (stdio)".into());
    }

    Ok(())
}

fn ensure_claude_settings(base_dir: &Path, result: &mut InitResult) -> Result<()> {
    let claude_dir = base_dir.join(".claude");
    if !claude_dir.is_dir() {
        return Ok(());
    }

    let settings_path = claude_dir.join("settings.local.json");
    let mut doc: serde_json::Value = if settings_path.is_file() {
        let content = fs::read_to_string(&settings_path)
            .context("failed to read settings.local.json")?;
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let servers = doc
        .as_object_mut()
        .unwrap()
        .entry("enabledMcpjsonServers")
        .or_insert_with(|| serde_json::json!([]));
    if let Some(arr) = servers.as_array_mut() {
        let corvia_val = serde_json::Value::String("corvia".into());
        if !arr.contains(&corvia_val) {
            arr.push(corvia_val);
            let output = serde_json::to_string_pretty(&doc)?;
            fs::write(&settings_path, format!("{output}\n"))
                .context("failed to write settings.local.json")?;
            result.actions.push("settings.local.json updated".into());
        }
    }

    Ok(())
}

fn ensure_models(
    corvia_dir: &Path,
    config: &Config,
    opts: &InitOptions,
    result: &mut InitResult,
) {
    use crate::embed::Embedder;

    let model_dir = opts
        .model_path
        .clone()
        .unwrap_or_else(|| corvia_dir.join("models"));
    if let Err(e) = fs::create_dir_all(&model_dir) {
        eprintln!("warning: failed to create models directory: {e}");
        result.actions.push("models: directory creation failed".into());
        return;
    }

    println!("  models:     checking {}...", config.embedding.model);
    match Embedder::new(
        Some(&model_dir),
        &config.embedding.model,
        &config.embedding.reranker_model,
    ) {
        Ok(_) => {
            result.actions.push(format!("models ready ({})", config.embedding.model));
        }
        Err(e) => {
            eprintln!("warning: failed to load embedding model: {e:#}");
            result.actions.push("models: download failed (will retry)".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn opts(dir: &Path) -> InitOptions {
        InitOptions {
            yes: true,
            base_dir: Some(dir.to_path_buf()),
            force: false,
            model_path: None,
        }
    }

    #[test]
    fn fresh_init_creates_directory_and_config() {
        let dir = TempDir::new().unwrap();
        let result = run_init(&opts(dir.path())).unwrap();

        assert!(result.created);
        assert!(dir.path().join(".corvia/corvia.toml").is_file());
        assert!(dir.path().join(".corvia/version").is_file());
        assert!(dir.path().join(".corvia/.gitignore").is_file());

        let version = fs::read_to_string(dir.path().join(".corvia/version")).unwrap();
        assert_eq!(version.trim(), STORE_SCHEMA_VERSION);
    }

    #[test]
    fn idempotent_second_run() {
        let dir = TempDir::new().unwrap();
        run_init(&opts(dir.path())).unwrap();
        let result = run_init(&opts(dir.path())).unwrap();

        assert!(!result.created);
        assert!(!result.config_migrated);
        assert!(!result.version_updated);
    }

    #[test]
    fn migrates_v1_config() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("corvia.toml"),
            "[embedding]\nmodel = \"custom-model\"\nreranker_model = \"custom-reranker\"\n",
        )
        .unwrap();

        let result = run_init(&opts(dir.path())).unwrap();

        assert!(result.config_migrated);
        assert!(dir.path().join(".corvia/corvia.toml").is_file());
        assert!(dir.path().join("corvia.toml.v1-backup").is_file());
        assert!(!dir.path().join("corvia.toml").exists());

        let config = Config::load(&dir.path().join(".corvia/corvia.toml")).unwrap();
        assert_eq!(config.embedding.model, "custom-model");
    }

    #[test]
    fn creates_mcp_json() {
        let dir = TempDir::new().unwrap();
        run_init(&opts(dir.path())).unwrap();

        let content = fs::read_to_string(dir.path().join(".mcp.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&content).unwrap();
        let entry = &doc["mcpServers"]["corvia"];
        assert_eq!(entry["type"], "stdio");
        assert_eq!(entry["command"], "corvia");
    }

    #[test]
    fn preserves_other_mcp_servers() {
        let dir = TempDir::new().unwrap();
        let existing = serde_json::json!({
            "mcpServers": {
                "other-server": {"type": "http", "url": "http://localhost:9999"}
            }
        });
        fs::write(
            dir.path().join(".mcp.json"),
            serde_json::to_string_pretty(&existing).unwrap(),
        )
        .unwrap();

        run_init(&opts(dir.path())).unwrap();

        let content = fs::read_to_string(dir.path().join(".mcp.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(doc["mcpServers"]["other-server"].is_object());
        assert!(doc["mcpServers"]["corvia"].is_object());
    }

    #[test]
    fn skips_mcp_update_when_already_correct() {
        let dir = TempDir::new().unwrap();
        run_init(&opts(dir.path())).unwrap();

        // Second run — .mcp.json should not be in actions.
        let result = run_init(&opts(dir.path())).unwrap();
        assert!(!result.actions.iter().any(|a| a.contains(".mcp.json")));
    }

    #[test]
    fn claude_settings_appends_not_overwrites() {
        let dir = TempDir::new().unwrap();
        let claude_dir = dir.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("settings.local.json"),
            r#"{"enabledMcpjsonServers": ["other-server"]}"#,
        )
        .unwrap();

        run_init(&opts(dir.path())).unwrap();

        let content = fs::read_to_string(claude_dir.join("settings.local.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&content).unwrap();
        let servers = doc["enabledMcpjsonServers"].as_array().unwrap();
        assert!(servers.contains(&serde_json::json!("other-server")));
        assert!(servers.contains(&serde_json::json!("corvia")));
    }

    #[test]
    fn skips_claude_settings_without_claude_dir() {
        let dir = TempDir::new().unwrap();
        run_init(&opts(dir.path())).unwrap();
        assert!(!dir.path().join(".claude").exists());
    }

    #[test]
    fn version_downgrade_warns_in_yes_mode() {
        let dir = TempDir::new().unwrap();
        let corvia = dir.path().join(".corvia");
        fs::create_dir_all(&corvia).unwrap();
        fs::write(corvia.join("corvia.toml"), "").unwrap();
        fs::write(corvia.join("version"), "99.0.0").unwrap();

        let result = run_init(&opts(dir.path()));
        assert!(result.is_ok());
    }
}
