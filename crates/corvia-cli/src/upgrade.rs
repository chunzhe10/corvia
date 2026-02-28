use anyhow::Result;
use corvia_common::config::{CorviaConfig, StoreType};
use corvia_common::types::EdgeDirection;
use corvia_kernel::knowledge_files;
use corvia_kernel::knowledge_store::SurrealStore;
use corvia_kernel::traits::{GraphStore, QueryableStore};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Migrate from LiteStore to SurrealDB (one-way upgrade).
///
/// Steps:
/// 1. Validate current store is LiteStore
/// 2. Read all knowledge entries from JSON files
/// 3. Connect to SurrealDB and initialize schema
/// 4. Bulk insert all entries
/// 5. Export graph edges from LiteStore and recreate in SurrealDB
/// 6. Verify entry and edge counts match
/// 7. Update corvia.toml to store_type = "surrealdb"
pub async fn cmd_upgrade(dry_run: bool) -> Result<()> {
    let config_path = CorviaConfig::config_path();
    if !config_path.exists() {
        anyhow::bail!("No corvia.toml found. Run 'corvia init' first.");
    }
    let config = CorviaConfig::load(&config_path)?;

    // Step 1: Validate current store is LiteStore
    if config.storage.store_type != StoreType::Lite {
        anyhow::bail!("Store is already SurrealDB. Nothing to upgrade.");
    }

    let data_dir = Path::new(&config.storage.data_dir);

    println!("Corvia upgrade: LiteStore -> SurrealDB");
    if dry_run {
        println!("  (dry run — no changes will be made)\n");
    } else {
        println!();
    }

    // Step 2: Read all knowledge entries from JSON files
    println!("Reading knowledge files from {}...", data_dir.display());
    let entries = knowledge_files::read_all(data_dir)?;

    if entries.is_empty() {
        println!("No knowledge entries found. Nothing to migrate.");
        return Ok(());
    }

    // Group entries by scope for reporting and verification
    let mut entries_by_scope: HashMap<String, Vec<&corvia_common::types::KnowledgeEntry>> =
        HashMap::new();
    for entry in &entries {
        entries_by_scope
            .entry(entry.scope_id.clone())
            .or_default()
            .push(entry);
    }

    let mut scope_names: Vec<_> = entries_by_scope.keys().cloned().collect();
    scope_names.sort();
    println!("Found {} entries across {} scope(s):", entries.len(), entries_by_scope.len());
    for scope in &scope_names {
        println!("  {}: {} entries", scope, entries_by_scope[scope].len());
    }

    // Step 3: Read graph edges from LiteStore
    println!("\nReading graph edges from LiteStore...");
    let (_, lite_graph) = corvia_kernel::create_store_with_graph(&config).await?;

    let mut all_edges = Vec::new();
    for entry in &entries {
        let outgoing = lite_graph.edges(&entry.id, EdgeDirection::Outgoing).await?;
        all_edges.extend(outgoing);
    }
    println!("Found {} graph edges.", all_edges.len());

    if dry_run {
        println!("\nDry run summary:");
        println!("  Entries to migrate: {}", entries.len());
        println!("  Edges to migrate: {}", all_edges.len());
        println!("  Scopes: {}", scope_names.join(", "));
        println!("\nRun without --dry-run to perform the migration.");
        return Ok(());
    }

    // Step 4: Connect to SurrealDB
    let surreal_url = config.storage.surrealdb_url.as_deref().unwrap_or("127.0.0.1:8000");
    let surreal_ns = config.storage.surrealdb_ns.as_deref().unwrap_or("corvia");
    let surreal_db = config.storage.surrealdb_db.as_deref().unwrap_or("main");
    let surreal_user = config.storage.surrealdb_user.as_deref().unwrap_or("root");
    let surreal_pass = config.storage.surrealdb_pass.as_deref().unwrap_or("root");

    println!("\nConnecting to SurrealDB at {}...", surreal_url);
    let surreal_store = SurrealStore::connect(
        surreal_url,
        surreal_ns,
        surreal_db,
        surreal_user,
        surreal_pass,
        config.embedding.dimensions,
    )
    .await?;

    // Step 5: Initialize schema
    println!("Initializing SurrealDB schema...");
    surreal_store.init_schema().await?;

    // Wrap in Arc for GraphStore trait access
    let surreal_arc = Arc::new(surreal_store);

    // Step 6: Bulk insert entries
    println!("Migrating {} entries...", entries.len());
    let mut inserted = 0;
    let mut failed = 0;
    for entry in &entries {
        match surreal_arc.insert(entry).await {
            Ok(()) => {
                inserted += 1;
                if inserted % 100 == 0 {
                    println!("  {}/{} entries inserted", inserted, entries.len());
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  Failed to insert entry {}: {}", entry.id, e);
            }
        }
    }
    println!("  {}/{} entries inserted ({} failed)", inserted, entries.len(), failed);

    if failed > 0 {
        eprintln!("\nWarning: {} entries failed to insert. Config NOT updated.", failed);
        eprintln!("Fix the issues and re-run 'corvia upgrade'.");
        anyhow::bail!("{} entries failed to insert", failed);
    }

    // Step 7: Migrate graph edges
    let mut edges_inserted = 0;
    if !all_edges.is_empty() {
        println!("\nMigrating {} graph edges...", all_edges.len());
        let mut edges_failed = 0;
        for edge in &all_edges {
            match surreal_arc
                .relate(&edge.from, &edge.relation, &edge.to, edge.metadata.clone())
                .await
            {
                Ok(()) => {
                    edges_inserted += 1;
                }
                Err(e) => {
                    edges_failed += 1;
                    eprintln!(
                        "  Failed to create edge {} --[{}]--> {}: {}",
                        edge.from, edge.relation, edge.to, e
                    );
                }
            }
        }
        println!(
            "  {}/{} edges migrated ({} failed)",
            edges_inserted,
            all_edges.len(),
            edges_failed
        );

        if edges_failed > 0 {
            eprintln!("\nWarning: {} edges failed to migrate. Config NOT updated.", edges_failed);
            eprintln!("Fix the issues and re-run 'corvia upgrade'.");
            anyhow::bail!("{} edges failed to migrate", edges_failed);
        }
    }

    // Step 8: Verify counts
    println!("\nVerifying migration...");
    let mut verification_ok = true;
    for (scope, source_entries) in &entries_by_scope {
        let expected = source_entries.len() as u64;
        let actual = surreal_arc.count(scope).await?;
        if actual == expected {
            println!("  scope '{}': {} entries (OK)", scope, actual);
        } else {
            eprintln!(
                "  scope '{}': expected {} entries, found {} (MISMATCH)",
                scope, expected, actual
            );
            verification_ok = false;
        }
    }

    if !verification_ok {
        eprintln!("\nVerification failed. Config NOT updated.");
        eprintln!("Investigate mismatches and re-run 'corvia upgrade'.");
        anyhow::bail!("Verification failed: entry count mismatch");
    }

    // Step 9: Update config
    println!("\nUpdating corvia.toml...");
    let mut updated_config = config.clone();
    updated_config.storage.store_type = StoreType::Surrealdb;
    // Ensure SurrealDB connection details are saved (use defaults if not set)
    if updated_config.storage.surrealdb_url.is_none() {
        updated_config.storage.surrealdb_url = Some(surreal_url.to_string());
    }
    if updated_config.storage.surrealdb_ns.is_none() {
        updated_config.storage.surrealdb_ns = Some(surreal_ns.to_string());
    }
    if updated_config.storage.surrealdb_db.is_none() {
        updated_config.storage.surrealdb_db = Some(surreal_db.to_string());
    }
    if updated_config.storage.surrealdb_user.is_none() {
        updated_config.storage.surrealdb_user = Some(surreal_user.to_string());
    }
    if updated_config.storage.surrealdb_pass.is_none() {
        updated_config.storage.surrealdb_pass = Some(surreal_pass.to_string());
    }
    updated_config.save(&config_path)?;
    println!("  store_type changed to 'surrealdb'");

    // Step 10: Print summary
    println!("\nUpgrade complete!");
    println!("  Entries migrated: {}", inserted);
    println!("  Edges migrated: {}", edges_inserted);
    println!("  Scopes: {}", scope_names.join(", "));
    println!("  SurrealDB: {} (ns={}, db={})", surreal_url, surreal_ns, surreal_db);
    println!("\nNote: LiteStore files in {}/ are preserved as backup.", data_dir.display());
    println!("You can remove them manually after confirming the migration.");

    Ok(())
}
