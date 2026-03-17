use anyhow::Result;
use corvia_common::config::{CorviaConfig, StoreType};
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry};
use corvia_kernel::knowledge_files;
use corvia_kernel::traits::{GraphStore, QueryableStore};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Load all entries and edges from the current (source) store.
async fn load_source(
    config: &CorviaConfig,
) -> Result<(Vec<KnowledgeEntry>, Vec<GraphEdge>)> {
    let data_dir = Path::new(&config.storage.data_dir);

    match config.storage.store_type {
        StoreType::Lite => {
            let lite = corvia_kernel::lite_store::LiteStore::open(
                data_dir,
                config.embedding.dimensions,
            )?;
            let entries = lite.fetch_all_entries()?;
            let all_edges = lite.graph().fetch_all_edges()?;

            Ok((entries, all_edges))
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            let pg = connect_postgres(config).await?;
            let entries = pg.fetch_all_entries().await?;
            let edges = pg.fetch_all_edges().await?;
            Ok((entries, edges))
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            anyhow::bail!(
                "Cannot read from PostgresStore: compiled without --features postgres"
            );
        }
    }
}

/// Connect to PostgreSQL using config defaults.
#[cfg(feature = "postgres")]
async fn connect_postgres(
    config: &CorviaConfig,
) -> Result<corvia_kernel::postgres_store::PostgresStore> {
    let url = config
        .storage
        .postgres_url
        .as_deref()
        .unwrap_or("postgres://corvia:corvia@127.0.0.1:5432/corvia");
    Ok(corvia_kernel::postgres_store::PostgresStore::connect(url, config.embedding.dimensions).await?)
}

/// Parse a target store name string into a StoreType.
fn parse_target(to: &str) -> Result<StoreType> {
    match to.to_lowercase().as_str() {
        "lite" => Ok(StoreType::Lite),
        "postgres" | "postgresql" | "pg" => Ok(StoreType::Postgres),
        other => anyhow::bail!(
            "Unknown store type '{other}'. Valid targets: lite, postgres"
        ),
    }
}

fn store_type_name(st: &StoreType) -> &'static str {
    match st {
        StoreType::Lite => "lite",
        StoreType::Postgres => "postgres",
    }
}

/// Migrate data between storage backends.
///
/// Reads all entries + edges from the current store, writes them to the target
/// store, verifies counts, and updates corvia.toml.
pub async fn cmd_migrate(to: &str, dry_run: bool) -> Result<()> {
    let config_path = CorviaConfig::config_path();
    if !config_path.exists() {
        anyhow::bail!("No corvia.toml found. Run 'corvia init' first.");
    }
    let config = CorviaConfig::load(&config_path)?;

    let target = parse_target(to)?;

    // Validate source != destination
    if config.storage.store_type == target {
        anyhow::bail!(
            "Already using {}. Nothing to migrate.",
            store_type_name(&target)
        );
    }

    // Feature-gate check for postgres target
    #[cfg(not(feature = "postgres"))]
    if target == StoreType::Postgres {
        anyhow::bail!(
            "PostgresStore requires compiling with --features postgres.\n\
             Rebuild with: cargo build --features postgres"
        );
    }

    let source_name = store_type_name(&config.storage.store_type);
    let target_name = store_type_name(&target);

    println!("Corvia migrate: {} -> {}", source_name, target_name);
    if dry_run {
        println!("  (dry run — no changes will be made)\n");
    } else {
        println!();
    }

    // Load all data from source
    println!("Reading entries from {} store...", source_name);
    let (entries, all_edges) = load_source(&config).await?;

    if entries.is_empty() {
        println!("No knowledge entries found. Nothing to migrate.");
        return Ok(());
    }

    // Group entries by scope for reporting and verification
    let mut entries_by_scope: HashMap<String, Vec<&KnowledgeEntry>> = HashMap::new();
    for entry in &entries {
        entries_by_scope
            .entry(entry.scope_id.clone())
            .or_default()
            .push(entry);
    }

    let mut scope_names: Vec<_> = entries_by_scope.keys().cloned().collect();
    scope_names.sort();
    println!(
        "Found {} entries across {} scope(s):",
        entries.len(),
        entries_by_scope.len()
    );
    for scope in &scope_names {
        println!("  {}: {} entries", scope, entries_by_scope[scope].len());
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

    // Write to destination
    let data_dir = Path::new(&config.storage.data_dir);
    match target {
        StoreType::Lite => {
            write_to_lite(&config, data_dir, &entries, &all_edges, &entries_by_scope, &scope_names).await?;
        }
        #[cfg(feature = "postgres")]
        StoreType::Postgres => {
            write_to_postgres(&config, &entries, &all_edges, &entries_by_scope, &scope_names).await?;
        }
        #[cfg(not(feature = "postgres"))]
        StoreType::Postgres => {
            unreachable!("Postgres feature gate checked above");
        }
    }

    // Update config
    println!("\nUpdating corvia.toml...");
    let mut updated_config = config.clone();
    updated_config.storage.store_type = target.clone();

    match target {
        StoreType::Postgres => {
            if updated_config.storage.postgres_url.is_none() {
                updated_config.storage.postgres_url =
                    Some("postgres://corvia:corvia@127.0.0.1:5432/corvia".to_string());
            }
        }
        StoreType::Lite => {
            // No connection details needed for Lite
        }
    }

    updated_config.save(&config_path)?;
    println!("  store_type changed to '{}'", target_name);

    println!("\nMigration complete!");
    println!("  Entries migrated: {}", entries.len());
    println!("  Edges migrated: {}", all_edges.len());
    println!("  Scopes: {}", scope_names.join(", "));

    if config.storage.store_type == StoreType::Lite {
        println!(
            "\nNote: LiteStore files in {}/ are preserved as backup.",
            data_dir.display()
        );
        println!("You can remove them manually after confirming the migration.");
    }

    Ok(())
}

/// Write entries and edges to a LiteStore destination.
async fn write_to_lite(
    config: &CorviaConfig,
    data_dir: &Path,
    entries: &[KnowledgeEntry],
    edges: &[GraphEdge],
    entries_by_scope: &HashMap<String, Vec<&KnowledgeEntry>>,
    scope_names: &[String],
) -> Result<()> {
    // Step 1: Write JSON files for each entry
    println!("\nWriting {} knowledge files to {}...", entries.len(), data_dir.display());
    for entry in entries {
        knowledge_files::write_entry(data_dir, entry)?;
    }
    println!("  {} JSON files written.", entries.len());

    // Step 2: Open LiteStore (rebuilds HNSW + Redb indexes from files)
    println!("Opening LiteStore and rebuilding indexes...");
    let lite = corvia_kernel::lite_store::LiteStore::open(data_dir, config.embedding.dimensions)?;
    let rebuilt = lite.rebuild_from_files()?;
    println!("  Rebuilt index with {} entries.", rebuilt);

    // Step 3: Recreate graph edges
    if !edges.is_empty() {
        println!("Migrating {} graph edges...", edges.len());
        let mut edges_inserted = 0;
        let mut edges_failed = 0;
        for edge in edges {
            match lite.relate(&edge.from, &edge.relation, &edge.to, edge.metadata.clone()).await {
                Ok(()) => edges_inserted += 1,
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
            edges.len(),
            edges_failed
        );
        if edges_failed > 0 {
            eprintln!("\nWarning: {} edges failed to migrate. Config NOT updated.", edges_failed);
            eprintln!("Note: partial data was written to the destination. Re-running the migration");
            eprintln!("after resolving the issue may require clearing the destination first.");
            anyhow::bail!("{} edges failed to migrate", edges_failed);
        }
    }

    // Step 4: Verify counts
    let lite = Arc::new(lite);
    verify_counts(
        &(lite.clone() as Arc<dyn QueryableStore>),
        &(lite as Arc<dyn GraphStore>),
        entries_by_scope,
        scope_names,
        edges.len(),
    ).await?;

    Ok(())
}

/// Write entries and edges to a PostgreSQL destination.
#[cfg(feature = "postgres")]
async fn write_to_postgres(
    config: &CorviaConfig,
    entries: &[KnowledgeEntry],
    edges: &[GraphEdge],
    entries_by_scope: &HashMap<String, Vec<&KnowledgeEntry>>,
    scope_names: &[String],
) -> Result<()> {
    let pg_url = config
        .storage
        .postgres_url
        .as_deref()
        .unwrap_or("postgres://corvia:corvia@127.0.0.1:5432/corvia");

    println!("\nConnecting to PostgreSQL at {}...", pg_url);
    let store = connect_postgres(config).await?;
    println!("Initializing PostgreSQL schema...");
    store.init_schema().await?;

    let store = Arc::new(store);
    bulk_insert_and_verify(
        store.clone() as Arc<dyn QueryableStore>,
        store as Arc<dyn GraphStore>,
        entries,
        edges,
        entries_by_scope,
        scope_names,
    )
    .await
}

/// Shared logic: bulk insert entries + edges, then verify counts.
async fn bulk_insert_and_verify(
    queryable: Arc<dyn QueryableStore>,
    graph: Arc<dyn GraphStore>,
    entries: &[KnowledgeEntry],
    edges: &[GraphEdge],
    entries_by_scope: &HashMap<String, Vec<&KnowledgeEntry>>,
    scope_names: &[String],
) -> Result<()> {
    let expected_edge_count = edges.len();
    // Insert entries
    println!("Migrating {} entries...", entries.len());
    let mut inserted = 0;
    let mut failed = 0;
    for entry in entries {
        match queryable.insert(entry).await {
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
    println!(
        "  {}/{} entries inserted ({} failed)",
        inserted,
        entries.len(),
        failed
    );

    if failed > 0 {
        eprintln!("\nWarning: {} entries failed to insert. Config NOT updated.", failed);
        eprintln!("Note: {} entries were written to the destination. Re-running the migration", inserted);
        eprintln!("after resolving the issue may require clearing the destination first.");
        anyhow::bail!("{} entries failed to insert", failed);
    }

    // Insert edges
    if !edges.is_empty() {
        println!("\nMigrating {} graph edges...", edges.len());
        let mut edges_inserted = 0;
        let mut edges_failed = 0;
        for edge in edges {
            match graph
                .relate(&edge.from, &edge.relation, &edge.to, edge.metadata.clone())
                .await
            {
                Ok(()) => edges_inserted += 1,
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
            edges.len(),
            edges_failed
        );

        if edges_failed > 0 {
            eprintln!(
                "\nWarning: {} edges failed to migrate. Config NOT updated.",
                edges_failed
            );
            anyhow::bail!("{} edges failed to migrate", edges_failed);
        }
    }

    // Verify counts
    verify_counts(&queryable, &graph, entries_by_scope, scope_names, expected_edge_count).await?;

    Ok(())
}

/// Verify that per-scope entry counts and edge counts in the destination match the source.
async fn verify_counts(
    store: &Arc<dyn QueryableStore>,
    graph: &Arc<dyn GraphStore>,
    entries_by_scope: &HashMap<String, Vec<&KnowledgeEntry>>,
    scope_names: &[String],
    expected_edge_count: usize,
) -> Result<()> {
    println!("\nVerifying migration...");
    let mut verification_ok = true;
    for scope in scope_names {
        let expected = entries_by_scope[scope].len() as u64;
        let actual = store.count(scope).await?;
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

    // Verify edge count by querying outgoing edges from all entries
    if expected_edge_count > 0 {
        let mut actual_edge_count = 0;
        let all_entry_ids: Vec<_> = entries_by_scope
            .values()
            .flat_map(|v| v.iter().map(|e| e.id))
            .collect();
        for id in &all_entry_ids {
            let outgoing = graph.edges(id, EdgeDirection::Outgoing).await?;
            actual_edge_count += outgoing.len();
        }
        if actual_edge_count == expected_edge_count {
            println!("  edges: {} (OK)", actual_edge_count);
        } else {
            eprintln!(
                "  edges: expected {}, found {} (MISMATCH)",
                expected_edge_count, actual_edge_count
            );
            verification_ok = false;
        }
    }

    if !verification_ok {
        eprintln!("\nVerification failed. Config NOT updated.");
        anyhow::bail!("Verification failed: count mismatch");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_target_valid() {
        assert_eq!(parse_target("lite").unwrap(), StoreType::Lite);
        assert_eq!(parse_target("postgres").unwrap(), StoreType::Postgres);
        assert_eq!(parse_target("postgresql").unwrap(), StoreType::Postgres);
        assert_eq!(parse_target("pg").unwrap(), StoreType::Postgres);
    }

    #[test]
    fn test_parse_target_case_insensitive() {
        assert_eq!(parse_target("Lite").unwrap(), StoreType::Lite);
        assert_eq!(parse_target("Postgres").unwrap(), StoreType::Postgres);
        assert_eq!(parse_target("PG").unwrap(), StoreType::Postgres);
    }

    #[test]
    fn test_parse_target_invalid() {
        assert!(parse_target("mysql").is_err());
        assert!(parse_target("").is_err());
        assert!(parse_target("redis").is_err());
        assert!(parse_target("surrealdb").is_err());
    }

    #[test]
    fn test_store_type_name() {
        assert_eq!(store_type_name(&StoreType::Lite), "lite");
        assert_eq!(store_type_name(&StoreType::Postgres), "postgres");
    }
}
