# corvia-cli

Command-line interface for corvia. Provides workspace management, knowledge operations,
server control, and diagnostic tools.

## Commands

```bash
corvia init                    # Initialize a new .corvia/ directory
corvia ingest <path>           # Parse, chunk, embed, and store knowledge
corvia search <query>          # Semantic search across knowledge
corvia reason                  # Run health checks on a scope
corvia history <entry-id>      # Supersession chain for an entry
corvia graph <entry-id>        # Graph edges for an entry
corvia serve                   # Start REST + MCP server on :8020
corvia status --metrics        # Extended metrics (entries, agents, latency)
corvia workspace init          # Clone repos, set up workspace
corvia workspace ingest        # Index all workspace repos
corvia workspace status        # Workspace + service health
corvia workspace init-hooks    # Generate doc-placement hooks from config
```

## Build

```bash
cargo build -p corvia-cli --release
# With PostgreSQL support:
cargo build -p corvia-cli --release --features postgres
```

The binary is named `corvia` (not `corvia-cli`).

## License

[AGPL-3.0-only](../../../LICENSE)
