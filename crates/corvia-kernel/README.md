# corvia-kernel

Knowledge storage, agent coordination, graph reasoning, and temporal queries for the Corvia organizational memory system.

## Overview

`corvia-kernel` is the primary library crate in the Corvia workspace. It implements
the storage backends, agent coordination layer, reasoning engine, and all core traits
that adapters and servers depend on.

The kernel follows a two-tier storage design: **LiteStore** (zero-Docker default using
hnsw_rs, petgraph, and Redb) and **SurrealStore** (opt-in SurrealDB backend). Both
implement the same trait interface, so callers never need to know which tier is active.

## Core Traits

| Trait | Purpose |
|-------|---------|
| `QueryableStore` | Insert, semantic search, get, count, delete |
| `InferenceEngine` | Text → embedding vector generation |
| `TemporalStore` | Point-in-time snapshots and evolution queries |
| `GraphStore` | Directed edges, BFS/DFS traversal, shortest path |
| `IngestionAdapter` | Domain-specific source → knowledge entries |

## Subsystems

- **Storage** — `lite_store`, `knowledge_store` (SurrealDB), `knowledge_files` (Git JSON)
- **Embeddings** — `ollama_engine`, `embedding_pipeline` (vLLM)
- **Agent Coordination** — `agent_coordinator`, `session_manager`, `staging`, `merge_worker`
- **Reasoning** — `reasoner` (5 deterministic + 2 LLM-powered health checks)
- **Graph** — `graph_store` (petgraph-based overlay for LiteStore)
- **Introspection** — `introspect` (self-validation pipeline)

## Usage

```rust,no_run
use corvia_common::config::CorviaConfig;
use corvia_kernel::{create_engine, create_store};

async fn example() -> corvia_common::errors::Result<()> {
    let config = CorviaConfig::default(); // LiteStore + Ollama
    let engine = create_engine(&config);
    let store = create_store(&config).await?;
    store.init_schema().await?;
    Ok(())
}
```

## Architecture

See the root [ARCHITECTURE.md](../../ARCHITECTURE.md) for how `corvia-kernel` fits
into the layered workspace design.

## License

AGPL-3.0-only — see [LICENSE](../../LICENSE) for details.
