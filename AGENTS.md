# corvia

> Organizational memory for AI agents. Local-first, single binary, stdio MCP.

## Architecture

- **corvia-core**: Storage, retrieval pipeline (tantivy + hnsw_rs + fastembed + redb)
- **corvia-cli**: CLI (5 commands) + stdio MCP server (3 tools) via rmcp

## Build

```bash
cargo build                  # debug
cargo build --release        # release
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `corvia ingest [path]` | Ingest documents into knowledge store |
| `corvia search <query>` | Hybrid search (vector + BM25 + rerank) |
| `corvia write <content>` | Write a knowledge entry (auto-dedup) |
| `corvia status` | Show system status |
| `corvia mcp` | Start stdio MCP server |

## MCP Tools

| Tool | Description |
|------|-------------|
| `corvia_search` | Semantic + BM25 hybrid search with reranking |
| `corvia_write` | Write entry (auto-supersedes near-duplicates) |
| `corvia_status` | System status (entry counts, index health) |

## Key Design Decisions

1. Local-first, no server. Runs as stdio MCP subprocess.
2. Single binary distribution. No Python, no Docker.
3. Caller is the LLM. No corvia_ask (no built-in generation).
4. Auto-dedup on write. Near-duplicate content auto-supersedes.
5. 4 knowledge kinds: decision, learning, instruction, reference.
6. Flat files (.corvia/entries/*.md) as source of truth. Redb for indexes only.
7. Git provides versioning, provenance, and audit trail.
8. Cross-encoder reranking for improved accuracy over v1.
9. Brute-force cosine for <10K vectors. HNSW above.
10. No tiers, no graph, no RBAC, no agent tracking in v1.0.

## Retrieval Pipeline

```
query -> [BM25 (tantivy)] \
                            -> RRF fusion (k=30) -> rerank (cross-encoder) -> results
query -> [vector (cosine)] /
```

## Storage Model

```
.corvia/
  entries/          # git-tracked flat files (TOML frontmatter + markdown)
    <uuid>.md
  index/            # gitignored, rebuilt via `corvia ingest`
    store.redb      # vectors, chunk mappings, supersession state
    tantivy/        # BM25 full-text index
```

## Config Defaults

| Parameter | Value | Source |
|-----------|-------|--------|
| chunk max_tokens | 512 | FloTorch 2026 benchmark |
| chunk overlap | 64 (~12.5%) | NVIDIA 15% optimum |
| RRF k | 30 | Tuned for small corpora |
| dedup threshold | 0.85 | Cosine similarity |
| search limit | 5 | Default results |
| embedding model | nomic-embed-text-v1.5 | 62.4 MTEB, proven in v1 |
| reranker model | ms-marco-MiniLM-L6-v2 | Smallest, fastest |
