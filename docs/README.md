# Corvia V2 Design Documents

Research and design documents for the corvia v2 rebuild (versioned as v1.0.0).

## Documents

| # | Document | Status | Summary |
|---|----------|--------|---------|
| 00 | [Pivot Evaluation](00-pivot-evaluation.md) | Complete | V1 lessons, competitive landscape, key decisions |
| 01 | [Agentic Pipeline Research](01-agentic-pipeline-research.md) | Complete | Framework landscape, guardrail patterns, decided: corvia is infra not agent |
| 02 | [Retrieval Learnings from V1](02-retrieval-learnings-from-v1.md) | Complete | What worked, what didn't, config defaults to keep |
| 03 | [Rust vs Python](03-rust-vs-python.md) | Complete | Hard numbers, decision: Rust (already know it, cold start, single binary) |
| 04 | [Embedding Inference](04-embedding-inference.md) | Complete | fastembed-rs + ort, nomic-embed-text-v1.5, cross-platform GPU (DirectML/CoreML free) |
| 05 | [Retrieval Accuracy](05-retrieval-accuracy.md) | Complete | Ranked techniques, reranking is top priority addition |
| 06 | [Build vs Reuse](06-build-vs-reuse-components.md) | Complete | Individual components (tantivy + hnsw_rs + fastembed-rs) |
| 07 | [Open Questions Design](07-open-questions-design.md) | Complete | All decisions finalized after 5-persona review |

## V1.0.0 Scope (Final)

- **3 MCP tools**: corvia_search, corvia_write, corvia_status
- **5 CLI commands**: ingest, search, write, status, mcp
- **2 crates**: corvia-core, corvia-cli
- **Stack**: tantivy (BM25) + hnsw_rs (vector, >10K only) + fastembed (embedding + reranking) + rmcp (MCP) + redb (indexes)
- **Single binary**, stdio MCP, no server, no dashboard
- **Flat files** (.corvia/entries/*.md) + Redb indexes (.corvia/index/)
- **Auto-dedup**: near-duplicate writes auto-supersede existing entries

## Key Decisions

1. Corvia is knowledge infrastructure, not an agent framework
2. Fresh start. No migration from v1. Version 1.0.0.
3. No corvia_ask (caller is the LLM)
4. No graph, no tiers, no RBAC, no agent tracking
5. Rust (cold start, single binary, already known)
6. fastembed + nomic-embed-text-v1.5 (proven in v1)
7. Individual components over turnkey framework
8. Cross-encoder reranking as top new addition
9. Flat files (TOML frontmatter + markdown) for entries, git-synced
10. Redb for indexes only (vectors, chunks, supersession state)
11. Auto-dedup on write with 0.85 cosine threshold
12. Single `kind` field with 4 values: decision, learning, instruction, reference
13. UUIDv7 for entry IDs (time-ordered, zero collision)
14. Brute-force cosine for <10K vectors, HNSW above
15. Deletion = delete file + ingest
