# Build vs Reuse: RAG Components

**Date**: 2026-04-14
**Key Question**: Is there a single package that provides vector + BM25 + reranking?

## Answer: LanceDB Comes Closest

**LanceDB** (9.9K stars, Apache 2.0, Rust core) provides:
- Embedded vector search (HNSW, billion-scale)
- Full-text BM25 (via integrated Tantivy)
- Built-in rerankers (RRF, CrossEncoder, Cohere, ColBERT)
- Zero-server: `lancedb.connect("data/my-db")`
- Hybrid search in one call
- Rust, Python, TypeScript SDKs

This is the only embedded store that bundles all three (vector + BM25 + reranking).

## Component-Level Options

### Embedded Vector Stores

| Store | Stars | License | Language | Hybrid? | Reranking? |
|-------|-------|---------|----------|---------|-----------|
| FAISS | 39.7K | MIT | C++/Python | No | No |
| Qdrant | 30.3K | Apache 2.0 | Rust | Yes | Yes (MMR) |
| ChromaDB | 27.4K | Apache 2.0 | Rust/Python | Yes (2025) | No |
| **LanceDB** | **9.9K** | **Apache 2.0** | **Rust** | **Yes** | **Yes** |
| sqlite-vec | 7.4K | MIT | C | No | No |

### BM25 / Full-Text

| Library | Stars | License | Language | Performance |
|---------|-------|---------|----------|-------------|
| **tantivy** | **15K** | **MIT** | **Rust** | 0.8ms avg, 2x faster than Lucene |
| SQLite FTS5 | N/A | Public domain | C | Simple but limited |
| Whoosh | ~300 | BSD | Python | ~50ms avg, unmaintained |

### Rerankers (Rust)

| Library | License | Models | Notes |
|---------|---------|--------|-------|
| **fastembed-rs** | Apache 2.0 | bge-reranker, jina-reranker | ONNX-based, CPU |
| synaptic-flashrank | Apache 2.0 | BM25-based scoring | Lightweight |

### Embedding (Rust)

| Library | License | Notes |
|---------|---------|-------|
| **fastembed-rs** | Apache 2.0 | 30+ models, ONNX, proven in v1 |
| ort (direct) | MIT | Lower-level, more control |
| candle | Apache 2.0 | 9-10x slower than ONNX on CPU |

## Three Viable Stacks

### Option A: LanceDB (Maximum Reuse)

```
corvia-cli
  └── lancedb (Rust crate)
        ├── vector search (HNSW)
        ├── BM25 (tantivy internal)
        └── reranking (built-in)
  └── fastembed-rs (embedding)
  └── rmcp (MCP server)
```

**Pros**: Least code to write. One dependency handles search + BM25 + reranking.
**Cons**: Less control over pipeline. LanceDB's Rust API is less mature than Python.
May not support corvia's graph expansion or per-memory-type channels natively.

### Option B: Individual Components (Maximum Control)

```
corvia-cli
  └── tantivy (BM25)
  └── hnsw_rs or qdrant-embedded (vector)
  └── fastembed-rs (embedding + reranking)
  └── petgraph (knowledge graph)
  └── rmcp (MCP server)
```

**Pros**: Full control. Can implement per-memory-type routing, graph expansion,
custom fusion. Matches v1's proven architecture.
**Cons**: More code to wire together. More dependencies to maintain.

### Option C: Qdrant Embedded (Middle Ground)

```
corvia-cli
  └── qdrant-client (embedded mode)
        ├── vector search (HNSW)
        └── sparse vectors (BM25-like)
  └── fastembed-rs (embedding + reranking)
  └── petgraph (knowledge graph)
  └── rmcp (MCP server)
```

**Pros**: Qdrant is the most mature Rust vector DB (30.3K stars). Hybrid search
via sparse vectors. Well-tested at scale.
**Cons**: Embedded mode is less documented than server mode. Adds significant
binary size. May be overkill for local-first single-user tool.

## Recommendation

**Option B (Individual Components)** for these reasons:

1. V1 already proved this stack works (tantivy + hnsw_rs + fastembed-rs + petgraph)
2. Full control over pipeline composition (per-memory-type routing, graph expansion)
3. Smallest binary size and dependency footprint
4. Each component is individually mature and well-maintained
5. No risk of LanceDB/Qdrant API changes breaking the integration

**What changes from v1**:
- Add cross-encoder reranking via fastembed-rs (new)
- Simplify storage to single embedded DB (drop PostgreSQL option)
- Embed inference in main binary (drop gRPC server)
- Drop multi-channel search complexity (start with single hybrid searcher)

**What to watch**:
- LanceDB's Rust API maturity. If it stabilizes, Option A becomes more attractive
  for reducing maintenance burden.
- Qdrant's embedded mode. If it becomes well-documented, Option C is a strong
  middle ground.

## Turnkey RAG Systems (Not Recommended)

| System | Stars | Why Not |
|--------|-------|---------|
| RAGFlow | 78K | Requires Docker, Python, too heavy |
| txtai | 12.4K | Python-only, can't embed in Rust binary |
| Haystack | 17K | Python-only, framework overhead |
| LightRAG | 33.2K | Python-only, graph focus |

These are Python frameworks designed for different deployment models (cloud,
Docker). They don't fit the single-binary local-first constraint.
