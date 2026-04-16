# Rust vs Python Decision

**Date**: 2026-04-14
**Decision**: TBD (see analysis below)

## The Question

Corvia v2 is a 3-tool MCP server (search, write, status) with semantic search +
BM25 + local embeddings. Is Rust's complexity justified, or would Python get to
market faster with adequate performance?

## Hard Numbers

### MCP Server Performance (50 concurrent users, 39.9M requests)

| Metric | Rust | Python | Ratio |
|--------|------|--------|-------|
| Throughput | 4,845 RPS | 259 RPS | 18.7x |
| Avg Latency | 5.09ms | 251.62ms | 49x |
| P95 Latency | 10.99ms | 342.41ms | 31x |
| RAM | 10.9 MB | 258.6 MB | 23.7x |

### Cold Start (Critical for stdio MCP)

| Approach | Time |
|----------|------|
| Rust binary | <10ms |
| Python + onnxruntime | 1-2 seconds |
| Python + sentence-transformers | 15-45 seconds |

Every time Claude/Cursor spawns the MCP subprocess, it pays the cold start.
10ms vs 2 seconds is noticeable. 10ms vs 45 seconds is disqualifying.

### Search Component Performance

| Component | Rust | Python | Ratio |
|-----------|------|--------|-------|
| BM25 query (tantivy vs Whoosh) | 0.8ms | 50ms | 60x |
| Embedding (ort vs onnxruntime) | 15-40ms | 50-120ms | 3-5x |
| Vector search (HNSW) | 1-6ms | 3-5ms (ChromaDB) | Similar |

### Lines of Code

| Component | Python | Rust |
|-----------|--------|------|
| MCP shell | ~20 LOC | ~60 LOC |
| Embedding | ~30 LOC | ~80 LOC |
| BM25 | ~15 LOC | ~60 LOC |
| Total | ~150 LOC | ~450 LOC |

## MCP SDK Maturity

| SDK | Stars | Status |
|-----|-------|--------|
| Python (mcp) | 23,000 | Stable, most tutorials |
| Rust (rmcp) | 3,300 | Pre-1.0, active development |

## Distribution

| | Python | Rust |
|---|---|---|
| Install method | pip install (requires Python) | Download single binary |
| Binary size | 100-500MB (PyInstaller) | 5-15MB |
| Dependencies | Virtual env + ML stack | None |
| Cross-platform | Wheels per platform | Cross-compile with cargo-zigbuild |

## The Honest Assessment

### Python wins if:
- You need to ship in 1-2 weeks (vs 4-8 weeks Rust)
- You want the largest MCP SDK community
- Performance is "good enough" (it is for single-user local tool)
- You plan to use LanceDB/ChromaDB/txtai (Python-first libraries)

### Rust wins if:
- Cold start matters (<10ms vs 2 seconds, felt every session)
- RAM matters (11MB vs 259MB, competing with IDE + LLM)
- Single binary distribution matters (no "install Python first")
- Dependency stability matters (cargo vs pip ML stack fragility)
- You already know Rust (corvia v1 is Rust)

### The hybrid option:
- Python MCP server + tantivy-py (Rust BM25) + onnxruntime (C++ engine)
- Gets ~80-90% of Rust search performance with Python dev speed
- Distribution still requires Python runtime

## Recommendation

For corvia specifically: **Rust is justified** because:
1. You already know Rust from v1 (no 3-6 month learning curve)
2. Cold start is critical for stdio MCP (felt every session)
3. Single binary distribution is a key differentiator
4. The codebase is ~450 LOC total, not a maintenance burden
5. tantivy and ort are mature Rust crates (not building from scratch)
6. fastembed-rs handles embedding (already proven in v1)

If you were starting fresh with no Rust experience, Python would be the pragmatic
choice. But given v1 experience, Rust's overhead is minimal and the UX benefits
(cold start, binary distribution, RAM) are real.
