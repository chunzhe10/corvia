# Ensuring Retrieval Accuracy

**Date**: 2026-04-14

## Techniques Ranked by ROI (accuracy gain per effort)

| Priority | Technique | Accuracy Gain | Effort | V2 Status |
|----------|-----------|---------------|--------|-----------|
| 1 | Hybrid search (vector + BM25) | +5-24% | Low | Keep from v1 |
| 2 | Cross-encoder reranking | +33% average | Low | NEW (v1 had stub only) |
| 3 | Better embedding model | +6 MTEB points | Low | Keep nomic v1.5 |
| 4 | Chunk size tuning (512 tokens, 15% overlap) | +15% over bad defaults | Low | Keep from v1 |
| 5 | Contextual Retrieval (Anthropic) | -67% failure rate | Medium | Consider for v2.1 |
| 6 | Graph-augmented retrieval | +24% complex reasoning | High | Keep internal |
| 7 | Adaptive query expansion (HyDE) | +7-42% selective | Medium | Defer |

## Priority 1: Hybrid Search (Already Have)

V1 already does vector + BM25 with RRF fusion (k=60). Per-memory-type routing:
BM25 for structural/code, vectors for design/episodic. Keep this.

Research confirms: hybrid improves 70-80% of cases. +5-24% depending on domain.
Cost is near-zero (BM25 indexes are tiny and fast).

## Priority 2: Cross-Encoder Reranking (NEW for V2)

**This is the single highest-ROI addition.** +33% accuracy at +120ms latency.

| Dataset | Without Reranking | With Reranking | Improvement |
|---------|------------------|----------------|-------------|
| MS MARCO | 37.2% | 52.8% | +42% |
| Natural Questions | 45.6% | 63.1% | +38% |
| Average across 8 datasets | | | +33% |

Approach: Retrieve top 50-150, rerank to top 10-20.

**Rust options**:
- fastembed-rs supports cross-encoder models (bge-reranker-base, jina-reranker)
- synaptic-flashrank crate (BM25-based relevance, lighter)

**Model**: ms-marco-MiniLM-L6-v2 cross-encoder. +35% accuracy at 50ms for 100
document pairs. Smallest, fastest option.

**Implementation**: Add reranker stage to pipeline after fusion, before return.
V1 already has the pipeline slot (`IdentityReranker` stub). Just need to wire in
a real cross-encoder via fastembed-rs.

## Priority 3-4: Model and Chunking (Already Good)

nomic-embed-text-v1.5 at 62.4 MTEB is solid. MiniLM (56.3) is outdated.
512 token chunks with ~12.5% overlap matches industry optimum.

## Priority 5: Contextual Retrieval (Consider for V2.1)

Anthropic's technique: prepend LLM-generated 100-token context summary to each
chunk before embedding. Reduced top-20 failure rate by 35% alone, 67% combined
with BM25 + reranking.

**Tradeoff**: Requires LLM call per chunk at ingest time. Cost: ~$1.02 per million
document tokens. For a local-first tool, this means either:
- Use a local LLM (Ollama) -- adds dependency
- Use API -- adds cost + breaks local-first
- Make it opt-in for users who want higher quality

Defer to v2.1. The hybrid + reranking already gets most of the benefit.

## Priority 6: Graph Expansion (Keep Internal)

GraphRAG benchmark results:

| Task | GraphRAG | Vanilla RAG | Winner |
|------|----------|-------------|--------|
| Fact retrieval | 60.14% | 60.92% | RAG wins |
| Complex reasoning | 53.38% | 42.93% | Graph wins (+24%) |
| Summarization | 64.40% | 51.30% | Graph wins (+26%) |

Graph helps for "which modules depend on X?" but hurts for simple facts.
Keep it as internal search optimization, not an exposed tool.
V1's approach (follow edges from seed results, 2 hops, relation-weighted) is correct.

## When RAG Beats Context Stuffing

| Context Size | Accuracy | Latency to First Token |
|-------------|----------|----------------------|
| 4K-32K | 98-99% | 200-500ms |
| 100K | ~95% | 2-5 seconds |
| 500K | ~92% | 20-30 seconds |
| 1M | ~90% | 20-30+ seconds |

RAG is necessary once corpus exceeds ~200K tokens. Even within context window,
RAG provides 50-200x cost reduction and avoids "lost in the middle" effect
(10-20pp accuracy drop for info in middle of long context).

## Evaluation Strategy for V2

V1 has 15 known-answer queries across 5 categories. Expand this:

**Metrics**: NDCG@10, MRR, Recall@20, Precision@10, end-to-end answer accuracy.

**Test set**: 50+ queries covering:
- Architecture questions (multi-hop)
- Config/API questions (exact match)
- Code structure questions (AST-aware)
- Decision recall (temporal)
- Cross-document reasoning

**Automated**: `corvia bench` runs evaluation against ground truth TOML file.
Report per-category and aggregate scores.

**Important caveat**: Traditional IR metrics (NDCG, MAP) show "fundamental
misalignment with modern RAG system requirements" (arxiv 2510.21440).
End-to-end answer accuracy is what ultimately matters. Measure both.
