# Retrieval Learnings from Corvia V1

**Date**: 2026-04-14

## What Worked

1. **Composable pipeline > monolithic retriever**. V1 started with a monolithic
   `VectorRetriever` and `GraphExpandRetriever`. Refactored to: Searcher → Fusion →
   Expander → Reranker. Each stage testable independently. Keep this pattern.

2. **Hybrid search (vector + BM25)**. Tantivy BM25 integration was a clear win.
   BM25 for structural/code content, vectors for design/episodic content. Per-memory-type
   routing (MultiChannelSearcher) is the right approach.

3. **RRF fusion (k=60)**. Reciprocal Rank Fusion is robust, parameter-free. Simple
   formula: `score(d) = sum_i 1 / (k + rank_i(d))`. Works in 70-80% of cases without
   tuning.

4. **Graph expansion with relation weighting**. Following edges from search results
   to related entries improved recall for architecture queries. Weight by relation type:
   implements(0.9) > extends(0.85) > calls(0.7) > imports(0.5) > references(0.2).

5. **Quality signals (confidence + suggestions)**. HIGH/MEDIUM/LOW confidence with
   actionable suggestions ("try broader terms") enables agent retry loops.
   Thresholds: HIGH >= 0.65 top_score + 3 results, LOW < 0.45.

6. **Tiered knowledge lifecycle**. Hot/Warm/Cold/Forgotten with tier weights
   (1.0/0.7/0.3/0.0). Retention scoring: 35% time-decay + 30% access + 20% graph
   edges + 15% confidence.

7. **Semantic sub-splitting (Max-Min algorithm)**. Self-calibrating: new sentence
   joins group if max_similarity >= min_pairwise_similarity. Better than naive
   sentence splitting for long sections.

8. **Oversample 3x for metadata filters**. Post-filter elimination removes ~70%
   of results. Fetching 3x compensates.

## What Didn't Work

1. **SurrealDB storage** -- 893 transitive deps, removed in v0.4.4.
2. **Monolithic retrievers** -- couldn't test or swap individual stages.
3. **Single HNSW index for all content types** -- replaced by per-memory-type channels.

## V1 Config Defaults Worth Keeping

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_tokens (chunk) | 512 | Matches FloTorch 2026 benchmark optimum |
| overlap_tokens | 64 (~12.5%) | Close to NVIDIA's 15% optimum |
| RRF k | 60 | Standard, robust |
| graph_alpha | 0.3 | Blend: 70% cosine + 30% edge score |
| graph_depth | 2 | Diminishing returns beyond 2 hops |
| min_tokens (chunk) | 32 | Merge threshold for tiny chunks |

## Not Yet Implemented in V1

- **Reranking** -- pipeline slot exists (`IdentityReranker` stub) but no cross-encoder.
  Research shows +33% accuracy gain. Priority for v2.
- **Contextual Retrieval** (Anthropic technique) -- prepend LLM-generated context
  summary to each chunk before embedding. -67% failure rate. Requires LLM at ingest
  time (tradeoff: cost vs quality).
- **Relation-weight scoring in pipeline** -- defined in retriever.rs but not ported
  to GraphExpander. Follow-up noted in code comments.
