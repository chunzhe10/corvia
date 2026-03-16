# Post 2 Benchmark Demo Design

> **Status:** Shipped (v0.3.0)

**Date:** 2026-03-03
**Purpose:** Generate side-by-side terminal output for LinkedIn Post 2: "I ran the same search two ways. The difference surprised me."

## Decisions

- **Backend:** LiteStore (zero Docker — reinforces positioning)
- **Data:** Self-ingest Corvia's own codebase via `corvia demo`
- **Rigor:** 3-4 curated query pairs, not a formal benchmark (save that for M6/Post 9)
- **Tone:** Honest nuance — graph wins some, not all. Explorer voice.

## Research Context

- Mem0's own paper (arXiv:2504.19413) found graph variant only improved ~2% over base Mem0
- Mem0 acknowledged "expected relational advantages do not translate into better outcomes in multi-step reasoning scenarios"
- Graph provably wins on: temporal evolution, multi-hop relationship traversal, aggregation across time ranges
- Graph does NOT universally beat vector — this is the honest, interesting story

## How Corvia Supports This

Both retrieval modes exist at M3:

```bash
# Vector-only
curl -X POST http://localhost:8020/v1/context \
  -d '{"query": "...", "scope_id": "default", "expand_graph": false}'

# Graph-expanded (BFS traversal, α=0.3 score blending)
curl -X POST http://localhost:8020/v1/context \
  -d '{"query": "...", "scope_id": "default", "expand_graph": true, "graph_depth": 2}'
```

Both return trace metrics: `retriever_name`, `vector_results`, `graph_expanded`, `post_filter_count`, `latency_ms`.

## The 3-4 Query Pairs

### Query 1: Vector wins (or ties) — direct concept lookup

**Query:** `"embedding generation"`

**Expected behavior:**
- **Vector:** Finds the embedding module directly. Clean, relevant results.
- **Graph:** Finds the same file plus neighbors via edges. Extra context isn't harmful but doesn't help for a direct lookup.
- **Lesson:** For "where is this thing?" questions, vector search is already great.

### Query 2: Graph wins clearly — multi-hop relationship

**Query:** `"how does the RAG pipeline use graph traversal?"`

**Expected behavior:**
- **Vector:** Finds `rag_pipeline.rs` OR `retriever.rs` OR `graph_store.rs` — but likely not the connection between them. Semantic similarity to the query text pulls in whichever file mentions the most matching terms.
- **Graph:** Starts from RAG pipeline results, follows `imports`/`calls` edges to the retriever, then to the graph store. Surfaces the full chain: pipeline → retriever → graph store → traversal implementation.
- **Lesson:** For "how do these things connect?" questions, graph edges surface the chain that vector search can't see.

### Query 3: Graph adds useful context — trait implementations

**Query:** `"what traits does LiteStore implement?"`

**Expected behavior:**
- **Vector:** Returns `lite_store.rs` (high similarity to the query). May also return trait definition files if they mention LiteStore.
- **Graph:** Returns `lite_store.rs` PLUS follows `implements` edges to `QueryableStore`, `TemporalStore`, `GraphStore` trait definitions. Pulls in the trait contracts that explain *what* LiteStore does.
- **Lesson:** Graph retrieval adds structural context — not just where something is, but what it connects to.

### Query 4 (optional): Temporal chain

**Query:** `"authentication"` or another topic where supersession chains exist

**Note:** This depends on whether `corvia demo` creates temporal data (supersession chains). If the self-ingest only creates current-state entries, this query won't show temporal differences. May need to manually create a few superseded entries to demonstrate, or skip this query for Post 2 and save temporal for Post 4.

## Post Narrative Arc

```
Hook:    "I ran the same search two ways. The difference surprised me."

Setup:   "I expected graph-expanded retrieval to always beat pure vector.
          It doesn't.

          Mem0's own paper found graph memory only improved ~2% overall.
          That's... not what the marketing says."

Turn:    "But then I tried a different kind of question.

          Not 'where is this thing?' — vector search nails that.
          But 'how do these things connect?'"

Demo:    [Side-by-side terminal screenshot]
          Left: vector-only results for 'how does the RAG pipeline use graph traversal?'
          Right: graph-expanded results — showing the full chain from pipeline → retriever → graph store

Insight: "Vector search answers 'where is this thing?'
          Graph search answers 'how do these things connect?'

          They're not competing. They're different tools for different questions.

          The real skill is knowing which question you're asking."

Credit:  "Mem0's graph memory and Zep's temporal graph are exploring this same
          territory from different angles. The honest finding: the value isn't
          'graph beats vector.' It's 'graph catches what vector misses — on the
          right kind of question.'"

Tease:   "Next: I pointed my reasoning engine at a well-known codebase.
          It found things. (Post 3 preview)"

CTA:     "What does your team re-explain to AI agents every single session?"
```

## Steps to Execute

1. `cd /root/corvia-project/corvia-workspace/repos/corvia`
2. `corvia init --store lite` (if not already initialized)
3. `corvia demo` — self-ingest Corvia's own codebase
4. `corvia serve` — start the REST API on :8020
5. Run each query pair (expand_graph false vs true)
6. Capture terminal output for comparison
7. Draft the post text
8. Take screenshots for LinkedIn

## Future Post Ideas Surfaced

- **Post N (backends):** "I ran the same queries across three backends. Same trait, same results, different tradeoffs." — shows QueryableStore trait abstraction working across LiteStore/SurrealDB/PostgreSQL
- **Post 9 (M6):** Formal benchmark with precision@k, MRR, NDCG on a larger dataset. The rigorous version of what Post 2 explores casually.

## 5-Star Checklist

- **Explorer voice:** "I expected X, found Y" — discovery journal
- **Standalone insight:** "Vector vs graph is about question type, not quality" — valuable without Corvia
- **Concrete artifact:** Side-by-side terminal screenshots
- **Generous credit:** Mem0 paper, Zep temporal graph
- **Open question:** "What does your team re-explain every session?"
- **Curiosity hook:** Tease Post 3 (reasoning engine on OSS codebase)
