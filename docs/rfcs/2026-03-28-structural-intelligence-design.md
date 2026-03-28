# Structural Intelligence: Native Code Relationship Extraction

> **Status:** Draft (reviewed)
> **Date:** 2026-03-28

**Goal:** Close the repo onboarding cold-start gap by extending adapter-git's tree-sitter
relation extraction to produce richer structural edges. Teams running `corvia workspace ingest`
on an established repo should get navigable code structure on day one, not just text chunks.

**Motivation:** Evaluated GitNexus (19k GitHub stars, Feb 2026) as a code intelligence
complement. GitNexus builds knowledge graphs with Leiden clustering, execution flow tracing,
blast radius analysis, and 14-language support. However, its output is locked in LadybugDB
with no portable export format, undocumented schema, and MCP responses designed for LLM
consumption, not data interchange. An adapter integration would be fragile and tightly coupled.
Building the 80/20 natively leverages corvia's existing infrastructure and preserves the
unified knowledge graph as a differentiator.

**Prior Art:** M3.4 (graph edge improvements) and M4.2 (relation wiring fix) established
the end-to-end pipeline. After M4.2: 552 relations stored, 324 edges rebuilt on the corvia
repo. The infrastructure works. This RFC extends what gets extracted, not the pipeline itself.

---

## Problem Statement

Teams onboarding corvia with an established repo hit a cold-start problem. Today's
`corvia workspace ingest` chunks code files with tree-sitter and creates text entries with
embeddings. Agents can search by text similarity. But they cannot answer structural questions:

**Scenario 1 (JS/TS):** An agent asks `corvia_ask "what calls the authentication middleware?"`
on a freshly ingested Express app. The retriever finds the middleware file by text similarity,
but graph expansion adds 0 results because JS/TS extraction only captures `import` edges.
The agent misses every route handler that calls the middleware. After P1: `calls` edges connect
route handlers to the middleware. Graph expansion surfaces them.

**Scenario 2 (Python):** An agent asks `corvia_ask "what classes extend BaseRouter?"`. No
graph-expanded results. Python extraction has no inheritance edges. After P2: `extends` edges
connect all subclasses. The retriever surfaces them via graph expansion.

**Scenario 3 (resolution):** An agent asks about a utility module in a TS monorepo. The import
`from '@/utils/auth'` created a relation during extraction, but `wire_pipeline_relations()`
couldn't resolve the path alias to a real file. The edge was silently dropped. After P3: the
alias resolves via tsconfig paths, the edge is wired.

**What "navigable code structure on day one" means:** After ingest, `corvia_search` and
`corvia_ask` return results that include structurally related code (callers, callees,
implementations, base classes) via graph expansion. Not just text-similar chunks.

---

## Current State

### Relation Extraction by Language

| Language | imports | extends | implements | contains | calls |
|----------|---------|---------|------------|----------|-------|
| Rust | `use` with CRATE_REF resolution | -- | `impl Trait for` | `mod` bodies | Function calls (deny-list filtered) |
| JS/TS | `import` statements | -- | -- | -- | -- |
| Python | `import` / `from...import` | -- | -- | -- | -- |

### End-to-End Pipeline (Working)

```
tree-sitter AST
  -> CodeRelation (treesitter.rs)
  -> ChunkRelation (ast_chunker.rs)
  -> wire_pipeline_relations() (ingest.rs)
  -> graph.relate() (GraphStore trait)
  -> LiteStore Redb GRAPH_EDGES / PostgreSQL edges table
  -> GraphExpandRetriever (retriever.rs) uses edges for search expansion
```

All changes in this RFC are backend-agnostic via the `GraphStore` trait. Both LiteStore (Redb
+ petgraph) and PostgresStore (pgvector) are supported without modification.

### Key Files

| File | Role |
|------|------|
| `adapters/corvia-adapter-git/rust/src/treesitter.rs` | Relation extraction per language |
| `adapters/corvia-adapter-git/rust/src/ast_chunker.rs` | CodeRelation to ChunkRelation mapping |
| `crates/corvia-kernel/src/ingest.rs:220` | `wire_pipeline_relations()` target resolution |
| `crates/corvia-kernel/src/retriever.rs:440` | GraphExpandRetriever graph expansion |
| `crates/corvia-kernel/src/graph_store.rs` | LiteGraphStore (Redb + petgraph) |

---

## Design

Five priorities, ordered by effort-to-impact ratio. Each is independently shippable.

### P1: JS/TS Class Inheritance and Function Calls

**File:** `treesitter.rs` (extend `extract_js_ts_relations()`)

**Current state:** Only extracts `import_statement` nodes. The function walks top-level
children and skips everything except imports. The `_file_path` parameter is unused (underscore
prefix). P1 will use it; remove the underscore.

**Changes:**

1. **`extends`/`implements` extraction:**
   - Match `class_declaration` nodes
   - Walk children to find the `class_heritage` node (node kind, not a field name).
     In tree-sitter-javascript, `class_heritage` directly contains the base class
     expression. In tree-sitter-typescript, `class_heritage` contains `extends_clause`
     and/or `implements_clause` child nodes.
   - **JavaScript `class Foo extends Bar`:**
     - Find `class_heritage` child node
     - Extract the expression child (the base class identifier)
     - Emit: `relation: "extends", to_name: Some("Bar")`
   - **TypeScript `class Foo extends Bar<T>`:**
     - Find `class_heritage` -> `extends_clause` child
     - Extract the value expression (identifier before type arguments)
     - Emit: `relation: "extends", to_name: Some("Bar")`
   - **TypeScript `class Foo implements IBar, IBaz`:**
     - Find `class_heritage` -> `implements_clause` child
     - Iterate type children
     - Emit: `relation: "implements"` per interface
   - Handle re-exported base classes: if the extends target matches an import name,
     use that import's `to_file` instead of current file

2. **Function/method call extraction:**
   - Use a recursive walker (pattern from `collect_call_expressions()` in Rust extraction)
     applied to the entire root node, not just top-level children. This catches calls
     inside arrow functions, nested functions, and class method bodies.
   - Match `call_expression` nodes. The `function` field (via `child_by_field_name("function")`)
     gives the callable expression:
     - Simple: `foo()` -> `to_name: "foo"`
     - Member: `this.bar()` or `obj.baz()` -> `to_name: "baz"` (last segment of member expression)
     - Chained: `a.b.c()` -> `to_name: "c"`
   - Also catch `import()` dynamic imports: if the `function` child is the keyword `import`,
     emit an `"imports"` relation instead of `"calls"`.
   - Deny-list for JS/TS (prevents edge explosion):
     ```
     ["log", "warn", "error", "info", "debug",                  // console
      "then", "catch", "finally",                                // promises
      "map", "filter", "reduce", "forEach", "find", "some",     // array
      "push", "pop", "shift", "unshift", "splice", "slice",     // array mutation
      "toString", "valueOf", "hasOwnProperty",                   // object
      "stringify", "parse",                                      // JSON
      "createElement",                                           // React DOM
      "resolve", "reject",                                       // Promise
      "require"]                                                 // CJS
     ```
   - Skip calls with names <= 1 character
   - `to_file`: current file for unqualified calls. For member expressions where the
     object matches an import name, use that import's source path.

**Expected yield:** 200-500 new edges per medium-sized JS/TS project. Validate against
a reference fixture (see Evaluation section).

**Testing:**
- Class extending another class produces `"extends"` relation
- TypeScript class implementing interfaces produces `"implements"` relation per interface
- Function call inside method body produces `"calls"` relation
- Call inside arrow function assigned to a variable is captured
- Deny-listed methods are skipped
- `this.method()` extracts the method name, not "this"
- `import('./module')` produces `"imports"` not `"calls"`
- Syntax error in middle of file: tree-sitter still extracts valid parts
- Empty class body produces no relations
- TypeScript generics in extends: `class Foo extends Bar<Baz>` extracts "Bar" not "Bar<Baz>"
- Minified single-line JS: extraction still works (tree-sitter handles this)

---

### P2: Python Class Inheritance and Function Calls

**File:** `treesitter.rs` (extend `extract_python_relations()`)

**Current state:** Only extracts `import_statement` and `import_from_statement` nodes.
The `_file_path` parameter is unused. P2 will use it; remove the underscore.

**Changes:**

1. **Class inheritance extraction:**
   - Match `class_definition` nodes
   - Access superclass list via `node.child_by_field_name("superclasses")`, which returns
     an `argument_list` node containing the base class expressions as named children.
   - Iterate the `argument_list`'s named children. Each is an `identifier` (simple base
     class) or `attribute` (dotted base class like `module.ClassName`).
   - For `class Foo(Bar, Baz):`, emit:
     ```
     CodeRelation {
         relation: "extends",
         to_file: <current_file>,
         to_name: Some("Bar"),
     }
     CodeRelation {
         relation: "extends",
         to_file: <current_file>,
         to_name: Some("Baz"),
     }
     ```
   - Skip common base classes that add noise: `object`, `ABC`, `Exception`,
     `BaseException`, `type`, `dict`, `list`, `tuple`, `set`

2. **Function call extraction:**
   - Use a recursive walker applied to `function_definition` and `class_definition` bodies
   - Match `call` nodes (tree-sitter-python's call expression kind, NOT `call_expression`).
     The `function` field (via `child_by_field_name("function")`) gives the callable:
     - Simple: `foo()` -> `to_name: "foo"`
     - Attribute: `self.bar()` or `obj.baz()` -> `to_name: "baz"` (last segment)
     - Chained: `a.b.c()` -> `to_name: "c"`
   - Deny-list for Python:
     ```
     ["print", "len", "str", "int", "float", "bool", "list", "dict",
      "set", "tuple", "range", "enumerate", "zip", "map", "filter",
      "sorted", "reversed", "isinstance", "issubclass", "hasattr",
      "getattr", "setattr", "super", "type", "id", "repr", "hash",
      "next", "iter", "open", "format", "input", "vars", "dir",
      "append", "extend", "insert", "remove", "pop", "get", "keys",
      "values", "items", "update", "join", "split", "strip",
      "replace", "startswith", "endswith", "lower", "upper"]
     ```
   - Skip calls with names <= 1 character
   - Specific dunder deny-list instead of blanket `_` prefix skip:
     `["__init__", "__str__", "__repr__", "__eq__", "__hash__", "__len__",
       "__getitem__", "__setitem__", "__delitem__", "__contains__"]`
   - Keep structurally meaningful dunders like `__enter__`, `__exit__`, `__aenter__`,
     `__aexit__` (context managers are architectural)

**Expected yield:** 150-400 new edges per medium-sized Python project. Validate against
a reference fixture (see Evaluation section).

**Testing:**
- `class Foo(Bar):` produces `"extends"` relation with `to_name: "Bar"`
- Multiple superclasses produce one relation each
- `object` and `ABC` are skipped
- `self.method()` extracts method name
- Deny-listed builtins are skipped
- `super().__init__()` is deny-listed (specific dunder deny, not blanket `_` skip)
- `_private_helper()` is NOT skipped (only specific dunders are denied)
- Decorated classes (`@dataclass class Foo:`) still extract correctly
- `class Foo(Bar, metaclass=ABCMeta):` extracts "Bar" but not keyword arguments

---

### P3: JS/TS Target Resolution Improvements

**File:** `ingest.rs` (`wire_pipeline_relations()`)

**Current state:** Target resolution uses direct file match, CRATE_REF resolution (Rust
only), and suffix fallback. JS/TS relative imports (`./utils`) try extension suffixes
(`.ts`, `.js`, `.tsx`, `.jsx`) but miss common patterns.

**Changes:**

1. **`index.ts`/`index.js` resolution:**
   - When `./components` doesn't match any file, also try:
     - `./components/index.ts`
     - `./components/index.tsx`
     - `./components/index.js`
     - `./components/index.jsx`
   - This is the most common miss. Barrel files (`index.ts`) are standard in JS/TS projects.

2. **Python `__init__.py` resolution (same pattern):**
   - When a Python module path like `package.submodule` doesn't resolve, also try:
     - `package/submodule/__init__.py`
   - Same barrel-file pattern as JS/TS `index.ts`.

3. **Path alias resolution (deferred to follow-up):**
   - tsconfig.json parsing is complex: `extends` chains, `baseUrl` interaction, JSON5
     comments (real-world tsconfigs use `//` comments which are invalid JSON).
   - Defer to a follow-up RFC. The `index.ts` resolution alone captures the largest
     class of missing edges.
   - When implemented, must parse `baseUrl` alongside `paths` (paths are relative to
     baseUrl, not tsconfig location). Must handle `//` comments in tsconfig.

**Expected yield:** 50-150 previously-failing import relations now resolve to edges.

**Testing:**
- `import { Foo } from './components'` resolves to `components/index.ts`
- `from package import module` resolves to `package/module/__init__.py`
- Fallback behavior unchanged for Rust
- Extension probing order: `.ts` -> `.tsx` -> `.js` -> `.jsx`

---

### P4: Edge Weight Scoring in Retriever

**File:** `retriever.rs` (GraphExpandRetriever)

**Current state:** All edges are treated equally. The graph expansion blend formula:
```
hop 1: ((1-alpha)*cosine + alpha*0.5) * tier_weight
hop 2+: ((1-alpha)*cosine + alpha*0.33) * tier_weight
```
The `0.5` and `0.33` proximity constants are the same regardless of relation type.
The reinforcement bonus (`alpha * 0.25 * decay`) also ignores relation type.

**Changes:**

1. **Relation-type weight map:**
   ```rust
   fn relation_weight(relation: &str) -> f32 {
       match relation {
           "implements" => 0.9,   // Interface conformance (Rust traits, TS implements)
           "extends"    => 0.85,  // Class inheritance (JS/TS extends, Python)
           "calls"      => 0.7,   // Direct invocation
           "imports"    => 0.5,   // Dependency (current default)
           "uses"       => 0.4,   // Cross-file symbol reference
           "contains"   => 0.3,   // Parent-child nesting
           "references" => 0.2,   // Weak: doc mentions code
           _            => 0.5,   // Unknown: use current default
       }
   }
   ```
   These are initial values subject to tuning. See Evaluation section.

2. **Modified blend formula (new neighbors):**
   ```
   hop 1: ((1-alpha)*cosine + alpha*relation_weight) * tier_weight
   hop 2+: ((1-alpha)*cosine + alpha*relation_weight*0.67) * tier_weight
   ```

3. **Modified reinforcement bonus (already-seen neighbors):**
   ```
   bonus = alpha * 0.25 * decay * relation_weight(&edge.relation)
   ```
   This ensures repeated low-value `references` edges don't out-score a single high-value
   `implements` edge.

4. **Per-relation direction weighting:**
   Rather than a blanket 80% discount on incoming edges, use per-relation direction bias:
   ```rust
   fn direction_bias(relation: &str, is_outgoing: bool) -> f32 {
       match (relation, is_outgoing) {
           // Outgoing preferred: what I depend on > what depends on me
           ("calls", true)      => 1.0,
           ("calls", false)     => 0.8,
           ("imports", true)    => 1.0,
           ("imports", false)   => 0.8,
           // Incoming preferred: summaries/docs pointing at me are valuable
           ("references", true) => 0.8,
           ("references", false)=> 1.0,
           // Symmetric: containment and type hierarchy
           ("contains", _)      => 1.0,
           ("implements", _)    => 1.0,
           ("extends", _)       => 1.0,
           ("uses", _)          => 1.0,
           _                    => 1.0,
           }
   }
   ```
   Applied as: `final_weight = relation_weight * direction_bias`

5. **Max-edges-per-node guard:**
   When a node has more than 50 edges, sort by `relation_weight * direction_bias` and
   only expand the top 50. This prevents high-fan-out utility files from dominating
   retrieval latency.

**Implementation:** The `edges()` call already returns `GraphEdge` with the `relation`
field. Changes touch the scoring loop (retriever.rs:454-506) and the reinforcement loop
(retriever.rs:462-477).

**Expected impact:** Queries about type hierarchies surface implementations more
prominently. Queries about function behavior surface callees before callers.

**Testing:**
- An "implements" edge neighbor scores higher than an "imports" edge neighbor
  (same cosine similarity, different relation)
- Per-relation direction: outgoing `calls` edge neighbor scores higher than incoming
- Per-relation direction: incoming `references` edge scores higher than outgoing
- Reinforcement bonus is scaled by relation weight
- Nodes with >50 edges: only top 50 expanded (sorted by weight)
- Unknown relation types fall back to 0.5 / direction 1.0 (no regression)

---

### P5: HashMap Optimization in wire_pipeline_relations()

**File:** `ingest.rs` (`wire_pipeline_relations()`)

**Current state:** For each relation, linearly scans all `processed` chunks to find
source and target. O(R * C) where R = relations, C = chunks.

**Note:** Ship alongside or before P1/P2. P1 and P2 increase relation count, making the
linear scan measurably slower. Shipping P5 first prevents a temporary performance regression.

**Changes:**

1. **Build file index before the loop:**
   ```rust
   // Map source_file -> Vec<(index, start_line)>
   let mut file_index: HashMap<&str, Vec<(usize, u32)>> = HashMap::new();
   for (i, pc) in processed.iter().enumerate() {
       file_index
           .entry(pc.metadata.source_file.as_str())
           .or_default()
           .push((i, pc.start_line));
   }
   ```

2. **Source resolution:** Look up `rel.from_source_file` in `file_index`, then find
   the entry with matching `start_line`. O(1) file lookup + O(chunks_per_file) scan.

3. **Target resolution:** Look up `rel.to_file` in `file_index`. If `to_name` is set,
   scan matching file's chunks for content match. Falls back to CRATE_REF and suffix
   strategies as today.

**Expected impact:** For the corvia repo (2200+ chunks, 552 relations), this reduces
wiring time from O(2200 * 552) = O(1.2M) comparisons to O(552 * ~5) = O(2.7K). Matters
for large repos with 10K+ chunks.

**Testing:**
- Same edge count AND same edge endpoints (from_id, to_id, relation) as linear scan.
  Not just count -- verify exact edge set matches to catch resolution order changes.
- Benchmark on corvia repo to measure wiring time reduction.

---

## Edge Lifecycle

### Re-ingestion

When entries are re-ingested (same scope, new content):

1. Superseded entries retain their edges until garbage collected.
2. New entries get fresh edges via `wire_pipeline_relations()`.
3. **Stale edge cleanup:** Before wiring new edges, call `remove_edges()` for each
   entry being replaced/superseded. This prevents ghost relationships from accumulating
   (e.g., a refactored file that no longer calls function X still having a `calls` edge).

### `--fresh` ingest

`corvia workspace ingest --fresh` rebuilds the entire store. All edges are cleared and
re-created from the new ingestion. No staleness concern.

### Duplicate edge prevention

The Redb composite key format (`{from}:{relation}:{to}`) provides persistence-level
deduplication. However, petgraph's `add_edge()` always creates a new in-memory edge.
Fix: in `relate()`, check for an existing edge between the same two nodes with the same
relation before calling `add_edge()`. This keeps the in-memory graph consistent with Redb.

---

## User-Facing Observability

Structural intelligence is only valuable if users and agents can verify it works, debug
failures, and leverage the edges in queries.

### Ingest Output

Break down relation wiring results by type in the ingest summary log:

```
Structural relations: 324 stored (142 imports, 87 calls, 65 extends, 18 implements, 12 contains)
Resolution: 324/552 succeeded, 228 target-miss (41% resolution rate)
```

### `corvia workspace status`

Add graph metrics to workspace status output:

```
Graph: 324 edges (imports: 142, calls: 87, extends: 65, implements: 18, contains: 12)
```

### `corvia_system_status` MCP Tool

Add `edge_count` and `relation_types` (map of relation name to count) to the
`SystemStatus` struct so agents can programmatically check graph health.

### `corvia_graph` MCP Tool Enhancement

Currently requires an `entry_id` UUID, which agents rarely have. Add an optional
`file_path` parameter: resolve the file to its entries internally and return
their combined edge neighborhoods. This enables agents to ask "show me the graph
around `retriever.rs`" without a prior search step.

### RAG Transparency

`corvia_context` and `corvia_ask` responses should include a `graph_sources` field
in metadata showing which results came from graph expansion and via which relation type.
This lets agents distinguish vector-similarity results from structural results.

### Dashboard

Different edge types should render with different colors/styles in the graph view.
Add a legend. Not required for P1-P5 launch but should follow shortly.

---

## Evaluation Plan

### Reference Fixtures

Define one reference fixture per language with known expected edge counts:

- **JS/TS fixture:** A small Express app (~20 files) with routes, middleware, services,
  and class inheritance. Pin expected: ~80 imports, ~60 calls, ~10 extends.
- **Python fixture:** A small Django/Flask app (~15 files) with views, models, services,
  and class inheritance. Pin expected: ~50 imports, ~40 calls, ~8 extends.

These fixtures live in `tests/fixtures/structural/` and are ingested by CI tests.

### Benchmark Queries

5 evaluation queries per language, run before and after each priority ships:

**JS/TS:**
1. "what calls the authentication middleware?" (expect route handlers via `calls` edges)
2. "what classes extend BaseRouter?" (expect subclasses via `extends` edges)
3. "what does the database service import?" (expect dependency chain via `imports`)
4. "how is the User model used?" (expect controllers/services via cross-file `uses`)
5. "what implements the Logger interface?" (expect concrete classes via `implements`)

**Python:**
1. "what calls the authenticate decorator?" (expect view functions via `calls` edges)
2. "what classes extend BaseModel?" (expect model subclasses via `extends`)
3. "what does the auth module import?" (expect dependency chain)
4. "how is the UserService used across files?" (expect controllers via `uses`)
5. "what inherits from APIView?" (expect view classes via `extends`)

For each query, record: result count, graph_expanded count, and whether expected files
appear in results.

---

## Comparison: After P1-P5 vs. GitNexus

### Where Corvia Matches or Exceeds GitNexus

| Capability | Corvia (post-P5) | GitNexus |
|---|---|---|
| Import extraction | Rust + JS/TS + Python with multi-strategy resolution | 14 languages |
| Call graph | Rust + JS/TS + Python with deny-list filtering | All languages |
| Class inheritance | Rust (implements) + JS/TS (extends/implements) + Python (extends) | All languages (+ constructor inference) |
| Weighted graph expansion | Relation-type-aware + direction-aware scoring | No scoring (equal edges) |
| Organizational memory | Decisions, designs, learnings, temporal queries | None |
| Multi-agent coordination | Session isolation, merge queue, conflict resolution | None |
| Knowledge lifecycle | Supersession, forgetting, retention scoring | Static index (full re-index) |
| Temporal reasoning | "What did we know at time T?" via as_of, history, evolution | None |
| Unified search | Code structure + org knowledge in one RAG query | Code structure only |
| Storage backends | LiteStore + PostgreSQL | LadybugDB only |

### Where GitNexus Remains Ahead

| Capability | Gap | Path to Close |
|---|---|---|
| Leiden community clustering | No equivalent. | Future: integrate a Leiden implementation (e.g., `grappolo` crate) |
| Execution flow tracing | No process-level abstraction. Have edges but not named paths. | Future: build on call graph (P1/P2) to trace entry-to-terminal flows |
| Blast radius analysis | Graph traversal exists but no confidence scoring or depth grouping. | Future: extend `traverse()` with confidence propagation |
| Change detection (git diff to impact) | No git-diff-to-graph mapping. | Future: map diff hunks to affected entries, traverse edges |
| 14 language support | 4 languages. Missing Java, Go, C/C++, Kotlin, PHP, Ruby, Swift, Dart. | Future: add tree-sitter grammars incrementally (Java and Go first) |
| Framework-aware entry points | No detection of Express routes, React components, Django views. | Future: language-specific heuristics in relation extraction |
| LLM-assisted architectural summaries | No equivalent to GitNexus wiki generation. | Future RFC: LLM synthesis pass post-ingest (deferred from this RFC) |
| Generated agent skills | No per-module `.claude/skills/` generation. | Future: use summaries to generate skill files |
| Coordinated multi-file rename | No equivalent. | Future: use call graph to plan renames |

### Strategic Position

After P1-P5, corvia's pitch is: "The single knowledge layer for AI agents. Code structure,
organizational memory, and temporal reasoning in one unified graph." GitNexus is deeper
on code analysis but narrower in scope. Corvia covers the full lifecycle. Teams wanting
both can run GitNexus alongside. Teams wanting simplicity get a good-enough structural
foundation from corvia alone.

---

## Implementation Order

| Phase | Depends On | Estimated Scope |
|-------|-----------|-----------------|
| P5 (HashMap optimization) | None | ~60 lines in ingest.rs |
| P1 (JS/TS relations) | None | ~300 lines in treesitter.rs |
| P2 (Python relations) | None | ~250 lines in treesitter.rs |
| P3 (JS/TS + Python resolution) | P1+P2 (more relations to resolve) | ~150 lines in ingest.rs |
| P4 (edge weights) | P1+P2 (more relation types to weight) | ~80 lines in retriever.rs |
| Observability | P1 (something to observe) | ~100 lines across CLI, server, MCP |

P5 ships first (prevents perf regression when P1/P2 land). P1 and P2 can be done in parallel.
P3 and P4 benefit from P1+P2 shipping first but are not blocked. Observability can ship
incrementally alongside any priority.

---

## Testing Strategy

### Regression Tests

Before P1/P2 ship, add a snapshot regression test:
- Ingest the corvia repo (or representative subset)
- Assert Rust relation counts are within tight tolerance of the known baseline (552 relations)
- Run after P1/P2 to verify no cross-language contamination

### Integration Tests

Each priority must include a pipeline integration test:
1. Create a temp `LiteStore` + `LiteGraphStore`
2. Chunk a multi-file test fixture (specific to the priority's language)
3. Run `wire_pipeline_relations()` against the chunked output
4. Assert graph edges exist with correct relation types and endpoints
5. Run a retriever query and verify scoring order (for P4)

### Edge Case Coverage

Minimum edge cases per language (P1/P2):
- Syntax error in middle of file (tree-sitter partial parse)
- Empty class body
- TypeScript generics in extends: `class Foo extends Bar<Baz>`
- Python decorated class: `@dataclass class Foo(Base):`
- Minified single-line JS
- File with only imports and no functions/classes

---

## Relation Types (Complete Set After P1-P5)

| Relation | Meaning | Direction | Languages |
|----------|---------|-----------|-----------|
| `imports` | File/module dependency | A imports B | Rust, JS/TS, Python |
| `extends` | Class inheritance | A extends B | JS/TS, Python |
| `implements` | Interface/trait conformance | A implements B | Rust, TypeScript |
| `calls` | Function/method invocation | A calls B | Rust, JS/TS, Python |
| `contains` | Parent-child nesting | A contains B | Rust |
| `uses` | Cross-file symbol reference | A uses B | All (from chunking pipeline) |
| `references` | Documentation mentions code | A references B | All (from markdown chunker) |

All relation names must be colon-free (enforced by `graph.relate()` validation).

---

## Decisions

### D-SI-1: Build Native, Not Integrate GitNexus
- **Decision:** Build structural intelligence in adapter-git rather than consuming GitNexus output
- **Rationale:** GitNexus has no export format. LadybugDB schema is undocumented. MCP responses
  are for LLM consumption. Adapter would be fragile and tightly coupled.

### D-SI-2: 80/20 Scope (P1-P5), Expand Later
- **Decision:** Ship the highest-impact relation extractors for 4 languages first. Defer Leiden
  clustering, execution flow tracing, blast radius, LLM summaries, and additional languages.
- **Rationale:** P1-P5 leverages existing infrastructure. Each is independently shippable.
  LLM-assisted summaries (originally P6) deferred to a separate RFC due to fundamentally
  different architecture (LLM dependency, new entry type, runtime optionality).

### D-SI-3: Reuse Rust's Deny-List Pattern for JS/TS and Python
- **Decision:** Each language gets its own deny-list of common methods to prevent edge explosion.
- **Rationale:** Proven effective in Rust extraction. Without deny-lists, `.map()`, `.filter()`,
  `print()` etc. create thousands of low-value edges.
- **Future:** Make deny-lists configurable via `corvia.toml` per language.

### D-SI-4: Separate Relation Types for Inheritance vs. Interface Conformance
- **Decision:** Use `"extends"` for class inheritance (JS/TS, Python) and `"implements"` for
  interface/trait conformance (Rust, TypeScript). Not a single overloaded `"implements"`.
- **Rationale:** These are semantically distinct. Conflating them prevents precise queries
  ("find all implementations of this interface" vs. "find all subclasses"). The weight map
  in P4 can differentiate them.

### D-SI-5: Edge Weights by Relation Type with Per-Relation Direction Bias
- **Decision:** Weight graph expansion by relation type and direction. `implements` > `extends`
  > `calls` > `imports` > `uses` > `contains` > `references`. Direction bias is per-relation,
  not global.
- **Rationale:** Not all edges are equally informative. Outgoing `calls`/`imports` are more
  relevant than incoming for most queries. But incoming `references` (docs pointing at code)
  are more valuable than outgoing. A blanket direction discount is too blunt.

### D-SI-6: Stale Edge Cleanup on Re-ingestion
- **Decision:** Call `remove_edges()` for superseded/replaced entries before wiring new edges.
- **Rationale:** Without cleanup, refactored files accumulate ghost relationships. The graph
  diverges from the code over time, degrading traversal quality.

### D-SI-7: Defer tsconfig Path Alias Resolution
- **Decision:** Ship `index.ts`/`__init__.py` barrel file resolution in P3. Defer tsconfig
  `paths` + `baseUrl` parsing to a follow-up.
- **Rationale:** tsconfig parsing is complex (JSON5 comments, `extends` chains, `baseUrl`
  interaction). Index resolution alone captures the largest class of missing edges. Path
  alias resolution is worth doing but needs its own focused design.

---

## Future Work (Separate RFCs)

- **LLM-assisted architectural summaries:** Deferred from this RFC. Generates high-level
  "how does this module work?" entries post-ingest using LLM synthesis. Requires its own
  design for inference dependency, module detection, incremental updates, and user controls
  (`--no-summaries`, `--dry-run`, separate `corvia workspace summarize` command).
- **Configurable deny-lists:** Per-language deny-list overrides in `corvia.toml`.
- **Dashboard edge visualization:** Render different edge types with different colors/styles.
- **Additional languages:** Java and Go as next priorities based on adoption.
