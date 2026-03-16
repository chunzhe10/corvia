# Dynamic Skill-Based System Prompt

**Date:** 2026-03-09
**Status:** Shipped (v0.4.0)
**Scope:** corvia-kernel (augmenter, rag_pipeline), corvia-common (config)

## Problem

The RAG generation layer uses a static system prompt:
```
"You are a knowledge assistant. Answer questions using only the provided context.
 Cite sources using [N] notation."
```

This means every query gets the same reasoning approach regardless of type. A debugging
question, an architecture question, and a how-to question all receive identical behavioral
instructions. The generation layer adapts *what* it knows (via retrieval) but not *how*
it reasons.

## Solution

Skills are markdown files that contain behavioral instructions for the generation layer.
At startup, skills are loaded and embedded into an in-memory registry. At query time, the
augmenter selects semantically similar skills and injects their content into the system
prompt, giving the LLM task-appropriate reasoning instructions.

## Architecture Decisions

- **Model-agnostic** — uses `InferenceEngine` trait for embedding, cosine similarity for
  matching. Works with any embedding model (Ollama, corvia-inference, vLLM).
- **Embedding-based selection** over keyword matching — semantic intent matching scales
  naturally, avoids brittle keyword maintenance, and dogfoods corvia's core capability.
- **Embed the description, not full content** — descriptions are concise intent signals.
  Full content has too much implementation noise for matching.
- **Skills are behavioral instructions (option A)** — they change *how* the LLM reasons,
  not *what* it retrieves. Retrieval is already handled by the existing pipeline.
- **Dashboard-extensible** — `skills_used` in metrics enables future dashboard visibility
  into which skills influenced each answer.

## Skill Tiers

1. **Bundled skills** — ship with corvia, always available. Generic patterns applicable
   to any knowledge domain.
2. **User skills** — users add their own to a configured directory. Domain-specific
   behavioral instructions.
3. **Demo/workspace skills** — specific skills for demos or particular deployments
   (e.g., `.agents/skills/` in this workspace).

Config supports multiple directories merged at load time:
```toml
[rag]
skills_dirs = ["skills", "/path/to/user/skills"]
```

Bundled skills are the default. User-provided directories are additive.

## Data Model

### Skill

```rust
pub struct Skill {
    pub name: String,           // filename stem: "ai-assisted-development"
    pub description: String,    // frontmatter description or first paragraph
    pub content: String,        // full markdown body (injected into system prompt)
    pub embedding: Vec<f32>,    // pre-computed at load time from description
}
```

### SkillRegistry

```rust
pub struct SkillRegistry {
    skills: Vec<Skill>,
    dimensions: usize,
}
```

**Loading flow:**
1. Read `skills_dirs` from `RagConfig` (default: bundled skills directory)
2. Glob `*.md` from each directory, merge (later dirs override same-named files)
3. Parse each file — name from filename, description from YAML frontmatter or first paragraph
4. Embed descriptions using `InferenceEngine`
5. Hold in memory (skills are small — dozens of files)

## Skill Selection

When a query arrives:
1. Reuse query embedding from retrieval stage (already computed — zero extra cost)
2. Compute cosine similarity against each skill's pre-computed embedding
3. Select skills above `skill_threshold` (default: `0.3`)
4. Sort by score descending, take top `max_skills` (default: `3`)
5. Inject selected skill content into system prompt, before the base prompt

### System Prompt Assembly

```
[Selected Skill 1 content]

[Selected Skill 2 content]

---
{base system prompt}
```

### Token Budget

Skills get their own budget carved from the context window:

```
Context window (e.g. 4096 tokens)
+-- Skills budget:    15%  (614 tokens)
+-- Context budget:   65%  (2662 tokens)
+-- Answer reserve:   20%  (820 tokens)
```

If selected skills exceed the skills budget, drop lowest-scoring until they fit.

## Config Additions

```toml
[rag]
skills_dirs = ["skills"]      # directories to load skills from (default: bundled)
max_skills = 3                # max skills injected per query
skill_threshold = 0.3         # minimum cosine similarity to select a skill
reserve_for_skills = 0.15     # fraction of context window for skills
```

All fields have serde defaults — existing configs work unchanged.

## Integration Points

### Changes by file

| File | Change | Breaking? |
|------|--------|-----------|
| `config.rs` | Add 4 fields to `RagConfig` | No (serde defaults) |
| `augmenter.rs` | `StructuredAugmenter` accepts optional `SkillRegistry` | No |
| `rag_pipeline.rs` | Pass query embedding from retrieval to augmenter | Minor |
| `rag_types.rs` | Add `skills_used: Vec<String>` to `AugmentationMetrics` | Additive |
| `rag_types.rs` | Add `query_embedding: Option<Vec<f32>>` to `RetrievalResult` | Additive |
| `lib.rs` | Load `SkillRegistry` if `skills_dirs` configured | No |
| New: `skill_registry.rs` | `SkillRegistry` struct, loading, matching | New file |

### What does NOT change

- `Augmenter` trait signature — skill injection is internal to `StructuredAugmenter`
- `GenerationEngine` trait — unchanged, receives a richer system prompt
- `Retriever` trait — unchanged
- MCP tools / REST endpoints — unchanged (skills_used flows via trace)
- All existing tests — no skills configured = identical behavior

### Query embedding reuse

The retriever already embeds the query. `RetrievalResult` gains an optional
`query_embedding: Option<Vec<f32>>`. The pipeline passes this to the augmenter
for skill matching. Zero extra inference calls.

### Observability

`AugmentationMetrics` gains `skills_used: Vec<String>` — names of injected skills.
Flows through `PipelineTrace` into MCP/REST responses. Future dashboard can display
which skills influenced each answer.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| No `skills_dirs` configured | Use bundled skills directory |
| Skills directory is empty | Empty registry, no skills injected |
| Skill file has no frontmatter | Use first paragraph as description |
| All skills below threshold | No skills injected — base system prompt only |
| Skills exceed token budget | Drop lowest-scoring until they fit |
| Embedding engine unavailable at startup | Warn, no skills loaded |
| Skill file very large (>5000 words) | Truncate content to skills token budget |
| Duplicate skill names across dirs | Later directory wins (user overrides bundled) |

## Backward Compatibility

Fully opt-in. No `skills_dirs` = bundled default behavior. Every existing deployment
works identically. All new config fields have serde defaults. `SkillRegistry` is
`Option<Arc<SkillRegistry>>` — `None` preserves current code path exactly.
