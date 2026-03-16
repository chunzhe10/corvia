# Dynamic Skill System — Implementation Plan

**Date:** 2026-03-10
**Status:** Shipped (v0.4.0)
**Design doc:** [2026-03-09-dynamic-skill-system-prompt-design.md](2026-03-09-dynamic-skill-system-prompt-design.md)

## Benchmarking Control

The feature is gated by a single boolean: `skills_enabled` in `RagConfig`.

```toml
[rag]
skills_enabled = false   # default: off — opt-in for now, flip to true once validated
```

When `skills_enabled = false`:
- No skill files are loaded at startup
- No skill matching at query time
- Augmenter uses the static system prompt exactly as today
- `skills_used` in metrics is always `[]`
- Zero performance overhead

This gives a clean A/B control for benchmarking: same config, toggle one field.

---

## Implementation Steps

### Step 1: Config — `corvia-common/src/config.rs`

Add 5 fields to `RagConfig` (all with serde defaults, backward compatible):

```rust
// In RagConfig:
#[serde(default)]
pub skills_enabled: bool,                        // default: false
#[serde(default = "default_skills_dirs")]
pub skills_dirs: Vec<String>,                    // default: ["skills"]
#[serde(default = "default_max_skills")]
pub max_skills: usize,                           // default: 3
#[serde(default = "default_skill_threshold")]
pub skill_threshold: f32,                        // default: 0.3
#[serde(default = "default_reserve_for_skills")]
pub reserve_for_skills: f32,                     // default: 0.15
```

Default functions:
```rust
fn default_skills_dirs() -> Vec<String> { vec!["skills".into()] }
fn default_max_skills() -> usize { 3 }
fn default_skill_threshold() -> f32 { 0.3 }
fn default_reserve_for_skills() -> f32 { 0.15 }
```

Update `RagConfig::default()` impl to include new fields.

**Tests:** Add `test_skills_config_defaults`, `test_skills_config_roundtrip`,
`test_existing_config_without_skills_still_parses`.

---

### Step 2: Data types — `corvia-kernel/src/rag_types.rs`

Add to `RetrievalResult`:
```rust
pub query_embedding: Option<Vec<f32>>,  // populated by retriever for skill matching
```

Add to `AugmentationMetrics`:
```rust
#[serde(default)]
pub skills_used: Vec<String>,  // names of injected skills
```

Update all places that construct `RetrievalResult` (VectorRetriever, GraphExpandRetriever)
to populate `query_embedding`.

Update all places that construct `AugmentationMetrics` to include `skills_used: vec![]`.

---

### Step 3: Retriever — pass through query embedding

In `retriever.rs`, both `VectorRetriever::retrieve()` and `GraphExpandRetriever::retrieve()`:
- After `let embedding = self.engine.embed(query).await?;`
- Include `query_embedding: Some(embedding.clone())` in the returned `RetrievalResult`

This is the zero-extra-cost embedding reuse from the design doc.

---

### Step 4: New file — `corvia-kernel/src/skill_registry.rs`

```rust
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub embedding: Vec<f32>,
}

pub struct SkillRegistry {
    skills: Vec<Skill>,
    dimensions: usize,
}
```

Methods:
- `SkillRegistry::load(dirs, engine) -> Result<Self>` — async, globs `*.md`, parses
  frontmatter/first-paragraph, embeds descriptions. Later dirs override same-name skills.
- `SkillRegistry::match_skills(query_embedding, threshold, max) -> Vec<(&Skill, f32)>` —
  cosine similarity matching, returns sorted by score descending.
- `SkillRegistry::len()` / `is_empty()` — for observability.

Skill file parsing:
- YAML frontmatter (`---\ndescription: ...\n---`) → use `description` field
- No frontmatter → first non-empty paragraph is the description
- Content = everything after frontmatter (or full file if no frontmatter)

Uses `cosine_similarity` from `reasoner.rs` (already pub).

**Tests:** Unit tests with mock embeddings — loading, matching, override behavior, threshold
filtering, empty registry.

---

### Step 5: Augmenter — skill injection

Modify `StructuredAugmenter`:

```rust
pub struct StructuredAugmenter {
    system_prompt: String,
    skill_registry: Option<Arc<SkillRegistry>>,  // NEW
}
```

New constructor:
```rust
pub fn with_skills(system_prompt: String, registry: Arc<SkillRegistry>) -> Self
```

Modify `Augmenter` trait — add `query_embedding` parameter:
```rust
fn augment(
    &self,
    query: &str,
    results: &[SearchResult],
    budget: &TokenBudget,
    query_embedding: Option<&[f32]>,  // NEW
    skill_config: &SkillConfig,       // NEW — threshold, max, reserve, enabled
) -> Result<AugmentedContext>;
```

Alternative (simpler, less trait churn): Keep the `Augmenter` trait unchanged. Instead,
add skill fields to `TokenBudget` or create a new `AugmentationOpts` struct that the
pipeline passes. Or, since `StructuredAugmenter` owns the `SkillRegistry`, it can do
skill matching internally if given the query embedding.

**Recommended approach:** Keep trait unchanged. Extend `TokenBudget` to carry optional
skill config. The augmenter checks internally:

```rust
pub struct TokenBudget {
    pub max_context_tokens: Option<usize>,
    pub reserve_for_answer: f32,
    pub reserve_for_skills: f32,          // NEW, default 0.0
    pub query_embedding: Option<Vec<f32>>, // NEW
    pub max_skills: usize,                // NEW, default 0
    pub skill_threshold: f32,             // NEW, default 0.3
}
```

When `StructuredAugmenter` has a `SkillRegistry` and `query_embedding` is Some:
1. Match skills using `registry.match_skills(embedding, threshold, max)`
2. Compute skills token budget: `effective_budget * reserve_for_skills`
3. Collect skill content within budget (drop lowest-scoring if over)
4. Prepend to system prompt: `[skill1 content]\n\n[skill2 content]\n\n---\n{base prompt}`
5. Record `skills_used` in metrics

When no registry or no embedding → identical to current behavior.

**Tests:** Augmenter tests with mock skills, verify system prompt modification,
token budget respected, metrics populated.

---

### Step 6: Pipeline — wire it up

In `rag_pipeline.rs`, `run_pipeline()`:

After retrieval, before augmentation:
```rust
// Pass query embedding through to augmenter via budget
let budget = TokenBudget {
    max_context_tokens,
    reserve_for_answer: self.config.reserve_for_answer,
    reserve_for_skills: if self.config.skills_enabled {
        self.config.reserve_for_skills
    } else {
        0.0
    },
    query_embedding: if self.config.skills_enabled {
        retrieval.query_embedding.clone()
    } else {
        None
    },
    max_skills: self.config.max_skills,
    skill_threshold: self.config.skill_threshold,
};
```

---

### Step 7: Factory — `corvia-kernel/src/lib.rs`

Update `create_rag_pipeline()`:

```rust
pub async fn create_rag_pipeline(     // becomes async
    store: Arc<dyn traits::QueryableStore>,
    engine: Arc<dyn traits::InferenceEngine>,
    graph: Option<Arc<dyn traits::GraphStore>>,
    generator: Option<Arc<dyn traits::GenerationEngine>>,
    config: &CorviaConfig,
) -> rag_pipeline::RagPipeline {
    // ... retriever setup unchanged ...

    let skill_registry = if config.rag.skills_enabled {
        match SkillRegistry::load(&config.rag.skills_dirs, engine.clone()).await {
            Ok(reg) if !reg.is_empty() => {
                info!(count = reg.len(), "skills loaded");
                Some(Arc::new(reg))
            }
            Ok(_) => { info!("no skills found"); None }
            Err(e) => { warn!("failed to load skills: {e}"); None }
        }
    } else {
        None
    };

    let aug = match skill_registry {
        Some(reg) => Arc::new(StructuredAugmenter::with_skills(system_prompt, reg)),
        None => Arc::new(StructuredAugmenter::new()),  // or with_system_prompt
    };
    // ...
}
```

Note: Making `create_rag_pipeline` async means updating all call sites (server, CLI).
This is a minor but necessary change.

**Update call sites:**
- `corvia-server` — wherever pipeline is constructed
- `corvia-cli` — wherever pipeline is constructed

---

### Step 8: Register module — `corvia-kernel/src/lib.rs`

Add `pub mod skill_registry;` to module declarations.

---

### Step 9: Bundled skills directory

Create `repos/corvia/skills/` with 2-3 starter skills:

1. `debugging.md` — guides LLM to ask about error messages, suggest root cause analysis
2. `architecture.md` — guides LLM to reference high-level patterns, explain relationships
3. `howto.md` — guides LLM to give step-by-step instructions, include examples

Each skill file:
```markdown
---
description: Short description used for semantic matching against queries
---

# Skill Title

Behavioral instructions for the LLM...
```

---

## Implementation Order

1. **Config** (Step 1) — no dependencies, can merge independently
2. **Data types** (Step 2) — minor, additive
3. **Retriever** (Step 3) — populate query_embedding
4. **SkillRegistry** (Step 4) — new file, self-contained
5. **Augmenter** (Step 5) — depends on 2, 4
6. **Pipeline** (Step 6) — depends on 2, 5
7. **Factory + module** (Steps 7, 8) — depends on 4, 6
8. **Bundled skills** (Step 9) — independent, can land anytime

Steps 1-3 are safe to land first as a prep PR. Steps 4-8 are the core feature.
Step 9 is content, can iterate independently.

## What Changes for Existing Users

**Nothing.** `skills_enabled` defaults to `false`. All existing configs parse unchanged.
All existing tests pass. The feature is invisible until explicitly enabled.

## Verification

```bash
# Baseline (skills off, must match current behavior exactly):
cargo test --workspace

# With skills enabled in test config:
# - augmenter tests verify skill injection
# - pipeline tests verify end-to-end with mock skills
# - config tests verify TOML roundtrip
```
