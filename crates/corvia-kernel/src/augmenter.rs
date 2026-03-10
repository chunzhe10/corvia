//! Augmenter trait and StructuredAugmenter for the RAG pipeline (D61).
//!
//! The augmenter is the second stage of the R->A->G pipeline. It takes
//! retrieved [`SearchResult`]s and formats them into an [`AugmentedContext`]
//! with source citations and token budget enforcement.
//!
//! # Implementations
//!
//! - [`StructuredAugmenter`] — numbered citations, token budget, configurable system prompt.

use corvia_common::errors::Result;
use corvia_common::types::SearchResult;
use std::sync::Arc;
use std::time::Instant;

use crate::rag_types::{AugmentationMetrics, AugmentedContext, TokenBudget};
use crate::skill_registry::SkillRegistry;

/// Default system prompt for the knowledge assistant.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a knowledge assistant. Answer questions using only the provided context. \
     Cite sources using [N] notation.";

/// Default token budget when no explicit max is set and no model context window is available.
const DEFAULT_TOKEN_BUDGET: usize = 4096;

/// Augmentation stage trait (D61). Formats search results into LLM context.
///
/// Implementations must be Send + Sync so they can be shared across
/// async tasks and wrapped in `Arc`.
pub trait Augmenter: Send + Sync {
    /// Human-readable name for metrics attribution (D62).
    fn name(&self) -> &str;

    /// Format search results into an augmented context with source citations.
    fn augment(
        &self,
        query: &str,
        results: &[SearchResult],
        budget: &TokenBudget,
    ) -> Result<AugmentedContext>;
}

/// Structured augmenter: numbered citations, token budget enforcement,
/// configurable system prompt.
///
/// Context format:
/// ```text
/// [1] (score: 0.94, source: src/auth.rs, type: function)
/// {content}
///
/// [2] (score: 0.87, source: src/db.rs, type: module)
/// {content}
///
/// ---
/// Answer the following question using ONLY the context above. Cite sources using [N] notation.
/// Question: {query}
/// ```
pub struct StructuredAugmenter {
    system_prompt: String,
    skill_registry: Option<Arc<SkillRegistry>>,
}

impl StructuredAugmenter {
    /// Create a new StructuredAugmenter with the default system prompt.
    pub fn new() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            skill_registry: None,
        }
    }

    /// Create a new StructuredAugmenter with a custom system prompt.
    pub fn with_system_prompt(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            skill_registry: None,
        }
    }

    /// Create a new StructuredAugmenter with skills support.
    pub fn with_skills(system_prompt: impl Into<String>, registry: Arc<SkillRegistry>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            skill_registry: Some(registry),
        }
    }
}

impl Default for StructuredAugmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Augmenter for StructuredAugmenter {
    fn name(&self) -> &str {
        "structured"
    }

    fn augment(
        &self,
        query: &str,
        results: &[SearchResult],
        budget: &TokenBudget,
    ) -> Result<AugmentedContext> {
        let start = Instant::now();

        // Resolve the effective token budget.
        let effective_budget = budget.max_context_tokens.unwrap_or(DEFAULT_TOKEN_BUDGET);

        // --- Skill matching and injection ---
        let mut skills_used: Vec<String> = Vec::new();
        let system_prompt = if let (Some(registry), Some(ref query_emb)) =
            (&self.skill_registry, &budget.query_embedding)
        {
            let skills_budget =
                (effective_budget as f32 * budget.reserve_for_skills) as usize;
            if skills_budget > 0 && budget.max_skills > 0 {
                let matches =
                    registry.match_skills(query_emb, budget.skill_threshold, budget.max_skills);
                let mut skill_parts: Vec<String> = Vec::new();
                let mut skill_tokens_used: usize = 0;
                for m in &matches {
                    let tokens = estimate_tokens(&m.skill.content);
                    if skill_tokens_used + tokens > skills_budget {
                        break;
                    }
                    skill_parts.push(m.skill.content.clone());
                    skills_used.push(m.skill.name.clone());
                    skill_tokens_used += tokens;
                }
                if skill_parts.is_empty() {
                    self.system_prompt.clone()
                } else {
                    let mut prompt = skill_parts.join("\n\n");
                    prompt.push_str("\n\n---\n");
                    prompt.push_str(&self.system_prompt);
                    prompt
                }
            } else {
                self.system_prompt.clone()
            }
        } else {
            self.system_prompt.clone()
        };

        // Reserve tokens for answer generation (skills budget already carved out).
        let context_budget = (effective_budget as f32
            * (1.0 - budget.reserve_for_answer - budget.reserve_for_skills))
            as usize;

        // Handle empty results.
        if results.is_empty() {
            let context = "No relevant context found.\n\n---\n".to_string()
                + &format!(
                    "Answer the following question using ONLY the context above. \
                     Cite sources using [N] notation.\nQuestion: {query}"
                );
            return Ok(AugmentedContext {
                system_prompt,
                context,
                sources: Vec::new(),
                metrics: AugmentationMetrics {
                    latency_ms: start.elapsed().as_millis() as u64,
                    token_estimate: estimate_tokens(&"No relevant context found."),
                    token_budget: context_budget,
                    sources_included: 0,
                    sources_truncated: 0,
                    augmenter_name: self.name().to_string(),
                    skills_used,
                },
            });
        }

        // Build context with numbered citations, enforcing token budget.
        let mut context_parts: Vec<String> = Vec::new();
        let mut included_sources: Vec<SearchResult> = Vec::new();
        let mut tokens_used: usize = 0;
        let mut sources_truncated: usize = 0;

        // Pre-compute footer tokens so we reserve space for it.
        let footer = format!(
            "\n---\nAnswer the following question using ONLY the context above. \
             Cite sources using [N] notation.\nQuestion: {query}"
        );
        let footer_tokens = estimate_tokens(&footer);

        for (i, sr) in results.iter().enumerate() {
            let citation_num = i + 1;

            // Build the citation header.
            let source_info = sr
                .entry
                .metadata
                .source_file
                .as_deref()
                .unwrap_or("unknown");
            let type_info = sr
                .entry
                .metadata
                .chunk_type
                .as_deref()
                .unwrap_or("unknown");

            let header = format!(
                "[{citation_num}] (score: {:.2}, source: {source_info}, type: {type_info})",
                sr.score
            );
            let block = format!("{header}\n{}\n", sr.entry.content);
            let block_tokens = estimate_tokens(&block);

            // Check if adding this block would exceed the budget.
            if tokens_used + block_tokens + footer_tokens > context_budget {
                sources_truncated += 1;
                continue;
            }

            context_parts.push(block);
            included_sources.push(sr.clone());
            tokens_used += block_tokens;
        }

        let mut context = context_parts.join("\n");
        context.push_str(&footer);

        let total_token_estimate = tokens_used + footer_tokens;

        Ok(AugmentedContext {
            system_prompt,
            context,
            sources: included_sources,
            metrics: AugmentationMetrics {
                latency_ms: start.elapsed().as_millis() as u64,
                token_estimate: total_token_estimate,
                token_budget: context_budget,
                sources_included: results.len() - sources_truncated,
                sources_truncated,
                augmenter_name: self.name().to_string(),
                skills_used,
            },
        })
    }
}

/// Estimate token count using the chars/4 heuristic.
///
/// This matches [`CharDivFourEstimator`](crate::token_estimator::CharDivFourEstimator)
/// but is a private free function so the synchronous `Augmenter` trait doesn't need
/// to depend on the `TokenEstimator` trait.
fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        (text.len() / 4).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corvia_common::agent_types::EntryStatus;
    use corvia_common::types::{EntryMetadata, KnowledgeEntry};

    /// Helper: create a SearchResult with given content, score, and optional source file.
    fn mock_result(content: &str, score: f32, source_file: Option<&str>) -> SearchResult {
        let mut entry = KnowledgeEntry::new(content.into(), "proj".into(), "v1".into());
        entry.entry_status = EntryStatus::Merged;
        entry.metadata = EntryMetadata {
            source_file: source_file.map(|s| s.into()),
            language: Some("rust".into()),
            chunk_type: Some("function".into()),
            ..Default::default()
        };
        SearchResult { entry, score }
    }

    #[test]
    fn test_structured_augmenter_formats_sources() {
        let aug = StructuredAugmenter::new();
        let results = vec![
            mock_result("fn authenticate() { /* ... */ }", 0.94, Some("src/auth.rs")),
            mock_result("fn connect_db() { /* ... */ }", 0.87, Some("src/db.rs")),
        ];
        let budget = TokenBudget {
            max_context_tokens: Some(4096),
            reserve_for_answer: 0.2,
        };

        let ctx = aug.augment("How does auth work?", &results, &budget).unwrap();

        // Verify citation markers.
        assert!(ctx.context.contains("[1]"), "context should contain [1] citation");
        assert!(ctx.context.contains("[2]"), "context should contain [2] citation");

        // Verify source file paths appear.
        assert!(ctx.context.contains("src/auth.rs"), "context should contain auth source path");
        assert!(ctx.context.contains("src/db.rs"), "context should contain db source path");

        // Verify query appears in the footer.
        assert!(
            ctx.context.contains("How does auth work?"),
            "context should contain the query"
        );

        // Verify metrics.
        assert_eq!(ctx.metrics.sources_included, 2);
        assert_eq!(ctx.metrics.sources_truncated, 0);
        assert_eq!(ctx.metrics.augmenter_name, "structured");
        assert!(ctx.metrics.token_estimate > 0);
        assert!(ctx.metrics.token_budget > 0);
    }

    #[test]
    fn test_structured_augmenter_respects_token_budget() {
        let aug = StructuredAugmenter::new();

        // One very long result (20000 chars) and one small result.
        let long_content = "x".repeat(20000);
        let results = vec![
            mock_result(&long_content, 0.95, Some("src/big.rs")),
            mock_result("fn small() {}", 0.80, Some("src/small.rs")),
        ];

        // Tiny budget: 100 tokens ~ 400 chars. The long result alone is 5000+ tokens.
        let budget = TokenBudget {
            max_context_tokens: Some(100),
            reserve_for_answer: 0.2,
        };

        let ctx = aug.augment("test query", &results, &budget).unwrap();

        // The 20000-char result should be truncated (skipped).
        assert!(
            ctx.metrics.sources_truncated >= 1,
            "at least one source should be truncated, got {}",
            ctx.metrics.sources_truncated
        );
        // Token estimate should not exceed the budget.
        assert!(
            ctx.metrics.token_estimate <= ctx.metrics.token_budget,
            "token_estimate ({}) should not exceed token_budget ({})",
            ctx.metrics.token_estimate,
            ctx.metrics.token_budget
        );
    }

    #[test]
    fn test_structured_augmenter_custom_system_prompt() {
        let custom = "You are a Rust expert. Be concise.";
        let aug = StructuredAugmenter::with_system_prompt(custom);

        let results = vec![mock_result("fn main() {}", 0.90, Some("src/main.rs"))];
        let budget = TokenBudget::default();

        let ctx = aug.augment("What does main do?", &results, &budget).unwrap();

        assert_eq!(ctx.system_prompt, custom, "system prompt should match custom value");
    }

    #[test]
    fn test_structured_augmenter_empty_results() {
        let aug = StructuredAugmenter::new();
        let budget = TokenBudget::default();

        let ctx = aug.augment("anything", &[], &budget).unwrap();

        assert!(
            ctx.context.contains("No relevant context found"),
            "empty results should produce 'No relevant context found' message"
        );
        assert_eq!(ctx.metrics.sources_included, 0);
        assert_eq!(ctx.metrics.sources_truncated, 0);
        assert!(ctx.sources.is_empty());
    }
}
