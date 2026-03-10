//! Skill registry for dynamic system prompt assembly.
//!
//! Skills are markdown files containing behavioral instructions for the
//! generation layer. At startup, skills are loaded and their descriptions
//! embedded into an in-memory registry. At query time, the augmenter selects
//! semantically similar skills and injects their content into the system prompt.

use corvia_common::errors::{CorviaError, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

use crate::reasoner::cosine_similarity;
use crate::traits::InferenceEngine;

/// A loaded skill with pre-computed embedding.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Filename stem, e.g. "ai-assisted-development".
    pub name: String,
    /// Short description used for semantic matching.
    pub description: String,
    /// Full markdown body injected into the system prompt.
    pub content: String,
    /// Pre-computed embedding of the description.
    pub embedding: Vec<f32>,
}

/// In-memory registry of skills with pre-computed embeddings.
#[derive(Debug)]
pub struct SkillRegistry {
    skills: Vec<Skill>,
    dimensions: usize,
}

/// A matched skill with its similarity score.
#[derive(Debug, Clone)]
pub struct SkillMatch<'a> {
    pub skill: &'a Skill,
    pub score: f32,
}

impl SkillRegistry {
    /// Load skills from one or more directories.
    ///
    /// Globs `*.md` from each directory. If multiple directories contain a
    /// file with the same stem, the later directory wins (user overrides bundled).
    /// Each skill's description is embedded using the provided engine.
    pub async fn load(
        dirs: &[String],
        engine: Arc<dyn InferenceEngine>,
    ) -> Result<Self> {
        let dimensions = engine.dimensions();

        // Collect skill files, later dirs override same-named earlier ones.
        let mut skill_files: HashMap<String, (String, String)> = HashMap::new(); // name -> (description, content)

        for dir in dirs {
            let dir_path = Path::new(dir);
            if !dir_path.is_dir() {
                info!(dir = dir, "skills directory not found, skipping");
                continue;
            }

            let pattern = dir_path.join("*.md");
            let pattern_str = pattern.to_string_lossy();
            let entries: Vec<_> = glob::glob(&pattern_str)
                .map_err(|e| CorviaError::Config(format!("invalid skills glob: {e}")))?
                .filter_map(|r| r.ok())
                .collect();

            for path in entries {
                let name = match path.file_stem() {
                    Some(s) => s.to_string_lossy().to_string(),
                    None => continue,
                };

                let raw = match std::fs::read_to_string(&path) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "failed to read skill file");
                        continue;
                    }
                };

                let (description, content) = parse_skill_file(&raw);
                if description.is_empty() {
                    warn!(path = %path.display(), "skill has no description, skipping");
                    continue;
                }

                skill_files.insert(name, (description, content));
            }
        }

        if skill_files.is_empty() {
            return Ok(Self {
                skills: Vec::new(),
                dimensions,
            });
        }

        // Embed all descriptions in a single batch.
        let names: Vec<String> = skill_files.keys().cloned().collect();
        let descriptions: Vec<String> = names
            .iter()
            .map(|n| skill_files[n].0.clone())
            .collect();

        let embeddings = engine.embed_batch(&descriptions).await?;

        let skills: Vec<Skill> = names
            .into_iter()
            .zip(descriptions)
            .zip(embeddings)
            .map(|((name, description), embedding)| {
                let content = skill_files[&name].1.clone();
                Skill {
                    name,
                    description,
                    content,
                    embedding,
                }
            })
            .collect();

        info!(count = skills.len(), "skills loaded into registry");
        Ok(Self { skills, dimensions })
    }

    /// Match skills against a query embedding.
    ///
    /// Returns skills above `threshold` sorted by score descending,
    /// limited to `max_skills`.
    pub fn match_skills(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        max_skills: usize,
    ) -> Vec<SkillMatch<'_>> {
        let mut matches: Vec<SkillMatch<'_>> = self
            .skills
            .iter()
            .filter_map(|skill| {
                let score = cosine_similarity(query_embedding, &skill.embedding);
                if score >= threshold {
                    Some(SkillMatch { skill, score })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending.
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(max_skills);
        matches
    }

    /// Number of skills in the registry.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Parse a skill markdown file into (description, content).
///
/// If the file has YAML frontmatter (`---\n...\n---`), the `description`
/// field is used. Otherwise, the first non-empty paragraph is the description.
/// Content is everything after the frontmatter (or the full file).
fn parse_skill_file(raw: &str) -> (String, String) {
    let trimmed = raw.trim();

    // Try YAML frontmatter.
    if trimmed.starts_with("---") {
        if let Some(end) = trimmed[3..].find("\n---") {
            let frontmatter = &trimmed[3..3 + end].trim();
            let content = trimmed[3 + end + 4..].trim().to_string();

            // Simple YAML parsing — look for `description:` line.
            for line in frontmatter.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("description:") {
                    let desc = rest.trim().trim_matches('"').trim_matches('\'').to_string();
                    if !desc.is_empty() {
                        return (desc, content);
                    }
                }
            }

            // Frontmatter exists but no description field — use first paragraph of content.
            let desc = first_paragraph(&content);
            return (desc, content);
        }
    }

    // No frontmatter — first paragraph is description, full file is content.
    let desc = first_paragraph(trimmed);
    (desc, trimmed.to_string())
}

/// Extract the first non-empty paragraph from text.
fn first_paragraph(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        // Skip headings.
        if line.starts_with('#') {
            if !lines.is_empty() {
                break;
            }
            continue;
        }
        if line.is_empty() {
            if !lines.is_empty() {
                break;
            }
            continue;
        }
        lines.push(line);
    }
    lines.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Mock engine that returns fixed embeddings for testing.
    struct MockEngine {
        dimensions: usize,
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            // Return distinct embeddings so we can test matching.
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0; self.dimensions];
                    v[i % self.dimensions] = 1.0;
                    v
                })
                .collect())
        }
        fn dimensions(&self) -> usize {
            self.dimensions
        }
    }

    #[test]
    fn test_parse_skill_with_frontmatter() {
        let raw = r#"---
description: Guide debugging and root cause analysis
---

# Debugging Skill

When debugging, always ask about error messages first.
"#;
        let (desc, content) = parse_skill_file(raw);
        assert_eq!(desc, "Guide debugging and root cause analysis");
        assert!(content.contains("# Debugging Skill"));
        assert!(content.contains("error messages"));
    }

    #[test]
    fn test_parse_skill_without_frontmatter() {
        let raw = r#"# Architecture Patterns

Guide the LLM to reference high-level patterns.

Explain component relationships clearly.
"#;
        let (desc, content) = parse_skill_file(raw);
        assert_eq!(desc, "Guide the LLM to reference high-level patterns.");
        assert!(content.contains("# Architecture Patterns"));
    }

    #[test]
    fn test_parse_skill_frontmatter_no_description() {
        let raw = r#"---
tags: [debugging]
---

First paragraph is the fallback description.

More content here.
"#;
        let (desc, content) = parse_skill_file(raw);
        assert_eq!(desc, "First paragraph is the fallback description.");
        assert!(content.contains("More content here."));
    }

    #[test]
    fn test_parse_skill_quoted_description() {
        let raw = "---\ndescription: \"Quoted description value\"\n---\n\nContent.";
        let (desc, _) = parse_skill_file(raw);
        assert_eq!(desc, "Quoted description value");
    }

    #[test]
    fn test_first_paragraph_skips_headings() {
        let text = "# Title\n\nActual paragraph content.\n\nSecond paragraph.";
        assert_eq!(first_paragraph(text), "Actual paragraph content.");
    }

    #[test]
    fn test_match_skills_threshold_filtering() {
        let registry = SkillRegistry {
            skills: vec![
                Skill {
                    name: "debugging".into(),
                    description: "debug".into(),
                    content: "debug content".into(),
                    embedding: vec![1.0, 0.0, 0.0],
                },
                Skill {
                    name: "architecture".into(),
                    description: "arch".into(),
                    content: "arch content".into(),
                    embedding: vec![0.0, 1.0, 0.0],
                },
            ],
            dimensions: 3,
        };

        // Query aligned with "debugging" skill.
        let query_emb = vec![0.9, 0.1, 0.0];
        let matches = registry.match_skills(&query_emb, 0.5, 3);

        assert_eq!(matches.len(), 1, "only debugging should pass threshold");
        assert_eq!(matches[0].skill.name, "debugging");
        assert!(matches[0].score > 0.5);
    }

    #[test]
    fn test_match_skills_max_limit() {
        let registry = SkillRegistry {
            skills: vec![
                Skill {
                    name: "a".into(),
                    description: "a".into(),
                    content: "a".into(),
                    embedding: vec![1.0, 0.0, 0.0],
                },
                Skill {
                    name: "b".into(),
                    description: "b".into(),
                    content: "b".into(),
                    embedding: vec![0.9, 0.1, 0.0],
                },
                Skill {
                    name: "c".into(),
                    description: "c".into(),
                    content: "c".into(),
                    embedding: vec![0.8, 0.2, 0.0],
                },
            ],
            dimensions: 3,
        };

        let query_emb = vec![1.0, 0.0, 0.0];
        let matches = registry.match_skills(&query_emb, 0.0, 2);

        assert_eq!(matches.len(), 2, "should be limited to max_skills=2");
        // Should be sorted by score descending.
        assert!(matches[0].score >= matches[1].score);
    }

    #[test]
    fn test_match_skills_empty_registry() {
        let registry = SkillRegistry {
            skills: Vec::new(),
            dimensions: 3,
        };
        let matches = registry.match_skills(&[1.0, 0.0, 0.0], 0.0, 10);
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn test_load_from_directory() {
        let dir = tempfile::tempdir().unwrap();

        // Create a skill file with frontmatter.
        let skill_path = dir.path().join("test-skill.md");
        std::fs::write(
            &skill_path,
            "---\ndescription: A test skill for testing\n---\n\n# Test\n\nDo testing things.",
        )
        .unwrap();

        let engine = Arc::new(MockEngine { dimensions: 3 }) as Arc<dyn InferenceEngine>;
        let dirs = vec![dir.path().to_string_lossy().to_string()];
        let registry = SkillRegistry::load(&dirs, engine).await.unwrap();

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert_eq!(registry.dimensions(), 3);
    }

    #[tokio::test]
    async fn test_load_later_dir_overrides() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        // Same filename in both dirs.
        std::fs::write(
            dir1.path().join("skill.md"),
            "---\ndescription: Version 1\n---\n\nContent v1.",
        )
        .unwrap();
        std::fs::write(
            dir2.path().join("skill.md"),
            "---\ndescription: Version 2\n---\n\nContent v2.",
        )
        .unwrap();

        let engine = Arc::new(MockEngine { dimensions: 3 }) as Arc<dyn InferenceEngine>;
        let dirs = vec![
            dir1.path().to_string_lossy().to_string(),
            dir2.path().to_string_lossy().to_string(),
        ];
        let registry = SkillRegistry::load(&dirs, engine).await.unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.skills[0].description, "Version 2");
    }

    #[tokio::test]
    async fn test_load_missing_dir_skipped() {
        let engine = Arc::new(MockEngine { dimensions: 3 }) as Arc<dyn InferenceEngine>;
        let dirs = vec!["/nonexistent/path".into()];
        let registry = SkillRegistry::load(&dirs, engine).await.unwrap();
        assert!(registry.is_empty());
    }
}
