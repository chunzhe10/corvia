//! Semantic sub-splitting for oversized Markdown heading-section chunks.
//!
//! Implements the Max-Min Semantic Chunking algorithm (Springer 2025) with a
//! pluggable `SimilaritySource` trait. V1 uses pre-computed embeddings from
//! `InferenceEngine`; future V2 can use inline embedding during the chunking pipeline.
//!
//! The algorithm groups consecutive sentences into coherent sub-chunks without
//! requiring a magic threshold. Internal coherence is self-calibrating: a new
//! sentence joins a group if its max similarity to the group >= the group's
//! min pairwise similarity.
//!
//! Design doc: corvia knowledge entry 019d3999-493e-7be2-ba58-8580b570730a

/// Source of similarity scores between text segments.
/// Implementations can use real embeddings, TF-IDF, or any other signal.
pub trait SimilaritySource {
    /// Compute cosine similarity between two segments identified by index.
    /// Indices correspond to the sentences slice passed to `max_min_split`.
    fn similarity(&self, i: usize, j: usize) -> f32;
}

/// Similarity source backed by pre-computed embedding vectors.
/// Computes cosine similarity on-demand from the stored vectors.
pub struct EmbeddingSimilarity {
    embeddings: Vec<Vec<f32>>,
}

impl EmbeddingSimilarity {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self { embeddings }
    }
}

impl SimilaritySource for EmbeddingSimilarity {
    fn similarity(&self, i: usize, j: usize) -> f32 {
        if i >= self.embeddings.len() || j >= self.embeddings.len() {
            return 0.0;
        }
        cosine_similarity(&self.embeddings[i], &self.embeddings[j])
    }
}

/// Cosine similarity between two vectors. Returns 0.0 for zero-length vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Maximum number of segments to process per chunk.
/// Beyond this, fall back to naive splitting (N^2 similarity is too expensive).
pub const MAX_SEGMENTS_PER_CHUNK: usize = 50;

/// Split sentences into semantically coherent groups using the Max-Min algorithm.
///
/// For each new sentence, if its maximum similarity to the current group >=
/// the minimum pairwise similarity within the group, it joins the group.
/// Otherwise, a new group starts.
///
/// Returns indices into the `sentences` slice, grouped by semantic coherence.
pub fn max_min_split(
    sentence_count: usize,
    similarity: &dyn SimilaritySource,
) -> Vec<Vec<usize>> {
    if sentence_count == 0 {
        return vec![];
    }
    if sentence_count == 1 {
        return vec![vec![0]];
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = vec![0];

    for i in 1..sentence_count {
        // Max similarity between new sentence and any sentence in current group
        let max_sim_to_group = current_group
            .iter()
            .map(|&j| similarity.similarity(i, j))
            .fold(f32::NEG_INFINITY, f32::max);

        // Min pairwise similarity within current group
        let min_sim_in_group = if current_group.len() == 1 {
            // Single-sentence group: no internal pairs exist yet.
            // The new sentence always joins, establishing the group's coherence.
            f32::NEG_INFINITY
        } else {
            let mut min_sim = f32::INFINITY;
            for (a_idx, &a) in current_group.iter().enumerate() {
                for &b in &current_group[a_idx + 1..] {
                    let sim = similarity.similarity(a, b);
                    if sim < min_sim {
                        min_sim = sim;
                    }
                }
            }
            min_sim
        };

        if max_sim_to_group >= min_sim_in_group {
            // Sentence fits the group's coherence level
            current_group.push(i);
        } else {
            // Start a new group
            groups.push(std::mem::take(&mut current_group));
            current_group.push(i);
        }
    }

    // Don't forget the last group
    if !current_group.is_empty() {
        groups.push(current_group);
    }

    groups
}

/// Split a Markdown heading-section chunk into semantic segments.
///
/// Segments are Markdown-structure-aware:
/// - Fenced code blocks (``` or ~~~) are atomic units
/// - Lines within a code block are never split
/// - Prose is split at blank-line (paragraph) boundaries
/// - Each paragraph becomes one segment (no intra-paragraph sentence splitting)
///
/// Returns a Vec of segment strings.
pub fn split_into_segments(content: &str) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut segments: Vec<String> = Vec::new();
    let mut current_segment: Vec<&str> = Vec::new();
    let mut in_code_block = false;

    for line in &lines {
        // Track fenced code blocks
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if in_code_block {
                // End of code block: add closing fence to current segment and emit
                current_segment.push(line);
                in_code_block = false;
                // Emit the code block as one atomic segment
                let text = current_segment.join("\n");
                if !text.trim().is_empty() {
                    segments.push(text);
                }
                current_segment.clear();
                continue;
            } else {
                // Start of code block: emit any accumulated text first
                if !current_segment.is_empty() {
                    let text = current_segment.join("\n");
                    if !text.trim().is_empty() {
                        segments.push(text);
                    }
                    current_segment.clear();
                }
                in_code_block = true;
                current_segment.push(line);
                continue;
            }
        }

        if in_code_block {
            current_segment.push(line);
            continue;
        }

        // Prose: split on blank lines (paragraph boundaries)
        if trimmed.is_empty() {
            if !current_segment.is_empty() {
                let text = current_segment.join("\n");
                if !text.trim().is_empty() {
                    segments.push(text);
                }
                current_segment.clear();
            }
        } else {
            current_segment.push(line);
        }
    }

    // Emit remaining content (including unclosed code blocks)
    if !current_segment.is_empty() {
        let text = current_segment.join("\n");
        if !text.trim().is_empty() {
            segments.push(text);
        }
    }

    segments
}

/// Reassemble grouped segments into sub-chunk content strings.
///
/// Each group of segment indices produces one sub-chunk string.
/// Groups are separated by blank lines for readability.
pub fn reassemble_groups(segments: &[String], groups: &[Vec<usize>]) -> Vec<String> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .filter_map(|&idx| segments.get(idx))
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect()
}

/// Check if any sub-chunk exceeds the token budget and split further if needed.
/// Uses simple chars/4 token estimation (matching the pipeline's CharDivFourEstimator).
/// Recursion is depth-limited to prevent infinite loops on pathological input
/// (e.g., single very long line with no newlines).
pub fn enforce_token_budget(sub_chunks: Vec<String>, max_tokens: usize) -> Vec<String> {
    let mut result = Vec::new();
    for chunk in sub_chunks {
        enforce_single(&mut result, chunk, max_tokens, 0);
    }
    result
}

/// Maximum recursion depth for budget enforcement splitting.
const MAX_BUDGET_SPLIT_DEPTH: usize = 10;

fn enforce_single(result: &mut Vec<String>, chunk: String, max_tokens: usize, depth: usize) {
    let tokens = chunk.len() / 4;
    if tokens <= max_tokens || max_tokens == 0 || depth >= MAX_BUDGET_SPLIT_DEPTH {
        result.push(chunk);
        return;
    }

    let mid = chunk.len() / 2;
    let split_point = chunk[..mid]
        .rfind("\n\n")
        .map(|p| p + 2)
        .or_else(|| chunk[..mid].rfind('\n').map(|p| p + 1))
        .unwrap_or(mid);

    let (first, second) = chunk.split_at(split_point);
    let first = first.trim().to_string();
    let second = second.trim().to_string();

    if !first.is_empty() {
        enforce_single(result, first, max_tokens, depth + 1);
    }
    if !second.is_empty() {
        enforce_single(result, second, max_tokens, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SimilaritySource + Max-Min algorithm tests
    // -----------------------------------------------------------------------

    /// Mock similarity source with a pre-defined matrix.
    struct MockSimilarity {
        matrix: Vec<Vec<f32>>,
    }

    impl SimilaritySource for MockSimilarity {
        fn similarity(&self, i: usize, j: usize) -> f32 {
            self.matrix[i][j]
        }
    }

    #[test]
    fn test_max_min_empty() {
        let sim = MockSimilarity { matrix: vec![] };
        let groups = max_min_split(0, &sim);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_max_min_single_sentence() {
        let sim = MockSimilarity {
            matrix: vec![vec![1.0]],
        };
        let groups = max_min_split(1, &sim);
        assert_eq!(groups, vec![vec![0]]);
    }

    #[test]
    fn test_max_min_two_similar_sentences() {
        // High similarity -> same group
        let sim = MockSimilarity {
            matrix: vec![
                vec![1.0, 0.9],
                vec![0.9, 1.0],
            ],
        };
        let groups = max_min_split(2, &sim);
        assert_eq!(groups, vec![vec![0, 1]]);
    }

    #[test]
    fn test_max_min_two_sentences_always_group() {
        // With only 2 sentences, they always form one group (no coherence
        // baseline exists for a single-sentence group). The split decision
        // requires at least 2 sentences in the group to establish a floor.
        let sim = MockSimilarity {
            matrix: vec![
                vec![1.0, 0.1],
                vec![0.1, 1.0],
            ],
        };
        let groups = max_min_split(2, &sim);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0, 1]);
    }

    #[test]
    fn test_max_min_topic_shift() {
        // Sentences 0,1,2 are about topic A (high mutual similarity)
        // Sentence 3 shifts to topic B (low similarity to A)
        // Sentence 4 continues topic B
        //
        // Group starts: [0], sentence 1 joins (single-sentence group, always joins) -> [0,1]
        // Sentence 2: min_sim_in_group = sim(0,1) = 0.90. max_sim_to_group = max(0.88, 0.91) = 0.91.
        // 0.91 >= 0.90 -> joins -> [0,1,2]
        // Sentence 3: min_sim_in_group = min(0.90, 0.88, 0.91) = 0.88.
        // max_sim_to_group = max(0.10, 0.12, 0.15) = 0.15. 0.15 < 0.88 -> new group [3]
        // Sentence 4: single-sentence group, always joins -> [3,4]
        let sim = MockSimilarity {
            matrix: vec![
                //    0     1     2     3     4
                vec![1.0,  0.90, 0.88, 0.10, 0.08], // 0: topic A
                vec![0.90, 1.0,  0.91, 0.12, 0.09], // 1: topic A
                vec![0.88, 0.91, 1.0,  0.15, 0.11], // 2: topic A
                vec![0.10, 0.12, 0.15, 1.0,  0.92], // 3: topic B
                vec![0.08, 0.09, 0.11, 0.92, 1.0],  // 4: topic B
            ],
        };
        let groups = max_min_split(5, &sim);
        assert_eq!(groups.len(), 2, "Should split into 2 topic groups, got {:?}", groups);
        assert_eq!(groups[0], vec![0, 1, 2], "First group should be topic A");
        assert_eq!(groups[1], vec![3, 4], "Second group should be topic B");
    }

    #[test]
    fn test_max_min_three_topics() {
        // Three distinct topic clusters with tight intra-cluster similarity
        // and low inter-cluster similarity.
        // Cluster A: [0,1], B: [2,3], C: [4,5]
        let sim = MockSimilarity {
            matrix: vec![
                //    0     1     2     3     4     5
                vec![1.0,  0.92, 0.10, 0.08, 0.05, 0.06], // 0: topic A
                vec![0.92, 1.0,  0.11, 0.09, 0.06, 0.07], // 1: topic A
                vec![0.10, 0.11, 1.0,  0.90, 0.08, 0.09], // 2: topic B
                vec![0.08, 0.09, 0.90, 1.0,  0.07, 0.10], // 3: topic B
                vec![0.05, 0.06, 0.08, 0.07, 1.0,  0.91], // 4: topic C
                vec![0.06, 0.07, 0.09, 0.10, 0.91, 1.0],  // 5: topic C
            ],
        };
        let groups = max_min_split(6, &sim);
        assert_eq!(groups.len(), 3, "Should split into 3 groups, got {:?}", groups);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2, 3]);
        assert_eq!(groups[2], vec![4, 5]);
    }

    // -----------------------------------------------------------------------
    // EmbeddingSimilarity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_similarity_identical() {
        let sim = EmbeddingSimilarity::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ]);
        let score = sim.similarity(0, 1);
        assert!((score - 1.0).abs() < 0.001, "Identical vectors should have similarity ~1.0");
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let sim = EmbeddingSimilarity::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]);
        let score = sim.similarity(0, 1);
        assert!(score.abs() < 0.001, "Orthogonal vectors should have similarity ~0.0");
    }

    #[test]
    fn test_embedding_similarity_out_of_bounds() {
        let sim = EmbeddingSimilarity::new(vec![vec![1.0, 0.0]]);
        assert_eq!(sim.similarity(0, 5), 0.0);
    }

    // -----------------------------------------------------------------------
    // Sentence splitting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_paragraphs() {
        let content = "First paragraph about HNSW.\n\nSecond paragraph about graph expansion.\n\nThird about agents.";
        let segments = split_into_segments(content);
        assert_eq!(segments.len(), 3);
        assert!(segments[0].contains("HNSW"));
        assert!(segments[1].contains("graph"));
        assert!(segments[2].contains("agents"));
    }

    #[test]
    fn test_split_preserves_code_blocks() {
        let content = "Some intro text.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nMore text after.";
        let segments = split_into_segments(content);
        assert_eq!(segments.len(), 3, "Should have intro, code block, and outro");
        assert!(segments[1].contains("fn main()"), "Code block should be atomic");
        assert!(segments[1].contains("```"), "Code block should include fences");
    }

    #[test]
    fn test_split_empty_content() {
        let segments = split_into_segments("");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_split_single_line() {
        let segments = split_into_segments("Just one line.");
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_split_preserves_tilde_code_blocks() {
        let content = "Some intro.\n\n~~~python\ndef hello():\n    print('hi')\n~~~\n\nAfter code.";
        let segments = split_into_segments(content);
        assert_eq!(segments.len(), 3, "Should have intro, tilde code block, and outro");
        assert!(segments[1].contains("def hello()"), "Tilde code block should be atomic");
        assert!(segments[1].contains("~~~"), "Tilde code block should include fences");
    }

    #[test]
    fn test_split_heading_preserved() {
        let content = "## Architecture\n\nThe system uses HNSW for vector search.\n\nIt also supports graph expansion.";
        let segments = split_into_segments(content);
        assert_eq!(segments.len(), 3);
        assert!(segments[0].starts_with("## Architecture"));
    }

    // -----------------------------------------------------------------------
    // Token budget enforcement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_budget_enforcement_fits() {
        let chunks = vec!["Short chunk.".to_string()];
        let result = enforce_token_budget(chunks, 512);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_budget_enforcement_splits_oversized() {
        // Create a chunk that is ~200 tokens (800 chars)
        let long_text = "A ".repeat(400); // 800 chars = ~200 tokens
        let chunks = vec![long_text];
        let result = enforce_token_budget(chunks, 100); // 100 token budget
        assert!(result.len() >= 2, "Should split oversized chunk, got {}", result.len());
        for chunk in &result {
            let tokens = chunk.len() / 4;
            assert!(tokens <= 100, "Sub-chunk should fit budget: {} tokens", tokens);
        }
    }

    // -----------------------------------------------------------------------
    // Reassembly tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reassemble_groups() {
        let segments = vec![
            "Sentence one.".to_string(),
            "Sentence two.".to_string(),
            "Sentence three.".to_string(),
            "Sentence four.".to_string(),
        ];
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = reassemble_groups(&segments, &groups);
        assert_eq!(result.len(), 2);
        assert!(result[0].contains("one") && result[0].contains("two"));
        assert!(result[1].contains("three") && result[1].contains("four"));
    }
}
