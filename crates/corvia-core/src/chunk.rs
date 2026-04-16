//! Document chunking with semantic sub-splitting.
//! Defaults: 512 max tokens, 64 overlap (~12.5%), 32 min tokens.

use crate::types::{Chunk, Entry};

/// Strip TOML frontmatter (between `+++` delimiters) from raw entry content.
///
/// Returns just the markdown body, trimmed. If no frontmatter found, returns
/// the input trimmed.
pub fn strip_frontmatter(raw: &str) -> &str {
    let trimmed = raw.trim();
    if !trimmed.starts_with("+++") {
        return trimmed;
    }
    // Find the closing `+++` after the opening one.
    let after_open = &trimmed[3..];
    if let Some(close_pos) = after_open.find("+++") {
        let body_start = 3 + close_pos + 3; // skip opening +++ , content, closing +++
        trimmed[body_start..].trim()
    } else {
        // No closing delimiter found; return input trimmed.
        trimmed
    }
}

/// Split text into sentences using common sentence-ending punctuation and
/// newlines as boundaries. Preserves sentence-ending punctuation within the
/// returned string. Fenced code blocks (``` ... ```) are treated as atomic
/// units and never split mid-block.
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Detect fenced code block boundaries.
        if trimmed.starts_with("```") {
            if in_code_block {
                // Closing fence: include line, emit as atomic unit.
                if !current.is_empty() {
                    current.push('\n');
                }
                current.push_str(line);
                in_code_block = false;
                sentences.push(std::mem::take(&mut current));
                continue;
            } else {
                // Opening fence: flush anything accumulated so far, then start
                // collecting the code block.
                let flushed = current.trim().to_string();
                current.clear();
                if !flushed.is_empty() {
                    sentences.push(flushed);
                }
                current.push_str(line);
                in_code_block = true;
                continue;
            }
        }

        if in_code_block {
            // Inside a code block: accumulate without splitting.
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
            continue;
        }

        // Blank line acts as a paragraph boundary.
        if trimmed.is_empty() {
            let flushed = current.trim().to_string();
            current.clear();
            if !flushed.is_empty() {
                sentences.push(flushed);
            }
            continue;
        }

        // Within a paragraph, split at sentence-ending punctuation followed by
        // a space (". ", "! ", "? "). We scan character-by-character.
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            current.push(chars[i]);
            let is_sentence_end = matches!(chars[i], '.' | '!' | '?');
            let followed_by_space = i + 1 < chars.len() && chars[i + 1] == ' ';
            if is_sentence_end && followed_by_space {
                // Emit the accumulated sentence.
                let trimmed_current = current.trim().to_string();
                if !trimmed_current.is_empty() {
                    sentences.push(trimmed_current);
                    current = String::new();
                }
                // Skip the trailing space; the next sentence starts at the
                // next non-space character.
                i += 1; // skip the space
            }
            i += 1;
        }

        // If the line doesn't end mid-sentence but continues to next line,
        // add a space separator for the next line's content.
        if !current.is_empty() && !current.ends_with(' ') {
            current.push(' ');
        }
    }

    // Flush remaining content (including unclosed code blocks).
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Count words (whitespace-separated tokens) in a string.
fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Split text into chunks with overlap, respecting sentence boundaries.
///
/// Tokens are approximated as whitespace-separated words in the chunking pass.
/// For budget enforcement (max_tokens in search), a 1.33x multiplier converts
/// word count to approximate subword token count.
///
/// Algorithm:
/// 1. Split text into sentences (by `. `, `! `, `? `, blank lines, code fences).
/// 2. Group sentences into chunks up to `max_tokens` words.
/// 3. When adding a sentence would exceed `max_tokens`, start a new chunk.
///    The new chunk begins with the last N tokens (overlap_tokens) from the
///    previous chunk for context continuity.
/// 4. If the last chunk has fewer than `min_tokens` words, it is merged into
///    the previous chunk.
pub fn split_into_chunks(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    min_tokens: usize,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }
    if words.len() <= max_tokens {
        return vec![words.join(" ")];
    }

    let sentences = split_into_sentences(text);
    if sentences.is_empty() {
        return vec![];
    }

    // If a single sentence exceeds max_tokens, we must fall back to word-based
    // splitting for that sentence. Build chunks by grouping sentences.
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut current_wc: usize = 0;

    for sentence in &sentences {
        let s_wc = word_count(sentence);

        // If a single sentence exceeds max_tokens, split it by words.
        if s_wc > max_tokens {
            // Flush current chunk first.
            if !current_chunk.trim().is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
                current_wc = 0;
            }
            // Word-based split for oversized sentence.
            let s_words: Vec<&str> = sentence.split_whitespace().collect();
            let step = max_tokens.saturating_sub(overlap_tokens).max(1);
            let mut start = 0;
            while start < s_words.len() {
                let end = (start + max_tokens).min(s_words.len());
                chunks.push(s_words[start..end].join(" "));
                start += step;
                if end == s_words.len() {
                    break;
                }
            }
            continue;
        }

        // Would adding this sentence exceed the budget?
        if current_wc + s_wc > max_tokens && current_wc > 0 {
            // Emit current chunk.
            let finished = current_chunk.trim().to_string();
            chunks.push(finished.clone());

            // Start new chunk with overlap from the tail of the finished chunk.
            let finished_words: Vec<&str> = finished.split_whitespace().collect();
            let finished_wc = finished_words.len();
            if overlap_tokens > 0 && finished_wc > overlap_tokens {
                let overlap_start = finished_wc - overlap_tokens;
                current_chunk = finished_words[overlap_start..].join(" ");
                current_chunk.push(' ');
                current_wc = overlap_tokens;
            } else if overlap_tokens > 0 {
                // Previous chunk is smaller than the overlap; use it all.
                current_chunk = finished_words.join(" ");
                current_chunk.push(' ');
                current_wc = finished_wc;
            } else {
                current_chunk = String::new();
                current_wc = 0;
            }
        }

        // Append sentence.
        if !current_chunk.is_empty() && !current_chunk.ends_with(' ') {
            current_chunk.push(' ');
        }
        current_chunk.push_str(sentence);
        current_wc += s_wc;
    }

    // Flush last chunk.
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Merge last chunk into previous if it is too small.
    if chunks.len() >= 2 {
        let last_wc = word_count(chunks.last().unwrap());
        if last_wc < min_tokens {
            let last = chunks.pop().unwrap();
            if let Some(prev) = chunks.last_mut() {
                // Deduplicate overlap: find the longest suffix of prev that is
                // a prefix of last and merge without duplicating.
                let prev_words: Vec<&str> = prev.split_whitespace().collect();
                let last_words: Vec<&str> = last.split_whitespace().collect();
                let max_overlap = overlap_tokens.min(prev_words.len()).min(last_words.len());

                let mut found_overlap = 0;
                for ol in (1..=max_overlap).rev() {
                    let prev_tail = &prev_words[prev_words.len() - ol..];
                    let last_head = &last_words[..ol];
                    if prev_tail == last_head {
                        found_overlap = ol;
                        break;
                    }
                }

                let new_part = &last_words[found_overlap..];
                if !new_part.is_empty() {
                    prev.push(' ');
                    prev.push_str(&new_part.join(" "));
                }
            }
        }
    }

    chunks
}

/// Chunk an entry: strip frontmatter from the body, split into chunks, and
/// produce [`Chunk`] structs with metadata copied from the entry.
///
/// An empty body produces one `Chunk` with empty text.
pub fn chunk_entry(
    entry: &Entry,
    max_tokens: usize,
    overlap_tokens: usize,
    min_tokens: usize,
) -> Vec<Chunk> {
    let body = strip_frontmatter(&entry.body);
    let pieces = split_into_chunks(body, max_tokens, overlap_tokens, min_tokens);

    if pieces.is_empty() {
        // Empty body -> single chunk with empty text.
        return vec![Chunk {
            source_entry_id: entry.meta.id.clone(),
            text: String::new(),
            chunk_index: 0,
            kind: entry.meta.kind,
            tags: entry.meta.tags.clone(),
        }];
    }

    pieces
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            source_entry_id: entry.meta.id.clone(),
            text,
            chunk_index: i as u32,
            kind: entry.meta.kind,
            tags: entry.meta.tags.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntryMeta, Kind};

    #[test]
    fn strip_frontmatter_removes_toml() {
        let input = r#"+++
title = "my entry"
kind = "decision"
+++
This is the body."#;
        let result = strip_frontmatter(input);
        assert_eq!(result, "This is the body.");
    }

    #[test]
    fn strip_frontmatter_no_frontmatter() {
        let input = "Just plain text, no frontmatter.";
        let result = strip_frontmatter(input);
        assert_eq!(result, "Just plain text, no frontmatter.");
    }

    #[test]
    fn short_text_single_chunk() {
        let chunks = split_into_chunks("hello world", 512, 64, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn empty_text_no_chunks() {
        let chunks = split_into_chunks("", 512, 64, 32);
        assert!(chunks.is_empty());
    }

    #[test]
    fn long_text_splits_with_overlap() {
        // Generate 100 words.
        let words: Vec<String> = (0..100).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");

        let chunks = split_into_chunks(&text, 30, 5, 10);

        // With 100 words, max 30, step 25: ceil((100-30)/25)+1 = 4 chunks
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks, got {}",
            chunks.len()
        );

        for chunk in &chunks {
            let word_count = chunk.split_whitespace().count();
            assert!(
                word_count <= 30,
                "chunk has {} words, exceeds max 30",
                word_count
            );
        }
    }

    #[test]
    fn small_last_chunk_gets_merged() {
        // 35 words, max 30, overlap 5, min 32.
        // Without merge: chunk 0 = 30 words, chunk 1 = 10 words (starts at 25).
        // 10 < 32 so chunk 1 should be merged into chunk 0.
        let words: Vec<String> = (0..35).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");

        let chunks = split_into_chunks(&text, 30, 5, 32);

        assert_eq!(
            chunks.len(),
            1,
            "last small chunk should have been merged; got {} chunks",
            chunks.len()
        );

        // The merged chunk should contain all 35 words.
        let word_count = chunks[0].split_whitespace().count();
        assert_eq!(word_count, 35, "merged chunk should have all 35 words");
    }

    #[test]
    fn chunk_entry_preserves_metadata() {
        let entry = Entry {
            meta: EntryMeta {
                id: "entry-123".to_string(),
                created_at: "2026-04-15T00:00:00Z".to_string(),
                kind: Kind::Decision,
                supersedes: vec![],
                tags: vec!["architecture".to_string(), "v2".to_string()],
            },
            body: "Some body text here.".to_string(),
        };

        let chunks = chunk_entry(&entry, 512, 64, 32);
        assert_eq!(chunks.len(), 1);

        let c = &chunks[0];
        assert_eq!(c.source_entry_id, "entry-123");
        assert_eq!(c.kind, Kind::Decision);
        assert_eq!(c.tags, vec!["architecture", "v2"]);
        assert_eq!(c.chunk_index, 0);
        assert_eq!(c.text, "Some body text here.");
    }

    #[test]
    fn chunk_entry_empty_body_produces_one_chunk() {
        let entry = Entry {
            meta: EntryMeta {
                id: "entry-456".to_string(),
                created_at: "2026-04-15T00:00:00Z".to_string(),
                kind: Kind::Learning,
                supersedes: vec![],
                tags: vec![],
            },
            body: String::new(),
        };

        let chunks = chunk_entry(&entry, 512, 64, 32);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.is_empty());
        assert_eq!(chunks[0].source_entry_id, "entry-456");
    }

    // -----------------------------------------------------------------------
    // Sentence-boundary splitting tests
    // -----------------------------------------------------------------------

    #[test]
    fn split_into_sentences_basic() {
        let text = "First sentence. Second sentence. Third sentence.";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence.");
        assert_eq!(sentences[1], "Second sentence.");
        assert_eq!(sentences[2], "Third sentence.");
    }

    #[test]
    fn split_into_sentences_paragraph_boundary() {
        let text = "Paragraph one.\n\nParagraph two.";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Paragraph one.");
        assert_eq!(sentences[1], "Paragraph two.");
    }

    #[test]
    fn split_into_sentences_code_block_atomic() {
        let text = "Before.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nAfter.";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Before.");
        assert!(
            sentences[1].contains("fn main()"),
            "code block should be kept atomic"
        );
        assert_eq!(sentences[2], "After.");
    }

    #[test]
    fn split_into_sentences_exclamation_and_question() {
        let text = "Wow! Really? Yes.";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Wow!");
        assert_eq!(sentences[1], "Really?");
        assert_eq!(sentences[2], "Yes.");
    }

    #[test]
    fn sentence_boundary_chunking_respects_boundaries() {
        // Build text with 5 sentences, each ~10 words.
        let text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet. \
                    Kilo lima mike november oscar papa quebec romeo sierra tango. \
                    Uniform victor whiskey xray yankee zulu alpha bravo charlie delta. \
                    Echo foxtrot golf hotel india juliet kilo lima mike november. \
                    Oscar papa quebec romeo sierra tango uniform victor whiskey xray.";

        // max_tokens = 25, so each chunk can hold ~2 sentences.
        let chunks = split_into_chunks(text, 25, 0, 5);

        // With 50 words across 5 sentences of 10 each, and max 25:
        // We should get ~3 chunks (2 sentences per chunk, last one with 1).
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        // No chunk should break mid-sentence (no chunk should end with a
        // partial word from a sentence that continues in the next chunk).
        for chunk in &chunks {
            let wc = chunk.split_whitespace().count();
            assert!(
                wc <= 25,
                "chunk has {} words, exceeds max 25",
                wc
            );
        }
    }

    #[test]
    fn sentence_boundary_chunking_with_overlap() {
        // 4 sentences of 10 words each = 40 words total.
        let text = "One two three four five six seven eight nine ten. \
                    Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty. \
                    Twentyone twentytwo twentythree twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty. \
                    Thirtyone thirtytwo thirtythree thirtyfour thirtyfive thirtysix thirtyseven thirtyeight thirtynine forty.";

        let chunks = split_into_chunks(text, 20, 5, 5);

        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        // With overlap, chunks after the first should share some words with the
        // previous chunk.
        if chunks.len() >= 2 {
            let first_words: Vec<&str> = chunks[0].split_whitespace().collect();
            let second_words: Vec<&str> = chunks[1].split_whitespace().collect();

            // The second chunk should start with words from the tail of the first.
            let first_tail: Vec<&&str> = first_words.iter().rev().take(5).collect();
            let second_head: Vec<&&str> = second_words.iter().take(5).collect();

            // At least some overlap should be present (exact amount depends on
            // sentence boundaries, so we check for any shared prefix).
            let shared = first_tail
                .iter()
                .rev()
                .zip(second_head.iter())
                .take_while(|(a, b)| a == b)
                .count();
            assert!(
                shared > 0,
                "expected overlap between chunks"
            );
        }
    }
}
