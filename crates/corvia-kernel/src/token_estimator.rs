/// Pluggable token estimation (shared by M3.2 RAG pipeline and M3.3 chunking).
///
/// Default implementation: chars / 4 heuristic. Swap in tiktoken or
/// sentencepiece later without changing consumers.
pub trait TokenEstimator: Send + Sync {
    fn estimate(&self, text: &str) -> usize;
}

/// Default estimator: character count divided by 4.
/// Industry-standard heuristic for English text. Overestimates slightly
/// for code (good — conservative budget enforcement).
pub struct CharDivFourEstimator;

impl TokenEstimator for CharDivFourEstimator {
    fn estimate(&self, text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            (text.len() / 4).max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let est = CharDivFourEstimator;
        assert_eq!(est.estimate(""), 0);
    }

    #[test]
    fn test_short_string() {
        let est = CharDivFourEstimator;
        assert_eq!(est.estimate("hi"), 1);
    }

    #[test]
    fn test_typical_text() {
        let est = CharDivFourEstimator;
        let text = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(est.estimate(text), 11);
    }

    #[test]
    fn test_code_text() {
        let est = CharDivFourEstimator;
        let code = "fn main() {\n    println!(\"hello\");\n}";
        let tokens = est.estimate(code);
        assert!(tokens > 0);
        assert_eq!(tokens, code.len() / 4);
    }

    #[test]
    fn test_object_safety() {
        let est: Box<dyn TokenEstimator> = Box::new(CharDivFourEstimator);
        assert_eq!(est.estimate("test"), 1);
    }
}
