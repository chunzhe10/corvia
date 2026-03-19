//! Configuration file chunking strategy for TOML, YAML, and JSON (D68).
//!
//! [`ConfigChunker`] splits configuration files into section-level chunks by
//! dispatching on the file extension:
//!
//! - **TOML**: splits on `[section]` headers
//! - **YAML**: splits on top-level keys (column-0 lines containing `:`)
//! - **JSON**: splits on top-level object keys (via `serde_json`)
//!
//! Each chunk has `chunk_type = "section"` and carries accurate line tracking.

use corvia_common::errors::Result;

use crate::chunking_strategy::{ChunkMetadata, ChunkResult, ChunkingStrategy, RawChunk, SourceMetadata};

/// Configuration file chunker for TOML, YAML, and JSON.
#[derive(Default)]
pub struct ConfigChunker;

impl ConfigChunker {
    /// Create a new `ConfigChunker`.
    pub fn new() -> Self {
        Self
    }

    /// Build a [`ChunkMetadata`] from the given source metadata.
    fn make_meta(meta: &SourceMetadata) -> ChunkMetadata {
        ChunkMetadata {
            source_file: meta.file_path.clone(),
            language: meta.language.clone(),
            ..Default::default()
        }
    }

    /// Chunk a TOML source by splitting on `[section]` headers.
    ///
    /// Lines matching `^\[` at the start of a line begin a new section.
    /// Content before the first section header (if any) becomes its own chunk.
    fn chunk_toml(source: &str, meta: &SourceMetadata) -> Vec<RawChunk> {
        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        // Find indices of section header lines.
        let section_starts: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.starts_with('['))
            .map(|(i, _)| i)
            .collect();

        // If no section headers, return the whole source as one chunk.
        if section_starts.is_empty() {
            return vec![RawChunk {
                content: source.to_owned(),
                chunk_type: "section".into(),
                start_line: 1,
                end_line: lines.len() as u32,
                metadata: Self::make_meta(meta),
            }];
        }

        let mut chunks = Vec::new();

        // Content before first section header (preamble).
        if section_starts[0] > 0 {
            let preamble: String = lines[..section_starts[0]].join("\n");
            if !preamble.trim().is_empty() {
                chunks.push(RawChunk {
                    content: preamble,
                    chunk_type: "section".into(),
                    start_line: 1,
                    end_line: section_starts[0] as u32,
                    metadata: Self::make_meta(meta),
                });
            }
        }

        // Each section: from section header to the line before the next header (or end).
        for (i, &start) in section_starts.iter().enumerate() {
            let end = if i + 1 < section_starts.len() {
                section_starts[i + 1]
            } else {
                lines.len()
            };

            let content: String = lines[start..end].join("\n");
            chunks.push(RawChunk {
                content,
                chunk_type: "section".into(),
                // Lines are 1-indexed.
                start_line: (start + 1) as u32,
                end_line: end as u32,
                metadata: Self::make_meta(meta),
            });
        }

        chunks
    }

    /// Chunk a YAML source by splitting on top-level keys.
    ///
    /// A top-level key is a non-empty, non-comment line that starts at column 0
    /// and contains `:`. Each key plus all following indented lines form one chunk.
    fn chunk_yaml(source: &str, meta: &SourceMetadata) -> Vec<RawChunk> {
        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        // Find top-level key line indices.
        let key_starts: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| {
                !line.is_empty()
                    && !line.starts_with('#')
                    && !line.starts_with(' ')
                    && !line.starts_with('\t')
                    && line.contains(':')
            })
            .map(|(i, _)| i)
            .collect();

        // No top-level keys found — return whole source as single chunk.
        if key_starts.is_empty() {
            return vec![RawChunk {
                content: source.to_owned(),
                chunk_type: "section".into(),
                start_line: 1,
                end_line: lines.len() as u32,
                metadata: Self::make_meta(meta),
            }];
        }

        let mut chunks = Vec::new();

        // Preamble before first key (comments, blank lines).
        if key_starts[0] > 0 {
            let preamble: String = lines[..key_starts[0]].join("\n");
            if !preamble.trim().is_empty() {
                chunks.push(RawChunk {
                    content: preamble,
                    chunk_type: "section".into(),
                    start_line: 1,
                    end_line: key_starts[0] as u32,
                    metadata: Self::make_meta(meta),
                });
            }
        }

        for (i, &start) in key_starts.iter().enumerate() {
            let end = if i + 1 < key_starts.len() {
                key_starts[i + 1]
            } else {
                lines.len()
            };

            let content: String = lines[start..end].join("\n");
            chunks.push(RawChunk {
                content,
                chunk_type: "section".into(),
                start_line: (start + 1) as u32,
                end_line: end as u32,
                metadata: Self::make_meta(meta),
            });
        }

        chunks
    }

    /// Chunk a JSON source by splitting on top-level object keys.
    ///
    /// Parses with `serde_json`. If the value is an object, each top-level key
    /// becomes a chunk with pretty-printed JSON. If parsing fails or the value
    /// is not an object, the whole source is returned as a single chunk.
    fn chunk_json(source: &str, meta: &SourceMetadata) -> Vec<RawChunk> {
        let parsed: serde_json::Value = match serde_json::from_str(source) {
            Ok(v) => v,
            Err(_) => {
                // Parse failure — return whole source as a single chunk.
                return vec![RawChunk {
                    content: source.to_owned(),
                    chunk_type: "section".into(),
                    start_line: 1,
                    end_line: source.lines().count().max(1) as u32,
                    metadata: Self::make_meta(meta),
                }];
            }
        };

        let obj = match parsed.as_object() {
            Some(o) => o,
            None => {
                // Not an object — return whole source as a single chunk.
                return vec![RawChunk {
                    content: source.to_owned(),
                    chunk_type: "section".into(),
                    start_line: 1,
                    end_line: source.lines().count().max(1) as u32,
                    metadata: Self::make_meta(meta),
                }];
            }
        };

        if obj.is_empty() {
            return Vec::new();
        }

        // Track line positions by building pretty-printed output for each key.
        // Since we reformat, line numbers are synthetic but ordered.
        let mut chunks = Vec::new();
        let mut current_line: u32 = 1;

        for (key, value) in obj {
            // Pretty-print this key-value pair as a small JSON object fragment.
            let pretty_value = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
            let content = format!("{}: {}", serde_json::to_string(key).unwrap_or_else(|_| key.clone()), pretty_value);
            let line_count = content.lines().count().max(1) as u32;

            chunks.push(RawChunk {
                content,
                chunk_type: "section".into(),
                start_line: current_line,
                end_line: current_line + line_count - 1,
                metadata: Self::make_meta(meta),
            });

            current_line += line_count;
        }

        chunks
    }
}

impl ChunkingStrategy for ConfigChunker {
    fn name(&self) -> &str {
        "config"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["toml", "yaml", "yml", "json"]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        if source.is_empty() {
            return Ok(ChunkResult::default());
        }

        let chunks = match meta.extension.as_str() {
            "toml" => Self::chunk_toml(source, meta),
            "yaml" | "yml" => Self::chunk_yaml(source, meta),
            "json" => Self::chunk_json(source, meta),
            _ => {
                // Unknown extension — return whole source as a single chunk.
                vec![RawChunk {
                    content: source.to_owned(),
                    chunk_type: "section".into(),
                    start_line: 1,
                    end_line: source.lines().count().max(1) as u32,
                    metadata: Self::make_meta(meta),
                }]
            }
        };

        Ok(ChunkResult { chunks, relations: vec![] })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn toml_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "config.toml".into(),
            extension: "toml".into(),
            language: Some("toml".into()),
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
            workstream: None,
            content_role: None,
            source_origin: None,
        }
    }

    fn json_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "config.json".into(),
            extension: "json".into(),
            language: Some("json".into()),
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
            workstream: None,
            content_role: None,
            source_origin: None,
        }
    }

    fn yaml_meta() -> SourceMetadata {
        SourceMetadata {
            file_path: "config.yaml".into(),
            extension: "yaml".into(),
            language: Some("yaml".into()),
            scope_id: "org:proj:ws:git:v1".into(),
            source_version: "abc123".into(),
            workstream: None,
            content_role: None,
            source_origin: None,
        }
    }

    #[test]
    fn test_toml_section_splitting() {
        let chunker = ConfigChunker::new();
        let source = "[package]\nname=\"test\"\n\n[deps]\nserde=\"1\"";
        let chunks = chunker.chunk(source, &toml_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}: {:?}",
            chunks.len(),
            chunks.iter().map(|c| &c.content).collect::<Vec<_>>()
        );

        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("");
        assert!(all_content.contains("[package]"), "should contain [package]");
        assert!(all_content.contains("[deps]"), "should contain [deps]");
    }

    #[test]
    fn test_json_top_level_keys() {
        let chunker = ConfigChunker::new();
        let source = r#"{"name": "test", "version": "1.0", "deps": {"serde": "1"}}"#;
        let chunks = chunker.chunk(source, &json_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks for 3 top-level keys, got {}: {:?}",
            chunks.len(),
            chunks.iter().map(|c| &c.content).collect::<Vec<_>>()
        );

        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("name"), "should contain 'name' key");
        assert!(all_content.contains("version"), "should contain 'version' key");
        assert!(all_content.contains("deps"), "should contain 'deps' key");
    }

    #[test]
    fn test_yaml_top_level_keys() {
        let chunker = ConfigChunker::new();
        let source = "name: test\nversion: \"1.0\"\ndeps:\n  serde: \"1\"\n  toml: \"0.8\"";
        let chunks = chunker.chunk(source, &yaml_meta()).unwrap().chunks;

        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks for 3 top-level keys, got {}: {:?}",
            chunks.len(),
            chunks.iter().map(|c| &c.content).collect::<Vec<_>>()
        );

        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("name"), "should contain 'name' key");
        assert!(all_content.contains("deps"), "should contain 'deps' key");
    }

    #[test]
    fn test_empty_config() {
        let chunker = ConfigChunker::new();
        let chunks = chunker.chunk("", &toml_meta()).unwrap().chunks;
        assert!(chunks.is_empty(), "expected empty vec for empty source");
    }

    #[test]
    fn test_chunk_type_is_section() {
        let chunker = ConfigChunker::new();
        let source = "[package]\nname=\"test\"";
        let chunks = chunker.chunk(source, &toml_meta()).unwrap().chunks;

        for chunk in &chunks {
            assert_eq!(chunk.chunk_type, "section", "all chunks should be type 'section'");
        }
    }

    #[test]
    fn test_name_and_extensions() {
        let chunker = ConfigChunker::new();
        assert_eq!(chunker.name(), "config");
        assert_eq!(chunker.supported_extensions(), &["toml", "yaml", "yml", "json"]);
    }
}
