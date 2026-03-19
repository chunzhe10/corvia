//! Central chunking orchestrator for the Corvia ingestion pipeline (D65/D66).
//!
//! [`ChunkingPipeline`] applies a six-step process to every source file:
//!
//! 1. **Resolve strategy** via [`FormatRegistry`] (adapter override > kernel default > fallback)
//! 2. **Domain chunking** — delegate to the resolved [`ChunkingStrategy`]
//! 3. **Merge small chunks** — strategy-specific or kernel default (adjacent, same merge group)
//! 4. **Enforce token budget** — recursive splitting of oversized chunks
//! 5. **Add overlap context** — strategy-specific or kernel default (trailing chars from prev chunk)
//! 6. **Build [`ProcessedChunk`]s** — attach token estimates and processing metadata
//!
//! The [`FormatRegistry`] routes file extensions to strategies with a three-level
//! priority: adapter overrides > kernel defaults > fallback (D66).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use corvia_common::config::ChunkingConfig;
use corvia_common::errors::Result;
use tracing::info;

use crate::chunking_strategy::{
    ChunkRelation, ChunkResult, ChunkingStrategy, ProcessedChunk, ProcessingInfo, RawChunk,
    SourceFile, SourceMetadata,
};
use crate::process_adapter::ProcessAdapter;
use crate::token_estimator::TokenEstimator;

// ---------------------------------------------------------------------------
// FormatRegistry
// ---------------------------------------------------------------------------

/// Routes file extensions to [`ChunkingStrategy`] implementations.
///
/// Resolution priority (D66): adapter override > kernel default > fallback.
pub struct FormatRegistry {
    defaults: HashMap<String, Arc<dyn ChunkingStrategy>>,
    overrides: HashMap<String, Arc<dyn ChunkingStrategy>>,
    fallback: Arc<dyn ChunkingStrategy>,
}

impl FormatRegistry {
    /// Create a new registry with the given fallback strategy.
    ///
    /// The fallback is used when no default or override matches the extension.
    pub fn new(fallback: Arc<dyn ChunkingStrategy>) -> Self {
        Self {
            defaults: HashMap::new(),
            overrides: HashMap::new(),
            fallback,
        }
    }

    /// Register a kernel default strategy for an extension.
    ///
    /// Kernel defaults are overridden by adapter overrides.
    pub fn register_default(&mut self, ext: &str, strategy: Arc<dyn ChunkingStrategy>) {
        self.defaults.insert(ext.to_string(), strategy);
    }

    /// Register an adapter override strategy for an extension.
    ///
    /// Adapter overrides take highest priority in resolution.
    pub fn register_override(&mut self, ext: &str, strategy: Arc<dyn ChunkingStrategy>) {
        self.overrides.insert(ext.to_string(), strategy);
    }

    /// Resolve the strategy for a file extension.
    ///
    /// Priority: adapter override > kernel default > fallback.
    pub fn resolve(&self, ext: &str) -> Arc<dyn ChunkingStrategy> {
        if let Some(s) = self.overrides.get(ext) {
            return Arc::clone(s);
        }
        if let Some(s) = self.defaults.get(ext) {
            return Arc::clone(s);
        }
        Arc::clone(&self.fallback)
    }
}

// ---------------------------------------------------------------------------
// ProcessingReport / StrategyStats
// ---------------------------------------------------------------------------

/// Aggregate statistics from a batch chunking run.
#[derive(Debug, Default)]
pub struct ProcessingReport {
    pub files_processed: usize,
    pub total_chunks: usize,
    pub chunks_split: usize,
    pub chunks_merged: usize,
    pub per_strategy: HashMap<String, StrategyStats>,
}

/// Per-strategy statistics within a [`ProcessingReport`].
#[derive(Debug, Default)]
pub struct StrategyStats {
    pub files: usize,
    pub chunks: usize,
    pub splits: usize,
    pub merges: usize,
}

// ---------------------------------------------------------------------------
// ChunkingPipeline
// ---------------------------------------------------------------------------

/// Central orchestrator that enforces universal chunking concerns.
///
/// Owns a [`FormatRegistry`] for strategy resolution, a [`TokenEstimator`]
/// for budget enforcement, and a [`ChunkingConfig`] for tuning parameters.
pub struct ChunkingPipeline {
    registry: FormatRegistry,
    estimator: Arc<dyn TokenEstimator>,
    config: ChunkingConfig,
}

impl ChunkingPipeline {
    /// Create a new pipeline.
    pub fn new(
        registry: FormatRegistry,
        estimator: Arc<dyn TokenEstimator>,
        config: ChunkingConfig,
    ) -> Self {
        Self {
            registry,
            estimator,
            config,
        }
    }

    /// Create a pipeline pre-loaded with all kernel default strategies.
    ///
    /// Registers [`FallbackChunker`], [`MarkdownChunker`], [`ConfigChunker`],
    /// and [`PdfChunker`] with the [`FormatRegistry`], using
    /// [`CharDivFourEstimator`] for token estimation.
    pub fn with_kernel_defaults(config: ChunkingConfig) -> Self {
        let fallback = Arc::new(crate::chunking_fallback::FallbackChunker::new(config.max_tokens));
        let mut registry = FormatRegistry::new(fallback);

        let md = Arc::new(crate::chunking_markdown::MarkdownChunker::new());
        for ext in md.supported_extensions() {
            registry.register_default(ext, md.clone());
        }

        let cfg_chunker = Arc::new(crate::chunking_config_fmt::ConfigChunker::new());
        for ext in cfg_chunker.supported_extensions() {
            registry.register_default(ext, cfg_chunker.clone());
        }

        let pdf = Arc::new(crate::chunking_pdf::PdfChunker::new());
        for ext in pdf.supported_extensions() {
            registry.register_default(ext, pdf.clone());
        }

        let estimator = Arc::new(crate::token_estimator::CharDivFourEstimator)
            as Arc<dyn crate::token_estimator::TokenEstimator>;
        Self::new(registry, estimator, config)
    }

    /// Mutable access to the format registry for adapter registration.
    pub fn registry_mut(&mut self) -> &mut FormatRegistry {
        &mut self.registry
    }

    /// Process a single source file through the six-step pipeline.
    ///
    /// Steps: resolve strategy, domain chunk, merge small, split oversized,
    /// add overlap context, build `ProcessedChunk`s.
    ///
    /// Returns processed chunks and any relations discovered by the strategy.
    pub fn process(
        &self,
        source: &str,
        meta: &SourceMetadata,
    ) -> Result<(Vec<ProcessedChunk>, Vec<ChunkRelation>)> {
        // Step 1: Resolve strategy via FormatRegistry.
        let strategy = self.registry.resolve(&meta.extension);
        let strategy_name = strategy.name().to_string();

        info!(
            strategy = %strategy_name,
            file = %meta.file_path,
            "chunking_started"
        );

        // Step 2: Domain chunking.
        let chunk_result = strategy.chunk(source, meta)?;
        let mut chunks = chunk_result.chunks;
        let relations = chunk_result.relations;

        // Step 3: Merge small chunks.
        let (chunks_after_merge, merge_flags_pre_split) =
            self.merge_step(&strategy, &mut chunks);

        // Step 4: Enforce token budget — split oversized chunks.
        let (chunks_after_split, split_flags, merge_count) =
            self.split_step(&strategy, chunks_after_merge, merge_flags_pre_split)?;

        // Step 5: Add overlap context.
        let (final_contents, overlap_tokens_vec) =
            self.overlap_step(&strategy, &chunks_after_split);

        // Step 6: Build ProcessedChunks.
        let processed: Vec<ProcessedChunk> = chunks_after_split
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let original_content = chunk.content.clone();
                let content = final_contents[i].clone();
                let token_estimate = self.estimator.estimate(&content);
                let overlap_toks = overlap_tokens_vec[i];

                ProcessedChunk {
                    content,
                    original_content,
                    chunk_type: chunk.chunk_type,
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    metadata: chunk.metadata,
                    token_estimate,
                    processing: ProcessingInfo {
                        strategy_name: strategy_name.clone(),
                        was_split: split_flags[i],
                        was_merged: merge_count[i],
                        overlap_tokens: overlap_toks,
                    },
                }
            })
            .collect();

        info!(
            strategy = %strategy_name,
            file = %meta.file_path,
            chunks = processed.len(),
            "chunking_completed"
        );

        Ok((processed, relations))
    }

    /// Process a batch of source files, returning all chunks, relations, and a report.
    pub fn process_batch(
        &self,
        files: &[SourceFile],
    ) -> Result<(Vec<ProcessedChunk>, Vec<ChunkRelation>, ProcessingReport)> {
        let mut all_chunks = Vec::new();
        let mut all_relations = Vec::new();
        let mut report = ProcessingReport::default();

        for file in files {
            let (chunks, relations) = self.process(&file.content, &file.metadata)?;

            let strategy_name = if let Some(first) = chunks.first() {
                first.processing.strategy_name.clone()
            } else {
                self.registry.resolve(&file.metadata.extension).name().to_string()
            };

            let stats = report
                .per_strategy
                .entry(strategy_name)
                .or_default();

            stats.files += 1;

            for chunk in &chunks {
                stats.chunks += 1;
                if chunk.processing.was_split {
                    stats.splits += 1;
                    report.chunks_split += 1;
                }
                if chunk.processing.was_merged {
                    stats.merges += 1;
                    report.chunks_merged += 1;
                }
            }

            report.files_processed += 1;
            report.total_chunks += chunks.len();
            all_chunks.extend(chunks);
            all_relations.extend(relations);
        }

        // After all files are processed, discover cross-file relations
        let cross_file_relations = Self::discover_cross_file_relations(&all_chunks);
        if !cross_file_relations.is_empty() {
            tracing::info!(
                count = cross_file_relations.len(),
                "discovered cross-file declaration→usage relations"
            );
            all_relations.extend(cross_file_relations);
        }

        Ok((all_chunks, all_relations, report))
    }

    // -- Internal pipeline steps -------------------------------------------

    /// Step 3: Merge small adjacent chunks.
    ///
    /// First tries the strategy's `merge_small`. If it returns `None`, applies
    /// the kernel default algorithm: merge adjacent chunks that are both below
    /// `min_tokens` if their combined size fits `max_tokens` and they share the
    /// same `merge_group`.
    fn merge_step(
        &self,
        strategy: &Arc<dyn ChunkingStrategy>,
        chunks: &mut Vec<RawChunk>,
    ) -> (Vec<RawChunk>, Vec<bool>) {
        // Try strategy-specific merge first.
        if let Some(merged) = strategy.merge_small(chunks, self.config.max_tokens) {
            let flags = vec![true; merged.len()];
            return (merged, flags);
        }

        // Kernel default merge algorithm.
        let mut result: Vec<RawChunk> = Vec::new();
        let mut merge_flags: Vec<bool> = Vec::new();
        let mut i = 0;

        while i < chunks.len() {
            let current = &chunks[i];
            let current_tokens = self.estimator.estimate(&current.content);

            if current_tokens < self.config.min_tokens && i + 1 < chunks.len() {
                let next = &chunks[i + 1];
                let next_tokens = self.estimator.estimate(&next.content);

                if next_tokens < self.config.min_tokens
                    && Self::same_merge_group(current, next)
                {
                    let combined = format!("{}\n{}", current.content, next.content);
                    let combined_tokens = self.estimator.estimate(&combined);

                    if combined_tokens <= self.config.max_tokens {
                        // Merge current and next.
                        let merged_chunk = RawChunk {
                            content: combined,
                            chunk_type: current.chunk_type.clone(),
                            start_line: current.start_line,
                            end_line: next.end_line,
                            metadata: current.metadata.clone(),
                        };
                        result.push(merged_chunk);
                        merge_flags.push(true);
                        i += 2;
                        continue;
                    }
                }
            }

            result.push(chunks[i].clone());
            merge_flags.push(false);
            i += 1;
        }

        (result, merge_flags)
    }

    /// Check whether two chunks share the same merge group.
    fn same_merge_group(a: &RawChunk, b: &RawChunk) -> bool {
        match (&a.metadata.merge_group, &b.metadata.merge_group) {
            (None, None) => true,
            (Some(ga), Some(gb)) => ga == gb,
            _ => false,
        }
    }

    /// Step 4: Split chunks that exceed the token budget.
    ///
    /// Recursively calls `strategy.split_oversized` until all chunks fit
    /// within `max_tokens` (or cannot be split further).
    fn split_step(
        &self,
        strategy: &Arc<dyn ChunkingStrategy>,
        chunks: Vec<RawChunk>,
        merge_flags: Vec<bool>,
    ) -> Result<(Vec<RawChunk>, Vec<bool>, Vec<bool>)> {
        let mut result = Vec::new();
        let mut split_flags = Vec::new();
        let mut expanded_merge_flags = Vec::new();

        for (idx, chunk) in chunks.into_iter().enumerate() {
            let was_merged = merge_flags[idx];
            let tokens = self.estimator.estimate(&chunk.content);
            if tokens > self.config.max_tokens {
                let split_chunks =
                    self.recursive_split(strategy, &chunk, self.config.max_tokens)?;
                let count = split_chunks.len();
                result.extend(split_chunks);
                // All chunks produced from a split are marked as split.
                split_flags.extend(std::iter::repeat_n(count > 1, count));
                // Propagate the merge flag to all split pieces.
                expanded_merge_flags.extend(std::iter::repeat_n(was_merged, count));
            } else {
                result.push(chunk);
                split_flags.push(false);
                expanded_merge_flags.push(was_merged);
            }
        }

        Ok((result, split_flags, expanded_merge_flags))
    }

    /// Recursively split a chunk until all pieces fit within the budget
    /// or cannot be split further.
    fn recursive_split(
        &self,
        strategy: &Arc<dyn ChunkingStrategy>,
        chunk: &RawChunk,
        max_tokens: usize,
    ) -> Result<Vec<RawChunk>> {
        let tokens = self.estimator.estimate(&chunk.content);
        if tokens <= max_tokens {
            return Ok(vec![chunk.clone()]);
        }

        let pieces = strategy.split_oversized(chunk, max_tokens)?;

        // Guard against infinite recursion: if split returned a single chunk
        // that is still oversized, we cannot split further.
        if pieces.len() <= 1 {
            return Ok(pieces);
        }

        let mut result = Vec::new();
        for piece in &pieces {
            let sub = self.recursive_split(strategy, piece, max_tokens)?;
            result.extend(sub);
        }

        Ok(result)
    }

    /// Step 5: Add overlap context between consecutive chunks.
    ///
    /// First tries `strategy.overlap_context`. If it returns `None` and
    /// `overlap_tokens > 0`, uses the kernel default: last N characters from
    /// the previous chunk (where N = overlap_tokens * 4).
    fn overlap_step(
        &self,
        strategy: &Arc<dyn ChunkingStrategy>,
        chunks: &[RawChunk],
    ) -> (Vec<String>, Vec<usize>) {
        let mut contents = Vec::with_capacity(chunks.len());
        let mut overlap_tokens_vec = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 || self.config.overlap_tokens == 0 {
                // First chunk or overlap disabled — no prefix.
                contents.push(chunk.content.clone());
                overlap_tokens_vec.push(0);
                continue;
            }

            let prev = &chunks[i - 1];

            // Try strategy-specific overlap.
            if let Some(overlap_text) = strategy.overlap_context(prev, chunk) {
                let overlap_est = self.estimator.estimate(&overlap_text);
                let with_overlap = format!("{}\n{}", overlap_text, chunk.content);
                contents.push(with_overlap);
                overlap_tokens_vec.push(overlap_est);
            } else {
                // Kernel default: last N chars from prev chunk.
                let char_budget = self.config.overlap_tokens * 4;
                let prev_content = &prev.content;
                let overlap_text = if prev_content.len() > char_budget {
                    let start = prev_content.len() - char_budget;
                    // Find the nearest char boundary at or after the byte offset.
                    let start = {
                        let mut i = start;
                        while i < prev_content.len() && !prev_content.is_char_boundary(i) {
                            i += 1;
                        }
                        i
                    };
                    &prev_content[start..]
                } else {
                    prev_content.as_str()
                };
                let overlap_est = self.estimator.estimate(overlap_text);
                let with_overlap = format!("{}\n{}", overlap_text, chunk.content);
                contents.push(with_overlap);
                overlap_tokens_vec.push(overlap_est);
            }
        }

        (contents, overlap_tokens_vec)
    }

    /// Discover cross-file relations by tracking type/trait definitions and usages.
    ///
    /// Pass 1: Collect definitions (structs, traits, enums) from chunk metadata.
    /// Pass 2: Find cross-file usages of those definitions in chunk content.
    /// Returns additional `ChunkRelation` entries with relation type `"uses"`.
    pub fn discover_cross_file_relations(
        chunks: &[ProcessedChunk],
    ) -> Vec<ChunkRelation> {
        // Types to skip — common standard library / serde / language primitives.
        let skip_types: HashSet<&str> = [
            "String", "Vec", "HashMap", "HashSet", "BTreeMap", "BTreeSet",
            "Option", "Result", "Box", "Arc", "Mutex", "RwLock", "Rc", "Cell",
            "RefCell", "Pin", "Future", "Stream", "Iterator", "Display", "Debug",
            "Clone", "Copy", "Send", "Sync", "Default", "From", "Into",
            "TryFrom", "TryInto", "AsRef", "AsMut", "Deref", "DerefMut", "Drop",
            "Fn", "FnMut", "FnOnce", "Error", "Ord", "PartialOrd", "Eq",
            "PartialEq", "Hash", "Sized", "Unpin", "Serialize", "Deserialize",
            "None", "Some", "Ok", "Err", "Self", "TRUE", "FALSE", "Path",
            "PathBuf", "File", "Read", "Write", "Seek", "Command", "Output",
            "ExitStatus",
        ]
        .iter()
        .copied()
        .collect();

        // Pass 1: Collect definitions.
        // symbol_name → Vec<(source_file, start_line)>
        let definition_types: HashSet<&str> = [
            "struct_item", "trait_item", "enum_item", "impl_item", "mod_item",
            "struct", "trait", "enum", "impl", "mod",
        ]
        .iter()
        .copied()
        .collect();

        let mut definitions: HashMap<String, Vec<(String, u32)>> = HashMap::new();

        for chunk in chunks {
            if !definition_types.contains(chunk.chunk_type.as_str()) {
                continue;
            }
            if let Some(name) = extract_definition_name(&chunk.content) {
                // Only keep CamelCase names >= 3 chars, not in skip set.
                if name.len() >= 3
                    && name.starts_with(|c: char| c.is_uppercase())
                    && !skip_types.contains(name.as_str())
                {
                    definitions
                        .entry(name)
                        .or_default()
                        .push((
                            chunk.metadata.source_file.clone(),
                            chunk.start_line,
                        ));
                }
            }
        }

        // Filter out symbols defined in 10+ unique files (too generic).
        definitions.retain(|_name, locs| {
            let unique_files: HashSet<&str> =
                locs.iter().map(|(f, _)| f.as_str()).collect();
            unique_files.len() < 10
        });

        // Pass 2: Find cross-file usages.
        let mut seen: HashSet<(String, u32, String, String)> = HashSet::new();
        let mut relations: Vec<ChunkRelation> = Vec::new();

        for chunk in chunks {
            let from_file = &chunk.metadata.source_file;
            let from_line = chunk.start_line;

            for (symbol, def_locs) in &definitions {
                if !content_references_symbol(&chunk.content, symbol) {
                    continue;
                }

                for (def_file, def_line) in def_locs {
                    // Skip self-references (same file + same start_line).
                    if from_file == def_file && from_line == *def_line {
                        continue;
                    }
                    // Skip same-file references.
                    if from_file == def_file {
                        continue;
                    }

                    let key = (
                        from_file.clone(),
                        from_line,
                        def_file.clone(),
                        symbol.clone(),
                    );
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);

                    relations.push(ChunkRelation {
                        from_source_file: from_file.clone(),
                        from_start_line: from_line,
                        relation: "uses".to_string(),
                        to_file: def_file.clone(),
                        to_name: Some(symbol.clone()),
                    });
                }
            }
        }

        relations
    }
}

/// Extract a definition name (struct/trait/enum) from the first line of chunk content.
fn extract_definition_name(content: &str) -> Option<String> {
    let first_line = content.lines().next()?;
    let trimmed = first_line.trim();
    let keywords = ["struct ", "trait ", "enum "];
    for keyword in &keywords {
        if let Some(pos) = trimmed.find(keyword) {
            let after = &trimmed[pos + keyword.len()..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Word-boundary-aware substring search: returns true if `symbol` appears
/// in `content` as a whole word (not part of a longer identifier).
fn content_references_symbol(content: &str, symbol: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = content[start..].find(symbol) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0
            || !content.as_bytes()[abs_pos - 1].is_ascii_alphanumeric()
                && content.as_bytes()[abs_pos - 1] != b'_';
        let after_pos = abs_pos + symbol.len();
        let after_ok = after_pos >= content.len()
            || !content.as_bytes()[after_pos].is_ascii_alphanumeric()
                && content.as_bytes()[after_pos] != b'_';
        if before_ok && after_ok {
            return true;
        }
        start = abs_pos + 1;
    }
    false
}

// ---------------------------------------------------------------------------
// ProcessChunkingStrategy — IPC delegation to adapter processes
// ---------------------------------------------------------------------------

/// A [`ChunkingStrategy`] that delegates chunking to an adapter process via IPC.
///
/// Used by [`register_adapter_chunking`] to register adapter-provided chunking
/// as overrides in the [`FormatRegistry`].
struct ProcessChunkingStrategy {
    adapter: Arc<Mutex<ProcessAdapter>>,
}

impl ProcessChunkingStrategy {
    fn new(adapter: Arc<Mutex<ProcessAdapter>>) -> Self {
        Self { adapter }
    }
}

impl ChunkingStrategy for ProcessChunkingStrategy {
    fn name(&self) -> &str {
        "process_adapter"
    }

    fn supported_extensions(&self) -> &[&str] {
        // This is only used for registration metadata; the registry routes by
        // extension key, not by asking the strategy.
        &[]
    }

    fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<ChunkResult> {
        let mut guard = self.adapter.lock().map_err(|e| {
            corvia_common::errors::CorviaError::Ingestion(format!("Adapter lock poisoned: {e}"))
        })?;
        let (chunks, relations) = guard.chunk(source, meta).map_err(|e| {
            corvia_common::errors::CorviaError::Ingestion(format!("Adapter chunk failed: {e}"))
        })?;
        Ok(ChunkResult { chunks, relations })
    }
}

/// Register an adapter's chunking extensions in the format registry.
///
/// For each extension in `chunking_extensions`, registers a
/// [`ProcessChunkingStrategy`] as an override in the registry.
pub fn register_adapter_chunking(
    registry: &mut FormatRegistry,
    adapter: Arc<Mutex<ProcessAdapter>>,
    chunking_extensions: &[String],
) {
    if chunking_extensions.is_empty() {
        return;
    }
    let strategy = Arc::new(ProcessChunkingStrategy::new(adapter));
    for ext in chunking_extensions {
        registry.register_override(ext, Arc::clone(&strategy) as Arc<dyn ChunkingStrategy>);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking_fallback::FallbackChunker;
    use crate::chunking_strategy::ChunkMetadata;
    use crate::token_estimator::CharDivFourEstimator;

    // -- Helpers -----------------------------------------------------------

    /// Simple strategy that returns each line as a separate chunk.
    struct LineChunker;

    impl ChunkingStrategy for LineChunker {
        fn name(&self) -> &str {
            "line_chunker"
        }

        fn supported_extensions(&self) -> &[&str] {
            &["txt", "log"]
        }

        fn chunk(&self, source: &str, meta: &SourceMetadata) -> Result<crate::chunking_strategy::ChunkResult> {
            let chunks = source
                .lines()
                .enumerate()
                .map(|(i, line)| RawChunk {
                    content: line.to_string(),
                    chunk_type: "line".into(),
                    start_line: (i + 1) as u32,
                    end_line: (i + 1) as u32,
                    metadata: ChunkMetadata {
                        source_file: meta.file_path.clone(),
                        language: meta.language.clone(),
                        ..Default::default()
                    },
                })
                .collect();
            Ok(crate::chunking_strategy::ChunkResult { chunks, relations: vec![] })
        }
    }

    fn test_meta(ext: &str) -> SourceMetadata {
        SourceMetadata {
            file_path: format!("test.{}", ext),
            extension: ext.into(),
            language: None,
            scope_id: "test:scope".into(),
            source_version: "v1".into(),
            workstream: None,
            content_role: None,
            source_origin: None,
        }
    }

    fn make_pipeline(config: ChunkingConfig) -> ChunkingPipeline {
        let fallback: Arc<dyn ChunkingStrategy> =
            Arc::new(FallbackChunker::new(config.max_tokens));
        let registry = FormatRegistry::new(fallback);
        let estimator: Arc<dyn TokenEstimator> = Arc::new(CharDivFourEstimator);
        ChunkingPipeline::new(registry, estimator, config)
    }

    fn make_pipeline_with_line_chunker(config: ChunkingConfig) -> ChunkingPipeline {
        let fallback: Arc<dyn ChunkingStrategy> =
            Arc::new(FallbackChunker::new(config.max_tokens));
        let mut registry = FormatRegistry::new(fallback);
        registry.register_default("txt", Arc::new(LineChunker));
        let estimator: Arc<dyn TokenEstimator> = Arc::new(CharDivFourEstimator);
        ChunkingPipeline::new(registry, estimator, config)
    }

    // -- Registry tests ----------------------------------------------------

    #[test]
    fn test_registry_resolve_fallback() {
        let fallback: Arc<dyn ChunkingStrategy> = Arc::new(FallbackChunker::new(512));
        let registry = FormatRegistry::new(fallback);

        let resolved = registry.resolve("unknown_ext");
        assert_eq!(resolved.name(), "fallback");
    }

    #[test]
    fn test_registry_default_takes_priority_over_fallback() {
        let fallback: Arc<dyn ChunkingStrategy> = Arc::new(FallbackChunker::new(512));
        let mut registry = FormatRegistry::new(fallback);
        registry.register_default("txt", Arc::new(LineChunker));

        let resolved = registry.resolve("txt");
        assert_eq!(resolved.name(), "line_chunker");

        // Other extensions still fall back.
        let other = registry.resolve("rs");
        assert_eq!(other.name(), "fallback");
    }

    #[test]
    fn test_registry_override_takes_priority_over_default() {
        let fallback: Arc<dyn ChunkingStrategy> = Arc::new(FallbackChunker::new(512));
        let mut registry = FormatRegistry::new(fallback);
        registry.register_default("txt", Arc::new(LineChunker));

        // Register an override for "txt" using another strategy.
        let override_strategy: Arc<dyn ChunkingStrategy> = Arc::new(FallbackChunker::new(100));
        registry.register_override("txt", override_strategy);

        let resolved = registry.resolve("txt");
        assert_eq!(
            resolved.name(),
            "fallback",
            "override (FallbackChunker) should beat default (LineChunker)"
        );
    }

    // -- Pipeline tests ----------------------------------------------------

    #[test]
    fn test_process_small_file() {
        let config = ChunkingConfig {
            max_tokens: 512,
            min_tokens: 32,
            overlap_tokens: 0,
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline_with_line_chunker(config);
        let source = "Hello, world! This is a small test file.";
        let meta = test_meta("txt");

        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert_eq!(chunks.len(), 1, "small single-line file should produce 1 chunk");
        assert_eq!(chunks[0].processing.strategy_name, "line_chunker");
        assert!(
            chunks[0].token_estimate > 0,
            "token_estimate should be positive"
        );
        assert_eq!(chunks[0].original_content, "Hello, world! This is a small test file.");
    }

    #[test]
    fn test_process_enforces_token_budget() {
        // Use a very small token budget so the content must be split.
        let config = ChunkingConfig {
            max_tokens: 10,
            min_tokens: 2,
            overlap_tokens: 0,
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline(config);

        // Generate content that exceeds 10 tokens (~40 chars).
        let source = "The quick brown fox jumps over the lazy dog. \
                       This sentence is here to ensure we exceed the budget. \
                       Another line to push the total higher still.";
        let meta = test_meta("xyz"); // unknown ext -> fallback

        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert!(
            chunks.len() > 1,
            "expected multiple chunks after budget enforcement, got {}",
            chunks.len()
        );

        // Every chunk should fit within the budget (with some tolerance for
        // single-line chunks that cannot be split further).
        let estimator = CharDivFourEstimator;
        for chunk in &chunks {
            let est = estimator.estimate(&chunk.content);
            // Allow up to 2x budget for unsplittable single-line chunks.
            assert!(
                est <= 20,
                "chunk token estimate {} exceeds 2x budget for content: {:?}",
                est,
                chunk.content
            );
        }
    }

    #[test]
    fn test_process_merges_small_chunks() {
        // Each line is ~5 chars / 4 = ~1 token. With min_tokens=10, they
        // should be merged.
        let config = ChunkingConfig {
            max_tokens: 100,
            min_tokens: 10,
            overlap_tokens: 0,
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline_with_line_chunker(config);
        let source = "one\ntwo\nthree\nfour\nfive\nsix";
        let meta = test_meta("txt");

        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        // 6 tiny lines should be merged into fewer chunks.
        assert!(
            chunks.len() < 6,
            "expected fewer than 6 chunks after merging, got {}",
            chunks.len()
        );

        // At least one chunk should be marked as merged.
        let merged_count = chunks.iter().filter(|c| c.processing.was_merged).count();
        assert!(
            merged_count > 0,
            "expected at least one merged chunk"
        );
    }

    #[test]
    fn test_process_batch() {
        let config = ChunkingConfig {
            max_tokens: 512,
            min_tokens: 32,
            overlap_tokens: 0,
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline_with_line_chunker(config);

        let files = vec![
            SourceFile {
                content: "Line one\nLine two".into(),
                metadata: test_meta("txt"),
            },
            SourceFile {
                content: "Another file content here.".into(),
                metadata: test_meta("txt"),
            },
        ];

        let (chunks, _relations, report) = pipeline.process_batch(&files).unwrap();
        assert!(
            !chunks.is_empty(),
            "expected chunks from batch processing"
        );
        assert_eq!(report.files_processed, 2);
        assert_eq!(report.total_chunks, chunks.len());
    }

    #[test]
    fn test_process_overlap_context() {
        let config = ChunkingConfig {
            max_tokens: 512,
            min_tokens: 2,
            overlap_tokens: 4, // 4 overlap tokens -> ~16 chars from prev chunk
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline_with_line_chunker(config);
        let source = "First line here with content.\nSecond line here with content.";
        let meta = test_meta("txt");

        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );

        // The second chunk should have overlap applied.
        let second = &chunks[1];
        assert!(
            second.processing.overlap_tokens > 0,
            "expected overlap_tokens > 0 on second chunk"
        );
        // The content should differ from original_content (overlap prefix prepended).
        assert_ne!(
            second.content, second.original_content,
            "content should include overlap prefix"
        );
    }

    #[test]
    fn test_process_sets_processing_info() {
        let config = ChunkingConfig {
            max_tokens: 512,
            min_tokens: 2,
            overlap_tokens: 0,
            strategy: "auto".into(),
        };
        let pipeline = make_pipeline_with_line_chunker(config);
        let source = "Simple content.";
        let meta = test_meta("txt");

        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert_eq!(chunks.len(), 1);

        let info = &chunks[0].processing;
        assert_eq!(info.strategy_name, "line_chunker");
        assert!(!info.was_split, "small chunk should not be split");
        assert!(!info.was_merged, "single chunk should not be merged");
        assert_eq!(info.overlap_tokens, 0);
    }

    // -- with_kernel_defaults tests ----------------------------------------

    fn default_config() -> ChunkingConfig {
        ChunkingConfig {
            max_tokens: 512,
            min_tokens: 32,
            overlap_tokens: 0,
            strategy: "auto".into(),
        }
    }

    #[test]
    fn test_with_kernel_defaults_routes_md() {
        let config = default_config();
        let pipeline = ChunkingPipeline::with_kernel_defaults(config);
        // Use different heading depths so merge_small won't combine them
        // (they have different merge groups: h1 vs h2).
        let source = "# Hello\n\nWorld.\n\n## Goodbye\n\nSee you.";
        let meta = test_meta("md");
        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {} (different heading depths should prevent merging)",
            chunks.len()
        );
        assert_eq!(chunks[0].processing.strategy_name, "markdown");
    }

    #[test]
    fn test_with_kernel_defaults_routes_toml() {
        let config = default_config();
        let pipeline = ChunkingPipeline::with_kernel_defaults(config);
        let source = "[package]\nname = \"test\"\n\n[deps]\nserde = \"1\"\n";
        let meta = test_meta("toml");
        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert_eq!(chunks[0].processing.strategy_name, "config");
    }

    #[test]
    fn test_with_kernel_defaults_routes_unknown_to_fallback() {
        let config = default_config();
        let pipeline = ChunkingPipeline::with_kernel_defaults(config);
        let source = "some random content";
        let meta = test_meta("xyz");
        let (chunks, _relations) = pipeline.process(source, &meta).unwrap();
        assert_eq!(chunks[0].processing.strategy_name, "fallback");
    }

    // -- E2E integration test ------------------------------------------------

    #[test]
    fn test_e2e_mixed_format_batch() {
        use crate::chunking_strategy::SourceFile;

        let config = ChunkingConfig {
            max_tokens: 100,
            min_tokens: 5,
            overlap_tokens: 4,
            strategy: "auto".into(),
        };
        let pipeline = ChunkingPipeline::with_kernel_defaults(config);

        let files = vec![
            SourceFile {
                content: "# Title\n\nIntro paragraph.\n\n## Section\n\nContent here.".into(),
                metadata: SourceMetadata {
                    file_path: "README.md".into(),
                    extension: "md".into(),
                    language: Some("markdown".into()),
                    scope_id: "test".into(),
                    source_version: "v1".into(),
                    workstream: None,
                    content_role: None,
                    source_origin: None,
                },
            },
            SourceFile {
                content: "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1\"\n".into(),
                metadata: SourceMetadata {
                    file_path: "Cargo.toml".into(),
                    extension: "toml".into(),
                    language: Some("toml".into()),
                    scope_id: "test".into(),
                    source_version: "v1".into(),
                    workstream: None,
                    content_role: None,
                    source_origin: None,
                },
            },
            SourceFile {
                content: "Just a plain text file with some content.".into(),
                metadata: SourceMetadata {
                    file_path: "notes.txt".into(),
                    extension: "txt".into(),
                    language: None,
                    scope_id: "test".into(),
                    source_version: "v1".into(),
                    workstream: None,
                    content_role: None,
                    source_origin: None,
                },
            },
        ];

        let (chunks, _relations, report) = pipeline.process_batch(&files).unwrap();

        // Verify strategy routing
        let strategies: Vec<&str> = chunks
            .iter()
            .map(|c| c.processing.strategy_name.as_str())
            .collect();
        assert!(
            strategies.contains(&"markdown"),
            "should route .md to MarkdownChunker, got: {:?}",
            strategies
        );
        assert!(
            strategies.contains(&"config"),
            "should route .toml to ConfigChunker, got: {:?}",
            strategies
        );
        assert!(
            strategies.contains(&"fallback"),
            "should route .txt to FallbackChunker, got: {:?}",
            strategies
        );

        // Verify token budget enforcement
        for chunk in &chunks {
            assert!(
                chunk.token_estimate <= 120,
                "chunk from {} has {} tokens, exceeds budget",
                chunk.metadata.source_file,
                chunk.token_estimate
            );
        }

        // Verify report
        assert_eq!(report.files_processed, 3);
        assert!(
            report.total_chunks >= 3,
            "expected at least 3 chunks, got {}",
            report.total_chunks
        );
        assert!(report.per_strategy.contains_key("markdown"));
        assert!(report.per_strategy.contains_key("config"));
        assert!(report.per_strategy.contains_key("fallback"));
    }

    // -- Cross-file relation discovery tests --------------------------------

    /// Helper to build a `ProcessedChunk` for cross-file relation tests.
    fn make_chunk(
        source_file: &str,
        chunk_type: &str,
        start_line: u32,
        content: &str,
    ) -> ProcessedChunk {
        ProcessedChunk {
            content: content.to_string(),
            original_content: content.to_string(),
            chunk_type: chunk_type.to_string(),
            start_line,
            end_line: start_line + 10,
            metadata: ChunkMetadata {
                source_file: source_file.to_string(),
                language: Some("rust".to_string()),
                ..Default::default()
            },
            token_estimate: 100,
            processing: ProcessingInfo {
                strategy_name: "test".to_string(),
                was_split: false,
                was_merged: false,
                overlap_tokens: 0,
            },
        }
    }

    #[test]
    fn test_discover_cross_file_uses() {
        // traits.rs defines QueryableStore, lite_store.rs implements it.
        let chunks = vec![
            make_chunk(
                "src/traits.rs",
                "trait_item",
                1,
                "pub trait QueryableStore {\n    fn query(&self);\n}",
            ),
            make_chunk(
                "src/lite_store.rs",
                "impl_item",
                10,
                "impl QueryableStore for LiteStore {\n    fn query(&self) {}\n}",
            ),
        ];

        let relations = ChunkingPipeline::discover_cross_file_relations(&chunks);

        // Should produce a "uses" relation from lite_store.rs to traits.rs
        assert!(
            !relations.is_empty(),
            "expected at least one cross-file relation"
        );
        let uses = relations
            .iter()
            .find(|r| r.relation == "uses" && r.to_name == Some("QueryableStore".to_string()));
        assert!(uses.is_some(), "expected a 'uses' relation for QueryableStore");
        let r = uses.unwrap();
        assert_eq!(r.from_source_file, "src/lite_store.rs");
        assert_eq!(r.to_file, "src/traits.rs");
    }

    #[test]
    fn test_skip_same_file_references() {
        // Both chunks in the same file — should produce NO relations.
        let chunks = vec![
            make_chunk(
                "src/traits.rs",
                "trait_item",
                1,
                "pub trait QueryableStore {\n    fn query(&self);\n}",
            ),
            make_chunk(
                "src/traits.rs",
                "impl_item",
                20,
                "impl QueryableStore for DefaultStore {}",
            ),
        ];

        let relations = ChunkingPipeline::discover_cross_file_relations(&chunks);
        assert!(
            relations.is_empty(),
            "expected no cross-file relations for same-file references, got {:?}",
            relations
        );
    }

    #[test]
    fn test_skip_std_types() {
        // Chunk defines "String" — should not produce edges even if other files use String.
        let chunks = vec![
            make_chunk(
                "src/string_wrapper.rs",
                "struct_item",
                1,
                "pub struct String {\n    inner: Vec<u8>,\n}",
            ),
            make_chunk(
                "src/main.rs",
                "function_item",
                1,
                "fn main() {\n    let s: String = String::new();\n}",
            ),
        ];

        let relations = ChunkingPipeline::discover_cross_file_relations(&chunks);
        assert!(
            relations.is_empty(),
            "expected no relations for std type 'String', got {:?}",
            relations
        );
    }

    #[test]
    fn test_word_boundary_matching() {
        // "QueryableStore" should match in "impl QueryableStore for Foo"
        assert!(content_references_symbol(
            "impl QueryableStore for Foo",
            "QueryableStore"
        ));

        // "QueryableStore" should NOT match in "NotQueryableStoreButSimilar"
        assert!(!content_references_symbol(
            "NotQueryableStoreButSimilar",
            "QueryableStore"
        ));

        // "LiteStore" should match in "use LiteStore;"
        assert!(content_references_symbol("use LiteStore;", "LiteStore"));

        // "LiteStore" should NOT match in "use LiteStoreExtra;"
        assert!(!content_references_symbol(
            "use LiteStoreExtra;",
            "LiteStore"
        ));
    }
}
