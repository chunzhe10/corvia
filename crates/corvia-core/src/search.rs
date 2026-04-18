//! Hybrid search pipeline: BM25 (tantivy) + vector (redb) -> RRF fusion -> rerank.
//! Cross-encoder reranking via fastembed-rs (configurable, default JINA reranker v1 turbo).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use tracing::{debug, info, info_span, warn, Span};

use crate::config::Config;
use crate::embed::Embedder;
use crate::entry::scan_entries;
use crate::index::RedbIndex;
use crate::tantivy_index::TantivyIndex;
use crate::types::{Confidence, Kind, QualitySignal, SearchResponse, SearchResult};

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Parameters for a search query.
pub struct SearchParams {
    pub query: String,
    pub limit: usize,
    pub max_tokens: Option<usize>,
    pub min_score: Option<f32>,
    pub kind: Option<Kind>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A candidate chunk from either BM25 or vector search, before fusion.
#[derive(Debug, Clone)]
struct FusedCandidate {
    chunk_id: String,
    entry_id: String,
    rrf_score: f64,
}

/// Encode parallel `chunk_ids` and `scores` arrays as JSON strings.
///
/// The OTLP file exporter serializes `opentelemetry::Value::Array` via debug
/// formatting (`trace.rs:72`), producing strings that are not machine-parseable.
/// Encoding as JSON strings and recording them as `stringValue` attrs lets
/// downstream consumers parse via `json::parse` cleanly.
///
/// Returns `("[]", "[]")` on any (impossible-for-these-types) serde error
/// so that attr presence is preserved even in unreachable edge cases.
fn encode_stage_scores(chunk_ids: &[String], scores: &[f32]) -> (String, String) {
    let ids = serde_json::to_string(chunk_ids).unwrap_or_else(|_| "[]".to_string());
    let sc = serde_json::to_string(scores).unwrap_or_else(|_| "[]".to_string());
    (ids, sc)
}

/// Record `chunk_ids` and `scores` as JSON-string attrs on the given span.
/// Both fields must have been declared on the `info_span!` with `tracing::field::Empty`.
fn record_stage_scores(span: &tracing::Span, chunk_ids: &[String], scores: &[f32]) {
    let (ids_json, scores_json) = encode_stage_scores(chunk_ids, scores);
    span.record("chunk_ids", ids_json.as_str());
    span.record("scores", scores_json.as_str());
}

// ---------------------------------------------------------------------------
// RRF fusion
// ---------------------------------------------------------------------------

/// Compute RRF (Reciprocal Rank Fusion) scores from multiple ranked lists.
///
/// For each unique chunk_id across all lists, the fused score is:
///   sum(1.0 / (k + rank + 1)) across all lists where the chunk appears.
///
/// `k` is the RRF smoothing constant (higher = less weight to top ranks).
/// `rank` is 0-indexed position in the result list.
fn rrf_fusion(
    bm25_results: &[(String, String, f32)],
    vector_results: &[(String, String, f32)],
    k: u32,
) -> Vec<FusedCandidate> {
    let k = k as f64;

    // Accumulate RRF scores and track entry_id per chunk_id.
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut entry_ids: HashMap<String, String> = HashMap::new();

    // BM25 contributions
    for (rank, (chunk_id, entry_id, _score)) in bm25_results.iter().enumerate() {
        let rrf = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(chunk_id.clone()).or_insert(0.0) += rrf;
        entry_ids
            .entry(chunk_id.clone())
            .or_insert_with(|| entry_id.clone());
    }

    // Vector contributions
    for (rank, (chunk_id, entry_id, _score)) in vector_results.iter().enumerate() {
        let rrf = 1.0 / (k + rank as f64 + 1.0);
        *scores.entry(chunk_id.clone()).or_insert(0.0) += rrf;
        entry_ids
            .entry(chunk_id.clone())
            .or_insert_with(|| entry_id.clone());
    }

    // Build candidate list sorted by descending RRF score.
    let mut candidates: Vec<FusedCandidate> = scores
        .into_iter()
        .map(|(chunk_id, rrf_score)| {
            let entry_id = entry_ids.remove(&chunk_id).unwrap_or_default();
            FusedCandidate {
                chunk_id,
                entry_id,
                rrf_score,
            }
        })
        .collect();

    candidates.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}

// ---------------------------------------------------------------------------
// Quality signal computation
// ---------------------------------------------------------------------------

/// Confidence thresholds (provisional).
const HIGH_SCORE_THRESHOLD: f32 = 0.5;
const MEDIUM_SCORE_THRESHOLD: f32 = 0.2;
const HIGH_MIN_COUNT: usize = 3;

/// Compute a quality signal from the final result scores.
fn compute_quality_signal(scores: &[f32], stale: bool) -> QualitySignal {
    let top_score = scores.first().copied().unwrap_or(0.0);
    let count = scores.len();

    let confidence = if top_score >= HIGH_SCORE_THRESHOLD && count >= HIGH_MIN_COUNT {
        Confidence::High
    } else if top_score >= MEDIUM_SCORE_THRESHOLD {
        Confidence::Medium
    } else if count > 0 {
        Confidence::Low
    } else {
        Confidence::None
    };

    let suggestion = if stale {
        Some("Index may be stale. Run 'corvia ingest' to update.".to_string())
    } else if confidence == Confidence::Low {
        Some("Results may not be relevant. Try rephrasing your query.".to_string())
    } else {
        None
    };

    QualitySignal {
        confidence,
        suggestion,
    }
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Deduplicate candidates by entry_id, keeping only the highest-scoring chunk
/// per entry.
fn deduplicate_by_entry(candidates: &mut Vec<(String, String, f32)>) {
    // Track the best score per entry_id.
    let mut best: HashMap<String, (usize, f32)> = HashMap::new();

    for (i, (_chunk_id, entry_id, score)) in candidates.iter().enumerate() {
        best.entry(entry_id.clone())
            .and_modify(|(idx, best_score)| {
                if *score > *best_score {
                    *idx = i;
                    *best_score = *score;
                }
            })
            .or_insert((i, *score));
    }

    let keep_indices: std::collections::HashSet<usize> =
        best.values().map(|(idx, _)| *idx).collect();

    let mut i = 0;
    candidates.retain(|_| {
        let keep = keep_indices.contains(&i);
        i += 1;
        keep
    });
}

// ---------------------------------------------------------------------------
// Main search function
// ---------------------------------------------------------------------------

/// Run the hybrid search pipeline with pre-opened index handles.
///
/// Use this when the caller holds persistent index handles (e.g. the HTTP MCP server).
/// For one-shot callers, use [`search`] which opens handles internally.
#[tracing::instrument(name = "corvia.search", skip(config, base_dir, embedder, params, redb, tantivy), fields(
    query = tracing::field::Empty,
    query_len = params.query.len(),
    limit = params.limit,
    kind_filter = ?params.kind,
    result_count = tracing::field::Empty,
    result_chunk_ids = tracing::field::Empty,
    confidence = tracing::field::Empty,
))]
pub fn search_with_handles(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: &SearchParams,
    redb: &RedbIndex,
    tantivy: &TantivyIndex,
) -> Result<SearchResponse> {
    // Record raw query for eval mining. Design RFC §4.2: no redaction toggle —
    // corvia is single-user local; raw query is the eval join key.
    Span::current().record("query", params.query.as_str());
    // Step 2: Cold start check.
    let indexed_count_str = redb
        .get_meta("entry_count")
        .context("reading entry_count from redb meta")?;
    let indexed_count: u64 = indexed_count_str
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if indexed_count == 0 {
        info!("cold start: no entries indexed");
        return Ok(SearchResponse {
            results: vec![],
            quality: QualitySignal {
                confidence: Confidence::None,
                suggestion: Some(
                    "No entries indexed. Run 'corvia ingest' first.".to_string(),
                ),
            },
        });
    }

    // Step 3: Drift detection.
    let entries_dir = base_dir.join(config.entries_dir());
    let actual_files = scan_entries(&entries_dir).context("scanning entries for drift detection")?;
    let actual_count = actual_files.len() as u64;
    let stale = actual_count != indexed_count;

    if stale {
        debug!(
            indexed = indexed_count,
            actual = actual_count,
            "index drift detected"
        );
    }

    // Step 4: Oversample if kind filter is set.
    let retrieval_limit = if params.kind.is_some() {
        params.limit * 3
    } else {
        params.limit
    };
    let retrieval_limit = retrieval_limit.max(config.search.reranker_candidates);

    // Step 5: BM25 search.
    let bm25_results = {
        let _span = info_span!(
            "corvia.search.bm25",
            result_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let results = tantivy
            .search(&params.query, params.kind, retrieval_limit)
            .context("BM25 search")?;
        Span::current().record("result_count", results.len());

        // Record chunk_ids + BM25 raw scores for eval mining.
        let ids: Vec<String> = results.iter().map(|(cid, _, _)| cid.clone()).collect();
        let scores_vec: Vec<f32> = results.iter().map(|(_, _, s)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        debug!(count = results.len(), "BM25 results");
        results
    };

    // Step 6: Vector search.
    let vector_scored = {
        let _span = info_span!(
            "corvia.search.vector",
            vector_count = tracing::field::Empty,
            result_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let query_vector = embedder
            .embed(&params.query)
            .context("embedding search query")?;

        let all_vectors = redb.all_vectors().context("loading all vectors from redb")?;
        let superseded_ids = redb.superseded_ids().context("loading superseded IDs")?;
        Span::current().record("vector_count", all_vectors.len());

        let mut scored: Vec<(String, String, f32)> = Vec::new();
        for (chunk_id, vector) in &all_vectors {
            let entry_id = match redb.chunk_entry_id(chunk_id)? {
                Some(eid) => eid,
                None => continue,
            };
            if superseded_ids.contains(&entry_id) {
                continue;
            }
            if let Some(ref kind_filter) = params.kind {
                if let Ok(Some(chunk_kind_str)) = redb.get_chunk_kind(chunk_id) {
                    if let Ok(chunk_kind) = chunk_kind_str.parse::<Kind>() {
                        if chunk_kind != *kind_filter {
                            continue;
                        }
                    }
                }
            }
            let similarity = Embedder::cosine_similarity(&query_vector, vector);
            scored.push((chunk_id.clone(), entry_id, similarity));
        }
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(retrieval_limit);
        Span::current().record("result_count", scored.len());

        // Record chunk_ids + cosine scores for eval mining.
        let ids: Vec<String> = scored.iter().map(|(cid, _, _)| cid.clone()).collect();
        let scores_vec: Vec<f32> = scored.iter().map(|(_, _, s)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        debug!(count = scored.len(), "vector results");
        scored
    };

    // Step 7: RRF fusion.
    let fused = {
        let _span = info_span!(
            "corvia.search.fusion",
            candidate_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();
        let result = rrf_fusion(&bm25_results, &vector_scored, config.search.rrf_k);
        Span::current().record("candidate_count", result.len());

        // Record chunk_ids + RRF scores (f64 → f32) for eval mining.
        let ids: Vec<String> = result.iter().map(|c| c.chunk_id.clone()).collect();
        let scores_vec: Vec<f32> = result.iter().map(|c| c.rrf_score as f32).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        debug!(count = result.len(), "fused candidates");
        result
    };

    // Step 8: Take top reranker_candidates.
    let reranker_count = config.search.reranker_candidates.min(fused.len());
    let top_candidates = &fused[..reranker_count];

    // Step 9: Cross-encoder rerank.
    let mut scored_results = {
        let _span = info_span!(
            "corvia.search.rerank",
            input_count = top_candidates.len(),
            output_count = tracing::field::Empty,
            chunk_ids = tracing::field::Empty,
            scores = tracing::field::Empty,
        )
        .entered();

        let mut candidate_texts: Vec<String> = Vec::with_capacity(top_candidates.len());
        let mut candidate_chunk_ids: Vec<String> = Vec::with_capacity(top_candidates.len());
        let mut candidate_entry_ids: Vec<String> = Vec::with_capacity(top_candidates.len());

        for candidate in top_candidates {
            match tantivy.get_chunk_text(&candidate.chunk_id)? {
                Some(text) => {
                    candidate_texts.push(text);
                    candidate_chunk_ids.push(candidate.chunk_id.clone());
                    candidate_entry_ids.push(candidate.entry_id.clone());
                }
                None => {
                    warn!(
                        chunk_id = %candidate.chunk_id,
                        "chunk text not found in tantivy, skipping"
                    );
                }
            }
        }

        let mut results: Vec<(String, String, f32, String)>;

        if candidate_texts.is_empty() {
            results = vec![];
        } else {
            let text_refs: Vec<&str> = candidate_texts.iter().map(|s| s.as_str()).collect();
            let rerank_limit = params.limit.max(candidate_texts.len());

            match embedder.rerank(&params.query, &text_refs, rerank_limit) {
                Ok(reranked) => {
                    results = Vec::with_capacity(reranked.len());
                    for rr in &reranked {
                        let idx = rr.index;
                        if idx < candidate_chunk_ids.len() {
                            results.push((
                                candidate_chunk_ids[idx].clone(),
                                candidate_entry_ids[idx].clone(),
                                rr.score,
                                candidate_texts[idx].clone(),
                            ));
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "reranker failed, falling back to RRF scores");
                    results = candidate_chunk_ids
                        .iter()
                        .zip(candidate_entry_ids.iter())
                        .zip(candidate_texts.iter())
                        .enumerate()
                        .map(|(i, ((cid, eid), text))| {
                            let rrf_score = if i < top_candidates.len() {
                                top_candidates[i].rrf_score as f32
                            } else {
                                0.0
                            };
                            (cid.clone(), eid.clone(), rrf_score, text.clone())
                        })
                        .collect();
                }
            }
        }

        // Record chunk_ids + reranker (or RRF-fallback) scores for eval mining.
        let ids: Vec<String> = results.iter().map(|(cid, _, _, _)| cid.clone()).collect();
        let scores_vec: Vec<f32> = results.iter().map(|(_, _, s, _)| *s).collect();
        record_stage_scores(&Span::current(), &ids, &scores_vec);

        Span::current().record("output_count", results.len());
        results
    };

    // Deduplicate by entry_id.
    {
        let mut dedup_input: Vec<(String, String, f32)> = scored_results
            .iter()
            .map(|(cid, eid, score, _)| (cid.clone(), eid.clone(), *score))
            .collect();
        deduplicate_by_entry(&mut dedup_input);
        let keep_chunks: std::collections::HashSet<String> =
            dedup_input.iter().map(|(cid, _, _)| cid.clone()).collect();
        scored_results.retain(|(cid, _, _, _)| keep_chunks.contains(cid));
    }

    scored_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Step 10: Apply min_score filter.
    let pre_min_score_count = scored_results.len();
    if let Some(min) = params.min_score {
        scored_results.retain(|(_, _, score, _)| *score >= min);
    }

    // Step 11: Apply max_tokens budget.
    if let Some(budget) = params.max_tokens {
        let mut token_count = 0usize;
        let mut keep_count = 0usize;
        for (_cid, _eid, _score, text) in &scored_results {
            let words = text.split_whitespace().count();
            let estimated_tokens = (words as f32 * 1.33) as usize;
            if token_count + estimated_tokens > budget && keep_count > 0 {
                break;
            }
            token_count += estimated_tokens;
            keep_count += 1;
        }
        scored_results.truncate(keep_count);
    }

    scored_results.truncate(params.limit);

    // Step 12: Build SearchResult for each.
    let final_scores: Vec<f32> = scored_results.iter().map(|(_, _, s, _)| *s).collect();

    let mut results: Vec<SearchResult> = Vec::with_capacity(scored_results.len());
    for (chunk_id, entry_id, score, content) in scored_results {
        let kind = tantivy
            .get_chunk_kind(&chunk_id)
            .ok()
            .flatten()
            .unwrap_or_default();
        results.push(SearchResult {
            id: entry_id,
            chunk_id,
            kind,
            score,
            content,
        });
    }

    // Step 13: Quality signal.
    let quality = {
        let _span = info_span!("corvia.search.quality", confidence = tracing::field::Empty, stale).entered();
        let mut q = compute_quality_signal(&final_scores, stale);
        if results.is_empty() && pre_min_score_count > 0 && params.min_score.is_some() {
            q.suggestion = Some(
                "No results above minimum score threshold. Try lowering min_score or broadening your query.".to_string(),
            );
        }
        Span::current().record("confidence", tracing::field::debug(q.confidence));
        q
    };

    Span::current().record("result_count", results.len());
    Span::current().record("confidence", tracing::field::debug(quality.confidence));

    info!(
        results = results.len(),
        confidence = ?quality.confidence,
        "search complete"
    );

    Ok(SearchResponse { results, quality })
}

/// Run the hybrid search pipeline, opening index handles internally.
///
/// For callers that hold persistent handles, use [`search_with_handles`] directly.
pub fn search(
    config: &Config,
    base_dir: &Path,
    embedder: &Embedder,
    params: &SearchParams,
) -> Result<SearchResponse> {
    let redb = RedbIndex::open(&base_dir.join(config.redb_path()))
        .context("opening redb index for search")?;
    let tantivy = TantivyIndex::open(&base_dir.join(config.tantivy_dir()))
        .context("opening tantivy index for search")?;
    search_with_handles(config, base_dir, embedder, params, &redb, &tantivy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fusion_math() {
        // With k=30, a single item at rank 0 should have score 1/(30+0+1) = 1/31.
        let bm25 = vec![("chunk-a".to_string(), "entry-a".to_string(), 1.0f32)];
        let vector: Vec<(String, String, f32)> = vec![];

        let fused = rrf_fusion(&bm25, &vector, 30);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].chunk_id, "chunk-a");

        let expected = 1.0 / 31.0;
        let diff = (fused[0].rrf_score - expected).abs();
        assert!(
            diff < 1e-9,
            "expected RRF score ~{expected}, got {}",
            fused[0].rrf_score
        );
    }

    #[test]
    fn rrf_fusion_two_lists_same_item() {
        // Item appearing in both lists at rank 0 should get 2/(k+1).
        let bm25 = vec![("chunk-a".to_string(), "entry-a".to_string(), 1.0f32)];
        let vector = vec![("chunk-a".to_string(), "entry-a".to_string(), 0.9f32)];

        let fused = rrf_fusion(&bm25, &vector, 30);
        assert_eq!(fused.len(), 1);

        let expected = 2.0 / 31.0;
        let diff = (fused[0].rrf_score - expected).abs();
        assert!(
            diff < 1e-9,
            "expected combined RRF score ~{expected}, got {}",
            fused[0].rrf_score
        );
    }

    #[test]
    fn rrf_fusion_ordering() {
        // Item appearing in both lists should rank higher than one appearing in only one.
        let bm25 = vec![
            ("chunk-a".to_string(), "entry-a".to_string(), 1.0f32),
            ("chunk-b".to_string(), "entry-b".to_string(), 0.8f32),
        ];
        let vector = vec![("chunk-a".to_string(), "entry-a".to_string(), 0.9f32)];

        let fused = rrf_fusion(&bm25, &vector, 30);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].chunk_id, "chunk-a", "chunk-a should rank first (appears in both)");
        assert_eq!(fused[1].chunk_id, "chunk-b", "chunk-b should rank second (appears in one)");
    }

    #[test]
    fn quality_signal_high_confidence() {
        let scores = vec![0.8, 0.6, 0.5, 0.3];
        let quality = compute_quality_signal(&scores, false);
        assert_eq!(quality.confidence, Confidence::High);
        assert!(quality.suggestion.is_none());
    }

    #[test]
    fn quality_signal_medium_confidence() {
        // Top score above medium threshold but below high threshold, or count < 3.
        let scores = vec![0.3, 0.2];
        let quality = compute_quality_signal(&scores, false);
        assert_eq!(quality.confidence, Confidence::Medium);
    }

    #[test]
    fn quality_signal_low_confidence() {
        let scores = vec![0.1, 0.05];
        let quality = compute_quality_signal(&scores, false);
        assert_eq!(quality.confidence, Confidence::Low);
        assert!(quality.suggestion.is_some());
    }

    #[test]
    fn quality_signal_none_when_empty() {
        let scores: Vec<f32> = vec![];
        let quality = compute_quality_signal(&scores, false);
        assert_eq!(quality.confidence, Confidence::None);
    }

    #[test]
    fn quality_signal_stale_suggestion() {
        let scores = vec![0.8, 0.7, 0.6];
        let quality = compute_quality_signal(&scores, true);
        assert_eq!(quality.confidence, Confidence::High);
        assert!(quality.suggestion.is_some());
        assert!(
            quality
                .suggestion
                .as_ref()
                .unwrap()
                .contains("stale"),
            "stale suggestion should mention 'stale'"
        );
    }

    #[test]
    fn deduplicate_keeps_best_per_entry() {
        let mut candidates = vec![
            ("chunk-a:0".to_string(), "entry-a".to_string(), 0.5f32),
            ("chunk-a:1".to_string(), "entry-a".to_string(), 0.9f32),
            ("chunk-b:0".to_string(), "entry-b".to_string(), 0.7f32),
        ];

        deduplicate_by_entry(&mut candidates);
        assert_eq!(candidates.len(), 2, "should keep one per entry");

        // Check that entry-a kept the higher scoring chunk.
        let entry_a = candidates
            .iter()
            .find(|(_, eid, _)| eid == "entry-a")
            .unwrap();
        assert_eq!(entry_a.0, "chunk-a:1", "should keep chunk with score 0.9");
        assert!((entry_a.2 - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn rrf_fusion_empty_inputs() {
        let bm25: Vec<(String, String, f32)> = vec![];
        let vector: Vec<(String, String, f32)> = vec![];

        let fused = rrf_fusion(&bm25, &vector, 30);
        assert!(fused.is_empty());
    }

    #[test]
    #[ignore]
    fn search_integration_cold_start() {
        // Integration test: requires embedder. Verifies cold start response.
        let dir = tempfile::tempdir().unwrap();
        let config = Config::default();
        let embedder = Embedder::new(None, "nomic-embed-text-v1.5", "jina-v1-turbo").expect("failed to init embedder");

        // Create the index directories so open succeeds.
        std::fs::create_dir_all(dir.path().join(config.index_dir())).unwrap();
        std::fs::create_dir_all(dir.path().join(config.entries_dir())).unwrap();

        let params = SearchParams {
            query: "test query".to_string(),
            limit: 5,
            max_tokens: None,
            min_score: None,
            kind: None,
        };

        let response = search(&config, dir.path(), &embedder, &params).unwrap();
        assert!(response.results.is_empty());
        assert_eq!(response.quality.confidence, Confidence::None);
        assert!(response.quality.suggestion.is_some());
        assert!(
            response
                .quality
                .suggestion
                .as_ref()
                .unwrap()
                .contains("No entries indexed"),
        );
    }

    #[test]
    fn search_with_handles_signature_exists() {
        // Compile-time check: verify search_with_handles is public with the right signature.
        let _fn: fn(
            &Config,
            &std::path::Path,
            &crate::embed::Embedder,
            &SearchParams,
            &crate::index::RedbIndex,
            &crate::tantivy_index::TantivyIndex,
        ) -> anyhow::Result<crate::types::SearchResponse> = search_with_handles;
        let _ = _fn;
    }

    #[test]
    fn encode_stage_scores_empty() {
        let (ids, scores) = encode_stage_scores(&[], &[]);
        assert_eq!(ids, "[]");
        assert_eq!(scores, "[]");
    }

    #[test]
    fn encode_stage_scores_parallel_arrays() {
        let chunk_ids = vec!["a:0".to_string(), "b:1".to_string(), "c:2".to_string()];
        let scores = vec![0.9f32, 0.5, 0.1];
        let (ids_json, scores_json) = encode_stage_scores(&chunk_ids, &scores);
        assert_eq!(ids_json, r#"["a:0","b:1","c:2"]"#);
        assert_eq!(scores_json, "[0.9,0.5,0.1]");
    }

    #[test]
    fn encode_stage_scores_length_mismatch_still_encodes_both() {
        let chunk_ids = vec!["a".to_string()];
        let scores = vec![0.1f32, 0.2, 0.3];
        let (ids_json, scores_json) = encode_stage_scores(&chunk_ids, &scores);
        assert_eq!(ids_json, r#"["a"]"#);
        assert_eq!(scores_json, "[0.1,0.2,0.3]");
    }
}
