mod common;

use corvia_core::config::Config;
use corvia_core::embed::Embedder;
use corvia_core::entry::{new_entry, parse_entry, serialize_entry, write_entry_atomic};
use corvia_core::ingest::ingest;
use corvia_core::search::{search, SearchParams};
use corvia_core::types::{Confidence, Kind};
use corvia_core::write::{write, WriteParams};

// ---------------------------------------------------------------------------
// Helper: create an Embedder from config defaults
// ---------------------------------------------------------------------------

fn make_embedder(config: &Config) -> Embedder {
    Embedder::new(
        None,
        &config.embedding.model,
        &config.embedding.reranker_model,
    )
    .expect("failed to create embedder")
}

// ---------------------------------------------------------------------------
// 1. ingest_and_search_known_answer
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn ingest_and_search_known_answer() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    h.copy_fixtures();

    let result = ingest(config, base, false).unwrap();
    assert!(result.entries_ingested >= 4, "should ingest fixture entries");

    let embedder = make_embedder(config);
    let params = SearchParams {
        query: "why did we choose Redb".to_string(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    assert!(
        !response.results.is_empty(),
        "should return results for known fixture content"
    );

    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&"fixture-decision-01"),
        "fixture-decision-01 should be in top results, got: {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. write_then_search_finds_entry
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn write_then_search_finds_entry() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let write_response = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Tantivy provides fast full-text search in Rust".to_string(),
            kind: Kind::Reference,
            tags: vec!["tantivy".to_string()],
            supersedes: vec![],
        },
    )
    .unwrap();

    assert_eq!(write_response.action, "created");

    let params = SearchParams {
        query: "tantivy full-text".to_string(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&write_response.id.as_str()),
        "written entry {} should appear in search results, got: {ids:?}",
        write_response.id
    );
}

// ---------------------------------------------------------------------------
// 3. auto_dedup_supersedes_similar_entry
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn auto_dedup_supersedes_similar_entry() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let first = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "We use nomic-embed-text-v1.5 for embeddings".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();
    assert_eq!(first.action, "created");

    let second = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "We use nomic-embed-text-v1.5 as the embedding model".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    assert_eq!(
        second.action, "superseded",
        "second write of similar content should trigger auto-dedup"
    );
    assert!(
        second.superseded.contains(&first.id),
        "superseded list should contain first entry ID {}, got: {:?}",
        first.id,
        second.superseded
    );
}

// ---------------------------------------------------------------------------
// 4. superseded_entry_excluded_from_search
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn superseded_entry_excluded_from_search() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let first = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "We chose SQLite for storage".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    let _second = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "We chose Redb for storage".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![first.id.clone()],
        },
    )
    .unwrap();

    let params = SearchParams {
        query: "storage choice".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        !ids.contains(&first.id.as_str()),
        "superseded entry {} should NOT appear in results, got: {ids:?}",
        first.id
    );
}

// ---------------------------------------------------------------------------
// 5. supersession_chain
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn supersession_chain() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let a = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Initial storage decision: we use flat files".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    let b = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Updated storage decision: we use SQLite".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![a.id.clone()],
        },
    )
    .unwrap();

    let c = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Final storage decision: we use Redb".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![b.id.clone()],
        },
    )
    .unwrap();

    let params = SearchParams {
        query: "storage decision".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();

    assert!(
        !ids.contains(&a.id.as_str()),
        "entry A should be superseded and absent from results"
    );
    assert!(
        !ids.contains(&b.id.as_str()),
        "entry B should be superseded and absent from results"
    );
    assert!(
        ids.contains(&c.id.as_str()),
        "entry C (latest) should appear in results, got: {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. cold_start_returns_helpful_message
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn cold_start_returns_helpful_message() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    // Ensure index dirs exist but are empty (no ingest).
    std::fs::create_dir_all(base.join(config.index_dir())).unwrap();
    std::fs::create_dir_all(base.join(config.entries_dir())).unwrap();

    let params = SearchParams {
        query: "anything at all".to_string(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    assert!(response.results.is_empty(), "cold start should return no results");
    assert_eq!(
        response.quality.confidence,
        Confidence::None,
        "confidence should be None on cold start"
    );
    assert!(
        response.quality.suggestion.is_some(),
        "should provide a suggestion on cold start"
    );
    let suggestion = response.quality.suggestion.unwrap();
    assert!(
        suggestion.contains("ingest"),
        "suggestion should mention 'ingest', got: {suggestion}"
    );
}

// ---------------------------------------------------------------------------
// 7. malformed_entry_skipped_during_ingest
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn malformed_entry_skipped_during_ingest() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    let entries_dir = base.join(config.entries_dir());
    std::fs::create_dir_all(&entries_dir).unwrap();

    // Good entry: valid TOML frontmatter with all required fields.
    let good_content = r#"+++
id = "good-entry-01"
created_at = "2026-04-15T00:00:00Z"
kind = "learning"
+++

This is a valid entry with useful content about Rust programming.
"#;
    std::fs::write(entries_dir.join("good-entry.md"), good_content).unwrap();

    // Bad entry: missing id field.
    let bad_content = r#"+++
kind = "learning"
created_at = "2026-04-15T00:00:00Z"
+++

This entry has no id field.
"#;
    std::fs::write(entries_dir.join("bad-entry.md"), bad_content).unwrap();

    let result = ingest(config, base, false).unwrap();
    assert_eq!(
        result.entries_ingested, 1,
        "should ingest only the valid entry"
    );
    assert_eq!(
        result.entries_skipped.len(),
        1,
        "should skip the malformed entry"
    );
}

// ---------------------------------------------------------------------------
// 8. entry_roundtrip_serialization (does NOT need embedder)
// ---------------------------------------------------------------------------

#[test]
fn entry_roundtrip_serialization() {
    let entry = new_entry(
        "Body with special chars: \"quotes\", [brackets], key=value, and unicode \u{1f600}"
            .to_string(),
        Kind::Decision,
        vec![
            "tag-with-dash".to_string(),
            "tag with spaces".to_string(),
            "key=value".to_string(),
        ],
        vec!["old-id-1".to_string()],
    );

    let serialized = serialize_entry(&entry).unwrap();
    let parsed = parse_entry(&serialized).unwrap();

    assert_eq!(parsed.meta.id, entry.meta.id);
    assert_eq!(parsed.meta.created_at, entry.meta.created_at);
    assert_eq!(parsed.meta.kind, entry.meta.kind);
    assert_eq!(parsed.meta.supersedes, entry.meta.supersedes);
    assert_eq!(parsed.meta.tags, entry.meta.tags);
    assert_eq!(parsed.body, entry.body);

    // Also test atomic file write + read roundtrip.
    let dir = tempfile::tempdir().unwrap();
    let entries_dir = dir.path().join("entries");
    let path = write_entry_atomic(&entries_dir, &entry).unwrap();
    let read_back = corvia_core::entry::read_entry(&path).unwrap();

    assert_eq!(read_back.meta.id, entry.meta.id);
    assert_eq!(read_back.meta.tags, entry.meta.tags);
    assert_eq!(read_back.body, entry.body);
}

// ---------------------------------------------------------------------------
// 9. write_with_missing_supersedes_warns
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn write_with_missing_supersedes_warns() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let response = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Some new content that supersedes a nonexistent entry".to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec!["nonexistent-id".to_string()],
        },
    )
    .unwrap();

    // Entry should still be created despite the missing supersedes reference.
    assert!(!response.id.is_empty(), "entry should be created");
    assert!(
        response.warning.is_some(),
        "should produce a warning for missing supersedes"
    );
    let warning = response.warning.unwrap();
    assert!(
        warning.contains("not found"),
        "warning should mention 'not found', got: {warning}"
    );
}

// ---------------------------------------------------------------------------
// 10. kind_filter_in_search
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn kind_filter_in_search() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    // Write a decision.
    let decision = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "We decided to use Redb for persistent vector storage in corvia".to_string(),
            kind: Kind::Decision,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    // Write an instruction.
    let instruction = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "To configure Redb storage, set data_dir in corvia.toml".to_string(),
            kind: Kind::Instruction,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    // Search with kind=Decision filter.
    let params = SearchParams {
        query: "Redb storage".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: Some(Kind::Decision),
    };

    let response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();

    assert!(
        ids.contains(&decision.id.as_str()),
        "decision entry should appear when filtering by Decision"
    );
    assert!(
        !ids.contains(&instruction.id.as_str()),
        "instruction entry should NOT appear when filtering by Decision"
    );
}

// ---------------------------------------------------------------------------
// 11. search_drift_detection
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn search_drift_detection() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    h.copy_fixtures();

    let result = ingest(config, base, false).unwrap();
    assert!(result.entries_ingested > 0, "should ingest fixtures");

    let embedder = make_embedder(config);

    // Add a new .md file to entries dir WITHOUT re-ingesting.
    let entries_dir = base.join(config.entries_dir());
    let extra_content = r#"+++
id = "drift-extra-01"
created_at = "2026-04-15T00:00:00Z"
kind = "learning"
+++

This entry was added after ingest and causes drift.
"#;
    std::fs::write(entries_dir.join("drift-extra.md"), extra_content).unwrap();

    let params = SearchParams {
        query: "any query".to_string(),
        limit: 5,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();

    // The quality signal should detect staleness.
    assert!(
        response.quality.suggestion.is_some(),
        "quality should have a suggestion when index is stale"
    );
    let suggestion = response.quality.suggestion.unwrap();
    assert!(
        suggestion.to_lowercase().contains("stale") || suggestion.to_lowercase().contains("ingest"),
        "suggestion should mention staleness or ingest, got: {suggestion}"
    );
}

// ---------------------------------------------------------------------------
// 12. deletion_via_ingest
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn deletion_via_ingest() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    h.copy_fixtures();

    let result = ingest(config, base, false).unwrap();
    let initial_count = result.entries_ingested;
    assert!(initial_count >= 4, "should ingest all fixtures");

    let embedder = make_embedder(config);

    // Verify fixture-decision-01 is searchable.
    let params = SearchParams {
        query: "chose Redb over SQLite".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    };
    let response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        ids.contains(&"fixture-decision-01"),
        "fixture-decision-01 should be found before deletion"
    );

    // Delete the fixture file.
    let entries_dir = base.join(config.entries_dir());
    std::fs::remove_file(entries_dir.join("decision-storage.md")).unwrap();

    // Re-ingest with --fresh.
    let result2 = ingest(config, base, true).unwrap();
    assert_eq!(
        result2.entries_ingested,
        initial_count - 1,
        "should have one fewer entry after deletion"
    );

    // Search again: the deleted entry should be gone.
    let response2 = search(config, base, &embedder, &params).unwrap();
    let ids2: Vec<&str> = response2.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        !ids2.contains(&"fixture-decision-01"),
        "fixture-decision-01 should NOT be found after deletion and fresh ingest"
    );
}

// ---------------------------------------------------------------------------
// 13. empty_content_write
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn empty_content_write() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let response = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "".to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    assert!(!response.id.is_empty(), "should return an entry ID even for empty content");

    // Search should not find it (no vector was indexed for empty content).
    let params = SearchParams {
        query: "empty content".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let search_response = search(config, base, &embedder, &params).unwrap();
    let ids: Vec<&str> = search_response.results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        !ids.contains(&response.id.as_str()),
        "empty content entry should not appear in search results"
    );
}

// ---------------------------------------------------------------------------
// 14. write_returns_similarity_on_dedup
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn write_returns_similarity_on_dedup() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();
    let embedder = make_embedder(config);

    let _first = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Rust provides memory safety without garbage collection through ownership and borrowing".to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    let second = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Rust provides memory safety without garbage collection through ownership and borrowing".to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    assert_eq!(
        second.action, "superseded",
        "identical content should trigger auto-dedup"
    );
    assert!(
        second.similarity.is_some(),
        "dedup response should include similarity score"
    );
    let sim = second.similarity.unwrap();
    assert!(
        sim > 0.85,
        "similarity should be above dedup threshold (0.85), got: {sim}"
    );
}

// ---------------------------------------------------------------------------
// 15. status_shows_correct_counts
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn status_shows_correct_counts() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    h.copy_fixtures();

    let result = ingest(config, base, false).unwrap();
    let expected_entries = result.entries_ingested;

    // Open the Redb index to check status fields.
    let redb = corvia_core::index::RedbIndex::open(&base.join(config.redb_path())).unwrap();

    let entry_count = redb.entry_count().unwrap();
    assert_eq!(
        entry_count as usize, expected_entries,
        "Redb entry count should match ingested count"
    );

    let entry_count_meta = redb.get_meta("entry_count").unwrap();
    assert!(
        entry_count_meta.is_some(),
        "entry_count metadata should be set"
    );
    let meta_count: usize = entry_count_meta.unwrap().parse().unwrap();
    assert_eq!(
        meta_count, expected_entries,
        "entry_count metadata should match"
    );

    let last_ingest = redb.get_meta("last_ingest").unwrap();
    assert!(
        last_ingest.is_some(),
        "last_ingest metadata should be set after ingest"
    );

    let vector_count = redb.vector_count().unwrap();
    assert!(
        vector_count > 0,
        "should have stored vectors after ingest"
    );
    assert!(
        vector_count >= expected_entries as u64,
        "should have at least one vector per entry"
    );
}

// ---------------------------------------------------------------------------
// 16. full_pipeline_ordering
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires embedding model
fn full_pipeline_ordering() {
    let h = common::TestHarness::new();
    let config = &h.config;
    let base = h.base_dir();

    h.copy_fixtures();

    let result = ingest(config, base, false).unwrap();
    assert!(result.entries_ingested > 0, "should ingest fixtures");

    let embedder = make_embedder(config);

    // Write 2 more entries related to the query we'll search for.
    let _w1 = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "Redb transactions are ACID compliant and use MVCC for concurrency"
                .to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    let _w2 = write(
        config,
        base,
        &embedder,
        WriteParams {
            content: "The embedding model produces 768-dimensional vectors for semantic search"
                .to_string(),
            kind: Kind::Learning,
            tags: vec![],
            supersedes: vec![],
        },
    )
    .unwrap();

    let params = SearchParams {
        query: "Redb storage transactions".to_string(),
        limit: 10,
        max_tokens: None,
        min_score: None,
        kind: None,
    };

    let response = search(config, base, &embedder, &params).unwrap();
    assert!(
        !response.results.is_empty(),
        "should return results from combined ingest + write"
    );

    // Verify results are ordered by score descending.
    let scores: Vec<f32> = response.results.iter().map(|r| r.score).collect();
    for i in 1..scores.len() {
        assert!(
            scores[i - 1] >= scores[i],
            "results should be ordered by score descending: {} (pos {}) < {} (pos {})",
            scores[i - 1],
            i - 1,
            scores[i],
            i,
        );
    }
}
