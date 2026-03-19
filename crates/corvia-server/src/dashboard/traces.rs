//! Structured log parsing and span aggregation.
//!
//! Ported from Python `corvia_dev/traces.py`.

use corvia_common::dashboard::{SpanStats, TraceEvent, TracesData};
use std::collections::HashMap;
use std::path::Path;

/// Classify a span name into a module.
/// Exact matches checked first, then prefix matches (first match wins).
pub fn span_to_module(span: &str) -> &'static str {
    // Exact overrides
    if span == "corvia.entry.embed" {
        return "inference";
    }

    // Prefix matches (order matters — first match wins)
    const PREFIX_MAP: &[(&str, &str)] = &[
        ("corvia.agent.", "agent"),
        ("corvia.session.", "agent"),
        ("corvia.entry.", "entry"),
        ("corvia.merge.", "merge"),
        ("corvia.store.", "storage"),
        ("corvia.rag.", "rag"),
        ("corvia.gc.", "gc"),
    ];

    for (prefix, module) in PREFIX_MAP {
        if span.starts_with(prefix) {
            return module;
        }
    }

    "unknown"
}

/// Classify a Rust module target path into a dashboard module.
pub fn target_to_module(target: &str) -> &'static str {
    const TARGET_MAP: &[(&str, &str)] = &[
        ("agent_coordinator", "agent"),
        ("merge_worker", "merge"),
        ("lite_store", "storage"),
        ("postgres_store", "storage"),
        ("rag_pipeline", "rag"),
        ("graph_store", "storage"),
        ("chunking", "entry"),
        ("embedding_service", "inference"),
        ("chat_service", "inference"),
        ("model_manager", "inference"),
    ];

    for (pattern, module) in TARGET_MAP {
        if target.contains(pattern) {
            return module;
        }
    }

    "unknown"
}

/// Parsed trace line — either a span with timing or a structured event
pub enum ParsedTrace {
    Span {
        level: String,
        timestamp: String,
        span_name: String,
        elapsed_ms: f64,
        trace_id: Option<String>,
        span_id: Option<String>,
        parent_span_id: Option<String>,
    },
    Event {
        level: String,
        timestamp: String,
        msg: String,
        target: String,
    },
}

/// Parse a single JSON-structured trace line.
/// Returns None for invalid or non-JSON lines.
pub fn parse_trace_line(line: &str) -> Option<ParsedTrace> {
    let line = line.trim();
    if line.is_empty() || !line.starts_with('{') {
        return None;
    }

    let v: serde_json::Value = serde_json::from_str(line).ok()?;

    let level = v.get("level")?.as_str()?.to_string();
    let timestamp = v.get("timestamp")?.as_str()?.to_string();

    // Check if it's a span with timing
    if let Some(span_name) = v
        .get("span")
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
        && let Some(elapsed_ms) = v.get("elapsed_ms").and_then(|e| e.as_f64()) {
            let trace_id = v.get("otel.trace_id").and_then(|t| t.as_str()).map(String::from);
            let span_id = v.get("otel.span_id").and_then(|t| t.as_str()).map(String::from);
            let parent_span_id = v.get("otel.parent_span_id").and_then(|t| t.as_str()).map(String::from);
            return Some(ParsedTrace::Span {
                level,
                timestamp,
                span_name: span_name.to_string(),
                elapsed_ms,
                trace_id,
                span_id,
                parent_span_id,
            });
        }

    // Structured event
    let msg = v
        .get("fields")
        .and_then(|f| f.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let target = v
        .get("target")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    // Skip lines with no useful content
    if msg.is_empty() && target.is_empty() {
        return None;
    }

    Some(ParsedTrace::Event {
        level,
        timestamp,
        msg,
        target,
    })
}

/// Normalize log level string to lowercase standard form
pub fn normalize_level(level: &str) -> &'static str {
    match level.to_lowercase().as_str() {
        "warn" | "warning" => "warn",
        "error" | "err" => "error",
        "debug" | "trace" => "debug",
        _ => "info",
    }
}

/// Extract HH:MM:SS from an ISO timestamp string
pub fn short_timestamp(ts: &str) -> String {
    if let Some(t_pos) = ts.find('T') {
        // 'T' is ASCII so t_pos + 1 is always a valid char boundary
        let rest = &ts[t_pos + 1..];
        if rest.len() >= 8 && rest.is_char_boundary(8) {
            return rest[..8].to_string();
        }
    }
    ts.to_string()
}

/// Parse ISO timestamp to epoch seconds (for 1-hour window filtering)
fn timestamp_to_epoch(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|dt| dt.timestamp())
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S")
                .ok()
                .map(|ndt| ndt.and_utc().timestamp())
        })
}

/// Compute the p-th percentile from a mutable slice of durations.
/// Sorts the slice in place. Returns 0.0 for empty slices.
pub fn compute_percentile(durations: &mut [f64], percentile: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((percentile / 100.0) * (durations.len() - 1) as f64).round() as usize;
    durations[idx.min(durations.len() - 1)]
}

/// Aggregate trace data from parsed log lines.
/// Computes span statistics (all-time + 1-hour window) and collects recent events.
pub fn collect_traces_from_lines(lines: &[&str]) -> TracesData {
    let now = chrono::Utc::now().timestamp();
    let one_hour_ago = now - 3600;

    let mut span_all: HashMap<String, Vec<f64>> = HashMap::new();
    let mut span_1h: HashMap<String, Vec<f64>> = HashMap::new();
    let mut span_errors: HashMap<String, u64> = HashMap::new();
    let mut events: Vec<TraceEvent> = Vec::new();

    for line in lines {
        let parsed = match parse_trace_line(line) {
            Some(p) => p,
            None => continue,
        };

        match parsed {
            ParsedTrace::Span {
                level,
                timestamp,
                span_name,
                elapsed_ms,
                ..
            } => {
                span_all
                    .entry(span_name.clone())
                    .or_default()
                    .push(elapsed_ms);

                if let Some(epoch) = timestamp_to_epoch(&timestamp)
                    && epoch >= one_hour_ago {
                        span_1h
                            .entry(span_name.clone())
                            .or_default()
                            .push(elapsed_ms);
                    }

                let level_lower = level.to_lowercase();
                if level_lower == "error" || level_lower == "err" {
                    *span_errors.entry(span_name).or_default() += 1;
                }
            }
            ParsedTrace::Event {
                level,
                timestamp,
                msg,
                target,
            } => {
                if !msg.is_empty() {
                    let module = target_to_module(&target);
                    events.push(TraceEvent {
                        ts: short_timestamp(&timestamp),
                        level: normalize_level(&level).to_string(),
                        module: module.to_string(),
                        msg,
                    });
                }
            }
        }
    }

    // Build SpanStats
    let mut spans = HashMap::new();
    for (name, timings) in &span_all {
        let count = timings.len() as u64;
        let avg_ms = timings.iter().sum::<f64>() / count as f64;
        let last_ms = *timings.last().unwrap_or(&0.0);
        let count_1h = span_1h.get(name).map(|v| v.len() as u64).unwrap_or(0);
        let errors = span_errors.get(name).copied().unwrap_or(0);

        let mut sorted = timings.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |p: f64| -> f64 {
            if sorted.is_empty() { return 0.0; }
            let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };
        let p50_ms = pct(50.0);
        let p95_ms = pct(95.0);
        let p99_ms = pct(99.0);

        spans.insert(
            name.clone(),
            SpanStats {
                count,
                count_1h,
                avg_ms,
                last_ms,
                errors,
                p50_ms,
                p95_ms,
                p99_ms,
            },
        );
    }

    // Keep last 50 events
    let recent_events = if events.len() > 50 {
        events[events.len() - 50..].to_vec()
    } else {
        events
    };

    TracesData {
        spans,
        recent_events,
    }
}

/// Build trace trees from log lines that have OTEL trace context.
/// Groups spans by trace_id, sorts by timestamp, builds parent-child trees.
/// Returns the most recent `limit` traces.
pub fn collect_trace_trees(lines: &[&str], limit: usize) -> Vec<corvia_common::dashboard::TraceTree> {
    use corvia_common::dashboard::{SpanNode, TraceTree};

    struct RawSpan {
        trace_id: String,
        span_id: String,
        parent_span_id: String,
        span_name: String,
        elapsed_ms: f64,
        timestamp: String,
    }

    let mut spans_by_trace: HashMap<String, Vec<RawSpan>> = HashMap::new();

    for line in lines {
        if let Some(ParsedTrace::Span {
            timestamp,
            span_name,
            elapsed_ms,
            trace_id: Some(tid),
            span_id: Some(sid),
            parent_span_id,
            ..
        }) = parse_trace_line(line)
        {
            spans_by_trace
                .entry(tid.clone())
                .or_default()
                .push(RawSpan {
                    trace_id: tid,
                    span_id: sid,
                    parent_span_id: parent_span_id.unwrap_or_default(),
                    span_name,
                    elapsed_ms,
                    timestamp,
                });
        }
    }

    let mut trees: Vec<TraceTree> = Vec::new();

    for (trace_id, mut trace_spans) in spans_by_trace {
        trace_spans.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let root_ts = trace_spans
            .first()
            .map(|s| s.timestamp.clone())
            .unwrap_or_default();

        let root_name = trace_spans
            .iter()
            .find(|s| s.parent_span_id.is_empty())
            .map(|s| s.span_name.clone())
            .unwrap_or_else(|| {
                trace_spans
                    .first()
                    .map(|s| s.span_name.clone())
                    .unwrap_or_default()
            });

        let total_ms = trace_spans
            .iter()
            .find(|s| s.parent_span_id.is_empty())
            .map(|s| s.elapsed_ms)
            .unwrap_or_else(|| {
                trace_spans
                    .iter()
                    .map(|s| s.elapsed_ms)
                    .fold(0.0f64, f64::max)
            });

        let span_count = trace_spans.len();

        // Build parent-child tree
        let mut node_map: HashMap<String, SpanNode> = trace_spans
            .iter()
            .map(|s| {
                let offset_ms = compute_offset_ms(&root_ts, &s.timestamp);
                (
                    s.span_id.clone(),
                    SpanNode {
                        span_id: s.span_id.clone(),
                        parent_span_id: s.parent_span_id.clone(),
                        trace_id: s.trace_id.clone(),
                        span_name: s.span_name.clone(),
                        elapsed_ms: s.elapsed_ms,
                        start_offset_ms: offset_ms,
                        depth: 0,
                        module: span_to_module(&s.span_name).to_string(),
                        fields: serde_json::Value::Null,
                        children: vec![],
                    },
                )
            })
            .collect();

        let mut children_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut root_ids: Vec<String> = Vec::new();

        for s in &trace_spans {
            if s.parent_span_id.is_empty() || !node_map.contains_key(&s.parent_span_id) {
                root_ids.push(s.span_id.clone());
            } else {
                children_map
                    .entry(s.parent_span_id.clone())
                    .or_default()
                    .push(s.span_id.clone());
            }
        }

        fn build_tree(
            id: &str,
            depth: usize,
            node_map: &mut HashMap<String, SpanNode>,
            children_map: &HashMap<String, Vec<String>>,
        ) -> Option<SpanNode> {
            let mut node = node_map.remove(id)?;
            node.depth = depth;
            if let Some(child_ids) = children_map.get(id) {
                for cid in child_ids {
                    if let Some(child) = build_tree(cid, depth + 1, node_map, children_map) {
                        node.children.push(child);
                    }
                }
            }
            Some(node)
        }

        let tree_roots: Vec<SpanNode> = root_ids
            .iter()
            .filter_map(|id| build_tree(id, 0, &mut node_map, &children_map))
            .collect();

        trees.push(TraceTree {
            trace_id,
            root_span: root_name,
            total_ms,
            span_count,
            started_at: root_ts,
            spans: tree_roots,
        });
    }

    trees.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    trees.truncate(limit);
    trees
}

/// Compute millisecond offset between two RFC 3339 timestamps.
fn compute_offset_ms(base: &str, ts: &str) -> f64 {
    let base_dt = chrono::DateTime::parse_from_rfc3339(base).ok();
    let ts_dt = chrono::DateTime::parse_from_rfc3339(ts).ok();
    match (base_dt, ts_dt) {
        (Some(b), Some(t)) => {
            let diff = t.signed_duration_since(b);
            diff.num_milliseconds() as f64
        }
        _ => 0.0,
    }
}

/// Maximum file size to read (50 MB) — skip larger files to prevent DoS
const MAX_LOG_FILE_SIZE: u64 = 50 * 1024 * 1024;

/// Read the last `n` lines from a file.
/// Files larger than 50 MB are skipped entirely.
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader};

    // Check file size before reading
    if let Ok(meta) = fs::metadata(path)
        && meta.len() > MAX_LOG_FILE_SIZE {
            return Vec::new();
        }

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
    let start = all_lines.len().saturating_sub(n);
    all_lines[start..].to_vec()
}

/// Resolve the log directory — checks env var, then default path
pub fn log_dir() -> std::path::PathBuf {
    std::env::var("CORVIA_LOG_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/corvia-dev-logs"))
}

/// Collect traces from all .log files in a directory.
/// Reads last 500 lines per file to bound memory.
pub fn collect_traces(log_dir: &Path) -> TracesData {
    let mut all_lines = Vec::new();

    if let Ok(entries) = std::fs::read_dir(log_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                let lines = tail_lines(&path, 500);
                all_lines.extend(lines);
            }
        }
    }

    let line_refs: Vec<&str> = all_lines.iter().map(|s| s.as_str()).collect();
    collect_traces_from_lines(&line_refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Module classification ---

    #[test]
    fn span_to_module_exact_override() {
        assert_eq!(span_to_module("corvia.entry.embed"), "inference");
    }

    #[test]
    fn span_to_module_prefix_match() {
        assert_eq!(span_to_module("corvia.agent.register"), "agent");
        assert_eq!(span_to_module("corvia.session.create"), "agent");
        assert_eq!(span_to_module("corvia.entry.write"), "entry");
        assert_eq!(span_to_module("corvia.merge.resolve"), "merge");
        assert_eq!(span_to_module("corvia.store.insert"), "storage");
        assert_eq!(span_to_module("corvia.rag.retrieve"), "rag");
        assert_eq!(span_to_module("corvia.gc.sweep"), "gc");
    }

    #[test]
    fn span_to_module_unknown() {
        assert_eq!(span_to_module("something.else"), "unknown");
    }

    #[test]
    fn target_to_module_matches_rust_paths() {
        assert_eq!(target_to_module("corvia_kernel::agent_coordinator"), "agent");
        assert_eq!(target_to_module("corvia_kernel::merge_worker"), "merge");
        assert_eq!(target_to_module("corvia_kernel::lite_store::write"), "storage");
        assert_eq!(target_to_module("corvia_kernel::rag_pipeline"), "rag");
        assert_eq!(target_to_module("corvia_kernel::graph_store"), "storage");
        assert_eq!(target_to_module("corvia_inference::embedding_service"), "inference");
        assert_eq!(target_to_module("corvia_inference::chat_service"), "inference");
        assert_eq!(target_to_module("corvia_inference::model_manager"), "inference");
        assert_eq!(target_to_module("corvia_kernel::chunking"), "entry");
    }

    #[test]
    fn target_to_module_unknown() {
        assert_eq!(target_to_module("some::other::module"), "unknown");
    }

    // --- Trace line parsing ---

    #[test]
    fn parse_span_with_timing() {
        let line = r#"{"timestamp":"2026-03-10T14:31:52Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{"session_id":"s1"},"elapsed_ms":12.5}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Span { level, span_name, elapsed_ms, .. } => {
                assert_eq!(level, "INFO");
                assert_eq!(span_name, "corvia.entry.write");
                assert!((elapsed_ms - 12.5).abs() < 0.01);
            }
            _ => panic!("expected Span variant"),
        }
    }

    #[test]
    fn parse_structured_event() {
        let line = r#"{"timestamp":"2026-03-10T14:31:52Z","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Event { level, msg, target, .. } => {
                assert_eq!(level, "WARN");
                assert_eq!(msg, "Slow embed: 210ms");
                assert_eq!(target, "corvia_kernel::agent_coordinator");
            }
            _ => panic!("expected Event variant"),
        }
    }

    #[test]
    fn parse_invalid_line_returns_none() {
        assert!(parse_trace_line("not json at all").is_none());
        assert!(parse_trace_line("").is_none());
        assert!(parse_trace_line("{}").is_none());
    }

    // --- Helpers ---

    #[test]
    fn normalize_level_variants() {
        assert_eq!(normalize_level("WARN"), "warn");
        assert_eq!(normalize_level("WARNING"), "warn");
        assert_eq!(normalize_level("ERROR"), "error");
        assert_eq!(normalize_level("ERR"), "error");
        assert_eq!(normalize_level("DEBUG"), "debug");
        assert_eq!(normalize_level("TRACE"), "debug");
        assert_eq!(normalize_level("INFO"), "info");
        assert_eq!(normalize_level("anything"), "info");
    }

    #[test]
    fn short_timestamp_extracts_time() {
        assert_eq!(short_timestamp("2026-03-10T14:31:52Z"), "14:31:52");
        assert_eq!(short_timestamp("2026-03-10T14:31:52.123Z"), "14:31:52");
        assert_eq!(short_timestamp("no-t-here"), "no-t-here");
    }

    // --- Aggregation ---

    #[test]
    fn collect_traces_aggregates_spans() {
        let lines = vec![
            r#"{"timestamp":"2026-03-10T14:31:50Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":10.0}"#,
            r#"{"timestamp":"2026-03-10T14:31:51Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":20.0}"#,
            r#"{"timestamp":"2026-03-10T14:31:52Z","level":"ERROR","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":30.0}"#,
        ];
        let data = collect_traces_from_lines(&lines);

        let span = data.spans.get("corvia.entry.write").unwrap();
        assert_eq!(span.count, 3);
        assert!((span.avg_ms - 20.0).abs() < 0.01);
        assert!((span.last_ms - 30.0).abs() < 0.01);
        assert_eq!(span.errors, 1);
    }

    #[test]
    fn collect_traces_captures_events() {
        let lines = vec![
            r#"{"timestamp":"2026-03-10T14:31:52Z","level":"WARN","fields":{"message":"Slow embed"},"target":"corvia_kernel::agent_coordinator"}"#,
        ];
        let data = collect_traces_from_lines(&lines);

        assert_eq!(data.recent_events.len(), 1);
        assert_eq!(data.recent_events[0].level, "warn");
        assert_eq!(data.recent_events[0].module, "agent");
        assert_eq!(data.recent_events[0].msg, "Slow embed");
        assert_eq!(data.recent_events[0].ts, "14:31:52");
    }

    #[test]
    fn collect_traces_limits_to_50_events() {
        let lines: Vec<String> = (0..60)
            .map(|i| {
                format!(
                    r#"{{"timestamp":"2026-03-10T14:{i:02}:00Z","level":"INFO","fields":{{"message":"event {i}"}},"target":"corvia_kernel::agent_coordinator"}}"#,
                )
            })
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let data = collect_traces_from_lines(&line_refs);

        assert_eq!(data.recent_events.len(), 50);
    }

    // --- File reading ---

    #[test]
    fn tail_lines_reads_last_n() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.log");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 0..10 {
            writeln!(f, "line {i}").unwrap();
        }
        drop(f);

        let lines = tail_lines(&path, 3);
        assert_eq!(lines, vec!["line 7", "line 8", "line 9"]);
    }

    #[test]
    fn tail_lines_missing_file_returns_empty() {
        let lines = tail_lines(Path::new("/nonexistent/file.log"), 10);
        assert!(lines.is_empty());
    }

    // --- OTEL trace context ---

    #[test]
    fn parse_span_with_otel_context() {
        let line = r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":12.5,"otel.trace_id":"abcdef1234567890abcdef1234567890","otel.span_id":"1234567890abcdef","otel.parent_span_id":"fedcba0987654321"}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Span {
                trace_id,
                span_id,
                parent_span_id,
                ..
            } => {
                assert_eq!(
                    trace_id.as_deref(),
                    Some("abcdef1234567890abcdef1234567890")
                );
                assert_eq!(span_id.as_deref(), Some("1234567890abcdef"));
                assert_eq!(parent_span_id.as_deref(), Some("fedcba0987654321"));
            }
            _ => panic!("expected Span variant"),
        }
    }

    #[test]
    fn parse_span_without_otel_context() {
        let line = r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":12.5}"#;
        let result = parse_trace_line(line).unwrap();
        match result {
            ParsedTrace::Span {
                trace_id,
                span_id,
                parent_span_id,
                ..
            } => {
                assert!(trace_id.is_none());
                assert!(span_id.is_none());
                assert!(parent_span_id.is_none());
            }
            _ => panic!("expected Span variant"),
        }
    }

    // --- Percentile computation ---

    #[test]
    fn percentile_computation() {
        let mut durations: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p50 = compute_percentile(&mut durations, 50.0);
        let p95 = compute_percentile(&mut durations, 95.0);
        let p99 = compute_percentile(&mut durations, 99.0);
        assert!((p50 - 50.0).abs() < 1.5);
        assert!((p95 - 95.0).abs() < 1.5);
        assert!((p99 - 99.0).abs() < 1.5);
    }

    #[test]
    fn percentile_empty_returns_zero() {
        let mut empty: Vec<f64> = vec![];
        assert_eq!(compute_percentile(&mut empty, 50.0), 0.0);
    }

    #[test]
    fn percentile_single_value() {
        let mut single = vec![42.0];
        assert_eq!(compute_percentile(&mut single, 50.0), 42.0);
        assert_eq!(compute_percentile(&mut single, 99.0), 42.0);
    }

    // --- Trace tree builder ---

    #[test]
    fn collect_trace_trees_builds_hierarchy() {
        let lines = vec![
            r#"{"timestamp":"2026-03-14T10:00:00.000Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":50.0,"otel.trace_id":"aaaa","otel.span_id":"1111","otel.parent_span_id":""}"#,
            r#"{"timestamp":"2026-03-14T10:00:00.010Z","level":"INFO","span":{"name":"corvia.entry.embed"},"fields":{},"elapsed_ms":30.0,"otel.trace_id":"aaaa","otel.span_id":"2222","otel.parent_span_id":"1111"}"#,
            r#"{"timestamp":"2026-03-14T10:00:00.020Z","level":"INFO","span":{"name":"corvia.store.insert"},"fields":{},"elapsed_ms":10.0,"otel.trace_id":"aaaa","otel.span_id":"3333","otel.parent_span_id":"1111"}"#,
        ];
        let trees = collect_trace_trees(&lines, 10);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].trace_id, "aaaa");
        assert_eq!(trees[0].root_span, "corvia.entry.write");
        assert_eq!(trees[0].span_count, 3);
        assert!(trees[0].total_ms >= 50.0);

        // Verify parent-child tree structure
        assert_eq!(trees[0].spans.len(), 1, "should have 1 root span");
        let root = &trees[0].spans[0];
        assert_eq!(root.span_name, "corvia.entry.write");
        assert_eq!(root.depth, 0);
        assert_eq!(root.children.len(), 2, "root should have 2 children");
        assert_eq!(root.children[0].depth, 1);
        assert_eq!(root.children[1].depth, 1);
    }

    #[test]
    fn collect_trace_trees_skips_lines_without_trace_id() {
        let lines = vec![
            r#"{"timestamp":"2026-03-14T10:00:00Z","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{},"elapsed_ms":10.0}"#,
        ];
        let trees = collect_trace_trees(&lines, 10);
        assert!(trees.is_empty());
    }
}
