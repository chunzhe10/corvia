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
        ("knowledge_store", "storage"),
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
    {
        if let Some(elapsed_ms) = v.get("elapsed_ms").and_then(|e| e.as_f64()) {
            return Some(ParsedTrace::Span {
                level,
                timestamp,
                span_name: span_name.to_string(),
                elapsed_ms,
            });
        }
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
        let rest = &ts[t_pos + 1..];
        if rest.len() >= 8 {
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
            } => {
                span_all
                    .entry(span_name.clone())
                    .or_default()
                    .push(elapsed_ms);

                if let Some(epoch) = timestamp_to_epoch(&timestamp) {
                    if epoch >= one_hour_ago {
                        span_1h
                            .entry(span_name.clone())
                            .or_default()
                            .push(elapsed_ms);
                    }
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

        spans.insert(
            name.clone(),
            SpanStats {
                count,
                count_1h,
                avg_ms,
                last_ms,
                errors,
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

/// Read the last `n` lines from a file
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
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
        assert_eq!(target_to_module("corvia_kernel::knowledge_store"), "storage");
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
}
