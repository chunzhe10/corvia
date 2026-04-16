//! OTLP JSON file exporter: writes span data as OTLP-compatible JSON lines
//! to `.corvia/traces.jsonl`.
//!
//! Each completed span produces one line using the OTLP span object format
//! (without the resourceSpans/scopeSpans wrapper):
//! ```json
//! {"traceId":"abc123","spanId":"def456","parentSpanId":"","name":"corvia.search","startTimeUnixNano":1234567890000000000,"endTimeUnixNano":1234567890045000000,"attributes":[{"key":"query_len","value":{"intValue":"24"}}],"status":{"code":"OK"}}
//! ```
//!
//! Only spans whose name starts with `corvia.` are recorded. The file is
//! automatically rotated (truncated) at 10 MB.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use futures_util::future::BoxFuture;
use opentelemetry::trace::Status;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};

/// Maximum trace file size before rotation (10 MB).
const MAX_TRACE_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// An OpenTelemetry SpanExporter that writes OTLP JSON lines to a file.
///
/// Thread-safe via `Mutex<BufWriter<File>>`. Auto-creates directories and
/// rotates the file at 10 MB.
#[derive(Debug)]
pub struct OtlpFileExporter {
    path: PathBuf,
    writer: Mutex<BufWriter<fs::File>>,
}

impl OtlpFileExporter {
    /// Create a new OtlpFileExporter writing to the given path.
    ///
    /// Creates parent directories and the file if they do not exist.
    pub fn new(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

/// Convert a `SystemTime` to nanoseconds since Unix epoch.
fn system_time_to_nanos(t: std::time::SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Serialize an OTel `Value` into OTLP JSON attribute value format.
fn value_to_otlp_json(value: &opentelemetry::Value) -> serde_json::Value {
    match value {
        opentelemetry::Value::Bool(b) => serde_json::json!({"boolValue": b}),
        opentelemetry::Value::I64(n) => serde_json::json!({"intValue": n.to_string()}),
        opentelemetry::Value::F64(n) => serde_json::json!({"doubleValue": n}),
        opentelemetry::Value::String(s) => {
            serde_json::json!({"stringValue": s.as_str()})
        }
        opentelemetry::Value::Array(_) => {
            // Arrays are uncommon in our spans; serialize as string fallback.
            serde_json::json!({"stringValue": format!("{:?}", value)})
        }
        _ => serde_json::json!({"stringValue": format!("{:?}", value)})
    }
}

/// Serialize a `SpanData` into an OTLP-compatible JSON line.
fn span_to_otlp_json(span: &SpanData) -> serde_json::Value {
    let trace_id = format!("{}", span.span_context.trace_id());
    let span_id = format!("{}", span.span_context.span_id());
    let parent_span_id = format!("{}", span.parent_span_id);

    let start_nanos = system_time_to_nanos(span.start_time);
    let end_nanos = system_time_to_nanos(span.end_time);

    let attributes: Vec<serde_json::Value> = span
        .attributes
        .iter()
        .map(|kv| {
            serde_json::json!({
                "key": kv.key.as_str(),
                "value": value_to_otlp_json(&kv.value),
            })
        })
        .collect();

    let status = match &span.status {
        Status::Unset => serde_json::json!({"code": "UNSET"}),
        Status::Ok => serde_json::json!({"code": "OK"}),
        Status::Error { description } => {
            serde_json::json!({"code": "ERROR", "message": description.as_ref()})
        }
    };

    serde_json::json!({
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": span.name.as_ref(),
        "startTimeUnixNano": start_nanos,
        "endTimeUnixNano": end_nanos,
        "attributes": attributes,
        "status": status,
    })
}

impl SpanExporter for OtlpFileExporter {
    fn export(&mut self, batch: Vec<SpanData>) -> BoxFuture<'static, OTelSdkResult> {
        // Filter to only corvia spans, serialize, and write synchronously.
        let mut lines = Vec::new();
        for span in &batch {
            if span.name.starts_with("corvia.") {
                let json = span_to_otlp_json(span);
                lines.push(json);
            }
        }

        let path = self.path.clone();
        if let Ok(mut writer) = self.writer.lock() {
            for line in &lines {
                let _ = writeln!(writer, "{}", line);
            }
            let _ = writer.flush();

            // Rotate if the file exceeds the size limit.
            if let Ok(meta) = fs::metadata(&path) {
                if meta.len() > MAX_TRACE_FILE_SIZE {
                    let old_path = path.with_extension("jsonl.old");
                    let _ = fs::rename(&path, &old_path);
                    if let Ok(file) = fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        *writer = BufWriter::new(file);
                    }
                }
            }
        }

        Box::pin(std::future::ready(Ok(())))
    }

    fn shutdown(&mut self) -> OTelSdkResult {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
        Ok(())
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
        Ok(())
    }
}

/// A parsed trace entry from the OTLP JSON trace file.
#[derive(Debug, Clone)]
pub struct ParsedTrace {
    pub name: String,
    pub elapsed_ms: u64,
    pub timestamp_ns: u64,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Parse an OTLP attribute value object into a plain serde_json::Value.
fn parse_otlp_attribute_value(value: &serde_json::Value) -> serde_json::Value {
    if let Some(s) = value.get("stringValue").and_then(|v| v.as_str()) {
        serde_json::Value::String(s.to_string())
    } else if let Some(s) = value.get("intValue") {
        // OTLP encodes intValue as a string.
        if let Some(s) = s.as_str() {
            if let Ok(n) = s.parse::<i64>() {
                serde_json::json!(n)
            } else {
                serde_json::Value::String(s.to_string())
            }
        } else if let Some(n) = s.as_i64() {
            serde_json::json!(n)
        } else {
            s.clone()
        }
    } else if let Some(n) = value.get("doubleValue").and_then(|v| v.as_f64()) {
        serde_json::json!(n)
    } else if let Some(b) = value.get("boolValue").and_then(|v| v.as_bool()) {
        serde_json::json!(b)
    } else {
        serde_json::Value::Null
    }
}

/// Read the last N lines from a traces.jsonl file.
///
/// Parses the OTLP JSON format and returns `ParsedTrace` objects, most recent last.
pub fn read_recent_traces(
    path: &std::path::Path,
    count: usize,
) -> Vec<ParsedTrace> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(count);

    lines[start..]
        .iter()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;

            let name = v.get("name")?.as_str()?.to_string();
            let start_ns = v.get("startTimeUnixNano")?.as_u64()?;
            let end_ns = v.get("endTimeUnixNano")?.as_u64()?;
            let elapsed_ms = (end_ns.saturating_sub(start_ns)) / 1_000_000;

            let mut attributes = HashMap::new();
            if let Some(attrs) = v.get("attributes").and_then(|a| a.as_array()) {
                for attr in attrs {
                    if let (Some(key), Some(val)) = (
                        attr.get("key").and_then(|k| k.as_str()),
                        attr.get("value"),
                    ) {
                        attributes.insert(key.to_string(), parse_otlp_attribute_value(val));
                    }
                }
            }

            Some(ParsedTrace {
                name,
                elapsed_ms,
                timestamp_ns: end_ns,
                attributes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_recent_traces_missing_file() {
        let traces = read_recent_traces(std::path::Path::new("/nonexistent/traces.jsonl"), 10);
        assert!(traces.is_empty());
    }

    #[test]
    fn read_recent_traces_parses_otlp_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traces.jsonl");

        let lines = vec![
            r#"{"traceId":"abc","spanId":"111","parentSpanId":"","name":"corvia.search","startTimeUnixNano":1000000000,"endTimeUnixNano":1010000000,"attributes":[{"key":"result_count","value":{"intValue":"5"}}],"status":{"code":"OK"}}"#,
            r#"{"traceId":"abc","spanId":"222","parentSpanId":"","name":"corvia.write","startTimeUnixNano":2000000000,"endTimeUnixNano":2020000000,"attributes":[{"key":"action","value":{"stringValue":"created"}}],"status":{"code":"OK"}}"#,
            r#"{"traceId":"abc","spanId":"333","parentSpanId":"","name":"corvia.ingest","startTimeUnixNano":3000000000,"endTimeUnixNano":3030000000,"attributes":[],"status":{"code":"OK"}}"#,
        ];
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let traces = read_recent_traces(&path, 2);
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].name, "corvia.write");
        assert_eq!(traces[0].elapsed_ms, 20); // (2020000000 - 2000000000) / 1_000_000
        assert_eq!(
            traces[0].attributes.get("action"),
            Some(&serde_json::json!("created"))
        );
        assert_eq!(traces[1].name, "corvia.ingest");
        assert_eq!(traces[1].elapsed_ms, 30);
    }

    #[test]
    fn read_recent_traces_parses_int_attributes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traces.jsonl");

        let line = r#"{"traceId":"t","spanId":"s","parentSpanId":"","name":"corvia.search","startTimeUnixNano":0,"endTimeUnixNano":45000000,"attributes":[{"key":"result_count","value":{"intValue":"5"}},{"key":"confidence","value":{"stringValue":"high"}}],"status":{"code":"OK"}}"#;
        fs::write(&path, format!("{line}\n")).unwrap();

        let traces = read_recent_traces(&path, 1);
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].elapsed_ms, 45);
        assert_eq!(
            traces[0].attributes.get("result_count"),
            Some(&serde_json::json!(5))
        );
        assert_eq!(
            traces[0].attributes.get("confidence"),
            Some(&serde_json::json!("high"))
        );
    }

    #[test]
    fn otlp_file_exporter_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("traces.jsonl");

        let _exporter = OtlpFileExporter::new(path.clone()).unwrap();
        assert!(path.exists());
    }
}
