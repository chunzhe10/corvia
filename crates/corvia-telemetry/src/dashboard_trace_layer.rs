//! A tracing Layer that writes span-close events as JSON lines to a file,
//! enriched with OTEL trace context (trace_id, span_id, parent_span_id)
//! and accurate elapsed timing.
//!
//! This layer exists because `tracing_subscriber::fmt::json()` doesn't include
//! data from span extensions (where OtelFields lives). The dashboard traces
//! page (`/api/dashboard/traces/recent`) parses these JSON lines to build
//! trace trees and module-level span statistics.

use crate::otel_context_layer::OtelFields;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Instant, SystemTime};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Stored in span extensions to track creation time.
struct SpanCreatedAt(Instant);

/// A tracing Layer that appends span-close JSON to a log file for dashboard consumption.
/// Maximum trace log file size before rotation (10 MB).
const MAX_TRACE_LOG_SIZE: u64 = 10 * 1024 * 1024;

/// A tracing Layer that appends span-close JSON to a log file for dashboard consumption.
pub struct DashboardTraceLayer {
    path: PathBuf,
    writer: Mutex<std::io::BufWriter<std::fs::File>>,
}

impl DashboardTraceLayer {
    /// Create a new DashboardTraceLayer writing to the given path.
    /// Creates parent directories and the file if they don't exist.
    pub fn new(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(std::io::BufWriter::new(file)),
        })
    }

    /// Rotate the trace log if it exceeds the size cap.
    /// Renames the current file to `.old` and opens a fresh file.
    fn maybe_rotate(&self) {
        let size = std::fs::metadata(&self.path)
            .map(|m| m.len())
            .unwrap_or(0);
        if size <= MAX_TRACE_LOG_SIZE {
            return;
        }
        let old_path = self.path.with_extension("log.old");
        let _ = std::fs::rename(&self.path, &old_path);
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            if let Ok(mut writer) = self.writer.lock() {
                *writer = std::io::BufWriter::new(file);
            }
        }
    }

    /// Default path: the same directory corvia-dev writes service logs to.
    pub fn default_path() -> PathBuf {
        let dir = std::env::var("CORVIA_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/corvia-dev-logs"));
        dir.join("corvia-traces.log")
    }
}

impl<S> Layer<S> for DashboardTraceLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            // Only track corvia.* spans
            if span.metadata().name().starts_with("corvia.") {
                span.extensions_mut().insert(SpanCreatedAt(Instant::now()));
            }
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = match ctx.span(&id) {
            Some(s) => s,
            None => return,
        };

        let metadata = span.metadata();
        let span_name = metadata.name();

        // Only emit lines for corvia.* spans (our instrumented spans)
        if !span_name.starts_with("corvia.") {
            return;
        }

        let extensions = span.extensions();

        // Read OTEL fields if available
        let (trace_id, span_id, parent_span_id) =
            if let Some(otel) = extensions.get::<OtelFields>() {
                (
                    Some(otel.trace_id.clone()),
                    Some(otel.span_id.clone()),
                    if otel.parent_span_id.is_empty() {
                        None
                    } else {
                        Some(otel.parent_span_id.clone())
                    },
                )
            } else {
                (None, None, None)
            };

        // Compute elapsed time from span creation
        let elapsed_ms = extensions
            .get::<SpanCreatedAt>()
            .map(|created| created.0.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        drop(extensions);

        // Format current time as ISO 8601
        let now = {
            let d = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            let secs = d.as_secs();
            let micros = d.subsec_micros();
            let (y, mo, dy, h, mi, s) = unix_to_utc(secs);
            format!("{y:04}-{mo:02}-{dy:02}T{h:02}:{mi:02}:{s:02}.{micros:06}Z")
        };

        // Build JSON line matching the format parse_trace_line expects
        let mut json = serde_json::json!({
            "timestamp": now,
            "level": metadata.level().as_str(),
            "span": { "name": span_name },
            "spans": [{ "name": span_name }],
            "elapsed_ms": elapsed_ms,
            "target": metadata.target(),
        });

        if let Some(tid) = trace_id {
            json["otel.trace_id"] = serde_json::Value::String(tid);
        }
        if let Some(sid) = span_id {
            json["otel.span_id"] = serde_json::Value::String(sid);
        }
        if let Some(psid) = parent_span_id {
            json["otel.parent_span_id"] = serde_json::Value::String(psid);
        }

        if let Ok(mut writer) = self.writer.lock() {
            let _ = writeln!(writer, "{}", json);
            let _ = writer.flush();
        }

        // Rotate after writing if file exceeds size cap
        self.maybe_rotate();
    }
}

/// Convert Unix epoch seconds to UTC (year, month, day, hour, minute, second).
/// Simplified — no leap second handling, valid for 2000-2099.
fn unix_to_utc(epoch: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = epoch % 86400;
    let h = s / 3600;
    let mi = (s % 3600) / 60;
    let sec = s % 60;

    let mut days = epoch / 86400;
    let mut y = 1970u64;

    loop {
        let days_in_year = if y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400)) {
            366
        } else {
            365
        };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        y += 1;
    }

    let leap = y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400));
    let month_days: [u64; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut mo = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            mo = i as u64 + 1;
            break;
        }
        days -= md;
    }
    if mo == 0 {
        mo = 12;
    }

    (y, mo, days + 1, h, mi, sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_path_uses_tmp() {
        let path = DashboardTraceLayer::default_path();
        assert!(path.to_str().unwrap().contains("corvia-traces.log"));
    }

    #[test]
    fn unix_to_utc_known_date() {
        // 2026-03-19 00:00:00 UTC = 1773878400
        let (y, mo, d, h, mi, s) = unix_to_utc(1773878400);
        assert_eq!((y, mo, d, h, mi, s), (2026, 3, 19, 0, 0, 0));
    }
}
