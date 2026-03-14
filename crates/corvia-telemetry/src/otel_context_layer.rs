//! Custom tracing Layer that bridges OpenTelemetry span context into
//! the fmt JSON output by storing trace_id/span_id in span extensions.

use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Extracted OTEL trace context fields, stored in span extensions.
#[derive(Clone, Debug)]
pub struct OtelFields {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
}

/// A tracing Layer that reads `tracing_opentelemetry::OtelData` from span
/// extensions (populated by the OTEL layer composed earlier) and stores
/// extracted trace/span IDs as `OtelFields` for the fmt layer to read.
///
/// Layer composition order matters: the OTEL layer must be `.with()`-ed
/// before this layer so `OtelData` is in extensions when `on_new_span` fires.
pub struct OtelContextLayer;

impl<S> Layer<S> for OtelContextLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };

        let extensions = span.extensions();

        // tracing-opentelemetry stores OtelData in extensions during on_new_span.
        // OtelData.builder has the span_id and trace_id for this span.
        // OtelData.parent_cx has the parent's span context.
        if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
            use opentelemetry::trace::TraceContextExt;

            let parent_span_ref = otel_data.parent_cx.span();
            let parent_sc = parent_span_ref.span_context();

            let trace_id = otel_data
                .builder
                .trace_id
                .map(|t| format!("{t}"))
                .or_else(|| {
                    if parent_sc.is_valid() {
                        Some(format!("{}", parent_sc.trace_id()))
                    } else {
                        None
                    }
                });

            let span_id = otel_data.builder.span_id.map(|s| format!("{s}"));

            let parent_span_id = if parent_sc.is_valid() {
                format!("{}", parent_sc.span_id())
            } else {
                String::new()
            };

            if let (Some(tid), Some(sid)) = (trace_id, span_id) {
                let fields = OtelFields {
                    trace_id: tid,
                    span_id: sid,
                    parent_span_id,
                };
                drop(extensions);
                span.extensions_mut().insert(fields);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otel_fields_stores_ids() {
        let fields = OtelFields {
            trace_id: "abcdef1234567890abcdef1234567890".to_string(),
            span_id: "1234567890abcdef".to_string(),
            parent_span_id: String::new(),
        };
        assert_eq!(fields.trace_id.len(), 32);
        assert_eq!(fields.span_id.len(), 16);
        assert!(fields.parent_span_id.is_empty());
    }

    #[test]
    fn otel_context_layer_is_constructible() {
        let _layer = OtelContextLayer;
    }
}
