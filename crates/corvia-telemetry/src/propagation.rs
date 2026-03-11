//! W3C trace context propagation adapters for tonic gRPC.

use opentelemetry::propagation::{Extractor, Injector};
use tonic::metadata::MetadataMap;

/// Injects OpenTelemetry context into tonic `MetadataMap` for outgoing gRPC calls.
pub struct MetadataInjector<'a>(pub &'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(name) = tonic::metadata::MetadataKey::from_bytes(key.as_bytes()) {
            if let Ok(val) = value.parse() {
                self.0.insert(name, val);
            }
        }
    }
}

/// Extracts OpenTelemetry context from tonic `MetadataMap` for incoming gRPC calls.
pub struct MetadataExtractor<'a>(pub &'a MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .filter_map(|k| match k {
                tonic::metadata::KeyRef::Ascii(key) => Some(key.as_str()),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    #[test]
    fn test_injector_sets_header() {
        let mut map = MetadataMap::new();
        let mut injector = MetadataInjector(&mut map);
        injector.set("x-custom-header", "test-value".to_string());
        assert_eq!(map.get("x-custom-header").unwrap().to_str().unwrap(), "test-value");
    }

    #[test]
    fn test_extractor_gets_header() {
        let mut map = MetadataMap::new();
        map.insert("x-custom-header", "test-value".parse().unwrap());
        let extractor = MetadataExtractor(&map);
        assert_eq!(extractor.get("x-custom-header"), Some("test-value"));
    }

    #[test]
    fn test_extractor_missing_key_returns_none() {
        let map = MetadataMap::new();
        let extractor = MetadataExtractor(&map);
        assert_eq!(extractor.get("nonexistent"), None);
    }

    #[test]
    fn test_metadata_injector_extractor_roundtrip() {
        let propagator = TraceContextPropagator::new();

        // Inject into a MetadataMap
        let mut map = MetadataMap::new();
        let cx = opentelemetry::Context::new();
        propagator.inject_context(&cx, &mut MetadataInjector(&mut map));

        // Extract from the MetadataMap
        let _extracted_cx = propagator.extract(&MetadataExtractor(&map));
        // The roundtrip should not panic; with a default (empty) context
        // the propagator may or may not set headers, but the flow must work.
    }
}
