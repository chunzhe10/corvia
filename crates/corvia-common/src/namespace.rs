use serde::{Deserialize, Serialize};
use std::fmt;
use crate::errors::{CorviaError, Result};

/// Five-segment hierarchical namespace (D17).
/// Format: {org}:{scope_id}:{workstream}:{source}:{version_ref}
/// Version ref: @{hash} (immutable) or :{label} (mutable)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace {
    pub org: String,
    pub scope: String,
    pub workstream: String,
    pub source: String,
    pub version_ref: VersionRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionRef {
    Immutable(String),
    Mutable(String),
}

impl Namespace {
    pub fn new(org: &str, scope: &str, workstream: &str, source: &str, version_ref: VersionRef) -> Self {
        Self {
            org: org.to_string(),
            scope: scope.to_string(),
            workstream: workstream.to_string(),
            source: source.to_string(),
            version_ref,
        }
    }

    /// Default local namespace for single-user use.
    pub fn local(scope: &str, source: &str) -> Self {
        Self::new("local", scope, "main", source, VersionRef::Mutable("latest".into()))
    }

    /// Parse from colon-separated string at system boundaries.
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(5, ':').collect();
        if parts.len() != 5 {
            return Err(CorviaError::Config(format!(
                "Invalid namespace '{}': expected 5 colon-separated segments", s
            )));
        }
        let version_ref = if parts[4].starts_with('@') {
            VersionRef::Immutable(parts[4][1..].to_string())
        } else if parts[4].starts_with(':') {
            VersionRef::Mutable(parts[4][1..].to_string())
        } else {
            VersionRef::Mutable(parts[4].to_string())
        };
        Ok(Self {
            org: parts[0].to_string(),
            scope: parts[1].to_string(),
            workstream: parts[2].to_string(),
            source: parts[3].to_string(),
            version_ref,
        })
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ref_str = match &self.version_ref {
            VersionRef::Immutable(hash) => format!("@{hash}"),
            VersionRef::Mutable(label) => label.clone(),
        };
        write!(f, "{}:{}:{}:{}:{}", self.org, self.scope, self.workstream, self.source, ref_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_local_default() {
        let ns = Namespace::local("my-repo", "code");
        assert_eq!(ns.org, "local");
        assert_eq!(ns.workstream, "main");
        assert_eq!(ns.version_ref, VersionRef::Mutable("latest".into()));
    }

    #[test]
    fn test_namespace_parse_immutable() {
        let ns = Namespace::parse("local:project-alpha:main:my-repo:@abc123").unwrap();
        assert_eq!(ns.org, "local");
        assert_eq!(ns.scope, "project-alpha");
        assert_eq!(ns.workstream, "main");
        assert_eq!(ns.source, "my-repo");
        assert_eq!(ns.version_ref, VersionRef::Immutable("abc123".into()));
    }

    #[test]
    fn test_namespace_parse_mutable() {
        let ns = Namespace::parse("local:project-alpha:main:my-repo::latest").unwrap();
        assert_eq!(ns.version_ref, VersionRef::Mutable("latest".into()));
    }

    #[test]
    fn test_namespace_display_roundtrip() {
        let ns = Namespace::new("acme", "prod", "feature-x", "contracts", VersionRef::Immutable("def456".into()));
        let s = ns.to_string();
        let parsed = Namespace::parse(&s).unwrap();
        assert_eq!(ns, parsed);
    }

    #[test]
    fn test_namespace_parse_invalid() {
        assert!(Namespace::parse("only:two:parts").is_err());
    }
}
