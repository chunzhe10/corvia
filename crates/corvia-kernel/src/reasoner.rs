use chrono::{DateTime, Utc};
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::traits::{GraphStore, InferenceEngine, QueryableStore};

/// Type of algorithmic check performed by the Reasoner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckType {
    /// Entry with `valid_to` set but no `superseded_by` replacement.
    StaleEntry,
    /// Entry whose `superseded_by` points to a nonexistent entry.
    BrokenChain,
    /// Entry with zero graph edges (no imports, no relations).
    OrphanedNode,
    /// Import edge whose target does not exist in the store.
    DanglingImport,
    /// Circular `depends_on` chain detected.
    DependencyCycle,
    // LLM-powered (opt-in)
    /// Low coverage for a probing topic detected via semantic search.
    SemanticGap,
    /// Two entries with high embedding similarity but different source versions.
    Contradiction,
    // Docs workflow checks
    /// Doc file in wrong directory for its content_role.
    MisplacedDoc,
    /// Two entries with conflicting claims at overlapping valid time ranges from different origins.
    TemporalContradiction,
    /// Topic cluster has only memory-role entries, no formal design doc.
    CoverageGap,
    /// Entry has a content_role value not in the MemoryType mapping table.
    UnmappedContentRole,
}

impl CheckType {
    /// Machine-readable string label for this check type.
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckType::StaleEntry => "stale_entry",
            CheckType::BrokenChain => "broken_chain",
            CheckType::OrphanedNode => "orphaned_node",
            CheckType::DanglingImport => "dangling_import",
            CheckType::DependencyCycle => "dependency_cycle",
            CheckType::SemanticGap => "semantic_gap",
            CheckType::Contradiction => "contradiction",
            CheckType::MisplacedDoc => "misplaced_doc",
            CheckType::TemporalContradiction => "temporal_contradiction",
            CheckType::CoverageGap => "coverage_gap",
            CheckType::UnmappedContentRole => "unmapped_content_role",
        }
    }
}

impl std::str::FromStr for CheckType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "stale" | "stale_entry" => Ok(CheckType::StaleEntry),
            "broken" | "broken_chain" => Ok(CheckType::BrokenChain),
            "orphan" | "orphaned_node" => Ok(CheckType::OrphanedNode),
            "dangling" | "dangling_import" => Ok(CheckType::DanglingImport),
            "cycle" | "dependency_cycle" => Ok(CheckType::DependencyCycle),
            "gap" | "semantic_gap" => Ok(CheckType::SemanticGap),
            "contradiction" => Ok(CheckType::Contradiction),
            "misplaced" | "misplaced_doc" => Ok(CheckType::MisplacedDoc),
            "temporal" | "temporal_contradiction" => Ok(CheckType::TemporalContradiction),
            "coverage" | "coverage_gap" => Ok(CheckType::CoverageGap),
            "unmapped" | "unmapped_content_role" => Ok(CheckType::UnmappedContentRole),
            _ => Err(format!(
                "Unknown check type: {s}. Valid: stale, broken, orphan, dangling, cycle, gap, contradiction, misplaced, temporal, coverage, unmapped"
            )),
        }
    }
}

/// A finding produced by the Reasoner's algorithmic checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub check_type: CheckType,
    pub scope_id: String,
    pub target_ids: Vec<Uuid>,
    pub confidence: f32,
    pub rationale: String,
    pub found_at: DateTime<Utc>,
}

impl Finding {
    /// Create a new finding with the current timestamp.
    pub fn new(
        check_type: CheckType,
        scope_id: impl Into<String>,
        target_ids: Vec<Uuid>,
        confidence: f32,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            check_type,
            scope_id: scope_id.into(),
            target_ids,
            confidence,
            rationale: rationale.into(),
            found_at: Utc::now(),
        }
    }

    /// Convert this finding to a `KnowledgeEntry` with `chunk_type = "finding"`.
    ///
    /// The caller decides whether to store the resulting entry.
    pub fn to_knowledge_entry(&self) -> KnowledgeEntry {
        let content = format!(
            "[{}] {}\nTargets: {}\nConfidence: {:.0}%",
            self.check_type.as_str(),
            self.rationale,
            self.target_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.confidence * 100.0,
        );
        let mut entry = KnowledgeEntry::new(
            content,
            self.scope_id.clone(),
            "reasoner".to_string(),
        );
        entry.metadata.chunk_type = Some("finding".to_string());
        entry
    }
}

/// The Reasoner is a compute module that reads from the store and graph
/// to produce findings about knowledge health. It does NOT store findings;
/// the caller decides what to do with them.
pub struct Reasoner<'a> {
    store: &'a dyn QueryableStore,
    graph: &'a dyn GraphStore,
    docs_rules: Option<corvia_common::config::DocsRulesConfig>,
}

impl<'a> Reasoner<'a> {
    /// Create a new Reasoner with access to the queryable store and graph store.
    pub fn new(store: &'a dyn QueryableStore, graph: &'a dyn GraphStore) -> Self {
        Self { store, graph, docs_rules: None }
    }

    /// Configure docs rules for MisplacedDoc checks.
    pub fn with_docs_rules(mut self, rules: corvia_common::config::DocsRulesConfig) -> Self {
        self.docs_rules = Some(rules);
        self
    }

    /// Run all algorithmic checks on the provided entries. Returns all findings.
    ///
    /// Includes the 5 core checks (stale, broken chain, orphan, dangling, cycle)
    /// plus the 3 docs workflow checks (misplaced doc, coverage gap, temporal
    /// contradiction). MisplacedDoc requires `with_docs_rules()` to be configured;
    /// if absent, that check is silently skipped.
    pub async fn run_all(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        // Core checks
        findings.extend(self.check_stale_entries(entries, scope_id).await?);
        findings.extend(self.check_broken_chains(entries, scope_id).await?);
        findings.extend(self.check_orphaned_nodes(entries, scope_id).await?);
        findings.extend(self.check_dangling_imports(entries, scope_id).await?);
        findings.extend(self.check_cycles(entries, scope_id).await?);
        // Docs workflow checks
        if let Some(ref rules) = self.docs_rules {
            findings.extend(check_misplaced_doc(entries, rules));
        }
        findings.extend(check_coverage_gap(entries, scope_id));
        findings.extend(check_temporal_contradiction(entries, 0.85));
        findings.extend(check_unmapped_content_role(entries, scope_id));
        Ok(findings)
    }

    /// Run a single check type on the provided entries.
    ///
    /// Note: `SemanticGap` and `Contradiction` require `run_llm_checks()` instead.
    /// Calling `run_check` with these types returns an empty vec.
    pub async fn run_check(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
        check: CheckType,
    ) -> Result<Vec<Finding>> {
        match check {
            CheckType::StaleEntry => self.check_stale_entries(entries, scope_id).await,
            CheckType::BrokenChain => self.check_broken_chains(entries, scope_id).await,
            CheckType::OrphanedNode => self.check_orphaned_nodes(entries, scope_id).await,
            CheckType::DanglingImport => self.check_dangling_imports(entries, scope_id).await,
            CheckType::DependencyCycle => self.check_cycles(entries, scope_id).await,
            // LLM checks require an InferenceEngine — use run_llm_checks() instead.
            CheckType::SemanticGap | CheckType::Contradiction => Ok(Vec::new()),
            // Docs workflow checks
            CheckType::MisplacedDoc => {
                if let Some(ref rules) = self.docs_rules {
                    Ok(check_misplaced_doc(entries, rules))
                } else {
                    Ok(Vec::new())
                }
            }
            CheckType::CoverageGap => Ok(check_coverage_gap(entries, scope_id)),
            CheckType::TemporalContradiction => {
                Ok(check_temporal_contradiction(entries, 0.85))
            }
            CheckType::UnmappedContentRole => Ok(check_unmapped_content_role(entries, scope_id)),
        }
    }

    /// Entries with `valid_to` set but no `superseded_by` replacement.
    /// These are "expired" knowledge with no successor — potentially stale.
    async fn check_stale_entries(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let scoped: Vec<&KnowledgeEntry> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id)
            .collect();

        let mut findings = Vec::new();
        for entry in scoped {
            if entry.valid_to.is_some() && entry.superseded_by.is_none() {
                findings.push(Finding::new(
                    CheckType::StaleEntry,
                    scope_id,
                    vec![entry.id],
                    1.0,
                    format!(
                        "Entry {} has valid_to set but no superseded_by replacement",
                        entry.id
                    ),
                ));
            }
        }
        Ok(findings)
    }

    /// Entries where `superseded_by` points to a nonexistent entry.
    async fn check_broken_chains(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let all_ids: HashSet<Uuid> = entries.iter().map(|e| e.id).collect();
        let scoped: Vec<&KnowledgeEntry> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id)
            .collect();

        let mut findings = Vec::new();
        for entry in scoped {
            let Some(successor_id) = entry.superseded_by else {
                continue;
            };
            if all_ids.contains(&successor_id) {
                continue;
            }
            // Double-check via the store in case the entry exists
            // in a different scope or was not provided in the slice.
            let exists = self.store.get(&successor_id).await?.is_some();
            if !exists {
                findings.push(Finding::new(
                    CheckType::BrokenChain,
                    scope_id,
                    vec![entry.id, successor_id],
                    1.0,
                    format!(
                        "Entry {} has superseded_by pointing to nonexistent entry {}",
                        entry.id, successor_id
                    ),
                ));
            }
        }
        Ok(findings)
    }

    /// Entries with zero graph edges (no imports, no relations).
    ///
    /// Entries with `chunk_type` of "finding" or "file" are excluded, as they
    /// legitimately may have no relations.
    async fn check_orphaned_nodes(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let scoped: Vec<&KnowledgeEntry> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id)
            .collect();

        let mut findings = Vec::new();
        for entry in &scoped {
            // Skip chunk types that are structurally orphaned — no edge extraction
            // exists for these types, so flagging them is noise, not signal.
            let chunk_type = entry.metadata.chunk_type.as_deref().unwrap_or("");
            let language = entry.metadata.language.as_deref().unwrap_or("");
            if matches!(chunk_type, "finding" | "file" | "text" | "section")
                || matches!(language, "" | "toml" | "yaml" | "json")
            {
                continue;
            }

            let edges = self.graph.edges(&entry.id, EdgeDirection::Both).await?;
            if edges.is_empty() {
                findings.push(Finding::new(
                    CheckType::OrphanedNode,
                    scope_id,
                    vec![entry.id],
                    0.8, // Slightly lower confidence — orphan status depends on context
                    format!(
                        "Entry {} has zero graph edges (no relations)",
                        entry.id
                    ),
                ));
            }
        }
        Ok(findings)
    }

    /// Import edges where the target does not exist in the store.
    async fn check_dangling_imports(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let all_ids: HashSet<Uuid> = entries.iter().map(|e| e.id).collect();
        let scoped: Vec<&KnowledgeEntry> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id)
            .collect();

        let mut findings = Vec::new();
        for entry in &scoped {
            let edges = self
                .graph
                .edges(&entry.id, EdgeDirection::Outgoing)
                .await?;
            for edge in edges {
                if edge.relation == "imports" && !all_ids.contains(&edge.to) {
                    // Double-check the store
                    let exists = self.store.get(&edge.to).await?.is_some();
                    if !exists {
                        findings.push(Finding::new(
                            CheckType::DanglingImport,
                            scope_id,
                            vec![entry.id, edge.to],
                            1.0,
                            format!(
                                "Entry {} imports nonexistent entry {}",
                                entry.id, edge.to
                            ),
                        ));
                    }
                }
            }
        }
        Ok(findings)
    }

    /// Detect circular `depends_on` chains using strongly connected components.
    ///
    /// Builds a mini directed graph from all `depends_on` edges among scoped entries
    /// and uses Kosaraju's SCC algorithm. Any SCC with more than one node is a cycle.
    async fn check_cycles(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let scoped_ids: HashSet<Uuid> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id)
            .map(|e| e.id)
            .collect();

        // Build a petgraph from depends_on edges
        let mut dep_graph = petgraph::graph::DiGraph::<Uuid, ()>::new();
        let mut node_map: HashMap<Uuid, petgraph::graph::NodeIndex> = HashMap::new();

        // Collect all depends_on edges for scoped entries
        let mut has_depends_on = false;
        for id in &scoped_ids {
            let edges: Vec<GraphEdge> = self
                .graph
                .edges(id, EdgeDirection::Outgoing)
                .await?;
            for edge in edges {
                if edge.relation == "depends_on" && scoped_ids.contains(&edge.to) {
                    has_depends_on = true;
                    let from_idx = *node_map
                        .entry(edge.from)
                        .or_insert_with(|| dep_graph.add_node(edge.from));
                    let to_idx = *node_map
                        .entry(edge.to)
                        .or_insert_with(|| dep_graph.add_node(edge.to));
                    dep_graph.add_edge(from_idx, to_idx, ());
                }
            }
        }

        if !has_depends_on {
            return Ok(Vec::new());
        }

        // Find strongly connected components (Kosaraju's algorithm)
        let sccs = petgraph::algo::kosaraju_scc(&dep_graph);

        let mut findings = Vec::new();
        for scc in sccs {
            if scc.len() > 1 {
                // Multi-node cycle (A -> B -> ... -> A)
                let cycle_ids: Vec<Uuid> = scc
                    .iter()
                    .map(|idx| dep_graph[*idx])
                    .collect();
                let desc = cycle_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ");
                findings.push(Finding::new(
                    CheckType::DependencyCycle,
                    scope_id,
                    cycle_ids,
                    1.0,
                    format!(
                        "Dependency cycle detected among {} entries: {}",
                        scc.len(),
                        desc,
                    ),
                ));
            } else if scc.len() == 1 {
                // Check for self-loop (A -> A)
                let idx = scc[0];
                if dep_graph.contains_edge(idx, idx) {
                    let id = dep_graph[idx];
                    findings.push(Finding::new(
                        CheckType::DependencyCycle,
                        scope_id,
                        vec![id],
                        1.0,
                        format!("Self-referential dependency: entry {} depends on itself", id),
                    ));
                }
            }
        }
        Ok(findings)
    }

    // ---- LLM-powered checks (opt-in) ----

    /// Run LLM-powered checks. Requires an InferenceEngine.
    /// Returns empty vec if no entries are provided.
    pub async fn run_llm_checks(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
        engine: &dyn InferenceEngine,
    ) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        findings.extend(self.check_semantic_gaps(scope_id, engine).await?);
        findings.extend(self.check_contradictions(entries, scope_id).await?);
        Ok(findings)
    }

    /// Check for topics with sparse coverage in the knowledge store.
    ///
    /// Embeds a set of probing queries and searches the store for each.
    /// If the best match score is below a threshold, the topic is flagged
    /// as a semantic gap.
    async fn check_semantic_gaps(
        &self,
        scope_id: &str,
        engine: &dyn InferenceEngine,
    ) -> Result<Vec<Finding>> {
        let probes = [
            "error handling and recovery",
            "authentication and authorization",
            "database access patterns",
            "API endpoint design",
            "testing strategy",
            "configuration management",
            "logging and observability",
            "security considerations",
        ];

        let mut findings = Vec::new();
        for probe in &probes {
            let embedding = engine.embed(probe).await?;
            let results = self.store.search(&embedding, scope_id, 3).await?;

            let best_score = results.first().map(|r| r.score).unwrap_or(0.0);
            if best_score < 0.3 {
                findings.push(Finding::new(
                    CheckType::SemanticGap,
                    scope_id,
                    vec![],
                    0.7, // heuristic confidence
                    format!(
                        "Low coverage for topic '{}' — best match score {:.2} (threshold: 0.30)",
                        probe, best_score
                    ),
                ));
            }
        }
        Ok(findings)
    }

    /// Detect potential contradictions among entries in the same scope.
    ///
    /// Uses pre-computed embeddings from stored entries (does NOT call the
    /// InferenceEngine). Grouped with LLM checks because it relies on
    /// embedding quality, which is an LLM-dependent property.
    ///
    /// Compares embeddings pairwise. If two entries have high cosine
    /// similarity (>0.85) but different `source_version` values (suggesting
    /// different time periods), they are flagged as potential contradictions.
    async fn check_contradictions(
        &self,
        entries: &[KnowledgeEntry],
        scope_id: &str,
    ) -> Result<Vec<Finding>> {
        let scoped: Vec<&KnowledgeEntry> = entries
            .iter()
            .filter(|e| e.scope_id == scope_id && e.embedding.is_some())
            .collect();

        let mut findings = Vec::new();
        // Cap pairwise comparisons to avoid O(n^2) explosion on large scopes.
        // With 1000 pairs, scopes up to ~45 entries get full coverage.
        // For larger scopes, findings are not exhaustive.
        let max_pairs = 1000;
        let mut pair_count = 0;

        for i in 0..scoped.len() {
            if pair_count >= max_pairs {
                break;
            }
            for j in (i + 1)..scoped.len() {
                if pair_count >= max_pairs {
                    break;
                }
                pair_count += 1;

                let a = scoped[i];
                let b = scoped[j];

                let emb_a = a.embedding.as_ref().unwrap();
                let emb_b = b.embedding.as_ref().unwrap();

                let similarity = cosine_similarity(emb_a, emb_b);

                // High similarity + different source versions = potential contradiction
                if similarity > 0.85 && a.source_version != b.source_version {
                    findings.push(Finding::new(
                        CheckType::Contradiction,
                        scope_id,
                        vec![a.id, b.id],
                        0.6, // lower confidence — heuristic
                        format!(
                            "Potential contradiction: entries {} and {} have {:.0}% embedding similarity but different source versions ({} vs {})",
                            a.id, b.id, similarity * 100.0, a.source_version, b.source_version
                        ),
                    ));
                }
            }
        }
        Ok(findings)
    }
}

/// Check for doc files placed in blocked paths per workspace rules.
///
/// Compares each entry's `source_file` against the `blocked_paths` globs
/// in the `DocsRulesConfig`. Any match is a misplaced doc.
///
/// **Scope:** This is a blocklist check only — it catches files in explicitly
/// forbidden directories but does not validate that a file's `content_role` is
/// consistent with its directory location. The `repo_docs_pattern` field in
/// `DocsRulesConfig` is reserved for future positive-placement validation.
pub fn check_misplaced_doc(
    entries: &[KnowledgeEntry],
    rules: &corvia_common::config::DocsRulesConfig,
) -> Vec<Finding> {
    let blocked: Vec<glob::Pattern> = rules
        .blocked_paths
        .iter()
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect();

    let mut findings = Vec::new();
    for entry in entries {
        let Some(ref source_file) = entry.metadata.source_file else {
            continue;
        };
        for pattern in &blocked {
            if pattern.matches(source_file) {
                findings.push(Finding::new(
                    CheckType::MisplacedDoc,
                    &entry.scope_id,
                    vec![entry.id],
                    1.0,
                    format!(
                        "File '{}' is in blocked path '{}'",
                        source_file, pattern
                    ),
                ));
            }
        }
    }
    findings
}

/// Check if a scope has only memory-role entries with no formal documentation.
///
/// Flags when entries exist with `content_role = "memory"` but none with
/// `design`, `decision`, or `plan` roles — suggesting knowledge exists only
/// as transient memory without formalization.
///
/// **Design note:** The spec describes a cluster-based approach ("cluster by embedding,
/// flag clusters with only memory role"). This simplified implementation does a
/// scope-global check; embedding-based per-topic clustering can be added as a
/// follow-up optimization.
pub fn check_coverage_gap(entries: &[KnowledgeEntry], scope_id: &str) -> Vec<Finding> {
    let scoped: Vec<&KnowledgeEntry> = entries
        .iter()
        .filter(|e| e.scope_id == scope_id)
        .collect();

    let memory_only: Vec<&&KnowledgeEntry> = scoped
        .iter()
        .filter(|e| e.metadata.content_role.as_deref() == Some("memory"))
        .collect();

    let has_formal = scoped.iter().any(|e| {
        matches!(
            e.metadata.content_role.as_deref(),
            Some("design" | "decision" | "plan")
        )
    });

    if !memory_only.is_empty() && !has_formal {
        vec![Finding::new(
            CheckType::CoverageGap,
            scope_id,
            memory_only.iter().map(|e| e.id).collect(),
            0.7,
            format!(
                "{} memory entries found but no design/decision/plan docs — consider formalizing",
                memory_only.len()
            ),
        )]
    } else {
        Vec::new()
    }
}

/// Check for entries with content_role values not in the MemoryType mapping table.
///
/// Warns about entries that have a content_role set but it doesn't map to any known
/// MemoryType. These entries will fall back to Episodic classification, which may not
/// be correct.
pub fn check_unmapped_content_role(entries: &[KnowledgeEntry], scope_id: &str) -> Vec<Finding> {
    let unmapped: Vec<&KnowledgeEntry> = entries
        .iter()
        .filter(|e| e.scope_id == scope_id)
        .filter(|e| {
            if let Some(ref role) = e.metadata.content_role {
                !corvia_common::types::MemoryType::is_known_content_role(role)
            } else {
                false
            }
        })
        .collect();

    if unmapped.is_empty() {
        return Vec::new();
    }

    let roles: std::collections::BTreeSet<&str> = unmapped
        .iter()
        .filter_map(|e| e.metadata.content_role.as_deref())
        .collect();

    vec![Finding::new(
        CheckType::UnmappedContentRole,
        scope_id,
        unmapped.iter().map(|e| e.id).collect(),
        0.6,
        format!(
            "{} entries have unmapped content_role values: {} — these default to Episodic memory type",
            unmapped.len(),
            roles.into_iter().collect::<Vec<_>>().join(", ")
        ),
    )]
}

/// Check for potential contradictions between current entries from different source origins.
///
/// Uses cosine similarity on pre-computed embeddings. Entries with high similarity
/// (>= threshold) from different `source_origin` values are flagged, since they
/// likely describe the same topic but may contain conflicting information.
///
/// **Design note:** The spec calls for overlapping `valid_from`/`valid_to` window checks.
/// This simplified implementation filters to `is_current()` entries instead — all current
/// entries implicitly overlap temporally (they are all "now"). A future refinement could
/// compare actual temporal windows for superseded entries.
///
/// Entries where both `source_origin` values are `None` are skipped (legacy/untagged
/// entries cannot be meaningfully compared cross-origin).
///
/// Pairwise comparisons are capped at 1000 pairs to bound O(n²) cost, matching the
/// existing `Contradiction` check's cap.
pub fn check_temporal_contradiction(
    entries: &[KnowledgeEntry],
    threshold: f32,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    // Only compare entries that have a source_origin set and are current
    let current: Vec<&KnowledgeEntry> = entries
        .iter()
        .filter(|e| e.is_current() && e.metadata.source_origin.is_some())
        .collect();

    let max_pairs = 1000;
    let mut pair_count = 0;

    for i in 0..current.len() {
        if pair_count >= max_pairs {
            break;
        }
        for j in (i + 1)..current.len() {
            if pair_count >= max_pairs {
                break;
            }
            pair_count += 1;

            let a = current[i];
            let b = current[j];

            // Must have different source_origin
            if a.metadata.source_origin == b.metadata.source_origin {
                continue;
            }

            // Must both have embeddings for similarity check
            let (Some(ea), Some(eb)) = (&a.embedding, &b.embedding) else {
                continue;
            };

            let sim = cosine_similarity(ea, eb);
            if sim >= threshold {
                findings.push(Finding::new(
                    CheckType::TemporalContradiction,
                    &a.scope_id,
                    vec![a.id, b.id],
                    0.6, // heuristic confidence (consistent with Contradiction check)
                    format!(
                        "High similarity ({:.2}) between entries from '{}' and '{}' — possible contradiction",
                        sim,
                        a.metadata.source_origin.as_deref().unwrap_or("unknown"),
                        b.metadata.source_origin.as_deref().unwrap_or("unknown"),
                    ),
                ));
            }
        }
    }
    findings
}

/// Compute cosine similarity between two embedding vectors.
/// Both vectors must have the same dimensionality.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "embedding vectors must have equal dimensions");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite_store::LiteStore;
    use crate::traits::{GraphStore, QueryableStore};
    use corvia_common::types::KnowledgeEntry;

    /// Helper: create a LiteStore in a tempdir with schema initialized.
    async fn test_store() -> (tempfile::TempDir, LiteStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = LiteStore::open(dir.path(), 3).unwrap();
        store.init_schema().await.unwrap();
        (dir, store)
    }

    /// Helper: create a KnowledgeEntry with a dummy 3-dim embedding.
    fn entry(content: &str, scope_id: &str) -> KnowledgeEntry {
        KnowledgeEntry::new(content.into(), scope_id.into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0])
    }

    // ---- Check 1: Stale entries ----

    #[tokio::test]
    async fn test_check_stale_entries() {
        let (_dir, store) = test_store().await;

        let mut e = entry("stale knowledge", "scope");
        e.valid_to = Some(Utc::now());
        // No superseded_by — this is stale
        store.insert(&e).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_stale_entries(&[e.clone()], "scope")
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::StaleEntry);
        assert_eq!(findings[0].target_ids, vec![e.id]);
        assert_eq!(findings[0].confidence, 1.0);
    }

    #[tokio::test]
    async fn test_check_stale_entries_not_stale() {
        let (_dir, store) = test_store().await;

        let mut e = entry("superseded knowledge", "scope");
        e.valid_to = Some(Utc::now());
        e.superseded_by = Some(Uuid::now_v7()); // Has replacement — not stale
        store.insert(&e).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_stale_entries(&[e], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "entry with superseded_by should not be flagged as stale");
    }

    // ---- Check 2: Broken chains ----

    #[tokio::test]
    async fn test_check_broken_chains() {
        let (_dir, store) = test_store().await;

        let nonexistent = Uuid::now_v7();
        let mut e = entry("broken chain", "scope");
        e.superseded_by = Some(nonexistent);
        store.insert(&e).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_broken_chains(&[e.clone()], "scope")
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::BrokenChain);
        assert_eq!(findings[0].target_ids, vec![e.id, nonexistent]);
    }

    #[tokio::test]
    async fn test_check_broken_chains_valid() {
        let (_dir, store) = test_store().await;

        let e1 = entry("original", "scope");
        let e2 = entry("replacement", "scope");
        let mut e1_mut = e1.clone();
        e1_mut.superseded_by = Some(e2.id);
        store.insert(&e1_mut).await.unwrap();
        store.insert(&e2).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_broken_chains(&[e1_mut, e2], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "valid supersession chain should not produce findings");
    }

    // ---- Check 3: Orphaned nodes ----

    #[tokio::test]
    async fn test_check_orphaned_nodes() {
        let (_dir, store) = test_store().await;

        let mut e = entry("lonely node", "scope");
        e.metadata.language = Some("rs".into()); // Must have a language to not be skipped
        store.insert(&e).await.unwrap();
        // No graph edges created — this entry is an orphan

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_orphaned_nodes(&[e.clone()], "scope")
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::OrphanedNode);
        assert_eq!(findings[0].target_ids, vec![e.id]);
        assert_eq!(findings[0].confidence, 0.8);
    }

    #[tokio::test]
    async fn test_check_orphaned_nodes_skips_findings_and_files() {
        let (_dir, store) = test_store().await;

        let mut e_finding = entry("a finding", "scope");
        e_finding.metadata.chunk_type = Some("finding".to_string());
        store.insert(&e_finding).await.unwrap();

        let mut e_file = entry("a whole file", "scope");
        e_file.metadata.chunk_type = Some("file".to_string());
        store.insert(&e_file).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_orphaned_nodes(&[e_finding, e_file], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "finding and file chunk types should not be flagged as orphans");
    }

    #[tokio::test]
    async fn test_check_orphaned_nodes_with_edges() {
        let (_dir, store) = test_store().await;

        let e1 = entry("connected A", "scope");
        let e2 = entry("connected B", "scope");
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_orphaned_nodes(&[e1, e2], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "entries with graph edges should not be orphans");
    }

    // ---- Check 4: Dangling imports ----

    #[tokio::test]
    async fn test_check_dangling_imports() {
        let (_dir, store) = test_store().await;

        let e = entry("importer", "scope");
        let ghost = Uuid::now_v7();
        store.insert(&e).await.unwrap();
        // Create an import edge to a nonexistent entry
        store.relate(&e.id, "imports", &ghost, None).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_dangling_imports(&[e.clone()], "scope")
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::DanglingImport);
        assert_eq!(findings[0].target_ids, vec![e.id, ghost]);
    }

    #[tokio::test]
    async fn test_check_dangling_imports_valid() {
        let (_dir, store) = test_store().await;

        let e1 = entry("importer", "scope");
        let e2 = entry("imported", "scope");
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();
        store.relate(&e1.id, "imports", &e2.id, None).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_dangling_imports(&[e1, e2], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "valid import should not be flagged as dangling");
    }

    // ---- Check 5: Dependency cycles ----

    #[tokio::test]
    async fn test_check_cycles() {
        let (_dir, store) = test_store().await;

        let a = entry("module A", "scope");
        let b = entry("module B", "scope");
        let c = entry("module C", "scope");
        store.insert(&a).await.unwrap();
        store.insert(&b).await.unwrap();
        store.insert(&c).await.unwrap();

        // Create a cycle: A -> B -> C -> A
        store.relate(&a.id, "depends_on", &b.id, None).await.unwrap();
        store.relate(&b.id, "depends_on", &c.id, None).await.unwrap();
        store.relate(&c.id, "depends_on", &a.id, None).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_cycles(&[a.clone(), b.clone(), c.clone()], "scope")
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::DependencyCycle);
        // The SCC should contain all 3 nodes
        assert_eq!(findings[0].target_ids.len(), 3);
        let cycle_set: HashSet<Uuid> = findings[0].target_ids.iter().cloned().collect();
        assert!(cycle_set.contains(&a.id));
        assert!(cycle_set.contains(&b.id));
        assert!(cycle_set.contains(&c.id));
    }

    #[tokio::test]
    async fn test_check_cycles_no_cycle() {
        let (_dir, store) = test_store().await;

        let a = entry("module A", "scope");
        let b = entry("module B", "scope");
        store.insert(&a).await.unwrap();
        store.insert(&b).await.unwrap();

        // Linear dependency, no cycle
        store.relate(&a.id, "depends_on", &b.id, None).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_cycles(&[a, b], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "linear dependency should not be a cycle");
    }

    // ---- run_all ----

    #[tokio::test]
    async fn test_run_all() {
        let (_dir, store) = test_store().await;

        // Create a stale entry (valid_to set, no superseded_by)
        let mut stale = entry("stale", "scope");
        stale.valid_to = Some(Utc::now());
        store.insert(&stale).await.unwrap();

        // Create an orphaned entry (no edges) — must have language to not be skipped
        let mut orphan = entry("orphan", "scope");
        orphan.metadata.language = Some("rs".into());
        store.insert(&orphan).await.unwrap();

        // Also set language on stale entry so it's eligible for orphan check
        stale.metadata.language = Some("rs".into());

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .run_all(&[stale.clone(), orphan.clone()], "scope")
            .await
            .unwrap();

        // Should have at least: 1 stale + 2 orphaned (both entries have no edges)
        let stale_count = findings
            .iter()
            .filter(|f| f.check_type == CheckType::StaleEntry)
            .count();
        let orphan_count = findings
            .iter()
            .filter(|f| f.check_type == CheckType::OrphanedNode)
            .count();
        assert_eq!(stale_count, 1);
        assert_eq!(orphan_count, 2);
    }

    // ---- run_check (single) ----

    #[tokio::test]
    async fn test_run_check_single() {
        let (_dir, store) = test_store().await;

        let mut stale = entry("stale", "scope");
        stale.valid_to = Some(Utc::now());
        store.insert(&stale).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);

        // Run only the stale check
        let stale_findings = reasoner
            .run_check(&[stale.clone()], "scope", CheckType::StaleEntry)
            .await
            .unwrap();
        assert_eq!(stale_findings.len(), 1);

        // Run the broken chain check — should find nothing for this data
        let chain_findings = reasoner
            .run_check(&[stale], "scope", CheckType::BrokenChain)
            .await
            .unwrap();
        assert!(chain_findings.is_empty());
    }

    // ---- Finding -> KnowledgeEntry ----

    #[tokio::test]
    async fn test_finding_to_knowledge_entry() {
        let target_id = Uuid::now_v7();
        let finding = Finding::new(
            CheckType::StaleEntry,
            "my-scope",
            vec![target_id],
            1.0,
            "test rationale",
        );

        let ke = finding.to_knowledge_entry();
        assert_eq!(ke.scope_id, "my-scope");
        assert_eq!(ke.source_version, "reasoner");
        assert_eq!(ke.metadata.chunk_type.as_deref(), Some("finding"));
        assert!(ke.content.contains("[stale_entry]"));
        assert!(ke.content.contains("test rationale"));
        assert!(ke.content.contains(&target_id.to_string()));
        assert!(ke.content.contains("100%"));
    }

    // ---- Scope filtering ----

    #[tokio::test]
    async fn test_checks_respect_scope_filter() {
        let (_dir, store) = test_store().await;

        // Stale entry in a different scope should not appear
        let mut stale_other = entry("stale in other scope", "other-scope");
        stale_other.valid_to = Some(Utc::now());
        store.insert(&stale_other).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_stale_entries(&[stale_other], "scope")
            .await
            .unwrap();
        assert!(findings.is_empty(), "entries in a different scope should be excluded");
    }

    // ---- LLM-powered checks ----

    /// Mock engine that returns deterministic embeddings for testing.
    struct MockEngine;

    #[async_trait::async_trait]
    impl InferenceEngine for MockEngine {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            // Return a simple deterministic embedding (3-dim to match test store)
            Ok(vec![0.1, 0.1, 0.1])
        }
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1, 0.1, 0.1]).collect())
        }
        fn dimensions(&self) -> usize {
            3
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should have similarity -1.0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should yield similarity 0.0");
    }

    #[tokio::test]
    async fn test_semantic_gap_detection() {
        let (_dir, store) = test_store().await;

        // Store is empty — all probes should return low scores → semantic gaps
        let reasoner = Reasoner::new(&store, &store);
        let engine = MockEngine;

        let findings = reasoner
            .check_semantic_gaps("scope", &engine)
            .await
            .unwrap();

        // All 8 probes should produce a finding (empty store = zero scores)
        assert_eq!(findings.len(), 8, "empty store should flag all probes as semantic gaps");
        for f in &findings {
            assert_eq!(f.check_type, CheckType::SemanticGap);
            assert_eq!(f.confidence, 0.7);
            assert!(f.target_ids.is_empty());
            assert!(f.rationale.contains("Low coverage"));
        }
    }

    #[tokio::test]
    async fn test_contradiction_detection() {
        let (_dir, store) = test_store().await;

        // Create two entries with identical embeddings but different source_version
        let embedding = vec![1.0, 0.0, 0.0];
        let mut a = KnowledgeEntry::new("auth uses JWT".into(), "scope".into(), "v1".into())
            .with_embedding(embedding.clone());
        let mut b = KnowledgeEntry::new("auth uses OAuth".into(), "scope".into(), "v2".into())
            .with_embedding(embedding.clone());
        // Assign unique IDs so they're different entries
        a.id = Uuid::now_v7();
        b.id = Uuid::now_v7();

        store.insert(&a).await.unwrap();
        store.insert(&b).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_contradictions(&[a.clone(), b.clone()], "scope")
            .await
            .unwrap();

        assert_eq!(findings.len(), 1, "identical embeddings + different versions = contradiction");
        assert_eq!(findings[0].check_type, CheckType::Contradiction);
        assert_eq!(findings[0].confidence, 0.6);
        assert_eq!(findings[0].target_ids.len(), 2);
        assert!(findings[0].target_ids.contains(&a.id));
        assert!(findings[0].target_ids.contains(&b.id));
        assert!(findings[0].rationale.contains("100%")); // 100% similarity
    }

    #[tokio::test]
    async fn test_contradiction_same_version_not_flagged() {
        let (_dir, store) = test_store().await;

        // Same embedding, same version — NOT a contradiction
        let embedding = vec![1.0, 0.0, 0.0];
        let a = KnowledgeEntry::new("same topic A".into(), "scope".into(), "v1".into())
            .with_embedding(embedding.clone());
        let b = KnowledgeEntry::new("same topic B".into(), "scope".into(), "v1".into())
            .with_embedding(embedding.clone());

        store.insert(&a).await.unwrap();
        store.insert(&b).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_contradictions(&[a, b], "scope")
            .await
            .unwrap();

        assert!(findings.is_empty(), "same version should not be flagged as contradiction");
    }

    #[tokio::test]
    async fn test_contradiction_low_similarity_not_flagged() {
        let (_dir, store) = test_store().await;

        // Different embeddings, different versions — not similar enough to be a contradiction
        let a = KnowledgeEntry::new("topic A".into(), "scope".into(), "v1".into())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let b = KnowledgeEntry::new("topic B".into(), "scope".into(), "v2".into())
            .with_embedding(vec![0.0, 1.0, 0.0]);

        store.insert(&a).await.unwrap();
        store.insert(&b).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner
            .check_contradictions(&[a, b], "scope")
            .await
            .unwrap();

        assert!(findings.is_empty(), "low similarity should not be flagged as contradiction");
    }

    #[tokio::test]
    async fn test_llm_checks_with_empty_entries() {
        let (_dir, store) = test_store().await;

        let reasoner = Reasoner::new(&store, &store);
        let engine = MockEngine;

        let findings = reasoner
            .run_llm_checks(&[], "scope", &engine)
            .await
            .unwrap();

        // Semantic gaps should still be found (store queries), but no contradictions (no entries)
        let gap_count = findings
            .iter()
            .filter(|f| f.check_type == CheckType::SemanticGap)
            .count();
        let contradiction_count = findings
            .iter()
            .filter(|f| f.check_type == CheckType::Contradiction)
            .count();
        assert_eq!(gap_count, 8, "semantic gaps should be detected even with no entries");
        assert_eq!(contradiction_count, 0, "no entries means no contradictions");
    }

    #[tokio::test]
    async fn test_run_check_llm_types_return_empty() {
        let (_dir, store) = test_store().await;
        let reasoner = Reasoner::new(&store, &store);

        // SemanticGap and Contradiction should return empty from run_check
        let gap_findings = reasoner
            .run_check(&[], "scope", CheckType::SemanticGap)
            .await
            .unwrap();
        assert!(gap_findings.is_empty(), "SemanticGap via run_check should return empty");

        let contradiction_findings = reasoner
            .run_check(&[], "scope", CheckType::Contradiction)
            .await
            .unwrap();
        assert!(contradiction_findings.is_empty(), "Contradiction via run_check should return empty");
    }

    #[test]
    fn test_check_type_as_str_llm_variants() {
        assert_eq!(CheckType::SemanticGap.as_str(), "semantic_gap");
        assert_eq!(CheckType::Contradiction.as_str(), "contradiction");
    }

    // ---- MisplacedDoc check ----

    fn test_docs_rules() -> corvia_common::config::DocsRulesConfig {
        corvia_common::config::DocsRulesConfig {
            blocked_paths: vec!["docs/superpowers/*".into()],
            repo_docs_pattern: Some("docs/".into()),
        }
    }

    // ---- TemporalContradiction check ----

    #[test]
    fn test_temporal_contradiction_skips_without_embeddings() {
        let mut e1 = KnowledgeEntry::new("auth uses JWT".into(), "test".into(), "v1".into());
        e1.metadata.source_origin = Some("repo:backend".into());
        let mut e2 = KnowledgeEntry::new("auth uses session cookies".into(), "test".into(), "v2".into());
        e2.metadata.source_origin = Some("repo:frontend".into());
        // No embeddings
        let findings = check_temporal_contradiction(&[e1, e2], 0.85);
        assert!(findings.is_empty());
    }

    #[test]
    fn test_temporal_contradiction_flags_similar_cross_origin() {
        let fake_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let mut e1 = KnowledgeEntry::new("auth uses JWT".into(), "test".into(), "v1".into());
        e1.metadata.source_origin = Some("repo:backend".into());
        e1.embedding = Some(fake_embedding.clone());

        let mut e2 = KnowledgeEntry::new("auth uses session cookies".into(), "test".into(), "v2".into());
        e2.metadata.source_origin = Some("repo:frontend".into());
        e2.embedding = Some(fake_embedding);

        let findings = check_temporal_contradiction(&[e1, e2], 0.85);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::TemporalContradiction);
    }

    #[test]
    fn test_temporal_contradiction_ignores_same_origin() {
        let fake_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let mut e1 = KnowledgeEntry::new("v1".into(), "test".into(), "v1".into());
        e1.metadata.source_origin = Some("repo:backend".into());
        e1.embedding = Some(fake_embedding.clone());
        let mut e2 = KnowledgeEntry::new("v2".into(), "test".into(), "v2".into());
        e2.metadata.source_origin = Some("repo:backend".into());
        e2.embedding = Some(fake_embedding);

        let findings = check_temporal_contradiction(&[e1, e2], 0.85);
        assert!(findings.is_empty());
    }

    // ---- CoverageGap check ----

    fn make_entry_with_role(role: &str, content: &str) -> KnowledgeEntry {
        let mut e = KnowledgeEntry::new(content.into(), "test".into(), "v1".into());
        e.metadata.content_role = Some(role.into());
        e
    }

    #[test]
    fn test_coverage_gap_check() {
        let entries = vec![
            make_entry_with_role("memory", "topic A discussion"),
            make_entry_with_role("memory", "topic A notes"),
        ];
        let findings = check_coverage_gap(&entries, "test");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::CoverageGap);
    }

    #[test]
    fn test_coverage_gap_no_flag_when_design_exists() {
        let entries = vec![
            make_entry_with_role("memory", "topic A discussion"),
            make_entry_with_role("design", "topic A design spec"),
        ];
        let findings = check_coverage_gap(&entries, "test");
        assert!(findings.is_empty());
    }

    #[test]
    fn test_coverage_gap_no_flag_when_decision_exists() {
        let entries = vec![
            make_entry_with_role("memory", "topic A discussion"),
            make_entry_with_role("decision", "topic A decision"),
        ];
        let findings = check_coverage_gap(&entries, "test");
        assert!(findings.is_empty());
    }

    #[test]
    fn test_coverage_gap_no_flag_when_plan_exists() {
        let entries = vec![
            make_entry_with_role("memory", "topic A discussion"),
            make_entry_with_role("plan", "topic A impl plan"),
        ];
        let findings = check_coverage_gap(&entries, "test");
        assert!(findings.is_empty());
    }

    #[test]
    fn test_temporal_contradiction_excludes_superseded() {
        let fake_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let mut e1 = KnowledgeEntry::new("auth uses JWT".into(), "test".into(), "v1".into());
        e1.metadata.source_origin = Some("repo:backend".into());
        e1.embedding = Some(fake_embedding.clone());
        // Supersede e1 so is_current() returns false
        e1.valid_to = Some(chrono::Utc::now());
        e1.superseded_by = Some(Uuid::now_v7());

        let mut e2 = KnowledgeEntry::new("auth uses cookies".into(), "test".into(), "v2".into());
        e2.metadata.source_origin = Some("repo:frontend".into());
        e2.embedding = Some(fake_embedding);

        let findings = check_temporal_contradiction(&[e1, e2], 0.85);
        assert!(findings.is_empty()); // e1 is superseded, should be excluded
    }

    // ---- MisplacedDoc check ----

    #[test]
    fn test_misplaced_doc_check_blocked() {
        let mut e = KnowledgeEntry::new("design doc".into(), "test".into(), "v1".into());
        e.metadata.content_role = Some("design".into());
        e.metadata.source_file = Some("docs/superpowers/specs/design.md".into());
        e.metadata.source_origin = Some("repo:corvia".into());

        let findings = check_misplaced_doc(&[e], &test_docs_rules());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::MisplacedDoc);
    }

    #[test]
    fn test_misplaced_doc_check_allowed() {
        let mut e = KnowledgeEntry::new("design doc".into(), "test".into(), "v1".into());
        e.metadata.content_role = Some("design".into());
        e.metadata.source_file = Some("repos/corvia/docs/design.md".into());

        let findings = check_misplaced_doc(&[e], &test_docs_rules());
        assert!(findings.is_empty());
    }

    #[test]
    fn test_new_check_types_parse() {
        assert_eq!("misplaced_doc".parse::<CheckType>().unwrap(), CheckType::MisplacedDoc);
        assert_eq!("misplaced".parse::<CheckType>().unwrap(), CheckType::MisplacedDoc);
        assert_eq!("temporal_contradiction".parse::<CheckType>().unwrap(), CheckType::TemporalContradiction);
        assert_eq!("temporal".parse::<CheckType>().unwrap(), CheckType::TemporalContradiction);
        assert_eq!("coverage_gap".parse::<CheckType>().unwrap(), CheckType::CoverageGap);
        assert_eq!("coverage".parse::<CheckType>().unwrap(), CheckType::CoverageGap);
    }

    #[test]
    fn test_new_check_types_as_str() {
        assert_eq!(CheckType::MisplacedDoc.as_str(), "misplaced_doc");
        assert_eq!(CheckType::TemporalContradiction.as_str(), "temporal_contradiction");
        assert_eq!(CheckType::CoverageGap.as_str(), "coverage_gap");
    }

    // ---- End-to-end Reasoner dispatch tests ----

    #[tokio::test]
    async fn test_reasoner_misplaced_doc_via_run_check() {
        let (_dir, store) = test_store().await;

        let mut e = entry("design doc", "test");
        e.metadata.content_role = Some("design".into());
        e.metadata.source_file = Some("docs/superpowers/specs/design.md".into());
        store.insert(&e).await.unwrap();

        let reasoner = Reasoner::new(&store, &store)
            .with_docs_rules(test_docs_rules());
        let findings = reasoner.run_check(&[e], "test", CheckType::MisplacedDoc).await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::MisplacedDoc);
    }

    #[tokio::test]
    async fn test_reasoner_misplaced_doc_without_rules_returns_empty() {
        let (_dir, store) = test_store().await;

        let mut e = entry("design doc", "test");
        e.metadata.source_file = Some("docs/superpowers/specs/design.md".into());
        store.insert(&e).await.unwrap();

        // No with_docs_rules() — should return empty
        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner.run_check(&[e], "test", CheckType::MisplacedDoc).await.unwrap();
        assert!(findings.is_empty());
    }

    #[tokio::test]
    async fn test_reasoner_coverage_gap_via_run_check() {
        let (_dir, store) = test_store().await;

        let mut e1 = entry("topic A discussion", "test");
        e1.metadata.content_role = Some("memory".into());
        let mut e2 = entry("topic A notes", "test");
        e2.metadata.content_role = Some("memory".into());
        store.insert(&e1).await.unwrap();
        store.insert(&e2).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner.run_check(&[e1, e2], "test", CheckType::CoverageGap).await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::CoverageGap);
    }

    // ---- UnmappedContentRole check ----

    #[test]
    fn test_unmapped_content_role_detects_unknown() {
        let mut e = entry("some entry", "test");
        e.metadata.content_role = Some("exotic_role".into());
        let findings = check_unmapped_content_role(&[e], "test");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::UnmappedContentRole);
        assert!(findings[0].rationale.contains("exotic_role"));
    }

    #[test]
    fn test_unmapped_content_role_ignores_known() {
        let mut e = entry("some entry", "test");
        e.metadata.content_role = Some("design".into());
        let findings = check_unmapped_content_role(&[e], "test");
        assert!(findings.is_empty());
    }

    #[test]
    fn test_unmapped_content_role_ignores_none() {
        let e = entry("some entry", "test");
        let findings = check_unmapped_content_role(&[e], "test");
        assert!(findings.is_empty());
    }

    #[test]
    fn test_unmapped_content_role_parse_and_str() {
        assert_eq!(CheckType::UnmappedContentRole.as_str(), "unmapped_content_role");
        assert_eq!("unmapped_content_role".parse::<CheckType>().unwrap(), CheckType::UnmappedContentRole);
        assert_eq!("unmapped".parse::<CheckType>().unwrap(), CheckType::UnmappedContentRole);
    }

    #[tokio::test]
    async fn test_reasoner_unmapped_content_role_via_run_check() {
        let (_dir, store) = test_store().await;

        let mut e = entry("exotic entry", "test");
        e.metadata.content_role = Some("banana".into());
        store.insert(&e).await.unwrap();

        let reasoner = Reasoner::new(&store, &store);
        let findings = reasoner.run_check(&[e], "test", CheckType::UnmappedContentRole).await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check_type, CheckType::UnmappedContentRole);
    }
}
