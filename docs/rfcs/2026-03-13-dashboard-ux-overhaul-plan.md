# Dashboard UX Overhaul — Implementation Plan

> **Status:** Shipped (v0.4.3)

**Goal:** Overhaul the corvia dashboard across 5 subsystems: multi-tiered graph LOD, agent registration with reconnect, activity-feed-based history, collapsible sidebar, and agent descriptions with topic drift.

**Architecture:** Server-side `ClusterStore` in `corvia-server/src/dashboard/clustering.rs` computes semantic clusters from entry embeddings, providing a shared topic vocabulary consumed by the graph, activity feed, and agent summaries. Frontend components (GraphView, HistoryView, Layout, AgentsView) are rewritten to consume the new APIs. Agent registration moves from passive MCP detection to explicit CLI-driven identity selection.

**Tech Stack:** Rust (axum 0.8, redb, tokio, rand), TypeScript/Preact (Vite), k-means clustering on 768-dim nomic-embed-text embeddings

**Note:** Cluster hierarchy is stored in-memory (`Arc<RwLock>`) and recomputed on server start. Redb persistence is deferred — recomputation is fast (~11 MB at current scale) and simplifies the initial implementation.

**Repo structure:** Backend code lives in the `repos/corvia/` git repo. Frontend/hooks live in the workspace root git repo. Subagents must branch in the correct repo.

**Design spec:** `2026-03-13-dashboard-ux-overhaul-design.md`

---

## File Map

### Backend (repos/corvia/)

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/corvia-server/src/dashboard/clustering.rs` | Create | ClusterStore: k-means, silhouette, cluster hierarchy, background recompute |
| `crates/corvia-server/src/dashboard/activity.rs` | Create | Activity feed endpoint: recent entries, semantic grouping, content deltas |
| `crates/corvia-server/src/dashboard/mod.rs` | Modify | Wire new endpoints, inject ClusterStore into AppState |
| `crates/corvia-common/src/agent_types.rs` | Modify | Add `description`, `ActivitySummary` to AgentRecord |
| `crates/corvia-kernel/src/agent_registry.rs` | Modify | Persist/retrieve new AgentRecord fields |
| `crates/corvia-kernel/src/agent_coordinator.rs` | Modify | Activity summary computation on session close, reconnectable endpoint logic |
| `crates/corvia-cli/src/commands/agent.rs` | Modify | Add `corvia agent connect` interactive CLI command |
| `crates/corvia-server/src/rest.rs` | Modify | Add `ClusterStore` to `AppState`, wire new dashboard routes |
| `crates/corvia-server/Cargo.toml` | Modify | Add `rand = "0.8"` dependency |

### Frontend (tools/corvia-dashboard/src/)

| File | Action | Responsibility |
|------|--------|---------------|
| `components/GraphView.tsx` | Modify | Multi-tiered LOD rendering, zoom-driven level switching, breadcrumbs, viewport culling |
| `components/HistoryView.tsx` | Rewrite | Activity feed default, semantic groups, topic filters, entry detail in sidebar |
| `components/Layout.tsx` | Modify | Collapsible sidebar with 3 states, auto-show rules, gear icon for config |
| `components/AgentsView.tsx` | Modify | Topic tag pills, drift indicator, description subtitle |
| `api.ts` | Modify | New fetch functions for clustered graph, activity feed, reconnectable agents |
| `types.ts` | Modify | New types for ClusterNode, ActivityItem, ActivitySummary |

### Hook Scripts

| File | Action | Responsibility |
|------|--------|---------------|
| `.claude/hooks/agent-check.sh` | Create | SessionStart display-only reminder for agent identity |

---

## Chunk 1: ClusterStore Backend

### Task 0: Add `rand` Dependency and `mod clustering`

**Files:**
- Modify: `crates/corvia-server/Cargo.toml`
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Add `rand` to corvia-server dependencies**

In `crates/corvia-server/Cargo.toml`, add under `[dependencies]`:
```toml
rand = "0.8"
```

-**Step 2: Add `mod clustering;` and `mod activity;` to dashboard module**

In `crates/corvia-server/src/dashboard/mod.rs`, add near the top with other module declarations:
```rust
pub mod clustering;
pub mod activity;
```

-**Step 3: Verify it compiles (files will be empty initially)**

Create empty files `clustering.rs` and `activity.rs` with placeholder content.
Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo check -p corvia-server 2>&1 | tail -5`

-**Step 4: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/Cargo.toml crates/corvia-server/src/dashboard/mod.rs crates/corvia-server/src/dashboard/clustering.rs crates/corvia-server/src/dashboard/activity.rs
git commit -m "chore(server): add rand dependency and clustering/activity modules"
```

### Task 1: K-means and Silhouette Core

**Files:**
- Modify: `crates/corvia-server/src/dashboard/clustering.rs`

-**Step 1: Write failing test for k-means on known data**

In `clustering.rs`, add a `#[cfg(test)] mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_two_obvious_clusters() {
        // Two well-separated 3D clusters
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![0.05, 0.05, 0.05],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 10.1, 10.1],
            vec![9.95, 9.95, 9.95],
        ];
        let assignments = kmeans(&data, 2, 100);
        // First 3 should be same cluster, last 3 should be same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }
}
```

-**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server clustering::tests::test_kmeans --no-default-features 2>&1 | tail -5`
Expected: FAIL — `kmeans` function not found

-**Step 3: Implement k-means**

```rust
use std::collections::HashMap;

/// Assign each vector to one of k clusters. Returns cluster assignments (0..k).
pub fn kmeans(data: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<usize> {
    let n = data.len();
    let dim = data[0].len();
    if n <= k {
        return (0..n).collect();
    }

    // Initialize centroids with k-means++ seeding
    let mut centroids = kmeans_pp_init(data, k);
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        // Assign step
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let nearest = nearest_centroid(point, &centroids);
            if nearest != assignments[i] {
                assignments[i] = nearest;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, val) in point.iter().enumerate() {
                sums[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
    }
    assignments
}

fn kmeans_pp_init(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut centroids = vec![data[rng.gen_range(0..data.len())].clone()];

    for _ in 1..k {
        let distances: Vec<f32> = data.iter().map(|p| {
            centroids.iter().map(|c| euclidean_dist_sq(p, c)).fold(f32::MAX, f32::min)
        }).collect();
        let total: f32 = distances.iter().sum();
        let threshold = rng.gen::<f32>() * total;
        let mut cumulative = 0.0;
        for (i, &d) in distances.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                centroids.push(data[i].clone());
                break;
            }
        }
    }
    centroids
}

fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids.iter().enumerate()
        .min_by(|(_, a), (_, b)| euclidean_dist_sq(point, a).partial_cmp(&euclidean_dist_sq(point, b)).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn euclidean_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
    dot / (mag_a * mag_b)
}
```

-**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server clustering::tests::test_kmeans --no-default-features 2>&1 | tail -5`
Expected: PASS

-**Step 5: Write failing test for silhouette score**

```rust
#[test]
fn test_silhouette_well_separated() {
    let data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1],
        vec![10.0, 10.0], vec![10.1, 10.0], vec![10.0, 10.1],
    ];
    let assignments = vec![0, 0, 0, 1, 1, 1];
    let score = silhouette_score(&data, &assignments, 2);
    assert!(score > 0.9, "Well-separated clusters should have silhouette > 0.9, got {score}");
}

#[test]
fn test_best_k_finds_obvious_clusters() {
    let data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![0.05, 0.05],
        vec![10.0, 10.0], vec![10.1, 10.0], vec![10.0, 10.1], vec![10.05, 10.05],
        vec![20.0, 0.0], vec![20.1, 0.0], vec![20.0, 0.1], vec![20.05, 0.05],
    ];
    let best = find_best_k(&data, 2, 6, 500);
    assert_eq!(best, 3, "Should find 3 clusters, got {best}");
}
```

-**Step 6: Implement silhouette and find_best_k**

```rust
/// Compute silhouette score for a clustering. Samples up to max_sample points.
pub fn silhouette_score(data: &[Vec<f32>], assignments: &[usize], k: usize) -> f32 {
    use rand::seq::SliceRandom;
    let n = data.len();
    if n <= 1 || k <= 1 { return 0.0; }

    let indices: Vec<usize> = (0..n).collect();
    let sample: Vec<usize> = if n > 500 {
        let mut rng = rand::thread_rng();
        let mut shuffled = indices.clone();
        shuffled.shuffle(&mut rng);
        shuffled.into_iter().take(500).collect()
    } else {
        indices
    };

    let mut total = 0.0f32;
    for &i in &sample {
        let ci = assignments[i];
        // a(i) = mean distance to same cluster
        let mut same_sum = 0.0f32;
        let mut same_count = 0usize;
        // b(i) = min mean distance to other clusters
        let mut other_sums = vec![0.0f32; k];
        let mut other_counts = vec![0usize; k];

        for (j, point) in data.iter().enumerate() {
            if i == j { continue; }
            let d = euclidean_dist_sq(&data[i], point).sqrt();
            if assignments[j] == ci {
                same_sum += d;
                same_count += 1;
            } else {
                other_sums[assignments[j]] += d;
                other_counts[assignments[j]] += 1;
            }
        }

        let a = if same_count > 0 { same_sum / same_count as f32 } else { 0.0 };
        let b = (0..k)
            .filter(|&c| c != ci && other_counts[c] > 0)
            .map(|c| other_sums[c] / other_counts[c] as f32)
            .fold(f32::MAX, f32::min);

        if b == f32::MAX { continue; }
        let s = (b - a) / a.max(b);
        total += s;
    }
    total / sample.len() as f32
}

/// Find best K by trying k_min..=k_max and picking highest silhouette.
pub fn find_best_k(data: &[Vec<f32>], k_min: usize, k_max: usize, max_sample: usize) -> usize {
    let mut best_k = k_min;
    let mut best_score = f32::NEG_INFINITY;
    for k in k_min..=k_max {
        let assignments = kmeans(data, k, 100);
        let score = silhouette_score(data, &assignments, k);
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }
    best_k
}
```

-**Step 7: Run tests to verify they pass**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server clustering::tests --no-default-features 2>&1 | tail -10`
Expected: all 3 tests PASS

-**Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/clustering.rs crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): add k-means clustering with silhouette scoring"
```

### Task 2: ClusterStore — Hierarchy Builder and Background Recompute

**Files:**
- Modify: `crates/corvia-server/src/dashboard/clustering.rs`
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Write failing test for ClusterHierarchy**

```rust
#[test]
fn test_cluster_hierarchy_from_embeddings() {
    // Create 9 entries with clear 3-cluster structure in 4D
    let entries: Vec<(String, Vec<f32>)> = vec![
        ("a1".into(), vec![0.0, 0.0, 0.0, 0.0]),
        ("a2".into(), vec![0.1, 0.0, 0.0, 0.0]),
        ("a3".into(), vec![0.0, 0.1, 0.0, 0.0]),
        ("b1".into(), vec![10.0, 10.0, 0.0, 0.0]),
        ("b2".into(), vec![10.1, 10.0, 0.0, 0.0]),
        ("b3".into(), vec![10.0, 10.1, 0.0, 0.0]),
        ("c1".into(), vec![0.0, 0.0, 10.0, 10.0]),
        ("c2".into(), vec![0.0, 0.0, 10.1, 10.0]),
        ("c3".into(), vec![0.0, 0.0, 10.0, 10.1]),
    ];
    let hierarchy = ClusterHierarchy::build(&entries, 2, 5);
    assert!(hierarchy.super_clusters.len() >= 2 && hierarchy.super_clusters.len() <= 5);
    // Each super-cluster should have 3 entries
    for sc in &hierarchy.super_clusters {
        assert_eq!(sc.entry_ids.len(), 3);
    }
}
```

-**Step 2: Run test, verify failure**

-**Step 3: Implement ClusterHierarchy**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub cluster_id: String,
    pub label: String,
    pub level: u8,
    pub parent_id: Option<String>,
    pub entry_ids: Vec<String>,
    pub centroid: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHierarchy {
    pub super_clusters: Vec<ClusterNode>,   // L0
    pub sub_clusters: Vec<ClusterNode>,     // L1
    pub entry_count: usize,
    pub computed_at: chrono::DateTime<chrono::Utc>,
}

impl ClusterHierarchy {
    /// Build hierarchy from (entry_id, embedding) pairs.
    pub fn build(entries: &[(String, Vec<f32>)], k_min: usize, k_max: usize) -> Self {
        let embeddings: Vec<Vec<f32>> = entries.iter().map(|(_, e)| e.clone()).collect();
        let k_max = k_max.min(entries.len());
        let k_min = k_min.min(k_max);

        // L0: super-clusters
        let best_k = find_best_k(&embeddings, k_min, k_max, 500);
        let l0_assignments = kmeans(&embeddings, best_k, 100);

        let mut super_clusters = Vec::new();
        for c in 0..best_k {
            let member_indices: Vec<usize> = l0_assignments.iter().enumerate()
                .filter(|(_, &a)| a == c).map(|(i, _)| i).collect();
            if member_indices.is_empty() { continue; }

            let entry_ids: Vec<String> = member_indices.iter().map(|&i| entries[i].0.clone()).collect();
            let centroid = compute_centroid(&member_indices.iter().map(|&i| &embeddings[i]).collect::<Vec<_>>());

            // Label from nearest entry to centroid
            let nearest_idx = member_indices.iter()
                .min_by(|&&a, &&b| euclidean_dist_sq(&embeddings[a], &centroid)
                    .partial_cmp(&euclidean_dist_sq(&embeddings[b], &centroid)).unwrap())
                .copied().unwrap();
            let label = entries[nearest_idx].0.clone(); // Will be replaced with content preview by caller

            super_clusters.push(ClusterNode {
                cluster_id: format!("sc-{c}"),
                label,
                level: 0,
                parent_id: None,
                entry_ids,
                centroid,
            });
        }

        // L1: sub-clusters within each super-cluster
        let mut sub_clusters = Vec::new();
        for sc in &super_clusters {
            if sc.entry_ids.len() < 4 { continue; } // Too small to sub-divide
            let sc_embeddings: Vec<Vec<f32>> = sc.entry_ids.iter()
                .map(|id| entries.iter().find(|(eid, _)| eid == id).unwrap().1.clone())
                .collect();
            let sc_entries: Vec<(String, Vec<f32>)> = sc.entry_ids.iter()
                .zip(sc_embeddings.iter())
                .map(|(id, e)| (id.clone(), e.clone()))
                .collect();

            let sub_k_max = (sc.entry_ids.len() / 2).min(8).max(2);
            let sub_k = find_best_k(&sc_embeddings, 2, sub_k_max, 200);
            let sub_assignments = kmeans(&sc_embeddings, sub_k, 100);

            for s in 0..sub_k {
                let member_indices: Vec<usize> = sub_assignments.iter().enumerate()
                    .filter(|(_, &a)| a == s).map(|(i, _)| i).collect();
                if member_indices.is_empty() { continue; }

                let entry_ids: Vec<String> = member_indices.iter().map(|&i| sc_entries[i].0.clone()).collect();
                let centroid = compute_centroid(&member_indices.iter().map(|&i| &sc_embeddings[i]).collect::<Vec<_>>());
                let nearest_idx = member_indices.iter()
                    .min_by(|&&a, &&b| euclidean_dist_sq(&sc_embeddings[a], &centroid)
                        .partial_cmp(&euclidean_dist_sq(&sc_embeddings[b], &centroid)).unwrap())
                    .copied().unwrap();

                sub_clusters.push(ClusterNode {
                    cluster_id: format!("{}-sub-{s}", sc.cluster_id),
                    label: sc_entries[nearest_idx].0.clone(),
                    level: 1,
                    parent_id: Some(sc.cluster_id.clone()),
                    entry_ids,
                    centroid,
                });
            }
        }

        ClusterHierarchy {
            super_clusters,
            sub_clusters,
            entry_count: entries.len(),
            computed_at: chrono::Utc::now(),
        }
    }

    /// Find which super-cluster an entry belongs to.
    pub fn cluster_for_entry(&self, entry_id: &str) -> Option<&ClusterNode> {
        self.super_clusters.iter().find(|sc| sc.entry_ids.contains(&entry_id.to_string()))
    }

    /// Get topic label for an entry (its super-cluster label).
    pub fn topic_for_entry(&self, entry_id: &str) -> Option<&str> {
        self.cluster_for_entry(entry_id).map(|sc| sc.label.as_str())
    }
}

fn compute_centroid(vectors: &[&Vec<f32>]) -> Vec<f32> {
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];
    for v in vectors {
        for (j, val) in v.iter().enumerate() {
            centroid[j] += val;
        }
    }
    let n = vectors.len() as f32;
    centroid.iter_mut().for_each(|x| *x /= n);
    centroid
}
```

-**Step 4: Run test, verify pass**

-**Step 5: Write test for background ClusterStore wrapper**

```rust
#[test]
fn test_cluster_store_degraded_when_empty() {
    let store = ClusterStore::new();
    assert!(store.current().is_none(), "Should be None before first computation");
    assert!(store.is_degraded());
}
```

-**Step 6: Implement ClusterStore wrapper with Arc<RwLock>**

```rust
use std::sync::{Arc, RwLock};

pub struct ClusterStore {
    hierarchy: Arc<RwLock<Option<ClusterHierarchy>>>,
    last_entry_count: Arc<RwLock<usize>>,
}

impl ClusterStore {
    pub fn new() -> Self {
        Self {
            hierarchy: Arc::new(RwLock::new(None)),
            last_entry_count: Arc::new(RwLock::new(0)),
        }
    }

    pub fn current(&self) -> Option<ClusterHierarchy> {
        self.hierarchy.read().unwrap().clone()
    }

    pub fn is_degraded(&self) -> bool {
        self.hierarchy.read().unwrap().is_none()
    }

    /// Update hierarchy if entry count changed. Returns true if recomputed.
    pub fn maybe_recompute(&self, entries: &[(String, Vec<f32>)]) -> bool {
        let current_count = entries.len();
        let last = *self.last_entry_count.read().unwrap();
        if current_count == last && !self.is_degraded() {
            return false;
        }
        if current_count < 3 {
            return false;
        }

        let hierarchy = ClusterHierarchy::build(entries, 3, 12);
        *self.hierarchy.write().unwrap() = Some(hierarchy);
        *self.last_entry_count.write().unwrap() = current_count;
        true
    }
}
```

-**Step 7: Run all clustering tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server clustering::tests --no-default-features 2>&1 | tail -10`
Expected: all tests PASS

-**Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/clustering.rs
git commit -m "feat(dashboard): add ClusterHierarchy builder and ClusterStore wrapper"
```

### Task 3: Add ClusterStore to AppState and Clustered Graph Endpoint

**Files:**
- Modify: `crates/corvia-server/src/rest.rs` (add ClusterStore to AppState)
- Modify: `crates/corvia-server/src/dashboard/mod.rs` (new handler + background task)

-**Step 1: Add ClusterStore to AppState**

In `rest.rs`, find the `AppState` struct definition. Add a new field:

```rust
use crate::dashboard::clustering::ClusterStore;

pub struct AppState {
    // ... existing fields ...
    pub cluster_store: Arc<ClusterStore>,
}
```

In the `AppState` construction (likely in `create_app` or server startup function), add:

```rust
let cluster_store = Arc::new(ClusterStore::new());
```

Pass it into the AppState struct.

-**Step 2: Wire background recompute task**

In the server startup (where `tokio::spawn` is used for merge_worker, etc.), add:

```rust
// Background cluster recompute every 60s
let cluster_store_bg = state.cluster_store.clone();
let store_bg = state.store.clone();
let scope_id_bg = state.default_scope_id.clone();
tokio::spawn(async move {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        // Load entries with embeddings
        let data_dir = /* resolve .corvia path */;
        match corvia_kernel::knowledge_files::read_scope(&data_dir, &scope_id_bg) {
            Ok(entries) => {
                let pairs: Vec<(String, Vec<f32>)> = entries.iter()
                    .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id.to_string(), emb.clone())))
                    .collect();
                if cluster_store_bg.maybe_recompute(&pairs) {
                    tracing::info!("Cluster hierarchy recomputed ({} entries)", pairs.len());
                }
            }
            Err(e) => tracing::warn!("Cluster recompute failed: {e}"),
        }
    }
});
```

Note: Entry IDs are `uuid::Uuid` in the codebase. Convert to `String` via `.to_string()` when passing to ClusterStore. The ClusterStore uses String IDs internally for simplicity.

-**Step 3: Add clustered graph handler**

In `dashboard/mod.rs`, add a query params struct and handler:

```rust
#[derive(Debug, Deserialize)]
pub struct ClusteredGraphParams {
    pub level: Option<u8>,
    pub parent: Option<String>,
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
}

pub async fn clustered_graph_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ClusteredGraphParams>,
) -> impl IntoResponse {
    let level = params.level.unwrap_or(255); // 255 = no level = legacy behavior

    if level == 255 {
        // Legacy: return existing graph_scope_handler behavior
        return graph_scope_handler(State(state), Query(/* forward existing params */)).await;
    }

    let hierarchy = match state.cluster_store.current() {
        Some(h) => h,
        None => {
            // Degraded mode: return legacy path-based grouping
            return Json(serde_json::json!({
                "nodes": [], "edges": [], "degraded": true
            })).into_response();
        }
    };

    match level {
        0 => {
            // Super-clusters as nodes
            let nodes: Vec<serde_json::Value> = hierarchy.super_clusters.iter().map(|sc| {
                serde_json::json!({
                    "id": sc.cluster_id,
                    "label": sc.label,
                    "level": 0,
                    "entry_count": sc.entry_ids.len(),
                })
            }).collect();

            // Aggregate inter-cluster edges
            let scope_id = &state.default_scope_id;
            let edges = compute_inter_cluster_edges(&hierarchy.super_clusters, &state).await;

            Json(serde_json::json!({ "nodes": nodes, "edges": edges })).into_response()
        }
        1 => {
            // Sub-clusters within a parent super-cluster
            let parent_id = params.parent.as_deref().unwrap_or("");
            let nodes: Vec<serde_json::Value> = hierarchy.sub_clusters.iter()
                .filter(|sc| sc.parent_id.as_deref() == Some(parent_id))
                .map(|sc| serde_json::json!({
                    "id": sc.cluster_id,
                    "label": sc.label,
                    "level": 1,
                    "entry_count": sc.entry_ids.len(),
                    "parent_id": sc.parent_id,
                }))
                .collect();
            Json(serde_json::json!({ "nodes": nodes, "edges": [] })).into_response()
        }
        2 => {
            // File groups within a parent (reuse existing path-based grouping for entries in this cluster)
            // Find the parent cluster, get its entry_ids, then group by source_file prefix
            let parent_id = params.parent.as_deref().unwrap_or("");
            let cluster = hierarchy.sub_clusters.iter()
                .find(|sc| sc.cluster_id == parent_id)
                .or_else(|| hierarchy.super_clusters.iter().find(|sc| sc.cluster_id == parent_id));
            // Build file-group nodes from entry source_file paths
            // (reuse existing buildClusters logic from the current graph_scope_handler)
            Json(serde_json::json!({ "nodes": [], "edges": [], "level": 2 })).into_response()
        }
        _ => {
            Json(serde_json::json!({ "error": "Invalid level" })).into_response()
        }
    }
}
```

-**Step 4: Add label enrichment at response time**

In the L0 node builder above, look up the nearest-centroid entry to get a human-readable label. The entries are already loaded by `knowledge_files::read_scope()`:

```rust
// Build entry map for label lookup
let entries = knowledge_files::read_scope(&data_dir, scope_id).unwrap_or_default();
let entry_map: HashMap<String, &KnowledgeEntry> = entries.iter()
    .map(|e| (e.id.to_string(), e)).collect();

// For each cluster, find the nearest-centroid entry and use its source_file or content preview
for node in &mut nodes {
    let cluster = hierarchy.super_clusters.iter().find(|sc| sc.cluster_id == node["id"]).unwrap();
    if let Some(entry) = cluster.entry_ids.iter().find_map(|id| entry_map.get(id.as_str())) {
        let label = entry.source_file.clone()
            .unwrap_or_else(|| entry.content.chars().take(60).collect());
        node["label"] = serde_json::Value::String(label);
    }
}
```

-**Step 5: Wire route**

In `dashboard/mod.rs` router setup, add alongside existing graph route:
```rust
.route("/api/dashboard/graph/scope", get(clustered_graph_handler))
```
This replaces the existing `graph_scope_handler` route — the new handler falls through to legacy behavior when no `level` param is provided.

-**Step 6: Test endpoint manually**

Run: `curl -s http://localhost:8020/api/dashboard/graph/scope?level=0 | jq '.nodes | length'`
Expected: a number between 3 and 12 (super-cluster count)

Run: `curl -s http://localhost:8020/api/dashboard/graph/scope | jq '.nodes | length'`
Expected: same as before (backward compatible, returns all entries)

-**Step 7: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/rest.rs crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): add ClusterStore to AppState and clustered graph endpoint with LOD levels"
```

### Task 4: Graph LOD Frontend

**Files:**
- Modify: `tools/corvia-dashboard/src/components/GraphView.tsx`
- Modify: `tools/corvia-dashboard/src/api.ts`
- Modify: `tools/corvia-dashboard/src/types.ts`

-**Step 1: Add types for clustered graph**

In `types.ts`, add:

```typescript
export interface ClusterGraphNode {
  id: string;
  label: string;
  level: number;
  entry_count?: number;
  preview?: string;
  source_file?: string;
  group?: string;
  content_role?: string;
  source_origin?: string;
}
```

-**Step 2: Add API function for clustered graph**

In `api.ts`, add:

```typescript
export async function fetchClusteredGraph(level: number, parent?: string): Promise<GraphScopeResponse> {
  let url = `${BASE}/graph/scope?level=${level}`;
  if (parent) url += `&parent=${encodeURIComponent(parent)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
  return res.json();
}
```

-**Step 3: Add zoom-level state to GraphView**

Add state tracking for current LOD level, breadcrumb trail, and last zoom for hysteresis:

```typescript
const [lodLevel, setLodLevel] = useState(0);
const [breadcrumbs, setBreadcrumbs] = useState<Array<{level: number, id?: string, label: string}>>([{level: 0, label: 'All'}]);
const lastZoomRef = useRef(1.0);
```

-**Step 4: Replace data fetch with level-aware fetch**

Replace the existing `fetchGraphScope` polling with `fetchClusteredGraph(lodLevel, parentId)`. When clustering is degraded (endpoint returns `{ degraded: true }`), fall back to existing path-based fetch.

-**Step 5: Add zoom-driven LOD switching**

In the `onWheel` handler, after computing new zoom:

```typescript
const newLevel = zoomToLevel(newZoom, lastZoomRef.current);
if (newLevel !== lodLevel) {
  setLodLevel(newLevel);
  // Trigger re-fetch at new level
}

function zoomToLevel(zoom: number, lastZoom: number): number {
  const HYSTERESIS = 0.05;
  // Only switch if we crossed a threshold by more than hysteresis
  if (zoom < 0.8 - (lastZoom >= 0.8 ? HYSTERESIS : 0)) return 0;
  if (zoom < 1.5 - (lastZoom >= 1.5 ? HYSTERESIS : 0)) return 1;
  if (zoom < 3.0 - (lastZoom >= 3.0 ? HYSTERESIS : 0)) return 2;
  return 3;
}
```

-**Step 6: Add breadcrumb bar**

Render a breadcrumb bar above the canvas:

```tsx
<div class="graph-breadcrumbs">
  {breadcrumbs.map((bc, i) => (
    <span key={i}>
      {i > 0 && ' → '}
      <button onClick={() => navigateToLevel(bc.level, bc.id)}>
        {bc.label}
      </button>
    </span>
  ))}
</div>
```

-**Step 7: Add viewport culling at L3**

At entry level (L3), filter nodes to only those within the visible canvas bounds + 20% margin before running the force simulation:

```typescript
function cullToViewport(nodes: GraphNode[], pan: {x:number,y:number}, zoom: number, canvasW: number, canvasH: number): GraphNode[] {
  const margin = 0.2;
  const left = -pan.x / zoom - canvasW * margin / zoom;
  const right = (canvasW - pan.x) / zoom + canvasW * margin / zoom;
  const top = -pan.y / zoom - canvasH * margin / zoom;
  const bottom = (canvasH - pan.y) / zoom + canvasH * margin / zoom;
  return nodes.filter(n => n.x >= left && n.x <= right && n.y >= top && n.y <= bottom);
}
```

-**Step 8: Test in browser**

Open `http://localhost:8021`, navigate to Graph tab. Verify:
1. Default view shows 5-12 super-cluster nodes
2. Zooming in transitions through LOD levels
3. Double-click drills into next level
4. Breadcrumbs appear and are clickable
5. Zooming to L3 does not crash with full dataset

-**Step 9: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/GraphView.tsx tools/corvia-dashboard/src/api.ts tools/corvia-dashboard/src/types.ts
git commit -m "feat(dashboard): multi-tiered LOD graph with zoom-driven level switching"
```

---

## Chunk 2: Agent Registration & Descriptions (Sections 2 + 5)

### Task 5: Extend AgentRecord with Description and ActivitySummary

**Files:**
- Modify: `crates/corvia-common/src/agent_types.rs`
- Modify: `crates/corvia-kernel/src/agent_registry.rs`

-**Step 1: Write failing test for new fields**

In `agent_registry.rs` tests:

```rust
#[test]
fn test_agent_description_and_summary() {
    let reg = test_registry();
    reg.register("test::agent", "Test Agent", IdentityType::Registered,
        AgentPermission::ReadOnly).unwrap();

    // Set description
    reg.set_description("test::agent", "working on graph refactor").unwrap();
    let agent = reg.get("test::agent").unwrap().unwrap();
    assert_eq!(agent.description.as_deref(), Some("working on graph refactor"));

    // Set activity summary
    let summary = ActivitySummary {
        entry_count: 12,
        topic_tags: vec!["graph store".into(), "edge handling".into()],
        last_topics: vec!["merge pipeline".into()],
        last_active: chrono::Utc::now(),
        session_count: 3,
    };
    reg.set_activity_summary("test::agent", &summary).unwrap();
    let agent = reg.get("test::agent").unwrap().unwrap();
    let summary = agent.activity_summary.as_ref().expect("activity_summary should be set");
    assert_eq!(summary.entry_count, 12);
    assert_eq!(summary.topic_tags.len(), 2);
}
```

-**Step 2: Run test, verify failure**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-kernel test_agent_description_and_summary 2>&1 | tail -5`
Expected: FAIL — fields don't exist

-**Step 3: Add fields to AgentRecord**

In `agent_types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySummary {
    pub entry_count: u64,
    pub topic_tags: Vec<String>,
    pub last_topics: Vec<String>,
    pub last_active: DateTime<Utc>,
    pub session_count: u64,
}

// Add to AgentRecord:
pub struct AgentRecord {
    // ... existing fields ...
    pub description: Option<String>,
    pub activity_summary: Option<ActivitySummary>,
}
```

-**Step 4: Add persistence methods to AgentRegistry**

In `agent_registry.rs`, add `set_description` and `set_activity_summary` methods that read the record, update the field, and write it back to Redb. Follow the existing pattern used by `set_status` and `touch`.

-**Step 5: Run test, verify pass**

-**Step 6: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-common/src/agent_types.rs crates/corvia-kernel/src/agent_registry.rs
git commit -m "feat(agent): add description and activity_summary to AgentRecord"
```

### Task 6: Reconnectable Agents Endpoint

**Files:**
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs`
- Modify: `crates/corvia-server/src/dashboard/mod.rs` or `crates/corvia-server/src/rest.rs`

-**Step 1: Write failing test for reconnectable logic**

In `agent_coordinator.rs` tests (or inline):

```rust
#[test]
fn test_reconnectable_agents() {
    // Setup: create agent, create session, make session stale
    let coord = test_coordinator();
    let identity = AgentIdentity::Registered { agent_id: "test::agent".into(), api_key: None };
    coord.register_agent(&identity, "Test Agent", AgentPermission::ReadWrite { scopes: vec!["test".into()] }).unwrap();
    let session = coord.create_session("test::agent", false).unwrap();

    // Transition to stale
    coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();

    let reconnectable = coord.list_reconnectable().unwrap();
    assert_eq!(reconnectable.len(), 1);
    assert_eq!(reconnectable[0].agent_id, "test::agent");
}
```

-**Step 2: Implement `list_reconnectable` on AgentCoordinator**

```rust
/// List agents that have stale or orphaned sessions (candidates for reconnection).
pub fn list_reconnectable(&self) -> Result<Vec<AgentRecord>> {
    let all_agents = self.registry.list_active()?;
    let mut reconnectable = Vec::new();
    for agent in all_agents {
        let sessions = self.sessions.list_by_agent(&agent.agent_id)?;
        let has_stale_or_orphaned = sessions.iter().any(|s|
            matches!(s.state, SessionState::Stale | SessionState::Orphaned)
        );
        if has_stale_or_orphaned {
            reconnectable.push(agent);
        }
    }
    reconnectable.sort_by(|a, b| b.last_seen.cmp(&a.last_seen));
    Ok(reconnectable)
}
```

-**Step 3: Run test, verify pass**

-**Step 4: Add REST endpoint**

In `dashboard/mod.rs` or `rest.rs`, add:

```rust
async fn reconnectable_agents_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match state.coordinator.list_reconnectable() {
        Ok(agents) => Json(agents).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}
```

Wire as `GET /api/dashboard/agents/reconnectable`.

-**Step 5: Add `POST /api/dashboard/agents/{agent_id}/connect` endpoint**

```rust
#[derive(Deserialize)]
pub struct ConnectAgentRequest {
    pub description: Option<String>,
}

async fn connect_agent_handler(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
    Json(body): Json<ConnectAgentRequest>,
) -> impl IntoResponse {
    // Update description if provided
    if let Some(desc) = &body.description {
        let _ = state.coordinator.registry.set_description(&agent_id, desc);
    }
    match state.coordinator.connect(&agent_id) {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}
```

Wire as `POST /api/dashboard/agents/{agent_id}/connect`.

-**Step 6: Add `POST /api/dashboard/agents/{agent_id}/refresh-summary` endpoint**

```rust
async fn refresh_summary_handler(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
) -> impl IntoResponse {
    // Recompute activity summary for this agent using ClusterStore
    let hierarchy = state.cluster_store.current();
    // Get entries written by this agent (query knowledge store)
    // Compute topic tags and update summary
    // (implementation details in Task 7)
    Json(serde_json::json!({ "status": "refreshed" })).into_response()
}
```

Wire as `POST /api/dashboard/agents/{agent_id}/refresh-summary`.

-**Step 7: Test endpoints**

Run: `curl -s http://localhost:8020/api/dashboard/agents/reconnectable | jq`
Run: `curl -s -X POST http://localhost:8020/api/dashboard/agents/test::agent/connect -H 'Content-Type: application/json' -d '{"description":"test"}' | jq`

-**Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-kernel/src/agent_coordinator.rs crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(agent): add reconnectable, connect, and refresh-summary endpoints"
```

### Task 7: Activity Summary Computation on Session Close

**Files:**
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs`
- Modify: `crates/corvia-server/src/dashboard/clustering.rs` (reuse cosine_similarity)

-**Step 1: Write failing test for topic extraction**

```rust
#[test]
fn test_compute_topic_tags_from_embeddings() {
    // Build a hierarchy with 2 known clusters
    let entries: Vec<(String, Vec<f32>)> = vec![
        ("a1".into(), vec![0.0, 0.0]),
        ("a2".into(), vec![0.1, 0.0]),
        ("b1".into(), vec![10.0, 10.0]),
        ("b2".into(), vec![10.1, 10.0]),
    ];
    let hierarchy = ClusterHierarchy::build(&entries, 2, 4);

    // Agent wrote entries near cluster A
    let agent_embeddings = vec![vec![0.05, 0.05], vec![0.02, 0.03]];
    let tags = compute_topic_tags(&hierarchy, &agent_embeddings);
    assert!(tags.len() >= 1);
    assert!(tags.len() <= 5);
}
```

-**Step 2: Implement topic tag extraction**

For each entry embedding, find the nearest super-cluster centroid. Count occurrences. Return top 3-5 cluster labels.

```rust
pub fn compute_topic_tags(hierarchy: &ClusterHierarchy, embeddings: &[Vec<f32>]) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for emb in embeddings {
        let nearest = hierarchy.super_clusters.iter()
            .min_by(|a, b| euclidean_dist_sq(emb, &a.centroid)
                .partial_cmp(&euclidean_dist_sq(emb, &b.centroid)).unwrap());
        if let Some(sc) = nearest {
            *counts.entry(sc.label.clone()).or_insert(0) += 1;
        }
    }
    let mut tags: Vec<(String, usize)> = counts.into_iter().collect();
    tags.sort_by(|a, b| b.1.cmp(&a.1));
    tags.into_iter().take(5).map(|(label, _)| label).collect()
}
```

-**Step 3: Wire into session close flow**

In the `close_session` or `commit_session` method on `AgentCoordinator`, after transitioning the session to Closed, compute the activity summary:

1. Get all entries written by this agent (from session's `entries_written` or knowledge store query)
2. Get their embeddings
3. Call `compute_topic_tags` with current ClusterHierarchy
4. Compare `last_topics` with existing `topic_tags` for drift detection
5. Update `AgentRecord.activity_summary` via `registry.set_activity_summary()`

Note: The ClusterStore lives in corvia-server, not corvia-kernel. The coordinator needs access to cluster data. Two options:
- Pass cluster hierarchy as a parameter to the close method
- Compute tags at the server layer (in the REST handler that calls close)

Prefer option B (server layer) to avoid kernel→server dependency.

-**Step 4: Run tests, verify pass**

-**Step 5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-kernel/src/agent_coordinator.rs crates/corvia-server/src/dashboard/clustering.rs
git commit -m "feat(agent): compute activity summary with topic tags on session close"
```

### Task 8: Topic Drift Detection

**Files:**
- Modify: `crates/corvia-common/src/agent_types.rs`
- Modify: `crates/corvia-server/src/dashboard/clustering.rs`

-**Step 1: Write failing test for drift detection**

```rust
#[test]
fn test_topic_drift_detected() {
    let historical = vec!["graph store".to_string(), "edge handling".to_string()];
    let current = vec!["merge pipeline".to_string(), "conflict resolution".to_string()];
    assert!(is_topic_drifted(&historical, &current));
}

#[test]
fn test_no_drift_when_overlap() {
    let historical = vec!["graph store".to_string(), "edge handling".to_string()];
    let current = vec!["graph store".to_string(), "traversal".to_string()];
    assert!(!is_topic_drifted(&historical, &current));
}
```

-**Step 2: Implement drift detection**

```rust
/// Returns true if overlap between historical and current topics is < 50%.
pub fn is_topic_drifted(historical: &[String], current: &[String]) -> bool {
    if historical.is_empty() || current.is_empty() {
        return false; // No data to compare
    }
    let overlap = current.iter().filter(|t| historical.contains(t)).count();
    let max_possible = historical.len().max(current.len());
    (overlap as f32 / max_possible as f32) < 0.5
}
```

-**Step 3: Run tests, verify pass**

-**Step 4: Add `drifted` field to ActivitySummary serialization**

Add a computed `drifted: bool` field that is set when updating the summary.

-**Step 5: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-common/src/agent_types.rs crates/corvia-server/src/dashboard/clustering.rs
git commit -m "feat(agent): topic drift detection for activity summaries"
```

### Task 9: CLI `corvia agent connect` Command

**Files:**
- Modify: `crates/corvia-cli/src/commands/agent.rs` (or create if subcommand structure differs)

-**Step 1: Explore existing CLI agent subcommands**

Read `crates/corvia-cli/src/commands/agent.rs` (or wherever agent CLI commands live) to understand the existing pattern for subcommands, HTTP client usage, and output formatting.

-**Step 2: Add `connect` subcommand**

```rust
/// Interactive agent selection for session identity.
async fn agent_connect(client: &HttpClient) -> Result<()> {
    // 1. Fetch reconnectable agents
    let agents: Vec<AgentRecord> = client.get("/api/dashboard/agents/reconnectable").await?;
    // Note: all agent endpoints use /api/dashboard/ prefix per existing convention

    if agents.is_empty() {
        println!("No reconnectable agents found.");
    } else {
        println!("Reconnectable agents:");
        for (i, agent) in agents.iter().enumerate() {
            println!("  [{}] {} ({})", i + 1, agent.display_name, agent.agent_id);
            if let Some(desc) = &agent.description {
                println!("      Purpose: {desc}");
            }
            if let Some(summary) = &agent.activity_summary {
                let tags = summary.topic_tags.join(", ");
                println!("      Activity: {} entries across [{}]", summary.entry_count, tags);
                if !summary.last_topics.is_empty() {
                    let last = summary.last_topics.join(", ");
                    let drifted = is_topic_drifted(&summary.topic_tags, &summary.last_topics);
                    if drifted {
                        println!("      ⚠ Last session drifted to: [{last}]");
                    }
                }
                let ago = humanize_duration(chrono::Utc::now() - summary.last_active);
                println!("      Last active: {ago}");
            }
        }
    }

    println!("  [N] Register new agent");
    print!("Pick one: ");
    // Read stdin
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();

    let agent_id = if input.eq_ignore_ascii_case("n") {
        print!("Agent name: ");
        let mut name = String::new();
        std::io::stdin().read_line(&mut name)?;
        let name = name.trim();
        print!("Purpose (optional): ");
        let mut desc = String::new();
        std::io::stdin().read_line(&mut desc)?;
        let desc = desc.trim();
        // Register new agent
        let resp = client.post("/api/dashboard/agents/register", json!({
            "display_name": name,
            "description": if desc.is_empty() { None } else { Some(desc) },
        })).await?;
        resp.agent_id
    } else {
        let idx: usize = input.parse().map_err(|_| anyhow::anyhow!("Invalid selection"))? ;
        let agent = agents.get(idx - 1).ok_or_else(|| anyhow::anyhow!("Invalid index"))?;
        // Connect existing agent
        client.post(&format!("/api/dashboard/agents/{}/connect", agent.agent_id), json!({})).await?;
        agent.agent_id.clone()
    };

    // Export environment variable
    println!("✓ Connected as {agent_id}. Set CORVIA_AGENT_ID={agent_id}");
    println!("  export CORVIA_AGENT_ID=\"{agent_id}\"");
    Ok(())
}
```

-**Step 3: Test CLI command**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo run -- agent connect`
Expected: Shows list of reconnectable agents or "No reconnectable agents found" + prompt

-**Step 4: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-cli/src/commands/agent.rs
git commit -m "feat(cli): add 'corvia agent connect' for interactive identity selection"
```

### Task 10: SessionStart Hook (Display-Only Reminder)

**Files:**
- Create: `.claude/hooks/agent-check.sh`
- Modify: `.claude/settings.json`

-**Step 1: Create hook script**

```bash
#!/usr/bin/env bash
# Display-only reminder for agent identity selection.
# Runs on Claude Code SessionStart — does NOT require interaction.

CORVIA_API="${CORVIA_API:-http://localhost:8020}"

if [ -n "$CORVIA_AGENT_ID" ]; then
    echo "🔗 Connected as: $CORVIA_AGENT_ID"
    exit 0
fi

# Check for reconnectable agents
COUNT=$(curl -s "$CORVIA_API/api/dashboard/agents/reconnectable" 2>/dev/null | jq 'length' 2>/dev/null)

if [ "$COUNT" -gt "0" ] 2>/dev/null; then
    echo "⚠ Run 'corvia agent connect' to select an identity ($COUNT agents available)"
else
    echo "ℹ No agent identity set. Run 'corvia agent connect' to register."
fi
```

-**Step 2: Add SessionStart hook to settings**

In `.claude/settings.json`, add under the hooks object. The existing file uses event-type keys (e.g., `"PreToolUse": [...]`, `"SessionEnd": [...]`), so add:

```json
"SessionStart": [
  {
    "command": "bash .claude/hooks/agent-check.sh",
    "timeout": 5000
  }
]
```

-**Step 3: Make script executable and test**

Run: `chmod +x .claude/hooks/agent-check.sh && bash .claude/hooks/agent-check.sh`
Expected: One of the three messages depending on state

-**Step 4: Commit**

```bash
cd /workspaces/corvia-workspace
git add .claude/hooks/agent-check.sh .claude/settings.json
git commit -m "feat(hooks): add SessionStart agent identity reminder"
```

### Task 11: Agent Description Display in Dashboard

**Files:**
- Modify: `tools/corvia-dashboard/src/components/AgentsView.tsx`
- Modify: `tools/corvia-dashboard/src/types.ts`

-**Step 1: Update types**

In `types.ts`, add `ActivitySummary` type and extend `AgentRecord`:

```typescript
export interface ActivitySummary {
  entry_count: number;
  topic_tags: string[];
  last_topics: string[];
  last_active: string;
  session_count: number;
  drifted?: boolean;
}

// Add to existing AgentRecord:
export interface AgentRecord {
  // ... existing fields ...
  description?: string;
  activity_summary?: ActivitySummary;
}
```

-**Step 2: Update AgentsView to show description and topics**

In `AgentsView.tsx`, enhance each agent card:

```tsx
{/* After display_name */}
{agent.description && (
  <div class="agent-description">{agent.description}</div>
)}
{agent.activity_summary && (
  <div class="agent-topics">
    {agent.activity_summary.topic_tags.map(tag => (
      <span class="topic-pill">{tag}</span>
    ))}
    {agent.activity_summary.drifted && (
      <span class="drift-indicator" title={`Last session: ${agent.activity_summary.last_topics.join(', ')}`}>
        ⚠ drifted
      </span>
    )}
  </div>
)}
```

-**Step 3: Add CSS for topic pills and drift indicator**

```css
.topic-pill {
  display: inline-block;
  padding: 2px 8px;
  margin: 2px;
  border-radius: 12px;
  background: rgba(147, 130, 220, 0.15); /* lavender from design system */
  color: #9382dc;
  font-size: 11px;
}
.drift-indicator {
  color: #f0a050;
  font-size: 11px;
  margin-left: 4px;
}
.agent-description {
  color: #8899aa;
  font-size: 12px;
  margin-top: 2px;
}
```

-**Step 4: Test in browser**

Open `http://localhost:8021`, go to Agents tab. Verify agent cards show description and topic pills.

-**Step 5: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/AgentsView.tsx tools/corvia-dashboard/src/types.ts
git commit -m "feat(dashboard): show agent description, topic tags, and drift indicator"
```

---

## Chunk 3: Collapsible Right Panel (Section 4)

### Task 12: Sidebar State Machine

**Files:**
- Modify: `tools/corvia-dashboard/src/components/Layout.tsx`

-**Step 1: Define sidebar state types**

```typescript
type SidebarState = 'collapsed' | 'narrow' | 'wide';
type SidebarContent =
  | { kind: 'config' }
  | { kind: 'health' }
  | { kind: 'cluster'; data: any }
  | { kind: 'entry'; data: any }
  | { kind: 'agent'; data: any }
  | { kind: 'finding'; data: any }
  | { kind: 'history'; entryId: string };

const SIDEBAR_WIDTHS: Record<SidebarState, number> = {
  collapsed: 0,
  narrow: 320,
  wide: 480,
};
```

-**Step 2: Replace current sidebar with state-driven sidebar**

Replace the existing always-visible `<aside>` with:

```tsx
const [sidebarState, setSidebarState] = useState<SidebarState>('collapsed');
const [sidebarContent, setSidebarContent] = useState<SidebarContent | null>(null);

function openSidebar(content: SidebarContent, width: SidebarState = 'narrow') {
  setSidebarContent(content);
  setSidebarState(width);
}

function closeSidebar() {
  setSidebarState('collapsed');
  setSidebarContent(null);
}

// Auto-collapse on tab switch
useEffect(() => {
  closeSidebar();
}, [activeTab]);
```

-**Step 3: Render sidebar with transition**

```tsx
<aside
  class="sidebar"
  style={{
    width: `${SIDEBAR_WIDTHS[sidebarState]}px`,
    transition: 'width 200ms ease',
    overflow: 'hidden',
  }}
>
  {sidebarState !== 'collapsed' && (
    <>
      <button class="sidebar-close" onClick={closeSidebar}>✕</button>
      {sidebarContent?.kind === 'config' && <ConfigPanel config={config} />}
      {sidebarContent?.kind === 'health' && <HealthPanel ... />}
      {/* Other content kinds rendered by their respective views */}
    </>
  )}
</aside>

{/* Chevron toggle always visible */}
<button
  class="sidebar-toggle"
  onClick={() => sidebarState === 'collapsed'
    ? openSidebar({ kind: 'config' })
    : closeSidebar()
  }
>
  {sidebarState === 'collapsed' ? '◀' : '▶'}
</button>
```

-**Step 4: Move Config to gear icon, Health to pulse dots**

Replace the existing sidebar tab strip (Config | Health) with:
- A gear icon (`⚙`) in the header bar that calls `openSidebar({ kind: 'config' })`
- The existing health pulse dots already call health — update to use `openSidebar({ kind: 'health' })`

-**Step 5: Pass openSidebar to child components**

Thread `openSidebar` and `closeSidebar` as props (or via Preact context) to GraphView, AgentsView, HistoryView, etc. Each component calls `openSidebar` with the appropriate content kind when the user clicks an interactive element.

-**Step 6: Test in browser**

Verify:
1. Sidebar starts collapsed on page load
2. Clicking gear icon opens Config in narrow mode
3. Clicking health dots opens Health in narrow mode
4. Switching tabs collapses sidebar
5. Close button and clicking empty area collapses

-**Step 7: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/Layout.tsx
git commit -m "feat(dashboard): collapsible context-aware sidebar with auto-show"
```

---

## Chunk 4: History Activity Feed (Section 3)

### Task 13: Activity Feed Backend

**Files:**
- Create: `crates/corvia-server/src/dashboard/activity.rs`
- Modify: `crates/corvia-server/src/dashboard/mod.rs`

-**Step 1: Define ActivityItem response type**

```rust
#[derive(Debug, Serialize)]
pub struct ActivityItem {
    pub entry_id: String,
    pub action: String,           // "wrote", "superseded", "merged"
    pub title: String,            // source_file or content preview (80 chars)
    pub agent_id: Option<String>,
    pub agent_name: Option<String>,
    pub topic_tags: Vec<String>,
    pub delta_bytes: Option<i64>, // positive = addition, negative = deletion
    pub recorded_at: String,
    pub superseded_id: Option<String>,
    pub group_id: Option<String>, // semantic group identifier
    pub group_count: Option<usize>, // how many items in this group
}

#[derive(Debug, Serialize)]
pub struct ActivityFeedResponse {
    pub items: Vec<ActivityItem>,
    pub total: usize,
    pub topics: Vec<String>,      // available topic filters
}
```

-**Step 2: Implement activity feed handler**

```rust
#[derive(Debug, Deserialize)]
pub struct ActivityFeedParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub agent: Option<String>,
    pub topic: Option<String>,
}

// Note: uses State<Arc<AppState>> (from rest.rs), NOT DashboardState
pub async fn activity_feed_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ActivityFeedParams>,
) -> impl IntoResponse {
    let scope_id = &state.default_scope_id;
    let data_dir = &state.data_dir;
    let entries = knowledge_files::read_scope(data_dir, scope_id).unwrap_or_default();

    // Sort by recorded_at descending
    let mut entries = entries;
    entries.sort_by(|a, b| b.recorded_at.cmp(&a.recorded_at));

    // Apply agent filter
    if let Some(ref agent) = params.agent {
        entries.retain(|e| e.metadata.get("agent_id").map(|a| a == agent).unwrap_or(false));
    }

    // Build entry map for content delta lookups (entry_id -> content)
    let entry_map: std::collections::HashMap<String, &str> = entries.iter()
        .map(|e| (e.id.to_string(), e.content.as_str())).collect();

    // Build activity items with topic tags from ClusterStore
    let hierarchy = state.cluster_store.current();

    let mut items: Vec<ActivityItem> = entries.iter()
        .skip(params.offset.unwrap_or(0))
        .take(params.limit.unwrap_or(50))
        .map(|entry| {
            let entry_id_str = entry.id.to_string();
            let topic_tags = hierarchy.as_ref()
                .and_then(|h| h.cluster_for_entry(&entry_id_str).map(|sc| vec![sc.label.clone()]))
                .unwrap_or_default();

            let action = if entry.superseded_by.is_some() { "superseded" }
                else { "wrote" };

            let title = entry.source_file.clone()
                .unwrap_or_else(|| entry.content.chars().take(80).collect());

            // Content delta: UTF-8 byte diff vs predecessor
            // Find predecessor: an entry whose superseded_by == this entry's ID
            let predecessor_content = entries.iter()
                .find(|e| e.superseded_by.as_deref() == Some(&entry_id_str))
                .map(|e| e.content.len() as i64);
            let current_bytes = entry.content.len() as i64;
            let delta_bytes = match predecessor_content {
                Some(prev) => current_bytes - prev,
                None => current_bytes, // New entry, full content is the delta
            };

            ActivityItem {
                entry_id: entry_id_str,
                action: action.to_string(),
                title,
                agent_id: entry.metadata.get("agent_id").cloned(),
                agent_name: entry.metadata.get("agent_name").cloned(),
                topic_tags,
                delta_bytes: Some(delta_bytes),
                recorded_at: entry.recorded_at.to_rfc3339(),
                superseded_id: entry.superseded_by.clone(),
                group_id: None,
                group_count: None,
            }
        })
        .collect();

    // Apply semantic grouping
    group_activity_items(&mut items);

    // Collect available topics
    let topics: Vec<String> = hierarchy.map(|h|
        h.super_clusters.iter().map(|sc| sc.label.clone()).collect()
    ).unwrap_or_default();

    // Apply topic filter
    let items = if let Some(ref topic) = params.topic {
        items.into_iter().filter(|i| i.topic_tags.contains(topic)).collect()
    } else {
        items
    };

    let total = items.len();
    Json(ActivityFeedResponse { items, total, topics })
}
```

-**Step 3: Wire endpoint**

In `mod.rs`, add route: `.route("/api/dashboard/activity", get(activity::activity_feed_handler))`

-**Step 4: Add semantic grouping logic**

After building the activity items list, group adjacent items that share the same cluster label and agent within a 5-minute window:

```rust
fn group_activity_items(items: &mut Vec<ActivityItem>) {
    // Walk through items, merge adjacent ones with same agent + topic within 5min
    let mut i = 0;
    while i < items.len() {
        let mut group_size = 1;
        let mut j = i + 1;
        while j < items.len() {
            let same_agent = items[i].agent_id == items[j].agent_id;
            let same_topic = !items[i].topic_tags.is_empty()
                && items[i].topic_tags[0] == items[j].topic_tags.get(0).cloned().unwrap_or_default();
            let time_i: DateTime<Utc> = items[i].recorded_at.parse().unwrap_or_default();
            let time_j: DateTime<Utc> = items[j].recorded_at.parse().unwrap_or_default();
            let within_5min = (time_i - time_j).num_seconds().abs() < 300;

            if same_agent && (same_topic || within_5min) {
                group_size += 1;
                j += 1;
            } else {
                break;
            }
        }

        if group_size > 1 {
            let group_id = format!("group-{i}");
            for k in i..j {
                items[k].group_id = Some(group_id.clone());
                items[k].group_count = Some(group_size);
            }
        }
        i = j;
    }
}
```

-**Step 5: Write unit test for grouping logic**

In `activity.rs`, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_activity_items_same_agent_same_topic() {
        let now = chrono::Utc::now();
        let mut items = vec![
            ActivityItem {
                entry_id: "a".into(), action: "wrote".into(), title: "test".into(),
                agent_id: Some("agent1".into()), agent_name: None,
                topic_tags: vec!["graph".into()], delta_bytes: Some(100),
                recorded_at: now.to_rfc3339(), superseded_id: None,
                group_id: None, group_count: None,
            },
            ActivityItem {
                entry_id: "b".into(), action: "wrote".into(), title: "test2".into(),
                agent_id: Some("agent1".into()), agent_name: None,
                topic_tags: vec!["graph".into()], delta_bytes: Some(50),
                recorded_at: (now - chrono::Duration::seconds(60)).to_rfc3339(),
                superseded_id: None, group_id: None, group_count: None,
            },
        ];
        group_activity_items(&mut items);
        assert!(items[0].group_id.is_some());
        assert_eq!(items[0].group_count, Some(2));
        assert_eq!(items[0].group_id, items[1].group_id);
    }

    #[test]
    fn test_no_group_different_agents() {
        let now = chrono::Utc::now();
        let mut items = vec![
            ActivityItem {
                entry_id: "a".into(), action: "wrote".into(), title: "test".into(),
                agent_id: Some("agent1".into()), agent_name: None,
                topic_tags: vec!["graph".into()], delta_bytes: Some(100),
                recorded_at: now.to_rfc3339(), superseded_id: None,
                group_id: None, group_count: None,
            },
            ActivityItem {
                entry_id: "b".into(), action: "wrote".into(), title: "test2".into(),
                agent_id: Some("agent2".into()), agent_name: None,
                topic_tags: vec!["graph".into()], delta_bytes: Some(50),
                recorded_at: (now - chrono::Duration::seconds(60)).to_rfc3339(),
                superseded_id: None, group_id: None, group_count: None,
            },
        ];
        group_activity_items(&mut items);
        assert!(items[0].group_id.is_none());
    }
}
```

-**Step 6: Run tests**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test -p corvia-server activity::tests --no-default-features 2>&1 | tail -10`
Expected: PASS

-**Step 7: Test endpoint manually**

Run: `curl -s http://localhost:8020/api/dashboard/activity?limit=10 | jq '.items | length'`
Expected: number ≤ 10

Run: `curl -s http://localhost:8020/api/dashboard/activity | jq '.topics'`
Expected: array of topic labels

-**Step 8: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-server/src/dashboard/activity.rs crates/corvia-server/src/dashboard/mod.rs
git commit -m "feat(dashboard): activity feed endpoint with semantic grouping and topic filters"
```

### Task 14: History Tab Frontend — Activity Feed

**Files:**
- Modify: `tools/corvia-dashboard/src/components/HistoryView.tsx`
- Modify: `tools/corvia-dashboard/src/api.ts`
- Modify: `tools/corvia-dashboard/src/types.ts`

-**Step 1: Add types**

In `types.ts`:

```typescript
export interface ActivityItem {
  entry_id: string;
  action: string;
  title: string;
  agent_id?: string;
  agent_name?: string;
  topic_tags: string[];
  delta_bytes?: number;
  recorded_at: string;
  superseded_id?: string;
  group_id?: string;
  group_count?: number;
}

export interface ActivityFeedResponse {
  items: ActivityItem[];
  total: number;
  topics: string[];
}
```

-**Step 2: Add API function**

In `api.ts`:

```typescript
export async function fetchActivityFeed(params?: {
  limit?: number; offset?: number; agent?: string; topic?: string;
}): Promise<ActivityFeedResponse> {
  const query = new URLSearchParams();
  if (params?.limit) query.set('limit', String(params.limit));
  if (params?.offset) query.set('offset', String(params.offset));
  if (params?.agent) query.set('agent', params.agent);
  if (params?.topic) query.set('topic', params.topic);
  const res = await fetch(`${BASE}/activity?${query}`);
  if (!res.ok) throw new Error(`Activity fetch failed: ${res.status}`);
  return res.json();
}
```

-**Step 3: Rewrite HistoryView as activity feed**

Replace the UUID-lookup-first design with an activity feed:

```tsx
export function HistoryView({ openSidebar }: { openSidebar: (content: SidebarContent, width: SidebarState) => void }) {
  const [feed, setFeed] = useState<ActivityFeedResponse | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  useEffect(() => {
    const load = () => fetchActivityFeed({
      limit: 50,
      topic: selectedTopic || undefined,
      agent: selectedAgent || undefined,
    }).then(setFeed).catch(console.error);
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  }, [selectedTopic, selectedAgent]);

  if (!feed) return <div class="loading">Loading activity...</div>;

  // Group items by group_id
  const grouped = groupFeedItems(feed.items);

  return (
    <div class="history-view">
      {/* Topic filter bar */}
      <div class="topic-filters">
        {feed.topics.map(topic => (
          <button
            class={`topic-pill ${selectedTopic === topic ? 'active' : ''}`}
            onClick={() => setSelectedTopic(selectedTopic === topic ? null : topic)}
          >
            {topic}
          </button>
        ))}
      </div>

      {/* Activity feed */}
      <div class="activity-feed">
        {grouped.map(item => (
          <div
            class="feed-item"
            key={item.entry_id}
            onClick={() => openSidebar({ kind: 'history', entryId: item.entry_id }, 'wide')}
          >
            <span class="agent-dot" style={{ background: agentColor(item.agent_id) }} />
            <span class="agent-name">{item.agent_name || item.agent_id || 'unknown'}</span>
            <span class="action">{item.action}</span>
            <span class="title">{item.title}</span>
            <span class="topics">
              {item.topic_tags.map(t => <span class="topic-pill small">{t}</span>)}
            </span>
            <span class={`delta ${(item.delta_bytes || 0) >= 0 ? 'positive' : 'negative'}`}>
              {(item.delta_bytes || 0) >= 0 ? '+' : ''}{item.delta_bytes}
            </span>
            <span class="timestamp">{relativeTime(item.recorded_at)}</span>
            {item.group_count && item.group_count > 1 && !expandedGroups.has(item.group_id!) && (
              <button class="expand-group" onClick={(e) => {
                e.stopPropagation();
                setExpandedGroups(prev => new Set([...prev, item.group_id!]));
              }}>
                +{item.group_count - 1} more
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

-**Step 4: Add helper functions**

```typescript
function agentColor(agentId?: string): string {
  if (!agentId) return '#666';
  // Deterministic color from agent ID hash
  let hash = 0;
  for (const ch of agentId) hash = ((hash << 5) - hash) + ch.charCodeAt(0);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 60%, 50%)`;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function groupFeedItems(items: ActivityItem[]): ActivityItem[] {
  // Show only first item of each group unless expanded
  const seen = new Set<string>();
  return items.filter(item => {
    if (!item.group_id) return true;
    if (seen.has(item.group_id)) return false;
    seen.add(item.group_id);
    return true;
  });
}
```

-**Step 5: Add CSS for activity feed**

Style the feed items with the dashboard's existing color system (lavender accents, dark background).

-**Step 6: Test in browser**

Open `http://localhost:8021`, go to History tab. Verify:
1. Activity feed loads with recent entries
2. Topic filter pills appear and filter works
3. Grouped items show "+N more" buttons
4. Clicking an item opens sidebar with entry detail
5. Timestamps show relative time

-**Step 7: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/HistoryView.tsx tools/corvia-dashboard/src/api.ts tools/corvia-dashboard/src/types.ts
git commit -m "feat(dashboard): activity feed with semantic grouping and topic filters in History tab"
```

### Task 15: Cross-Tab Deeplinks

**Files:**
- Modify: `tools/corvia-dashboard/src/components/Layout.tsx`
- Modify: `tools/corvia-dashboard/src/components/GraphView.tsx`
- Modify: `tools/corvia-dashboard/src/components/AgentsView.tsx`
- Modify: `tools/corvia-dashboard/src/components/HealthPanel.tsx`
- Modify: `tools/corvia-dashboard/src/components/RagView.tsx`
- Modify: `tools/corvia-dashboard/src/components/TracesView.tsx`
- Modify: `tools/corvia-dashboard/src/components/LogsView.tsx`

-**Step 1: Add navigation function to Layout**

```typescript
function navigateToHistory(entryId: string) {
  setActiveTab('history');
  // Pass entryId to HistoryView via state or URL param
  setDeeplinkEntryId(entryId);
}
```

Thread `navigateToHistory` as a prop to all tab components.

-**Step 2: Make entry IDs clickable in each tab**

For each component, find where entry IDs are displayed and wrap them in clickable elements:

- **GraphView:** Entry node click → sidebar already opens. Add "View history →" link in the sidebar panel.
- **AgentsView:** Session entry list → wrap IDs with `onClick={() => navigateToHistory(id)}`
- **HealthPanel:** Finding target IDs → wrap truncated UUIDs with `onClick`
- **RagView:** Source entries → wrap source content cards with "View history →" link
- **TracesView:** Entry IDs in span events → wrap with `onClick`
- **LogsView:** Entry IDs in log messages → wrap with `onClick` (regex match UUID pattern in log text)

-**Step 3: Handle deeplink in HistoryView**

When HistoryView receives a `deeplinkEntryId` prop, auto-open the sidebar with that entry's timeline:

```tsx
useEffect(() => {
  if (deeplinkEntryId) {
    openSidebar({ kind: 'history', entryId: deeplinkEntryId }, 'wide');
  }
}, [deeplinkEntryId]);
```

-**Step 4: Test deeplinks**

1. Graph tab → click entry → sidebar → "View history" → navigates to History tab with entry loaded
2. Health tab → click finding target ID → History tab with entry loaded
3. RAG tab → click source → History tab with entry loaded

-**Step 5: Commit**

```bash
cd /workspaces/corvia-workspace
git add tools/corvia-dashboard/src/components/Layout.tsx tools/corvia-dashboard/src/components/GraphView.tsx tools/corvia-dashboard/src/components/AgentsView.tsx tools/corvia-dashboard/src/components/HealthPanel.tsx tools/corvia-dashboard/src/components/RagView.tsx tools/corvia-dashboard/src/components/TracesView.tsx tools/corvia-dashboard/src/components/LogsView.tsx
git commit -m "feat(dashboard): cross-tab deeplinks from all tabs to History"
```

---

## Chunk 5: Integration and GC Adjustment

### Task 16: GC — Preserve Agent Records

**Files:**
- Modify: `crates/corvia-kernel/src/agent_coordinator.rs`

-**Step 1: Write test confirming agents survive GC**

```rust
#[test]
fn test_gc_preserves_agent_records() {
    let coord = test_coordinator();
    let identity = AgentIdentity::Registered { agent_id: "test::agent".into(), api_key: None };
    coord.register_agent(&identity, "Test", AgentPermission::ReadOnly).unwrap();
    let session = coord.create_session("test::agent", false).unwrap();

    // Make session orphaned
    coord.sessions.transition(&session.session_id, SessionState::Stale).unwrap();
    coord.sessions.transition(&session.session_id, SessionState::Orphaned).unwrap();

    // Run GC
    let report = coord.gc_sweep().unwrap();
    assert_eq!(report.orphans_rolled_back, 1);

    // Agent record still exists
    let agent = coord.registry.get("test::agent").unwrap();
    assert!(agent.is_some(), "Agent record should survive GC");
    assert_eq!(agent.unwrap().status, AgentStatus::Active);
}
```

-**Step 2: Verify existing GC code doesn't delete agents**

Read the `gc_sweep` method. If it deletes agent records alongside sessions, modify it to only roll back sessions (transition to Closed, clean staging) while preserving the agent record.

-**Step 3: Run test, verify pass**

-**Step 4: Commit**

```bash
cd /workspaces/corvia-workspace/repos/corvia
git add crates/corvia-kernel/src/agent_coordinator.rs
git commit -m "fix(gc): preserve agent records during GC sweep, only rollback sessions"
```

### Task 17: End-to-End Smoke Test

-**Step 1: Start server and verify all new endpoints**

```bash
# Ensure server is running
curl -s http://localhost:8020/api/dashboard/status | jq '.entry_count'

# Test clustered graph
curl -s http://localhost:8020/api/dashboard/graph/scope?level=0 | jq '.nodes | length'

# Test activity feed
curl -s http://localhost:8020/api/dashboard/activity?limit=5 | jq '.items | length'

# Test reconnectable agents
curl -s http://localhost:8020/api/dashboard/agents/reconnectable | jq 'length'
```

-**Step 2: Verify dashboard loads all tabs**

Open `http://localhost:8021` and verify:
1. Graph tab renders super-clusters (not the old fractured view)
2. History tab shows activity feed (not UUID lookup)
3. Sidebar is collapsed by default
4. Agents tab shows descriptions and topic pills (if data present)

-**Step 3: Run full Rust test suite**

Run: `cd /workspaces/corvia-workspace/repos/corvia && cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass (existing + new)

-**Step 4: Build frontend**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dashboard && npm run build`
Expected: Build succeeds with no errors

-**Step 5: Final commit if any fixups**

---

## Parallelization Guide

These task groups can be dispatched to independent subagents:

| Subagent | Tasks | Branch | Git Repo | Worktree Port | Dependencies |
|----------|-------|--------|----------|---------------|-------------|
| **A: ClusterStore** | 0, 1, 2, 3 | `feature/cluster-store` | repos/corvia | 8023 | None — build first |
| **B: Agent Identity** | 5, 6, 7, 8, 9, 10 | `feature/agent-identity` | repos/corvia | 8024 | Needs ClusterStore for topic tags (Task 7) |
| **C: Collapsible Panel** | 12 | `feature/collapsible-panel` | workspace root | 8025 | None — pure frontend |
| **D: Graph LOD Frontend** | 4 | `feature/graph-lod` | workspace root | 8026 | Needs ClusterStore endpoint (Task 3) |
| **E: History Feed** | 13, 14, 15 | `feature/history-feed` | both repos | 8027 | Needs ClusterStore (Task 2) + Collapsible Panel (Task 12) |
| **F: Agent Display** | 11 | `feature/agent-display` | workspace root | N/A | Needs Agent Identity types (Task 5) |
| **G: Integration** | 16, 17 | N/A | both repos | N/A | All above complete |

**Recommended execution order:**
1. **Phase 1:** A (ClusterStore) + C (Collapsible Panel) — fully independent
2. **Phase 2:** B (Agent Identity) + D (Graph LOD Frontend) — both need ClusterStore
3. **Phase 3:** E (History Feed) + F (Agent Display) — need Phase 2 outputs
4. **Phase 4:** G (Integration) — smoke test everything

**Merge order:** A → C → B → D → F → E → G (each rebases on previous)
