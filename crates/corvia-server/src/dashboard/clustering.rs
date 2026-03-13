//! Semantic clustering for the dashboard knowledge graph.
//!
//! Provides k-means clustering with k-means++ initialization, silhouette scoring,
//! and a two-level cluster hierarchy (super-clusters L0, sub-clusters L1) for
//! level-of-detail graph rendering.

use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Distance / similarity primitives
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two vectors.
fn euclidean_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity between two vectors. Returns 0 if either has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

// ---------------------------------------------------------------------------
// K-means with k-means++ initialization
// ---------------------------------------------------------------------------

/// Index of the nearest centroid to `point`.
fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean_dist_sq(point, a)
                .partial_cmp(&euclidean_dist_sq(point, b))
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
}

/// K-means++ centroid initialization.
fn kmeans_pp_init(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids = vec![data[rng.gen_range(0..data.len())].clone()];

    for _ in 1..k {
        let distances: Vec<f32> = data
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| euclidean_dist_sq(p, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        let total: f32 = distances.iter().sum();
        if total == 0.0 {
            // All points coincide with existing centroids — pick randomly.
            centroids.push(data[rng.gen_range(0..data.len())].clone());
            continue;
        }
        let threshold = Rng::r#gen::<f32>(&mut rng) * total;
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

/// Assign each vector to one of `k` clusters via Lloyd's algorithm.
/// Returns cluster assignments (0..k) for each data point.
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

// ---------------------------------------------------------------------------
// Silhouette scoring
// ---------------------------------------------------------------------------

/// Compute the mean silhouette coefficient for a clustering.
/// Samples up to 500 points for efficiency on large datasets.
pub fn silhouette_score(data: &[Vec<f32>], assignments: &[usize], k: usize) -> f32 {
    let n = data.len();
    if n <= 1 || k <= 1 {
        return 0.0;
    }

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
    let mut counted = 0usize;
    for &i in &sample {
        let ci = assignments[i];
        // a(i) = mean distance to same cluster
        let mut same_sum = 0.0f32;
        let mut same_count = 0usize;
        // b(i) = min mean distance to other clusters
        let mut other_sums = vec![0.0f32; k];
        let mut other_counts = vec![0usize; k];

        for (j, point) in data.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = euclidean_dist_sq(&data[i], point).sqrt();
            if assignments[j] == ci {
                same_sum += d;
                same_count += 1;
            } else {
                other_sums[assignments[j]] += d;
                other_counts[assignments[j]] += 1;
            }
        }

        let a = if same_count > 0 {
            same_sum / same_count as f32
        } else {
            0.0
        };
        let b = (0..k)
            .filter(|&c| c != ci && other_counts[c] > 0)
            .map(|c| other_sums[c] / other_counts[c] as f32)
            .fold(f32::MAX, f32::min);

        if b == f32::MAX {
            continue;
        }
        let s = (b - a) / a.max(b);
        total += s;
        counted += 1;
    }
    if counted == 0 {
        return 0.0;
    }
    total / counted as f32
}

/// Find the best K by trying k_min..=k_max and picking highest silhouette score.
pub fn find_best_k(data: &[Vec<f32>], k_min: usize, k_max: usize, _max_sample: usize) -> usize {
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

// ---------------------------------------------------------------------------
// Centroid computation
// ---------------------------------------------------------------------------

/// Compute the centroid (mean) of a set of vectors.
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

// ---------------------------------------------------------------------------
// ClusterNode & ClusterHierarchy
// ---------------------------------------------------------------------------

/// A single cluster node in the hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub cluster_id: String,
    pub label: String,
    pub level: u8,
    pub parent_id: Option<String>,
    pub entry_ids: Vec<String>,
    pub centroid: Vec<f32>,
}

/// Two-level cluster hierarchy: super-clusters (L0) and sub-clusters (L1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHierarchy {
    pub super_clusters: Vec<ClusterNode>, // L0
    pub sub_clusters: Vec<ClusterNode>,   // L1
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
            let member_indices: Vec<usize> = l0_assignments
                .iter()
                .enumerate()
                .filter(|&(_, a)| *a == c)
                .map(|(i, _)| i)
                .collect();
            if member_indices.is_empty() {
                continue;
            }

            let entry_ids: Vec<String> =
                member_indices.iter().map(|&i| entries[i].0.clone()).collect();
            let centroid = compute_centroid(
                &member_indices
                    .iter()
                    .map(|&i| &embeddings[i])
                    .collect::<Vec<_>>(),
            );

            // Label from nearest entry to centroid
            let nearest_idx = member_indices
                .iter()
                .min_by(|&&a, &&b| {
                    euclidean_dist_sq(&embeddings[a], &centroid)
                        .partial_cmp(&euclidean_dist_sq(&embeddings[b], &centroid))
                        .unwrap()
                })
                .copied()
                .unwrap();
            let label = entries[nearest_idx].0.clone();

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
            if sc.entry_ids.len() < 4 {
                continue; // Too small to sub-divide
            }
            let sc_embeddings: Vec<Vec<f32>> = sc
                .entry_ids
                .iter()
                .map(|id| {
                    entries
                        .iter()
                        .find(|(eid, _)| eid == id)
                        .unwrap()
                        .1
                        .clone()
                })
                .collect();
            let sc_entries: Vec<(String, Vec<f32>)> = sc
                .entry_ids
                .iter()
                .zip(sc_embeddings.iter())
                .map(|(id, e)| (id.clone(), e.clone()))
                .collect();

            let sub_k_max = (sc.entry_ids.len() / 2).min(8).max(2);
            let sub_k = find_best_k(&sc_embeddings, 2, sub_k_max, 200);
            let sub_assignments = kmeans(&sc_embeddings, sub_k, 100);

            for s in 0..sub_k {
                let member_indices: Vec<usize> = sub_assignments
                    .iter()
                    .enumerate()
                    .filter(|&(_, a)| *a == s)
                    .map(|(i, _)| i)
                    .collect();
                if member_indices.is_empty() {
                    continue;
                }

                let entry_ids: Vec<String> =
                    member_indices.iter().map(|&i| sc_entries[i].0.clone()).collect();
                let centroid = compute_centroid(
                    &member_indices
                        .iter()
                        .map(|&i| &sc_embeddings[i])
                        .collect::<Vec<_>>(),
                );
                let nearest_idx = member_indices
                    .iter()
                    .min_by(|&&a, &&b| {
                        euclidean_dist_sq(&sc_embeddings[a], &centroid)
                            .partial_cmp(&euclidean_dist_sq(&sc_embeddings[b], &centroid))
                            .unwrap()
                    })
                    .copied()
                    .unwrap();

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
        self.super_clusters
            .iter()
            .find(|sc| sc.entry_ids.contains(&entry_id.to_string()))
    }

    /// Get topic label for an entry (its super-cluster label).
    pub fn topic_for_entry(&self, entry_id: &str) -> Option<&str> {
        self.cluster_for_entry(entry_id)
            .map(|sc| sc.label.as_str())
    }
}

// ---------------------------------------------------------------------------
// ClusterStore — thread-safe wrapper with in-memory caching
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around an optional `ClusterHierarchy`.
/// Stored in `AppState` as `Arc<ClusterStore>` and updated by a background task.
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

    /// Return a snapshot of the current hierarchy, if computed.
    pub fn current(&self) -> Option<ClusterHierarchy> {
        self.hierarchy.read().unwrap().clone()
    }

    /// Returns true if no hierarchy has been computed yet.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    #[test]
    fn test_silhouette_well_separated() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &assignments, 2);
        assert!(
            score > 0.9,
            "Well-separated clusters should have silhouette > 0.9, got {score}"
        );
    }

    #[test]
    fn test_best_k_finds_obvious_clusters() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.05, 10.05],
            vec![20.0, 0.0],
            vec![20.1, 0.0],
            vec![20.0, 0.1],
            vec![20.05, 0.05],
        ];
        let best = find_best_k(&data, 2, 6, 500);
        assert_eq!(best, 3, "Should find 3 clusters, got {best}");
    }

    #[test]
    fn test_kmeans_single_point_per_cluster() {
        let data: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let assignments = kmeans(&data, 5, 100);
        // n <= k, should return identity assignments
        assert_eq!(assignments, vec![0, 1]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

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
        assert!(
            hierarchy.super_clusters.len() >= 2 && hierarchy.super_clusters.len() <= 5,
            "Expected 2-5 super-clusters, got {}",
            hierarchy.super_clusters.len()
        );
        // Each super-cluster should have 3 entries
        for sc in &hierarchy.super_clusters {
            assert_eq!(
                sc.entry_ids.len(),
                3,
                "Super-cluster {} has {} entries, expected 3",
                sc.cluster_id,
                sc.entry_ids.len()
            );
        }
    }

    #[test]
    fn test_cluster_store_degraded_when_empty() {
        let store = ClusterStore::new();
        assert!(
            store.current().is_none(),
            "Should be None before first computation"
        );
        assert!(store.is_degraded());
    }

    #[test]
    fn test_cluster_store_recompute() {
        let store = ClusterStore::new();
        let entries: Vec<(String, Vec<f32>)> = vec![
            ("a".into(), vec![0.0, 0.0]),
            ("b".into(), vec![0.1, 0.0]),
            ("c".into(), vec![10.0, 10.0]),
            ("d".into(), vec![10.1, 10.0]),
        ];
        // First compute should succeed
        assert!(store.maybe_recompute(&entries));
        assert!(!store.is_degraded());
        assert!(store.current().is_some());

        // Same count should not recompute
        assert!(!store.maybe_recompute(&entries));
    }

    #[test]
    fn test_cluster_store_too_few_entries() {
        let store = ClusterStore::new();
        let entries: Vec<(String, Vec<f32>)> = vec![
            ("a".into(), vec![0.0, 0.0]),
            ("b".into(), vec![1.0, 1.0]),
        ];
        // Less than 3 entries should not recompute
        assert!(!store.maybe_recompute(&entries));
        assert!(store.is_degraded());
    }

    #[test]
    fn test_cluster_for_entry() {
        // Use 8 entries with 2 tight clusters — forces k=2 via silhouette
        let entries: Vec<(String, Vec<f32>)> = vec![
            ("a1".into(), vec![0.0, 0.0]),
            ("a2".into(), vec![0.01, 0.0]),
            ("a3".into(), vec![0.0, 0.01]),
            ("a4".into(), vec![0.01, 0.01]),
            ("b1".into(), vec![100.0, 100.0]),
            ("b2".into(), vec![100.01, 100.0]),
            ("b3".into(), vec![100.0, 100.01]),
            ("b4".into(), vec![100.01, 100.01]),
        ];
        let hierarchy = ClusterHierarchy::build(&entries, 2, 4);
        // a1 and a2 should be in the same cluster
        let c1 = hierarchy.cluster_for_entry("a1").unwrap();
        let c2 = hierarchy.cluster_for_entry("a2").unwrap();
        assert_eq!(c1.cluster_id, c2.cluster_id);
        // a1 and b1 should be in different clusters
        let c3 = hierarchy.cluster_for_entry("b1").unwrap();
        assert_ne!(c1.cluster_id, c3.cluster_id);
    }
}
