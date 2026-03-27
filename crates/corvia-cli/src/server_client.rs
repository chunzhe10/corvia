//! HTTP client for routing CLI operations through a running corvia-server.
//!
//! When a corvia server is already running, the CLI routes read-only operations
//! through the REST API instead of opening the Redb database directly. This
//! avoids the exclusive file lock conflict between the server and CLI.

#![allow(dead_code)]

use anyhow::Result;
use corvia_common::config::CorviaConfig;
use serde::{Deserialize, Serialize};

/// HTTP client that wraps a running corvia-server's REST API.
pub struct ServerClient {
    client: reqwest::Client,
    base_url: String,
}

// --- Response DTOs (lightweight, decoupled from server crate internals) ---

#[derive(Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultDto>,
    pub count: usize,
}

#[derive(Deserialize)]
pub struct SearchResultDto {
    pub content: String,
    pub score: f32,
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    #[serde(default)]
    pub tier: Option<String>,
    #[serde(default)]
    pub retention_score: Option<f32>,
}

#[derive(Deserialize)]
pub struct HistoryEntryDto {
    pub id: String,
    pub content: String,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    pub valid_from: chrono::DateTime<chrono::Utc>,
    pub valid_to: Option<chrono::DateTime<chrono::Utc>>,
    pub is_current: bool,
}

#[derive(Deserialize)]
pub struct EdgeDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct FindingDto {
    pub check_type: String,
    pub confidence: f32,
    pub rationale: String,
    pub target_ids: Vec<String>,
}

#[derive(Deserialize)]
pub struct ReasonResponse {
    pub scope_id: String,
    pub findings: Vec<FindingDto>,
    pub count: usize,
}

#[derive(Deserialize)]
pub struct EvolutionEntryDto {
    pub id: String,
    pub content: String,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    pub valid_from: chrono::DateTime<chrono::Utc>,
    pub valid_to: Option<chrono::DateTime<chrono::Utc>>,
    pub is_current: bool,
}

// --- Request bodies ---

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    scope_id: String,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct ReasonRequest {
    scope_id: String,
    check: Option<String>,
}

impl ServerClient {
    /// Detect a running corvia-server by health-checking the configured endpoint.
    /// Returns `Some(client)` if the server responds, `None` otherwise.
    pub async fn detect(config: &CorviaConfig) -> Option<Self> {
        let host = &config.server.host;
        let port = config.server.port;
        let base_url = format!("http://{}:{}", host, port);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .ok()?;

        let resp = client
            .get(format!("{}/health", base_url))
            .send()
            .await
            .ok()?;

        if resp.status().is_success() {
            // Use a longer timeout for actual operations
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .ok()?;
            Some(Self { client, base_url })
        } else {
            None
        }
    }

    /// Base URL of the connected server (for display).
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// Semantic search.
    pub async fn search(
        &self,
        query: &str,
        scope_id: &str,
        limit: usize,
    ) -> Result<SearchResponse> {
        let resp = self
            .client
            .post(format!("{}/v1/memories/search", self.base_url))
            .json(&SearchRequest {
                query: query.to_string(),
                scope_id: scope_id.to_string(),
                limit: Some(limit),
            })
            .send()
            .await?
            .error_for_status()?;

        Ok(resp.json().await?)
    }

    /// Run reasoning checks.
    pub async fn reason(
        &self,
        scope_id: &str,
        check: Option<&str>,
    ) -> Result<ReasonResponse> {
        let resp = self
            .client
            .post(format!("{}/v1/reason", self.base_url))
            .json(&ReasonRequest {
                scope_id: scope_id.to_string(),
                check: check.map(String::from),
            })
            .send()
            .await?
            .error_for_status()?;

        Ok(resp.json().await?)
    }

    /// Get history chain for an entry.
    pub async fn history(&self, entry_id: &str) -> Result<Vec<HistoryEntryDto>> {
        let resp = self
            .client
            .get(format!("{}/v1/entries/{}/history", self.base_url, entry_id))
            .send()
            .await?
            .error_for_status()?;

        Ok(resp.json().await?)
    }

    /// Get edges for an entry.
    pub async fn edges(
        &self,
        entry_id: &str,
        relation: Option<&str>,
    ) -> Result<Vec<EdgeDto>> {
        let mut url = format!("{}/v1/entries/{}/edges", self.base_url, entry_id);
        if let Some(rel) = relation {
            url = format!("{}?relation={}", url, rel);
        }

        let resp = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?;

        Ok(resp.json().await?)
    }

    /// Get evolution (entries changed within a time range).
    pub async fn evolution(
        &self,
        scope: &str,
        since: &str,
    ) -> Result<Vec<EvolutionEntryDto>> {
        let url = format!(
            "{}/v1/evolution?scope={}&since={}",
            self.base_url, scope, since
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?;

        Ok(resp.json().await?)
    }

    /// Pin an entry by ID.
    pub async fn pin_entry(&self, entry_id: &str, agent_id: &str) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/v1/entries/{}/pin", self.base_url, entry_id))
            .json(&serde_json::json!({ "agent_id": agent_id }))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Unpin an entry by ID.
    pub async fn unpin_entry(&self, entry_id: &str) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/v1/entries/{}/unpin", self.base_url, entry_id))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Inspect an entry's lifecycle metadata.
    pub async fn inspect_entry(&self, entry_id: &str) -> Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}/v1/entries/{}/inspect", self.base_url, entry_id))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Get GC status (tier distribution).
    pub async fn gc_status(&self, scope_id: Option<&str>) -> Result<serde_json::Value> {
        let url = format!("{}/v1/gc/status", self.base_url);
        let mut req = self.client.get(&url);
        if let Some(scope) = scope_id {
            req = req.query(&[("scope_id", scope)]);
        }
        let resp = req.send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Trigger a manual GC knowledge cycle.
    pub async fn gc_run(&self) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/v1/gc/run", self.base_url))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Get recent GC knowledge cycle history.
    pub async fn gc_history(&self) -> Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}/v1/gc/history", self.base_url))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }
}
