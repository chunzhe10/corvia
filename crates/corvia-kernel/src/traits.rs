use async_trait::async_trait;
use corvia_common::errors::Result;
use corvia_common::types::{EdgeDirection, GraphEdge, KnowledgeEntry, SearchResult};
use std::any::Any;

/// Queryable store for knowledge entries (D15).
/// LiteStore is the default implementation. Trait-bounded for backend swap.
#[async_trait]
pub trait QueryableStore: Send + Sync {
    /// Insert a knowledge entry (must have embedding set).
    async fn insert(&self, entry: &KnowledgeEntry) -> Result<()>;

    /// Semantic search by embedding vector.
    async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<Vec<SearchResult>>;

    /// Get a single entry by ID.
    async fn get(&self, id: &uuid::Uuid) -> Result<Option<KnowledgeEntry>>;

    /// Count entries in a scope.
    async fn count(&self, scope_id: &str) -> Result<u64>;

    /// Initialize the store schema (create tables, indexes).
    async fn init_schema(&self) -> Result<()>;

    /// Delete all entries in a scope. Used for test/demo teardown.
    async fn delete_scope(&self, scope_id: &str) -> Result<()>;

    /// Look up an entry by scope and source_version (exact match).
    /// Used for cross-batch graph edge resolution (e.g., spawned_by parent lookup).
    async fn get_by_source_version(
        &self,
        scope_id: &str,
        source_version: &str,
    ) -> Result<Option<KnowledgeEntry>>;

    /// Record access for a batch of entries (updates last_accessed and access_count).
    /// All updates are batched into a single write transaction.
    /// Failures are non-fatal — implementations should log warnings, not propagate errors.
    async fn record_access(&self, entry_ids: &[uuid::Uuid]) -> Result<()>;

    /// Downcast support for store-specific operations (e.g., LiteStore::rebuild_from_files).
    fn as_any(&self) -> &dyn Any;
}

/// Embedding and inference provider (D5).
/// Ollama is the first implementation. Provider-agnostic via trait.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Return the embedding dimension for this model.
    fn dimensions(&self) -> usize;
}

/// Ingestion adapter for domain-specific sources (D11/D69).
///
/// D69 clean break: adapters return raw [`SourceFile`]s and register
/// [`ChunkingStrategy`]s. The kernel's [`ChunkingPipeline`] handles
/// splitting, merging, and token budget enforcement.
#[async_trait]
pub trait IngestionAdapter: Send + Sync {
    /// Domain name (e.g., "git", "general").
    fn domain(&self) -> &str;

    /// Register chunking strategies this adapter provides.
    /// Called once during pipeline setup. The adapter registers
    /// its strategies (e.g., AstChunker) as overrides in the registry.
    fn register_chunking(&self, registry: &mut crate::chunking_pipeline::FormatRegistry);

    /// Return raw source files for the kernel's ChunkingPipeline.
    /// Files are returned as-is (content + metadata); the pipeline
    /// handles all chunking, embedding, and storage.
    async fn ingest_sources(&self, source_path: &str) -> Result<Vec<crate::chunking_strategy::SourceFile>>;
}

/// Temporal query interface for bi-temporal knowledge (D38).
/// Implemented by LiteStore (Redb range scans) and PostgresStore (SQL).
#[async_trait]
pub trait TemporalStore: Send + Sync {
    /// Return entries valid at a point in time within a scope.
    async fn as_of(
        &self,
        scope_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Follow the supersession chain for an entry (newest -> oldest).
    async fn history(&self, entry_id: &uuid::Uuid) -> Result<Vec<KnowledgeEntry>>;

    /// Return entries that changed within a time range (created, superseded, or expired).
    async fn evolution(
        &self,
        scope_id: &str,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<KnowledgeEntry>>;
}

/// Knowledge graph interface (D37).
/// Implemented by LiteStore (petgraph + Redb) and PostgresStore (relational edges).
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Create a directed edge between two entries.
    async fn relate(
        &self,
        from: &uuid::Uuid,
        relation: &str,
        to: &uuid::Uuid,
        metadata: Option<serde_json::Value>,
    ) -> Result<()>;

    /// Get all edges from/to an entry.
    async fn edges(
        &self,
        entry_id: &uuid::Uuid,
        direction: EdgeDirection,
    ) -> Result<Vec<GraphEdge>>;

    /// BFS traversal from a starting node, optionally filtering by relation type.
    async fn traverse(
        &self,
        start: &uuid::Uuid,
        relation: Option<&str>,
        direction: EdgeDirection,
        max_depth: usize,
    ) -> Result<Vec<KnowledgeEntry>>;

    /// Shortest path between two entries (returns the entries along the path).
    async fn shortest_path(
        &self,
        from: &uuid::Uuid,
        to: &uuid::Uuid,
    ) -> Result<Option<Vec<KnowledgeEntry>>>;

    /// Delete all edges involving an entry.
    async fn remove_edges(&self, entry_id: &uuid::Uuid) -> Result<()>;
}

// Re-export ChatMessage so kernel consumers don't need a direct corvia-common dependency.
pub use corvia_common::types::ChatMessage;

/// Result from a GenerationEngine call.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Text generation engine for RAG answers and LLM-assisted merge (D63).
///
/// Wire protocol name (`ChatService` in proto) is separate from this
/// Rust capability trait. Implementations: GrpcChatEngine (corvia-inference),
/// OllamaChatEngine (HTTP), GrpcVllmChatEngine (vLLM).
#[async_trait]
pub trait GenerationEngine: Send + Sync {
    /// Human-readable name for metrics attribution (D62).
    fn name(&self) -> &str;

    /// Generate a text response from a system prompt and user message.
    async fn generate(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<GenerationResult>;

    /// Return the model's context window size in tokens.
    fn context_window(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time test: verify traits are object-safe
    fn _assert_generation_engine_object_safe(_: &dyn GenerationEngine) {}
    fn _assert_ingestion_adapter_object_safe(_: &dyn IngestionAdapter) {}

    #[test]
    fn test_chat_message_construction() {
        let msg = ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        };
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_generation_result_construction() {
        let result = GenerationResult {
            text: "answer".into(),
            model: "test-model".into(),
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(result.text, "answer");
        assert_eq!(result.model, "test-model");
        assert_eq!(result.input_tokens, 100);
        assert_eq!(result.output_tokens, 50);
    }
}
