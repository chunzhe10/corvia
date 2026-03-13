# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7] - 2026-03-13

### Added

- **Dashboard UX Overhaul**: Complete overhaul of the standalone dashboard (18 tasks)
- **K-means clustering**: `ClusterStore` with k-means++ initialization on 768-dim embeddings,
  silhouette scoring for optimal K (3..12), shared across graph/activity/agent views
- **Multi-tiered LOD graph**: 4-level rendering (L0 super-clusters → L1 sub-clusters →
  L2 file groups → L3 entries) with zoom-driven level switching and viewport culling
- **Activity feed**: `GET /api/dashboard/activity` with semantic grouping (same agent +
  topic/5-min window), content deltas, topic tags from ClusterStore
- **Agent enrichment**: `description` and `activity_summary` fields on AgentRecord,
  topic drift detection, `GET /api/dashboard/agents/reconnectable`,
  `POST /api/dashboard/agents/{id}/connect`, `POST /api/dashboard/agents/{id}/refresh-summary`
- **Collapsible sidebar**: 3-state sidebar (collapsed/narrow/wide) with context-aware content
  types (config, health, cluster, entry, agent, finding, history)
- **Cross-tab navigation**: `navigateToHistory(entryId)` deeplinks from any view to History tab
- **Agent identity CLI**: `corvia agent connect` interactive command for agent session management
- **Human-readable cluster labels**: `build_with_labels()` resolves entry IDs to source_file
  names or content previews throughout the dashboard

### Changed

- History tab rewritten from UUID-lookup to activity feed consumer with topic filter pills,
  agent dropdown, and semantic group expansion
- Graph view upgraded from flat force-directed to multi-tiered LOD with breadcrumb navigation
- Agent cards enhanced with description subtitle, topic tag pills, and drift indicator

## [0.3.6] - 2026-03-10

### Added

- **Real chat inference**: `corvia-inference` now runs real chat completion via
  llama-cpp-2/hf-hub instead of a stub, wired into the RAG pipeline for `ask()` mode
- **HNSW persistence**: LiteStore persists the HNSW index to disk for fast startup
  (no full rebuild on every `corvia serve`)
- **Stateless MCP**: MCP server is fully stateless — no session management needed,
  agent identity via `agent_id` tool parameter
- **M4 Observability**: `corvia-telemetry` crate with structured tracing, D45 span name contracts, configurable exporters (stdout, file, OTLP), and `TelemetryGuard` for log flushing
- **M4 Shared operations**: `ops.rs` module with 12 shared kernel operations callable from CLI and MCP (system_status, agents_list, config_get/set, gc_run, rebuild_index, etc.)
- **M4 MCP control plane**: 10 new MCP tools across 3 safety tiers — Tier 1 read-only (system_status, config_get, adapters_list, agents_list), Tier 2 low-risk (config_set, gc_run, rebuild_index), Tier 3 medium-risk (agent_suspend, merge_retry, merge_queue)
- **M4 Kernel instrumentation**: `#[tracing::instrument]` spans on agent_coordinator, merge_worker, lite_store, rag_pipeline using D45 span contracts
- **M4 CLI metrics**: `corvia status --metrics` flag for extended telemetry, agent counts, adapter discovery, and coordination status
- **M4 Config hot-reload**: `AppState` wraps `CorviaConfig` in `Arc<RwLock<>>` for runtime config updates via MCP
- **TelemetryConfig**: New config section in `corvia.toml` with exporter, log_format, and metrics_enabled settings

### Changed

- MCP server is always enabled on `corvia serve` (removed `--mcp` flag)
- `corvia_write` and `corvia_agent_status` accept `agent_id` as a regular tool
  parameter instead of requiring `_meta.agent_id`

### Fixed

- `corvia-inference` batch-feeds prompts to avoid llama.cpp assertion failures
- Eliminated unsafe transmute in corvia-inference, fixed defaults
- Adapter paths corrected in documentation
- `rebuild_index` now uses existing `LiteStore` reference via `as_any()` downcasting instead of opening a duplicate Redb lock
- Fixed duplicate `corvia.merge.process` span name — added `corvia.merge.process_entry` for per-entry granularity
- Fixed `corvia_config_set` MCP tool description listing wrong (restart-required) sections
- Removed always-zero `GcReport` fields (`closed_sessions_cleaned`, `inactive_agents_cleaned`)

## [0.3.4] - 2026-03-08

### Added

- **CLI server-aware routing**: CLI commands (`search`, `reason`, `history`, `graph`,
  `evolution`) detect a running corvia-server and route through REST API instead of
  opening Redb directly. Falls back to direct access when no server is running.
  Eliminates the Redb lock conflict during development/dogfooding.
- New `server_client` module in corvia-cli with HTTP client for all read-only operations

### Fixed

- Workspace ingest now auto-provisions inference model via `ensure_inference_ready()`
  (both `corvia ingest` workspace mode and `corvia workspace ingest`)

## [0.3.3] - 2026-03-08

### Added

- **Adapter plugin system** (D72-D79): Runtime adapter discovery via PATH scan,
  JSONL IPC protocol for adapter processes, ProcessAdapter wrapper, and
  ProcessChunkingStrategy for adapter-provided chunking
- **corvia-adapter-basic**: Filesystem ingestion adapter binary for non-Git sources
- **Graph edge improvements** (M3.4): Cross-file relation discovery, call-site
  extraction from Rust AST, markdown cross-reference extraction via pulldown-cmark,
  graph reinforcement scoring with configurable oversample factor
- **PostgresStore**: pgvector-backed storage tier with full trait implementation and
  `corvia migrate` command for moving data between backends
- **Benchmark script**: Vector vs graph-expanded retrieval comparison tool

### Changed

- Moved corvia-adapter-git into the monorepo as a workspace crate (was a separate repo)
- Refactored CLI ingestion to use runtime adapter discovery instead of hardcoded GitAdapter
- Updated IngestionAdapter trait to return SourceFile values (D69 clean break)
- Bumped workspace version to 0.3.3

### Fixed

- Auto-start corvia-inference when serving with provider=corvia
- Wire graph relations across files with multi-strategy resolution
- Blend cosine similarity into graph expansion scoring
- Replace unstable `ceil_char_boundary` with stable char boundary scan
- Implement missing `register_adapter_chunking` function in chunking pipeline
- Move corvia-adapter-git from dev-dependency to regular dependency in CLI
- Remove stale `GitAdapter::prepare()` calls (method was removed from trait)
- Remove unused `EntryStatus` import from context_builder

## [0.3.1] - 2026-03-04

### Added

- **M3.2 RAG pipeline**: Retriever, Augmenter, RagPipeline orchestrator, graph-expanded
  retrieval with alpha blending, MCP `corvia_context` and `corvia_ask` tools, REST
  `/v1/context` and `/v1/ask` endpoints
- **M3.3 Chunking pipeline**: ChunkingPipeline + FormatRegistry orchestrator,
  FallbackChunker, MarkdownChunker, ConfigChunker, PdfChunker, ChunkingStrategy trait
- **M3.1 gRPC inference server**: corvia-inference crate with fastembed ONNX backend,
  corvia-proto with gRPC service definitions, GrpcInferenceEngine and GrpcChatEngine clients,
  InferenceProvisioner for auto-lifecycle management
- **GenerationEngine trait**: Unified chat/generation interface (replaced ChatEngine)
- Prebuilt binary releases via GitHub Actions (D71)

### Changed

- Refactored ChatEngine → GenerationEngine trait (D63)
- Made AgentCoordinator mandatory, removed direct write fallback
- Returned `Arc<dyn InferenceEngine>` from `create_engine()` factory

### Fixed

- gRPC tests now self-bootstrapping, no silent skips
- Code review items for RAG pipeline (D62 compliance, naming, RBAC dedup)

## [0.3.0] - 2026-03-02

### Added

- **M3 Temporal queries**: Bi-temporal model with point-in-time snapshots,
  supersession chains, time-range evolution queries
- **M3 Knowledge graph**: Directed edges between entries, BFS traversal,
  shortest path, cycle detection (petgraph + Redb)
- **M3 Automated reasoning**: 5 deterministic health checks (StaleEntry,
  BrokenChain, OrphanedNode, DanglingImport, DependencyCycle) plus 2
  LLM-assisted checks (SemanticGap, Contradiction)
- **M2 Multi-agent coordination**: Session isolation, staging, crash
  recovery, LLM-assisted merge
- **M1 Core kernel**: LiteStore + SurrealStore, Ollama embedding pipeline,
  REST API, MCP server, CLI

[0.3.7]: https://github.com/corvia/corvia/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/corvia/corvia/compare/v0.3.4...v0.3.6
[0.3.4]: https://github.com/corvia/corvia/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/corvia/corvia/compare/v0.3.1...v0.3.3
[0.3.1]: https://github.com/corvia/corvia/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/corvia/corvia/releases/tag/v0.3.0
