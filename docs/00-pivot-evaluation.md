# Corvia V2 Pivot Evaluation

**Date**: 2026-04-14
**Status**: In Progress
**Author**: chunzhe

## Context

Corvia v1 has served its purpose. The codebase has grown too large for a solo developer
to maintain. It contains unnecessary complexity that generates hard-to-debug issues.
This document captures the competitive landscape research and strategic direction for v2.

## V1 Lessons Learned

- **Too much surface area**: Server, dashboard, inference, adapters, CLI, MCP, hooks,
  devcontainer tooling. Each adds maintenance burden.
- **Centralized server was premature**: A running `corvia serve` process adds infra
  complexity. For the actual use case (solo/small team), everything should run on-machine.
- **vLLM/inference server not needed for solo use**: Embedding can be simpler. Local
  ONNX or external API is sufficient without a dedicated gRPC inference service.
- **Bugs compound with complexity**: More crates, more features, more integration
  points means more edge cases a single developer cannot cover.
- **Git is the right persistence layer**: Docs and knowledge synced to git. People
  build and consume from there. No database needed.

## V2 Design Principles

1. **Local-first, no centralized server** -- runs entirely on the developer's machine
2. **Git-synced knowledge** -- all docs/knowledge stored as files, synced via git
3. **Lean and maintainable** -- small enough for one person to understand and debug
4. **Reuse open-source where possible** -- don't build what others maintain better
5. **RBAC from a local perspective** -- access control via filesystem/git permissions,
   not a server-side auth layer
6. **No vLLM/inference server for solo** -- use simpler embedding (ONNX direct, or API)

## Competitive Landscape Research (April 2026)

### Market Overview

The AI agent memory market is $6.27B (2025), projected $28.45B by 2030. Almost entirely
focused on individual user personalization. "Organizational memory with quality control
and knowledge evolution" is essentially unoccupied.

### Claude's Memory Systems

Claude has three memory layers:

1. **Chat Memory (claude.ai)**: RAG search through past conversations. 24-hour synthesis
   cycle. Per-user, per-project. No vector DB, no graph.

2. **CLAUDE.md (Claude Code)**: Human-written markdown instructions loaded at session
   start. Cascading hierarchy (managed policy > project > user > local).

3. **Auto Memory (Claude Code)**: File-based. MEMORY.md index + topic files. No vector
   search. Uses LLM-as-selector (sends manifest to Sonnet). autoDream consolidation
   runs every 24h + 5 sessions. Four types: User, Feedback, Project, Reference.

**Key architectural choice**: Anthropic chose files over vectors, betting that large
context windows make retrieval unnecessary. Client-side only. Single-user. No sharing.

**Open-sourced**: Memory Tool API (`memory_20250818`) in Python/TypeScript SDKs.
Six file operations (view, create, str_replace, insert, delete, rename). Entirely
client-side file I/O.

### Direct Competitors

| System | Approach | Stars | License | Key Strength | Key Weakness |
|--------|----------|-------|---------|-------------|-------------|
| **Mem0** | LLM extraction + vector | 53K | Apache 2.0 | Ecosystem (21 integrations) | 97.8% junk rate in audit |
| **Graphiti/Zep** | Bi-temporal knowledge graph | ~8K | Apache 2.0 | Best temporal reasoning (63.8%) | No org scoping, infra-heavy |
| **Cognee** | Poly-store (vector + graph) | ~7K | Apache 2.0 | Simple onboarding, 93% accuracy | No developer focus |
| **Letta** | Tiered memory (core/recall/archival) | ~15K | Apache 2.0 | Agent-managed memory | Requires adopting their runtime |
| **LangMem** | LangChain toolkit | ~3K | MIT | Procedural memory | Coupled to LangChain |
| **GitHub Copilot Memory** | Structured objects + citations | N/A | Proprietary | Code-aware, repo-scoped | Proprietary, single-repo |

### What Nobody Does (corvia's unique territory)

- Organizational knowledge scoping (team-wide, not user-specific)
- Multi-repo workspace awareness
- Knowledge quality signals (Mem0 calls this "unsolved")
- Supersession/evolution tracking with provenance
- Code structural intelligence combined with persistent knowledge
- Multi-agent coordination with quality control
- Enterprise governance over agent knowledge
- Self-hosted, zero-cloud organizational memory

### Claude Memory vs Corvia

| Dimension | Claude Memory | Corvia |
|-----------|-------------|--------|
| Retrieval | LLM reads whole file | Semantic + BM25 + graph |
| Quality | autoDream (non-deterministic) | 5 deterministic health checks |
| Temporal | Age warnings only | Bi-temporal, supersession chains |
| Multi-agent | Single user | Session isolation + merge |
| Code awareness | None | tree-sitter (330+ edges) |
| Org scope | Per-user | Multi-repo workspaces |

## Key Decisions Made

### Architecture: Option A (Knowledge Infrastructure)
Corvia is a knowledge engine that agents use. Not an agent framework itself.
Multiple agents (Claude, OpenAI, LangGraph, Rig) share one corvia instance via MCP.
Like PostgreSQL is to web apps, corvia is to AI agents.

### Memory Takeover
Corvia replaces native AI memory entirely (Claude auto-memory, OpenAI memory disabled).
One source of truth. No dual-read, no redundancy.

### No corvia_ask (Drop RAG Generation)
Every AI assistant that calls corvia IS an LLM. Pre-generating answers is redundant.
corvia_search returns ranked chunks. The calling LLM synthesizes.
This eliminates the need for: LLM client, inference runtime, API keys, token budgeting.
Matches how OpenAI file_search, Claude Projects RAG, and Claude chat search all work.

### Guardrails Inside MCP (No External Hooks)
Claude Code hooks are fragile (binary mismatch, shell execution, exit codes).
Guardrails live inside corvia's MCP tools. corvia_check lets agents validate actions.
Cedar policies (pure Rust, microsecond evaluation) for complex rules.

### Interface: stdio MCP Only
No HTTP server. No dashboard. corvia runs as an MCP stdio subprocess.
Works across Claude Code, OpenAI Codex, Cursor, Gemini, etc.

### Embedding: Embedded ONNX
Small model (all-MiniLM-L6-v2, 384d) embedded in binary or downloaded on first run.
No inference server, no API keys, fully local.

### Minimal Tool Surface (Final)
MCP tools: corvia_search, corvia_write, corvia_status (3 total)
CLI commands: ingest, search, write, bench, status, mcp (6 total)

Dropped: corvia_ask (caller is LLM), corvia_graph (internal optimization),
corvia_history (git provides this), corvia_check (rules in AGENTS.md),
corvia_bench as MCP (CLI only).

## Open Questions for V2

- [ ] Which open-source components to reuse? (Graphiti for temporal? Cognee for ingest?)
- [ ] What's the right embedded DB? (Redb? SQLite? Both?)
- [ ] How does RBAC work in a git-synced, local-first model?
- [ ] Embedding model: bundle in binary or download on first run?
- [ ] Should corvia-guard (Cedar) exist as separate crate or inline in core?

## Research Documents

- `docs/decisions/2026-04-14-v2-pivot-evaluation.md` -- this file
- `docs/decisions/2026-04-14-agentic-pipeline-research.md` -- frameworks and guardrails

## Sources

- Mem0: github.com/mem0ai/mem0 (Apache 2.0)
- Graphiti: github.com/getzep/graphiti (Apache 2.0)
- Cognee: github.com/topoteretes/cognee (Apache 2.0)
- Letta: github.com/letta-ai/letta (Apache 2.0)
- LangMem: github.com/langchain-ai/langmem (MIT)
- Claude Memory Tool: anthropic SDK examples
- Market data: Atlan analysis, OSS Insight agent memory race 2026
- Academic: arxiv 2512.13564, 2501.13956, 2502.12110
