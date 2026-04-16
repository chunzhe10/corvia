# Agentic Pipeline and Guardrails Research

**Date**: 2026-04-14
**Context**: Evaluating whether corvia v2 should include agentic pipeline capabilities

## Landscape Summary

| Framework | Stars | License | Language | Guardrails | Knowledge-Native | Local? |
|-----------|-------|---------|----------|------------|-------------------|--------|
| LangGraph | 29.2k | MIT | Python | Interrupts + checkpoints | No | No (needs Postgres) |
| CrewAI | 48.8k | MIT | Python | None built-in | No | Partial |
| AutoGen | 57.1k | MIT | Python/.NET | None built-in | No | Partial |
| Claude Agent SDK | -- | Commercial | Python/TS | 15+ hook events | Via MCP | No (API) |
| OpenAI Agents SDK | 20.8k | MIT | Python | Tripwire decorators | No | No (API) |
| Google ADK | 19k | Apache 2.0 | Py/TS/Go/Java | Tool confirmation | No | Partial |
| Pydantic AI | 16.3k | MIT | Python | Via capabilities | No | Partial |
| Mastra | 23k | Apache 2.0 | TypeScript | None built-in | Built-in RAG | Partial |
| Rig | 6.9k | MIT | Rust | None | Via vector stores | Yes |
| rs-graph-llm | 290 | OSS | Rust | WaitForInput | Via examples | Yes |
| NeMo Guardrails | 6k | Apache 2.0 | Python | Core purpose (5 types) | Retrieval rails | Yes |
| Guardrails AI | 6.7k | Apache 2.0 | Python | Core purpose | No | Yes |
| MS Agent Gov | 1k | MIT | Multi-lang | Core purpose (OWASP) | No | Yes |

## Key Gap

No framework treats knowledge retrieval as a native pipeline step with built-in
guardrails, running locally with no cloud dependency.

## Rust vs Python Performance

| Metric | Rust agents | Python agents |
|--------|-------------|---------------|
| Peak Memory | ~1 GB | ~5 GB |
| Cold Start | 4 ms | 62 ms |
| Throughput | 4.97 rps | 4.15 rps |
| 50 instances RAM | ~51 GB | ~279 GB |

## Guardrail Patterns (ranked for solo Rust dev)

### Already have (keep)
1. Claude Code hook consumer (hooks/mod.rs) -- typed Rust handlers, stdin JSON protocol
2. Kernel EventBus (event_bus.rs) -- tokio::broadcast, 7 event types

### High value, low effort
3. Tower middleware for tool-call pipeline -- already in dep tree via Axum
4. Git-style hook directories -- .corvia/hooks/pre-tool-use/ executables

### Medium value, medium effort (v2 candidates)
5. Extism WASM plugins -- sandboxed user-supplied guardrails, adds wasmtime dep
6. Cedar policies (Amazon, pure Rust) -- microsecond evaluation, forbid-overrides-permit

### Low priority for solo dev
7. Unix pipe composition (corvia pipe command)
8. OPA/Rego to WASM (more complex than Cedar)

## Minimum Viable Agentic Abstraction

Five primitives needed:
1. Step -- unit of work (function, LLM call, tool invocation)
2. Pipeline -- DAG of Steps with typed state
3. Guard -- pre/post hook on any Step (allow/deny/modify)
4. Retriever -- knowledge retrieval Step querying local index
5. Runner -- executes Pipeline with optional checkpointing

## MCP as Cross-Platform Interface

stdio MCP subprocess (not HTTP server) works across Claude Code, OpenAI Codex,
Cursor, Windsurf, Gemini. One binary, no port management:

```json
{
  "mcpServers": {
    "corvia": {
      "command": "corvia",
      "args": ["mcp"],
      "type": "stdio"
    }
  }
}
```

## Sources

See full research notes in conversation history (2026-04-14).
Key references: LangGraph docs, Rig docs, Claude Agent SDK hooks reference,
OpenAI Agents SDK guardrails docs, NeMo Guardrails Colang architecture,
Microsoft Agent Governance Toolkit deep dive, Extism plugin system,
Cedar policy language docs, Rust agent benchmarks (dev.to/saivishwak).
