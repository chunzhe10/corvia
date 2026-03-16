# MCP Agent Identity Fix — Design

> **Status:** Shipped (v0.3.6)

**Goal:** Enable `corvia_write` and `corvia_agent_status` to work from any MCP client
(including Claude Code) by accepting `agent_id` as a regular tool parameter.

**Problem:** The MCP server requires `_meta.agent_id` for write operations, but MCP
clients (Claude Code, Codex, etc.) only send declared `inputSchema` parameters. Since
`_meta` isn't in the schema, writes fail with "anonymous clients are read-only".

**Approach:** Add `agent_id` as an optional parameter to `corvia_write` and
`corvia_agent_status` inputSchemas. The server resolves agent identity with priority:
`arguments.agent_id` > `_meta.agent_id` > `None`.

## Changes

### 1. Tool schema (`tool_definitions()`)

Add to `corvia_write`:
```json
"agent_id": { "type": "string", "description": "Agent identity for attribution (e.g. 'claude-code')" }
```

Add to `corvia_agent_status`:
```json
"agent_id": { "type": "string", "description": "Agent identity (e.g. 'claude-code')" }
```

### 2. Agent ID resolution (`handle_tools_call`)

Extract agent_id from arguments first, fall back to `_meta.agent_id`:
```rust
let agent_id = arguments.get("agent_id").and_then(|v| v.as_str())
    .or(meta.and_then(|m| m.get("agent_id")).and_then(|v| v.as_str()));
```

### 3. No other changes

- Write still requires agent_id (rejects None)
- Auto-registration and session lifecycle unchanged
- `_meta.agent_id` still works for backwards compatibility
- Search, ask, context, history, graph, reason tools unchanged

## Files

- `crates/corvia-server/src/mcp.rs` — tool_definitions, handle_tools_call
- Tests in same file — update test for write with agent_id in arguments
