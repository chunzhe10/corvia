# Multi-Spoke Phase 0: Foundation & Security — Design

**Date:** 2026-03-28
**Status:** Active
**Issue:** chunzhe10/corvia-workspace#35
**Parent:** Multi-Spoke Workspace brainstorm (2026-03-28)

---

## Goal

Implement the security and infrastructure foundation required before any spoke
container management. These are blockers identified by the 5-persona design review.

## Scope

5 deliverables:

1. **Anthropic usage terms verification** (investigation, documented)
2. **MCP bearer token auth** on write operations
3. **Spoke Dockerfile** with pinned Claude Code version
4. **Server bind address** verification (0.0.0.0 support)
5. **Concurrent MCP connection** verification under load

## Task 1: Anthropic Usage Terms

**Type:** Investigation (no code)

**Findings:**
- `claude -p` (headless mode) is officially designed for CI/container use
- `ANTHROPIC_API_KEY` env var is documented for multi-instance container scenarios
- Multiple concurrent sessions are a supported feature of Claude Code
- Rate limits are the natural throttle for concurrent usage
- Max subscription provides Opus model access (`subscriptionType: "max"`)

**Recommendation:** Use `ANTHROPIC_API_KEY` as the primary auth mode for spokes.
Subscription credential mounting as documented fallback. This aligns with Anthropic's
documented CI/container use patterns.

## Task 2: MCP Bearer Token Auth

**Design:** Token-gated write operations on the MCP endpoint.

### Token Lifecycle

1. On server startup, if `config.server.host` is NOT loopback (parsed via `IpAddr::is_loopback()`):
   - If `{data_dir}/mcp-token` exists and is non-empty: reuse it (stable across restarts)
   - Otherwise: generate a UUID v4 token, write to `{data_dir}/mcp-token` with 0600 permissions
   - Store in `AppState` as `Option<String>`
   - Token comparison uses constant-time algorithm to prevent timing attacks
2. If host IS loopback: no token generated (backward compatible)

### Protected Operations (write tools)

These MCP tools require `Authorization: Bearer <token>`:
- `corvia_write`
- `corvia_gc_run`
- `corvia_config_set`
- `corvia_rebuild_index`
- `corvia_agent_suspend`
- `corvia_merge_retry`
- `corvia_pin`
- `corvia_unpin`

### Unprotected Operations (read tools)

These remain open for dashboard and debugging:
- `corvia_search`, `corvia_ask`, `corvia_context`
- `corvia_system_status`, `corvia_config_get`
- `corvia_history`, `corvia_graph`
- `corvia_agents_list`, `corvia_adapters_list`
- `corvia_agent_status`, `corvia_merge_queue`
- `corvia_reason`

### Implementation

**Middleware approach:** Extract Bearer token from `Authorization` header in
`handle_mcp_post()`. After dispatching to identify the tool name, check if it's
a write tool. If so, validate the token. Return JSON-RPC error on mismatch.

```
POST /mcp
Authorization: Bearer <token>
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"corvia_write",...}}
```

On auth failure: JSON-RPC error response with code -32000 and message
"Authorization required for write operations".

### Backward Compatibility

- Loopback binding (`127.0.0.1`): No token required. Existing workflows unchanged.
- Non-loopback binding (`0.0.0.0`): Token required on writes. Token file created
  automatically.

## Task 3: Spoke Dockerfile

**File:** `docker/spoke/Dockerfile`

### Contents

- Base: `node:22-bookworm`
- System deps: git, curl, build-essential, gh CLI
- Claude Code: pinned version via `ARG CLAUDE_CODE_VERSION`
- Non-root user: `spoke` user for security
- Health check: `pgrep -f "claude"`
- Entrypoint: `spoke-entrypoint.sh`

### Entrypoint Script

**File:** `docker/spoke/spoke-entrypoint.sh`

1. Copy credentials to user-owned location
2. Clone repo (with retry, `--depth 100`)
3. Create/checkout feature branch
4. Write `.mcp.json` pointing to hub
5. Health-check hub MCP connectivity (10 retries, 3s interval)
6. Start Claude Code with dev-loop
7. Error trap: report failures to corvia (best-effort)

## Task 4: Server Bind Address

**Status:** Already implemented. `ServerConfig.host` reads from `corvia.toml`.

**Verification:** Change `host = "0.0.0.0"` and confirm the server accepts
connections from Docker network addresses. This is a test, not new code.

## Task 5: Concurrent MCP Connections

**Type:** Verification test

Spawn N concurrent HTTP clients hitting MCP endpoints simultaneously.
Verify no request dropping, no panics, correct response ordering.

Test: 10 concurrent `corvia_search` + 5 concurrent `corvia_write` (with valid token).

---

## Implementation Order

1. MCP bearer token auth (Task 2) — core security gate
2. Spoke Dockerfile + entrypoint (Task 3) — depends on knowing token injection pattern
3. Bind address verification (Task 4) — quick test
4. Concurrent connection test (Task 5) — validates the whole stack
5. Usage terms documentation (Task 1) — written into design doc above

## Key Files to Modify

| File | Change |
|------|--------|
| `crates/corvia-server/src/mcp.rs` | Add token validation to `handle_mcp_post()` |
| `crates/corvia-server/src/rest.rs` | Add `mcp_token: Option<String>` to `AppState` |
| `crates/corvia-cli/src/main.rs` | Generate token on startup, write to file |
| `crates/corvia-common/src/config.rs` | No changes needed (ServerConfig already has host) |
| `docker/spoke/Dockerfile` | New file |
| `docker/spoke/spoke-entrypoint.sh` | New file |
