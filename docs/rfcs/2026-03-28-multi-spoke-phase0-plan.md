# Multi-Spoke Phase 0: Implementation Plan

**Date:** 2026-03-28
**Design:** `2026-03-28-multi-spoke-phase0-design.md`
**Branch:** `feat/35-multi-spoke-phase0`

---

## Task 1: Add `mcp_token` to AppState and generate on startup

**Files:** `crates/corvia-server/src/rest.rs`, `crates/corvia-cli/src/main.rs`

1. Add `pub mcp_token: Option<String>` field to `AppState`
2. In `cmd_serve()`, after loading config, check if `config.server.host` is NOT
   loopback (`127.0.0.1` or `::1`). If so:
   - Generate `uuid::Uuid::new_v4().to_string()`
   - Write token to `{data_dir}/mcp-token`
   - Pass `Some(token)` to AppState
3. If loopback: pass `None` to AppState
4. Print token file location on startup when generated

**Verify:** `cargo build`

---

## Task 2: Add bearer token validation to MCP write operations

**Files:** `crates/corvia-server/src/mcp.rs`

1. Define a `const WRITE_TOOLS` array listing all write tool names:
   `corvia_write`, `corvia_gc_run`, `corvia_config_set`, `corvia_rebuild_index`,
   `corvia_agent_suspend`, `corvia_merge_retry`, `corvia_pin`, `corvia_unpin`
2. In `handle_mcp_post()`, extract the Bearer token from the `Authorization` header
3. Pass the extracted token (Option<&str>) through to `handle_tools_call()`
4. In `handle_tools_call()`, after extracting `tool_name`, check:
   - If `state.mcp_token` is `Some(expected)` AND `tool_name` is in `WRITE_TOOLS`:
     - If provided token doesn't match: return JSON-RPC error (-32000, "Authorization required")
5. Return a clear error message distinguishing missing vs invalid token

**Verify:** `cargo test --workspace`

---

## Task 3: Create spoke Dockerfile

**Files:** `docker/spoke/Dockerfile` (new)

1. Base image: `node:22-bookworm`
2. Install system deps: git, curl, build-essential, gh CLI
3. `ARG CLAUDE_CODE_VERSION=1.0.18` with pinned install
4. Create non-root `spoke` user
5. HEALTHCHECK via `pgrep -f "claude"`
6. COPY entrypoint script
7. ENTRYPOINT to entrypoint script

**Verify:** File exists, Dockerfile syntax valid

---

## Task 4: Create spoke entrypoint script

**Files:** `docker/spoke/spoke-entrypoint.sh` (new)

1. Error trap with best-effort failure reporting to corvia
2. Copy credentials to user-owned location
3. Clone repo (with retry, --depth 100, query default branch)
4. Create/checkout feature branch from issue
5. Write `.mcp.json` pointing to hub with bearer token
6. Health-check hub MCP connectivity (10 retries, 3s interval)
7. Start Claude Code with dev-loop

**Verify:** `shellcheck docker/spoke/spoke-entrypoint.sh` (if available)

---

## Task 5: Add concurrent MCP connection test

**Files:** `crates/corvia-server/tests/` or inline test in mcp.rs

1. Write a test that spawns the server and hits it with concurrent requests
2. Verify N parallel search requests all return correctly
3. Verify write requests with valid token succeed
4. Verify write requests with invalid token get rejected
5. Verify write requests without token (when required) get rejected

**Verify:** `cargo test --workspace`

---

## Task 6: Verify bind address 0.0.0.0

**Type:** Manual verification, document results

1. Temporarily set `host = "0.0.0.0"` in corvia.toml
2. Start server, verify it binds
3. Verify token file is created in `.corvia/mcp-token`
4. Restore `host = "127.0.0.1"`

---

## Commit Plan

- Commit after Task 2 (token generation + auth validation)
- Commit after Task 4 (Dockerfile + entrypoint)
- Commit after Task 5 (tests)
