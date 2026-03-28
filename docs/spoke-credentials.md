# Spoke Credential Management

## Authentication Modes

Spokes authenticate to Claude Code using one of three modes:

### 1. Mounted credentials (default)

The hub's `.credentials.json` is mounted read-only into each spoke container.
This uses the owner's subscription (e.g., Max plan with Opus access).

**Pros:** No extra API key management. Uses existing subscription.
**Cons:** Token can expire. All spokes share one identity.

### 2. ANTHROPIC_API_KEY

Set `ANTHROPIC_API_KEY` in the hub's environment. Injected into spokes at
creation time when `auth_mode = "api_key"` in corvia.toml.

```toml
[workspace.spokes]
auth_mode = "api_key"
```

**Pros:** No expiry. Clear billing. Works in headless/CI environments.
**Cons:** Pay-per-token (not subscription). API key must be managed.

### 3. apiKeyHelper (enterprise)

For rotating tokens (Vault, AWS STS), configure a helper script. Claude Code
calls this script at startup and on 401 errors.

```json
{"apiKeyHelper": "/usr/local/bin/get-token.sh"}
```

Set `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` to control refresh interval.

**Pros:** Automatic rotation. Works with enterprise secret managers.
**Cons:** Requires custom infrastructure.

## Credential Rotation

### Rotating mounted credentials

1. Log in to Claude Code on the hub (interactive `claude` session).
2. Restart all spokes to pick up the new credentials:
   ```bash
   corvia workspace spoke restart --all
   ```

### Rotating ANTHROPIC_API_KEY

1. Generate a new key at https://console.anthropic.com/settings/keys.
2. Update the hub's environment (`.env` or secrets manager).
3. Destroy and recreate spokes (env vars are set at creation time):
   ```bash
   corvia workspace spoke destroy --all --yes
   # Re-create spokes with the new key in the environment
   corvia workspace spoke create --repo corvia --issue 42
   ```

### Rotating GITHUB_TOKEN

1. Generate a new fine-grained PAT at https://github.com/settings/tokens.
2. Minimum scopes: `contents:write`, `pull_requests:write`, `issues:write`, `metadata:read`.
3. Destroy and recreate spokes (same as API key rotation).

## Credential Revocation

### Revoking a compromised spoke's access

1. Destroy the spoke immediately:
   ```bash
   corvia workspace spoke destroy <name> --immediate
   ```
2. The MCP token for that spoke is invalidated when the container is removed.
3. If the shared credentials may be compromised:
   - For ANTHROPIC_API_KEY: revoke at https://console.anthropic.com/settings/keys
   - For GITHUB_TOKEN: revoke at https://github.com/settings/tokens
   - Generate new credentials and restart remaining spokes.

### Emergency: revoke all spoke access

```bash
# Kill all spokes immediately
corvia workspace spoke destroy --all --yes --immediate

# Regenerate the MCP token (requires server restart)
rm .corvia/mcp-token
corvia-dev down && sleep 3 && corvia-dev up --no-foreground
```

## Hub Restart Procedure

When the hub (corvia server) restarts, all spoke MCP connections drop.
Claude Code's MCP client retries automatically on the next tool call, so
brief restarts (under 30 seconds) are usually transparent.

For longer restarts or binary updates:

```bash
# 1. Restart the hub
corvia-dev down
sleep 3
corvia-dev up --no-foreground

# 2. Verify hub is healthy
curl -sf http://localhost:8020/api/dashboard/status

# 3. If spokes are unresponsive, restart them all
corvia workspace spoke restart --all
```

When to use `spoke restart --all`:
- Hub was down for more than 30 seconds
- Spokes show "unhealthy" in the dashboard after hub restart
- MCP write operations from spokes are failing

When you do NOT need to restart spokes:
- Brief hub restart (under 30 seconds). Claude Code retries MCP calls.
- Hub config changes that don't affect the MCP endpoint.

## Long-Running Session Considerations

For spokes running dev-loop on large issues (hours), credential expiry is a
risk with mounted subscription credentials. Mitigations:

1. **Prefer ANTHROPIC_API_KEY** for long sessions. API keys don't expire.
2. **Use apiKeyHelper** with a TTL-based refresh for enterprise deployments.
3. **Monitor spoke health** via the dashboard. Unhealthy spokes may indicate
   credential issues.
4. The entrypoint validates credentials before starting Claude Code. If
   validation fails, the spoke exits with a clear error message.
