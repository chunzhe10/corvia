#!/bin/bash
set -euo pipefail

# --- corvia spoke entrypoint ---
# Runs inside a spoke container. Clones a repo, connects to the hub's MCP
# server, and starts Claude Code on an assigned issue or branch.
#
# Required env vars:
#   REPO_URL          - Git repo URL (e.g., https://github.com/owner/repo.git)
#   CORVIA_MCP_URL    - Hub MCP endpoint (e.g., http://app:8020/mcp)
#
# Optional env vars:
#   CORVIA_ISSUE      - GitHub issue number to work on
#   CORVIA_BRANCH     - Branch name (if not derived from issue)
#   CORVIA_MCP_TOKEN  - Bearer token for MCP write operations
#   CORVIA_AGENT_ID   - Agent identity for corvia attribution
#   GITHUB_TOKEN      - GitHub PAT for push/PR operations
#   ANTHROPIC_API_KEY  - API key for Claude Code (alternative to mounted credentials)

# --- Validate required env vars ---

: "${REPO_URL:?REPO_URL is required (e.g., https://github.com/owner/repo.git)}"
: "${CORVIA_MCP_URL:?CORVIA_MCP_URL is required (e.g., http://app:8020/mcp)}"

# --- Error reporting ---

report_failure() {
    local msg="$1"
    echo "SPOKE FAILURE: ${msg}" >&2
    # Best-effort write to corvia using jq for safe JSON construction
    local payload
    payload=$(jq -n \
        --arg agent "${CORVIA_AGENT_ID:-spoke-unknown}" \
        --arg msg "Spoke ${CORVIA_AGENT_ID:-unknown} failed: ${msg}" \
        '{jsonrpc:"2.0",id:1,method:"tools/call",params:{name:"corvia_write",arguments:{scope_id:"corvia",agent_id:$agent,content_role:"finding",source_origin:"workspace",content:$msg}}}')
    curl -sf -X POST "${CORVIA_MCP_URL:-}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${CORVIA_MCP_TOKEN:-}" \
        -d "$payload" 2>/dev/null || true
}

trap 'report_failure "Entrypoint failed at line $LINENO (exit $?)"' ERR

# --- Credential setup ---

# Copy mounted credentials to user-owned location (mounted as read-only)
mkdir -p ~/.claude
if [ -f /spoke-config/.credentials.json ]; then
    cp /spoke-config/.credentials.json ~/.claude/.credentials.json
    chmod 600 ~/.claude/.credentials.json
    echo "Credentials: mounted file"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Credentials: ANTHROPIC_API_KEY"
else
    report_failure "No credentials: mount .credentials.json or set ANTHROPIC_API_KEY"
    exit 1
fi

# --- Repository setup ---

clone_with_retry() {
    # Extract owner/repo from GitHub URL for API call
    local repo_slug
    repo_slug=$(echo "${REPO_URL}" | sed 's|.*github\.com[/:]||; s|\.git$||')
    local default_branch
    default_branch=$(gh api "repos/${repo_slug}" --jq '.default_branch' 2>/dev/null || echo "master")

    for i in 1 2 3; do
        if git clone --depth 100 --branch "${default_branch}" "${REPO_URL}" /workspace 2>&1; then
            return 0
        fi
        echo "Clone failed (attempt $i/3), retrying in $((3 + RANDOM % 3))s..." >&2
        sleep $((3 + RANDOM % 3))
    done
    return 1
}

# Skip clone if workspace already populated (spoke restart case)
if [ -d "/workspace/.git" ]; then
    echo "Workspace already populated, skipping clone."
    git -C /workspace fetch origin && git -C /workspace pull --rebase || true
else
    if ! clone_with_retry; then
        report_failure "Git clone failed after 3 attempts"
        exit 1
    fi
fi

cd /workspace

# --- Branch setup ---

if [ -n "${CORVIA_ISSUE:-}" ]; then
    # Derive branch name from issue
    ISSUE_TITLE=$(gh issue view "${CORVIA_ISSUE}" --json title --jq '.title' 2>/dev/null || echo "")
    SLUG=$(echo "${ISSUE_TITLE}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | head -c 40)
    BRANCH="feat/${CORVIA_ISSUE}-${SLUG}"

    # Check if branch exists on remote
    if git ls-remote --heads origin "${BRANCH}" | grep -q .; then
        git fetch origin "${BRANCH}" && git checkout "${BRANCH}"
    else
        git checkout -b "${BRANCH}"
    fi
elif [ -n "${CORVIA_BRANCH:-}" ]; then
    if git ls-remote --heads origin "${CORVIA_BRANCH}" | grep -q .; then
        git fetch origin "${CORVIA_BRANCH}" && git checkout "${CORVIA_BRANCH}"
    else
        git checkout -b "${CORVIA_BRANCH}"
    fi
fi

# --- MCP configuration ---

# Write .mcp.json to user home (NOT workspace) to prevent accidental git commits
MCP_CONFIG='{"mcpServers":{"corvia":{"type":"http","url":"'"${CORVIA_MCP_URL}"'"'
if [ -n "${CORVIA_MCP_TOKEN:-}" ]; then
    MCP_CONFIG="${MCP_CONFIG}"',"headers":{"Authorization":"Bearer '"${CORVIA_MCP_TOKEN}"'"}'
fi
MCP_CONFIG="${MCP_CONFIG}"'}}}'
mkdir -p ~/.claude
echo "${MCP_CONFIG}" > ~/.claude/.mcp.json
echo "MCP config written to ~/.claude/.mcp.json (hub: ${CORVIA_MCP_URL})"

# --- Hub health check ---

echo "Checking MCP connectivity..."
HUB_STATUS_URL="${CORVIA_MCP_URL%/mcp}/api/dashboard/status"
for i in $(seq 1 10); do
    if curl -sf "${HUB_STATUS_URL}" >/dev/null 2>&1; then
        echo "Hub MCP reachable."
        break
    fi
    if [ "$i" -eq 10 ]; then
        report_failure "Hub MCP unreachable at ${HUB_STATUS_URL} after 10 retries"
        exit 1
    fi
    echo "Waiting for hub MCP... (attempt $i/10)"
    sleep $((3 + RANDOM % 3))
done

# --- Start Claude Code ---

echo "Starting Claude Code..."
if [ -n "${CORVIA_ISSUE:-}" ]; then
    exec claude -p "/dev-loop ${CORVIA_ISSUE}"
elif [ -n "${CORVIA_PROMPT:-}" ]; then
    exec claude -p "${CORVIA_PROMPT}"
else
    exec claude
fi
