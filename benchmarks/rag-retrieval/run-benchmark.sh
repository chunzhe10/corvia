#!/usr/bin/env bash
set -euo pipefail

# ── RAG Retrieval A/B Benchmark ─────────────────────────────────────────────
# Runs eval.py across 3 pipeline configurations via hot-swap, then compares.
#
# Usage:
#   ./run-benchmark.sh                    # Use defaults
#   ./run-benchmark.sh --url http://host  # Custom server URL
#   ./run-benchmark.sh --results-dir dir  # Custom output directory
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="http://127.0.0.1:8020"
SCOPE_ID="corvia"
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)         URL="$2"; shift 2 ;;
    --scope-id)    SCOPE_ID="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%S)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

MCP_URL="$URL/mcp"

phase() { echo ""; echo "== $1 =="; }

# ── Helper: call MCP tool (uses sys.argv to avoid shell injection) ─────────
mcp_call() {
  local tool="$1" args="$2"
  curl -s -X POST "$MCP_URL" -H "Content-Type: application/json" -d "$(python3 -c "
import json, sys
tool, args_json = sys.argv[1], sys.argv[2]
print(json.dumps({
    'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
    'params': {
        'name': tool,
        '_meta': {'confirmed': True},
        'arguments': json.loads(args_json)
    }
}))
" "$tool" "$args")"
}

# ── Helper: hot-swap config via MCP ────────────────────────────────────────
config_set() {
  local section="$1" key="$2" value="$3"
  local payload
  payload=$(python3 -c "
import json, sys
scope_id, section, key, value_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
print(json.dumps({
    'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
    'params': {
        'name': 'corvia_config_set',
        '_meta': {'confirmed': True},
        'arguments': {
            'scope_id': scope_id,
            'section': section,
            'key': key,
            'value': json.loads(value_json)
        }
    }
}))
" "$SCOPE_ID" "$section" "$key" "$value")
  local resp
  resp=$(curl -s -X POST "$MCP_URL" -H "Content-Type: application/json" -d "$payload")
  # Check for errors
  if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}).get('content',[{}])[0].get('text',''); p=json.loads(r) if r else {}; print(p.get('status',''))" 2>/dev/null | grep -q "updated"; then
    echo "  Config updated: $section.$key"
  else
    echo "  Config response: $resp" | head -c 200
  fi
}

# ── Cleanup trap: restore config on failure ────────────────────────────────
restore_config() {
  echo ""
  echo "== Restoring default config =="
  config_set "rag" "pipeline.searchers" '["vector"]' 2>/dev/null
  config_set "rag" "pipeline.fusion" '"passthrough"' 2>/dev/null
  config_set "rag" "pipeline.expander" '"graph"' 2>/dev/null
}
trap restore_config EXIT

# ── Phase 0: Preflight ────────────────────────────────────────────────────
phase "Phase 0: Preflight"
curl -sf "$URL/health" > /dev/null || { echo "FAIL: Server not responding at $URL"; exit 1; }
echo "Server OK at $URL"
echo "Results -> $RESULTS_DIR"

# ── Phase 0b: Ensure tantivy index is populated ───────────────────────────
phase "Phase 0b: Initialize tantivy index"
# Activate pipeline with BM25 to wire tantivy into LiteStore
config_set "rag" "pipeline.searchers" '["vector", "bm25"]'
# Rebuild index to populate tantivy from existing entries
echo "  Rebuilding index (this may take a few minutes)..."
mcp_call "corvia_rebuild_index" '{}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d.get('result',{}).get('content',[{}])[0].get('text','{}')
r = json.loads(text)
print(f'  Index rebuilt: {r.get(\"entries_indexed\", 0)} entries indexed')
" 2>/dev/null || echo "  (rebuild may have been skipped if already populated)"

# ── Phase 1: Vector-only baseline ──────────────────────────────────────────
phase "Phase 1: Vector-only baseline"
config_set "rag" "pipeline.searchers" '["vector"]'
config_set "rag" "pipeline.fusion" '"passthrough"'
config_set "rag" "pipeline.expander" '"graph"'
sleep 1  # Let pipeline swap settle
python3 "$SCRIPT_DIR/eval.py" \
  --url "$URL" --scope-id "$SCOPE_ID" \
  --label "vector-only" \
  -o "$RESULTS_DIR/vector-only.json"

# ── Phase 2: Hybrid (vector + BM25 + RRF) ──────────────────────────────────
phase "Phase 2: Hybrid (vector + BM25 + RRF)"
config_set "rag" "pipeline.searchers" '["vector", "bm25"]'
config_set "rag" "pipeline.fusion" '"rrf"'
config_set "rag" "pipeline.expander" '"graph"'
sleep 1
python3 "$SCRIPT_DIR/eval.py" \
  --url "$URL" --scope-id "$SCOPE_ID" \
  --label "hybrid-rrf" \
  -o "$RESULTS_DIR/hybrid-rrf.json"

# ── Phase 3: BM25-only (isolation) ─────────────────────────────────────────
phase "Phase 3: BM25-only (isolation)"
config_set "rag" "pipeline.searchers" '["bm25"]'
config_set "rag" "pipeline.fusion" '"passthrough"'
config_set "rag" "pipeline.expander" '"noop"'
sleep 1
python3 "$SCRIPT_DIR/eval.py" \
  --url "$URL" --scope-id "$SCOPE_ID" \
  --label "bm25-only" \
  -o "$RESULTS_DIR/bm25-only.json"

# ── Phase 4: Compare results ──────────────────────────────────────────────
# Config is restored by the EXIT trap handler.
phase "Phase 4: Compare results"
echo ""
python3 "$SCRIPT_DIR/compare.py" \
  "$RESULTS_DIR/vector-only.json" \
  "$RESULTS_DIR/hybrid-rrf.json" \
  "$RESULTS_DIR/bm25-only.json"

echo ""
echo "Results saved to: $RESULTS_DIR"
ls -1 "$RESULTS_DIR"
