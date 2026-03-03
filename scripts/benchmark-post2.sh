#!/usr/bin/env bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORVIA="$REPO_ROOT/target/release/corvia"
CORVIA_INFERENCE="$REPO_ROOT/target/release/corvia-inference"
INFERENCE_PORT=8030
SERVER_PORT=8020
SCOPE_ID="corvia-introspect"
RESULTS_DIR=""
SKIP_SETUP=false
INFERENCE_PID=""
SERVER_PID=""

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)  SKIP_SETUP=true; shift ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%S)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmark-results/$TIMESTAMP}"

# ── Cleanup trap ────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "── Teardown ──"
  [[ -n "$SERVER_PID" ]]    && kill "$SERVER_PID" 2>/dev/null && echo "Stopped corvia serve (pid $SERVER_PID)"
  [[ -n "$INFERENCE_PID" ]] && kill "$INFERENCE_PID" 2>/dev/null && echo "Stopped corvia-inference (pid $INFERENCE_PID)"
  echo "Results saved to: $RESULTS_DIR"
}
trap cleanup EXIT

# ── Helpers ─────────────────────────────────────────────────────────────
phase() { echo ""; echo "── $1 ──"; }

wait_for_http() {
  local url="$1" timeout="$2" elapsed=0
  while ! curl -sf "$url" > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [[ $elapsed -ge $timeout ]]; then
      echo "FAIL: $url not ready after ${timeout}s"
      exit 1
    fi
  done
}

wait_for_grpc() {
  local port="$1" timeout="$2" elapsed=0
  while ! curl -sf "http://127.0.0.1:$port" > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [[ $elapsed -ge $timeout ]]; then
      echo "FAIL: gRPC on port $port not ready after ${timeout}s"
      exit 1
    fi
  done
}

# ── Phase 1: Preflight ─────────────────────────────────────────────────
phase "Phase 1: Preflight"

[[ -x "$CORVIA" ]]           || { echo "FAIL: $CORVIA not found. Run: cargo build --release"; exit 1; }
[[ -x "$CORVIA_INFERENCE" ]] || { echo "FAIL: $CORVIA_INFERENCE not found. Run: cargo build --release"; exit 1; }
command -v curl >/dev/null    || { echo "FAIL: curl not found"; exit 1; }
command -v jq >/dev/null      || { echo "FAIL: jq not found"; exit 1; }

mkdir -p "$RESULTS_DIR"
echo "Binaries OK. Results → $RESULTS_DIR"

# ── Phase 2-5: Setup (unless --skip-setup) ──────────────────────────────
WORK_DIR="$REPO_ROOT/.benchmark-workspace"

if [[ "$SKIP_SETUP" == false ]]; then

  # Phase 2: Inference
  phase "Phase 2: Start corvia-inference"
  "$CORVIA_INFERENCE" serve --port "$INFERENCE_PORT" &
  INFERENCE_PID=$!
  echo "corvia-inference starting (pid $INFERENCE_PID)..."
  sleep 3
  echo "corvia-inference ready on :$INFERENCE_PORT"

  # Phase 3: Init
  phase "Phase 3: Initialize LiteStore workspace"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
  cd "$WORK_DIR"
  "$CORVIA" init --store lite

  # Patch corvia.toml: switch from ollama to corvia-inference
  if command -v sed >/dev/null; then
    sed -i 's/provider = "ollama"/provider = "corvia"/' corvia.toml
    sed -i 's|url = "http://127.0.0.1:11434"|url = "http://127.0.0.1:'"$INFERENCE_PORT"'"|' corvia.toml
    sed -i 's/model = "nomic-embed-text"/model = "nomic-embed-text-v1.5"/' corvia.toml
  fi
  echo "Config patched: provider=corvia, port=$INFERENCE_PORT"
  cat corvia.toml

  # Phase 4: Ingest
  phase "Phase 4: Self-ingest Corvia codebase"
  "$CORVIA" demo --keep
  echo "Ingest complete"

  # Phase 5: Serve
  phase "Phase 5: Start REST server"
  "$CORVIA" serve &
  SERVER_PID=$!
  echo "corvia serve starting (pid $SERVER_PID)..."
  wait_for_http "http://127.0.0.1:$SERVER_PORT/health" 30
  echo "Server ready on :$SERVER_PORT"

else
  phase "Skipping setup (--skip-setup)"
  cd "$WORK_DIR"
fi

# ── Phase 6: Benchmark ─────────────────────────────────────────────────
phase "Phase 6: Running benchmark queries"

BASE_URL="http://127.0.0.1:$SERVER_PORT"

# Define queries
QUERIES=(
  'How do I configure the embedding model and what dimensions does it use?'
  'I'\''m adding a new storage backend. Walk me through what traits I need to implement and how the existing backends do it.'
  'A search request is returning unexpected results. Trace the full path from the REST API endpoint through the RAG pipeline to the storage layer.'
  'How does the agent session lifecycle interact with the staging system and conflict resolution during merge?'
)
EXPECTED_WINNERS=("vector" "graph" "graph" "graph")

run_query() {
  local query="$1" expand_graph="$2"
  curl -s -X POST "$BASE_URL/v1/context" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg q "$query" \
      --arg s "$SCOPE_ID" \
      --argjson eg "$expand_graph" \
      '{query: $q, scope_id: $s, limit: 10, expand_graph: $eg}'
    )"
}

for i in "${!QUERIES[@]}"; do
  n=$((i + 1))
  query="${QUERIES[$i]}"
  echo ""
  echo "Query $n: ${query:0:70}..."

  echo "  → vector-only..."
  run_query "$query" false | jq '.' > "$RESULTS_DIR/query-${n}-vector.json"

  echo "  → graph-expanded..."
  run_query "$query" true  | jq '.' > "$RESULTS_DIR/query-${n}-graph.json"

  echo "  → saved"
done

# ── Build summary.json ──────────────────────────────────────────────────
phase "Phase 7: Building summary"

GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# Extract per-query metrics and build summary
SUMMARY_QUERIES="[]"
for i in "${!QUERIES[@]}"; do
  n=$((i + 1))
  query="${QUERIES[$i]}"
  expected="${EXPECTED_WINNERS[$i]}"

  v_file="$RESULTS_DIR/query-${n}-vector.json"
  g_file="$RESULTS_DIR/query-${n}-graph.json"

  entry=$(jq -n \
    --argjson id "$n" \
    --arg query "$query" \
    --arg expected "$expected" \
    --argjson vf "$(cat "$v_file")" \
    --argjson gf "$(cat "$g_file")" \
    '{
      id: $id,
      query: $query,
      expected_winner: $expected,
      vector: {
        sources_count:  ($vf.sources | length),
        top_score:      ($vf.sources[0].score // 0),
        top_source_file:($vf.sources[0].entry.metadata.source_file // "none"),
        source_files:   [($vf.sources[].entry.metadata.source_file // "none")],
        latency_ms:     ($vf.trace.retrieval.latency_ms // 0),
        vector_results: ($vf.trace.retrieval.vector_results // 0),
        graph_expanded: ($vf.trace.retrieval.graph_expanded // 0)
      },
      graph: {
        sources_count:  ($gf.sources | length),
        top_score:      ($gf.sources[0].score // 0),
        top_source_file:($gf.sources[0].entry.metadata.source_file // "none"),
        source_files:   [($gf.sources[].entry.metadata.source_file // "none")],
        latency_ms:     ($gf.trace.retrieval.latency_ms // 0),
        vector_results: ($gf.trace.retrieval.vector_results // 0),
        graph_expanded: ($gf.trace.retrieval.graph_expanded // 0)
      }
    }')

  SUMMARY_QUERIES=$(echo "$SUMMARY_QUERIES" | jq --argjson e "$entry" '. + [$e]')
done

# Write metadata.json
jq -n \
  --arg run_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg version "0.3.0" \
  --arg git_sha "$GIT_SHA" \
  '{
    run_at: $run_at,
    corvia_version: $version,
    git_sha: $git_sha,
    storage_backend: "lite",
    inference_provider: "corvia",
    inference_model: "nomic-embed-text-v1.5",
    graph_alpha: 0.3,
    graph_depth: 2,
    scope_id: "corvia-introspect"
  }' > "$RESULTS_DIR/metadata.json"

# Write summary.json
jq -n \
  --arg run_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson queries "$SUMMARY_QUERIES" \
  '{run_at: $run_at, queries: $queries}' > "$RESULTS_DIR/summary.json"

echo "metadata.json written"
echo "summary.json written"
echo ""
echo "── Done ──"
echo "Results: $RESULTS_DIR"
echo "Files:"
ls -1 "$RESULTS_DIR"
