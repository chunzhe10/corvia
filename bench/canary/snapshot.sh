#!/usr/bin/env bash
# snapshot.sh — emit a deterministic corpus snapshot hash for the canary query set.
#
# Hash is sha256 over sorted lines of "<entry_id>\t<sha256_of_file_bytes>".
# Filesystem-order-independent, mtime-independent.
# Invoke from anywhere; entries path is resolved relative to the workspace root.
set -euo pipefail

WORKSPACE_ROOT="${CORVIA_WORKSPACE:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
ENTRIES_DIR="${WORKSPACE_ROOT}/.corvia/entries"

if [[ ! -d "${ENTRIES_DIR}" ]]; then
  echo "error: entries dir not found: ${ENTRIES_DIR}" >&2
  echo "hint: set CORVIA_WORKSPACE to the workspace root (directory containing .corvia/)" >&2
  exit 1
fi

# shellcheck disable=SC2012
find "${ENTRIES_DIR}" -maxdepth 1 -name '*.md' -type f \
  | sort \
  | while read -r f; do
      eid="$(basename "$f" .md)"
      h="$(sha256sum "$f" | awk '{print $1}')"
      printf '%s\t%s\n' "${eid}" "${h}"
    done \
  | sha256sum \
  | awk '{print "sha256:" $1}'
