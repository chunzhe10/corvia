# StoreClient Trait Abstraction — Future Work

> **Status:** Design decision, not yet scheduled. Beyond M4 scope.
> **Date:** 2026-03-08

## Problem

The CLI and server both call `QueryableStore` directly, which opens the underlying database.
With LiteStore (Redb), only one process can hold the write lock. This creates conflicts when
the CLI and server need to run concurrently (the primary dogfooding pain point).

Phase 1 (v0.3.4) introduced CLI server-aware routing as a workaround: the CLI detects a
running server via health check and routes read-only operations through REST. This works
but introduces two code paths per command.

## Proposed Abstraction

Introduce a `StoreClient` trait that implements the same `QueryableStore` interface but
routes operations through the server when one is available.

```
CLI ───→ StoreClient trait ───→ DirectStore (opens DB directly)
                            ├─→ RemoteStore (talks to server via REST/gRPC/UDS)
Server ───→ DirectStore (owns the DB)
```

The CLI doesn't know or care whether it's talking to Redb, SurrealDB, PostgreSQL, or a
remote server — it just calls `store.search()`.

## Deployment Modes

| Mode | StoreClient impl | Transport | Process management |
|------|------------------|-----------|-------------------|
| **Devcontainer** | RemoteStore | TCP (localhost:8020) | Server pre-started by post-start.sh |
| **Solo dev (macOS/Linux)** | RemoteStore | Unix domain socket (.corvia/server.sock) | Auto-start server on first CLI command, idle timeout |
| **Team / shared server** | RemoteStore | TCP/gRPC (remote host) | External process management (systemd, k8s) |
| **CI/CD** | DirectStore | None (in-process) | Single process, no server needed |

## Unix Domain Socket Notes

UDS is IPC via a file path instead of TCP host:port. ~4x throughput, ~10x lower latency
than TCP for local communication. No port conflicts between workspaces (each gets its own
socket file). Supported on Linux, macOS, and Windows 10+.

Key advantage: detection is `stat .corvia/server.sock` (instant) vs TCP health check (500ms timeout).

## Why Not Now

- Current users are in devcontainers where the server is pre-started
- Phase 1 routing covers the immediate lock conflict for read operations
- The abstraction should be designed alongside M4 (observability) and M5 (VS Code extension)
  which will also need to decide how clients connect to corvia
- Building for deployment modes that don't have users yet adds complexity without value

## When to Build

- When corvia supports non-devcontainer workflows (native macOS/Linux install)
- When team/multi-machine use cases emerge
- When the VS Code extension needs a connection strategy
- If LiteStore is replaced or Redb's locking model changes

## Relationship to Existing Stores

LiteStore, SurrealStore, and PostgresStore all implement `QueryableStore`. The `StoreClient`
abstraction sits above them — it's about *how the client connects*, not *what database backs it*.
SurrealDB and PostgreSQL already handle concurrent access natively, so the lock conflict is
LiteStore-specific. But the `StoreClient` pattern benefits all backends by standardizing the
client connection model.
