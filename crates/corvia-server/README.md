# corvia-server

REST API and MCP JSON-RPC server for the Corvia knowledge system.

Part of the [Corvia](https://github.com/corvia/corvia) workspace. This crate exposes
the kernel's storage, graph, temporal, reasoning, and agent coordination capabilities
over HTTP. Typically started via `corvia serve` rather than used as a library.

## Protocols

- **REST** — Axum HTTP API on port `8020` with endpoints for knowledge ingestion, semantic search, graph queries, temporal history, reasoning, and agent coordination
- **MCP** — JSON-RPC 2.0 Model Context Protocol at `POST /mcp` for AI agent tool use

## License

AGPL-3.0-only — see [LICENSE](../../LICENSE) for details.
