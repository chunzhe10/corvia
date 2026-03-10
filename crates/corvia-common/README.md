# corvia-common

Shared types, configuration, errors, and namespace model for the Corvia knowledge system.

Part of the [Corvia](https://github.com/corvia/corvia) workspace. This crate provides
the foundational types that every other Corvia crate depends on. It carries no runtime
dependencies on storage or inference, making it suitable for external adapters.

## Modules

- **types** — `KnowledgeEntry`, `SearchResult`, `GraphEdge`, and other core data structures
- **config** — `CorviaConfig` loaded from `corvia.toml`, storage, embedding, and telemetry settings (`TelemetryConfig`)
- **namespace** — Five-segment namespace model (`org:scope:workstream:source:version`)
- **errors** — Unified error types and `Result` alias
- **events** — Domain events for inter-subsystem communication
- **agent_types** — Agent identity, session state, and coordination types

## License

AGPL-3.0-only — see [LICENSE](../../LICENSE) for details.
