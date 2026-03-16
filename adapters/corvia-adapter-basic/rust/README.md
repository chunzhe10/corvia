# corvia-adapter-basic

Basic filesystem ingestion adapter for corvia. Reads files from a directory and
produces text chunks for the knowledge store without language-specific parsing.

## How it works

1. Walks the target directory
2. Reads each file as plain text
3. Splits into chunks based on configured token limits
4. Outputs chunks via JSONL IPC protocol to the kernel

## When to use

Use this adapter for content that doesn't benefit from AST-aware parsing:
documentation directories, plain text files, configuration files, or any
file type not covered by specialized adapters like `corvia-adapter-git`.

## Usage

```bash
# Direct invocation (for testing)
echo '{"path": "/docs"}' | corvia-adapter-basic

# Normal usage — the kernel invokes it automatically based on adapter discovery
corvia ingest /path/to/docs
```

## License

[AGPL-3.0-only](../../../../LICENSE)
