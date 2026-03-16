# corvia-adapter-git

Git repository and source code ingestion adapter for corvia. Parses source files
using tree-sitter to produce semantic chunks for the knowledge store.

## Supported Languages

- Rust, Python, JavaScript, TypeScript (via tree-sitter grammars)
- Markdown, TOML, JSON, YAML (via format-aware chunking)

## How it works

1. Walks the git repository (respects `.gitignore`)
2. Detects file language from extension
3. Parses supported languages with tree-sitter into AST nodes
4. Extracts semantic chunks (functions, structs, classes, modules)
5. Outputs chunks via JSONL IPC protocol to the kernel

## Usage

The adapter is discovered automatically by the kernel via PATH scan:

```bash
# Direct invocation (for testing)
echo '{"path": "/repo"}' | corvia-adapter-git

# Normal usage — the kernel invokes it automatically
corvia ingest /path/to/repo
```

## License

[AGPL-3.0-only](../../../../LICENSE)
