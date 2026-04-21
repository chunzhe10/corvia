# Third-Party Licenses

corvia is distributed under **AGPL-3.0-or-later**. This file lists the third-party components that ship with or are used by corvia and its companion tooling.

## Runtime Rust dependencies

Collected from the `Cargo.lock` manifest; all are compatible with AGPL-3.0 (permissive licenses or AGPL-compatible copyleft). Full text of each license is preserved in the corresponding crate source.

A machine-generated list can be produced with `cargo license --manifest-path Cargo.toml` if precise version + license mapping is needed.

## Python eval-harness dependencies (`bench/ragas/`)

The Ragas synthetic testset generator at `bench/ragas/generate.py` is an **external** Python tool. It is invoked out-of-process at eval time; it is not linked into any Rust crate, and it is not required to build or run corvia itself.

| Package | Version | License | Upstream |
|---|---|---|---|
| [Ragas](https://github.com/explodinggradients/ragas) | 0.4.3 | Apache-2.0 | `pip install ragas` |
| [LangChain](https://github.com/langchain-ai/langchain) | 1.2.15 | MIT | `pip install langchain` |
| [LangChain Core](https://github.com/langchain-ai/langchain) | 1.3.0 | MIT | `pip install langchain-core` |
| [LangChain Community](https://github.com/langchain-ai/langchain) | 0.4.1 | MIT | `pip install langchain-community` |
| [langchain-google-genai](https://github.com/langchain-ai/langchain-google) | 3.0.5 | MIT | `pip install langchain-google-genai` |
| [langchain-openai](https://github.com/langchain-ai/langchain) *(optional)* | — | MIT | `pip install langchain-openai` |
| [langchain-anthropic](https://github.com/langchain-ai/langchain) *(optional)* | — | MIT | `pip install langchain-anthropic` |
| [pydantic](https://github.com/pydantic/pydantic) | (pulled by ragas) | MIT | `pip install pydantic` |
| [google-generativeai](https://github.com/google/generative-ai-python) | (pulled by langchain-google-genai) | Apache-2.0 | `pip install google-generativeai` |
| [datasets](https://github.com/huggingface/datasets) | (pulled by ragas) | Apache-2.0 | `pip install datasets` |
| [nest-asyncio](https://github.com/erdewit/nest_asyncio) | (pulled by ragas) | BSD-2-Clause | `pip install nest-asyncio` |
| [pytest](https://github.com/pytest-dev/pytest) | 8.3.5 | MIT | test-only |

### Apache-2.0 NOTICE

Apache-2.0-licensed packages above (Ragas, google-generativeai, datasets) require preservation of their NOTICE file contents where present. See each project's upstream repository for the canonical NOTICE.

### Compatibility

- Apache-2.0 is one-way compatible with AGPL-3.0-or-later (Apache → AGPL) per the FSF license compatibility table.
- MIT is compatible with AGPL-3.0-or-later.
- BSD-2-Clause is compatible with AGPL-3.0-or-later.

Because `bench/ragas/` ships no binary artifact and is not linked into corvia's build, distribution obligations for these packages are limited to this attribution file and to preserving LICENSE / NOTICE files for any package whose source is vendored (none are, at present — all are pulled from PyPI at eval time).

## Reporting issues

If you believe a license attribution is missing or incorrect, please open an issue at https://github.com/chunzhe10/corvia/issues.
