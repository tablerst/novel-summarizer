# Repository Guidelines

## Project Structure & Module Organization
Project entry points and core modules:

- `main.py`: thin wrapper for CLI entry (kept minimal).
- `novel_summarizer/cli.py`: CLI command definitions and orchestration.
- `novel_summarizer/config/`: config schema + loader (YAML + ENV merge).
- `novel_summarizer/ingest/`: text normalization, chapter parsing, chunk splitting.
- `novel_summarizer/llm/`: model clients, prompts, caching.
- `novel_summarizer/embeddings/`: vector index service and retrieval.
- `novel_summarizer/storage/`: database models, repositories, and vector DB helpers.
- `novel_summarizer/export/`: Markdown output rendering.
- `novel_summarizer/utils/`: shared utilities (logging, concurrency, etc.).

Guidelines:

- Put shared data access in `storage/` and call via `SQLAlchemyRepo` rather than raw SQL in services.
- Keep config access centralized in `config/loader.py`; avoid reading env directly outside config layer.
- Keep CLI thin; put business logic in services/modules under `novel_summarizer/`.
- Prefer pure functions for parsing/splitting to simplify tests.

## Build, Test, and Development Commands
- Package management is **uv-only**.
- `uv sync` installs locked dependencies (from `uv.lock`).
- `uv run pytest` runs the test suite.
- `uv run python -m ruff check .` runs lint checks (if configured).
- `uv run python -m black .` formats code (line length 120).

## Coding Style & Naming Conventions
- Python 3.12+; format with Black (line length 120) and lint with Ruff (Google-style docstrings).
- Use snake_case for functions and variables, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants; prefer type hints even though annotations are not enforced.
- Write comments in English and keep them brief and purposeful.
- Keep async boundaries clear: async I/O in services/repos; keep CPU-heavy ops sync.
- Avoid side effects at import time; keep initialization in explicit entry points.

## Testing Guidelines
- Place tests under `tests/` and name files `test_*.py`.
- Use `pytest` fixtures like `tmp_path` for filesystem I/O.
- Do not call external services in unit tests; mock LLM/embedding clients.
- Prefer small, deterministic tests for parsing, hashing, and rendering helpers.
