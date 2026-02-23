# Repository Guidelines

This document defines repository conventions for both humans and coding agents (Copilot, bots, CI helpers). It is intentionally:

- **English-only** (for consistent collaboration and tooling)
- **Unambiguous** (explicit "current vs planned" boundaries)

If something here conflicts with the code or tests, **the code/tests win** and this file should be updated.

## Architecture Overview

### Current status (important)

The repository is in a **transition** from the v1 "Map-Reduce summarization" pipeline to the v2 **Storyteller Rewrite** architecture.

- **What exists today**: v1 pipeline under `novel_summarizer/summarize/` plus ingest/export/storage/LLM utilities.
- **What is the target**: v2 chapter-by-chapter rewrite workflow orchestrated by LangGraph, with "hard" world state in SQLite and "soft" semantic memory in LanceDB.

The authoritative v2 design (data model, nodes, CLI, milestones) lives in [`PLAN.md`](PLAN.md).

### v2 design in one paragraph

The v2 architecture is a **rewrite** system, not a compression system: it rewrites each chapter into a "storyteller-style narration" while staying consistent with a persisted world state.

Three engines are separated by responsibility:

- **SQLite (World State / Hard Logic)**: the single source of truth for facts that must not drift (character state, item ownership, event timeline).
- **LanceDB (Semantic Memory / Soft Recall)**: vectorized memory for long-range callbacks (source chunks + generated narrations), used to "awaken" relevant context.
- **LangGraph (Workflow Orchestration)**: chapter loop orchestration as a directed graph of nodes (planned: 6 nodes).

## Project Structure & Module Organization

### Implemented (present in the repo)

- `main.py`: thin wrapper for CLI entry (keep minimal).
- `novel_summarizer/cli.py`: CLI command definitions and orchestration.
- `novel_summarizer/config/`: configuration schema + loader (YAML + ENV merge).
- `novel_summarizer/ingest/`: text normalization, chapter parsing, chunk splitting.
- `novel_summarizer/llm/`: model clients and caching.
- `novel_summarizer/embeddings/`: vector index service and retrieval helpers.
- `novel_summarizer/storage/`: SQLAlchemy models/repositories plus vector DB helpers.
  - `books/`, `chapters/`, `chunks/`, `summaries/`: v1 tables and CRUD.
- `novel_summarizer/export/`: Markdown export.
- `novel_summarizer/utils/`: shared utilities (logging, etc.).
- `novel_summarizer/summarize/`: **v1** Map-Reduce summarization pipeline (legacy; still used until v2 lands).

### Planned for v2 (may not exist yet)

The following modules/paths are part of the v2 plan and may be missing until the migration milestones are completed:

- `novel_summarizer/storyteller/`: core v2 chapter-by-chapter rewrite workflow (LangGraph)
  - `graph.py`: `StateGraph` definition and compilation
  - `state.py`: `StorytellerState` `TypedDict`
  - `nodes/`: planned nodes: `entity_extract`, `state_lookup`, `memory_retrieve`, `storyteller_generate`, `state_update`, `memory_commit`
  - `prompts/`: prompt templates split by function (entity, narration, state mutation)
  - `service.py`: outer chapter loop orchestrator
- `novel_summarizer/storage/narrations/`: per-chapter narrations (v2 output) CRUD
- `novel_summarizer/storage/world_state/`: characters/items/plot events/world facts (v2 hard-logic tables) CRUD

When implementing v2, prefer adding new v2 tables/modules rather than mutating v1 tables in-place, unless migration is explicitly planned.

## Key Design Principles

- **Hard logic vs soft memory**
  - SQLite stores hard constraints (facts). Treat it as authoritative.
  - Vector retrieval is assistive recall. Treat it as a hint, not a fact.

- **Chapter-by-chapter processing**
  - The rewrite workflow is naturally sequential because chapter $N$ depends on world state produced by chapters $1..N-1$.
  - Concurrency is mainly for embedding/vector indexing work; keep the chapter loop deterministic.

- **Idempotency and caching**
  - All expensive LLM steps must be cached and reproducible.
  - Cache keys should include **at least** `(prompt_version, model, input_hash)` and any other parameter that changes outputs (e.g., temperature).
  - Re-running the tool should not reprocess completed work unless inputs or prompt versions change.

- **Temperature tiering**
  - Extraction tasks (NER/state mutation) should use low temperature (e.g., 0.1).
  - Narration generation can use a higher temperature (e.g., 0.4-0.6) but must remain constrained by world state.

- **Testable graph nodes**
  - Each workflow node should be independently testable.
  - Nodes communicate via a typed state object (`TypedDict`), enabling clean mocks and deterministic tests.

## Build, Test, and Development Commands

- Package management is **uv-only** (do not introduce pip/poetry commands in docs).
- `uv sync` installs locked dependencies from `uv.lock`.
- `uv run pytest` runs the test suite.
- `uv run python -m ruff check .` runs lint checks.
- `uv run python -m black .` formats code (line length: 120).

## Coding Style & Naming Conventions

- Python **3.12+**.
- Format with Black (line length: 120) and lint with Ruff.
- Naming:
  - functions/variables: `snake_case`
  - classes: `PascalCase`
  - constants: `SCREAMING_SNAKE_CASE`
- Comments must be **English** and brief.
- Keep async boundaries clear:
  - async I/O in services/repos
  - CPU-heavy code stays sync unless proven necessary
- Avoid side effects at import time; initialization belongs in explicit entry points.
- Centralize data access in `novel_summarizer/storage/` and use `SQLAlchemyRepo` (avoid ad-hoc SQL in services).
- Keep configuration access centralized in `novel_summarizer/config/loader.py`; do not read env vars directly outside the config layer.
- Keep the CLI thin; put business logic under `novel_summarizer/` modules.
- Prefer pure functions for parsing/splitting to simplify tests.

## Testing Guidelines

- Place tests under `tests/` and name files `test_*.py`.
- Use `pytest` fixtures (e.g., `tmp_path`) for filesystem I/O.
- Unit tests must not call external services (LLM, embeddings, vector DB). Mock those clients.
- Prefer small, deterministic tests for parsing, hashing, and rendering.
- For future LangGraph nodes, test each node independently with mocked dependencies and fixed input state.
- For DB CRUD tests, prefer in-memory SQLite (e.g., `sqlite+aiosqlite://`) for fast, isolated runs.
