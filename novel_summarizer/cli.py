from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from loguru import logger

from novel_summarizer.config import load_config
from novel_summarizer.config.loader import masked_env_snapshot
from novel_summarizer.embeddings.service import prepare_retrieval_assets
from novel_summarizer.export.markdown import export_book_markdown
from novel_summarizer.ingest.service import ingest_book
from novel_summarizer.storyteller.service import storytell_book
from novel_summarizer.summarize.service import summarize_book
from novel_summarizer.storage.db import init_db_service, shutdown_db_service
from novel_summarizer.utils.logging import setup_logging

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novel-summarizer")
    parser.add_argument("--config", type=Path, default=None, help="Path to custom config YAML")
    parser.add_argument("--profile", type=str, default=None, help="Config profile name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="Validate and print effective config")

    ingest_parser = subparsers.add_parser("ingest", help="Parse chapters, split chunks, store in SQLite")
    ingest_parser.add_argument("--input", type=Path, required=True, help="Path to novel text file")
    ingest_parser.add_argument("--title", type=str, default=None, help="Book title")
    ingest_parser.add_argument("--author", type=str, default=None, help="Book author")
    ingest_parser.add_argument("--chapter-regex", type=str, default=None, help="Override chapter regex")

    summarize_parser = subparsers.add_parser("summarize", help="LEGACY: Generate v1 map-reduce summaries")
    summarize_parser.add_argument("--book-id", type=int, required=True, help="Book id to summarize")
    summarize_parser.add_argument("--no-export", action="store_true", help="Skip markdown export")

    storytell_parser = subparsers.add_parser("storytell", help="Run chapter-by-chapter storyteller rewrite")
    storytell_parser.add_argument("--book-id", type=int, required=True, help="Book id to process")
    storytell_parser.add_argument("--from-chapter", type=int, default=None, help="Start chapter idx (inclusive)")
    storytell_parser.add_argument("--to-chapter", type=int, default=None, help="End chapter idx (inclusive)")

    export_parser = subparsers.add_parser("export", help="Export markdown outputs from storyteller or legacy data")
    export_parser.add_argument("--book-id", type=int, required=True, help="Book id to export")
    export_parser.add_argument(
        "--mode",
        choices=["storyteller", "legacy", "auto"],
        default="storyteller",
        help="Export mode: storyteller (default), legacy, or auto fallback",
    )

    embed_parser = subparsers.add_parser("embed", help="Build LanceDB vector index for chunks")
    embed_parser.add_argument("--book-id", type=int, required=True, help="Book id to embed")
    embed_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    run_parser = subparsers.add_parser("run", help="Run pipeline (ingest -> storytell -> export)")
    run_parser.add_argument("--book-id", type=int, default=None, help="Existing book id (skip ingest)")
    run_parser.add_argument("--input", type=Path, default=None, help="Path to novel text file (required if --book-id absent)")
    run_parser.add_argument("--title", type=str, default=None, help="Book title (used during ingest)")
    run_parser.add_argument("--author", type=str, default=None, help="Book author (used during ingest)")
    run_parser.add_argument("--chapter-regex", type=str, default=None, help="Override chapter regex during ingest")
    run_parser.add_argument("--from-chapter", type=int, default=None, help="Start chapter idx (inclusive)")
    run_parser.add_argument("--to-chapter", type=int, default=None, help="End chapter idx (inclusive)")
    run_parser.add_argument("--no-export", action="store_true", help="Skip markdown export")

    return parser


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    app_overrides: dict[str, Any] = {}
    if args.output_dir:
        app_overrides["output_dir"] = str(args.output_dir)
    if args.data_dir:
        app_overrides["data_dir"] = str(args.data_dir)
    if app_overrides:
        overrides["app"] = app_overrides
    return overrides


def _print_config(config) -> None:
    env_snapshot = masked_env_snapshot(config)
    console.print(Panel(Pretty(config.model_dump(mode="json")), title="Effective Config"))
    console.print(Panel(Pretty(env_snapshot), title="Env Snapshot"))


async def _main_async() -> None:
    def _coerce_export_mode(value: str) -> Literal["storyteller", "legacy", "auto"]:
        if value not in {"storyteller", "legacy", "auto"}:
            raise ValueError(f"Unsupported export mode: {value}")
        return value

    parser = _build_parser()
    args = parser.parse_args()

    overrides = _build_overrides(args)
    config = load_config(
        config_path=args.config,
        profile=args.profile,
        overrides=overrides,
    )

    setup_logging(config.app.log_level)
    logger.info("Loaded configuration")

    await init_db_service(config.storage.sqlite_path)

    try:
        if args.command == "config":
            _print_config(config)
            return

        if args.command == "ingest":
            stats = await ingest_book(
                input_path=args.input,
                config=config,
                title=args.title,
                author=args.author,
                chapter_regex_override=args.chapter_regex,
            )
            table = Table(title="Ingest Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(stats.book_id))
            table.add_row("Book Hash", stats.book_hash)
            table.add_row("Chapters (total/new)", f"{stats.chapters_total}/{stats.chapters_inserted}")
            table.add_row("Chunks (total/new)", f"{stats.chunks_total}/{stats.chunks_inserted}")
            console.print(table)
            return

        if args.command == "summarize":
            console.print(
                Panel(
                    "Legacy command: `summarize` belongs to v1 map-reduce pipeline. "
                    "Prefer `storytell`/`run` for the current default workflow.",
                    title="Legacy Notice",
                )
            )
            stats = await summarize_book(book_id=args.book_id, config=config)
            table = Table(title="Summarize Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(stats.book_id))
            table.add_row("Chapters (total/new)", f"{stats.chapters_total}/{stats.chapters_new}")
            table.add_row("Chunks (total/new)", f"{stats.chunks_total}/{stats.chunks_new}")
            table.add_row("Book summary new", str(stats.book_summary_new))
            table.add_row("Characters new", str(stats.characters_new))
            table.add_row("Timeline new", str(stats.timeline_new))
            table.add_row("Story new", str(stats.story_new))
            console.print(table)

            if not args.no_export:
                export_result = await export_book_markdown(book_id=args.book_id, config=config, mode="legacy")
                console.print(
                    Panel(
                        f"Exported to {export_result.output_dir}",
                        title="Export",
                    )
                )
            return

        if args.command == "storytell":
            stats = await storytell_book(
                book_id=args.book_id,
                config=config,
                from_chapter=args.from_chapter,
                to_chapter=args.to_chapter,
            )
            table = Table(title="Storytell Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(stats.book_id))
            table.add_row("Chapters total", str(stats.chapters_total))
            table.add_row("Chapters processed", str(stats.chapters_processed))
            table.add_row("Chapters skipped", str(stats.chapters_skipped))
            table.add_row("LLM calls (est)", str(stats.llm_calls_estimated))
            table.add_row("Refine LLM calls", str(stats.refine_llm_calls_estimated))
            table.add_row("Cache hits", str(stats.llm_cache_hits))
            table.add_row("Cache misses", str(stats.llm_cache_misses))
            table.add_row("Input tokens (est)", str(stats.input_tokens_estimated))
            table.add_row("Output tokens (est)", str(stats.output_tokens_estimated))
            table.add_row(
                "Refine tokens in/out (est)",
                f"{stats.refine_input_tokens_estimated}/{stats.refine_output_tokens_estimated}",
            )
            table.add_row("Consistency warnings", str(stats.consistency_warnings))
            table.add_row("Consistency actions", str(stats.consistency_actions))
            table.add_row("Evidence supported", str(stats.evidence_supported_claims))
            table.add_row("Evidence unsupported", str(stats.evidence_unsupported_claims))
            table.add_row("Runtime (s)", f"{stats.runtime_seconds:.2f}")
            console.print(table)
            return

        if args.command == "export":
            export_result = await export_book_markdown(
                book_id=args.book_id,
                config=config,
                mode=_coerce_export_mode(args.mode),
            )
            console.print(
                Panel(
                    f"Exported to {export_result.output_dir}",
                    title="Export",
                )
            )
            return

        if args.command == "embed":
            stats = await prepare_retrieval_assets(book_id=args.book_id, config=config, batch_size=args.batch_size)
            table = Table(title="Embedding Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(stats.book_id))
            table.add_row("Chunk vectors embedded", str(stats.chunk_vectors_embedded))
            table.add_row("Narration vectors embedded", str(stats.narration_vectors_embedded))
            table.add_row("Chunk FTS rows", str(stats.chunk_fts_rows))
            table.add_row("Narration FTS rows", str(stats.narration_fts_rows))
            console.print(table)
            return

        if args.command == "run":
            if args.book_id is None and args.input is None:
                raise ValueError("run requires --book-id or --input")

            book_id = args.book_id
            ingest_stats = None

            if args.input is not None:
                ingest_stats = await ingest_book(
                    input_path=args.input,
                    config=config,
                    title=args.title,
                    author=args.author,
                    chapter_regex_override=args.chapter_regex,
                )
                book_id = ingest_stats.book_id

            if book_id is None:
                raise ValueError("Unable to determine book_id for run pipeline")

            storytell_stats = await storytell_book(
                book_id=book_id,
                config=config,
                from_chapter=args.from_chapter,
                to_chapter=args.to_chapter,
            )

            export_dir = "(skipped)"
            if not args.no_export:
                export_result = await export_book_markdown(book_id=book_id, config=config, mode="storyteller")
                export_dir = str(export_result.output_dir)

            table = Table(title="Run Pipeline Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(book_id))
            if ingest_stats is not None:
                table.add_row("Ingest chapters (total/new)", f"{ingest_stats.chapters_total}/{ingest_stats.chapters_inserted}")
                table.add_row("Ingest chunks (total/new)", f"{ingest_stats.chunks_total}/{ingest_stats.chunks_inserted}")
            table.add_row("Storytell chapters (total)", str(storytell_stats.chapters_total))
            table.add_row("Storytell chapters (processed)", str(storytell_stats.chapters_processed))
            table.add_row("Storytell chapters (skipped)", str(storytell_stats.chapters_skipped))
            table.add_row("Storytell LLM calls (est)", str(storytell_stats.llm_calls_estimated))
            table.add_row("Storytell refine calls", str(storytell_stats.refine_llm_calls_estimated))
            table.add_row("Storytell cache hit/miss", f"{storytell_stats.llm_cache_hits}/{storytell_stats.llm_cache_misses}")
            table.add_row(
                "Storytell tokens in/out (est)",
                f"{storytell_stats.input_tokens_estimated}/{storytell_stats.output_tokens_estimated}",
            )
            table.add_row(
                "Storytell refine tokens in/out",
                f"{storytell_stats.refine_input_tokens_estimated}/{storytell_stats.refine_output_tokens_estimated}",
            )
            table.add_row("Consistency warn/action", f"{storytell_stats.consistency_warnings}/{storytell_stats.consistency_actions}")
            table.add_row(
                "Evidence support/unsupport",
                f"{storytell_stats.evidence_supported_claims}/{storytell_stats.evidence_unsupported_claims}",
            )
            table.add_row("Storytell runtime (s)", f"{storytell_stats.runtime_seconds:.2f}")
            table.add_row("Export", export_dir)
            console.print(table)
            return

        console.print(Panel("M0 skeleton ready. Pipeline nodes will be implemented next.", title="Status"))
    finally:
        await shutdown_db_service()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
