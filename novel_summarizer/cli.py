from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from loguru import logger

from novel_summarizer.config import load_config
from novel_summarizer.config.loader import masked_env_snapshot
from novel_summarizer.embeddings.service import embed_book_chunks
from novel_summarizer.export.markdown import export_book_markdown
from novel_summarizer.ingest.service import ingest_book
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

    summarize_parser = subparsers.add_parser("summarize", help="Generate chunk and chapter summaries")
    summarize_parser.add_argument("--book-id", type=int, required=True, help="Book id to summarize")
    summarize_parser.add_argument("--no-export", action="store_true", help="Skip markdown export")

    export_parser = subparsers.add_parser("export", help="Export markdown outputs from stored summaries")
    export_parser.add_argument("--book-id", type=int, required=True, help="Book id to export")

    embed_parser = subparsers.add_parser("embed", help="Build LanceDB vector index for chunks")
    embed_parser.add_argument("--book-id", type=int, required=True, help="Book id to embed")
    embed_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    subparsers.add_parser("run", help="Run pipeline (M0 stub)")

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
    env_snapshot = masked_env_snapshot()
    console.print(Panel(Pretty(config.model_dump(mode="json")), title="Effective Config"))
    console.print(Panel(Pretty(env_snapshot), title="Env Snapshot"))


async def _main_async() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    overrides = _build_overrides(args)
    config = load_config(
        config_path=args.config,
        profile=args.profile,
        overrides=overrides,
        require_api_key=args.command not in {"config", "ingest"},
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
                export_result = await export_book_markdown(book_id=args.book_id, config=config)
                console.print(
                    Panel(
                        f"Exported to {export_result.output_dir}",
                        title="Export",
                    )
                )
            return

        if args.command == "export":
            export_result = await export_book_markdown(book_id=args.book_id, config=config)
            console.print(
                Panel(
                    f"Exported to {export_result.output_dir}",
                    title="Export",
                )
            )
            return

        if args.command == "embed":
            stats = await embed_book_chunks(book_id=args.book_id, config=config, batch_size=args.batch_size)
            table = Table(title="Embedding Summary", show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Book ID", str(stats.book_id))
            table.add_row("Chunks total", str(stats.chunks_total))
            table.add_row("Chunks embedded", str(stats.chunks_embedded))
            table.add_row("Chunks skipped", str(stats.chunks_skipped))
            console.print(table)
            return

        console.print(Panel("M0 skeleton ready. Pipeline nodes will be implemented next.", title="Status"))
    finally:
        await shutdown_db_service()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
