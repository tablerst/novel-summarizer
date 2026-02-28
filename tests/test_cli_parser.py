from __future__ import annotations

from novel_summarizer.cli import _build_overrides, _build_parser


def test_export_mode_defaults_to_storyteller() -> None:
    parser = _build_parser()
    args = parser.parse_args(["export", "--book-id", "1"])

    assert args.mode == "storyteller"


def test_export_mode_accepts_legacy() -> None:
    parser = _build_parser()
    args = parser.parse_args(["export", "--book-id", "1", "--mode", "legacy"])

    assert args.mode == "legacy"


def test_export_mode_accepts_auto() -> None:
    parser = _build_parser()
    args = parser.parse_args(["export", "--book-id", "1", "--mode", "auto"])

    assert args.mode == "auto"


def test_summarize_command_still_available_as_legacy() -> None:
    parser = _build_parser()
    args = parser.parse_args(["summarize", "--book-id", "1"])

    assert args.command == "summarize"
    assert args.book_id == 1


def test_storytell_accepts_step_size_and_builds_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["storytell", "--book-id", "1", "--step-size", "5"])

    overrides = _build_overrides(args)

    assert overrides["storyteller"]["step_size"] == 5
