from __future__ import annotations

import pytest

from novel_summarizer.storyteller.step_utils import (
    align_from_chapter,
    align_to_chapter,
    iter_step_ranges,
    step_end_for_start,
    step_start_for_chapter,
)


def test_step_start_for_chapter() -> None:
    assert step_start_for_chapter(chapter_idx=1, step_size=5) == 1
    assert step_start_for_chapter(chapter_idx=5, step_size=5) == 1
    assert step_start_for_chapter(chapter_idx=6, step_size=5) == 6
    assert step_start_for_chapter(chapter_idx=7, step_size=5) == 6


def test_step_end_for_start_clamps_to_max() -> None:
    assert step_end_for_start(step_start=1, step_size=5, max_chapter_idx=12) == 5
    assert step_end_for_start(step_start=11, step_size=5, max_chapter_idx=12) == 12


def test_align_from_chapter_aligns_down() -> None:
    assert align_from_chapter(from_chapter=7, step_size=5) == 6


def test_align_to_chapter_aligns_up() -> None:
    assert align_to_chapter(to_chapter=7, step_size=5, max_chapter_idx=40) == 10


def test_iter_step_ranges_builds_ranges() -> None:
    assert iter_step_ranges(start_chapter=1, end_chapter=12, step_size=5) == [(1, 5), (6, 10), (11, 12)]


def test_iter_step_ranges_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        step_start_for_chapter(chapter_idx=0, step_size=5)
    with pytest.raises(ValueError):
        step_start_for_chapter(chapter_idx=1, step_size=0)
