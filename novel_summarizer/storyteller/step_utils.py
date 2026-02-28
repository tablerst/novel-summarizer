from __future__ import annotations


def step_start_for_chapter(*, chapter_idx: int, step_size: int) -> int:
    """Returns the 1-based start chapter idx of the step containing chapter_idx."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if chapter_idx <= 0:
        raise ValueError("chapter_idx must be positive")
    return ((chapter_idx - 1) // step_size) * step_size + 1


def step_end_for_start(*, step_start: int, step_size: int, max_chapter_idx: int) -> int:
    """Returns the inclusive end idx of a step, clamped to max_chapter_idx."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if step_start <= 0:
        raise ValueError("step_start must be positive")
    if max_chapter_idx <= 0:
        raise ValueError("max_chapter_idx must be positive")

    return min(max_chapter_idx, step_start + step_size - 1)


def align_from_chapter(*, from_chapter: int, step_size: int) -> int:
    """Aligns a user-provided from_chapter down to the step start."""

    return step_start_for_chapter(chapter_idx=from_chapter, step_size=step_size)


def align_to_chapter(*, to_chapter: int, step_size: int, max_chapter_idx: int) -> int:
    """Aligns a user-provided to_chapter up to the step end, clamped to max_chapter_idx."""

    start = step_start_for_chapter(chapter_idx=to_chapter, step_size=step_size)
    return step_end_for_start(step_start=start, step_size=step_size, max_chapter_idx=max_chapter_idx)


def iter_step_ranges(*, start_chapter: int, end_chapter: int, step_size: int) -> list[tuple[int, int]]:
    """Builds inclusive (start, end) step ranges for a chapter interval."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if start_chapter <= 0 or end_chapter <= 0:
        raise ValueError("chapter idx must be positive")
    if start_chapter > end_chapter:
        return []

    ranges: list[tuple[int, int]] = []
    current = start_chapter
    while current <= end_chapter:
        step_start = step_start_for_chapter(chapter_idx=current, step_size=step_size)
        step_end = min(end_chapter, step_start + step_size - 1)
        ranges.append((step_start, step_end))
        current = step_end + 1

    # De-duplicate in case start_chapter is not aligned (caller may choose to do so).
    deduped: list[tuple[int, int]] = []
    for item in ranges:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped
