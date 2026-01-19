from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    idx: int
    text: str
    start_pos: int
    end_pos: int
    token_count: int


def _estimate_tokens(text: str) -> int:
    return len(text)


def split_text(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    if not text:
        return []

    length = len(text)
    if length <= chunk_size_tokens:
        return [Chunk(idx=1, text=text, start_pos=0, end_pos=length, token_count=_estimate_tokens(text))]

    chunks: list[Chunk] = []
    start = 0
    idx = 1
    while start < length:
        end = min(start + chunk_size_tokens, length)
        segment = text[start:end]
        token_count = _estimate_tokens(segment)

        if token_count < min_chunk_tokens and chunks:
            prev = chunks[-1]
            merged_text = prev.text + segment
            chunks[-1] = Chunk(
                idx=prev.idx,
                text=merged_text,
                start_pos=prev.start_pos,
                end_pos=end,
                token_count=_estimate_tokens(merged_text),
            )
            break

        chunks.append(Chunk(idx=idx, text=segment, start_pos=start, end_pos=end, token_count=token_count))
        idx += 1
        if end == length:
            break
        start = max(0, end - chunk_overlap_tokens)
        if start == end:
            start = end + 1

    return chunks
