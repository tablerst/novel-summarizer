from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def import_all_models() -> None:
    from novel_summarizer.storage.books.base import Book
    from novel_summarizer.storage.chapters.base import Chapter
    from novel_summarizer.storage.chunks.base import Chunk
    from novel_summarizer.storage.narrations.base import Narration
    from novel_summarizer.storage.narration_outputs.base import NarrationOutput
    from novel_summarizer.storage.summaries.base import Summary
    from novel_summarizer.storage.world_state.characters import CharacterState
    from novel_summarizer.storage.world_state.checkpoints import WorldStateCheckpoint
    from novel_summarizer.storage.world_state.items import ItemState
    from novel_summarizer.storage.world_state.plot_events import PlotEvent
    from novel_summarizer.storage.world_state.world_facts import WorldFact

    _ = (
        Book,
        Chapter,
        Chunk,
        Summary,
        Narration,
        NarrationOutput,
        CharacterState,
        ItemState,
        PlotEvent,
        WorldFact,
        WorldStateCheckpoint,
    )
