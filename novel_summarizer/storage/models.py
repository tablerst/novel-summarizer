"""Deprecated: use per-table models in storage/<table>/base.py."""

from novel_summarizer.storage.base import Base
from novel_summarizer.storage.books.base import Book
from novel_summarizer.storage.chapters.base import Chapter
from novel_summarizer.storage.chunks.base import Chunk
from novel_summarizer.storage.narrations.base import Narration
from novel_summarizer.storage.summaries.base import Summary
from novel_summarizer.storage.world_state.characters import CharacterState
from novel_summarizer.storage.world_state.items import ItemState
from novel_summarizer.storage.world_state.plot_events import PlotEvent
from novel_summarizer.storage.world_state.world_facts import WorldFact

__all__ = [
	"Base",
	"Book",
	"Chapter",
	"Chunk",
	"Summary",
	"Narration",
	"CharacterState",
	"ItemState",
	"PlotEvent",
	"WorldFact",
]
