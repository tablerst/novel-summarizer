from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.books import crud as books_crud
from novel_summarizer.storage.chapters import crud as chapters_crud
from novel_summarizer.storage.chunks import crud as chunks_crud
from novel_summarizer.storage.narrations import crud as narrations_crud
from novel_summarizer.storage.summaries import crud as summaries_crud
from novel_summarizer.storage.types import (
    BookRow,
    ChapterRow,
    CharacterRow,
    ChunkRow,
    InsertResult,
    ItemRow,
    NarrationRow,
    PlotEventRow,
    SummaryRow,
)
from novel_summarizer.storage.world_state import characters as characters_crud
from novel_summarizer.storage.world_state import items as items_crud
from novel_summarizer.storage.world_state import plot_events as plot_events_crud


class SQLAlchemyRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_book(
        self,
        title: str | None,
        author: str | None,
        book_hash: str,
        source_path: str,
    ) -> InsertResult:
        return await books_crud.get_or_create_book(self.session, title, author, book_hash, source_path)

    async def upsert_chapter(
        self,
        book_id: int,
        idx: int,
        title: str,
        chapter_hash: str,
        start_pos: int,
        end_pos: int,
    ) -> InsertResult:
        return await chapters_crud.upsert_chapter(
            self.session,
            book_id=book_id,
            idx=idx,
            title=title,
            chapter_hash=chapter_hash,
            start_pos=start_pos,
            end_pos=end_pos,
        )

    async def upsert_chunk(
        self,
        chapter_id: int,
        idx: int,
        chunk_hash: str,
        text: str,
        token_count: int,
        start_pos: int,
        end_pos: int,
        meta_json: str | None = None,
    ) -> InsertResult:
        return await chunks_crud.upsert_chunk(
            self.session,
            chapter_id=chapter_id,
            idx=idx,
            chunk_hash=chunk_hash,
            text=text,
            token_count=token_count,
            start_pos=start_pos,
            end_pos=end_pos,
            meta_json=meta_json,
        )

    async def list_chapters(self, book_id: int) -> list[ChapterRow]:
        return await chapters_crud.list_chapters(self.session, book_id)

    async def get_book(self, book_id: int) -> BookRow:
        return await books_crud.get_book(self.session, book_id)

    async def list_chunks(self, chapter_id: int) -> list[ChunkRow]:
        return await chunks_crud.list_chunks(self.session, chapter_id)

    async def get_narration(
        self,
        chapter_id: int,
        prompt_version: str,
        model: str,
        input_hash: str,
    ) -> NarrationRow | None:
        return await narrations_crud.get_narration(
            self.session,
            chapter_id=chapter_id,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
        )

    async def get_latest_narration(self, chapter_id: int) -> NarrationRow | None:
        return await narrations_crud.get_latest_narration(self.session, chapter_id)

    async def list_narrations_by_book(self, book_id: int) -> list[NarrationRow]:
        return await narrations_crud.list_narrations_by_book(self.session, book_id)

    async def upsert_narration(
        self,
        book_id: int,
        chapter_id: int,
        chapter_idx: int,
        narration_text: str,
        key_events_json: str | None,
        prompt_version: str,
        model: str,
        input_hash: str,
    ) -> InsertResult:
        return await narrations_crud.upsert_narration(
            self.session,
            book_id=book_id,
            chapter_id=chapter_id,
            chapter_idx=chapter_idx,
            narration_text=narration_text,
            key_events_json=key_events_json,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
        )

    async def list_character_states(self, book_id: int, canonical_names: list[str] | None = None) -> list[CharacterRow]:
        return await characters_crud.list_character_states(
            self.session,
            book_id=book_id,
            canonical_names=canonical_names,
        )

    async def upsert_character_state(
        self,
        book_id: int,
        canonical_name: str,
        aliases_json: str = "[]",
        first_chapter_idx: int | None = None,
        last_chapter_idx: int | None = None,
        status: str = "active",
        location: str | None = None,
        abilities_json: str | None = None,
        relationships_json: str | None = None,
        motivation: str | None = None,
        notes: str | None = None,
    ) -> InsertResult:
        return await characters_crud.upsert_character_state(
            self.session,
            book_id=book_id,
            canonical_name=canonical_name,
            aliases_json=aliases_json,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            status=status,
            location=location,
            abilities_json=abilities_json,
            relationships_json=relationships_json,
            motivation=motivation,
            notes=notes,
        )

    async def list_item_states(self, book_id: int, names: list[str] | None = None) -> list[ItemRow]:
        return await items_crud.list_item_states(self.session, book_id=book_id, names=names)

    async def upsert_item_state(
        self,
        book_id: int,
        name: str,
        owner_name: str | None = None,
        first_chapter_idx: int | None = None,
        last_chapter_idx: int | None = None,
        description: str | None = None,
        status: str = "active",
    ) -> InsertResult:
        return await items_crud.upsert_item_state(
            self.session,
            book_id=book_id,
            name=name,
            owner_name=owner_name,
            first_chapter_idx=first_chapter_idx,
            last_chapter_idx=last_chapter_idx,
            description=description,
            status=status,
        )

    async def list_recent_plot_events(
        self,
        book_id: int,
        chapter_idx: int | None = None,
        window: int = 5,
        limit: int = 20,
    ) -> list[PlotEventRow]:
        return await plot_events_crud.list_recent_plot_events(
            self.session,
            book_id=book_id,
            chapter_idx=chapter_idx,
            window=window,
            limit=limit,
        )

    async def insert_plot_event(
        self,
        book_id: int,
        chapter_idx: int,
        event_summary: str,
        involved_characters_json: str | None = None,
        event_type: str | None = None,
        impact: str | None = None,
    ) -> InsertResult:
        return await plot_events_crud.insert_plot_event(
            self.session,
            book_id=book_id,
            chapter_idx=chapter_idx,
            event_summary=event_summary,
            involved_characters_json=involved_characters_json,
            event_type=event_type,
            impact=impact,
        )

    async def get_summary(
        self,
        scope: str,
        ref_id: int,
        summary_type: str,
        prompt_version: str,
        model: str,
        input_hash: str,
    ) -> SummaryRow | None:
        return await summaries_crud.get_summary(
            self.session,
            scope=scope,
            ref_id=ref_id,
            summary_type=summary_type,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
        )

    async def get_latest_summary(self, scope: str, ref_id: int, summary_type: str) -> SummaryRow | None:
        return await summaries_crud.get_latest_summary(self.session, scope, ref_id, summary_type)

    async def upsert_summary(
        self,
        scope: str,
        ref_id: int,
        summary_type: str,
        prompt_version: str,
        model: str,
        input_hash: str,
        content: str,
        params_json: str | None = None,
    ) -> InsertResult:
        return await summaries_crud.upsert_summary(
            self.session,
            scope=scope,
            ref_id=ref_id,
            summary_type=summary_type,
            prompt_version=prompt_version,
            model=model,
            input_hash=input_hash,
            content=content,
            params_json=params_json,
        )
