from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from novel_summarizer.storage.books import crud as books_crud
from novel_summarizer.storage.chapters import crud as chapters_crud
from novel_summarizer.storage.chunks import crud as chunks_crud
from novel_summarizer.storage.summaries import crud as summaries_crud
from novel_summarizer.storage.types import BookRow, ChapterRow, ChunkRow, InsertResult, SummaryRow


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
