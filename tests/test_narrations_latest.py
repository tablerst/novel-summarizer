from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from novel_summarizer.storage.base import Base, import_all_models
from novel_summarizer.storage.books import crud as books_crud
from novel_summarizer.storage.chapters import crud as chapters_crud
from novel_summarizer.storage.narrations import crud as narrations_crud


async def _build_test_db(tmp_path: Path):
    db_path = tmp_path / "narrations_latest_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}", future=True)
    import_all_models()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            sa_text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS narrations_fts
                USING fts5(narration_id UNINDEXED, book_id UNINDEXED, chapter_idx UNINDEXED, chapter_title, text);
                """
            )
        )
    return engine


async def _seed_book_with_multi_version_narrations(tmp_path: Path):
    engine = await _build_test_db(tmp_path)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with session_maker() as session:
        book = await books_crud.get_or_create_book(
            session,
            title="测试书",
            author="测试作者",
            book_hash="book-hash-latest",
            source_path="input.txt",
        )
        chapter_1 = await chapters_crud.upsert_chapter(
            session,
            book_id=book.id,
            idx=1,
            title="第1章",
            chapter_hash="chapter-1-hash",
            start_pos=0,
            end_pos=100,
        )
        chapter_2 = await chapters_crud.upsert_chapter(
            session,
            book_id=book.id,
            idx=2,
            title="第2章",
            chapter_hash="chapter-2-hash",
            start_pos=101,
            end_pos=200,
        )

        # Chapter 1 has two versions; latest should win.
        await narrations_crud.upsert_narration(
            session,
            book_id=book.id,
            chapter_id=chapter_1.id,
            chapter_idx=1,
            narration_text="第一版",
            key_events_json=None,
            prompt_version="v1",
            model="m",
            input_hash="hash-v1",
        )
        await narrations_crud.upsert_narration(
            session,
            book_id=book.id,
            chapter_id=chapter_1.id,
            chapter_idx=1,
            narration_text="第二版",
            key_events_json=None,
            prompt_version="v1",
            model="m",
            input_hash="hash-v2",
        )

        # Chapter 2 has only one version.
        await narrations_crud.upsert_narration(
            session,
            book_id=book.id,
            chapter_id=chapter_2.id,
            chapter_idx=2,
            narration_text="第二章正文",
            key_events_json=None,
            prompt_version="v1",
            model="m",
            input_hash="hash-v3",
        )

        await session.commit()

    return engine, session_maker


def test_list_latest_narrations_by_book_returns_latest_per_chapter(tmp_path: Path) -> None:
    async def _run() -> None:
        engine, session_maker = await _seed_book_with_multi_version_narrations(tmp_path)
        try:
            async with session_maker() as session:
                latest_rows = await narrations_crud.list_latest_narrations_by_book(session, book_id=1)

            assert [row.chapter_idx for row in latest_rows] == [1, 2]
            assert latest_rows[0].narration_text == "第二版"
            assert latest_rows[1].narration_text == "第二章正文"
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_rebuild_narrations_fts_for_book_indexes_latest_only(tmp_path: Path) -> None:
    async def _run() -> None:
        engine, session_maker = await _seed_book_with_multi_version_narrations(tmp_path)
        try:
            async with session_maker() as session:
                rows = await narrations_crud.rebuild_narrations_fts_for_book(session, book_id=1)
                fts_result = await session.execute(
                    sa_text(
                        "SELECT CAST(chapter_idx AS INTEGER), text FROM narrations_fts WHERE book_id = :book_id ORDER BY CAST(chapter_idx AS INTEGER)"
                    ),
                    {"book_id": 1},
                )
                fts_rows = fts_result.all()

            assert rows == 2
            assert len(fts_rows) == 2
            assert fts_rows[0][0] == 1
            assert fts_rows[0][1] == "第二版"
            assert fts_rows[1][0] == 2
            assert fts_rows[1][1] == "第二章正文"
        finally:
            await engine.dispose()

    asyncio.run(_run())
