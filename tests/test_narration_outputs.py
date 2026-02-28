from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from novel_summarizer.storage.base import Base, import_all_models
from novel_summarizer.storage.books import crud as books_crud
from novel_summarizer.storage.chapters import crud as chapters_crud
from novel_summarizer.storage.narrations import crud as narrations_crud
from novel_summarizer.storage.repo import SQLAlchemyRepo


async def _build_test_db(tmp_path: Path):
    db_path = tmp_path / "narration_outputs_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}", future=True)
    import_all_models()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def test_upsert_and_get_latest_narration_output(tmp_path: Path) -> None:
    async def _run() -> None:
        engine = await _build_test_db(tmp_path)
        session_maker = async_sessionmaker(engine, expire_on_commit=False)

        try:
            async with session_maker() as session:
                repo = SQLAlchemyRepo(session)
                book = await books_crud.get_or_create_book(
                    session,
                    title="测试书",
                    author="作者",
                    book_hash="hash",
                    source_path="input.txt",
                )
                ch = await chapters_crud.upsert_chapter(
                    session,
                    book_id=book.id,
                    idx=1,
                    title="第1章",
                    chapter_hash="c1",
                    start_pos=0,
                    end_pos=10,
                )

                nar = await narrations_crud.upsert_narration(
                    session,
                    book_id=book.id,
                    chapter_id=ch.id,
                    chapter_idx=1,
                    narration_text="正文",
                    key_events_json=None,
                    prompt_version="v0",
                    model="m",
                    input_hash="h1",
                )

                payload = {
                    "entities_mentioned": ["韩立"],
                    "key_events": [{"what": "获得掌天瓶"}],
                    "character_updates": [],
                    "new_items": [],
                }
                payload_json = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")

                await repo.upsert_narration_output(
                    narration_id=nar.id,
                    book_id=book.id,
                    chapter_id=ch.id,
                    chapter_idx=1,
                    prompt_version="v0-step",
                    model="m",
                    input_hash="h1",
                    payload_json=payload_json,
                )

                row = await repo.get_narration_output(narration_id=nar.id)
                assert row is not None
                assert row.narration_id == nar.id

                latest = await repo.get_latest_narration_output_for_chapter(chapter_id=ch.id)
                assert latest is not None
                loaded = orjson.loads(latest.payload_json)
                assert loaded["entities_mentioned"] == ["韩立"]

                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())
