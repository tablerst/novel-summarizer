from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

import orjson
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from novel_summarizer.domain.hashing import sha256_text
from novel_summarizer.storage.base import Base, import_all_models
from novel_summarizer.storage.books import crud as books_crud
from novel_summarizer.storage.repo import SQLAlchemyRepo


async def _build_test_db(tmp_path: Path):
    db_path = tmp_path / "world_state_checkpoints_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}", future=True)
    import_all_models()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def test_checkpoint_restore_clears_and_rebuilds_world_state(tmp_path: Path) -> None:
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
                    book_hash="book-hash",
                    source_path="input.txt",
                )
                book_id = int(book.id)

                await repo.upsert_character_state(book_id=book_id, canonical_name="韩立", aliases_json='["韩跑跑"]')
                await repo.upsert_item_state(book_id=book_id, name="掌天瓶", owner_name="韩立")
                await repo.insert_plot_event(book_id=book_id, chapter_idx=1, event_summary="获得掌天瓶")
                await repo.upsert_world_fact(book_id=book_id, fact_key="fact:1", fact_value="初始事实")

                # Build snapshot from DB rows.
                characters = [asdict(row) for row in await repo.list_character_states(book_id=book_id)]
                items = [asdict(row) for row in await repo.list_item_states(book_id=book_id)]
                plot_events = [asdict(row) for row in await repo.list_plot_events_by_book(book_id=book_id)]
                world_facts = [asdict(row) for row in await repo.list_world_facts(book_id=book_id, limit=1000)]
                snapshot_obj = {
                    "characters": characters,
                    "items": items,
                    "plot_events": plot_events,
                    "world_facts": world_facts,
                }
                snapshot_json = orjson.dumps(snapshot_obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")
                snapshot_hash = sha256_text(snapshot_json)

                await repo.upsert_world_state_checkpoint(
                    book_id=book_id,
                    chapter_idx=1,
                    step_size=5,
                    snapshot_json=snapshot_json,
                    snapshot_hash=snapshot_hash,
                )

                # Mutate DB after checkpoint.
                await repo.upsert_character_state(book_id=book_id, canonical_name="韩立", aliases_json="[]", status="injured")
                await repo.insert_plot_event(book_id=book_id, chapter_idx=2, event_summary="后续事件")

                checkpoint = await repo.get_latest_world_state_checkpoint_at_or_before(book_id=book_id, chapter_idx=2)
                assert checkpoint is not None
                assert checkpoint.chapter_idx == 1

                await repo.restore_world_state_checkpoint(checkpoint=checkpoint)

                # Verify restore.
                restored_characters = await repo.list_character_states(book_id=book_id)
                assert len(restored_characters) == 1
                assert restored_characters[0].canonical_name == "韩立"
                assert restored_characters[0].aliases_json == '["韩跑跑"]'

                restored_events = await repo.list_plot_events_by_book(book_id=book_id)
                assert [row.chapter_idx for row in restored_events] == [1]
                assert restored_events[0].event_summary == "获得掌天瓶"

                restored_facts = await repo.list_world_facts(book_id=book_id, limit=1000)
                assert any(row.fact_key == "fact:1" for row in restored_facts)

                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())
