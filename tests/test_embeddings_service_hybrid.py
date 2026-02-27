from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.embeddings import service
from novel_summarizer.storage.types import ChapterRow, NarrationRow


def test_extract_keyword_terms_deduplicate_and_cap() -> None:
    terms = service._extract_keyword_terms(
        "韩立在天南天南遇到掌天瓶并筑基成功",
        ["韩立", "掌天瓶", "韩立"],
        max_terms=4,
    )
    assert len(terms) <= 4
    assert terms[0] == "韩立"
    assert terms[1] == "掌天瓶"


def test_build_fts_query_wraps_quotes() -> None:
    query = service._build_fts_query(["韩立", "掌天瓶"])
    assert query == '"韩立" OR "掌天瓶"'


def test_norm_rank_and_proximity() -> None:
    assert service._norm_rank(1, 4) == 1.0
    assert service._norm_rank(4, 4) == 0.25
    assert service._proximity_score(10, 9) == 0.5
    assert service._proximity_score(10, 10) == 0.0


def test_retrieve_vector_records_skip_when_table_missing(monkeypatch) -> None:
    class _FakeStore:
        def get_table(self):
            return None

        def similarity_search_by_vector(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("similarity_search_by_vector should not be called when table is missing")

    class _FakeEmbedClient:
        def __init__(self, config):
            _ = config

        def embed_query(self, text):  # pragma: no cover
            raise AssertionError("embed_query should not be called when table is missing")

    monkeypatch.setattr(service, "_build_vector_store", lambda config, table_name: _FakeStore())
    monkeypatch.setattr(service, "OpenAIEmbeddingClient", _FakeEmbedClient)

    result = service._retrieve_vector_records(
        config=AppConfigRoot(),
        table_name="narrations_vectors_1",
        query_text="韩立",
        top_k=3,
        source_type="narration",
        source_id_key="narration_id",
    )
    assert result == []


def test_embed_book_narrations_uses_latest_list(monkeypatch) -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.add_calls = 0
            self.rows = []

        def get_table(self):
            return None

        def add_texts(self, texts, metadatas=None, ids=None):
            self.add_calls += 1
            self.rows.append((texts, metadatas, ids))

    class _FakeRepo:
        def __init__(self) -> None:
            self.called_latest = False

        async def list_latest_narrations_by_book(self, book_id: int):
            _ = book_id
            self.called_latest = True
            return [
                NarrationRow(
                    id=10,
                    book_id=1,
                    chapter_id=100,
                    chapter_idx=1,
                    narration_text="第二版正文",
                    key_events_json=None,
                    prompt_version="v1",
                    model="m",
                    input_hash="h2",
                )
            ]

        async def list_narrations_by_book(self, book_id: int):  # pragma: no cover
            _ = book_id
            raise AssertionError("embedding should use latest narrations list")

        async def list_chapters(self, book_id: int):
            _ = book_id
            return [ChapterRow(id=100, idx=1, title="第1章")]

    fake_store = _FakeStore()
    fake_repo = _FakeRepo()

    @asynccontextmanager
    async def _fake_session_scope():
        yield object()

    monkeypatch.setattr(service, "_build_vector_store", lambda config, table_name: fake_store)
    monkeypatch.setattr(service, "session_scope", _fake_session_scope)
    monkeypatch.setattr(service, "SQLAlchemyRepo", lambda session: fake_repo)

    stats = asyncio.run(service.embed_book_narrations(book_id=1, config=AppConfigRoot(), batch_size=32))

    assert fake_repo.called_latest is True
    assert fake_store.add_calls == 1
    assert stats.narrations_total == 1
    assert stats.narrations_embedded == 1
    assert stats.narrations_skipped == 0
