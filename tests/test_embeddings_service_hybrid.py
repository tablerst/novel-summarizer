from __future__ import annotations

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.embeddings import service


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
