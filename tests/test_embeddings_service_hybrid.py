from __future__ import annotations

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
