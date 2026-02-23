from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from loguru import logger

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document

from novel_summarizer.config.schema import AppConfigRoot
from novel_summarizer.llm.embeddings import OpenAIEmbeddingClient
from novel_summarizer.storage.db import session_scope
from novel_summarizer.storage.repo import SQLAlchemyRepo


@dataclass
class EmbedStats:
    book_id: int
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int


@dataclass
class NarrationEmbedStats:
    book_id: int
    narrations_total: int
    narrations_embedded: int
    narrations_skipped: int


@dataclass
class RetrievalAssetsStats:
    book_id: int
    chunk_vectors_embedded: int
    narration_vectors_embedded: int
    chunk_fts_rows: int
    narration_fts_rows: int


def _chunks_table_name(book_id: int) -> str:
    return f"chunks_vectors_{book_id}"


def _narrations_table_name(book_id: int) -> str:
    return f"narrations_vectors_{book_id}"


def _build_vector_store(config: AppConfigRoot, table_name: str) -> LanceDB:
    client = OpenAIEmbeddingClient(config)
    return LanceDB(
        uri=str(config.storage.lancedb_dir),
        table_name=table_name,
        embedding=client.model,
        mode="append",
    )


def _list_existing_ids(store: LanceDB, id_keys: tuple[str, ...]) -> set[int]:
    table = store.get_table()
    if table is None:
        return set()
    df = table.to_pandas()
    values = None
    for key in id_keys:
        if key in df.columns:
            values = df[key].tolist()
            break

    if values is not None:
        pass
    elif "id" in df.columns:
        values = df["id"].tolist()
    elif "metadata" in df.columns:
        values = []
        for meta in df["metadata"].tolist():
            if not isinstance(meta, dict):
                continue
            for key in id_keys:
                if key in meta:
                    values.append(meta[key])
                    break
    else:
        return set()
    ids: set[int] = set()
    for value in values:
        try:
            ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _docs_to_records(docs: list[Document]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for doc in docs:
        record = dict(doc.metadata or {})
        record["text"] = doc.page_content
        records.append(record)
    return records


async def embed_book_chunks(book_id: int, config: AppConfigRoot, batch_size: int = 32) -> EmbedStats:
    store = _build_vector_store(config, _chunks_table_name(book_id))
    existing_ids = _list_existing_ids(store, ("chunk_id",))

    chunks_total = 0
    chunks_embedded = 0
    chunks_skipped = 0

    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        chapters = await repo.list_chapters(book_id)

        pending_texts: list[str] = []
        pending_metadata: list[dict[str, Any]] = []
        pending_ids: list[str] = []

        def flush() -> None:
            nonlocal chunks_embedded
            if not pending_texts:
                return
            store.add_texts(pending_texts, metadatas=pending_metadata, ids=pending_ids)
            chunks_embedded += len(pending_texts)
            pending_texts.clear()
            pending_metadata.clear()
            pending_ids.clear()

        for chapter in chapters:
            chunks = await repo.list_chunks(chapter.id)
            for chunk in chunks:
                chunks_total += 1
                if chunk.id in existing_ids:
                    chunks_skipped += 1
                    continue

                pending_metadata.append(
                    {
                        "chunk_id": chunk.id,
                        "chunk_hash": chunk.chunk_hash,
                        "chapter_id": chapter.id,
                        "chapter_idx": chapter.idx,
                        "chapter_title": chapter.title,
                    }
                )
                pending_ids.append(str(chunk.id))
                pending_texts.append(chunk.text)

                if len(pending_texts) >= batch_size:
                    flush()

        flush()

    logger.info(
        "Embedding complete: total={} embedded={} skipped={}",
        chunks_total,
        chunks_embedded,
        chunks_skipped,
    )
    return EmbedStats(
        book_id=book_id,
        chunks_total=chunks_total,
        chunks_embedded=chunks_embedded,
        chunks_skipped=chunks_skipped,
    )


async def embed_book_narrations(book_id: int, config: AppConfigRoot, batch_size: int = 32) -> NarrationEmbedStats:
    store = _build_vector_store(config, _narrations_table_name(book_id))
    existing_ids = _list_existing_ids(store, ("narration_id",))

    narrations_total = 0
    narrations_embedded = 0
    narrations_skipped = 0

    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        narrations = await repo.list_narrations_by_book(book_id)
        chapters = await repo.list_chapters(book_id)
        chapter_title_map = {chapter.id: chapter.title for chapter in chapters}

        pending_texts: list[str] = []
        pending_metadata: list[dict[str, Any]] = []
        pending_ids: list[str] = []

        def flush() -> None:
            nonlocal narrations_embedded
            if not pending_texts:
                return
            store.add_texts(pending_texts, metadatas=pending_metadata, ids=pending_ids)
            narrations_embedded += len(pending_texts)
            pending_texts.clear()
            pending_metadata.clear()
            pending_ids.clear()

        for row in narrations:
            narrations_total += 1
            if row.id in existing_ids:
                narrations_skipped += 1
                continue

            pending_metadata.append(
                {
                    "narration_id": row.id,
                    "chapter_id": row.chapter_id,
                    "chapter_idx": row.chapter_idx,
                    "chapter_title": chapter_title_map.get(row.chapter_id, ""),
                    "prompt_version": row.prompt_version,
                    "model": row.model,
                    "input_hash": row.input_hash,
                }
            )
            pending_ids.append(str(row.id))
            pending_texts.append(row.narration_text)

            if len(pending_texts) >= batch_size:
                flush()

        flush()

    logger.info(
        "Narration embedding complete: total={} embedded={} skipped={}",
        narrations_total,
        narrations_embedded,
        narrations_skipped,
    )
    return NarrationEmbedStats(
        book_id=book_id,
        narrations_total=narrations_total,
        narrations_embedded=narrations_embedded,
        narrations_skipped=narrations_skipped,
    )


def retrieve_evidence(
    *,
    book_id: int,
    config: AppConfigRoot,
    query_text: str,
    top_k: int,
    chapter_id: int | None = None,
) -> list[dict[str, Any]]:
    store = _build_vector_store(config, _chunks_table_name(book_id))
    if top_k <= 0:
        return []

    client = OpenAIEmbeddingClient(config)
    query_vector = client.embed_query(query_text)
    docs = store.similarity_search_by_vector(query_vector, k=max(top_k * 3, top_k))
    raw_results = _docs_to_records(docs)
    if not raw_results:
        return []

    if chapter_id is None:
        return raw_results[:top_k]

    filtered = [item for item in raw_results if int(item.get("chapter_id", -1)) == chapter_id]
    if len(filtered) >= top_k:
        return filtered[:top_k]

    extras = [item for item in raw_results if item not in filtered]
    return (filtered + extras)[:top_k]


def _extract_keyword_terms(query_text: str, terms: list[str] | None, max_terms: int = 8) -> list[str]:
    values = terms[:] if terms else []
    values.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,20}", query_text))
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= max_terms:
            break
    return output


def _build_fts_query(terms: list[str]) -> str:
    quoted = []
    for term in terms:
        escaped = term.replace('"', "")
        if escaped:
            quoted.append(f'"{escaped}"')
    return " OR ".join(quoted)


def _norm_rank(rank: int, size: int) -> float:
    if size <= 0:
        return 0.0
    return max(0.0, 1.0 - ((rank - 1) / max(size, 1)))


def _proximity_score(current_chapter_idx: int | None, source_chapter_idx: int) -> float:
    if current_chapter_idx is None:
        return 0.0
    if source_chapter_idx >= current_chapter_idx:
        return 0.0
    distance = current_chapter_idx - source_chapter_idx
    return 1.0 / (1.0 + float(distance))


def _retrieve_vector_records(
    *,
    config: AppConfigRoot,
    table_name: str,
    query_text: str,
    top_k: int,
    source_type: str,
    source_id_key: str,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    store = _build_vector_store(config, table_name)
    table = store.get_table()
    if table is None:
        return []

    client = OpenAIEmbeddingClient(config)
    query_vector = client.embed_query(query_text)
    docs = store.similarity_search_by_vector(query_vector, k=max(top_k, 1))
    raw_results = _docs_to_records(docs)

    records: list[dict[str, Any]] = []
    for rank, item in enumerate(raw_results, start=1):
        source_id = item.get(source_id_key)
        if source_id is None:
            continue
        try:
            source_id = int(source_id)
            chapter_idx = int(item.get("chapter_idx"))
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "chapter_idx": chapter_idx,
                "chapter_title": str(item.get("chapter_title", "")),
                "text": str(item.get("text", "")),
                "vector_rank_score": _norm_rank(rank, len(raw_results)),
                "keyword_rank_score": 0.0,
            }
        )
    return records


async def prepare_retrieval_assets(book_id: int, config: AppConfigRoot, batch_size: int = 32) -> RetrievalAssetsStats:
    chunk_stats = await embed_book_chunks(book_id=book_id, config=config, batch_size=batch_size)
    narration_stats = await embed_book_narrations(book_id=book_id, config=config, batch_size=batch_size)

    chunk_fts_rows = 0
    narration_fts_rows = 0
    async with session_scope() as session:
        repo = SQLAlchemyRepo(session)
        try:
            chunk_fts_rows = await repo.rebuild_chunks_fts_for_book(book_id)
            narration_fts_rows = await repo.rebuild_narrations_fts_for_book(book_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FTS index rebuild skipped: {}", exc)

    return RetrievalAssetsStats(
        book_id=book_id,
        chunk_vectors_embedded=chunk_stats.chunks_embedded,
        narration_vectors_embedded=narration_stats.narrations_embedded,
        chunk_fts_rows=chunk_fts_rows,
        narration_fts_rows=narration_fts_rows,
    )


async def retrieve_hybrid_memories(
    *,
    book_id: int,
    config: AppConfigRoot,
    query_text: str,
    top_k: int,
    current_chapter_idx: int | None,
    keyword_terms: list[str] | None = None,
    alpha: float = 0.7,
    beta: float = 0.2,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    candidates: list[dict[str, Any]] = []

    try:
        candidates.extend(
            _retrieve_vector_records(
                config=config,
                table_name=_chunks_table_name(book_id),
                query_text=query_text,
                top_k=max(top_k * 3, top_k),
                source_type="chunk",
                source_id_key="chunk_id",
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chunk vector retrieval failed: {}", exc)

    try:
        candidates.extend(
            _retrieve_vector_records(
                config=config,
                table_name=_narrations_table_name(book_id),
                query_text=query_text,
                top_k=max(top_k * 2, top_k),
                source_type="narration",
                source_id_key="narration_id",
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Narration vector retrieval failed: {}", exc)

    terms = _extract_keyword_terms(query_text, keyword_terms)
    fts_query = _build_fts_query(terms)

    keyword_hits: list[dict[str, Any]] = []
    if fts_query:
        try:
            async with session_scope() as session:
                repo = SQLAlchemyRepo(session)
                chunk_hits = await repo.search_chunks_fts(
                    book_id=book_id,
                    query=fts_query,
                    before_chapter_idx=current_chapter_idx,
                    limit=max(top_k * 3, top_k),
                )
                narration_hits = await repo.search_narrations_fts(
                    book_id=book_id,
                    query=fts_query,
                    before_chapter_idx=current_chapter_idx,
                    limit=max(top_k * 2, top_k),
                )
            merged_hits = chunk_hits + narration_hits
            for rank, hit in enumerate(merged_hits, start=1):
                keyword_hits.append(
                    {
                        "source_type": hit.source_type,
                        "source_id": hit.source_id,
                        "chapter_idx": hit.chapter_idx,
                        "chapter_title": hit.chapter_title,
                        "text": hit.text,
                        "vector_rank_score": 0.0,
                        "keyword_rank_score": _norm_rank(rank, len(merged_hits)),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FTS retrieval failed: {}", exc)

    aggregated: dict[tuple[str, int], dict[str, Any]] = {}

    def merge(item: dict[str, Any]) -> None:
        try:
            source_type = str(item["source_type"])
            source_id = int(item["source_id"])
            chapter_idx = int(item["chapter_idx"])
        except (KeyError, TypeError, ValueError):
            return
        if current_chapter_idx is not None and chapter_idx >= current_chapter_idx:
            return

        key = (source_type, source_id)
        entry = aggregated.setdefault(
            key,
            {
                "source_type": source_type,
                "source_id": source_id,
                "chapter_idx": chapter_idx,
                "chapter_title": str(item.get("chapter_title", "")),
                "text": str(item.get("text", ""))[:800],
                "vector_rank_score": 0.0,
                "keyword_rank_score": 0.0,
            },
        )
        entry["vector_rank_score"] = max(float(entry["vector_rank_score"]), float(item.get("vector_rank_score", 0.0)))
        entry["keyword_rank_score"] = max(
            float(entry["keyword_rank_score"]), float(item.get("keyword_rank_score", 0.0))
        )

    for item in candidates:
        merge(item)
    for item in keyword_hits:
        merge(item)

    results: list[dict[str, Any]] = []
    for entry in aggregated.values():
        proximity = _proximity_score(current_chapter_idx, int(entry["chapter_idx"]))
        vector_component = float(entry["vector_rank_score"])
        keyword_component = float(entry["keyword_rank_score"])
        score = alpha * vector_component + (1 - alpha) * keyword_component + beta * proximity
        entry["score"] = score
        entry["proximity_score"] = proximity
        results.append(entry)

    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return results[:top_k]
