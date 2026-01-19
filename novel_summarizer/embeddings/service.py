from __future__ import annotations

from dataclasses import dataclass
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


def _table_name(book_id: int) -> str:
    return f"chunks_vectors_{book_id}"


def _build_vector_store(config: AppConfigRoot, table_name: str) -> LanceDB:
    client = OpenAIEmbeddingClient(config)
    return LanceDB(
        uri=str(config.storage.lancedb_dir),
        table_name=table_name,
        embedding=client.model,
        mode="append",
    )


def _list_existing_ids(store: LanceDB) -> set[int]:
    table = store.get_table()
    if table is None:
        return set()
    df = table.to_pandas()
    if "chunk_id" in df.columns:
        values = df["chunk_id"].tolist()
    elif "id" in df.columns:
        values = df["id"].tolist()
    elif "metadata" in df.columns:
        values = []
        for meta in df["metadata"].tolist():
            if isinstance(meta, dict) and "chunk_id" in meta:
                values.append(meta["chunk_id"])
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
    store = _build_vector_store(config, _table_name(book_id))
    existing_ids = _list_existing_ids(store)

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


def retrieve_evidence(
    *,
    book_id: int,
    config: AppConfigRoot,
    query_text: str,
    top_k: int,
    chapter_id: int | None = None,
) -> list[dict[str, Any]]:
    store = _build_vector_store(config, _table_name(book_id))
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
