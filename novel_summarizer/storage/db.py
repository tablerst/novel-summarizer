from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from novel_summarizer.storage.base import Base, import_all_models

_db_service: "DatabaseService | None" = None


def _build_sqlite_url(db_path: Path) -> str:
    resolved = db_path.resolve()
    return f"sqlite+aiosqlite:///{resolved.as_posix()}"


class DatabaseService:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine: AsyncEngine = create_async_engine(db_url, future=True)
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

        @event.listens_for(self.engine.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    async def init_models(self) -> None:
        import_all_models()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def with_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._sessionmaker() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()


async def init_db_service(db_path: Path) -> DatabaseService:
    global _db_service
    if _db_service is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = _build_sqlite_url(db_path)
        service = DatabaseService(db_url)
        await service.init_models()
        _db_service = service
    return _db_service


def get_db_service() -> DatabaseService:
    if _db_service is None:
        raise RuntimeError("Database service not initialized. Call init_db_service() first.")
    return _db_service


async def shutdown_db_service() -> None:
    global _db_service
    if _db_service is None:
        return
    await _db_service.dispose()
    _db_service = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Retrieves an async session from the database service.

    Yields:
        AsyncSession: An async session object.

    """

    async with get_db_service().with_session() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for managing an async session scope.

    This context manager is used to manage an async session scope for database operations.
    It ensures that the session is properly committed if no exceptions occur,
    and rolled back if an exception is raised.

    Yields:
        AsyncSession: The async session object.

    Raises:
        Exception: If an error occurs during the session scope.

    """

    db_service = get_db_service()
    async with db_service.with_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            logger.exception("An error occurred during the session scope.")
            await session.rollback()
            raise
