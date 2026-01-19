from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time


@dataclass
class CacheResult:
    value: str | None
    hit: bool


class SimpleCache:
    def __init__(self, enabled: bool, backend: str, base_dir: Path, ttl_seconds: int):
        self.enabled = enabled
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self._conn: sqlite3.Connection | None = None

        if not enabled:
            return

        if backend == "sqlite":
            cache_path = (base_dir / "cache.sqlite").resolve()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(cache_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()
        else:
            self.enabled = False

    def get(self, key: str) -> CacheResult:
        if not self.enabled or not self._conn:
            return CacheResult(value=None, hit=False)

        row = self._conn.execute("SELECT value, created_at FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            return CacheResult(value=None, hit=False)

        value, created_at = row
        if self.ttl_seconds > 0 and (time.time() - float(created_at)) > self.ttl_seconds:
            self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
            self._conn.commit()
            return CacheResult(value=None, hit=False)

        return CacheResult(value=str(value), hit=True)

    def set(self, key: str, value: str) -> None:
        if not self.enabled or not self._conn:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    def delete(self, key: str) -> None:
        if not self.enabled or not self._conn:
            return
        self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
