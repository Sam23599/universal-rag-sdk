"""Small async TTL cache interface and in-memory implementation."""

from __future__ import annotations

import asyncio
import time
from typing import Generic, TypeVar


T = TypeVar("T")


class AsyncTTLCache(Generic[T]):
    def __init__(self, *, ttl_seconds: float = 60.0, max_entries: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._values: dict[str, tuple[float, T]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        item = self._values.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= time.monotonic():
            async with self._lock:
                self._values.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: T) -> None:
        async with self._lock:
            if len(self._values) >= self.max_entries:
                oldest = min(self._values, key=lambda item: self._values[item][0])
                self._values.pop(oldest, None)
            self._values[key] = (time.monotonic() + self.ttl_seconds, value)

    async def invalidate(self) -> None:
        async with self._lock:
            self._values.clear()
