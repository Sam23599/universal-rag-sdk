"""Bounded asynchronous retry policy for provider adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    initial_delay_seconds: float = 0.1
    maximum_delay_seconds: float = 2.0
    retryable: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


async def with_retry(operation: Callable[[], Awaitable[T]], policy: RetryPolicy | None = None) -> T:
    policy = policy or RetryPolicy()
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except policy.retryable:
            if attempt == policy.attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(policy.maximum_delay_seconds, delay * 2)
    raise RuntimeError("Retry loop exited unexpectedly")
