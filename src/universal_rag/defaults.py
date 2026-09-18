"""Default strategies that keep the core SDK dependency-free."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Sequence

from .models import TranscriptTurn


class LatestTurnQueryPlanner:
    """Build a bounded conversational query without another LLM call."""

    def __init__(self, *, include_previous_turns: int = 2, max_characters: int = 1200) -> None:
        self.include_previous_turns = max(0, include_previous_turns)
        self.max_characters = max(100, max_characters)

    async def build_query(self, turns: Sequence[TranscriptTurn]) -> str:
        if not turns:
            return ""
        latest = turns[-1]
        earlier = turns[max(0, len(turns) - self.include_previous_turns - 1) : -1]
        parts = [f"{turn.speaker}: {turn.text}" for turn in earlier]
        parts.append(f"{latest.speaker}: {latest.text}")
        query = "\n".join(parts)
        return query[-self.max_characters :]


class RegexPIIRedactor:
    """Conservative baseline redactor; applications can inject a stronger policy."""

    _patterns = (
        (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL]"),
        (re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)"), "[PHONE]"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    )

    def redact(self, text: str) -> str:
        redacted = text
        for pattern, replacement in self._patterns:
            redacted = pattern.sub(replacement, redacted)
        return redacted


class IdentityRedactor:
    def redact(self, text: str) -> str:
        return text


class NullEventSink:
    async def emit(self, event: str, payload: dict) -> None:
        return None


class CallableLLMProvider:
    """Adapter for an async or sync callable with no vendor dependency."""

    def __init__(self, function: Callable[..., str | Awaitable[str]]) -> None:
        self._function = function

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        result = self._function(system_prompt=system_prompt, user_prompt=user_prompt)
        return str(await result if inspect.isawaitable(result) else result)


class CallableEmbeddingProvider:
    """Adapter for an async or sync batch embedding callable."""

    def __init__(
        self, function: Callable[[Sequence[str]], Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]]]
    ) -> None:
        self._function = function

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        result = self._function(texts)
        return await result if inspect.isawaitable(result) else result
