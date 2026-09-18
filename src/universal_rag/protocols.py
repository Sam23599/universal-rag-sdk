"""Provider interfaces used by the offline and online pipelines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, Sequence

from .models import (
    Chunk,
    Document,
    IndexState,
    ParsedDocument,
    QueryContext,
    RawDocument,
    RetrievalResult,
    TranscriptTurn,
)


class DocumentSource(Protocol):
    def __aiter__(self) -> AsyncIterator[RawDocument]: ...


class DocumentParser(Protocol):
    async def parse(self, document: RawDocument) -> ParsedDocument: ...


class DocumentCleaner(Protocol):
    def clean(self, document: ParsedDocument) -> ParsedDocument: ...


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> Sequence[Chunk]: ...


class ChunkEnricher(Protocol):
    async def enrich(self, chunk: Chunk) -> Chunk: ...


class DocumentStore(Protocol):
    async def put(self, document: RawDocument) -> None: ...

    async def get(self, document_id: str) -> RawDocument | None: ...


class IndexBackend(Protocol):
    async def publish(self, namespace: str, generation: int, chunks: Sequence[Chunk]) -> IndexState: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class Reranker(Protocol):
    async def score(self, query: str, documents: Sequence[Document]) -> Sequence[float]: ...


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RetrievalResult: ...


class LLMProvider(Protocol):
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class QueryPlanner(Protocol):
    async def build_query(self, turns: Sequence[TranscriptTurn]) -> str: ...


class QueryProcessor(Protocol):
    async def process(self, context: QueryContext) -> QueryContext: ...


class Redactor(Protocol):
    def redact(self, text: str) -> str: ...


class EventSink(Protocol):
    async def emit(self, event: str, payload: dict[str, Any]) -> None: ...
