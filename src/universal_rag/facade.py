"""Optional facade composing the full offline and online RAG lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .ingestion import IngestionPipeline
from .models import IngestionReport, RAGResponse, TranscriptTurn
from .pipeline import RAGPipeline, RAGSession
from .protocols import DocumentSource


class RAG:
    """Start-to-end facade; every underlying component remains independently public."""

    def __init__(self, *, ingestion: IngestionPipeline, query: RAGPipeline) -> None:
        self.ingestion = ingestion
        self.query = query

    async def ingest(
        self,
        source: DocumentSource,
        *,
        namespace: str = "default",
        generation: int | None = None,
    ) -> IngestionReport:
        return await self.ingestion.run(source, namespace=namespace, generation=generation)

    async def ask(
        self,
        query: str | Sequence[TranscriptTurn],
        *,
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RAGResponse:
        return await self.query.respond(
            query,
            namespace=namespace,
            filter_metadata=filter_metadata,
            filter_tags=filter_tags,
            principals=principals,
        )

    def new_session(self) -> RAGSession:
        return self.query.new_session()
