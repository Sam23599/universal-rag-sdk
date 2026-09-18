"""Asynchronous source-to-index ingestion pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .config import IngestionConfig
from .defaults import NullEventSink
from .models import Chunk, IngestionReport, ParsedDocument, RawDocument
from .parsing import ParserRegistry
from .processing import DocumentDeduplicator, MetadataEnricher, StandardCleaner, StructureAwareChunker
from .protocols import ChunkEnricher, Chunker, DocumentCleaner, DocumentSource, DocumentStore, EventSink, IndexBackend


class IngestionPipeline:
    """Orchestrates independent ingestion stages and atomically publishes one generation."""

    def __init__(
        self,
        *,
        index: IndexBackend,
        parsers: ParserRegistry | None = None,
        cleaners: Sequence[DocumentCleaner] | None = None,
        chunker: Chunker | None = None,
        enrichers: Sequence[ChunkEnricher] | None = None,
        document_store: DocumentStore | None = None,
        event_sink: EventSink | None = None,
        config: IngestionConfig | None = None,
    ) -> None:
        self.config = config or IngestionConfig()
        self.index = index
        self.parsers = parsers or ParserRegistry.with_defaults()
        self.cleaners = tuple(cleaners or (StandardCleaner(),))
        self.chunker = chunker or StructureAwareChunker(self.config)
        self.enrichers = tuple(enrichers or (MetadataEnricher(),))
        self.document_store = document_store
        self.event_sink = event_sink or NullEventSink()

    async def run(
        self,
        source: DocumentSource,
        *,
        namespace: str = "default",
        generation: int | None = None,
        continue_on_error: bool = True,
    ) -> IngestionReport:
        if generation is None:
            next_generation = getattr(self.index, "next_generation", None)
            if next_generation is None:
                raise ValueError("generation is required for index backends without next_generation()")
            generation = await next_generation(namespace)

        documents = [document async for document in source]
        await self._emit(
            "ingestion.started",
            {"namespace": namespace, "generation": generation, "document_count": len(documents)},
        )
        parser_semaphore = asyncio.Semaphore(self.config.parser_concurrency)
        failures: list[str] = []

        async def process(document: RawDocument) -> ParsedDocument | None:
            try:
                if len(document.content) > self.config.max_document_bytes:
                    raise ValueError(f"document exceeds {self.config.max_document_bytes} bytes")
                if self.document_store is not None:
                    await self.document_store.put(document)
                async with parser_semaphore:
                    parsed = await self.parsers.parse(document)
                for cleaner in self.cleaners:
                    parsed = cleaner.clean(parsed)
                if not parsed.text:
                    raise ValueError("document contains no extractable text")
                return parsed
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append(f"{document.id}: {type(exc).__name__}: {exc}")
                await self._emit(
                    "ingestion.document_failed",
                    {"document_id": document.id, "error_type": type(exc).__name__},
                )
                return None

        parsed_results = await asyncio.gather(*(process(document) for document in documents))
        deduplicator = DocumentDeduplicator()
        parsed_documents: list[ParsedDocument] = []
        duplicate_count = 0
        for parsed in parsed_results:
            if parsed is None:
                continue
            if deduplicator.is_duplicate(parsed):
                duplicate_count += 1
            else:
                parsed_documents.append(parsed)

        chunks = [chunk for document in parsed_documents for chunk in self.chunker.chunk(document)]
        unique_chunks: list[Chunk] = []
        chunk_fingerprints: set[str] = set()
        for chunk in chunks:
            scope = tuple((key, chunk.metadata.get(key)) for key in ("tenant", "organisation", "project", "workspace"))
            fingerprint = f"{chunk.checksum}:{chunk.acl!r}:{scope!r}"
            if fingerprint in chunk_fingerprints:
                continue
            chunk_fingerprints.add(fingerprint)
            unique_chunks.append(chunk)

        enrichment_semaphore = asyncio.Semaphore(self.config.enrichment_concurrency)

        async def enrich(chunk: Chunk) -> Chunk | None:
            try:
                async with enrichment_semaphore:
                    for enricher in self.enrichers:
                        chunk = await enricher.enrich(chunk)
                    return chunk
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append(f"{chunk.id}: {type(exc).__name__}: {exc}")
                return None

        enriched_results = await asyncio.gather(*(enrich(chunk) for chunk in unique_chunks))
        enriched = [chunk for chunk in enriched_results if chunk is not None]
        state = await self.index.publish(namespace, generation, enriched)
        report = IngestionReport(
            namespace=namespace,
            generation=generation,
            documents_seen=len(documents),
            documents_indexed=len(parsed_documents),
            documents_deduplicated=duplicate_count,
            chunks_indexed=len(enriched),
            failures=tuple(failures),
            state=state,
        )
        await self._emit(
            "ingestion.completed",
            {
                "namespace": namespace,
                "generation": generation,
                "documents_indexed": report.documents_indexed,
                "chunks_indexed": report.chunks_indexed,
                "failure_count": len(report.failures),
            },
        )
        return report

    async def _emit(self, event: str, payload: dict) -> None:
        try:
            await self.event_sink.emit(event, payload)
        except Exception:
            return
