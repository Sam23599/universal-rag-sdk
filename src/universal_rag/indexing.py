"""Versioned index backend and document storage reference implementations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any
from pathlib import Path

from .config import RetrievalConfig
from .models import Chunk, Document, IndexState, RawDocument, RetrievalResult
from .protocols import EmbeddingProvider, Reranker
from .retrieval import InMemoryHybridRetriever


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, RawDocument] = {}
        self._lock = asyncio.Lock()

    async def put(self, document: RawDocument) -> None:
        async with self._lock:
            self._documents[document.id] = document

    async def get(self, document_id: str) -> RawDocument | None:
        return self._documents.get(document_id)


class FileSystemDocumentStore:
    """Persistent, traversal-safe raw document store with atomic writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, document_id: str) -> tuple[Path, Path]:
        key = hashlib.sha256(document_id.encode()).hexdigest()
        return self.root / f"{key}.bin", self.root / f"{key}.json"

    async def put(self, document: RawDocument) -> None:
        data_path, metadata_path = self._paths(document.id)
        payload = {
            "id": document.id,
            "mime_type": document.mime_type,
            "source_uri": document.source_uri,
            "metadata": document.metadata,
            "acl": document.acl,
            "checksum": hashlib.sha256(document.content).hexdigest(),
        }

        def write() -> None:
            token = uuid.uuid4().hex
            temporary_data = self.root / f".{data_path.name}.{token}.tmp"
            temporary_metadata = self.root / f".{metadata_path.name}.{token}.tmp"
            try:
                temporary_data.write_bytes(document.content)
                temporary_metadata.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
                os.replace(temporary_data, data_path)
                os.replace(temporary_metadata, metadata_path)
            finally:
                temporary_data.unlink(missing_ok=True)
                temporary_metadata.unlink(missing_ok=True)

        await asyncio.to_thread(write)

    async def get(self, document_id: str) -> RawDocument | None:
        data_path, metadata_path = self._paths(document_id)
        if not data_path.is_file() or not metadata_path.is_file():
            return None

        def read() -> RawDocument:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("id") != document_id:
                raise ValueError("Stored document identity mismatch")
            content = data_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != payload.get("checksum"):
                raise ValueError("Stored document checksum mismatch")
            return RawDocument(
                id=document_id,
                content=content,
                mime_type=str(payload.get("mime_type") or "application/octet-stream"),
                source_uri=payload.get("source_uri"),
                metadata=dict(payload.get("metadata") or {}),
                acl=tuple(payload.get("acl") or ()),
            )

        return await asyncio.to_thread(read)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    state: IndexState
    retriever: InMemoryHybridRetriever | None


class InMemoryVersionedIndex:
    """Atomic immutable generations; suitable for tests and small deployments."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self._desired_generations: dict[str, int] = {}
        self._snapshots: dict[str, _Snapshot] = {}
        self._lock = asyncio.Lock()

    async def next_generation(self, namespace: str) -> int:
        async with self._lock:
            generation = self._desired_generations.get(namespace, 0) + 1
            self._desired_generations[namespace] = generation
            return generation

    async def publish(self, namespace: str, generation: int, chunks: Sequence[Chunk]) -> IndexState:
        desired = self._desired_generations.get(namespace, 0)
        if generation < desired:
            raise ValueError(f"Generation {generation} is stale; desired generation is {desired}")
        self._desired_generations[namespace] = generation

        documents = [
            Document(
                id=chunk.id,
                content=chunk.content,
                index_text=chunk.index_text,
                title=str(chunk.metadata.get("title") or chunk.metadata.get("file_name") or "Knowledge Base"),
                source_url=chunk.metadata.get("source_uri"),
                page_number=chunk.metadata.get("page_number"),
                metadata={**chunk.metadata, "document_id": chunk.document_id, "chunk_id": chunk.id},
                tags=tuple(chunk.metadata.get("tags") or ()),
                acl=chunk.acl,
            )
            for chunk in chunks
        ]
        retriever = (
            InMemoryHybridRetriever(documents, embedder=self.embedder, reranker=self.reranker, config=self.config)
            if documents
            else None
        )
        if retriever is not None:
            await retriever.prepare()
        digest = hashlib.sha256()
        for chunk in sorted(chunks, key=lambda item: item.id):
            digest.update(f"{chunk.id}:{chunk.checksum}".encode())
        state = IndexState(
            namespace=namespace,
            generation=generation,
            document_count=len({chunk.document_id for chunk in chunks}),
            chunk_count=len(chunks),
            checksum=digest.hexdigest(),
        )
        async with self._lock:
            if generation < self._desired_generations.get(namespace, 0):
                raise ValueError(f"Generation {generation} was superseded before publication")
            self._snapshots[namespace] = _Snapshot(state=state, retriever=retriever)
        return state

    def state(self, namespace: str = "default") -> IndexState | None:
        snapshot = self._snapshots.get(namespace)
        return snapshot.state if snapshot else None

    def desired_generation(self, namespace: str = "default") -> int:
        return self._desired_generations.get(namespace, 0)

    def published_generation(self, namespace: str = "default") -> int:
        state = self.state(namespace)
        return state.generation if state else 0

    def cache_identity(self, namespace: str = "default") -> str | None:
        state = self.state(namespace)
        return f"{namespace}:{state.generation}:{state.checksum}" if state else None

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RetrievalResult:
        snapshot = self._snapshots.get(namespace)
        if snapshot is None:
            return RetrievalResult(query=query, hits=(), mode="unpublished")
        if snapshot.retriever is None:
            return RetrievalResult(query=query, hits=(), mode="empty", generation=snapshot.state.generation)
        result = await snapshot.retriever.search(
            query,
            filter_metadata=filter_metadata,
            filter_tags=filter_tags,
            principals=principals,
        )
        return replace(result, generation=snapshot.state.generation)
