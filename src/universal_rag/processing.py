"""Cleaning, deduplication, structure-aware chunking, and enrichment."""

from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from .config import IngestionConfig
from .models import Chunk, ParsedDocument, ParsedSection


def content_fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


class StandardCleaner:
    def clean(self, document: ParsedDocument) -> ParsedDocument:
        sections = []
        for section in document.sections:
            text = unicodedata.normalize("NFKC", section.text).replace("\x00", "")
            text = "\n".join(line.rstrip() for line in text.splitlines())
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                sections.append(replace(section, text=text))
        return replace(document, sections=tuple(sections))


class DocumentDeduplicator:
    def __init__(
        self,
        *,
        scope_metadata_keys: Sequence[str] = ("tenant", "organisation", "project", "workspace"),
    ) -> None:
        self._seen: set[str] = set()
        self.scope_metadata_keys = tuple(scope_metadata_keys)

    def is_duplicate(self, document: ParsedDocument) -> bool:
        scope = tuple((key, document.metadata.get(key)) for key in self.scope_metadata_keys)
        fingerprint = content_fingerprint(f"{document.acl!r}:{scope!r}:{document.text}")
        if fingerprint in self._seen:
            return True
        self._seen.add(fingerprint)
        return False


class StructureAwareChunker:
    """Keeps headings/page provenance and avoids splitting table rows."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()

    def chunk(self, document: ParsedDocument) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for section_index, section in enumerate(document.sections):
            parts = self._split_section(section)
            for part_index, content in enumerate(parts):
                if len(content.strip()) < self.config.minimum_chunk_size and len(parts) > 1:
                    continue
                prefix = f"{section.heading}\n" if section.heading else ""
                grounding_content = f"{prefix}{content}".strip()
                index_text = grounding_content
                checksum = content_fingerprint(grounding_content)
                chunk_id = hashlib.sha256(
                    f"{document.id}:{section_index}:{part_index}:{checksum}".encode()
                ).hexdigest()[:24]
                metadata = {
                    **document.metadata,
                    **section.metadata,
                    "heading": section.heading,
                    "page_number": section.page_number,
                    "section_kind": section.kind,
                    "section_index": section_index,
                    "source_uri": document.source_uri,
                }
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        content=grounding_content,
                        index_text=index_text,
                        chunk_index=len(chunks),
                        metadata={key: value for key, value in metadata.items() if value is not None},
                        acl=document.acl,
                        checksum=checksum,
                    )
                )
        return chunks

    def _split_section(self, section: ParsedSection) -> list[str]:
        if len(section.text) <= self.config.chunk_size:
            return [section.text]
        if section.kind == "table" or self._looks_like_table(section.text):
            return self._split_table(section.text)
        return self._split_prose(section.text)

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        return bool(lines) and sum("|" in line for line in lines) / len(lines) >= 0.5

    def _split_table(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header = lines[:2] if len(lines) > 1 else []
        rows = lines[len(header) :]
        chunks: list[str] = []
        current = list(header)
        for row in rows:
            candidate = "\n".join((*current, row))
            if len(candidate) > self.config.chunk_size and len(current) > len(header):
                chunks.append("\n".join(current))
                current = [*header, row]
            else:
                current.append(row)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _split_prose(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.config.chunk_size)
            if end < len(text):
                search_start = start + self.config.minimum_chunk_size
                boundary = max(text.rfind("\n\n", search_start, end), text.rfind(". ", search_start, end))
                if boundary > start:
                    end = boundary + (1 if text[boundary] == "." else 0)
            content = text[start:end].strip()
            if content:
                chunks.append(content)
            if end >= len(text):
                break
            start = max(start + 1, end - self.config.chunk_overlap)
        return chunks


class MetadataEnricher:
    """Adds deterministic provenance to searchable text without changing grounding content."""

    async def enrich(self, chunk: Chunk) -> Chunk:
        fields = [
            chunk.metadata.get("heading"),
            chunk.metadata.get("title"),
            chunk.metadata.get("file_name"),
            *chunk.metadata.get("keywords", []),
        ]
        context = " | ".join(str(value) for value in fields if value)
        index_text = f"{context}\n{chunk.index_text}" if context else chunk.index_text
        return replace(chunk, index_text=index_text)


class CallableEnricher:
    def __init__(self, function: Callable[[Chunk], Chunk | Awaitable[Chunk]]) -> None:
        self._function = function

    async def enrich(self, chunk: Chunk) -> Chunk:
        value = self._function(chunk)
        return await value if inspect.isawaitable(value) else value
