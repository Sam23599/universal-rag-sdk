"""Parser registry with dependency-free text, HTML, CSV, and JSON support."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from io import StringIO

from .models import ParsedDocument, ParsedSection, RawDocument
from .protocols import DocumentParser


class UnsupportedDocumentType(ValueError):
    pass


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _sections_from_text(text: str) -> tuple[ParsedSection, ...]:
    heading = None
    parts: list[ParsedSection] = []
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            if any(value.strip() for value in buffer):
                parts.append(ParsedSection(text="\n".join(buffer).strip(), heading=heading))
            heading = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)
    if any(value.strip() for value in buffer):
        parts.append(ParsedSection(text="\n".join(buffer).strip(), heading=heading))
    return tuple(parts) or (ParsedSection(text=text.strip()),)


class TextParser:
    async def parse(self, document: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            id=document.id,
            sections=_sections_from_text(_decode(document.content)),
            source_uri=document.source_uri,
            metadata=document.metadata,
            acl=document.acl,
        )


class JsonParser:
    async def parse(self, document: RawDocument) -> ParsedDocument:
        value = json.loads(_decode(document.content))
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        return ParsedDocument(
            document.id,
            (ParsedSection(text=text, kind="structured"),),
            document.source_uri,
            document.metadata,
            document.acl,
        )


class CsvParser:
    async def parse(self, document: RawDocument) -> ParsedDocument:
        rows = list(csv.reader(StringIO(_decode(document.content))))
        text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows if any(cell.strip() for cell in row))
        return ParsedDocument(
            document.id, (ParsedSection(text=text, kind="table"),), document.source_uri, document.metadata, document.acl
        )


class _HTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: list[ParsedSection] = []
        self._heading: str | None = None
        self._parts: list[str] = []
        self._skip = 0
        self._heading_depth = 0
        self._table_depth = 0

    def _flush(self, kind: str = "prose") -> None:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip(" |") for line in raw.splitlines()]
        text = "\n".join(line for line in lines if line).strip()
        if text:
            self.sections.append(ParsedSection(text=text, heading=self._heading, kind=kind))
        self._parts = []

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._heading_depth += 1
        elif tag == "table":
            self._flush()
            self._table_depth += 1
        elif tag in {"p", "div", "li", "tr", "br"}:
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_depth:
            self._heading = " ".join(self._parts).strip() or self._heading
            self._parts = []
            self._heading_depth -= 1
        elif tag == "table" and self._table_depth:
            self._flush("table")
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)


class HtmlParser:
    async def parse(self, document: RawDocument) -> ParsedDocument:
        parser = _HTMLExtractor()
        parser.feed(_decode(document.content))
        parser._flush("table" if parser._table_depth else "prose")
        return ParsedDocument(document.id, tuple(parser.sections), document.source_uri, document.metadata, document.acl)


class PdfParser:
    """PDF parser with an injectable OCR fallback per page."""

    def __init__(self, ocr: Callable[[bytes, int], str] | None = None, *, minimum_page_characters: int = 40) -> None:
        self.ocr = ocr
        self.minimum_page_characters = minimum_page_characters

    async def parse(self, document: RawDocument) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF parsing requires pypdf. Reinstall with: pip install universal-rag-sdk") from exc

        def extract() -> list[str]:
            from io import BytesIO

            reader = PdfReader(BytesIO(document.content))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")
            return [(page.extract_text() or "").strip() for page in reader.pages]

        pages = await asyncio.to_thread(extract)
        sections = []
        for page_number, text in enumerate(pages, start=1):
            used_ocr = False
            if len(re.sub(r"\W", "", text)) < self.minimum_page_characters and self.ocr is not None:
                text = await asyncio.to_thread(self.ocr, document.content, page_number)
                used_ocr = True
            if text.strip():
                sections.append(
                    ParsedSection(text=text.strip(), page_number=page_number, metadata={"used_ocr": used_ocr})
                )
        return ParsedDocument(document.id, tuple(sections), document.source_uri, document.metadata, document.acl)


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}

    @classmethod
    def with_defaults(cls) -> ParserRegistry:
        registry = cls()
        registry.register(("text/plain", "text/markdown"), TextParser())
        registry.register(("text/html", "application/xhtml+xml"), HtmlParser())
        registry.register(("text/csv",), CsvParser())
        registry.register(("application/json",), JsonParser())
        registry.register(("application/pdf",), PdfParser())
        return registry

    def register(self, mime_types: Sequence[str], parser: DocumentParser) -> None:
        for mime_type in mime_types:
            self._parsers[mime_type.casefold().split(";", 1)[0]] = parser

    async def parse(self, document: RawDocument) -> ParsedDocument:
        mime_type = document.mime_type.casefold().split(";", 1)[0]
        parser = self._parsers.get(mime_type)
        if parser is None:
            raise UnsupportedDocumentType(f"No parser registered for {document.mime_type}")
        return await parser.parse(document)
