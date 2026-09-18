"""Public data contracts for the RAG SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Speaker = Literal["agent", "caller", "system", "unknown"]


@dataclass(frozen=True, slots=True)
class RawDocument:
    id: str
    content: bytes
    mime_type: str = "text/plain"
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    acl: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Raw document id cannot be empty")


@dataclass(frozen=True, slots=True)
class ParsedSection:
    text: str
    heading: str | None = None
    page_number: int | None = None
    kind: str = "prose"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    id: str
    sections: tuple[ParsedSection, ...]
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    acl: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Parsed document id cannot be empty")

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    content: str
    index_text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    acl: tuple[str, ...] = ()
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.document_id.strip() or not self.content.strip():
            raise ValueError("Chunk id, document id, and content are required")


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    text: str
    speaker: Speaker = "unknown"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Transcript text cannot be empty")


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    content: str
    index_text: str | None = None
    title: str = "Knowledge Base"
    source_url: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    acl: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Document id cannot be empty")
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: Document
    score: float
    retrieval_mode: str
    grounding_eligible: bool
    dense_score: float | None = None
    sparse_score: float | None = None
    lexical_coverage: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    hits: tuple[SearchHit, ...]
    mode: str
    degraded: bool = False
    warning: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class IndexState:
    namespace: str
    generation: int
    document_count: int
    chunk_count: int
    checksum: str


@dataclass(frozen=True, slots=True)
class IngestionReport:
    namespace: str
    generation: int
    documents_seen: int
    documents_indexed: int
    documents_deduplicated: int
    chunks_indexed: int
    failures: tuple[str, ...]
    state: IndexState | None


@dataclass(frozen=True, slots=True)
class QueryContext:
    query: str
    history: tuple[TranscriptTurn, ...] = ()
    classification: str | None = None
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    relevant_document_ids: tuple[str, ...] = ()
    should_answer: bool = True
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    principals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    abstention_accuracy: float
    citation_precision: float
    mean_latency_ms: float


@dataclass(frozen=True, slots=True)
class Feedback:
    query: str
    answer: str
    rating: float
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    id: str
    document_id: str
    title: str
    chunk_id: str | None = None
    source_url: str | None = None
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    confidence: float
    reasoning: str
    citations: tuple[Citation, ...]
    query: str
    evidence: tuple[SearchHit, ...]
    retrieval_mode: str
    degraded: bool
    retrieval_warning: str | None
    abstained: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
