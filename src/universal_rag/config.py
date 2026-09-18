"""Configuration with safe, useful defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    grounding_threshold: float = 0.35
    reranker_threshold: float = 0.5
    minimum_lexical_coverage: float = 0.25
    minimum_dense_similarity: float = 0.35
    dense_weight: float = 0.55
    sparse_weight: float = 0.45
    rrf_k: int = 60
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.dense_weight < 0 or self.sparse_weight < 0 or self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive")
        thresholds = (
            self.grounding_threshold,
            self.reranker_threshold,
            self.minimum_lexical_coverage,
            self.minimum_dense_similarity,
        )
        if any(not 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("Retrieval thresholds must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    max_history_turns: int = 10
    max_context_characters: int = 6000
    llm_timeout_seconds: float = 15.0
    answer_on_speakers: tuple[str, ...] = ("caller", "unknown")
    redact_pii: bool = True
    abstain_on_no_evidence: bool = True
    require_citations: bool = True
    no_evidence_message: str = "I don't have enough verified information in the knowledge base to answer that."
    system_prompt: str = (
        "You are a real-time assistant helping a customer-service agent decide what to say next. "
        "Answer concisely and empathetically. Use only the supplied knowledge-base evidence for factual claims. "
        "Never invent policies, prices, account details, or completed actions."
    )

    def __post_init__(self) -> None:
        if self.max_history_turns < 1 or self.max_context_characters < 200:
            raise ValueError("History must be positive and context budget must be at least 200 characters")


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 150
    minimum_chunk_size: int = 80
    max_document_bytes: int = 25_000_000
    parser_concurrency: int = 4
    enrichment_concurrency: int = 8

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if self.minimum_chunk_size < 1:
            raise ValueError("minimum_chunk_size must be positive")
        if self.max_document_bytes < 1 or self.parser_concurrency < 1 or self.enrichment_concurrency < 1:
            raise ValueError("Ingestion limits and concurrency must be positive")
