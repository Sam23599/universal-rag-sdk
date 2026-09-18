"""Portable in-memory hybrid retrieval with sparse fallback."""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .config import RetrievalConfig
from .models import Document, RetrievalResult, SearchHit
from .protocols import EmbeddingProvider, Reranker


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _lexical_coverage(query: str, document_tokens: Sequence[str]) -> float:
    query_terms = {token for token in _tokens(query) if token not in _STOPWORDS}
    if not query_terms:
        return 0.0
    return len(query_terms.intersection(document_tokens)) / len(query_terms)


class InMemoryHybridRetriever:
    """BM25 + optional dense retrieval, fused by weighted reciprocal rank."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        ids = [document.id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Document ids must be unique")

        self.documents = tuple(documents)
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self._document_vectors: tuple[tuple[float, ...], ...] | None = None
        self._vector_lock = asyncio.Lock()
        self._dense_warning: str | None = None

        self._document_tokens = tuple(_tokens(document.index_text or document.content) for document in self.documents)
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._document_tokens)
        self._average_length = sum(map(len, self._document_tokens)) / len(self._document_tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._document_tokens:
            document_frequency.update(set(tokens))
        count = len(self.documents)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    async def _ensure_vectors(self) -> tuple[tuple[float, ...], ...]:
        if self._document_vectors is not None:
            return self._document_vectors
        if self.embedder is None:
            return ()
        async with self._vector_lock:
            if self._document_vectors is None:
                vectors = await self.embedder.embed(
                    [document.index_text or document.content for document in self.documents]
                )
                if len(vectors) != len(self.documents):
                    raise ValueError("Embedding provider returned the wrong number of vectors")
                self._document_vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
        return self._document_vectors

    async def prepare(self) -> None:
        """Build dense artifacts eagerly; BM25 artifacts are built during construction."""
        if self.embedder is None or self._dense_warning is not None:
            return
        try:
            await self._ensure_vectors()
        except Exception as exc:
            self._dense_warning = f"Dense indexing unavailable; using sparse index ({type(exc).__name__})"

    @staticmethod
    def _matches(
        document: Document,
        metadata: dict[str, Any] | None,
        tags: Sequence[str] | None,
        principals: Sequence[str] | None,
    ) -> bool:
        if document.acl and not set(document.acl).intersection(principals or ()):
            return False
        if tags and not set(tags).issubset(document.tags):
            return False
        for key, expected in (metadata or {}).items():
            actual = document.metadata.get(key)
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _bm25(self, query: str, eligible: set[int]) -> list[tuple[int, float]]:
        query_terms = _tokens(query)
        scores: list[tuple[int, float]] = []
        k1, b = 1.5, 0.75
        for index in eligible:
            frequencies = self._term_frequencies[index]
            length = len(self._document_tokens[index])
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (k1 + 1)
                denominator = frequency + k1 * (1 - b + b * length / max(self._average_length, 1))
                score += self._idf.get(term, 0.0) * numerator / denominator
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: (-item[1], self.documents[item[0]].id))

    async def _dense(self, query: str, eligible: set[int]) -> list[tuple[int, float]]:
        vectors = await self._ensure_vectors()
        if not vectors or self.embedder is None:
            return []
        query_vectors = await self.embedder.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("Embedding provider must return one query vector")
        scores = [(index, _cosine(query_vectors[0], vectors[index])) for index in eligible]
        return sorted((item for item in scores if item[1] > 0), key=lambda item: (-item[1], self.documents[item[0]].id))

    async def search(
        self,
        query: str,
        *,
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RetrievalResult:
        del namespace
        clean_query = " ".join(query.split())
        if not clean_query:
            return RetrievalResult(query="", hits=(), mode="none")

        eligible = {
            index
            for index, document in enumerate(self.documents)
            if self._matches(document, filter_metadata, filter_tags, principals)
        }
        sparse = self._bm25(clean_query, eligible)[: self.config.candidate_k]
        dense: list[tuple[int, float]] = []
        warning = self._dense_warning
        if self.embedder is not None and self._dense_warning is None:
            try:
                dense = (await self._dense(clean_query, eligible))[: self.config.candidate_k]
            except Exception as exc:
                warning = f"Dense retrieval unavailable; used sparse fallback ({type(exc).__name__})"

        channels = [("sparse", sparse, self.config.sparse_weight)]
        if dense:
            channels.append(("dense", dense, self.config.dense_weight))
        channels = [channel for channel in channels if channel[1] and channel[2] > 0]
        if not channels:
            return RetrievalResult(query=clean_query, hits=(), mode="none", degraded=bool(warning), warning=warning)

        weight_total = sum(channel[2] for channel in channels)
        fused: defaultdict[int, float] = defaultdict(float)
        raw_scores: dict[str, dict[int, float]] = {"dense": {}, "sparse": {}}
        for name, ranked, weight in channels:
            normalized_weight = weight / weight_total
            raw_scores[name] = dict(ranked)
            for rank, (index, _) in enumerate(ranked, start=1):
                # Normalize RRF so a rank-one result present in every active channel scores 1.0.
                fused[index] += normalized_weight * (self.config.rrf_k + 1) / (self.config.rrf_k + rank)

        ranked_indexes = sorted(fused, key=lambda index: (-fused[index], self.documents[index].id))
        rerank_scores: dict[int, float] = {}
        if self.reranker is not None and ranked_indexes:
            candidates = ranked_indexes[: self.config.candidate_k]
            try:
                scores = await self.reranker.score(clean_query, [self.documents[index] for index in candidates])
                if len(scores) != len(candidates):
                    raise ValueError("Reranker returned the wrong number of scores")
                rerank_scores = {
                    index: max(0.0, min(1.0, float(score))) for index, score in zip(candidates, scores, strict=True)
                }
                ranked_indexes = sorted(
                    candidates,
                    key=lambda index: (-rerank_scores[index], self.documents[index].id),
                )
            except Exception as exc:
                detail = f"Reranker unavailable; used fused retrieval ({type(exc).__name__})"
                warning = f"{warning}; {detail}" if warning else detail

        mode = "hybrid" if dense and sparse else "dense" if dense else "bm25"
        hits = []
        for index in ranked_indexes[: self.config.top_k]:
            score = rerank_scores.get(index, fused[index])
            lexical_coverage = _lexical_coverage(clean_query, self._document_tokens[index])
            dense_score = raw_scores["dense"].get(index)
            if rerank_scores:
                grounding_eligible = score >= self.config.reranker_threshold
            else:
                has_relevance_signal = lexical_coverage >= self.config.minimum_lexical_coverage or (
                    dense_score is not None and dense_score >= self.config.minimum_dense_similarity
                )
                grounding_eligible = score >= self.config.grounding_threshold and has_relevance_signal
            hits.append(
                SearchHit(
                    document=self.documents[index],
                    score=score,
                    dense_score=dense_score,
                    sparse_score=raw_scores["sparse"].get(index),
                    lexical_coverage=lexical_coverage,
                    retrieval_mode="reranked" if rerank_scores else mode,
                    grounding_eligible=grounding_eligible,
                )
            )
        return RetrievalResult(
            query=clean_query,
            hits=tuple(hits),
            mode=mode,
            degraded=bool(warning),
            warning=warning,
        )
