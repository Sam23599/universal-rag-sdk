"""Offline retrieval/answer evaluation and lightweight feedback storage."""

from __future__ import annotations

import time
from collections.abc import Sequence

from .models import EvaluationCase, EvaluationReport, Feedback
from .pipeline import RAGPipeline


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self._items: list[Feedback] = []

    async def add(self, feedback: Feedback) -> None:
        self._items.append(feedback)

    async def list(self) -> tuple[Feedback, ...]:
        return tuple(self._items)


class Evaluator:
    """Measures retrieval, citation, abstention, and end-to-end latency separately."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    async def evaluate(self, cases: Sequence[EvaluationCase], *, namespace: str = "default") -> EvaluationReport:
        if not cases:
            raise ValueError("At least one evaluation case is required")
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        abstention_matches = 0
        citation_scores: list[float] = []
        latencies: list[float] = []

        for case in cases:
            started = time.perf_counter()
            response = await self.pipeline.respond(
                case.query,
                namespace=namespace,
                filter_metadata=case.metadata_filters,
                principals=case.principals,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            retrieved = [str(hit.document.metadata.get("document_id") or hit.document.id) for hit in response.evidence]
            relevant = set(case.relevant_document_ids)
            recalls.append(len(relevant.intersection(retrieved)) / len(relevant) if relevant else 1.0)
            rank = next((index for index, value in enumerate(retrieved, start=1) if value in relevant), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            abstention_matches += int(response.abstained is (not case.should_answer))
            if response.citations:
                citation_scores.append(
                    sum(citation.document_id in relevant for citation in response.citations) / len(response.citations)
                    if relevant
                    else 0.0
                )
            else:
                citation_scores.append(1.0 if not case.should_answer else 0.0)

        count = len(cases)
        return EvaluationReport(
            case_count=count,
            recall_at_k=sum(recalls) / count,
            mean_reciprocal_rank=sum(reciprocal_ranks) / count,
            abstention_accuracy=abstention_matches / count,
            citation_precision=sum(citation_scores) / count,
            mean_latency_ms=sum(latencies) / count,
        )
