"""Online query-to-answer pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

from .cache import AsyncTTLCache
from .config import PipelineConfig
from .context import ContextBuilder
from .defaults import IdentityRedactor, LatestTurnQueryPlanner, NullEventSink, RegexPIIRedactor
from .models import Document, QueryContext, RAGResponse, RetrievalResult, TranscriptTurn
from .prompting import GroundedPromptBuilder, GroundingValidator, InvalidLLMResponse, NO_EVIDENCE, build_citations
from .protocols import (
    EmbeddingProvider,
    EventSink,
    LLMProvider,
    QueryPlanner,
    QueryProcessor,
    Redactor,
    Reranker,
    Retriever,
)
from .query import NormalizeQuery, QueryProcessorChain
from .retrieval import InMemoryHybridRetriever


class RAGPipeline:
    """Facade coordinating query planning, retrieval, grounding, and generation."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        config: PipelineConfig | None = None,
        query_planner: QueryPlanner | None = None,
        redactor: Redactor | None = None,
        event_sink: EventSink | None = None,
        prompt_builder: GroundedPromptBuilder | None = None,
        query_processors: Sequence[QueryProcessor] | None = None,
        context_builder: ContextBuilder | None = None,
        grounding_validator: GroundingValidator | None = None,
        cache: AsyncTTLCache[RAGResponse] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = retriever
        self.llm = llm
        self.query_planner = query_planner or LatestTurnQueryPlanner()
        self.redactor = redactor or (RegexPIIRedactor() if self.config.redact_pii else IdentityRedactor())
        self.event_sink = event_sink or NullEventSink()
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()
        self.query_processors = QueryProcessorChain(query_processors or (NormalizeQuery(),))
        self.context_builder = context_builder or ContextBuilder(self.prompt_builder)
        self.grounding_validator = grounding_validator or GroundingValidator()
        self.cache = cache

    def new_session(self) -> RAGSession:
        return RAGSession(self)

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        llm: LLMProvider,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        config: PipelineConfig | None = None,
        **pipeline_strategies: Any,
    ) -> RAGPipeline:
        """Convenience factory that keeps pipeline and retrieval configuration aligned."""
        effective_config = config or PipelineConfig()
        retriever = InMemoryHybridRetriever(
            documents,
            embedder=embedder,
            reranker=reranker,
            config=effective_config.retrieval,
        )
        return cls(retriever=retriever, llm=llm, config=effective_config, **pipeline_strategies)

    async def respond(
        self,
        transcript: str | Sequence[TranscriptTurn],
        *,
        speaker: str = "caller",
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RAGResponse:
        turns = self._normalize_turns(transcript, speaker)
        safe_turns = tuple(TranscriptTurn(self.redactor.redact(turn.text), turn.speaker) for turn in turns)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        raw_query = await self.query_planner.build_query(safe_turns)
        query_context = await self.query_processors.process(
            QueryContext(
                query=raw_query,
                history=safe_turns,
                metadata_filters=dict(filter_metadata or {}),
                tags=tuple(filter_tags or ()),
            )
        )
        query = query_context.query
        timings["query_planning"] = self._elapsed(started)
        await self._emit("query.planned", {"query": query})

        cache_key = self._cache_key(
            namespace=namespace,
            turns=safe_turns,
            query=query,
            filter_metadata=query_context.metadata_filters,
            filter_tags=query_context.tags,
            principals=principals or (),
        )
        if self.cache is not None and cache_key is not None:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.config.retrieval.timeout_seconds):
                retrieval = await self.retriever.search(
                    query,
                    namespace=namespace,
                    filter_metadata=query_context.metadata_filters,
                    filter_tags=query_context.tags,
                    principals=principals,
                )
        except TimeoutError:
            retrieval = RetrievalResult(
                query=query,
                hits=(),
                mode="timeout",
                degraded=True,
                warning="Retrieval timed out",
            )
        except Exception as exc:
            retrieval = RetrievalResult(
                query=query,
                hits=(),
                mode="error",
                degraded=True,
                warning=f"Retrieval unavailable ({type(exc).__name__})",
            )
        timings["retrieval"] = self._elapsed(started)
        await self._emit(
            "retrieval.completed",
            {"mode": retrieval.mode, "hit_count": len(retrieval.hits), "degraded": retrieval.degraded},
        )

        evidence = tuple(hit for hit in retrieval.hits if hit.grounding_eligible)
        context, evidence_by_id, evidence = self.context_builder.build(
            evidence,
            max_characters=self.config.max_context_characters,
        )
        if not evidence_by_id and self.config.abstain_on_no_evidence:
            response = RAGResponse(
                answer=self.config.no_evidence_message,
                confidence=1.0,
                reasoning="No retrieval result passed the grounding threshold.",
                citations=(),
                query=query,
                evidence=evidence,
                retrieval_mode=retrieval.mode,
                degraded=retrieval.degraded,
                retrieval_warning=retrieval.warning,
                abstained=True,
                timings_ms=timings,
            )
            if self.cache is not None and cache_key is not None:
                await self.cache.set(cache_key, response)
            return response

        system_prompt, user_prompt = self.prompt_builder.build(
            turns=safe_turns[-self.config.max_history_turns :],
            evidence_context=context or NO_EVIDENCE,
            system_prompt=self.config.system_prompt,
        )
        started = time.perf_counter()
        async with asyncio.timeout(self.config.llm_timeout_seconds):
            raw = await self.llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        timings["generation"] = self._elapsed(started)
        parsed = self.prompt_builder.parse(raw, set(evidence_by_id))
        if self.config.require_citations and evidence_by_id and not parsed.citation_ids:
            raise InvalidLLMResponse("LLM response must cite at least one supplied evidence item")
        self.grounding_validator.validate(parsed, evidence_by_id)
        await self._emit(
            "response.generated",
            {"confidence": parsed.confidence, "citation_count": len(parsed.citation_ids)},
        )
        response = RAGResponse(
            answer=parsed.answer,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            citations=build_citations(parsed, evidence_by_id),
            query=query,
            evidence=evidence,
            retrieval_mode=retrieval.mode,
            degraded=retrieval.degraded,
            retrieval_warning=retrieval.warning,
            abstained=False,
            timings_ms=timings,
        )
        if self.cache is not None and cache_key is not None:
            await self.cache.set(cache_key, response)
        return response

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    async def _emit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            await self.event_sink.emit(event, payload)
        except Exception:
            # Observability is best-effort and must not break answer generation.
            return

    def _cache_key(
        self,
        *,
        namespace: str,
        turns: Sequence[TranscriptTurn],
        query: str,
        filter_metadata: dict[str, Any],
        filter_tags: Sequence[str],
        principals: Sequence[str],
    ) -> str | None:
        identity_provider = getattr(self.retriever, "cache_identity", None)
        identity = identity_provider(namespace) if identity_provider else None
        if not identity:
            return None
        payload = {
            "index": identity,
            "query": query,
            "turns": [(turn.speaker, turn.text) for turn in turns[-self.config.max_history_turns :]],
            "metadata": filter_metadata,
            "tags": sorted(filter_tags),
            "principals": sorted(principals),
            "system_prompt": self.config.system_prompt,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _normalize_turns(transcript: str | Sequence[TranscriptTurn], speaker: str) -> tuple[TranscriptTurn, ...]:
        if isinstance(transcript, str):
            return (TranscriptTurn(transcript, speaker),)
        turns = tuple(transcript)
        if not turns:
            raise ValueError("Transcript cannot be empty")
        return turns


class RAGSession:
    """Stateful session wrapper; the underlying pipeline remains reusable."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._turns: list[TranscriptTurn] = []

    @property
    def turns(self) -> tuple[TranscriptTurn, ...]:
        return tuple(self._turns)

    def add_turn(self, text: str, *, speaker: str) -> None:
        self._turns.append(TranscriptTurn(text=text, speaker=speaker))

    async def respond(
        self,
        text: str,
        *,
        speaker: str = "caller",
        namespace: str = "default",
        filter_metadata: dict[str, Any] | None = None,
        filter_tags: Sequence[str] | None = None,
        principals: Sequence[str] | None = None,
    ) -> RAGResponse | None:
        self.add_turn(text, speaker=speaker)
        if speaker not in self._pipeline.config.answer_on_speakers:
            return None
        return await self._pipeline.respond(
            self._turns,
            namespace=namespace,
            filter_metadata=filter_metadata,
            filter_tags=filter_tags,
            principals=principals,
        )
