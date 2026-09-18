import json

import pytest

from universal_rag import (
    AsyncTTLCache,
    EvaluationCase,
    Evaluator,
    FileSystemDocumentStore,
    HtmlParser,
    IngestionPipeline,
    IterableSource,
    ParserRegistry,
    RAG,
    RAGPipeline,
    RawDocument,
    StructureAwareChunker,
    InMemoryVersionedIndex,
)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "answer": "The annual Gold deductible is $500.",
                "confidence": 0.95,
                "reasoning": "The indexed benefits guide states the value.",
                "citations": ["KB1"],
            }
        )


def source():
    content = b"""# Gold Medical Plan

For employee-only coverage, the annual Gold deductible is $500.

| Tier | Deductible |
| Employee only | $500 |
"""
    return IterableSource(
        [
            RawDocument(
                id="benefits-2026",
                content=content,
                mime_type="text/markdown",
                source_uri="https://example.test/benefits.md",
                metadata={"title": "2026 Benefits", "tenant": "acme", "tags": ["benefits"]},
                acl=("benefits-team",),
            ),
            RawDocument(
                id="duplicate",
                content=content,
                mime_type="text/markdown",
                metadata={"title": "2026 Benefits", "tenant": "acme", "tags": ["benefits"]},
                acl=("benefits-team",),
            ),
        ]
    )


@pytest.mark.asyncio
async def test_full_ingest_to_acl_scoped_cited_answer():
    index = InMemoryVersionedIndex()
    ingestion = IngestionPipeline(index=index)
    llm = FakeLLM()
    query = RAGPipeline(retriever=index, llm=llm)
    rag = RAG(ingestion=ingestion, query=query)

    report = await rag.ingest(source(), namespace="benefits")
    denied = await rag.ask("What is the Gold deductible?", namespace="benefits")
    answer = await rag.ask(
        "What is the Gold deductible?",
        namespace="benefits",
        principals=["benefits-team"],
        filter_metadata={"tenant": "acme"},
    )

    assert report.documents_seen == 2
    assert report.documents_indexed == 1
    assert report.documents_deduplicated == 1
    assert report.chunks_indexed >= 1
    assert denied.abstained is True
    assert answer.abstained is False
    assert answer.citations[0].document_id == "benefits-2026"
    assert answer.citations[0].chunk_id
    assert answer.evidence[0].document.metadata["heading"] == "Gold Medical Plan"


@pytest.mark.asyncio
async def test_deletion_only_generation_publishes_empty_index():
    index = InMemoryVersionedIndex()
    ingestion = IngestionPipeline(index=index)
    await ingestion.run(source(), namespace="benefits")

    report = await ingestion.run(IterableSource([]), namespace="benefits")
    result = await index.search("deductible", namespace="benefits", principals=["benefits-team"])

    assert report.generation == 2
    assert report.chunks_indexed == 0
    assert result.mode == "empty"
    assert result.generation == 2


@pytest.mark.asyncio
async def test_query_cache_isolated_by_index_generation():
    index = InMemoryVersionedIndex()
    ingestion = IngestionPipeline(index=index)
    await ingestion.run(source(), namespace="benefits")
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=index, llm=llm, cache=AsyncTTLCache())

    kwargs = {"namespace": "benefits", "principals": ["benefits-team"]}
    await pipeline.respond("Gold deductible", **kwargs)
    await pipeline.respond("Gold deductible", **kwargs)
    assert llm.calls == 1

    await ingestion.run(source(), namespace="benefits")
    await pipeline.respond("Gold deductible", **kwargs)
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_html_parser_preserves_heading_and_table_structure():
    document = RawDocument(
        id="rates",
        mime_type="text/html",
        content=b"<h2>Rates</h2><table><tr><th>Tier</th><th>Rate</th></tr><tr><td>Family</td><td>$20</td></tr></table>",
    )

    parsed = await HtmlParser().parse(document)
    chunks = StructureAwareChunker().chunk(parsed)

    assert parsed.sections[0].heading == "Rates"
    assert "Family" in parsed.sections[0].text
    assert chunks[0].content.startswith("Rates\n")


@pytest.mark.asyncio
async def test_evaluator_reports_retrieval_and_citation_metrics():
    index = InMemoryVersionedIndex()
    await IngestionPipeline(index=index).run(source(), namespace="benefits")
    evaluator = Evaluator(RAGPipeline(retriever=index, llm=FakeLLM()))

    report = await evaluator.evaluate(
        [
            EvaluationCase(
                id="deductible",
                query="Gold deductible",
                relevant_document_ids=("benefits-2026",),
                principals=("benefits-team",),
            )
        ],
        namespace="benefits",
    )

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.citation_precision == 1.0


@pytest.mark.asyncio
async def test_parser_registry_can_be_extended_independently():
    registry = ParserRegistry.with_defaults()
    document = RawDocument(id="data", content=b'{"plan": "Gold", "deductible": 500}', mime_type="application/json")

    parsed = await registry.parse(document)

    assert '"deductible": 500' in parsed.text


@pytest.mark.asyncio
async def test_deduplication_never_collapses_different_acl_scopes():
    content = b"Restricted policy number 42."
    source_with_scopes = IterableSource(
        [
            RawDocument(id="team-a", content=content, acl=("team-a",)),
            RawDocument(id="team-b", content=content, acl=("team-b",)),
        ]
    )
    index = InMemoryVersionedIndex()

    report = await IngestionPipeline(index=index).run(source_with_scopes)
    team_a = await index.search("policy number 42", principals=["team-a"])
    team_b = await index.search("policy number 42", principals=["team-b"])

    assert report.documents_indexed == 2
    assert report.chunks_indexed == 2
    assert {hit.document.metadata["document_id"] for hit in team_a.hits} == {"team-a"}
    assert {hit.document.metadata["document_id"] for hit in team_b.hits} == {"team-b"}


@pytest.mark.asyncio
async def test_filesystem_document_store_round_trip(tmp_path):
    store = FileSystemDocumentStore(tmp_path)
    document = RawDocument(
        id="../../unsafe/id",
        content=b"source bytes",
        mime_type="application/octet-stream",
        metadata={"tenant": "acme"},
        acl=("team-a",),
    )

    await store.put(document)
    loaded = await store.get(document.id)

    assert loaded == document
    assert all(path.parent == tmp_path for path in tmp_path.iterdir())
