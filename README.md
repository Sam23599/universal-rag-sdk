# Universal RAG SDK

A modular, provider-neutral Python library for complete retrieval-augmented generation:

`sources -> parse/OCR -> clean/deduplicate -> chunk -> enrich -> embed/index -> preprocess query -> retrieve -> filter/rerank -> build context -> generate -> validate/cite -> evaluate`

The package is application-independent. FastAPI, MongoDB, Redis, Celery, FAISS, and any specific LLM vendor are not
required.

## What is included

| Stage | Built-in implementation | Extension point |
|---|---|---|
| Sources | memory, directory, sync/async callable | `DocumentSource` for APIs, databases, object stores, queues |
| Parse | TXT/Markdown, HTML, CSV, JSON, optional PDF | `DocumentParser`; injectable page OCR |
| Clean | Unicode normalization, whitespace cleanup | `DocumentCleaner` chain |
| Deduplicate | deterministic document and chunk fingerprints | replaceable deduplication policy |
| Chunk | heading/page-aware prose and table-row chunking | `Chunker` |
| Enrich | provenance, headings, titles, keywords in `index_text` | `ChunkEnricher`, including LLM/entity enrichers |
| Store | in-memory and atomic local-filesystem document stores | `DocumentStore` for S3, Blob, database |
| Index | immutable, atomic in-memory generations | `IndexBackend` for vector DB/search services |
| Query processing | normalization, expansion, LLM rewrite, callable classification | `QueryProcessor` chain |
| Retrieve | BM25 or dense+BM25 weighted RRF | `Retriever` |
| Filter | exact metadata, tags, ACL principals before ranking | backend-specific authorization policies |
| Rerank | optional provider strategy | `Reranker`, such as a cross-encoder |
| Context | near-duplicate removal, relevance ordering, budget, prompt fencing | `ContextBuilder` |
| Generate | vendor-neutral LLM contract, timeout, PII redaction | `LLMProvider`, `Redactor` |
| Validate | strict JSON, citations, abstention, numeric-claim evidence checks | custom strategy |
| Operate | version-aware TTL cache, retries, events, feedback | distributed cache, metrics/tracing sinks |
| Evaluate | Recall@K, MRR, abstention, citation precision, latency | domain answer evals |

Every component is public and can be used independently. `IngestionPipeline`, `RAGPipeline`, and `RAG` are optional
orchestrators.

## Install

```bash
pip install -e ./rag_sdk
```

Core functionality has no runtime dependencies and supports Python 3.11+. PDF parsing is optional:

```bash
pip install -e './rag_sdk[pdf]'
```

## Complete pipeline

```python
from universal_rag import (
    CallableLLMProvider,
    InMemoryVersionedIndex,
    IngestionPipeline,
    IterableSource,
    RAG,
    RAGPipeline,
    RawDocument,
)


async def call_llm(*, system_prompt: str, user_prompt: str) -> str:
    # Return the JSON response requested by the prompt.
    return await llm_client.complete(system=system_prompt, user=user_prompt)


index = InMemoryVersionedIndex()
rag = RAG(
    ingestion=IngestionPipeline(index=index),
    query=RAGPipeline(retriever=index, llm=CallableLLMProvider(call_llm)),
)

source = IterableSource(
    [
        RawDocument(
            id="benefits-2026",
            content=b"# Gold plan\nThe annual employee-only deductible is $500.",
            mime_type="text/markdown",
            source_uri="https://docs.example.test/benefits.md",
            metadata={"tenant": "acme", "year": 2026, "tags": ["benefits"]},
            acl=("benefits-team",),
        )
    ]
)

report = await rag.ingest(source, namespace="acme-benefits")
response = await rag.ask(
    "What is the Gold plan deductible?",
    namespace="acme-benefits",
    filter_metadata={"tenant": "acme", "year": 2026},
    filter_tags=["benefits"],
    principals=["benefits-team"],
)
```

An empty later ingestion generation is still published. This prevents deleted documents from surviving in an old
index snapshot.

## Use individual features

### Parse only

```python
from universal_rag import ParserRegistry, RawDocument

parsed = await ParserRegistry.with_defaults().parse(
    RawDocument(id="rates", content=html_bytes, mime_type="text/html")
)
```

### Chunk only

```python
from universal_rag import IngestionConfig, StructureAwareChunker

chunker = StructureAwareChunker(IngestionConfig(chunk_size=1000, chunk_overlap=120))
chunks = chunker.chunk(parsed)
```

### Retrieve only

```python
from universal_rag import Document, InMemoryHybridRetriever

retriever = InMemoryHybridRetriever(
    [Document(id="1", content="Grounding text", metadata={"tenant": "acme"})]
)
result = await retriever.search("grounding", filter_metadata={"tenant": "acme"})
```

### Custom API or database source

```python
from universal_rag import CallableSource, RawDocument

async def load_rows():
    rows = await database.fetch_documents()
    return [
        RawDocument(
            id=str(row.id),
            content=row.body.encode(),
            mime_type="text/plain",
            metadata={"tenant": row.tenant_id},
        )
        for row in rows
    ]

source = CallableSource(load_rows)
```

### Dense retrieval and reranking

Pass an `EmbeddingProvider` to `InMemoryVersionedIndex` or `InMemoryHybridRetriever`. Pass a `Reranker` to add a
bounded second-stage cross-encoder. Dense failures serve BM25 results; reranker failures serve fused results and mark
the retrieval degraded.

### Query preprocessing

```python
from universal_rag import NormalizeQuery, StaticQueryExpansion

processors = [
    NormalizeQuery(),
    StaticQueryExpansion({"pto": ["paid time off", "vacation"]}),
]

pipeline = RAGPipeline(..., query_processors=processors)
```

`LLMQueryRewriter` and `CallableQueryProcessor` support conversational rewriting and project-specific intent
classification. Filter values should come from authenticated application context, not be inferred from user text.

## Production adapters

The bundled in-memory stores are reference implementations for tests, local applications, and small corpora. Larger
systems should implement the same ports for durable infrastructure:

- `DocumentStore`: S3, GCS, Azure Blob, database, encrypted filesystem.
- `IndexBackend` / `Retriever`: OpenSearch, pgvector, Qdrant, Pinecone, Weaviate, MongoDB Atlas, or another service.
- `EmbeddingProvider`, `Reranker`, `LLMProvider`: any local or hosted model.
- `EventSink`: OpenTelemetry, Prometheus, Datadog, CloudWatch, or structured logs.
- `AsyncTTLCache`: replace with a distributed generation-aware cache where multiple processes serve queries.
- Durable job runner: schedule `IngestionPipeline.run(...)` with the project's worker/queue system.

Index adapters should retain desired and published generations separately, build immutable snapshots, atomically
publish only the desired generation, and include the generation in cache identity. ACL and tenant filters must be
applied before ranking or content hydration.

## Safety defaults

- Retrieved and transcript text are fenced as untrusted data.
- Metadata filters are exact: fields are ANDed; collection values are alternatives within one field.
- Documents with ACLs are invisible unless a supplied principal matches.
- Rank position alone cannot make evidence eligible; lexical coverage, dense similarity, or reranker confidence must
  also pass its configured gate.
- Only eligible evidence reaches generation.
- No evidence returns a deterministic abstention without calling the LLM.
- Grounded answers require citations, and unknown citation IDs are rejected.
- Numeric values in answers must occur in the evidence actually cited.
- Query cache keys include ACL principals, filters, conversation, configuration, and immutable index identity.

## Evaluation

`Evaluator` accepts labeled `EvaluationCase` objects and reports Recall@K, MRR, abstention accuracy, citation
precision, and end-to-end latency. Keep retrieval evaluation separate from answer-quality or model-judge evaluation;
projects can add domain-specific exact-value, groundedness, hallucination, and source-contamination checks using the
returned evidence and citations.

## Validation

```bash
PYTHONPATH=src python -m pytest -q
ruff check src tests
```
