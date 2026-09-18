import pytest

from universal_rag import CallableEmbeddingProvider, Document, InMemoryHybridRetriever, RetrievalConfig


@pytest.mark.asyncio
async def test_hybrid_retrieval_fuses_dense_and_sparse_rankings():
    vectors = {
        "Alpha plan covers emergency care.": [1.0, 0.0],
        "Beta plan covers routine care.": [0.0, 1.0],
        "Which plan covers emergency care?": [1.0, 0.0],
    }

    async def embed(texts):
        return [vectors[text] for text in texts]

    retriever = InMemoryHybridRetriever(
        [
            Document(id="alpha", content="Alpha plan covers emergency care."),
            Document(id="beta", content="Beta plan covers routine care."),
        ],
        embedder=CallableEmbeddingProvider(embed),
        config=RetrievalConfig(top_k=2, candidate_k=2),
    )

    result = await retriever.search("Which plan covers emergency care?")

    assert result.mode == "hybrid"
    assert result.hits[0].document.id == "alpha"
    assert result.hits[0].dense_score == pytest.approx(1.0)
    assert result.hits[0].sparse_score is not None
