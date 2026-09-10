from __future__ import annotations

from collections.abc import Sequence

import pytest

from tech_ai.errors import KnowledgeRetrievalError
from tech_ai.models.knowledge import KnowledgeSearchResult
from tech_ai.services.knowledge_retrieval_service import KnowledgeRetrievalService


class StubEmbeddingClient:
    def __init__(self, embeddings: tuple[tuple[float, ...], ...]) -> None:
        self.embeddings = embeddings
        self.received: Sequence[str] = ()

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.received = texts
        return self.embeddings


class StubKnowledgeRepository:
    def __init__(self, results: tuple[KnowledgeSearchResult, ...] = ()) -> None:
        self.results = results
        self.received_embedding: Sequence[float] = ()
        self.received_limit = 0

    def query(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> tuple[KnowledgeSearchResult, ...]:
        self.received_embedding = embedding
        self.received_limit = limit
        return self.results


def test_search_embeds_normalized_query_and_queries_repository() -> None:
    expected = (
        KnowledgeSearchResult("id-1", "texto", {"focus": "asma"}, 0.1),
    )
    embedding = StubEmbeddingClient(((0.1, 0.2),))
    repository = StubKnowledgeRepository(expected)

    result = KnowledgeRetrievalService(embedding, repository).search("  asma  ", limit=3)

    assert result == expected
    assert embedding.received == ["asma"]
    assert repository.received_embedding == (0.1, 0.2)
    assert repository.received_limit == 3


@pytest.mark.parametrize(("query", "limit"), [(" ", 1), ("asma", 0)])
def test_search_rejects_invalid_input(query: str, limit: int) -> None:
    service = KnowledgeRetrievalService(StubEmbeddingClient(((1.0,),)), StubKnowledgeRepository())

    with pytest.raises(KnowledgeRetrievalError):
        service.search(query, limit=limit)


def test_search_rejects_unexpected_embedding_count() -> None:
    service = KnowledgeRetrievalService(StubEmbeddingClient(()), StubKnowledgeRepository())

    with pytest.raises(KnowledgeRetrievalError, match="exatamente um"):
        service.search("asma", limit=1)


def test_search_filters_results_above_maximum_distance() -> None:
    results = (
        KnowledgeSearchResult("close", "texto", {}, 0.3),
        KnowledgeSearchResult("limit", "texto", {}, 0.55),
        KnowledgeSearchResult("far", "texto", {}, 0.56),
    )
    service = KnowledgeRetrievalService(
        StubEmbeddingClient(((1.0,),)),
        StubKnowledgeRepository(results),
        max_distance=0.55,
    )

    assert [result.record_id for result in service.search("asma", limit=3)] == [
        "close",
        "limit",
    ]


def test_retrieval_service_rejects_invalid_maximum_distance() -> None:
    with pytest.raises(KnowledgeRetrievalError, match="max_distance"):
        KnowledgeRetrievalService(
            StubEmbeddingClient(((1.0,),)),
            StubKnowledgeRepository(),
            max_distance=0,
        )
