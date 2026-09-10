from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from tech_ai.config.settings import ChromaSettings
from tech_ai.errors import KnowledgeStoreError
from tech_ai.integrations.chroma.knowledge_repository import ChromaKnowledgeRepository


class FakeCollection:
    def __init__(self, result: Mapping[str, object] | None = None) -> None:
        self.result = result or {
            "ids": [["id-1"]],
            "documents": [["Medical topic: Asma"]],
            "metadatas": [[{"focus": "Asma", "ignored": None}]],
            "distances": [[0.25]],
        }
        self.query_embeddings: Sequence[Sequence[float]] = ()
        self.n_results = 0

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str],
    ) -> Mapping[str, object]:
        self.query_embeddings = query_embeddings
        self.n_results = n_results
        assert include == ["documents", "metadatas", "distances"]
        return self.result


class BrokenCollection(FakeCollection):
    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str],
    ) -> Mapping[str, object]:
        raise RuntimeError("offline")


def test_query_normalizes_chroma_response() -> None:
    collection = FakeCollection()
    repository = ChromaKnowledgeRepository(ChromaSettings(), collection=collection)

    results = repository.query((0.1, 0.2), limit=4)

    assert collection.query_embeddings == [[0.1, 0.2]]
    assert collection.n_results == 4
    assert results[0].record_id == "id-1"
    assert results[0].metadata == {"focus": "Asma"}
    assert results[0].distance == 0.25


def test_query_translates_client_failure() -> None:
    repository = ChromaKnowledgeRepository(ChromaSettings(), collection=BrokenCollection())

    with pytest.raises(KnowledgeStoreError, match="consulta"):
        repository.query((0.1,), limit=1)


@pytest.mark.parametrize(
    "result",
    [
        {"ids": [], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        {
            "ids": [["id-1"]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        },
        {
            "ids": [[123]],
            "documents": [["texto"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        },
        {
            "ids": [["id-1"]],
            "documents": [["texto"]],
            "metadatas": [["invalid"]],
            "distances": [[0.1]],
        },
        {
            "ids": [["id-1"]],
            "documents": [["texto"]],
            "metadatas": [[{}]],
            "distances": [["invalid"]],
        },
    ],
)
def test_query_rejects_malformed_responses(result: Mapping[str, object]) -> None:
    repository = ChromaKnowledgeRepository(
        ChromaSettings(),
        collection=FakeCollection(result),
    )

    with pytest.raises(KnowledgeStoreError):
        repository.query((0.1,), limit=1)
