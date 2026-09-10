"""Caso de uso de recuperação semântica do MedQuAD."""

from __future__ import annotations

from tech_ai.errors import KnowledgeRetrievalError
from tech_ai.integrations.embeddings.client import EmbeddingClient
from tech_ai.models.knowledge import KnowledgeSearchResult
from tech_ai.repositories.knowledge_repository import KnowledgeRepository


class KnowledgeRetrievalService:
    """Coordena embedding da pergunta e busca somente-leitura no ChromaDB."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        repository: KnowledgeRepository,
        *,
        max_distance: float | None = None,
    ) -> None:
        if max_distance is not None and max_distance <= 0:
            raise KnowledgeRetrievalError("max_distance deve ser maior que zero.")
        self._embedding_client = embedding_client
        self._repository = repository
        self._max_distance = max_distance

    def search(self, query: str, *, limit: int) -> tuple[KnowledgeSearchResult, ...]:
        normalized_query = query.strip()
        if not normalized_query:
            raise KnowledgeRetrievalError("A consulta semântica não pode ser vazia.")
        if limit <= 0:
            raise KnowledgeRetrievalError("limit deve ser maior que zero.")
        embeddings = self._embedding_client.embed([normalized_query])
        if len(embeddings) != 1:
            raise KnowledgeRetrievalError(
                "O provedor não retornou exatamente um embedding para a consulta."
            )
        results = self._repository.query(embeddings[0], limit=limit)
        if self._max_distance is None:
            return results
        return tuple(
            result for result in results if result.distance <= self._max_distance
        )
