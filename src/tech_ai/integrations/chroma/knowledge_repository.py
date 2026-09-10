"""Implementação ChromaDB do repositório de conhecimento."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import chromadb

from tech_ai.config.settings import ChromaSettings
from tech_ai.errors import KnowledgeStoreError
from tech_ai.models.knowledge import KnowledgeSearchResult, MetadataValue


class _ChromaCollection(Protocol):
    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str],
    ) -> Mapping[str, object]: ...


def _first_batch(result: Mapping[str, object], key: str) -> list[object]:
    value = result.get(key)
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise KnowledgeStoreError(f"Resposta inválida do ChromaDB no campo {key}.")
    return cast(list[object], value[0])


def _normalize_metadata(value: object) -> dict[str, MetadataValue]:
    if not isinstance(value, dict):
        raise KnowledgeStoreError("Resposta inválida do ChromaDB no campo metadatas.")
    normalized: dict[str, MetadataValue] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, (str, int, float, bool)):
            normalized[key] = item
    return normalized


class ChromaKnowledgeRepository:
    """Consulta embeddings prontos sem delegar geração de vetor ao ChromaDB."""

    def __init__(
        self,
        settings: ChromaSettings,
        *,
        collection: _ChromaCollection | None = None,
    ) -> None:
        if collection is not None:
            self._collection = collection
            return
        try:
            client = chromadb.HttpClient(
                host=settings.host,
                port=settings.port,
                ssl=settings.ssl,
            )
            chroma_collection = client.get_collection(
                name=settings.collection,
                embedding_function=None,
            )
        except Exception as error:
            raise KnowledgeStoreError(
                "Não foi possível acessar a coleção existente no ChromaDB."
            ) from error
        self._collection = cast(_ChromaCollection, chroma_collection)

    def query(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> tuple[KnowledgeSearchResult, ...]:
        try:
            result = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise KnowledgeStoreError("A consulta ao ChromaDB falhou.") from error

        ids = _first_batch(result, "ids")
        documents = _first_batch(result, "documents")
        metadatas = _first_batch(result, "metadatas")
        distances = _first_batch(result, "distances")
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise KnowledgeStoreError("O ChromaDB retornou colunas com tamanhos diferentes.")

        normalized: list[KnowledgeSearchResult] = []
        for record_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            if not isinstance(record_id, str) or not isinstance(document, str):
                raise KnowledgeStoreError("O ChromaDB retornou um registro sem texto ou ID.")
            if not isinstance(distance, (int, float)):
                raise KnowledgeStoreError("O ChromaDB retornou uma distância inválida.")
            normalized.append(
                KnowledgeSearchResult(
                    record_id=record_id,
                    text=document,
                    metadata=_normalize_metadata(metadata),
                    distance=float(distance),
                )
            )
        return tuple(normalized)
