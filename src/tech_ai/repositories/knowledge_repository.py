"""Contrato somente-leitura da base de conhecimento."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tech_ai.models.knowledge import KnowledgeSearchResult


class KnowledgeRepository(Protocol):
    """Consulta vetorial necessária pelo fluxo RAG."""

    def query(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> tuple[KnowledgeSearchResult, ...]: ...
