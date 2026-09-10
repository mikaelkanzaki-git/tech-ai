"""Contrato do provedor de embeddings usado na consulta."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingClient(Protocol):
    """Gera vetores sem expor o SDK do provedor."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...
