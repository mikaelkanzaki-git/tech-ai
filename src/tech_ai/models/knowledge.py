"""Modelos internos da recuperação semântica."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MetadataValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    """Documento recuperado do ChromaDB sem expor objetos do SDK."""

    record_id: str
    text: str
    metadata: dict[str, MetadataValue]
    distance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "metadata": self.metadata,
            "distance": self.distance,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeCitation:
    """Referência pequena retornada ao consumidor junto da resposta."""

    number: int
    record_id: str
    distance: float
    focus: str | None
    source_url: str | None

    @classmethod
    def from_result(
        cls,
        result: KnowledgeSearchResult,
        *,
        number: int,
    ) -> KnowledgeCitation:
        focus = result.metadata.get("focus")
        source_url = result.metadata.get("source_url")
        return cls(
            number=number,
            record_id=result.record_id,
            distance=result.distance,
            focus=focus if isinstance(focus, str) else None,
            source_url=source_url if isinstance(source_url, str) else None,
        )

    def as_dict(self) -> dict[str, str | float | None]:
        return {
            "citation": f"[{self.number}]",
            "record_id": self.record_id,
            "distance": self.distance,
            "focus": self.focus,
            "source_url": self.source_url,
        }
