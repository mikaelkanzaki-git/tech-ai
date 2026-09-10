"""Estruturas de entrada e saída do assistente médico."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tech_ai.models.knowledge import KnowledgeCitation

ACADEMIC_DISCLAIMER = (
    "Conteúdo experimental para fins acadêmicos; não substitui avaliação médica."
)


@dataclass(frozen=True, slots=True)
class AssistantQuery:
    """Pergunta enviada ao fluxo RAG."""

    question: str


@dataclass(frozen=True, slots=True)
class AssistantAnswer:
    """Resposta fundamentada, acompanhada da versão do modelo e das fontes."""

    question: str
    answer: str
    model_id: str
    sources: tuple[KnowledgeCitation, ...]
    grounded: bool
    warnings: tuple[str, ...] = ()
    disclaimer: str = ACADEMIC_DISCLAIMER

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "model": self.model_id,
            "sources": [source.as_dict() for source in self.sources],
            "warnings": list(self.warnings),
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
