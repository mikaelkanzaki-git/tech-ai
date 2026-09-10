"""Contrato mínimo de inferência usado pelos casos de uso."""

from __future__ import annotations

from typing import Protocol


class LlmClient(Protocol):
    """Gera texto sem expor Transformers, Torch ou outro SDK ao serviço."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...
