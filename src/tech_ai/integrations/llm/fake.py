"""Implementação determinística para testes e demonstrações leves."""

from __future__ import annotations


class FakeLlmClient:
    """Retorna uma resposta configurada e registra o último prompt."""

    def __init__(self, response: str = "Resposta simulada.") -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.call_count = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response
