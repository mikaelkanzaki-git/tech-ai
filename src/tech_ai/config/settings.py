"""Configuração explícita do runtime de modelo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tech_ai.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    """Local do manifesto publicado, substituível por variável de ambiente."""

    model_manifest_path: Path = Path(
        "../tech-fine-tuning/artifacts/model/model-manifest.json"
    )

    def __post_init__(self) -> None:
        if not self.model_manifest_path.name:
            raise ConfigurationError("TECH_AI_MODEL_MANIFEST_PATH não pode ser vazio.")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        values = environment if environment is not None else os.environ
        return cls(
            model_manifest_path=Path(
                values.get(
                    "TECH_AI_MODEL_MANIFEST_PATH",
                    "../tech-fine-tuning/artifacts/model/model-manifest.json",
                )
            ),
        )
