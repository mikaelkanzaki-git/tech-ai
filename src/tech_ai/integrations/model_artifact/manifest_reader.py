"""Leitor do manifesto publicado pelo serviço de fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tech_ai.errors import ModelArtifactReadError


def read_model_manifest(path: Path) -> dict[str, Any]:
    """Lê um objeto JSON sem expor detalhes do filesystem à camada de serviço."""

    try:
        with path.expanduser().resolve().open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelArtifactReadError(f"Não foi possível ler {path}: {error}") from error
    if not isinstance(data, dict):
        raise ModelArtifactReadError(f"O manifesto {path} deve conter um objeto JSON.")
    return data
