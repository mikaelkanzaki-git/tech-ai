"""Validação da fronteira entre o fine-tuning e o runtime de IA."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tech_ai.errors import ModelArtifactValidationError
from tech_ai.integrations.model_artifact.manifest_reader import read_model_manifest
from tech_ai.models.model_artifact import ArtifactType, ModelArtifact

MODEL_ARTIFACT_SCHEMA_VERSION = "1.0"
_ARTIFACT_TYPES: set[ArtifactType] = {"peft_adapter", "merged_transformers"}
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_mapping(
    data: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise ModelArtifactValidationError(f"{context}: campo {field!r} deve ser um objeto.")
    return value


def _required_string(data: Mapping[str, Any], field: str, *, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ModelArtifactValidationError(
            f"{context}: campo {field!r} deve ser uma string não vazia."
        )
    return value.strip()


def _optional_string(data: Mapping[str, Any], field: str, *, context: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelArtifactValidationError(
            f"{context}: campo {field!r} deve ser string não vazia ou null."
        )
    return value.strip()


def inspect_model_artifact(manifest_path: Path) -> ModelArtifact:
    """Valida o contrato mínimo antes que um runtime tente carregar os pesos."""

    data = read_model_manifest(manifest_path)
    context = f"manifesto {manifest_path}"
    schema_version = _required_string(data, "schema_version", context=context)
    if schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactValidationError(
            f"{context}: schema {schema_version!r} não suportado; "
            f"esperado {MODEL_ARTIFACT_SCHEMA_VERSION!r}."
        )

    artifact_type_value = _required_string(data, "artifact_type", context=context)
    if artifact_type_value not in _ARTIFACT_TYPES:
        raise ModelArtifactValidationError(
            f"{context}: artifact_type {artifact_type_value!r} não suportado."
        )
    artifact_type: ArtifactType = artifact_type_value

    artifact = _required_mapping(data, "artifact", context=context)
    base_model = _required_mapping(data, "base_model", context=context)
    tokenizer = _required_mapping(data, "tokenizer", context=context)
    provenance = _required_mapping(data, "provenance", context=context)
    producer = _required_string(provenance, "producer", context=context)
    if producer != "tech-fine-tuning":
        raise ModelArtifactValidationError(
            f"{context}: produtor {producer!r} não é aceito pelo runtime."
        )
    producer_commit = _required_string(provenance, "producer_commit", context=context)
    if not _GIT_COMMIT.fullmatch(producer_commit):
        raise ModelArtifactValidationError(f"{context}: producer_commit inválido.")
    dataset_hash = _required_string(
        provenance,
        "dataset_manifest_sha256",
        context=context,
    )
    if not _SHA256.fullmatch(dataset_hash):
        raise ModelArtifactValidationError(f"{context}: dataset_manifest_sha256 inválido.")

    return ModelArtifact(
        schema_version=schema_version,
        artifact_id=_required_string(data, "artifact_id", context=context),
        artifact_type=artifact_type,
        artifact_uri=_required_string(artifact, "uri", context=context),
        base_model_id=_required_string(base_model, "id", context=context),
        base_model_revision=_optional_string(base_model, "revision", context=context),
        tokenizer_id=_required_string(tokenizer, "id", context=context),
        tokenizer_revision=_optional_string(tokenizer, "revision", context=context),
        producer_commit=producer_commit,
        dataset_manifest_sha256=dataset_hash,
    )
