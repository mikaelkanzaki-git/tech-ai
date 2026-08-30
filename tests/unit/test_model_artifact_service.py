from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tech_ai.errors import ModelArtifactReadError, ModelArtifactValidationError
from tech_ai.integrations.model_artifact.manifest_reader import read_model_manifest
from tech_ai.services.model_artifact_service import inspect_model_artifact


def valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_id": "medquad-model-001",
        "artifact_type": "peft_adapter",
        "artifact": {"uri": "hf://group/medquad-model-001"},
        "base_model": {"id": "provider/base-model", "revision": "base-revision"},
        "tokenizer": {"id": "provider/base-model", "revision": None},
        "provenance": {
            "producer": "tech-fine-tuning",
            "producer_commit": "a" * 40,
            "dataset_manifest_sha256": "b" * 64,
        },
    }


def write_manifest(path: Path, data: object | None = None) -> Path:
    path.write_text(
        json.dumps(valid_manifest() if data is None else data),
        encoding="utf-8",
    )
    return path


def test_inspect_model_artifact_returns_runtime_projection(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "model-manifest.json")

    artifact = inspect_model_artifact(path)

    assert artifact.artifact_id == "medquad-model-001"
    assert artifact.artifact_type == "peft_adapter"
    assert artifact.artifact_uri == "hf://group/medquad-model-001"
    assert artifact.base_model_revision == "base-revision"
    assert artifact.tokenizer_revision is None
    assert artifact.as_dict()["provenance"]["producer"] == "tech-fine-tuning"
    assert json.loads(artifact.to_json())["artifact_id"] == "medquad-model-001"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "schema"),
        ("artifact_type", "gguf", "artifact_type"),
    ],
)
def test_inspect_model_artifact_rejects_unsupported_values(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = valid_manifest()
    manifest[field] = value
    path = write_manifest(tmp_path / "model-manifest.json", manifest)

    with pytest.raises(ModelArtifactValidationError, match=message):
        inspect_model_artifact(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("producer", "other-service", "produtor"),
        ("producer_commit", "short", "producer_commit"),
        ("dataset_manifest_sha256", "short", "dataset_manifest_sha256"),
    ],
)
def test_inspect_model_artifact_rejects_invalid_provenance(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = valid_manifest()
    manifest["provenance"][field] = value
    path = write_manifest(tmp_path / "model-manifest.json", manifest)

    with pytest.raises(ModelArtifactValidationError, match=message):
        inspect_model_artifact(path)


def test_inspect_model_artifact_rejects_missing_object_and_empty_string(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["artifact"] = None
    path = write_manifest(tmp_path / "missing-object.json", manifest)
    with pytest.raises(ModelArtifactValidationError, match="deve ser um objeto"):
        inspect_model_artifact(path)

    manifest = valid_manifest()
    manifest["artifact_id"] = " "
    path = write_manifest(tmp_path / "empty-string.json", manifest)
    with pytest.raises(ModelArtifactValidationError, match="string não vazia"):
        inspect_model_artifact(path)


def test_inspect_model_artifact_rejects_invalid_optional_revision(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["tokenizer"]["revision"] = 123
    path = write_manifest(tmp_path / "model-manifest.json", manifest)

    with pytest.raises(ModelArtifactValidationError, match="string não vazia ou null"):
        inspect_model_artifact(path)


def test_manifest_reader_translates_file_and_json_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactReadError, match="Não foi possível ler"):
        read_model_manifest(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ModelArtifactReadError, match="Não foi possível ler"):
        read_model_manifest(invalid)

    non_object = write_manifest(tmp_path / "non-object.json", [])
    with pytest.raises(ModelArtifactReadError, match="objeto JSON"):
        read_model_manifest(non_object)
