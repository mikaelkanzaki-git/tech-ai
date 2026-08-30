"""Modelo interno do artefato pronto consumido pelo runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

ArtifactType = Literal["peft_adapter", "merged_transformers"]


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Projeção mínima e versionada publicada pelo tech-fine-tuning."""

    schema_version: str
    artifact_id: str
    artifact_type: ArtifactType
    artifact_uri: str
    base_model_id: str
    base_model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    producer_commit: str
    dataset_manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "artifact_uri": self.artifact_uri,
            "base_model": {
                "id": self.base_model_id,
                "revision": self.base_model_revision,
            },
            "tokenizer": {
                "id": self.tokenizer_id,
                "revision": self.tokenizer_revision,
            },
            "provenance": {
                "producer": "tech-fine-tuning",
                "producer_commit": self.producer_commit,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
