from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_ai.runner import main

from .test_model_artifact_service import valid_manifest, write_manifest


def test_inspect_model_command_prints_validated_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_manifest(tmp_path / "model-manifest.json")

    exit_code = main(["inspect-model", "--manifest", str(path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["artifact_id"] == "medquad-model-001"


def test_inspect_model_command_returns_error_for_invalid_manifest(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["artifact_type"] = "unknown"
    path = write_manifest(tmp_path / "model-manifest.json", manifest)

    assert main(["inspect-model", "--manifest", str(path)]) == 2
