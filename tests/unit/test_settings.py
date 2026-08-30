from __future__ import annotations

from pathlib import Path

import pytest

from tech_ai.config.settings import Settings
from tech_ai.errors import ConfigurationError


def test_settings_use_documented_default() -> None:
    settings = Settings.from_environment({})

    assert settings.model_manifest_path == Path(
        "../tech-fine-tuning/artifacts/model/model-manifest.json"
    )


def test_settings_read_environment_override() -> None:
    settings = Settings.from_environment(
        {"TECH_AI_MODEL_MANIFEST_PATH": "published/model-manifest.json"}
    )

    assert settings.model_manifest_path == Path("published/model-manifest.json")


def test_settings_reject_path_without_filename() -> None:
    with pytest.raises(ConfigurationError, match="não pode ser vazio"):
        Settings(model_manifest_path=Path("."))
