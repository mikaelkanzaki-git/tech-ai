from __future__ import annotations

from pathlib import Path

import pytest

from tech_ai.config.settings import (
    ChromaSettings,
    GenerationSettings,
    LocalEmbeddingSettings,
    Settings,
)
from tech_ai.errors import ConfigurationError


def test_settings_use_documented_default() -> None:
    settings = Settings.from_environment({})

    assert settings.model_manifest_path == Path("configs/models/tech3-v3.json")
    assert settings.retrieval_limit == 4
    assert settings.max_retrieval_distance == 0.55
    assert settings.max_source_characters == 2_000


def test_settings_read_environment_override() -> None:
    settings = Settings.from_environment(
        {
            "TECH_AI_MODEL_MANIFEST_PATH": "published/model-manifest.json",
            "TECH_AI_RETRIEVAL_LIMIT": "7",
            "TECH_AI_MAX_RETRIEVAL_DISTANCE": "0.42",
            "TECH_AI_MAX_SOURCE_CHARACTERS": "900",
        }
    )

    assert settings.model_manifest_path == Path("published/model-manifest.json")
    assert settings.retrieval_limit == 7
    assert settings.max_retrieval_distance == 0.42
    assert settings.max_source_characters == 900


def test_settings_reject_path_without_filename() -> None:
    with pytest.raises(ConfigurationError, match="não pode ser vazio"):
        Settings(model_manifest_path=Path("."))


@pytest.mark.parametrize(
    "environment",
    [
        {"TECH_AI_RETRIEVAL_LIMIT": "0"},
        {"TECH_AI_RETRIEVAL_LIMIT": "abc"},
        {"TECH_AI_MAX_RETRIEVAL_DISTANCE": "0"},
        {"TECH_AI_MAX_RETRIEVAL_DISTANCE": "invalid"},
        {"TECH_AI_MAX_SOURCE_CHARACTERS": "-1"},
    ],
)
def test_settings_reject_invalid_positive_integers(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="deve ser"):
        Settings.from_environment(environment)


def test_chroma_settings_use_shared_contract_and_overrides() -> None:
    defaults = ChromaSettings.from_environment({})
    configured = ChromaSettings.from_environment(
        {
            "CHROMA_HOST": "chroma",
            "CHROMA_PORT": "9000",
            "CHROMA_COLLECTION": "medical-v2",
            "CHROMA_SSL": "true",
        }
    )

    assert defaults.collection == "medquad_knowledge_minilm_v1"
    assert configured == ChromaSettings(
        host="chroma", port=9000, collection="medical-v2", ssl=True
    )


def test_chroma_cloud_settings_use_secure_defaults() -> None:
    settings = ChromaSettings.from_environment(
        {
            "CHROMA_MODE": "cloud",
            "CHROMA_API_KEY": "secret-value",
            "CHROMA_TENANT": "tenant-id",
            "CHROMA_DATABASE": "fiap-tech-challenge-3",
        }
    )

    assert settings.mode == "cloud"
    assert settings.host == "api.trychroma.com"
    assert settings.port == 443
    assert settings.ssl is True
    assert settings.api_key == "secret-value"
    assert settings.tenant == "tenant-id"
    assert settings.database == "fiap-tech-challenge-3"
    assert "secret-value" not in repr(settings)


@pytest.mark.parametrize(
    "missing_variable",
    ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"],
)
def test_chroma_cloud_settings_require_credentials(missing_variable: str) -> None:
    environment = {
        "CHROMA_MODE": "cloud",
        "CHROMA_API_KEY": "secret-value",
        "CHROMA_TENANT": "tenant-id",
        "CHROMA_DATABASE": "database-id",
    }
    del environment[missing_variable]

    with pytest.raises(ConfigurationError, match=missing_variable):
        ChromaSettings.from_environment(environment)


@pytest.mark.parametrize(
    "environment",
    [
        {"CHROMA_MODE": "other"},
        {"CHROMA_HOST": " "},
        {"CHROMA_COLLECTION": " "},
        {"CHROMA_PORT": "0"},
        {"CHROMA_SSL": "perhaps"},
    ],
)
def test_chroma_settings_reject_invalid_values(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        ChromaSettings.from_environment(environment)


def test_local_embedding_settings_preserve_index_contract() -> None:
    assert LocalEmbeddingSettings.from_environment({}) == LocalEmbeddingSettings(
        model="all-MiniLM-L6-v2",
        dimensions=384,
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"LOCAL_EMBEDDING_MODEL": "other-model"},
        {"LOCAL_EMBEDDING_DIMENSIONS": "768"},
        {"LOCAL_EMBEDDING_DIMENSIONS": "invalid"},
    ],
)
def test_local_embedding_settings_reject_incompatible_contract(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ConfigurationError):
        LocalEmbeddingSettings.from_environment(environment)


def test_generation_settings_read_loop_controls() -> None:
    settings = GenerationSettings.from_environment(
        {
            "TECH_AI_MAX_NEW_TOKENS": "128",
            "TECH_AI_REPETITION_PENALTY": "1.2",
            "TECH_AI_NO_REPEAT_NGRAM_SIZE": "5",
        }
    )

    assert settings == GenerationSettings(
        max_new_tokens=128,
        repetition_penalty=1.2,
        no_repeat_ngram_size=5,
    )


def test_generation_settings_reject_invalid_penalty() -> None:
    with pytest.raises(ConfigurationError, match="número"):
        GenerationSettings.from_environment({"TECH_AI_REPETITION_PENALTY": "abc"})
