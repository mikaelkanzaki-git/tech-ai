"""Configurações externas do runtime e da recuperação semântica."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tech_ai.errors import ConfigurationError

_DEFAULT_MANIFEST_PATH = "configs/models/tech3-v3.json"


def _positive_integer(value: str, variable: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} deve ser um número inteiro.") from error
    if parsed <= 0:
        raise ConfigurationError(f"{variable} deve ser maior que zero.")
    return parsed


def _positive_float(value: str, variable: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} deve ser um número.") from error
    if parsed <= 0:
        raise ConfigurationError(f"{variable} deve ser maior que zero.")
    return parsed


def _boolean(value: str, variable: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{variable} deve ser true ou false.")


@dataclass(frozen=True, slots=True)
class Settings:
    """Parâmetros dos casos de uso, sem credenciais ou detalhes de SDK."""

    model_manifest_path: Path = Path(_DEFAULT_MANIFEST_PATH)
    retrieval_limit: int = 4
    max_retrieval_distance: float = 0.55
    max_source_characters: int = 2_000

    def __post_init__(self) -> None:
        if not self.model_manifest_path.name:
            raise ConfigurationError("TECH_AI_MODEL_MANIFEST_PATH não pode ser vazio.")
        if self.retrieval_limit <= 0:
            raise ConfigurationError("TECH_AI_RETRIEVAL_LIMIT deve ser maior que zero.")
        if self.max_retrieval_distance <= 0:
            raise ConfigurationError(
                "TECH_AI_MAX_RETRIEVAL_DISTANCE deve ser maior que zero."
            )
        if self.max_source_characters <= 0:
            raise ConfigurationError("TECH_AI_MAX_SOURCE_CHARACTERS deve ser maior que zero.")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        values = environment if environment is not None else os.environ
        return cls(
            model_manifest_path=Path(
                values.get("TECH_AI_MODEL_MANIFEST_PATH", _DEFAULT_MANIFEST_PATH)
            ),
            retrieval_limit=_positive_integer(
                values.get("TECH_AI_RETRIEVAL_LIMIT", "4"),
                "TECH_AI_RETRIEVAL_LIMIT",
            ),
            max_retrieval_distance=_positive_float(
                values.get("TECH_AI_MAX_RETRIEVAL_DISTANCE", "0.55"),
                "TECH_AI_MAX_RETRIEVAL_DISTANCE",
            ),
            max_source_characters=_positive_integer(
                values.get("TECH_AI_MAX_SOURCE_CHARACTERS", "2000"),
                "TECH_AI_MAX_SOURCE_CHARACTERS",
            ),
        )


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    """Conexão com a coleção escrita pelo ``tech-ingestao``."""

    mode: str = "local"
    host: str = "localhost"
    port: int = 8000
    collection: str = "medquad_knowledge_minilm_v1"
    ssl: bool = False
    api_key: str | None = field(default=None, repr=False)
    tenant: str | None = None
    database: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"local", "cloud"}:
            raise ConfigurationError("CHROMA_MODE deve ser local ou cloud.")
        if not self.host:
            raise ConfigurationError("CHROMA_HOST não pode ser vazio.")
        if self.port <= 0:
            raise ConfigurationError("CHROMA_PORT deve ser maior que zero.")
        if not self.collection:
            raise ConfigurationError("CHROMA_COLLECTION não pode ser vazia.")
        if self.mode == "cloud":
            missing = [
                variable
                for variable, value in (
                    ("CHROMA_API_KEY", self.api_key),
                    ("CHROMA_TENANT", self.tenant),
                    ("CHROMA_DATABASE", self.database),
                )
                if not value
            ]
            if missing:
                variables = ", ".join(missing)
                raise ConfigurationError(
                    f"Modo cloud requer as variáveis: {variables}."
                )

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ChromaSettings:
        values = os.environ if environment is None else environment
        mode = values.get("CHROMA_MODE", "local").strip().lower()
        cloud_mode = mode == "cloud"
        default_host = "api.trychroma.com" if cloud_mode else "localhost"
        default_port = "443" if cloud_mode else "8000"
        default_ssl = "true" if cloud_mode else "false"
        host = values.get("CHROMA_HOST", default_host).strip()
        collection = values.get(
            "CHROMA_COLLECTION", "medquad_knowledge_minilm_v1"
        ).strip()
        api_key = values.get("CHROMA_API_KEY", "").strip() or None
        tenant = values.get("CHROMA_TENANT", "").strip() or None
        database = values.get("CHROMA_DATABASE", "").strip() or None
        return cls(
            mode=mode,
            host=host,
            port=_positive_integer(values.get("CHROMA_PORT", default_port), "CHROMA_PORT"),
            collection=collection,
            ssl=_boolean(values.get("CHROMA_SSL", default_ssl), "CHROMA_SSL"),
            api_key=api_key,
            tenant=tenant,
            database=database,
        )


@dataclass(frozen=True, slots=True)
class LocalEmbeddingSettings:
    """Contrato do modelo ONNX compartilhado com o ``tech-ingestao``."""

    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> LocalEmbeddingSettings:
        values = os.environ if environment is None else environment
        model = values.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        dimensions = _positive_integer(
            values.get("LOCAL_EMBEDDING_DIMENSIONS", "384"),
            "LOCAL_EMBEDDING_DIMENSIONS",
        )
        if model != "all-MiniLM-L6-v2":
            raise ConfigurationError(
                "LOCAL_EMBEDDING_MODEL deve ser 'all-MiniLM-L6-v2' nesta versão."
            )
        if dimensions != 384:
            raise ConfigurationError(
                "LOCAL_EMBEDDING_DIMENSIONS deve ser 384 para all-MiniLM-L6-v2."
            )
        return cls(model=model, dimensions=dimensions)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Parâmetros conservadores para inferência determinística e sem loops."""

    max_new_tokens: int = 384
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> GenerationSettings:
        values = os.environ if environment is None else environment
        return cls(
            max_new_tokens=_positive_integer(
                values.get("TECH_AI_MAX_NEW_TOKENS", "384"),
                "TECH_AI_MAX_NEW_TOKENS",
            ),
            repetition_penalty=_positive_float(
                values.get("TECH_AI_REPETITION_PENALTY", "1.1"),
                "TECH_AI_REPETITION_PENALTY",
            ),
            no_repeat_ngram_size=_positive_integer(
                values.get("TECH_AI_NO_REPEAT_NGRAM_SIZE", "4"),
                "TECH_AI_NO_REPEAT_NGRAM_SIZE",
            ),
        )
