"""Interface de linha de comando do runtime de IA."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tech_ai.config.settings import Settings
from tech_ai.errors import (
    ConfigurationError,
    ModelArtifactReadError,
    ModelArtifactValidationError,
)
from tech_ai.services.model_artifact_service import inspect_model_artifact


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tech-ai",
        description="Consome o modelo médico pronto do Tech Challenge.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subcommands.add_parser(
        "inspect-model",
        help="Valida e exibe o manifesto de um modelo publicado.",
    )
    inspect_parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.model_manifest_path,
        help="Manifesto publicado pelo tech-fine-tuning.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = Settings.from_environment()
        parser = _build_parser(settings)
        arguments = parser.parse_args(argv)
        if arguments.command == "inspect-model":
            artifact = inspect_model_artifact(arguments.manifest)
            print(artifact.to_json())
            return 0
    except (
        ConfigurationError,
        ModelArtifactReadError,
        ModelArtifactValidationError,
    ) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2

    parser.error(f"Comando desconhecido: {arguments.command}")
    return 2
