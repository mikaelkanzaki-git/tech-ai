"""Erros estáveis apresentados pelo runtime do tech-ai."""


class ConfigurationError(ValueError):
    """Indica uma configuração ausente ou inválida."""


class ModelArtifactReadError(ValueError):
    """Indica que o manifesto de um modelo pronto não pôde ser lido."""


class ModelArtifactValidationError(ValueError):
    """Indica que um artefato publicado não atende ao contrato do runtime."""
