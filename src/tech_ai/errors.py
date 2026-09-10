"""Erros estáveis apresentados pelo runtime do tech-ai."""


class ConfigurationError(ValueError):
    """Indica uma configuração ausente ou inválida."""


class ModelArtifactReadError(ValueError):
    """Indica que o manifesto de um modelo pronto não pôde ser lido."""


class ModelArtifactValidationError(ValueError):
    """Indica que um artefato publicado não atende ao contrato do runtime."""


class ModelRuntimeError(RuntimeError):
    """Indica que o modelo publicado não pôde ser carregado ou executado."""


class EmbeddingError(RuntimeError):
    """Indica falha ao gerar o vetor de uma consulta."""


class KnowledgeStoreError(RuntimeError):
    """Indica falha ao consultar a base vetorial."""


class KnowledgeRetrievalError(ValueError):
    """Indica uma consulta semântica inválida ou inconsistente."""


class AssistantError(RuntimeError):
    """Indica que o assistente não conseguiu produzir uma resposta válida."""
