# Instruções de arquitetura

Este serviço adota a Arquitetura em Camadas Pragmática descrita em
`docs/architecture/pragmatic-layered-architecture.md`.

Ao criar ou alterar código:

- coloque estruturas de dados e enums em `models/`;
- coloque carregamento de modelo, inferência e orquestração em `services/`;
- coloque runtimes de modelo e sistemas externos em `integrations/`;
- coloque settings e montagem de dependências em `config/`;
- crie `repositories/` ou `api/` somente quando houver responsabilidade real;
- não crie diretórios genéricos `ports/`, `adapters/`, `utils/` ou `helpers/`;
- mantenha o fluxo de dependência `runner/api -> services -> models`;
- não importe Transformers, llama.cpp ou SDKs externos em `models/`;
- não implemente preparação de dataset, treinamento ou avaliação de checkpoint neste serviço;
- consuma somente artefatos publicados pelo `tech-fine-tuning`;
- consulte o ChromaDB em modo somente-leitura; indexação e upsert pertencem ao `tech-ingestao`;
- preserve na busca o mesmo modelo e a mesma dimensão de embeddings usados na indexação;
- respostas médicas devem expor fontes e o aviso de uso acadêmico;
- não faça commit, push ou PR sem autorização explícita do usuário.

Antes de entregar mudanças, execute:

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```
