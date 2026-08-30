# Tech AI

Runtime responsável por carregar e utilizar um modelo médico já publicado. Preparação de
dataset, fine-tuning, avaliação de checkpoints e geração do artefato pertencem exclusivamente ao
`tech-fine-tuning`.

## Estado atual

A primeira fatia estabelece a fronteira entre os dois serviços. O `tech-ai` consegue ler e
validar o manifesto de um modelo pronto, mas ainda não baixa pesos nem executa inferência. O
runtime concreto será escolhido depois que o primeiro modelo for produzido.

## Arquitetura

```text
tech-ai/
├── docs/
├── src/tech_ai/
│   ├── config/         Local do manifesto publicado
│   ├── integrations/   Leitura do artefato; futuramente runtime do modelo
│   ├── models/         Projeção interna do modelo pronto
│   ├── services/       Validação e, futuramente, inferência
│   ├── errors.py
│   └── runner.py
└── tests/unit/
```

O limite arquitetural está em
[`docs/architecture/pragmatic-layered-architecture.md`](docs/architecture/pragmatic-layered-architecture.md).

## Requisitos

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- um manifesto de modelo publicado pelo `tech-fine-tuning`.

## Preparação

```powershell
uv sync --dev
```

## Inspecionar um modelo publicado

```powershell
uv run tech-ai inspect-model `
  --manifest "..\tech-fine-tuning\artifacts\model\model-manifest.json"
```

O comando valida versão do contrato, tipo do artefato, modelo base, tokenizer e procedência do
treinamento. O contrato está em
[`docs/contracts/model-artifact.md`](docs/contracts/model-artifact.md).

O caminho padrão pode ser substituído por `TECH_AI_MODEL_MANIFEST_PATH`.

## Validar o projeto

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```

## Responsabilidades que não pertencem a este repositório

- leitura ou tratamento do MedQuAD;
- geração de JSONL conversacional;
- Unsloth, LoRA/QLoRA e treinamento;
- avaliação ou seleção de checkpoint;
- publicação dos pesos finais.

Essas atividades ficam no `tech-fine-tuning`. Depois da publicação, o `tech-ai` poderá carregar
o artefato, executar inferência e ser integrado ao assistente com LangGraph.
