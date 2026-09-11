# Tech AI

Runtime RAG responsável por consultar o conhecimento médico indexado no ChromaDB e gerar uma
resposta com o modelo publicado pelo `tech-fine-tuning`.

Este projeto apenas consome artefatos e conhecimento prontos. Leitura do MedQuAD, indexação,
preparação de datasets, treinamento e avaliação pertencem aos outros serviços.

> **Aviso:** o modelo `tech3-v3` é experimental, foi rejeitado para uso clínico na avaliação
> humana e foi publicado exclusivamente para aprendizado. As respostas não substituem avaliação
> médica.

## Fluxo

```text
pergunta
  -> ONNX local all-MiniLM-L6-v2
  -> ChromaDB / medquad_knowledge_minilm_v1
  -> filtro de relevância das evidências
  -> Qwen3-4B + mikaelkanzaki/tech3-v3
  -> validação de citações e contatos críticos
  -> resposta + fontes numeradas + aviso acadêmico
```

## Arquitetura

O serviço segue o mesmo padrão legível para quem vem de Java/Spring adotado no `ms-ai-agent`:

```text
src/tech_ai/
├── models/          DTOs e estruturas internas
├── services/        Casos de uso e orquestração
├── repositories/    Contratos de persistência
├── integrations/    ONNX, ChromaDB e Hugging Face
├── config/          Settings e montagem de dependências
├── errors.py
└── runner.py
```

Consulte
[`docs/architecture/pragmatic-layered-architecture.md`](docs/architecture/pragmatic-layered-architecture.md)
para a equivalência completa com Java/Spring.

## Requisitos

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- ChromaDB local ou Chroma Cloud previamente indexado pelo `tech-ingestao`;
- GPU NVIDIA recomendada para executar o Qwen3-4B localmente.

O modelo no Hugging Face é público. `HF_TOKEN` não é necessário para baixá-lo.

## 1. Preparar o ambiente leve

```powershell
uv sync --dev
```

Esse ambiente permite validar o manifesto e consultar o ChromaDB sem instalar PyTorch.

## 2. Configurar o ChromaDB

### Chroma Cloud (recomendado para o ambiente compartilhado)

Crie o arquivo local `.env` a partir do exemplo e informe as credenciais geradas no painel do
Chroma Cloud:

```powershell
Copy-Item .env.example .env
```

No `.env`, use:

```dotenv
CHROMA_MODE=cloud
CHROMA_HOST=api.trychroma.com
CHROMA_PORT=443
CHROMA_SSL=true
CHROMA_API_KEY=sua-chave
CHROMA_TENANT=seu-tenant
CHROMA_DATABASE=fiap-tech-challenge-3
CHROMA_COLLECTION=medquad_knowledge_minilm_v1
```

O arquivo `.env` é ignorado pelo Git. Nunca publique `CHROMA_API_KEY`. A coleção deve ter sido
criada e preenchida pelo `tech-ingestao`; o `tech-ai` possui acesso somente de leitura no fluxo da
aplicação.

### ChromaDB local (alternativa)

No repositório `tech-ingestao`:

```powershell
docker compose up -d chroma
```

A coleção esperada é `medquad_knowledge_minilm_v1`. O modo local continua sendo o padrão quando
`CHROMA_MODE` não é informado. Caso a coleção ainda esteja vazia, execute a indexação pelo
`tech-ingestao` antes de continuar.

## 3. Embedding local

O projeto usa o mesmo contrato local da indexação:

- modelo ONNX `all-MiniLM-L6-v2`;
- dimensão `384`;
- coleção `medquad_knowledge_minilm_v1`;
- destino ChromaDB selecionado por `CHROMA_MODE=local|cloud`.

O modelo de embeddings é baixado na primeira busca ou indexação e depois reutilizado do cache
local. Nenhuma chave de API é necessária. Veja todas as opções em [`.env.example`](.env.example).

## 4. Validar o modelo publicado

```powershell
uv run tech-ai inspect-model
```

O manifesto padrão em [`configs/models/tech3-v3.json`](configs/models/tech3-v3.json) fixa:

- adaptador `mikaelkanzaki/tech3-v3`;
- revisão publicada do adaptador;
- modelo base `Qwen/Qwen3-4B-Instruct-2507`;
- revisão do modelo base, tokenizer e procedência do treinamento.

## 5. Testar apenas a recuperação semântica

```powershell
uv run --env-file .env tech-ai search `
  --question "What is post-traumatic stress disorder?" `
  --limit 4
```

Este comando não carrega o Qwen e é o primeiro diagnóstico recomendado para a integração com o
ChromaDB. Por padrão, resultados com distância maior que `0.55` são descartados. O valor é uma
calibração inicial para a coleção atual e pode ser alterado com `--max-distance` ou
`TECH_AI_MAX_RETRIEVAL_DISTANCE`.

## 6. Instalar o runtime e responder

```powershell
uv sync --dev --extra runtime
```

Na primeira execução, o Hugging Face pode baixar vários gigabytes do modelo base. Depois:

```powershell
uv run --env-file .env tech-ai ask `
  --question "What is post-traumatic stress disorder?" `
  --limit 4
```

O carregamento usa BF16 quando CUDA está disponível, geração determinística, penalidade de
repetição e bloqueio de n-gramas repetidos. Esses parâmetros podem ser ajustados pelas variáveis
`TECH_AI_MAX_NEW_TOKENS`, `TECH_AI_REPETITION_PENALTY` e
`TECH_AI_NO_REPEAT_NGRAM_SIZE`.

## Fundamentação e segurança

O comando `ask` não confia apenas no prompt para controlar o modelo:

- cada fonte recebe um número estável (`[1]`, `[2]` etc.);
- quando a fonte é um fragmento, `record_id` expõe o ID canônico do documento de origem informado
  em `parent_record_id`;
- a resposta precisa citar pelo menos uma fonte existente;
- uma resposta sem citações válidas é refeita uma vez e, se continuar inválida, substituída pelo
  trecho `Answer:` da fonte mais relevante;
- contatos numéricos associados a emergência, hotline ou telefone são refeitos e, como última
  proteção, omitidos da saída;
- `grounded` informa se a resposta passou pelo contrato de citações;
- `warnings` torna visível qualquer intervenção da camada de segurança;
- quando nenhum resultado passa pelo filtro de distância, o modelo nem sequer é chamado.

Essas proteções reduzem respostas não fundamentadas, mas não transformam o projeto acadêmico em
um sistema adequado para uso clínico.

## Validação do projeto

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```

## Responsabilidades fora deste repositório

- leitura ou tratamento do MedQuAD;
- criação ou atualização da coleção ChromaDB;
- geração de JSONL conversacional;
- Unsloth, LoRA/QLoRA e treinamento;
- avaliação ou seleção de checkpoint;
- publicação de pesos ou adaptadores.

O contexto estruturado de pacientes e uma API HTTP serão integrados em fatias posteriores, sem
alterar o núcleo RAG já testado.
