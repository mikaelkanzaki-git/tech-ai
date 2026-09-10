# Arquitetura em Camadas Pragmática

## Status

Adotada pelo `tech-ai` como runtime consumidor de modelos prontos. A preparação e o treinamento
foram isolados no `tech-fine-tuning`.

## Equivalência com Java/Spring

| Java/Spring | Convenção adotada |
| --- | --- |
| `src/main/java` | `src/tech_ai` |
| `src/test/java` | `tests/` |
| `model` / DTO | `models/` |
| `service` | `services/` |
| `repository` | `repositories/` |
| client de API ou SDK | `integrations/<sistema>/` |
| `@Configuration` | `config/dependencies.py` |
| `application.yml` | `config/settings.py` e variáveis de ambiente |
| `Application.java` | `__main__.py` e `runner.py` |

## Estrutura atual

```text
src/tech_ai/
├── models/          Artefato, pergunta, resposta e fontes
├── services/        Recuperação semântica e orquestração RAG
├── repositories/    Contrato somente-leitura da base vetorial
├── integrations/    ONNX local, ChromaDB, Hugging Face e filesystem
├── config/          Settings e montagem explícita das dependências
├── errors.py        Erros estáveis
└── runner.py        Entrada local/CLI
```

`api/` será criada quando o transporte HTTP entrar em uma fatia própria. Diretórios vazios não
devem ser criados somente para reproduzir o diagrama.

## Direção das dependências

```text
runner ───────────────> services ───────────────> models
  │                         │  \
  │                         │   └───────────────> integration clients
  │                         └───────────────────> repository contracts
  └────> config/dependencies ───────────────────> implementações concretas
```

O runtime nunca importa o pacote Python do `tech-fine-tuning`. A fronteira é um manifesto JSON
versionado e um artefato endereçável. Unsloth, datasets e hiperparâmetros não atravessam essa
fronteira.

O runtime também nunca executa `upsert` no ChromaDB. O `tech-ingestao` escreve a coleção e o
`tech-ai` apenas pesquisa, usando o mesmo contrato de embedding.

## Divisão de responsabilidades

| Serviço | Responsabilidade |
| --- | --- |
| `tech-ingestao` | Curar, sanitizar, dividir e indexar dados médicos. |
| `tech-fine-tuning` | Preparar SFT, treinar, avaliar e publicar o modelo. |
| `tech-ai` | Validar, carregar e utilizar o modelo publicado. |

Essa separação permite trocar a infraestrutura de treinamento sem alterar o runtime e trocar o
runtime de inferência sem regenerar o dataset.
