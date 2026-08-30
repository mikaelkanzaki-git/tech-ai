# Arquitetura em Camadas Pragmática

## Status

Adotada pelo `tech-ai` como runtime consumidor de modelos prontos. A preparação e o treinamento
foram isolados no `tech-fine-tuning`.

## Estrutura atual

```text
src/tech_ai/
├── models/          Contrato interno do artefato publicado
├── services/        Validação, carregamento e inferência
├── integrations/    Filesystem e runtimes externos
├── config/          Settings e composição
├── errors.py        Erros estáveis
└── runner.py        Entrada local
```

`repositories/` e `api/` só serão criados quando houver persistência ou transporte HTTP reais.

## Direção das dependências

```text
runner ───────────────> services ───────────────> models
  │                         │
  └────> config             └────> integrations/model_artifact
```

O runtime nunca importa o pacote Python do `tech-fine-tuning`. A fronteira é um manifesto JSON
versionado e um artefato endereçável. Unsloth, datasets e hiperparâmetros não atravessam essa
fronteira.

## Divisão de responsabilidades

| Serviço | Responsabilidade |
| --- | --- |
| `tech-ingestao` | Curar, sanitizar, dividir e indexar dados médicos. |
| `tech-fine-tuning` | Preparar SFT, treinar, avaliar e publicar o modelo. |
| `tech-ai` | Validar, carregar e utilizar o modelo publicado. |

Essa separação permite trocar a infraestrutura de treinamento sem alterar o runtime e trocar o
runtime de inferência sem regenerar o dataset.
