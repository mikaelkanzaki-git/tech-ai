# Contrato do artefato de modelo

O `tech-fine-tuning` publicará um manifesto junto dos pesos ou adaptadores escolhidos. O
`tech-ai` consome apenas esta fronteira; arquivos intermediários de dataset e treinamento não
são necessários no runtime.

```json
{
  "schema_version": "1.0",
  "artifact_id": "medquad-model-001",
  "artifact_type": "peft_adapter",
  "artifact": {
    "uri": "hf://grupo/medquad-model-001",
    "revision": "cccccccccccccccccccccccccccccccccccccccc"
  },
  "base_model": {
    "id": "provider/base-model",
    "revision": "revision-or-null"
  },
  "tokenizer": {
    "id": "provider/base-model",
    "revision": "revision-or-null"
  },
  "provenance": {
    "producer": "tech-fine-tuning",
    "producer_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "dataset_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }
}
```

Tipos inicialmente aceitos:

- `peft_adapter`: adaptador LoRA/QLoRA acompanhado da referência ao modelo base;
- `merged_transformers`: pesos mesclados no formato Transformers.

O runtime valida o manifesto antes de escolher a integração capaz de resolver `artifact.uri`.
A revisão opcional fixa a versão publicada e evita que uma alteração futura no repositório mude
silenciosamente o modelo executado. O runtime atual aceita repositórios Hugging Face com o
prefixo `hf://` e caminhos locais.
