"""Runtime Transformers para o modelo e o adaptador publicados no Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from tech_ai.config.settings import GenerationSettings
from tech_ai.errors import ModelRuntimeError
from tech_ai.models.model_artifact import ModelArtifact


def resolve_artifact_source(artifact_uri: str) -> str:
    """Converte ``hf://owner/repo`` em uma referência aceita pelo Hub."""

    prefix = "hf://"
    if artifact_uri.startswith(prefix):
        return artifact_uri[len(prefix) :]
    return str(Path(artifact_uri).expanduser())


class HuggingFaceLlmClient:
    """Mantém o modelo carregado e oferece uma fronteira pequena de geração."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        settings: GenerationSettings,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings

    @classmethod
    def from_artifact(
        cls,
        artifact: ModelArtifact,
        settings: GenerationSettings,
    ) -> HuggingFaceLlmClient:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ModelRuntimeError(
                "Extras de runtime ausentes. Instale com 'uv sync --extra runtime'."
            ) from error

        source = resolve_artifact_source(artifact.artifact_uri)
        has_cuda = torch.cuda.is_available()
        model: Any
        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16 if has_cuda else torch.float32,
        }
        if has_cuda:
            load_kwargs["device_map"] = "auto"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                artifact.tokenizer_id,
                revision=artifact.tokenizer_revision,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if artifact.artifact_type == "merged_transformers":
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    revision=artifact.artifact_revision,
                    **load_kwargs,
                )
            else:
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    artifact.base_model_id,
                    revision=artifact.base_model_revision,
                    **load_kwargs,
                )
                model = PeftModel.from_pretrained(
                    base_model,
                    source,
                    revision=artifact.artifact_revision,
                )
            model.eval()
        except Exception as error:
            raise ModelRuntimeError(
                f"Não foi possível carregar o artefato {artifact.artifact_id!r}: {error}"
            ) from error
        return cls(model=model, tokenizer=tokenizer, settings=settings)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {name: value.to(device) for name, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self._settings.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=self._settings.repetition_penalty,
                    no_repeat_ngram_size=self._settings.no_repeat_ngram_size,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            generated_tokens = output[0][input_length:]
            answer = cast(
                str,
                self._tokenizer.decode(generated_tokens, skip_special_tokens=True),
            ).strip()
        except Exception as error:
            raise ModelRuntimeError("A inferência do modelo falhou.") from error
        if not answer:
            raise ModelRuntimeError("O modelo retornou uma resposta vazia.")
        return answer
