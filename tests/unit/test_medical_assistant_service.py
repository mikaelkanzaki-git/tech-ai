from __future__ import annotations

import pytest

from tech_ai.errors import AssistantError
from tech_ai.integrations.llm.fake import FakeLlmClient
from tech_ai.models.assistant import AssistantQuery
from tech_ai.models.knowledge import KnowledgeSearchResult
from tech_ai.services.knowledge_retrieval_service import KnowledgeRetrievalService
from tech_ai.services.medical_assistant_service import MedicalAssistantService

from .test_knowledge_retrieval_service import (
    StubEmbeddingClient,
    StubKnowledgeRepository,
)


def build_service(
    results: tuple[KnowledgeSearchResult, ...],
    *,
    response: str = "Use a evidência fornecida. [1]",
) -> tuple[MedicalAssistantService, FakeLlmClient]:
    retrieval = KnowledgeRetrievalService(
        StubEmbeddingClient(((0.1,),)),
        StubKnowledgeRepository(results),
    )
    llm = FakeLlmClient(response)
    return (
        MedicalAssistantService(
            retrieval_service=retrieval,
            llm_client=llm,
            model_id="mikaelkanzaki/tech3-v3",
            retrieval_limit=4,
            max_source_characters=12,
        ),
        llm,
    )


def test_answer_builds_grounded_prompt_and_returns_citations() -> None:
    result = KnowledgeSearchResult(
        record_id="id-1-chunk-1",
        text="0123456789 texto que será truncado",
        metadata={
            "parent_record_id": "id-1",
            "focus": "Asma",
            "source_url": "https://example.test/source",
        },
        distance=0.15,
    )
    service, llm = build_service((result,))

    answer = service.answer(AssistantQuery("O que é asma?"))

    assert answer.model_id == "mikaelkanzaki/tech3-v3"
    assert answer.grounded is True
    assert answer.sources[0].record_id == "id-1"
    assert answer.sources[0].focus == "Asma"
    assert answer.as_dict()["sources"][0]["citation"] == "[1]"
    assert answer.as_dict()["sources"][0]["distance"] == 0.15
    assert llm.system_prompt is not None and "SOMENTE" in llm.system_prompt
    assert llm.user_prompt is not None and '<fonte numero="1"' in llm.user_prompt
    assert "0123456789 t" in llm.user_prompt
    assert "será truncado" not in llm.user_prompt


def test_answer_does_not_call_model_without_evidence() -> None:
    service, llm = build_service(())

    answer = service.answer(AssistantQuery("O que é uma condição desconhecida?"))

    assert "Não encontrei evidências" in answer.answer
    assert answer.grounded is False
    assert answer.sources == ()
    assert llm.user_prompt is None


def test_answer_rejects_empty_question_and_empty_model_response() -> None:
    result = KnowledgeSearchResult("id-1", "texto", {}, 0.2)
    service, _ = build_service((result,))
    with pytest.raises(AssistantError, match="pergunta"):
        service.answer(AssistantQuery(" "))

    empty_service, _ = build_service((result,), response=" ")
    with pytest.raises(AssistantError, match="resposta vazia"):
        empty_service.answer(AssistantQuery("pergunta"), limit=2)


def test_answer_retries_then_uses_extractive_fallback_without_valid_citations() -> None:
    result = KnowledgeSearchResult(
        "id-1",
        "Medical topic: Asma\nQuestion: O que é?\nAnswer: Resposta da fonte.",
        {},
        0.2,
    )
    service, llm = build_service((result,), response="Resposta sem referência.")

    answer = service.answer(AssistantQuery("pergunta"))

    assert answer.grounded is True
    assert answer.answer == "Resposta da [1]"
    assert any("trecho da fonte" in warning for warning in answer.warnings)
    assert llm.call_count == 2
    assert llm.user_prompt is not None and "resposta anterior" in llm.user_prompt


def test_answer_rejects_citation_for_source_that_was_not_supplied() -> None:
    result = KnowledgeSearchResult("id-1", "texto", {}, 0.2)
    service, llm = build_service((result,), response="Resposta com fonte inexistente. [2]")

    answer = service.answer(AssistantQuery("pergunta"))

    assert answer.grounded is True
    assert answer.answer == "texto [1]"
    assert llm.call_count == 2


def test_answer_rejects_ungrounded_response_when_source_text_is_empty() -> None:
    result = KnowledgeSearchResult("id-1", " ", {}, 0.2)
    service, _ = build_service((result,), response="Resposta sem referência.")

    answer = service.answer(AssistantQuery("pergunta"))

    assert answer.grounded is False
    assert "referências verificáveis" in answer.answer
    assert any("descartada" in warning for warning in answer.warnings)


def test_answer_redacts_critical_contact_even_after_retry() -> None:
    result = KnowledgeSearchResult("id-1", "texto", {}, 0.2)
    service, llm = build_service(
        (result,),
        response="For help, call 1-800-273-TALK. [1]",
    )

    answer = service.answer(AssistantQuery("question"))

    assert answer.grounded is True
    assert "1-800-273-TALK" not in answer.answer
    assert "[contato omitido]" in answer.answer
    assert answer.warnings
    assert llm.call_count == 2


def test_answer_retries_and_warns_when_long_evidence_gets_short_answer() -> None:
    result = KnowledgeSearchResult("id-1", "evidência " * 80, {}, 0.2)
    service, llm = build_service((result,), response="Resposta breve. [1]")

    answer = service.answer(AssistantQuery("pergunta"))

    assert answer.grounded is True
    assert llm.call_count == 2
    assert any("permaneceu curta" in warning for warning in answer.warnings)
