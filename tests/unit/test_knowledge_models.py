from __future__ import annotations

from tech_ai.models.knowledge import KnowledgeCitation, KnowledgeSearchResult


def test_citation_projects_supported_metadata() -> None:
    result = KnowledgeSearchResult(
        record_id="id-1",
        text="texto",
        metadata={"focus": "Asma", "source_url": "https://example.test", "count": 2},
        distance=0.12,
    )

    citation = KnowledgeCitation.from_result(result, number=1)

    assert citation.as_dict() == {
        "citation": "[1]",
        "record_id": "id-1",
        "distance": 0.12,
        "focus": "Asma",
        "source_url": "https://example.test",
    }
    assert result.as_dict()["text"] == "texto"


def test_citation_ignores_non_string_optional_metadata() -> None:
    result = KnowledgeSearchResult("id-1", "texto", {"focus": 1}, 0.3)

    citation = KnowledgeCitation.from_result(result, number=1)

    assert citation.focus is None
    assert citation.source_url is None
