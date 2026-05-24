from __future__ import annotations

from datasets.llm import parse_entity_response


def test_parse_entity_response_json_array() -> None:
    assert parse_entity_response('["sepsis", "heart failure", "sepsis"]') == [
        "sepsis",
        "heart failure",
    ]


def test_parse_entity_response_bullets() -> None:
    assert parse_entity_response("- pneumonia\n- acute kidney injury") == [
        "pneumonia",
        "acute kidney injury",
    ]
