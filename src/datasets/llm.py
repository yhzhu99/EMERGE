from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

from datasets.rag import (
    RetrievedNode,
    build_patient_summary,
    clean_entity,
    deduplicate_entities,
    extract_note_entities,
)

if TYPE_CHECKING:
    from config import LLMConfig


@dataclass(frozen=True)
class OpenAICompatibleChatClient:
    base_url: str
    api_key: str
    timeout_seconds: int = 60

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = requests.post(
            _chat_completion_url(self.base_url),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])


def extract_note_entities_with_llm(text: str, cfg: LLMConfig, *, max_entities: int) -> list[str]:
    if cfg.entity_backend == "rules":
        return extract_note_entities(text, max_entities=max_entities)
    if cfg.entity_backend != "openai_compatible":
        raise ValueError(f"Unsupported entity extraction backend: {cfg.entity_backend}")

    client = _client_or_none(cfg)
    if client is None:
        if cfg.fallback_to_local:
            return extract_note_entities(text, max_entities=max_entities)
        raise RuntimeError(f"Missing API key environment variable: {cfg.api_key_env}")

    entities: list[str] = []
    for _ in range(cfg.max_entity_rounds):
        response = client.complete(
            model=cfg.entity_model_name,
            system_prompt=ENTITY_SYSTEM_PROMPT,
            user_prompt=_entity_user_prompt(text, entities),
            max_tokens=cfg.max_entity_tokens,
            temperature=cfg.temperature,
        )
        entities.extend(parse_entity_response(response))
        entities = _keep_entities_appearing_in_text(entities, text)
        if len(entities) >= max_entities:
            break
    if not entities and cfg.fallback_to_local:
        return extract_note_entities(text, max_entities=max_entities)
    return deduplicate_entities(entities)[:max_entities]


def summarize_patient_with_llm(
    abnormal_features: list[str],
    note_entities: list[str],
    retrieved: list[RetrievedNode],
    cfg: LLMConfig,
) -> str:
    if cfg.summary_backend == "template":
        return build_patient_summary(abnormal_features, note_entities, retrieved)
    if cfg.summary_backend != "openai_compatible":
        raise ValueError(f"Unsupported summary backend: {cfg.summary_backend}")

    client = _client_or_none(cfg)
    if client is None:
        if cfg.fallback_to_local:
            return build_patient_summary(abnormal_features, note_entities, retrieved)
        raise RuntimeError(f"Missing API key environment variable: {cfg.api_key_env}")

    response = client.complete(
        model=cfg.summary_model_name,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=_summary_user_prompt(abnormal_features, note_entities, retrieved),
        max_tokens=cfg.max_summary_tokens,
        temperature=cfg.temperature,
    )
    summary = " ".join(response.split())
    if not summary and cfg.fallback_to_local:
        return build_patient_summary(abnormal_features, note_entities, retrieved)
    return summary


def parse_entity_response(response: str) -> list[str]:
    text = response.strip()
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("entities") or parsed.get("diseases")
    if isinstance(parsed, list):
        return deduplicate_entities([str(item) for item in parsed])

    candidates = []
    for line in re.split(r"[\n;]", text):
        line = re.sub(r"^\s*[-*]?\s*\d*[\).:-]?\s*", "", line).strip()
        if not line:
            continue
        candidates.extend(part.strip(" \"'`[]") for part in line.split(","))
    return deduplicate_entities(candidates)


def _try_parse_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", text)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _keep_entities_appearing_in_text(entities: list[str], text: str) -> list[str]:
    lowered = text.lower()
    kept = []
    for entity in entities:
        clean = clean_entity(entity)
        if clean and clean.lower() in lowered:
            kept.append(clean)
    return deduplicate_entities(kept)


def _client_or_none(cfg: LLMConfig) -> OpenAICompatibleChatClient | None:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        return None
    return OpenAICompatibleChatClient(
        base_url=cfg.api_base_url,
        api_key=api_key,
        timeout_seconds=cfg.timeout_seconds,
    )


def _chat_completion_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _entity_user_prompt(note_text: str, existing_entities: list[str]) -> str:
    existing = ", ".join(existing_entities) if existing_entities else "None"
    return (
        "Extract disease entities that the patient may suffer from. "
        "Return only a JSON array of entity strings. Every entity must appear verbatim in the note.\n\n"
        f"Existing extracted entities: {existing}\n\n"
        f"Clinical note:\n{note_text}"
    )


def _summary_user_prompt(
    abnormal_features: list[str],
    note_entities: list[str],
    retrieved: list[RetrievedNode],
) -> str:
    kg_context = []
    for item in retrieved:
        triples = "; ".join(item.triples) if item.triples else "None"
        kg_context.append(
            f"- entity={item.entity}; kg_node={item.node_name}; description={item.document}; triples={triples}"
        )
    return (
        "Generate a concise patient health-status summary for mortality and 30-day readmission prediction. "
        "Use only the provided abnormal time-series features, note entities, and PrimeKG evidence.\n\n"
        f"Abnormal time-series features: {', '.join(abnormal_features) or 'None'}\n"
        f"Clinical-note disease entities: {', '.join(note_entities) or 'None'}\n"
        "Retrieved PrimeKG evidence:\n"
        + ("\n".join(kg_context) if kg_context else "None")
    )


ENTITY_SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. Extract only disease or disorder entities, "
    "avoid procedures, medications, generic anatomy, and administrative phrases."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a clinical summarization assistant. Produce a compact, factual summary grounded in the "
    "given EHR and PrimeKG evidence."
)
