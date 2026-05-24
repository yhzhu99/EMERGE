from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


DISEASE_HINTS = (
    "disease",
    "syndrome",
    "failure",
    "infection",
    "pneumonia",
    "sepsis",
    "diabetes",
    "hypertension",
    "hypotension",
    "ischemia",
    "infarction",
    "obstruction",
    "edema",
    "cancer",
    "tumor",
    "fracture",
    "anemia",
    "failure",
    "insufficiency",
    "disorder",
    "deficiency",
)

GENERIC_ENTITY_STOPWORDS = {
    "other",
    "natural",
    "graft",
    "bypass",
    "anastomosis",
    "procedure",
    "surgical operation",
    "artificial tissues used as implant causing abnormal patient reaction",
    "structure during a circulatory system procedure",
    "mention of misadventure at the time of the procedure",
    "of later complication",
}


@dataclass(frozen=True)
class RetrievedNode:
    entity: str
    node_name: str
    score: float
    document: str
    triples: list[str]


class KnowledgeRetriever:
    def __init__(
        self,
        node_names: list[str],
        documents: list[str],
        triples_by_name: dict[str, list[str]] | None = None,
        *,
        max_features: int = 20000,
    ) -> None:
        if not node_names:
            node_names = ["unknown disease"]
            documents = ["unknown disease"]
        self.node_names = node_names
        self.documents = documents
        self.triples_by_name = triples_by_name or {}
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            strip_accents="unicode",
        )
        self.node_matrix = normalize(self.vectorizer.fit_transform(documents), norm="l2", axis=1)

    @classmethod
    def from_primekg(
        cls,
        kg_dir: Path | None,
        *,
        full_triples: bool = True,
        max_features: int = 20000,
    ) -> "KnowledgeRetriever":
        if kg_dir is None:
            return cls([], [], max_features=max_features)
        kg_dir = Path(kg_dir)
        disease_path = _first_existing(
            kg_dir,
            ["disease_features.tab", "disease_features.tsv", "disease_features.csv"],
        )
        if disease_path is None:
            return cls([], [], max_features=max_features)

        sep = "\t" if disease_path.suffix in {".tab", ".tsv"} else ","
        disease = pd.read_csv(disease_path, sep=sep)
        disease = disease.drop_duplicates("mondo_name", keep="first")
        node_names = disease["mondo_name"].fillna("").astype(str).tolist()
        documents = [_build_node_document(row) for _, row in disease.iterrows()]
        triples_by_name = _load_primekg_triples(kg_dir) if full_triples else {}
        return cls(node_names, documents, triples_by_name, max_features=max_features)

    def retrieve(
        self,
        entities: list[str],
        *,
        threshold: float = 0.35,
        top_k: int = 10,
        triples_per_node: int = 5,
    ) -> list[RetrievedNode]:
        results: list[RetrievedNode] = []
        seen_nodes: set[str] = set()
        for entity in deduplicate_entities(entities):
            query = self.vectorizer.transform([entity])
            if query.nnz == 0:
                continue
            sims = cosine_similarity(query, self.node_matrix)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score < threshold:
                continue
            node_name = self.node_names[best_idx]
            if node_name in seen_nodes:
                continue
            seen_nodes.add(node_name)
            results.append(
                RetrievedNode(
                    entity=entity,
                    node_name=node_name,
                    score=best_score,
                    document=self.documents[best_idx],
                    triples=self.triples_by_name.get(node_name, [])[:triples_per_node],
                )
            )
            if len(results) >= top_k:
                break
        return results


def extract_note_entities(text: str, *, max_entities: int = 16) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    entities: list[str] = []
    for pattern in [
        r"\bdiagnosis\s+([^.;]+)",
        r"\bdiagnoses\s+([^.;]+)",
        r"\bhistory of\s+([^.;]+)",
        r"\bpast medical history\s+([^.;]+)",
        r"\bchief complaint\s+([^.;]+)",
    ]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.extend(_split_entity_phrase(match.group(1)))
    if not entities:
        for sentence in re.split(r"[.;\n]", text):
            sent = sentence.strip()
            lowered = sent.lower()
            if any(hint in lowered for hint in DISEASE_HINTS):
                entities.extend(_split_entity_phrase(sent))
    return deduplicate_entities(entities)[:max_entities]


def deduplicate_entities(entities: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entity in entities:
        clean = clean_entity(entity)
        if not clean:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def clean_entity(entity: object) -> str:
    text = re.sub(r"\s+", " ", str(entity)).strip(" .,:;[]()")
    text = re.sub(r"^(and|or|with|without|the|a|an)\s+", "", text, flags=re.IGNORECASE)
    if text.lower() in GENERIC_ENTITY_STOPWORDS:
        return ""
    if len(text) < 3:
        return ""
    return text


def build_patient_summary(
    abnormal_features: list[str],
    note_entities: list[str],
    retrieved: list[RetrievedNode],
    *,
    max_words: int = 180,
) -> str:
    parts: list[str] = []
    if abnormal_features:
        parts.append(
            "Abnormal early time-series signals include "
            + ", ".join(abnormal_features[:12])
            + "."
        )
    if note_entities:
        parts.append("Clinical text suggests " + ", ".join(note_entities[:12]) + ".")
    if retrieved:
        node_bits = []
        for item in retrieved[:8]:
            doc = _shorten(item.document, 34)
            node_bits.append(f"{item.node_name}: {doc}")
        parts.append("Retrieved KG context: " + " ".join(node_bits))
        triples = [triple for item in retrieved for triple in item.triples]
        if triples:
            parts.append("Related KG triples: " + "; ".join(triples[:8]) + ".")
    if not parts:
        parts.append("No abnormal early signals or disease entities were retrieved.")
    return _shorten(" ".join(parts), max_words)


def retrieval_records_to_json(items: list[RetrievedNode]) -> str:
    return json.dumps(
        [
            {
                "entity": item.entity,
                "node_name": item.node_name,
                "score": item.score,
                "triples": item.triples,
            }
            for item in items
        ],
        ensure_ascii=False,
    )


def _split_entity_phrase(phrase: str) -> list[str]:
    chunks = re.split(r"\s+(?:and|or|with|without)\s+|[,/]", phrase)
    return [clean_entity(chunk) for chunk in chunks if clean_entity(chunk)]


def _shorten(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _build_node_document(row: pd.Series) -> str:
    fields = [
        "mondo_name",
        "mondo_definition",
        "umls_description",
        "orphanet_definition",
        "orphanet_prevalence",
        "orphanet_epidemiology",
        "orphanet_clinical_description",
        "orphanet_management_and_treatment",
        "mayo_symptoms",
        "mayo_causes",
        "mayo_risk_factors",
        "mayo_complications",
        "mayo_prevention",
        "mayo_see_doc",
    ]
    values = []
    for field in fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip():
            values.append(str(value).strip())
    return " ".join(values) or str(row.get("mondo_name", "unknown disease"))


def _load_primekg_triples(kg_dir: Path) -> dict[str, list[str]]:
    kg_path = _first_existing(kg_dir, ["kg.csv", "disease_subkg.csv"])
    if kg_path is None:
        return {}
    triples: dict[str, list[str]] = {}
    usecols = None
    sample = pd.read_csv(kg_path, nrows=1)
    if {"x_name", "display_relation", "y_name"}.issubset(sample.columns):
        usecols = ["x_name", "display_relation", "y_name", "x_type", "y_type"]
    for chunk in pd.read_csv(kg_path, usecols=usecols, chunksize=200000):
        if not {"x_name", "display_relation", "y_name"}.issubset(chunk.columns):
            return {}
        for row in chunk.itertuples(index=False):
            triple = f"({row.x_name}, {row.display_relation}, {row.y_name})"
            triples.setdefault(str(row.x_name), []).append(triple)
            triples.setdefault(str(row.y_name), []).append(triple)
    return triples
