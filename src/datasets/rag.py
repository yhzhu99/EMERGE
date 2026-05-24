from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

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
        retriever_backend: str = "tfidf",
        retriever_model_name: str = "BAAI/bge-m3",
        retriever_batch_size: int = 64,
        retriever_device: str = "auto",
        precomputed_node_matrix: np.ndarray | None = None,
    ) -> None:
        if not node_names:
            node_names = ["unknown disease"]
            documents = ["unknown disease"]
        self.node_names = node_names
        self.documents = documents
        self.triples_by_name = triples_by_name or {}
        self.retriever_backend = retriever_backend
        self.vectorizer: TfidfVectorizer | None = None
        self.encoder: Any | None = None
        if retriever_backend == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words="english",
                strip_accents="unicode",
            )
            self.node_matrix = normalize(self.vectorizer.fit_transform(documents), norm="l2", axis=1)
        elif retriever_backend == "sentence_transformer":
            self.encoder = _load_sentence_transformer(retriever_model_name, retriever_device)
            if precomputed_node_matrix is not None:
                self.node_matrix = precomputed_node_matrix.astype(np.float32)
            else:
                self.node_matrix = _encode_documents(
                    self.encoder,
                    documents,
                    batch_size=retriever_batch_size,
                )
        else:
            raise ValueError(f"Unsupported KG retriever backend: {retriever_backend}")

    @classmethod
    def from_primekg(
        cls,
        kg_dir: Path | None,
        *,
        full_triples: bool = True,
        require_triples: bool = False,
        triples_per_node_limit: int | None = None,
        max_features: int = 20000,
        retriever_backend: str = "tfidf",
        retriever_model_name: str = "BAAI/bge-m3",
        retriever_batch_size: int = 64,
        retriever_device: str = "auto",
        cache_dir: Path | None = None,
        use_cache: bool = True,
        rebuild_cache: bool = False,
    ) -> KnowledgeRetriever:
        if kg_dir is None:
            if require_triples:
                raise FileNotFoundError(_primekg_missing_message(Path("data/kg/primekg"), "kg.csv"))
            return cls(
                [],
                [],
                max_features=max_features,
                retriever_backend=retriever_backend,
                retriever_model_name=retriever_model_name,
                retriever_batch_size=retriever_batch_size,
                retriever_device=retriever_device,
            )
        kg_dir = Path(kg_dir)
        disease_path = _first_existing(
            kg_dir,
            ["disease_features.tab", "disease_features.tsv", "disease_features.csv"],
        )
        if disease_path is None:
            if require_triples:
                raise FileNotFoundError(
                    _primekg_missing_message(
                        kg_dir, "disease_features.tab or disease_features.csv"
                    )
                )
            return cls(
                [],
                [],
                max_features=max_features,
                retriever_backend=retriever_backend,
                retriever_model_name=retriever_model_name,
                retriever_batch_size=retriever_batch_size,
                retriever_device=retriever_device,
            )

        kg_path = _primekg_triples_path(kg_dir, require_official=require_triples) if full_triples else None
        cache_path = (
            _primekg_cache_path(
                cache_dir,
                disease_path=disease_path,
                kg_path=kg_path,
                full_triples=full_triples,
                triples_per_node_limit=triples_per_node_limit,
                retriever_backend=retriever_backend,
                retriever_model_name=retriever_model_name,
                max_features=max_features,
            )
            if use_cache and cache_dir is not None
            else None
        )
        if cache_path is not None and not rebuild_cache:
            cached = _read_primekg_cache(cache_path, retriever_backend=retriever_backend)
            if cached is not None:
                return cls(
                    cached["node_names"],
                    cached["documents"],
                    cached["triples_by_name"],
                    max_features=max_features,
                    retriever_backend=retriever_backend,
                    retriever_model_name=retriever_model_name,
                    retriever_batch_size=retriever_batch_size,
                    retriever_device=retriever_device,
                    precomputed_node_matrix=cached["node_matrix"],
                )

        sep = "\t" if disease_path.suffix in {".tab", ".tsv"} else ","
        disease = pd.read_csv(disease_path, sep=sep)
        disease = disease.drop_duplicates("mondo_name", keep="first")
        node_names = disease["mondo_name"].fillna("").astype(str).tolist()
        documents = [_build_node_document(row) for _, row in disease.iterrows()]
        triples_by_name = (
            _load_primekg_triples(
                kg_dir,
                require_official=require_triples,
                node_names=set(node_names),
                triples_per_node_limit=triples_per_node_limit,
            )
            if full_triples
            else {}
        )
        retriever = cls(
            node_names,
            documents,
            triples_by_name,
            max_features=max_features,
            retriever_backend=retriever_backend,
            retriever_model_name=retriever_model_name,
            retriever_batch_size=retriever_batch_size,
            retriever_device=retriever_device,
        )
        if cache_path is not None:
            node_matrix = retriever.node_matrix if retriever_backend == "sentence_transformer" else None
            _write_primekg_cache(cache_path, node_names, documents, triples_by_name, node_matrix=node_matrix)
        return retriever

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
            sims = self._similarities(entity)
            if sims is None:
                continue
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

    def _similarities(self, entity: str) -> np.ndarray | None:
        if self.retriever_backend == "tfidf":
            if self.vectorizer is None:
                return None
            query = self.vectorizer.transform([entity])
            if query.nnz == 0:
                return None
            return cosine_similarity(query, self.node_matrix)[0]
        if self.encoder is None:
            return None
        query_embedding = _encode_query(self.encoder, entity)
        return np.matmul(query_embedding, self.node_matrix.T)[0]


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


def _load_primekg_triples(
    kg_dir: Path,
    *,
    require_official: bool = False,
    node_names: set[str] | None = None,
    triples_per_node_limit: int | None = None,
) -> dict[str, list[str]]:
    kg_path = _primekg_triples_path(kg_dir, require_official=require_official)
    if kg_path is None:
        return {}
    triples: dict[str, list[str]] = {}
    usecols = None
    sample = pd.read_csv(kg_path, nrows=1)
    if {"x_name", "display_relation", "y_name"}.issubset(sample.columns):
        usecols = ["x_name", "display_relation", "y_name", "x_type", "y_type"]
    for chunk in pd.read_csv(kg_path, usecols=usecols, chunksize=200000):
        if not {"x_name", "display_relation", "y_name"}.issubset(chunk.columns):
            if require_official:
                raise ValueError(
                    f"PrimeKG triples file {kg_path} must contain x_name, display_relation, and y_name columns."
                )
            return {}
        if node_names:
            chunk = chunk[chunk["x_name"].isin(node_names) | chunk["y_name"].isin(node_names)]
        for row in chunk.itertuples(index=False):
            triple = f"({row.x_name}, {row.display_relation}, {row.y_name})"
            _append_triple(
                triples,
                str(row.x_name),
                triple,
                node_names=node_names,
                triples_per_node_limit=triples_per_node_limit,
            )
            _append_triple(
                triples,
                str(row.y_name),
                triple,
                node_names=node_names,
                triples_per_node_limit=triples_per_node_limit,
            )
    return triples


def _primekg_triples_path(kg_dir: Path, *, require_official: bool) -> Path | None:
    official_kg_path = kg_dir / "kg.csv"
    if require_official and not official_kg_path.exists():
        raise FileNotFoundError(_primekg_missing_message(kg_dir, "kg.csv"))
    if official_kg_path.exists():
        return official_kg_path
    return _first_existing(kg_dir, ["disease_subkg.csv"])


def _append_triple(
    triples: dict[str, list[str]],
    node_name: str,
    triple: str,
    *,
    node_names: set[str] | None,
    triples_per_node_limit: int | None,
) -> None:
    if node_names is not None and node_name not in node_names:
        return
    node_triples = triples.setdefault(node_name, [])
    if triples_per_node_limit is not None and len(node_triples) >= triples_per_node_limit:
        return
    node_triples.append(triple)


def _load_sentence_transformer(model_name: str, device: str) -> Any:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence-transformer KG retrieval requires optional LM dependencies. "
            "Install them with `uv sync --extra lm`."
        ) from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    return SentenceTransformer(model_name, device=resolved_device)


def _encode_documents(encoder: Any, documents: list[str], *, batch_size: int) -> np.ndarray:
    if hasattr(encoder, "encode_document"):
        return encoder.encode_document(
            documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
    return encoder.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def _encode_query(encoder: Any, query: str) -> np.ndarray:
    if hasattr(encoder, "encode_query"):
        return encoder.encode_query(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
    return encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def _primekg_cache_path(
    cache_dir: Path | None,
    *,
    disease_path: Path,
    kg_path: Path | None,
    full_triples: bool,
    triples_per_node_limit: int | None,
    retriever_backend: str,
    retriever_model_name: str,
    max_features: int,
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {
        "disease": _file_signature(disease_path),
        "kg": _file_signature(kg_path) if kg_path is not None else None,
        "full_triples": full_triples,
        "triples_per_node_limit": triples_per_node_limit,
        "retriever_backend": retriever_backend,
        "retriever_model_name": retriever_model_name,
        "max_features": max_features,
        "version": 1,
    }
    key = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / key


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_primekg_cache(cache_path: Path, *, retriever_backend: str) -> dict[str, Any] | None:
    metadata_path = cache_path / "metadata.json"
    triples_path = cache_path / "triples.json"
    if not metadata_path.exists() or not triples_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    node_matrix = None
    matrix_path = cache_path / "node_matrix.npy"
    if retriever_backend == "sentence_transformer":
        if not matrix_path.exists():
            return None
        node_matrix = np.load(matrix_path).astype(np.float32)
    return {
        "node_names": metadata["node_names"],
        "documents": metadata["documents"],
        "triples_by_name": json.loads(triples_path.read_text(encoding="utf-8")),
        "node_matrix": node_matrix,
    }


def _write_primekg_cache(
    cache_path: Path,
    node_names: list[str],
    documents: list[str],
    triples_by_name: dict[str, list[str]],
    *,
    node_matrix: Any | None,
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "metadata.json").write_text(
        json.dumps({"node_names": node_names, "documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    (cache_path / "triples.json").write_text(
        json.dumps(triples_by_name, ensure_ascii=False),
        encoding="utf-8",
    )
    if isinstance(node_matrix, np.ndarray):
        np.save(cache_path / "node_matrix.npy", node_matrix.astype(np.float32))


def _primekg_missing_message(kg_dir: Path, filename: str) -> str:
    return (
        f"PrimeKG {filename} is required for complete EMERGE RAG preprocessing but was not found in "
        f"{kg_dir}. Run `python src/main.py --stage download --config configs/mimic4_demo.toml` "
        "to download the official required PrimeKG files, or add `--primekg-full` to download every "
        "official Dataverse file."
    )
