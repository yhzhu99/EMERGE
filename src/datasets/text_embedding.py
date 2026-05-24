from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TextEncoder(Protocol):
    def fit(self, texts: list[str]) -> TextEncoder: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class TfidfSvdEncoder:
    embedding_dim: int = 768
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents="unicode",
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None
        self.actual_dim = 0

    def fit(self, texts: list[str]) -> TfidfSvdEncoder:
        clean_texts = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.fit_transform(clean_texts)
        max_components = min(self.embedding_dim, max(1, matrix.shape[0] - 1), max(1, matrix.shape[1] - 1))
        if max_components >= 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            self.svd.fit(matrix)
            self.actual_dim = max_components
        else:
            self.svd = None
            self.actual_dim = min(self.embedding_dim, matrix.shape[1])
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        clean_texts = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.transform(clean_texts)
        if self.svd is not None:
            emb = self.svd.transform(matrix).astype(np.float32)
        else:
            emb = matrix.toarray().astype(np.float32)
        emb = normalize(emb, norm="l2", axis=1, copy=False)
        return _pad_or_trim(emb, self.embedding_dim)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


@dataclass
class TransformerClsEncoder:
    model_name: str
    embedding_dim: int = 768
    batch_size: int = 8
    device: str = "auto"
    max_length: int = 4096

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer text embeddings require optional LM dependencies. "
                "Install them with `uv sync --extra lm`."
            ) from exc

        self.torch = torch
        resolved_device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.resolved_device = resolved_device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.resolved_device)
        self.model.eval()

    def fit(self, texts: list[str]) -> TransformerClsEncoder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        clean_texts = _ensure_nonempty_texts(texts)
        batches: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(clean_texts), self.batch_size):
                batch_texts = clean_texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.resolved_device) for key, value in encoded.items()}
                output = self.model(**encoded)
                pooled = output.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
                batches.append(pooled)
        matrix = np.vstack(batches) if batches else np.zeros((0, self.embedding_dim), dtype=np.float32)
        matrix = normalize(matrix, norm="l2", axis=1, copy=False)
        return _pad_or_trim(matrix, self.embedding_dim)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)


def build_text_encoder(
    *,
    backend: str,
    embedding_dim: int,
    model_name: str,
    batch_size: int,
    device: str,
    max_length: int,
) -> TextEncoder:
    if backend == "tfidf_svd":
        return TfidfSvdEncoder(embedding_dim=embedding_dim)
    if backend == "transformer_cls":
        return TransformerClsEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported text embedding backend: {backend}")


def _ensure_nonempty_texts(texts: list[str]) -> list[str]:
    out = [" ".join(str(text).split()) for text in texts]
    return [text if text else "none" for text in out]


def _pad_or_trim(matrix: np.ndarray, dim: int) -> np.ndarray:
    if matrix.shape[1] == dim:
        return matrix.astype(np.float32)
    if matrix.shape[1] > dim:
        return matrix[:, :dim].astype(np.float32)
    pad_width = dim - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)
