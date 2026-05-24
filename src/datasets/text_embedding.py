from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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
