"""Dense vector retrieval over SEC filing chunks.

Uses sentence-transformers for local embeddings (no extra API key required) and
NumPy cosine similarity for search. Each indexed chunk keeps a reference to its
source filing so retrieved passages can be cited back to the original document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from sec import Filing

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Passage:
    text: str
    filing: Filing
    chunk_index: int


@dataclass
class Hit:
    passage: Passage
    score: float


class DenseRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self._passages: list[Passage] = []
        self._matrix: np.ndarray | None = None

    def add(self, filing: Filing, chunks: Iterable[str]) -> None:
        for i, chunk in enumerate(chunks):
            self._passages.append(Passage(text=chunk, filing=filing, chunk_index=i))

    def build(self) -> None:
        if not self._passages:
            raise ValueError("No passages to index. Call add() first.")
        embeddings = self.model.encode(
            [p.text for p in self._passages],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._matrix = np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("Index not built. Call build() first.")
        q = self.model.encode([query], normalize_embeddings=True)
        scores = (self._matrix @ np.asarray(q, dtype=np.float32).T).squeeze(-1)
        top_idx = np.argsort(-scores)[:top_k]
        return [Hit(passage=self._passages[i], score=float(scores[i])) for i in top_idx]

    @property
    def size(self) -> int:
        return len(self._passages)
