"""Sentence-transformers embedder with batch support."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Semantic embedding requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model {self.model_name!r}. "
                "Check network/model cache, or install/download it manually."
            ) from exc

    def embed(self, texts: Sequence[str]) -> "object":
        if not texts:
            return []
        model = self._load()
        return model.encode(list(texts), normalize_embeddings=True)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list:
        if not texts:
            return []
        if len(texts) <= batch_size:
            return list(self.embed(texts))
        model = self._load()
        all_embeddings = []
        text_list = list(texts)
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            batch_embeddings = model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
