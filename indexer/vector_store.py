"""FAISS vector store with embedding-id to chunk-id mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filemind.config import FileAgentConfig


class VectorStore:
    def __init__(self, index_path: str | Path | None = None):
        self.index_path = Path(index_path).expanduser() if index_path else FileAgentConfig().vector_index_path
        self.mapping_path = self.index_path.with_suffix(self.index_path.suffix + ".map.json")
        self._index = None
        self._mapping: list[int] = []

    def _imports(self):
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires faiss-cpu and numpy. Install with: pip install faiss-cpu numpy"
            ) from exc
        return faiss, np

    def load(self) -> None:
        faiss, _ = self._imports()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        if self.mapping_path.exists():
            self._mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self._index is None:
            return
        faiss, _ = self._imports()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.mapping_path.write_text(json.dumps(self._mapping), encoding="utf-8")

    def add_embeddings(self, embeddings: Any, chunk_ids: list[int]) -> list[int]:
        faiss, np = self._imports()
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self._index is None:
            self.load()
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        start = len(self._mapping)
        self._index.add(vectors)
        self._mapping.extend(chunk_ids)
        self.save()
        return list(range(start, start + len(chunk_ids)))

    def search(self, query_embedding: Any, top_k: int = 5) -> list[dict[str, float | int]]:
        _, np = self._imports()
        if self._index is None:
            self.load()
        if self._index is None or not self._mapping:
            return []
        vector = np.asarray(query_embedding, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, ids = self._index.search(vector, top_k)
        results: list[dict[str, float | int]] = []
        for embedding_id, score in zip(ids[0], scores[0]):
            if embedding_id < 0 or embedding_id >= len(self._mapping):
                continue
            results.append(
                {
                    "embedding_id": int(embedding_id),
                    "chunk_id": int(self._mapping[int(embedding_id)]),
                    "score": float(score),
                }
            )
        return results
