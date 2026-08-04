"""Index documents into chunks and FAISS with parallel embedding support."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filemind.config import FileAgentConfig
from filemind.indexer.chunker import TextChunker
from filemind.indexer.embedder import Embedder
from filemind.indexer.vector_store import VectorStore
from filemind.storage.chunk_repository import ChunkRepository

_PARALLEL_EMBED_THRESHOLD = 50


class IndexService:
    def __init__(
        self,
        config: FileAgentConfig | None = None,
        db_path: str | Path | None = None,
        vector_index_path: str | Path | None = None,
    ):
        self.config = config or FileAgentConfig()
        self.chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap)
        self.chunks = ChunkRepository(db_path or self.config.db_path)
        self.embedder = Embedder(self.config.embedding_model)
        self.vector_store = VectorStore(vector_index_path or self.config.vector_index_path)

    def index_text(self, file_id: int, text: str, *, semantic: bool = True) -> dict[str, object]:
        chunks = self.chunker.chunk(text)
        chunk_ids = self.chunks.insert_chunks(file_id, chunks)
        result: dict[str, object] = {"chunk_count": len(chunks), "semantic_indexed": False}
        if not chunks or not semantic:
            return result
        texts = [str(chunk["content"]) for chunk in chunks]
        if len(texts) > _PARALLEL_EMBED_THRESHOLD:
            embeddings = self._parallel_embed(texts)
        else:
            embeddings = self.embedder.embed(texts)
        embedding_ids = self.vector_store.add_embeddings(embeddings, chunk_ids)
        self.chunks.set_embedding_ids(chunk_ids, embedding_ids)
        result["semantic_indexed"] = True
        return result

    def _parallel_embed(self, texts: list[str]) -> list:
        mid = len(texts) // 2
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(self.embedder.embed_batch, texts[:mid])
            future_b = pool.submit(self.embedder.embed_batch, texts[mid:])
            result_a = future_a.result()
            result_b = future_b.result()
        return result_a + result_b
