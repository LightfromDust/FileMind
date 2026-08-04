"""Repository for chunks table with thread-safe write operations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filemind.storage.database import get_connection, get_write_lock, init_database


class ChunkRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        init_database(db_path)

    def insert_chunks(self, file_id: int, chunks: list[dict[str, Any]]) -> list[int]:
        now = datetime.now(timezone.utc).isoformat()
        ids: list[int] = []
        with get_write_lock():
            with get_connection(self.db_path, check_same_thread=False) as conn:
                conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                for chunk in chunks:
                    cur = conn.execute(
                        """
                        INSERT INTO chunks (file_id, chunk_index, content, embedding_id, created_time)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (file_id, chunk["chunk_index"], chunk["content"], chunk.get("embedding_id"), now),
                    )
                    ids.append(int(cur.lastrowid))
        return ids

    def set_embedding_ids(self, chunk_ids: list[int], embedding_ids: list[int]) -> None:
        with get_write_lock():
            with get_connection(self.db_path, check_same_thread=False) as conn:
                for chunk_id, embedding_id in zip(chunk_ids, embedding_ids):
                    conn.execute(
                        "UPDATE chunks SET embedding_id = ? WHERE id = ?",
                        (embedding_id, chunk_id),
                    )

    def get_chunks_by_file_id(self, file_id: int) -> list[dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE file_id = ? ORDER BY chunk_index", (file_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def search_chunks_keyword(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT chunks.*, files.file_name, files.file_path, files.summary, files.category
                FROM chunks JOIN files ON chunks.file_id = files.id
                WHERE chunks.content LIKE ? OR files.file_name LIKE ? OR files.summary LIKE ?
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit),
            ).fetchall()
            return [dict(row) for row in rows]
