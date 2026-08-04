"""Repository for files table with thread-safe write operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from filemind.storage.database import get_connection, get_write_lock, init_database


class FileRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        init_database(db_path)

    def upsert_file(self, metadata: dict[str, Any]) -> int:
        columns = [
            "file_name",
            "file_path",
            "file_ext",
            "file_type",
            "file_size",
            "created_time",
            "modified_time",
            "content_hash",
            "summary",
            "keywords",
            "category",
            "indexed_time",
            "status",
        ]
        values = [metadata.get(col) for col in columns]
        update = ", ".join(f"{col}=excluded.{col}" for col in columns if col != "file_path")
        with get_write_lock():
            with get_connection(self.db_path, check_same_thread=False) as conn:
                cur = conn.execute(
                    f"""
                    INSERT INTO files ({", ".join(columns)})
                    VALUES ({", ".join("?" for _ in columns)})
                    ON CONFLICT(file_path) DO UPDATE SET {update}
                    """,
                    values,
                )
                if cur.lastrowid:
                    return int(cur.lastrowid)
                row = conn.execute("SELECT id FROM files WHERE file_path = ?", (metadata["file_path"],)).fetchone()
                return int(row["id"])

    def get_file_by_path(self, path: str | Path) -> dict[str, Any] | None:
        file_path = str(Path(path).expanduser().resolve())
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM files WHERE file_path = ?", (file_path,)).fetchone()
            return dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
            return dict(row) if row else None

    def list_files(
        self,
        *,
        status: str | None = "active",
        category: str | None = None,
        directory: str | Path | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append("category = ?")
            params.append(category)
        if directory:
            root = str(Path(directory).expanduser().resolve())
            clauses.append("file_path LIKE ?")
            params.append(root.rstrip("\\/") + "%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM files {where} ORDER BY modified_time DESC LIMIT ?", params
            ).fetchall()
            return [dict(row) for row in rows]
