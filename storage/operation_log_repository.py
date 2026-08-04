"""Repository for operation logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filemind.storage.database import get_connection, init_database


class OperationLogRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        init_database(db_path)

    def insert_log(
        self,
        operation_type: str,
        source_path: str | None = None,
        target_path: str | None = None,
        status: str | None = None,
        rollback_info: dict[str, Any] | None = None,
    ) -> int:
        with get_connection(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO operation_logs
                    (operation_type, source_path, target_path, status, rollback_info, created_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_type,
                    source_path,
                    target_path,
                    status,
                    json.dumps(rollback_info or {}, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def list_recent_logs(self, limit: int = 20) -> list[dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM operation_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
