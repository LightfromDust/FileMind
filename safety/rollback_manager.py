"""Rollback manager for move/rename operation logs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from filemind.safety.conflict_checker import ConflictChecker
from filemind.storage.operation_log_repository import OperationLogRepository


class RollbackManager:
    def __init__(self, db_path: str | Path | None = None):
        self.logs = OperationLogRepository(db_path)
        self.conflicts = ConflictChecker()

    def rollback_last(self) -> dict[str, object]:
        for log in self.logs.list_recent_logs(limit=50):
            info = json.loads(log.get("rollback_info") or "{}")
            source = info.get("source_path")
            target = info.get("target_path")
            if not source or not target:
                continue
            current = Path(target)
            original = self.conflicts.safe_path(Path(source))
            if not current.exists():
                return {"status": "skipped", "reason": f"Target no longer exists: {current}"}
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(original))
            self.logs.insert_log(
                "rollback",
                source_path=str(current),
                target_path=str(original),
                status="success",
                rollback_info={},
            )
            return {"status": "success", "restored_path": str(original)}
        return {"status": "skipped", "reason": "No rollbackable operation found"}
