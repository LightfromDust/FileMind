"""Safe rename planning/execution tool wrapper."""

from __future__ import annotations

import shutil
from pathlib import Path

from filemind.config import FileAgentConfig
from filemind.safety.conflict_checker import ConflictChecker
from filemind.safety.path_guard import PathGuard
from filemind.services.rename_service import RenameService
from filemind.storage.file_repository import FileRepository
from filemind.storage.operation_log_repository import OperationLogRepository


def rename_files_tool(directory: str, dry_run: bool = True) -> dict[str, object]:
    try:
        root = PathGuard().normalize(directory)
        repo = FileRepository(FileAgentConfig.from_env().db_path)
        renamer = RenameService()
        conflicts = ConflictChecker()
        logs = OperationLogRepository(FileAgentConfig.from_env().db_path)
        plans: list[dict[str, object]] = []
        for item in repo.list_files(directory=root, limit=1000):
            old_path = PathGuard().ensure_allowed(item["file_path"], [root])
            suggested = old_path.with_name(renamer.suggest_name(item))
            target = conflicts.safe_path(suggested)
            plan = {
                "old_path": str(old_path),
                "new_path": str(target),
                "reason": "Generated from date, category, and topic keywords.",
                "conflict_resolved": target != suggested,
                "status": "planned" if dry_run else "renamed",
            }
            if not dry_run and old_path != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(target))
                logs.insert_log(
                    "rename",
                    source_path=str(old_path),
                    target_path=str(target),
                    status="success",
                    rollback_info={"source_path": str(old_path), "target_path": str(target)},
                )
            plans.append(plan)
        return {"ok": True, "dry_run": dry_run, "rename_plan": plans}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
