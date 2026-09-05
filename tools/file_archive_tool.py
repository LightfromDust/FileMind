"""Safe archive planning/execution tool wrapper."""

from __future__ import annotations

import shutil
from pathlib import Path

from filemind.config import FileMindConfig
from filemind.safety.conflict_checker import ConflictChecker
from filemind.safety.path_guard import PathGuard
from filemind.storage.file_repository import FileRepository
from filemind.storage.operation_log_repository import OperationLogRepository


def archive_files_tool(source_dir: str, target_root: str, dry_run: bool = True) -> dict[str, object]:
    try:
        guard = PathGuard()
        source_root = guard.normalize(source_dir)
        archive_root = guard.normalize(target_root)
        repo = FileRepository(FileMindConfig.from_env().db_path)
        conflicts = ConflictChecker()
        logs = OperationLogRepository(FileMindConfig.from_env().db_path)
        plans: list[dict[str, object]] = []
        for item in repo.list_files(directory=source_root, limit=1000):
            category = str(item.get("category") or "其他文件")
            source = guard.ensure_allowed(item["file_path"], [source_root])
            target_dir = archive_root / category
            target = conflicts.safe_path(target_dir / source.name)
            guard.ensure_allowed(target, [archive_root])
            plan = {
                "source_path": str(source),
                "target_path": str(target),
                "category": category,
                "status": "planned" if dry_run else "archived",
            }
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
                logs.insert_log(
                    "archive",
                    source_path=str(source),
                    target_path=str(target),
                    status="success",
                    rollback_info={"source_path": str(source), "target_path": str(target)},
                )
            plans.append(plan)
        return {"ok": True, "dry_run": dry_run, "archive_plan": plans}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
