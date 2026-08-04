"""V1 placeholder archive service."""

from __future__ import annotations

from pathlib import Path

from filemind.safety.conflict_checker import ConflictChecker


class ArchiveService:
    def plan_target(self, source_path: str, target_root: str, category: str) -> Path:
        target_dir = Path(target_root).expanduser().resolve() / category
        return ConflictChecker().safe_path(target_dir / Path(source_path).name)
