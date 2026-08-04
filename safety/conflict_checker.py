"""Conflict-free target path generation."""

from __future__ import annotations

from pathlib import Path


class ConflictChecker:
    def safe_path(self, target_path: str | Path) -> Path:
        path = Path(target_path).expanduser().resolve()
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        for index in range(1, 10_000):
            candidate = path.with_name(f"{stem}_{index}{suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Unable to find non-conflicting path for {target_path}")
