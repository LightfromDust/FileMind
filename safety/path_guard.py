"""Path safety guard for move/rename operations."""

from __future__ import annotations

from pathlib import Path


class PathGuard:
    blocked_roots = [
        Path("C:/Windows"),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
        Path("C:/ProgramData"),
        Path("/bin"),
        Path("/sbin"),
        Path("/usr"),
        Path("/etc"),
        Path("/System"),
        Path("/Library"),
    ]

    def normalize(self, path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    def ensure_allowed(self, path: str | Path, allowed_roots: list[str | Path]) -> Path:
        resolved = self.normalize(path)
        if any(self._is_under(resolved, root.resolve()) for root in self.blocked_roots):
            raise PermissionError(f"Refusing to operate in system directory: {resolved}")
        roots = [self.normalize(root) for root in allowed_roots]
        if not any(self._is_under(resolved, root) for root in roots):
            raise PermissionError(f"Path is outside allowed roots: {resolved}")
        return resolved

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
