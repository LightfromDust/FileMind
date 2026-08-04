"""File metadata extraction."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


class MetadataExtractor:
    def extract(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path).expanduser().resolve()
        stat = path.stat()
        ext = path.suffix.lower()
        return {
            "file_name": path.name,
            "file_path": str(path),
            "file_ext": ext,
            "file_type": mimetypes.guess_type(path.name)[0] or ext.lstrip(".") or "unknown",
            "file_size": stat.st_size,
            "created_time": _iso_timestamp(stat.st_ctime),
            "modified_time": _iso_timestamp(stat.st_mtime),
            "content_hash": self.sha256(path),
            "status": "active",
        }

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
