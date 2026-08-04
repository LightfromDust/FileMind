"""V1 placeholder for safe rename planning."""

from __future__ import annotations

import re
from pathlib import Path


WINDOWS_ILLEGAL = r'<>:"/\|?*'


def sanitize_filename(name: str) -> str:
    cleaned = "".join("_" if ch in WINDOWS_ILLEGAL or ord(ch) < 32 else ch for ch in name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    return cleaned or "untitled"


class RenameService:
    def suggest_name(self, metadata: dict[str, object]) -> str:
        ext = str(metadata.get("file_ext") or Path(str(metadata.get("file_name"))).suffix)
        date = str(metadata.get("modified_time") or "")[:10] or "unknown-date"
        category = sanitize_filename(str(metadata.get("category") or "其他文件"))
        topic = sanitize_filename(str(metadata.get("keywords") or Path(str(metadata.get("file_name"))).stem))
        return f"{date}_{category}_{topic}{ext}"
