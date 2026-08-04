"""Directory scan tool wrapper."""

from __future__ import annotations

from filemind.config import FileAgentConfig
from filemind.scanner.file_scanner import FileScanner


def scan_directory_tool(directory: str, recursive: bool = True) -> dict[str, object]:
    try:
        return {
            "ok": True,
            "result": FileScanner(FileAgentConfig.from_env()).scan(directory, recursive=recursive),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
