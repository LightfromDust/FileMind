"""Search tool wrapper."""

from __future__ import annotations

from filemind.config import FileMindConfig
from filemind.services.search_service import SearchService


def search_files_tool(query: str, top_k: int = 5) -> dict[str, object]:
    try:
        return {"ok": True, "results": SearchService(FileMindConfig.from_env()).search_files(query, top_k=top_k)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
