"""RAG question answering tool wrapper."""

from __future__ import annotations

import asyncio

from filemind.config import FileAgentConfig
from filemind.services.qa_service import QAService


def file_qa_tool(question: str, top_k: int = 5) -> dict[str, object]:
    try:
        result = asyncio.run(QAService(FileAgentConfig.from_env()).answer_question(question, top_k=top_k))
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
