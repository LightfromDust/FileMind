"""Single-file summary tool wrapper with optional LLM enhancement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from filemind.config import FileMindConfig
from filemind.parser import DEFAULT_PARSERS
from filemind.services.classify_service import ClassifyService
from filemind.services.summarize_service import SummarizeService


def summarize_file_tool(
    file_path: str,
    llm_client: Any | None = None,
    model: str | None = None,
) -> dict[str, object]:
    try:
        FileMindConfig.from_env().ensure_dirs()
        path = Path(file_path).expanduser().resolve()
        parser = next((item for item in DEFAULT_PARSERS if item.can_parse(str(path))), None)
        if parser is None:
            return {"ok": False, "error": f"Unsupported file type: {path.suffix}"}
        text = parser.parse(str(path))
        summarizer = SummarizeService(llm_client=llm_client, model=model)
        classifier = ClassifyService(llm_client=llm_client, model=model)
        classification = classifier.classify_multi(path.name, text)

        if llm_client:
            try:
                smart = asyncio.run(summarizer.smart_summarize(text, path.name))
                summary = smart.get("summary", summarizer.summarize(text))
                keywords = smart.get("keywords", summarizer.keywords(path.name, text))
                source = smart.get("source", "unknown")
            except Exception:
                summary = summarizer.summarize(text)
                keywords = summarizer.keywords(path.name, text)
                source = "rule_based_fallback"
        else:
            summary = summarizer.summarize(text)
            keywords = summarizer.keywords(path.name, text)
            source = "rule_based"

        return {
            "ok": True,
            "summary": summary,
            "keywords": keywords,
            "category": classification["final_category"],
            "classification": classification,
            "summary_source": source,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
