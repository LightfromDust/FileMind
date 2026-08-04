"""LLM-enhanced summary and keyword extraction."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_LLM_SUMMARY_TIMEOUT = 20


class SummarizeService:
    def __init__(self, llm_client: Any | None = None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model

    def summarize(self, text: str, max_chars: int = 300) -> str:
        return re.sub(r"\s+", " ", text or "").strip()[:max_chars]

    def keywords(self, file_name: str, text: str, top_k: int = 8) -> str:
        source = f"{Path(file_name).stem} {text[:2000]}"
        words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[一-鿿]{2,}", source.lower())
        stop = {"the", "and", "for", "with", "from", "this", "that", "introduction", "references"}
        ranked = [word for word, _ in Counter(w for w in words if w not in stop).most_common(top_k)]
        return ", ".join(ranked)

    async def smart_summarize(self, text: str, file_name: str, max_chars: int = 300) -> dict[str, str]:
        """Generate summary and keywords via LLM; fall back to rule-based on failure."""
        if not text.strip():
            summary = self.summarize(text, max_chars)
            keywords = self.keywords(file_name, text)
            return {"summary": summary, "keywords": keywords, "source": "rule_based_empty"}

        llm_result = await self._llm_summarize(text, file_name, max_chars)
        if llm_result:
            return llm_result

        summary = self.summarize(text, max_chars)
        keywords = self.keywords(file_name, text)
        return {"summary": summary, "keywords": keywords, "source": "rule_based"}

    async def _llm_summarize(self, text: str, file_name: str, max_chars: int) -> dict[str, str] | None:
        if not self.llm_client:
            return None
        prompt = (
            f"请为以下文件内容生成一段精炼的中文摘要（不超过{max_chars}个字符），"
            f"并提取6-10个关键词，用JSON格式返回：\n"
            f'{{"summary": "摘要文本", "keywords": "keyword1, keyword2, ..."}}\n\n'
            f"文件名：{file_name}\n"
            f"文件内容：\n{text[:3000]}"
        )
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的文档分析师。请严格按JSON格式返回，不要添加其他说明文字。",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            coro = self._call_llm(messages)
            response = await asyncio.wait_for(coro, timeout=_LLM_SUMMARY_TIMEOUT)
            content = getattr(response, "content", None)
            if not content:
                return None
            result = self._parse_json_response(content)
            if result:
                result["source"] = "llm"
            return result
        except Exception:
            return None

    async def _call_llm(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=None,
            model=self.model,
            max_tokens=600,
            temperature=0.2,
            tool_choice="none",
        )
        if hasattr(self.llm_client, "chat_with_retry"):
            return await self.llm_client.chat_with_retry(**kwargs)
        if hasattr(self.llm_client, "chat"):
            return await self.llm_client.chat(**kwargs)
        return None

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, str] | None:
        content = content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
            summary = str(data.get("summary", "")).strip()
            keywords = str(data.get("keywords", "")).strip()
            if not summary:
                return None
            return {"summary": summary, "keywords": keywords}
        except (json.JSONDecodeError, ValueError, KeyError):
            return None