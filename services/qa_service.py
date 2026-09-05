"""RAG-style file question answering."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from filemind.config import FileMindConfig
from filemind.services.search_service import SearchService

_LLM_QA_TIMEOUT = 30


class QAService:
    def __init__(
        self,
        config: FileMindConfig | None = None,
        *,
        search_service: SearchService | None = None,
        llm_client: Any | None = None,
        model: str | None = None,
    ):
        self.config = config or FileMindConfig.from_env()
        self.search_service = search_service or SearchService(self.config)
        self.llm_client = llm_client
        self.model = model

    async def answer_question(self, question: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.search_service.search_chunks(question, top_k=max(top_k * 2, 10), semantic=True)
        hits = self._filter_category_mismatch(question, hits)[:top_k]
        matched_chunks = [
            {
                "file_name": hit.get("file_name"),
                "file_path": hit.get("file_path"),
                "content": hit.get("matched_chunk") or "",
                "score": hit.get("score"),
                "category": hit.get("category"),
                "matched_terms": hit.get("matched_terms", []),
            }
            for hit in hits
        ]
        search_mode = self._search_mode(hits)
        source_files = self._source_files(matched_chunks)
        match_reasons = [
            self._match_reason(question, chunk["content"], chunk["file_name"], chunk.get("matched_terms"))
            for chunk in matched_chunks
        ]

        answer = await self._llm_answer(question, matched_chunks)
        if not answer:
            answer = self._extractive_answer(question, matched_chunks)

        return {
            "answer": answer,
            "source_files": source_files,
            "matched_chunks": matched_chunks,
            "match_reasons": match_reasons,
            "search_mode": search_mode,
        }

    @staticmethod
    def _filter_category_mismatch(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expected = SearchService.expected_category(question)
        if not expected:
            return hits
        matching = [hit for hit in hits if hit.get("category") == expected]
        return matching if matching else hits

    async def _llm_answer(self, question: str, matched_chunks: list[dict[str, Any]]) -> str | None:
        if not self.llm_client or not matched_chunks:
            return None
        context = self._build_context(matched_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的文件分析师。请严格根据下方提供的文件片段回答用户问题。\n"
                    "规则：\n"
                    "1. 只使用提供的文件片段中的信息作答，不要编造或推测。\n"
                    "2. 在回答中用 [1]、[2] 等编号标注信息来源对应的文件。\n"
                    "3. 如果文件片段中的证据不足以完整回答问题，请明确指出哪些部分缺乏证据。\n"
                    "4. 回答应简洁、结构化，使用要点列表或短段落。\n"
                    "5. 如果片段完全不相关，请直接说明「提供的文件片段中未找到相关信息」。"
                ),
            },
            {
                "role": "user",
                "content": f"## 问题\n{question}\n\n## 召回的文件片段\n{context}",
            },
        ]
        try:
            coro = self._call_llm(messages)
            response = await asyncio.wait_for(coro, timeout=_LLM_QA_TIMEOUT)
            content = getattr(response, "content", None)
            return content.strip() if isinstance(content, str) and content.strip() else None
        except Exception:
            return None

    async def _call_llm(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=None,
            model=self.model,
            max_tokens=1200,
            temperature=0.2,
            tool_choice="none",
        )
        if hasattr(self.llm_client, "chat_with_retry"):
            return await self.llm_client.chat_with_retry(**kwargs)
        if hasattr(self.llm_client, "chat"):
            return await self.llm_client.chat(**kwargs)
        return None

    @staticmethod
    def _build_context(matched_chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(matched_chunks, start=1):
            text = str(chunk.get("content") or "")[:1600]
            source = QAService._format_citation(index, chunk.get("file_name"), chunk.get("file_path"))
            parts.append(f"{source}\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_citation(index: int, file_name: Any, file_path: Any) -> str:
        name = str(file_name or "unknown")
        path = str(file_path or "")
        if path and path != name:
            return f"[{index}] {name} ({path})"
        return f"[{index}] {name}"

    @staticmethod
    def _extractive_answer(question: str, matched_chunks: list[dict[str, Any]]) -> str:
        if not matched_chunks:
            return "没有找到足够相关的文件片段来回答这个问题。"

        q = question.lower()
        if "合同" in q and ("费用" in q or "服务" in q):
            terms = ("服务内容", "服务范围", "项目内容", "第一条", "费用", "总费用", "支付", "人民币")
            return QAService._focused_answer("合同要点", matched_chunks, terms)
        if "发票" in q and "金额" in q:
            terms = ("合计金额", "价税合计", "金额", "税额", "人民币", "¥")
            return QAService._focused_answer("发票金额", matched_chunks, terms)
        if "论文" in q or "paper" in q:
            terms = ("abstract", "introduction", "conclusion", "摘要", "引言", "结论", "优化", "GPU", "CUDA")
            return QAService._focused_answer("论文内容", matched_chunks, terms)

        terms = tuple(str(term) for chunk in matched_chunks for term in (chunk.get("matched_terms") or []))
        return QAService._focused_answer("基于召回片段", matched_chunks, terms or ("",))

    @staticmethod
    def _focused_answer(title: str, matched_chunks: list[dict[str, Any]], terms: tuple[str, ...]) -> str:
        lines: list[str] = []
        for chunk in matched_chunks[:5]:
            sentences = QAService._relevant_sentences(str(chunk.get("content") or ""), terms)
            if sentences:
                lines.append(f"{chunk.get('file_name')}: " + "；".join(sentences[:4]))
        if not lines:
            for chunk in matched_chunks[:2]:
                text = " ".join(str(chunk.get("content") or "").split())
                if text:
                    lines.append(f"{chunk.get('file_name')}: {text[:220]}")
        return f"{title}：\n" + "\n".join(lines) if lines else "没有找到足够相关的文件片段来回答这个问题。"

    @staticmethod
    def _relevant_sentences(text: str, terms: tuple[str, ...]) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s*|\n+", normalized)
            if part.strip()
        ]
        selected: list[str] = []
        lower_terms = [term.lower() for term in terms if term]
        for index, sentence in enumerate(sentences):
            lower = sentence.lower()
            if any(term in lower for term in lower_terms):
                window = sentences[max(0, index - 1) : min(len(sentences), index + 2)]
                for item in window:
                    if item not in selected:
                        selected.append(item)
        return selected[:6]

    @staticmethod
    def _source_files(matched_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        sources: list[dict[str, str]] = []
        for chunk in matched_chunks:
            path = str(chunk.get("file_path") or "")
            if path and path not in seen:
                seen.add(path)
                sources.append({"file_name": str(chunk.get("file_name") or ""), "file_path": path})
        return sources

    @staticmethod
    def _search_mode(hits: list[dict[str, Any]]) -> str:
        modes = {str(hit.get("search_mode") or "") for hit in hits}
        if "semantic" in modes:
            return "semantic"
        return "keyword_fallback"

    @staticmethod
    def _match_reason(question: str, content: str, file_name: Any, matched_terms: Any = None) -> str:
        if matched_terms:
            return f"{file_name}: matched query terms {', '.join(list(matched_terms)[:5])}."
        q_terms = {term.lower() for term in question.split() if len(term) > 1}
        lower = content.lower()
        matched = [term for term in q_terms if term in lower]
        if matched:
            return f"{file_name}: matched query terms {', '.join(matched[:5])}."
        return f"{file_name}: retrieved as a high-scoring relevant chunk."
