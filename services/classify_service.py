"""Multi-strategy file classification with LLM enhancement."""

from __future__ import annotations

import asyncio
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from filemind.indexer.embedder import Embedder

_LLM_CLASSIFY_TIMEOUT = 15

CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "发票/报销": ("invoice", "发票", "税额", "金额", "报销", "receipt", "reimbursement"),
    "合同文件": ("contract", "合同", "甲方", "乙方", "租期", "违约", "agreement"),
    "简历求职": ("resume", "cv", "简历", "教育经历", "项目经历", "工作经历", "求职"),
    "论文资料": ("abstract", "introduction", "references", "摘要", "关键词", "参考文献", "paper"),
}

EXTENSION_RULES = {
    "项目代码": {".py", ".java", ".c", ".cpp", ".js"},
    "表格数据": {".csv", ".xlsx"},
    "图片资料": {".jpg", ".jpeg", ".png"},
}

CATEGORY_PROTOTYPES = {
    "发票/报销": "invoice receipt tax amount reimbursement expense 发票 税额 金额 报销",
    "合同文件": "contract agreement party breach lease 合同 甲方 乙方 租期 违约",
    "简历求职": "resume cv education work experience project 简历 教育经历 项目经历 工作经历",
    "论文资料": "academic paper abstract introduction references 摘要 关键词 参考文献",
    "项目代码": "source code function class module python java javascript project",
    "表格数据": "spreadsheet table csv xlsx rows columns data",
    "图片资料": "image photo picture jpg png jpeg",
    "其他文件": "general document miscellaneous file",
}

CATEGORY_LIST = list(CATEGORY_PROTOTYPES.keys())


class ClassifyService:
    def __init__(self, embedder: Embedder | None = None, llm_client: Any | None = None, model: str | None = None):
        self.embedder = embedder or Embedder()
        self.llm_client = llm_client
        self.model = model

    def classify(self, file_name: str, text: str = "") -> str:
        return str(self.classify_multi(file_name, text)["final_category"])

    def classify_multi(self, file_name: str, text: str = "") -> dict[str, Any]:
        ext = Path(file_name).suffix.lower()
        haystack_name = file_name.lower()
        haystack_text = (text or "")[:5000].lower()
        votes: list[dict[str, Any]] = []

        ext_vote = self._extension_vote(ext)
        if ext_vote:
            votes.append(ext_vote)

        votes.extend(self._keyword_votes("file_name_keywords", haystack_name, weight=0.8))
        votes.extend(self._keyword_votes("content_summary_keywords", haystack_text, weight=0.7))

        embedding_vote = self._embedding_vote(f"{file_name}\n{(text or '')[:1200]}")
        if embedding_vote:
            votes.append(embedding_vote)

        llm_vote: dict[str, Any] | None = None
        if self.llm_client:
            try:
                llm_vote = asyncio.run(self._llm_vote(file_name, text))
            except Exception:
                llm_vote = None
        if llm_vote:
            votes.append(llm_vote)

        if not votes:
            votes.append(
                {
                    "strategy": "default",
                    "category": "其他文件",
                    "confidence": 0.35,
                    "evidence": "No strategy matched; defaulted to other files.",
                }
            )

        scores: Counter[str] = Counter()
        for vote in votes:
            cat = str(vote["category"])
            conf = float(vote.get("confidence", 0.0))
            if llm_vote and vote is llm_vote and conf >= 0.8:
                conf = min(conf * 2, 0.99)
            scores[cat] += conf
        final_category, score = scores.most_common(1)[0]
        max_possible = max(1.0, sum(float(vote.get("confidence", 0.0)) for vote in votes))
        confidence = min(0.99, max(0.1, score / max_possible))

        return {
            "final_category": final_category,
            "confidence": round(confidence, 3),
            "evidence": [vote.get("evidence", "") for vote in votes if vote.get("category") == final_category],
            "strategy_votes": votes,
        }

    @staticmethod
    def _extension_vote(ext: str) -> dict[str, Any] | None:
        for category, extensions in EXTENSION_RULES.items():
            if ext in extensions:
                return {
                    "strategy": "extension_rule",
                    "category": category,
                    "confidence": 0.95,
                    "evidence": f"Extension {ext} maps to {category}.",
                }
        return None

    @staticmethod
    def _keyword_votes(strategy: str, haystack: str, weight: float) -> list[dict[str, Any]]:
        votes: list[dict[str, Any]] = []
        for category, keywords in CATEGORY_RULES.items():
            matches = [kw for kw in keywords if kw.lower() in haystack]
            if matches:
                confidence = min(0.95, weight + 0.03 * min(len(matches), 5))
                votes.append(
                    {
                        "strategy": strategy,
                        "category": category,
                        "confidence": confidence,
                        "evidence": f"Matched keywords: {', '.join(matches[:6])}.",
                    }
                )
        return votes

    def _embedding_vote(self, text: str) -> dict[str, Any] | None:
        if not text.strip():
            return None
        try:
            import numpy as np  # type: ignore

            labels = list(CATEGORY_PROTOTYPES)
            embeddings = self.embedder.embed([text] + [CATEGORY_PROTOTYPES[label] for label in labels])
            vectors = np.asarray(embeddings, dtype="float32")
            query = vectors[0]
            prototypes = vectors[1:]
            scores = prototypes @ query
            best_index = int(scores.argmax())
            score = float(scores[best_index])
            if score < 0.35:
                return None
            return {
                "strategy": "embedding_prototype_similarity",
                "category": labels[best_index],
                "confidence": min(0.9, max(0.4, score)),
                "evidence": f"Closest category prototype: {labels[best_index]} ({score:.3f}).",
            }
        except Exception:
            return None

    async def _llm_vote(self, file_name: str, text: str) -> dict[str, Any] | None:
        if not self.llm_client:
            return None
        prompt = (
            f"根据以下文件名和内容摘要，将其分类到以下类别之一：{', '.join(CATEGORY_LIST)}。\n"
            f"文件名：{file_name}\n"
            f"内容摘要：{text[:800] if text else '(无内容)'}\n"
            "请严格以JSON格式返回，不要添加任何其他说明文字：\n"
            '{"category": "类别名称", "confidence": 0.0-1.0, "evidence": "简要说明"}'
        )
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的文件分类专家。请严格按JSON格式返回分类结果，不要添加其他说明文字。",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            coro = self._call_llm(messages)
            response = await asyncio.wait_for(coro, timeout=_LLM_CLASSIFY_TIMEOUT)
            content = getattr(response, "content", None)
            if not content:
                return None
            result = self._parse_json_response(content)
            if result and result.get("category"):
                return {
                    "strategy": "llm_classification",
                    "category": str(result["category"]),
                    "confidence": float(result.get("confidence", 0.75)),
                    "evidence": str(result.get("evidence", "LLM classification result.")),
                }
        except Exception:
            pass
        return None

    async def _call_llm(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            tools=None,
            model=self.model,
            max_tokens=300,
            temperature=0.1,
            tool_choice="none",
        )
        if hasattr(self.llm_client, "chat_with_retry"):
            return await self.llm_client.chat_with_retry(**kwargs)
        if hasattr(self.llm_client, "chat"):
            return await self.llm_client.chat(**kwargs)
        return None

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any] | None:
        content = content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
            if data.get("category"):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None
