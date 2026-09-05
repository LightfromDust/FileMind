"""Keyword and semantic search service."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from filemind.config import FileMindConfig
from filemind.indexer.embedder import Embedder
from filemind.indexer.vector_store import VectorStore
from filemind.storage.chunk_repository import ChunkRepository
from filemind.storage.database import get_connection, init_database


QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "费用": ("金额", "总费用", "支付"),
    "发票金额": ("合计金额", "价税合计", "金额"),
    "服务内容": ("服务范围", "项目内容", "第一条"),
    "论文": ("abstract", "introduction", "references", "paper"),
    "优化": ("调优", "性能优化", "optimization"),
}

DOMAIN_KEYWORDS = (
    "GPU",
    "CUDA",
    "合同",
    "服务内容",
    "服务范围",
    "项目内容",
    "费用",
    "总费用",
    "支付",
    "发票金额",
    "发票",
    "合计金额",
    "价税合计",
    "金额",
    "论文",
    "优化",
    "性能优化",
    "调优",
)

CATEGORY_BOOSTS = (
    ("论文", "论文资料", 4.0),
    ("paper", "论文资料", 4.0),
    ("合同", "合同文件", 4.0),
    ("contract", "合同文件", 4.0),
    ("发票", "发票/报销", 4.0),
    ("报销", "发票/报销", 3.0),
)


class SearchService:
    def __init__(
        self,
        config: FileMindConfig | None = None,
        db_path: str | Path | None = None,
        vector_index_path: str | Path | None = None,
    ):
        self.config = config or FileMindConfig()
        self.db_path = db_path or self.config.db_path
        self.chunks = ChunkRepository(self.db_path)
        self.embedder = Embedder(self.config.embedding_model)
        self.vector_store = VectorStore(vector_index_path or self.config.vector_index_path)
        init_database(self.db_path)

    def search_files(self, query: str, top_k: int = 5, *, semantic: bool = True) -> list[dict[str, Any]]:
        combined: dict[int, dict[str, Any]] = {}
        mode = "semantic" if semantic else "keyword_fallback"
        for row in self._keyword_search(query, limit=max(top_k * 5, 20)):
            self._merge_result(combined, row, float(row.get("_score", 0.5)), mode)
        if semantic:
            try:
                query_embedding = self.embedder.embed([query])
                for hit in self.vector_store.search(query_embedding, top_k=max(top_k * 3, 10)):
                    row = self._chunk_with_file(int(hit["chunk_id"]))
                    if row:
                        self._merge_result(combined, row, float(hit["score"]), "semantic")
            except Exception:
                combined.clear()
                mode = "keyword_fallback"
                for row in self._keyword_search(query, limit=max(top_k * 5, 20)):
                    self._merge_result(combined, row, float(row.get("_score", 0.5)), mode)
        return sorted(combined.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def search_chunks(self, query: str, top_k: int = 5, *, semantic: bool = True) -> list[dict[str, Any]]:
        mode = "semantic" if semantic else "keyword_fallback"
        hits: list[dict[str, Any]] = []
        if semantic:
            try:
                query_embedding = self.embedder.embed([query])
                for hit in self.vector_store.search(query_embedding, top_k=top_k):
                    row = self._chunk_with_file(int(hit["chunk_id"]))
                    if row:
                        hits.append(self._row_to_result(row, float(hit["score"]), "semantic"))
                if hits:
                    return hits[:top_k]
            except Exception:
                mode = "keyword_fallback"
        return [
            self._row_to_result(row, float(row.get("_score", 0.5)), mode)
            for row in self._keyword_search(query, limit=top_k)
        ][:top_k]

    def _keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = self._query_terms(query)
        if not terms:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        for term in terms:
            pattern = f"%{term}%"
            clauses.append(
                """
                (
                    files.file_name LIKE ?
                    OR files.file_path LIKE ?
                    OR files.summary LIKE ?
                    OR files.keywords LIKE ?
                    OR files.category LIKE ?
                    OR chunks.content LIKE ?
                )
                """
            )
            params.extend([pattern, pattern, pattern, pattern, pattern, pattern])
        params.append(max(limit * 5, 50))
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    chunks.id AS chunk_id,
                    files.id AS file_id,
                    files.file_name,
                    files.file_path,
                    files.summary,
                    files.keywords,
                    files.category,
                    chunks.content
                FROM files
                LEFT JOIN chunks ON chunks.file_id = files.id
                WHERE {" OR ".join(clauses)}
                LIMIT ?
                """,
                params,
            ).fetchall()
        scored = [self._score_keyword_row(dict(row), query, terms) for row in rows]
        scored = [row for row in scored if row["_score"] > 0]
        scored.sort(key=lambda item: item["_score"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        raw_terms = [query.strip()]
        raw_terms.extend(re.findall(r"[A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", query))
        lower_query = query.lower()
        for keyword in DOMAIN_KEYWORDS:
            if keyword.lower() in lower_query:
                raw_terms.append(keyword)
                raw_terms.extend(QUERY_SYNONYMS.get(keyword, ()))
        if "费用" in query:
            raw_terms.extend(QUERY_SYNONYMS["费用"])
        if "发票" in query and "金额" in query:
            raw_terms.extend(QUERY_SYNONYMS["发票金额"])
        if "服务" in query and "内容" in query:
            raw_terms.extend(QUERY_SYNONYMS["服务内容"])
        if "论文" in query:
            raw_terms.extend(QUERY_SYNONYMS["论文"])
        if "优化" in query:
            raw_terms.extend(QUERY_SYNONYMS["优化"])

        seen: set[str] = set()
        terms: list[str] = []
        for term in raw_terms:
            normalized = term.strip(" ，。？！?；;：:")
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                terms.append(normalized)
        return terms

    @staticmethod
    def expected_category(query: str) -> str | None:
        lower = query.lower()
        for needle, category, _boost in CATEGORY_BOOSTS:
            if needle.lower() in lower:
                return category
        return None

    @staticmethod
    def _score_keyword_row(row: dict[str, Any], query: str, terms: list[str]) -> dict[str, Any]:
        fields = {
            "file_name": str(row.get("file_name") or ""),
            "file_path": str(row.get("file_path") or ""),
            "summary": str(row.get("summary") or ""),
            "keywords": str(row.get("keywords") or ""),
            "category": str(row.get("category") or ""),
            "content": str(row.get("content") or ""),
        }
        haystack = "\n".join(fields.values()).lower()
        query_norm = query.strip().lower()
        score = 0.0
        if query_norm and query_norm in haystack:
            score += 8.0

        matched_terms: list[str] = []
        for term in terms:
            t = term.lower()
            if not t:
                continue
            term_hits = 0
            for field, value in fields.items():
                if t in value.lower():
                    term_hits += 1
                    if field == "content":
                        score += 1.6
                    elif field in {"summary", "keywords"}:
                        score += 1.3
                    elif field == "file_name":
                        score += 1.1
                    elif field == "category":
                        score += 1.0
                    else:
                        score += 0.6
            if term_hits:
                matched_terms.append(term)
                score += min(1.0, term_hits * 0.2)

        lower_query = query.lower()
        for needle, target_category, boost in CATEGORY_BOOSTS:
            if needle.lower() in lower_query and fields["category"] == target_category:
                score += boost

        row["_score"] = round(score, 4)
        row["_matched_terms"] = matched_terms
        return row

    def _chunk_with_file(self, chunk_id: int) -> dict[str, Any] | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    chunks.*,
                    files.file_name,
                    files.file_path,
                    files.summary,
                    files.keywords,
                    files.category
                FROM chunks JOIN files ON chunks.file_id = files.id
                WHERE chunks.id = ?
                """,
                (chunk_id,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def _merge_result(
        combined: dict[int, dict[str, Any]],
        row: dict[str, Any],
        score: float,
        search_mode: str,
    ) -> None:
        file_id = int(row["file_id"])
        current = combined.get(file_id)
        result = SearchService._row_to_result(row, score, search_mode)
        if current is None or score > float(current["score"]):
            combined[file_id] = result

    @staticmethod
    def _row_to_result(row: dict[str, Any], score: float, search_mode: str) -> dict[str, Any]:
        return {
            "file_name": row.get("file_name"),
            "file_path": row.get("file_path"),
            "summary": row.get("summary"),
            "matched_chunk": row.get("content"),
            "score": score,
            "category": row.get("category"),
            "search_mode": search_mode,
            "matched_terms": row.get("_matched_terms", []),
        }
