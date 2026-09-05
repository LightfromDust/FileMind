"""Directory scanner and indexing orchestration with concurrency support."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filemind.config import FileMindConfig
from filemind.indexer.index_service import IndexService
from filemind.parser import DEFAULT_PARSERS, BaseParser
from filemind.scanner.metadata_extractor import MetadataExtractor
from filemind.services.classify_service import ClassifyService
from filemind.services.summarize_service import SummarizeService
from filemind.storage.file_repository import FileRepository


class FileScanner:
    temp_prefixes = ("~$", ".~",)
    temp_suffixes = (".tmp", ".temp", ".part", ".crdownload")

    def __init__(
        self,
        config: FileMindConfig | None = None,
        *,
        db_path: str | Path | None = None,
        parsers: list[BaseParser] | None = None,
        semantic_index: bool = True,
        max_workers: int = 4,
        llm_client: Any | None = None,
        model: str | None = None,
    ):
        self.config = config or FileMindConfig()
        if db_path:
            self.config.db_path = Path(db_path).expanduser()
        self.config.ensure_dirs()
        self.files = FileRepository(self.config.db_path)
        self.extractor = MetadataExtractor()
        self.parsers = parsers or DEFAULT_PARSERS
        self.summarizer = SummarizeService(llm_client=llm_client, model=model)
        self.classifier = ClassifyService(llm_client=llm_client, model=model)
        self.indexer = IndexService(self.config)
        self.semantic_index = semantic_index
        self.max_workers = max_workers

    def scan(self, directory: str | Path, recursive: bool | None = None) -> dict[str, Any]:
        root = Path(directory).expanduser().resolve()
        stats: dict[str, Any] = {"scanned": 0, "indexed": 0, "updated": 0, "skipped": 0, "failed": 0, "errors": []}
        if not root.exists() or not root.is_dir():
            return {**stats, "failed": 1, "errors": [f"Directory not found: {root}"]}
        iterator = root.rglob("*") if (self.config.recursive_scan if recursive is None else recursive) else root.iterdir()
        candidates = [path for path in iterator if path.is_file() and not self._should_skip(path)]
        stats["scanned"] = len(candidates)

        if self.max_workers <= 1 or len(candidates) <= 1:
            for path in candidates:
                try:
                    result = self._scan_file(path)
                    stats[result] += 1
                except Exception as exc:
                    stats["failed"] += 1
                    stats["errors"].append(f"{path}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._scan_file, path): path for path in candidates}
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                        stats[result] += 1
                    except Exception as exc:
                        stats["failed"] += 1
                        stats["errors"].append(f"{path}: {exc}")
        return stats

    def _scan_file(self, path: Path) -> str:
        metadata = self.extractor.extract(path)
        existing = self.files.get_file_by_path(path)
        unchanged = existing and all(
            existing.get(key) == metadata.get(key)
            for key in ("file_size", "modified_time", "content_hash")
        )
        if unchanged:
            return "skipped"

        text = self._parse_text(path)
        summary = self.summarizer.summarize(text)
        keywords = self.summarizer.keywords(path.name, text)
        classification = self.classifier.classify_multi(path.name, text)
        category = str(classification["final_category"])
        metadata.update(
            {
                "summary": summary,
                "keywords": keywords,
                "category": category,
                "indexed_time": datetime.now(timezone.utc).isoformat(),
            }
        )
        file_id = self.files.upsert_file(metadata)
        if text:
            try:
                self.indexer.index_text(file_id, text, semantic=self.semantic_index)
            except RuntimeError:
                self.indexer.index_text(file_id, text, semantic=False)
        return "updated" if existing else "indexed"

    def _parse_text(self, path: Path) -> str:
        parser = next((item for item in self.parsers if item.can_parse(str(path))), None)
        if parser is None:
            return ""
        return parser.parse(str(path))

    def _should_skip(self, path: Path) -> bool:
        name = path.name
        if name.startswith(".") or name.startswith(self.temp_prefixes):
            return True
        if path.suffix.lower() not in self.config.supported_extensions:
            return True
        if name.lower() in {"thumbs.db", "desktop.ini", ".ds_store"}:
            return True
        if name.lower().endswith(self.temp_suffixes):
            return True
        try:
            return path.stat().st_size > self.config.max_file_size
        except OSError:
            return True
