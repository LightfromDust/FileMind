"""Command line interface for FileMind.

All commands print JSON so they can be consumed by scripts, other agents,
or CI pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from filemind.config import FileMindConfig
from filemind.parser import DEFAULT_PARSERS
from filemind.scanner.file_scanner import FileScanner
from filemind.services.archive_service import ArchiveService
from filemind.services.classify_service import ClassifyService
from filemind.services.rename_service import RenameService
from filemind.storage.file_repository import FileRepository
from filemind.tools import search_files_tool, summarize_file_tool
from filemind.tools.file_qa_tool import file_qa_tool


def _json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _semantic_enabled() -> bool:
    # FILEMIND_SEMANTIC_INDEX is canonical; FILE_AGENT_SEMANTIC_INDEX is the
    # legacy name kept for backward compatibility.
    value = os.environ.get("FILEMIND_SEMANTIC_INDEX") or os.environ.get(
        "FILE_AGENT_SEMANTIC_INDEX", "1"
    )
    return value.lower() not in {"0", "false", "no"}


def _parse_text(path: Path) -> str:
    parser = next((item for item in DEFAULT_PARSERS if item.can_parse(str(path))), None)
    if parser is None:
        return ""
    return parser.parse(str(path))


def _classify_path(target: str) -> dict[str, Any]:
    path = Path(target).expanduser().resolve()
    classifier = ClassifyService()
    if path.is_file():
        text = _parse_text(path)
        return {
            "ok": True,
            "results": [{"file_path": str(path), **classifier.classify_multi(path.name, text)}],
        }
    if path.is_dir():
        config = FileMindConfig.from_env()
        files = FileRepository(config.db_path).list_files(directory=path, limit=1000)
        results = [
            {"file_path": item["file_path"], "file_name": item["file_name"], "category": item["category"]}
            for item in files
        ]
        return {"ok": True, "results": results}
    return {"ok": False, "error": f"Path not found: {path}"}


def _rename_plan(directory: str) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    repo = FileRepository(FileMindConfig.from_env().db_path)
    renamer = RenameService()
    plans: list[dict[str, Any]] = []
    for item in repo.list_files(directory=root, limit=1000):
        old_path = Path(item["file_path"])
        new_path = old_path.with_name(renamer.suggest_name(item))
        plans.append(
            {
                "old_path": str(old_path),
                "new_path": str(new_path),
                "reason": "日期_类别_主题.后缀",
                "conflict_resolved": new_path.exists(),
            }
        )
    return {"ok": True, "rename_plan": plans}


def _archive_plan(source_dir: str, target_root: str) -> dict[str, Any]:
    root = Path(source_dir).expanduser().resolve()
    repo = FileRepository(FileMindConfig.from_env().db_path)
    archiver = ArchiveService()
    plans: list[dict[str, Any]] = []
    for item in repo.list_files(directory=root, limit=1000):
        category = item.get("category") or "其他文件"
        target = archiver.plan_target(item["file_path"], target_root, category)
        plans.append(
            {
                "source_path": item["file_path"],
                "target_path": str(target),
                "category": category,
                "status": "planned",
            }
        )
    return {"ok": True, "archive_plan": plans}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m filemind")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan")
    scan.add_argument("directory")
    scan.add_argument("--no-recursive", action="store_true")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)

    qa = sub.add_parser("qa")
    qa.add_argument("question")
    qa.add_argument("--top-k", type=int, default=5)

    summary = sub.add_parser("summary")
    summary.add_argument("file_path")

    classify = sub.add_parser("classify")
    classify.add_argument("target")

    rename = sub.add_parser("rename-plan")
    rename.add_argument("directory")

    archive = sub.add_parser("archive-plan")
    archive.add_argument("source_dir")
    archive.add_argument("target_root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan":
            config = FileMindConfig.from_env()
            result = FileScanner(config, semantic_index=_semantic_enabled()).scan(
                args.directory,
                recursive=not args.no_recursive,
            )
            _json({"ok": True, "result": result})
        elif args.command == "search":
            _json(search_files_tool(args.query, top_k=args.top_k))
        elif args.command == "qa":
            _json(file_qa_tool(args.question, top_k=args.top_k))
        elif args.command == "summary":
            _json(summarize_file_tool(args.file_path))
        elif args.command == "classify":
            _json(_classify_path(args.target))
        elif args.command == "rename-plan":
            _json(_rename_plan(args.directory))
        elif args.command == "archive-plan":
            _json(_archive_plan(args.source_dir, args.target_root))
        return 0
    except Exception as exc:
        _json({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
