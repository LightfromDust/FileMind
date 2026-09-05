"""Path safety guard: dangerous paths must be refused everywhere they are enforced.

Substantiates the claim: "对批量文件操作引入 dry-run、冲突检测、路径越权检测、
日志记录和回滚机制…在异常测试中拦截所有危险操作".

Covers the previously untested PathGuard component, both at unit level and
through the tool layer, including a lexical-prefix escape (a DB record whose
stored path starts with the allowed root but resolves outside it).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from filemind.safety.path_guard import PathGuard
from filemind.storage.file_repository import FileRepository
from filemind.tools import archive_files_tool, rename_files_tool


def _tmp() -> Path:
    path = Path("tests") / "_runtime" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _seed_file(db_path: Path, file_path: Path, name: str = None, category: str = "其他文件") -> None:
    FileRepository(db_path).upsert_file(
        {
            "file_name": name or file_path.name,
            "file_path": str(file_path),
            "file_ext": file_path.suffix,
            "file_type": "text/plain",
            "file_size": 16,
            "created_time": "2026-01-01T00:00:00+00:00",
            "modified_time": "2026-01-01T00:00:00+00:00",
            "content_hash": str(file_path),
            "summary": "seed",
            "keywords": "seed",
            "category": category,
            "indexed_time": "2026-01-01T00:00:00+00:00",
            "status": "active",
        }
    )


def test_accepts_paths_inside_allowed_roots():
    tmp = _tmp()
    root = tmp / "allowed"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("x", encoding="utf-8")

    resolved = PathGuard().ensure_allowed(target, [root])

    assert resolved == target.resolve()


def test_rejects_paths_outside_allowed_roots():
    tmp = _tmp()
    root_a = tmp / "root_a"
    root_a.mkdir()
    outsider = tmp / "root_b" / "secret.txt"
    outsider.parent.mkdir()
    outsider.write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        PathGuard().ensure_allowed(outsider, [root_a])


def test_rejects_parent_directory_escape():
    tmp = _tmp()
    root = tmp / "allowed"
    root.mkdir()
    escape = root / ".." / "escaped.txt"  # resolves outside the allowed root

    with pytest.raises(PermissionError):
        PathGuard().ensure_allowed(escape, [root])


def test_rejects_blocked_system_roots_even_when_allowed(monkeypatch):
    tmp = _tmp()
    system_like = tmp / "Windows"
    system_like.mkdir()
    target = system_like / "system32.txt"
    target.write_text("x", encoding="utf-8")

    monkeypatch.setattr(PathGuard, "blocked_roots", [system_like])
    with pytest.raises(PermissionError, match="system directory"):
        PathGuard().ensure_allowed(target, [system_like])


def test_rename_tool_blocks_lexical_escape_record(monkeypatch):
    tmp = _tmp()
    root = tmp / "docs"
    root.mkdir()
    db_path = tmp / "filemind.db"

    # Record whose stored path starts with the root (passes the SQL LIKE
    # filter) but resolves outside it once `..` components are normalized.
    escape = root / "sub" / ".." / ".." / "stolen.txt"
    _seed_file(db_path, escape)

    monkeypatch.setenv("FILEMIND_DB_PATH", str(db_path))
    monkeypatch.setenv("FILEMIND_VECTOR_INDEX_PATH", str(tmp / "faiss.index"))
    monkeypatch.setenv("FILEMIND_WORKSPACE_DIR", str(tmp / "workspace"))

    result = rename_files_tool(str(root), dry_run=False)

    assert result["ok"] is False
    assert "outside allowed roots" in result["error"]
    assert not (tmp / "stolen.txt").exists()


def test_archive_tool_refuses_blocked_target_root(monkeypatch):
    tmp = _tmp()
    source = tmp / "docs"
    source.mkdir()
    doc = source / "invoice.txt"
    doc.write_text("发票 金额", encoding="utf-8")
    db_path = tmp / "filemind.db"
    _seed_file(db_path, doc, category="发票/报销")

    archive_root = tmp / "Windows" / "archive"
    monkeypatch.setattr(PathGuard, "blocked_roots", [tmp / "Windows"])
    monkeypatch.setenv("FILEMIND_DB_PATH", str(db_path))
    monkeypatch.setenv("FILEMIND_VECTOR_INDEX_PATH", str(tmp / "faiss.index"))
    monkeypatch.setenv("FILEMIND_WORKSPACE_DIR", str(tmp / "workspace"))

    result = archive_files_tool(str(source), str(archive_root), dry_run=False)

    assert result["ok"] is False
    assert "system directory" in result["error"]
    assert doc.exists(), "source file must not be moved when the target is refused"
