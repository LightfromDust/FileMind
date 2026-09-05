"""Incremental indexing benchmark.

Substantiates the claim: "基于文件路径、修改时间、大小和哈希值判断文件变更，
避免重复解析和重复向量化，在 1000+ 文件、约 5% 文件发生变更的场景下，
相比全量重建索引，索引更新耗时大幅降低"（宣传值约 91%）。

Deterministic part: unchanged files are reported as skipped, changed files re-indexed.
Timing part: incremental scan must be at least 2x faster than a full rebuild
(robust lower bound; the measured ratio is printed for the record).
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from filemind.config import FileMindConfig
from filemind.indexer.embedder import Embedder
from filemind.scanner.file_scanner import FileScanner

N_FILES = 1000            # "1000+ files" scenario
N_CHANGED = N_FILES // 20  # ~5% of files change

TEMPLATES = (
    ("invoice_{i:04d}.txt", "增值税普通发票 编号 INV-{i:04d} 合计金额 {i} 元 价税合计 报销"),
    ("contract_{i:04d}.txt", "服务合同 甲方 乙方 项目内容 费用 违约金 编号 C-{i:04d}"),
    ("paper_{i:04d}.txt", "Abstract Introduction References 编号 P-{i:04d} 实验 结果"),
    ("resume_{i:04d}.txt", "简历 教育经历 项目经历 工作经历 编号 R-{i:04d}"),
    ("notes_{i:04d}.txt", "工作笔记 编号 N-{i:04d} 待办事项 会议纪要"),
)


def _tmp() -> Path:
    path = Path("tests") / "_runtime" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _make_config(tmp: Path) -> FileMindConfig:
    return FileMindConfig(
        db_path=tmp / "filemind.db",
        vector_index_path=tmp / "faiss.index",
        workspace_dir=tmp / "workspace",
    )


def _make_corpus(root: Path) -> None:
    for i in range(N_FILES):
        name, template = TEMPLATES[i % len(TEMPLATES)]
        (root / name.format(i=i)).write_text(template.format(i=i) + "\n", encoding="utf-8")


def _disable_embedding(monkeypatch) -> None:
    def fail_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(Embedder, "embed", fail_embed)


def test_incremental_scan_skips_all_unchanged_files(monkeypatch):
    _disable_embedding(monkeypatch)
    tmp = _tmp()
    config = _make_config(tmp)
    corpus = tmp / "corpus"
    corpus.mkdir()
    _make_corpus(corpus)

    first = FileScanner(config, semantic_index=False).scan(corpus)
    assert first["scanned"] == N_FILES
    assert first["indexed"] == N_FILES
    assert first["failed"] == 0

    second = FileScanner(config, semantic_index=False).scan(corpus)
    assert second["skipped"] == N_FILES
    assert second["indexed"] == 0
    assert second["updated"] == 0
    assert second["failed"] == 0


def test_incremental_update_beats_full_rebuild(monkeypatch):
    """3 rounds of each phase, compared on medians to damp timing noise."""
    _disable_embedding(monkeypatch)
    tmp = _tmp()
    config_incremental = _make_config(tmp / "incr")
    (tmp / "incr").mkdir(parents=True, exist_ok=True)
    corpus = tmp / "corpus"
    corpus.mkdir()
    _make_corpus(corpus)

    # Production state: an existing index built by a previous scan.
    FileScanner(config_incremental, semantic_index=False).scan(corpus)

    # Full-rebuild cost, 3 runs on fresh databases -> median.
    full_times: list[float] = []
    for round_index in range(3):
        config_rebuild = _make_config(tmp / f"full{round_index}")
        start = time.perf_counter()
        stats = FileScanner(config_rebuild, semantic_index=False).scan(corpus)
        full_times.append(time.perf_counter() - start)
        assert stats["indexed"] == N_FILES and stats["failed"] == 0
    t_full = sorted(full_times)[1]

    # Incremental cost: 3 rounds, each changing the *next* 5% of files.
    incremental_times: list[float] = []
    for round_index in range(3):
        first = round_index * N_CHANGED
        for i in range(first, first + N_CHANGED):
            name, _ = TEMPLATES[i % len(TEMPLATES)]
            target = corpus / name.format(i=i)
            target.write_text(
                target.read_text(encoding="utf-8") + f"变更内容 round{round_index}-{i}\n",
                encoding="utf-8",
            )
        start = time.perf_counter()
        stats = FileScanner(config_incremental, semantic_index=False).scan(corpus)
        incremental_times.append(time.perf_counter() - start)
        assert stats["skipped"] == N_FILES - N_CHANGED, stats
        assert stats["updated"] + stats["indexed"] == N_CHANGED, stats
        assert stats["failed"] == 0
    t_incremental = sorted(incremental_times)[1]

    reduction = 1.0 - (t_incremental / t_full)
    print(
        f"\n[incremental benchmark] files={N_FILES} changed_per_round={N_CHANGED} "
        f"full(median of {['%.2fs' % t for t in full_times]})={t_full:.2f}s "
        f"incremental(median of {['%.2fs' % t for t in incremental_times]})={t_incremental:.2f}s "
        f"reduction={reduction:.1%}"
    )
    # Robust lower bound; the advertised number is ~91%.
    assert t_incremental < t_full * 0.5, (
        f"incremental median ({t_incremental:.2f}s) should be clearly cheaper than "
        f"full-rebuild median ({t_full:.2f}s)"
    )
