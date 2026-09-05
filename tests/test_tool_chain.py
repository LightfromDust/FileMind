"""End-to-end multi-tool chain.

Substantiates the claim: "将文件扫描、解析、搜索、分类、重命名、归档等能力封装为
Agent Tools，支持连续调用多个工具完成复杂文件整理流程"（整体任务完成率的
可复现下界：本测试的 6 步链路必须全部成功）。

Simulates the natural-language task "帮我整理这个目录，先找到发票，总结一下，
再规划重命名和归档" as a deterministic 6-step tool sequence. Steps 5 and 6
run in dry-run mode (the default) and must leave the filesystem untouched.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from filemind.indexer.embedder import Embedder
from filemind.tools import (
    archive_files_tool,
    file_qa_tool,
    rename_files_tool,
    scan_directory_tool,
    search_files_tool,
    summarize_file_tool,
)

MESSY_FILES = {
    "invoice_gpu_server.txt": (
        "GPU服务器采购发票 编号 INV-GPU-088 价税合计 89,000 元 报销凭证。\n"
        "商品名称 GPU服务器 数量 2 单价 44,500 元。"
    ),
    "contract_cloud_service.txt": (
        "服务合同 甲方上海云创科技 乙方北京智达信息 GPU集群优化服务内容 "
        "总费用 120,000 元 违约金条款。"
    ),
    "meeting_notes.txt": "会议纪要 团队周会 产品迭代讨论 下周计划。",
    "data_clean.py": "class DataCleaner:\n    def clean(self, rows):\n        return [r for r in rows if r]\n",
}


def _tmp() -> Path:
    path = Path("tests") / "_runtime" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _setup_env(monkeypatch, tmp: Path) -> Path:
    workspace = tmp / "messy"
    workspace.mkdir()
    for name, content in MESSY_FILES.items():
        (workspace / name).write_text(content, encoding="utf-8")
    monkeypatch.setenv("FILEMIND_DB_PATH", str(tmp / "filemind.db"))
    monkeypatch.setenv("FILEMIND_VECTOR_INDEX_PATH", str(tmp / "faiss.index"))
    monkeypatch.setenv("FILEMIND_WORKSPACE_DIR", str(tmp / "workspace"))

    def fail_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(Embedder, "embed", fail_embed)
    return workspace


def test_file_management_tool_chain_completes(monkeypatch):
    tmp = _tmp()
    workspace = _setup_env(monkeypatch, tmp)
    completed = 0

    # Step 1 — scan: parse, summarize, classify and index the directory.
    scan = scan_directory_tool(str(workspace), recursive=True)
    assert scan["ok"], scan
    assert scan["result"]["indexed"] == len(MESSY_FILES)
    assert scan["result"]["failed"] == 0
    completed += 1

    # Step 2 — search: locate the invoice by natural-language-ish query.
    search = search_files_tool("GPU服务器采购 发票 价税合计", top_k=3)
    assert search["ok"], search
    assert search["results"][0]["file_name"] == "invoice_gpu_server.txt"
    completed += 1

    # Step 3 — Q&A: ask about the invoice amount (extractive fallback, no LLM).
    qa = file_qa_tool("发票 价税合计 金额", top_k=3)
    assert qa["ok"], qa
    assert any(
        "invoice_gpu_server.txt" in (source.get("file_name") or "")
        for source in qa["source_files"]
    )
    completed += 1

    # Step 4 — summarize one file found by the previous steps.
    summary = summarize_file_tool(str(workspace / "invoice_gpu_server.txt"))
    assert summary["ok"], summary
    assert summary["category"] == "发票/报销"
    completed += 1

    # Step 5 — rename plan (dry-run): nothing moved on disk.
    rename = rename_files_tool(str(workspace), dry_run=True)
    assert rename["ok"], rename
    assert len(rename["rename_plan"]) == len(MESSY_FILES)
    assert all(plan["status"] == "planned" for plan in rename["rename_plan"])
    assert all((workspace / name).exists() for name in MESSY_FILES)
    completed += 1

    # Step 6 — archive plan (dry-run): categories routed, nothing moved on disk.
    archive_root = tmp / "archive"
    archive = archive_files_tool(str(workspace), str(archive_root), dry_run=True)
    assert archive["ok"], archive
    assert len(archive["archive_plan"]) == len(MESSY_FILES)
    assert all(plan["status"] == "planned" for plan in archive["archive_plan"])
    assert {plan["category"] for plan in archive["archive_plan"]} >= {"发票/报销", "合同文件", "项目代码"}
    assert all((workspace / name).exists() for name in MESSY_FILES)
    assert not archive_root.exists()
    completed += 1

    assert completed == 6, "every step of the tool chain must succeed"
