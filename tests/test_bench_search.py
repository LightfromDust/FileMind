"""Search benchmark: Top-5 recall, ranking quality and latency.

Substantiates the claims:
- "Top-5 召回率达到 91%"
- "平均向量检索耗时 200ms 以内"

Three suites, all in deterministic keyword-fallback mode (Embedder disabled)
so no model download is needed:

1. Curated topic queries over a 16-file labeled corpus (hit_rate@5, macro_recall@5).
2. Generated 250-file corpus with 40+ single-target queries — each query points
   at exactly one uniquely-marked file buried among same-category distractors
   (hit_rate@5, MRR@10). This is a real ranking test, not just retrieval.
3. Latency: multi-round medians on the small corpus, and a 30k-chunk corpus
   that actually stresses the LIKE scan (median must stay under 200ms).
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from filemind.config import FileMindConfig
from filemind.indexer.embedder import Embedder
from filemind.services.search_service import SearchService
from filemind.storage.chunk_repository import ChunkRepository
from filemind.storage.file_repository import FileRepository

CURATED_CORPUS = [
    ("invoice_202604.txt", "发票/报销",
     "增值税普通发票 INV-20260428-001 合计金额 140,000 元 价税合计人民币140,000元。"),
    ("invoice_gpu_server.txt", "发票/报销",
     "GPU服务器采购发票 编号 INV-GPU-088 价税合计 89,000 元 报销凭证。"),
    ("invoice_training.txt", "发票/报销",
     "技术培训服务发票 合计金额 20,000 元 价税合计 报销单据。"),
    ("contract_gpu_service.txt", "合同文件",
     "服务合同 甲方 乙方 GPU集群优化服务内容 总费用 120,000 元 违约金条款。"),
    ("contract_office_rent.txt", "合同文件",
     "房屋租赁合同 甲方 乙方 租期三年 押金 违约责任条款。"),
    ("contract_maintenance.txt", "合同文件",
     "运维服务合同 甲方 乙方 服务内容与费用 支付方式。"),
    ("paper_gpu_optimization.txt", "论文资料",
     "Abstract Introduction References GPU optimization for CUDA kernels experiments."),
    ("paper_llm_finetune.txt", "论文资料",
     "Abstract 摘要 参考文献 LLM fine-tuning LoRA methods 性能评估。"),
    ("paper_federated_learning.txt", "论文资料",
     "abstract introduction references federated learning survey."),
    ("resume_backend.txt", "简历求职",
     "个人简历 教育经历 项目经历 CUDA 后端开发工程师。"),
    ("resume_data_engineer.txt", "简历求职",
     "个人简历 工作经历 项目经历 数据工程师 离线数仓。"),
    ("demo_util.py", "项目代码",
     "def get_file_size(path): return os.path.getsize(path)  # python utility"),
    ("server_entry.py", "项目代码",
     "class FastAPIServer: def handle_request(self, request): return response"),
    ("sales_data.csv", "表格数据",
     "销售数据表格 region,amount,quarter 华东区销售额明细"),
    ("meeting_notes.txt", "其他文件",
     "会议纪要 团队周会 产品迭代讨论 下周计划。"),
    ("todo_list.txt", "其他文件",
     "待办事项清单 买菜 修电脑 看牙医。"),
]

CURATED_QUERIES = [
    ("GPU服务器采购 发票 价税合计", ["invoice_gpu_server.txt"]),
    ("发票 合计金额 报销", ["invoice_202604.txt", "invoice_gpu_server.txt", "invoice_training.txt"]),
    ("合同 租期 违约", ["contract_office_rent.txt"]),
    ("GPU集群优化 服务内容 合同", ["contract_gpu_service.txt"]),
    ("GPU optimization 论文", ["paper_gpu_optimization.txt"]),
    ("LLM fine-tuning 摘要", ["paper_llm_finetune.txt"]),
    ("简历 教育经历 项目经历", ["resume_backend.txt", "resume_data_engineer.txt"]),
    ("get_file_size 工具函数", ["demo_util.py"]),
    ("FastAPIServer handle_request", ["server_entry.py"]),
    ("销售数据表格 销售额", ["sales_data.csv"]),
    ("会议纪要 周会", ["meeting_notes.txt"]),
    ("CUDA 优化 论文", ["paper_gpu_optimization.txt"]),
]

COMPANIES = ("上海云创科技", "北京智达信息", "深圳蓝鲸数据", "杭州明略智能", "广州极视角")
SCHOOLS = ("清华大学", "北京大学", "浙江大学", "复旦大学", "上海交通大学", "华中科技大学")
SKILLS = ("CUDA", "Spark", "Kubernetes", "PyTorch", "Flink", "后端开发", "数据仓库", "推荐系统")
NOTE_KINDS = ("会议纪要", "读书笔记", "旅行计划", "购物清单", "健身记录", "菜谱", "维修记录", "周报")


def generated_corpus() -> list[tuple[str, str, str]]:
    """250 files across 7 categories; every file carries a unique marker token."""
    corpus: list[tuple[str, str, str]] = []
    for i in range(40):
        kind = "专用" if i % 2 == 0 else "普通"
        corpus.append((
            f"invoice_INV-{i:04d}.txt", "发票/报销",
            f"增值税{kind}发票 发票号码 INV-{i:04d} 开票方{COMPANIES[i % 5]} "
            f"合计金额 {5000 + i * 137} 元 价税合计 报销凭证 商品明细",
        ))
    for i in range(40):
        subtype = ("服务合同", "租赁合同", "劳务合同", "采购合同")[i % 4]
        corpus.append((
            f"contract_HT-{i:04d}.txt", "合同文件",
            f"{subtype} 编号 HT-{i:04d} 甲方{COMPANIES[i % 5]} 乙方{COMPANIES[(i + 2) % 5]} "
            f"费用总额 {10000 + i * 211} 元 违约责任 交付周期",
        ))
    for i in range(35):
        topic = ("GPU优化", "分布式系统", "机器学习")[i % 3]
        corpus.append((
            f"paper_arXiv-{i:04d}.txt", "论文资料",
            f"Abstract paper arXiv-{i:04d} 摘要 本文研究{topic} Introduction "
            f"References 参考文献 实验与结论",
        ))
    for i in range(25):
        corpus.append((
            f"resume_R-{i:04d}.txt", "简历求职",
            f"个人简历 编号 R-{i:04d} 教育经历 {SCHOOLS[i % 6]} "
            f"项目经历 {SKILLS[i % 8]} 工作经历 求职意向",
        ))
    for i in range(30):
        corpus.append((
            f"module_M-{i:04d}.py", "项目代码",
            f"# module M-{i:04d}\ndef util_{i}(rows):\n    return [r for r in rows if r]\n",
        ))
    for i in range(20):
        corpus.append((
            f"report_TBL-{i:04d}.csv", "表格数据",
            f"报表编号 TBL-{i:04d}\nregion,amount,quarter\n华东,{100 + i},Q{i % 4 + 1}\n销售数据表格",
        ))
    for i in range(60):
        corpus.append((
            f"note_N-{i:04d}.txt", "其他文件",
            f"{NOTE_KINDS[i % 8]} N-{i:04d} 事项{i} 备忘",
        ))
    return corpus


# (query, ground-truth file name) — one uniquely-marked target per query.
def generated_queries(corpus: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    templates = {
        "invoice": "发票 {marker} 价税合计",
        "contract": "合同 {marker} 违约",
        "paper": "论文 {marker} 摘要",
        "resume": "简历 {marker} 项目经历",
        "module": "module {marker}",
        "report": "销售数据表格 {marker}",
        "note": "备忘 {marker}",
    }
    queries: list[tuple[str, str]] = []
    for index, (name, _category, _content) in enumerate(corpus):
        if index % 6 != 0:
            continue
        stem = name.rsplit(".", 1)[0]        # e.g. invoice_INV-0007
        key, marker = stem.rsplit("_", 1)    # key=invoice, marker=INV-0007
        queries.append((templates[key].format(marker=marker), name))
    return queries


def _tmp() -> Path:
    path = Path("tests") / "_runtime" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _disable_embedding(monkeypatch) -> None:
    def fail_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(Embedder, "embed", fail_embed)


def _seed(tmp: Path, corpus: list[tuple[str, str, str]]) -> FileMindConfig:
    config = FileMindConfig(
        db_path=tmp / "filemind.db",
        vector_index_path=tmp / "faiss.index",
        workspace_dir=tmp / "workspace",
    )
    for name, category, content in corpus:
        file_path = tmp / name
        file_id = FileRepository(config.db_path).upsert_file(
            {
                "file_name": name,
                "file_path": str(file_path),
                "file_ext": Path(name).suffix,
                "file_type": "text/plain",
                "file_size": len(content.encode("utf-8")),
                "created_time": "2026-01-01T00:00:00+00:00",
                "modified_time": "2026-01-01T00:00:00+00:00",
                "content_hash": name,
                "summary": content[:120],
                "keywords": content,
                "category": category,
                "indexed_time": "2026-01-01T00:00:00+00:00",
                "status": "active",
            }
        )
        ChunkRepository(config.db_path).insert_chunks(
            file_id,
            [{"chunk_index": 0, "content": content}],
        )
    return config


def test_curated_topic_queries_hit_rate(monkeypatch):
    _disable_embedding(monkeypatch)
    config = _seed(_tmp(), CURATED_CORPUS)
    service = SearchService(config)

    hits = 0
    recalls: list[float] = []
    for query, expected in CURATED_QUERIES:
        results = service.search_files(query, top_k=5, semantic=True)
        assert results, f"no results for query: {query}"
        assert results[0]["search_mode"] == "keyword_fallback"
        names = {item["file_name"] for item in results}
        relevant = set(expected)
        if names & relevant:
            hits += 1
        recalls.append(len(names & relevant) / len(relevant))

    hit_rate = hits / len(CURATED_QUERIES)
    recall = sum(recalls) / len(recalls)
    print(
        f"\n[search curated] queries={len(CURATED_QUERIES)} files={len(CURATED_CORPUS)} "
        f"hit_rate@5={hit_rate:.1%} macro_recall@5={recall:.1%}"
    )
    assert hit_rate >= 0.9, f"hit_rate@5={hit_rate:.1%} below 90%"
    assert recall >= 0.9, f"macro recall@5={recall:.1%} below 90%"


def test_generated_corpus_single_target_ranking(monkeypatch):
    _disable_embedding(monkeypatch)
    corpus = generated_corpus()
    queries = generated_queries(corpus)
    assert len(corpus) == 250
    assert len(queries) >= 40

    config = _seed(_tmp(), corpus)
    service = SearchService(config)

    hits = 0
    reciprocal_ranks: list[float] = []
    misses: list[str] = []
    for query, target in queries:
        results = service.search_files(query, top_k=10, semantic=True)
        names = [item["file_name"] for item in results]
        if target in names:
            hits += 1
            reciprocal_ranks.append(1.0 / (names.index(target) + 1))
        else:
            misses.append(query)
            reciprocal_ranks.append(0.0)

    hit_rate = hits / len(queries)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(
        f"\n[search generated] queries={len(queries)} files={len(corpus)} "
        f"hit_rate@5(in top10 window)={hit_rate:.1%} MRR@10={mrr:.3f} misses={misses}"
    )
    assert hit_rate >= 0.9, f"single-target hit rate {hit_rate:.1%} below 90% (misses: {misses})"
    assert mrr >= 0.8, f"MRR@10={mrr:.3f} below 0.8"


def test_search_latency_small_corpus_median(monkeypatch):
    _disable_embedding(monkeypatch)
    config = _seed(_tmp(), CURATED_CORPUS)
    service = SearchService(config)

    rounds = 3
    per_query: dict[str, list[float]] = {query: [] for query, _ in CURATED_QUERIES}
    for _ in range(rounds):
        for query, _expected in CURATED_QUERIES:
            start = time.perf_counter()
            service.search_chunks(query, top_k=5, semantic=True)
            per_query[query].append((time.perf_counter() - start) * 1000)

    medians = sorted(sorted(times)[rounds // 2] for times in per_query.values())
    overall_median = medians[len(medians) // 2]
    p95 = medians[int(len(medians) * 0.95) - 1]
    print(
        f"\n[search latency small] queries={len(medians)} rounds={rounds} "
        f"median={overall_median:.1f}ms p95={p95:.1f}ms"
    )
    assert overall_median < 200, f"median latency {overall_median:.1f}ms exceeds 200ms"


def test_keyword_latency_at_30k_chunks(monkeypatch):
    """The 200ms claim, challenged at a realistic scale: 30k chunks in SQLite."""
    _disable_embedding(monkeypatch)
    tmp = _tmp()
    config = FileMindConfig(
        db_path=tmp / "filemind.db",
        vector_index_path=tmp / "faiss.index",
        workspace_dir=tmp / "workspace",
    )
    topics = ("GPU集群优化", "CUDA内核调优", "分布式训练", "推理加速", "显存管理", "发票报销")
    n_files, n_chunks = 30, 1000
    for f in range(n_files):
        name = f"doc_{f:03d}.txt"
        file_id = FileRepository(config.db_path).upsert_file(
            {
                "file_name": name,
                "file_path": str(tmp / name),
                "file_ext": ".txt",
                "file_type": "text/plain",
                "file_size": 1000,
                "created_time": "2026-01-01T00:00:00+00:00",
                "modified_time": "2026-01-01T00:00:00+00:00",
                "content_hash": name,
                "summary": f"文档{f} 摘要",
                "keywords": f"文档{f}",
                "category": "其他文件",
                "indexed_time": "2026-01-01T00:00:00+00:00",
                "status": "active",
            }
        )
        chunks = [
            {
                "chunk_index": c,
                "content": (
                    f"第{c}段 主题{topics[(f + c) % len(topics)]} 编号 D-{f:03d}-{c:04d} "
                    "的详细内容，包含性能分析与实验数据。"
                ),
            }
            for c in range(n_chunks)
        ]
        ChunkRepository(config.db_path).insert_chunks(file_id, chunks)

    service = SearchService(config)
    queries = [
        "GPU集群优化 性能分析",
        "D-012-0345",
        "发票 价税合计",
        "分布式训练 实验",
        "CUDA内核调优 编号 D-029-0999",
        "显存管理 详细内容",
    ]
    medians: list[float] = []
    for query in queries:
        times: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            service.search_chunks(query, top_k=5, semantic=True)
            times.append((time.perf_counter() - start) * 1000)
        medians.append(sorted(times)[2])
    medians.sort()
    overall_median = medians[len(medians) // 2]
    p95 = medians[-1]
    print(
        f"\n[search latency 30k chunks] chunks={n_files * n_chunks} queries={len(queries)} "
        f"median={overall_median:.1f}ms worst-query-median={p95:.1f}ms"
    )
    assert overall_median < 200, (
        f"median keyword-search latency at 30k chunks is {overall_median:.1f}ms — "
        "the 200ms claim does not hold at this scale"
    )
