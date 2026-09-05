"""Classification accuracy benchmark: single-strategy baseline vs multi-strategy ensemble.

Substantiates the claim: "结合文件后缀规则、内容摘要、Embedding 相似度和 LLM 分类结果，
论文/发票/合同/代码等文件识别准确性相比单一方案提高 15% 以上".

The ensemble is measured with Embedder disabled (keyword + extension votes +
default fallback), keeping the benchmark hermetic. The corpus is deliberately
extension-hostile: most business documents are .txt files where extension
rules alone say nothing.
"""

from __future__ import annotations

from pathlib import Path

from filemind.indexer.embedder import Embedder
from filemind.services.classify_service import EXTENSION_RULES, ClassifyService

CORPUS = [
    # (file_name, ground-truth category, content)
    ("invoice_cloud_01.txt", "发票/报销", "增值税普通发票 合计金额 45,000 元 税额 报销单据。"),
    ("invoice_cloud_02.txt", "发票/报销", "增值税专用发票 价税合计 128,000 元 金额 报销凭证。"),
    ("invoice_cloud_03.txt", "发票/报销", "receipt invoice total amount 8,900 reimbursement。"),
    ("invoice_cloud_04.txt", "发票/报销", "差旅费发票 金额 3,200 元 报销 附明细。"),
    ("invoice_cloud_05.txt", "发票/报销", "办公用品采购发票 合计金额 1,540 元 税额。"),
    ("invoice_cloud_06.txt", "发票/报销", "云服务账单发票 金额 6,800 元 价税合计。"),
    ("contract_vendor_01.txt", "合同文件", "采购合同 甲方 乙方 违约责任 交付周期 条款。"),
    ("contract_vendor_02.txt", "合同文件", "劳务合同 甲方 乙方 合同期限 保密条款。"),
    ("contract_vendor_03.txt", "合同文件", "软件开发合同 甲方 乙方 验收标准 违约金。"),
    ("contract_vendor_04.txt", "合同文件", "agreement contract party breach delivery terms。"),
    ("contract_vendor_05.txt", "合同文件", "租赁合同 甲方 乙方 租期 押金 违约。"),
    ("contract_vendor_06.txt", "合同文件", "服务合同 甲方 乙方 费用 支付 违约条款。"),
    ("paper_nlp_01.txt", "论文资料", "Abstract 本文提出一种新方法 Introduction References 参考文献。"),
    ("paper_nlp_02.txt", "论文资料", "摘要 关键词 实验对比分析 参考文献 结论 展望。"),
    ("paper_sys_03.txt", "论文资料", "abstract introduction references related work evaluation。"),
    ("paper_sys_04.txt", "论文资料", "paper abstract 摘要 introduction 参考文献 方法。"),
    ("resume_dev_01.txt", "简历求职", "个人简历 教育经历 项目经历 求职意向 后端开发。"),
    ("resume_dev_02.txt", "简历求职", "简历 工作经历 项目经历 专业技能 自我评价。"),
    ("resume_dev_03.txt", "简历求职", "resume cv education work experience projects。"),
    ("note_random_01.txt", "其他文件", "购物清单 牛奶 面包 鸡蛋。"),
    ("note_random_02.txt", "其他文件", "读书笔记 第三章要点整理。"),
    ("pipeline_helper.py", "项目代码", "def run_pipeline(): pass  # python module"),
    ("data_clean.py", "项目代码", "class DataCleaner: def clean(self): return None"),
    ("quarterly_report.csv", "表格数据", "region,amount,quarter\nnorth,1200,Q1"),
]


def _disable_embedding(monkeypatch) -> None:
    def fail_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(Embedder, "embed", fail_embed)


def _extension_only(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    for category, extensions in EXTENSION_RULES.items():
        if ext in extensions:
            return category
    return "其他文件"


VENDORS = ("上海云创科技", "北京智达信息", "深圳蓝鲸数据", "杭州明略智能", "广州极视角")
NOTE_SUBJECTS = ("会议纪要", "读书笔记", "旅行计划", "购物清单", "健身记录", "装修记录")


def generated_corpus() -> list[tuple[str, str, str]]:
    """200 files in realistic proportions plus tricky edge cases.

    Unlike the curated adversarial corpus above, the extension mix here
    mirrors a plausible real directory, so the extension-only baseline is
    not artificially crushed (it hovers around 45-50%).
    """
    corpus: list[tuple[str, str, str]] = []
    for i in range(30):
        corpus.append((
            f"发票_{i:03d}.txt", "发票/报销",
            f"增值税发票 发票号码 F{i:04d} 开票方{VENDORS[i % 5]} 合计金额 {3000 + i * 97} 元 税额 报销",
        ))
    for i in range(30):
        subtype = ("服务", "租赁", "劳务", "采购")[i % 4]
        corpus.append((
            f"{subtype}合同_{i:03d}.txt", "合同文件",
            f"{subtype}合同 编号 H{i:04d} 甲方{VENDORS[i % 5]} 乙方{VENDORS[(i + 1) % 5]} 违约责任 条款",
        ))
    for i in range(25):
        corpus.append((
            f"研究笔记论文_{i:03d}.txt", "论文资料",
            f"Abstract 摘要 论文编号 P{i:04d} Introduction References 参考文献 实验结论",
        ))
    for i in range(20):
        corpus.append((
            f"候选人简历_{i:03d}.txt", "简历求职",
            f"个人简历 教育经历 项目经历 工作经历 编号 R{i:04d} 求职意向",
        ))
    for i in range(25):
        corpus.append((
            f"helper_{i:03d}.py", "项目代码",
            f"# helper {i:03d}\ndef run_{i}(items):\n    return sorted(items)\n",
        ))
    for i in range(15):
        corpus.append((
            f"export_{i:03d}.csv", "表格数据",
            f"id,name,value\n{i},row{i},{i * 10}\n数据导出",
        ))
    for i in range(55):
        corpus.append((
            f"note_{i:03d}.txt" if i % 2 else f"note_{i:03d}.md", "其他文件",
            f"{NOTE_SUBJECTS[i % 6]} 备忘 事项{i}",
        ))
    # Tricky cases: name-only evidence, English-only content, no signal at all.
    corpus.append(("2024年度发票汇总.txt", "发票/报销", ""))
    corpus.append(("receipt_0513.txt", "发票/报销", "receipt invoice total amount reimbursement"))
    corpus.append(("agreement_draft.txt", "合同文件", "agreement contract party breach terms"))
    corpus.append(("随机文件.txt", "其他文件", "今天天气不错"))
    return corpus


def test_multi_strategy_beats_extension_only_baseline(monkeypatch):
    _disable_embedding(monkeypatch)
    classifier = ClassifyService()

    baseline_correct = 0
    ensemble_correct = 0
    for file_name, expected, content in CORPUS:
        if _extension_only(file_name) == expected:
            baseline_correct += 1
        result = classifier.classify_multi(file_name, content)
        if result["final_category"] == expected:
            ensemble_correct += 1

    baseline_acc = baseline_correct / len(CORPUS)
    ensemble_acc = ensemble_correct / len(CORPUS)
    improvement = ensemble_acc - baseline_acc
    print(
        f"\n[classify benchmark] files={len(CORPUS)} "
        f"extension_only={baseline_acc:.1%} multi_strategy={ensemble_acc:.1%} "
        f"improvement={improvement:.1%}"
    )
    assert ensemble_acc >= 0.9, f"multi-strategy accuracy {ensemble_acc:.1%} below 90%"
    assert improvement >= 0.15, (
        f"multi-strategy improvement over extension-only baseline is "
        f"{improvement:.1%}, below the claimed 15pp"
    )


def test_generated_corpus_multi_strategy_beats_baseline(monkeypatch):
    """200-file realistic-mix corpus; per-category accuracy for both strategies."""
    _disable_embedding(monkeypatch)
    classifier = ClassifyService()
    corpus = generated_corpus()
    assert len(corpus) == 204

    baseline_hits: dict[str, list[int]] = {}
    ensemble_hits: dict[str, list[int]] = {}
    for file_name, expected, content in corpus:
        baseline_hits.setdefault(expected, [0, 0])
        ensemble_hits.setdefault(expected, [0, 0])
        baseline_hits[expected][1] += 1
        ensemble_hits[expected][1] += 1
        if _extension_only(file_name) == expected:
            baseline_hits[expected][0] += 1
        if classifier.classify_multi(file_name, content)["final_category"] == expected:
            ensemble_hits[expected][0] += 1

    def _accuracy(hits: dict[str, list[int]]) -> float:
        correct = sum(ok for ok, _total in hits.values())
        total = sum(total for _ok, total in hits.values())
        return correct / total

    baseline_acc = _accuracy(baseline_hits)
    ensemble_acc = _accuracy(ensemble_hits)
    per_category = {
        cat: f"{ensemble_hits[cat][0]}/{ensemble_hits[cat][1]}"
        for cat in ensemble_hits
    }
    print(
        f"\n[classify generated] files={len(corpus)} "
        f"extension_only={baseline_acc:.1%} multi_strategy={ensemble_acc:.1%} "
        f"improvement={ensemble_acc - baseline_acc:.1%} per_category={per_category}"
    )
    assert ensemble_acc >= 0.9, f"multi-strategy accuracy {ensemble_acc:.1%} below 90%"
    assert ensemble_acc - baseline_acc >= 0.15, (
        f"improvement over extension-only baseline is {ensemble_acc - baseline_acc:.1%}, "
        "below the claimed 15pp"
    )


def test_classify_multi_returns_strategy_votes(monkeypatch):
    _disable_embedding(monkeypatch)
    result = ClassifyService().classify_multi(
        "invoice_march.txt", "增值税发票 合计金额 9,999 元 报销"
    )

    assert result["final_category"] == "发票/报销"
    assert 0.0 < result["confidence"] <= 1.0
    assert result["strategy_votes"], "expected at least one strategy vote"
    assert any("keyword" in vote["strategy"] for vote in result["strategy_votes"])
    assert result["evidence"], "expected evidence for the winning category"
