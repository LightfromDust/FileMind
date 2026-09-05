# FileMind 🧠

**AI 驱动的本地文件智能工具包。**扫描、搜索、问答、摘要、分类、整理——一切都在你自己的机器上运行。

FileMind 将你的本地文件构建为可检索的知识库：解析文档、文本分块、生成向量嵌入（FAISS + sentence-transformers），全部存入 SQLite。你可以获得语义搜索、RAG 风格的文件问答、AI 摘要、多策略分类，以及安全的批量重命名和归档功能。

---

## 快速开始

```bash
# 安装
pip install -e .

# 安装完整文件格式支持（PDF、DOCX）
pip install -e ".[parsers]"

# 扫描目录，建立索引
python -m filemind scan ~/Documents

# 语义搜索
python -m filemind search "GPU优化"

# 对文件内容提问
python -m filemind qa "发票的合计金额是多少？"

# 摘要单个文件
python -m filemind summary ~/Documents/paper_gpu.txt

# 分类文件
python -m filemind classify ~/Documents

# 规划批量重命名（默认仅预览，不实际执行）
python -m filemind rename-plan ~/Documents

# 规划按类别归档（默认仅预览，不实际执行）
python -m filemind archive-plan ~/Documents ~/Archive
```

---

## 功能

| 功能 | 说明 |
|------|------|
| 🔍 **智能扫描** | 递归目录扫描，并发处理文件。通过内容哈希跳过未变更的文件。 |
| 🔎 **语义搜索** | 混合搜索：FAISS 向量搜索 + 关键词回退。支持中文、领域同义词扩展、分类加权。 |
| 💬 **文件问答** | 基于 RAG 的文件问答——LLM 生成带引文标注 `[1]` `[2]` 的答案，外加抽取式回退。 |
| 📝 **智能摘要** | LLM 增强的摘要和关键词提取，带规则回退。 |
| 🏷️ **文件分类** | 4 层策略投票：扩展名规则 → 关键词匹配 → 嵌入向量相似度 → LLM 分类。 |
| ✏️ **批量重命名** | 标准化为 `日期_类别_主题.后缀` 格式。预览先行、冲突处理、支持回滚。 |
| 📦 **分类归档** | 按类别整理到分类目录。安全——不删除原文件，始终默认预览模式。 |

---

## 支持的文件类型

| 格式 | 解析器 | 扩展名 |
|------|--------|--------|
| 纯文本 | 内置 | `.txt` `.md` `.csv` `.json` `.yaml` `.yml` |
| 源代码 | 内置 | `.py` `.java` `.c` `.cpp` `.js` |
| PDF | PyMuPDF（可选） | `.pdf` |
| Word 文档 | python-docx（可选） | `.docx` |

---

## 架构

```
filemind/
├── scanner/        # 目录扫描与索引编排
├── parser/         # 多格式文件解析（文本、代码、PDF、DOCX）
├── indexer/        # 语义分块、嵌入向量化、FAISS 向量存储
├── services/       # 核心逻辑：搜索、问答、摘要、分类、重命名、归档
├── storage/        # SQLite 仓库（files、chunks、operation_logs）
├── safety/         # 路径守卫、冲突检测、回滚管理
├── tools/          # 公开函数包装（可从任何代码中调用）
├── config.py       # 配置（dataclass + 环境变量）
└── __main__.py     # JSON 输出 CLI
```

### 数据流

```
文件 → 解析器 → 文本 → 分块器 → 嵌入器 → FAISS 索引
                    ↘          ↙
                 SQLite（文件 + 块 + 操作日志）
                    ↓
           搜索 / 问答 / 摘要 / 分类
```

---

## 配置

所有设置都有合理的默认值，可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FILEMIND_DB_PATH` | `~/.filemind/filemind.db` | SQLite 数据库路径 |
| `FILEMIND_VECTOR_INDEX_PATH` | `~/.filemind/faiss.index` | FAISS 索引路径 |
| `FILEMIND_WORKSPACE_DIR` | `~/.filemind/workspace` | 工作目录 |
| `FILEMIND_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `FILEMIND_SEMANTIC_INDEX` | `1` | 设为 `0` 禁用语义索引 |

> 兼容性:旧前缀 `FILE_AGENT_*`(如 `FILE_AGENT_DB_PATH`)仍然有效;当两个前缀同时设置时,`FILEMIND_*` 优先。

---

## 接入其他 Agent

FileMind 是框架无关的——可以在任何 AI Agent 中使用。详见 [INTEGRATION.md](INTEGRATION.md)，提供了四种接入方式：

1. **直接函数调用**——最简单的 6 行代码即可使用
2. **LLM 增强**——对接你的 LLM 客户端，获得更智能的结果
3. **Agent 工具注册**——注册为 Agent 可调用的工具
4. **CLI 子进程**——通过 shell 调用，解析 JSON 输出

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
