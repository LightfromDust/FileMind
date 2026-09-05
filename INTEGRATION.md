# Integrating FileMind with Other Agent Frameworks

FileMind is **framework-agnostic** — it has zero dependencies on any agent framework. It works as a standalone Python library, a CLI tool, or a plugin inside any AI agent.

---

## Dependency Summary

FileMind only needs:

| Dependency | Required? | Purpose |
|-----------|-----------|---------|
| `sentence-transformers` | ⚠️ Yes | Embedding model for semantic search |
| `faiss-cpu` | ⚠️ Yes | Vector index for similarity search |
| `numpy` | ⚠️ Yes | Numerical operations on embeddings |
| `PyMuPDF` | Optional | PDF parsing |
| `python-docx` | Optional | Word document parsing |

**If you skip semantic features**, you can strip down to: `faiss-cpu` + `numpy`. The keyword search, classification, rename, and archive features work without `sentence-transformers` (they fall back gracefully).

The LLM is **never required** — all LLM-dependent features have rule-based fallbacks.

---

## Option 1: Direct Function Calls (Simplest)

FileMind exposes 6 public functions in `filemind.tools`. Call them directly — no agent framework needed.

```python
from filemind.tools import (
    scan_directory_tool,
    search_files_tool,
    summarize_file_tool,
    file_qa_tool,
    rename_files_tool,
    archive_files_tool,
)

# 1. Scan a directory to build the index
result = scan_directory_tool("/home/user/Documents", recursive=True)
# {"ok": true, "result": {"scanned": 42, "indexed": 30, "updated": 5, ...}}

# 2. Semantic search
result = search_files_tool("GPU optimization论文", top_k=5)
# {"ok": true, "results": [{...file metadata + score...}]}

# 3. Summarize a single file
result = summarize_file_tool("/path/to/some_file.pdf")
# {"ok": true, "summary": "...", "keywords": "...", "category": "论文资料"}

# 4. RAG-style Q&A (no LLM needed — extractive fallback built in)
result = file_qa_tool("合同里的服务内容是什么？", top_k=5)
# {"ok": true, "answer": "...", "source_files": [...], ...}

# 5. Plan safe batch renames (dry_run=True by default)
result = rename_files_tool("/home/user/Downloads", dry_run=True)
# {"ok": true, "rename_plan": [{old_path, new_path, reason, ...}]}

# 6. Plan category-based archiving (dry_run=True by default)
result = archive_files_tool("/home/user/Downloads", "/home/user/Archive", dry_run=True)
# {"ok": true, "archive_plan": [{source_path, target_path, category, ...}]}
```

All functions return `{"ok": True, ...}` or `{"ok": False, "error": "..."}`.

---

## Option 2: With LLM Enhancement

Pass your agent's LLM client to get smarter summaries, classifications, and Q&A answers.

### What FileMind expects from an LLM client

```python
# Your LLM client needs ONE of these two async methods:
#   await client.chat_with_retry(messages=..., tools=..., model=..., max_tokens=..., temperature=..., tool_choice=...)
#   await client.chat(messages=..., tools=..., model=..., max_tokens=..., temperature=..., tool_choice=...)
# The response object needs a .content attribute (str).

# Minimal example — wrap any OpenAI-compatible client:
class LLMAdapter:
    def __init__(self, client, model="gpt-4o"):
        self.client = client
        self.model = model

    async def chat_with_retry(self, **kwargs):
        # FileMind always passes: messages, tools=None, model, max_tokens, temperature, tool_choice="none"
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=kwargs["messages"],
            max_tokens=kwargs.get("max_tokens", 600),
            temperature=kwargs.get("temperature", 0.2),
        )
        # Wrap response so .content works
        response.content = response.choices[0].message.content
        return response
```

### Using the LLM adapter

```python
from openai import AsyncOpenAI
from filemind.config import FileMindConfig
from filemind.services.qa_service import QAService
from filemind.services.summarize_service import SummarizeService
from filemind.services.classify_service import ClassifyService
from filemind.tools import summarize_file_tool

# Wrap your LLM
openai_client = AsyncOpenAI()
llm = LLMAdapter(openai_client, model="gpt-4o")

# Now FileMind uses LLM for better results, with automatic fallback
result = summarize_file_tool("/path/to/file.pdf", llm_client=llm, model="gpt-4o")

# Or use services directly for more control
qa = QAService(FileMindConfig.from_env(), llm_client=llm, model="gpt-4o")
answer = await qa.answer_question("这份合同的违约金是多少？", top_k=5)
```

---

## Option 3: As Agent Tools (Tool Registration Pattern)

If your agent framework has a tool registry (like nanobot, Claude Code, LangChain, etc.), wrap FileMind's functions as tools and register them.

### Generic tool registration pattern

```python
# Define tool schemas for your agent framework
TOOLS = [
    {
        "name": "filemind_scan",
        "description": "Scan a local directory into the FileMind index. Call this before searching or Q&A.",
        "parameters": {
            "directory": {"type": "str", "description": "Directory path to scan"},
            "recursive": {"type": "bool", "description": "Scan subdirectories recursively", "default": True},
        },
    },
    {
        "name": "filemind_search",
        "description": "Semantic search over indexed local files with keyword fallback.",
        "parameters": {
            "query": {"type": "str", "description": "Search query"},
            "top_k": {"type": "int", "description": "Max results", "default": 5},
        },
    },
    {
        "name": "filemind_qa",
        "description": "Answer questions about indexed files using RAG with LLM or extractive fallback.",
        "parameters": {
            "question": {"type": "str", "description": "The question to answer"},
            "top_k": {"type": "int", "description": "Max chunks to retrieve", "default": 5},
        },
    },
    {
        "name": "filemind_summary",
        "description": "Summarize one local file and return keywords plus category.",
        "parameters": {
            "file_path": {"type": "str", "description": "Absolute path of the file to summarize"},
        },
    },
    {
        "name": "filemind_rename",
        "description": "Plan safe batch renames. Never deletes files. Dry-run by default.",
        "parameters": {
            "directory": {"type": "str", "description": "Directory containing files to rename"},
            "dry_run": {"type": "bool", "description": "Preview only without executing", "default": True},
        },
    },
    {
        "name": "filemind_archive",
        "description": "Plan safe category-based archiving. Never deletes files. Dry-run by default.",
        "parameters": {
            "source_dir": {"type": "str", "description": "Source directory"},
            "target_root": {"type": "str", "description": "Root directory for categorized archives"},
            "dry_run": {"type": "bool", "description": "Preview only without executing", "default": True},
        },
    },
]

# Tool dispatch
def handle_filemind_tool(name: str, params: dict) -> dict:
    from filemind.tools import (
        scan_directory_tool,
        search_files_tool,
        summarize_file_tool,
        file_qa_tool,
        rename_files_tool,
        archive_files_tool,
    )

    dispatch = {
        "filemind_scan":    lambda: scan_directory_tool(params["directory"], params.get("recursive", True)),
        "filemind_search":  lambda: search_files_tool(params["query"], params.get("top_k", 5)),
        "filemind_qa":      lambda: file_qa_tool(params["question"], params.get("top_k", 5)),
        "filemind_summary": lambda: summarize_file_tool(params["file_path"]),
        "filemind_rename":  lambda: rename_files_tool(params["directory"], params.get("dry_run", True)),
        "filemind_archive": lambda: archive_files_tool(params["source_dir"], params["target_root"], params.get("dry_run", True)),
    }

    handler = dispatch.get(name)
    if handler:
        return handler()
    return {"ok": False, "error": f"Unknown tool: {name}"}
```

### nanobot-specific integration

FileMind already ships with a nanobot bridge at `nanobot/nanobot/agent/tools/file_agent.py`. It registers 6 tools into nanobot's `ToolRegistry`:

```python
from nanobot.agent.tools.file_agent import register_file_agent_tools

# In your AgentLoop setup:
register_file_agent_tools(self.tools, provider=self.provider, model=self.model)
```

---

## Option 4: CLI Subprocess

If your agent framework can call shell commands, invoke FileMind as a CLI subprocess. All commands output JSON.

```bash
# Scan
python -m filemind scan /path/to/dir

# Search
python -m filemind search "invoice amount"

# Q&A
python -m filemind qa "What services are in the contract?"

# Summarize
python -m filemind summary /path/to/file.pdf

# Classify
python -m filemind classify /path/to/dir

# Rename plan
python -m filemind rename-plan /path/to/dir

# Archive plan
python -m filemind archive-plan /path/to/source /path/to/target
```

Parse stdout as JSON to get structured results.

---

## What Needs No LLM

These features work with **zero LLM dependency**:

| Feature | LLM-free approach |
|---------|-------------------|
| `scan` | File parsing + metadata extraction + embedding — no LLM needed |
| `search` | Semantic (FAISS) + keyword SQLite — no LLM needed |
| `classify` | Extension rules + keywords + embedding prototype similarity — LLM is an optional 4th voter |
| `rename` | Rule-based naming pattern `date_category_topic.ext` — no LLM needed |
| `archive` | Category-based directory routing — no LLM needed |
| `summarize` | First 300 chars + keyword frequency — LLM is an optional enhancement |
| `qa` | Extractive sentence matching by keyword + sentence boundary — LLM is an optional enhancement |

---

## Quick Reference: All Public Functions

```python
# Tools (synchronous wrappers)
scan_directory_tool(directory: str, recursive: bool = True) -> dict
search_files_tool(query: str, top_k: int = 5) -> dict
summarize_file_tool(file_path: str, llm_client=None, model=None) -> dict
file_qa_tool(question: str, top_k: int = 5) -> dict
rename_files_tool(directory: str, dry_run: bool = True) -> dict
archive_files_tool(source_dir: str, target_root: str, dry_run: bool = True) -> dict

# Services (async, more control)
QAService.answer_question(question: str, top_k: int = 5) -> dict       # async
SummarizeService.smart_summarize(text: str, file_name: str) -> dict    # async, needs llm_client
ClassifyService.classify_multi(file_name: str, text: str) -> dict      # sync, llm optional
SearchService.search_files(query: str, top_k: int = 5) -> list[dict]   # sync
SearchService.search_chunks(query: str, top_k: int = 5) -> list[dict]  # sync
RenameService.suggest_name(metadata: dict) -> str                       # sync
ArchiveService.plan_target(source_path: str, target_root: str, category: str) -> Path  # sync

# Scanner
FileScanner(config).scan(directory: str, recursive: bool = True) -> dict  # sync

# Config
FileMindConfig.from_env() -> FileMindConfig   # alias: FileAgentConfig
```
