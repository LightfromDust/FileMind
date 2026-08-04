"""Configuration defaults for the Personal File Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


DEFAULT_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".xlsx",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".js",
    ".json",
    ".yaml",
    ".yml",
}


@dataclass(slots=True)
class FileAgentConfig:
    db_path: Path = Path("~/.nanobot/filemind/filemind.db").expanduser()
    vector_index_path: Path = Path("~/.nanobot/filemind/faiss.index").expanduser()
    workspace_dir: Path = Path("~/.nanobot/workspace").expanduser()
    max_file_size: int = 50 * 1024 * 1024
    recursive_scan: bool = True
    supported_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_SUPPORTED_EXTENSIONS)
    )
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    @classmethod
    def from_env(cls) -> "FileAgentConfig":
        config = cls()
        if value := os.environ.get("FILE_AGENT_DB_PATH"):
            config.db_path = Path(value).expanduser()
        if value := os.environ.get("FILE_AGENT_VECTOR_INDEX_PATH"):
            config.vector_index_path = Path(value).expanduser()
        if value := os.environ.get("FILE_AGENT_WORKSPACE_DIR"):
            config.workspace_dir = Path(value).expanduser()
        if value := os.environ.get("FILE_AGENT_EMBEDDING_MODEL"):
            config.embedding_model = value
        return config

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
