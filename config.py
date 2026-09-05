"""Configuration defaults for FileMind."""

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


def _env(*names: str) -> str | None:
    """Return the first environment variable that is set among ``names``."""
    for name in names:
        if value := os.environ.get(name):
            return value
    return None


@dataclass(slots=True)
class FileMindConfig:
    db_path: Path = Path("~/.filemind/filemind.db").expanduser()
    vector_index_path: Path = Path("~/.filemind/faiss.index").expanduser()
    workspace_dir: Path = Path("~/.filemind/workspace").expanduser()
    max_file_size: int = 50 * 1024 * 1024
    recursive_scan: bool = True
    supported_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_SUPPORTED_EXTENSIONS)
    )
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    @classmethod
    def from_env(cls) -> "FileMindConfig":
        config = cls()
        # FILEMIND_* is the canonical prefix; FILE_AGENT_* stays supported
        # for backward compatibility with earlier releases.
        if value := _env("FILEMIND_DB_PATH", "FILE_AGENT_DB_PATH"):
            config.db_path = Path(value).expanduser()
        if value := _env("FILEMIND_VECTOR_INDEX_PATH", "FILE_AGENT_VECTOR_INDEX_PATH"):
            config.vector_index_path = Path(value).expanduser()
        if value := _env("FILEMIND_WORKSPACE_DIR", "FILE_AGENT_WORKSPACE_DIR"):
            config.workspace_dir = Path(value).expanduser()
        if value := _env("FILEMIND_EMBEDDING_MODEL", "FILE_AGENT_EMBEDDING_MODEL"):
            config.embedding_model = value
        return config

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)


# Backward-compatible alias: earlier releases (and the nanobot bridge) know
# this config class under its original name.
FileAgentConfig = FileMindConfig
