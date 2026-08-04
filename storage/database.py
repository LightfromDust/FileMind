"""SQLite database initialization for File Agent with thread-safe write lock."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from filemind.config import FileAgentConfig

_db_write_lock = threading.Lock()


def get_write_lock() -> threading.Lock:
    return _db_write_lock


def get_connection(db_path: str | Path | None = None, check_same_thread: bool = False) -> sqlite3.Connection:
    path = Path(db_path).expanduser() if db_path else FileAgentConfig().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    return conn


def init_database(db_path: str | Path | None = None) -> None:
    with _db_write_lock:
        with get_connection(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_ext TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    created_time TEXT,
                    modified_time TEXT,
                    content_hash TEXT,
                    summary TEXT,
                    keywords TEXT,
                    category TEXT,
                    indexed_time TEXT,
                    status TEXT DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_id INTEGER,
                    created_time TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id)
                );

                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    source_path TEXT,
                    target_path TEXT,
                    status TEXT,
                    rollback_info TEXT,
                    created_time TEXT
                );
                """
            )
