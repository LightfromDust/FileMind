"""DOCX parser backed by python-docx."""

from __future__ import annotations

from pathlib import Path

from filemind.parser.base_parser import BaseParser, ParserError


class DOCXParser(BaseParser):
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".docx"

    def parse(self, file_path: str) -> str:
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:
            raise ParserError("DOCX parsing requires python-docx. Install with: pip install python-docx") from exc

        try:
            document = Document(file_path)
            parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            return "\n".join(parts)
        except Exception as exc:
            raise ParserError(f"Failed to parse DOCX {file_path}: {exc}") from exc
