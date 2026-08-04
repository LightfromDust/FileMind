"""Plain text parser with tolerant encoding handling."""

from __future__ import annotations

from pathlib import Path

from filemind.parser.base_parser import BaseParser, ParserError


class TextParser(BaseParser):
    extensions = {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.extensions

    def parse(self, file_path: str) -> str:
        path = Path(file_path)
        if path.stat().st_size == 0:
            return ""
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ParserError(f"Unable to decode text file: {file_path}")
