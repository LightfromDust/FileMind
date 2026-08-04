"""Code parser. V1 extracts source text only."""

from __future__ import annotations

from pathlib import Path

from filemind.parser.text_parser import TextParser


class CodeParser(TextParser):
    extensions = {".py", ".java", ".c", ".cpp", ".js"}

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.extensions
