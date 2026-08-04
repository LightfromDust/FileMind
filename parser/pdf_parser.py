"""PDF parser backed by PyMuPDF."""

from __future__ import annotations

from pathlib import Path

from filemind.parser.base_parser import BaseParser, ParserError


class PDFParser(BaseParser):
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> str:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ParserError("PDF parsing requires PyMuPDF. Install with: pip install PyMuPDF") from exc

        try:
            doc = fitz.open(file_path)
            try:
                return "\n\n".join(page.get_text().strip() for page in doc if page.get_text().strip())
            finally:
                doc.close()
        except Exception as exc:
            raise ParserError(f"Failed to parse PDF {file_path}: {exc}") from exc
