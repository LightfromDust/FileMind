"""Document parsers."""

from filemind.parser.base_parser import BaseParser, ParserError
from filemind.parser.code_parser import CodeParser
from filemind.parser.docx_parser import DOCXParser
from filemind.parser.pdf_parser import PDFParser
from filemind.parser.text_parser import TextParser

DEFAULT_PARSERS = [PDFParser(), DOCXParser(), CodeParser(), TextParser()]

__all__ = [
    "BaseParser",
    "ParserError",
    "PDFParser",
    "DOCXParser",
    "TextParser",
    "CodeParser",
    "DEFAULT_PARSERS",
]
