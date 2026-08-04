"""Parser interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ParserError(RuntimeError):
    """Raised when a parser cannot extract text from a supported file."""


class BaseParser(ABC):
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        ...

    @abstractmethod
    def parse(self, file_path: str) -> str:
        ...
