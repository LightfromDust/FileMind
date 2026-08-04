"""Text chunking utilities with semantic-aware splitting."""

from __future__ import annotations

import re


class SemanticChunker:
    """Three-layer regex-based semantic chunking.

    Layer 1: Split by Markdown headers (^#{1,6}\\s+)
    Layer 2: Split by double-newline (paragraph boundaries)
    Layer 3: Split by single newline / sentence boundary

    Adjacent small segments are merged up to min_chunk_size.
    Segments exceeding chunk_size are split on sentence boundaries.
    """

    _MD_HEADER = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    _DOUBLE_NEWLINE = re.compile(r"\n\s*\n")
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;.])\s*")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, min_chunk_size: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[dict[str, object]]:
        if not text or not text.strip():
            return []

        segments = self._layered_split(text)
        segments = self._split_large(segments)
        segments = self._apply_overlap(segments)
        return [{"chunk_index": i, "content": seg} for i, seg in enumerate(segments) if seg.strip()]

    def _layered_split(self, text: str) -> list[str]:
        """Three-layer splitting: headers > paragraphs > sentences."""
        if self._MD_HEADER.search(text):
            return self._split_markdown(text)
        if self._DOUBLE_NEWLINE.search(text):
            return self._split_paragraphs(text)
        return self._split_sentences(text)

    def _split_markdown(self, text: str) -> list[str]:
        """Split by Markdown headers, keeping header with its body."""
        parts = self._MD_HEADER.split(text)
        result: list[str] = []
        if parts[0].strip():
            result.append(parts[0].strip())
        i = 1
        while i < len(parts):
            header = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            combined = f"{header}\n{body}".strip() if body else header
            result.append(combined)
            i += 2
        return [p for p in result if p]

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split by double-newline (paragraph boundaries)."""
        paragraphs = self._DOUBLE_NEWLINE.split(text.strip())
        result: list[str] = []
        for para in paragraphs:
            stripped = para.strip()
            if stripped:
                result.append(stripped)
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split by sentence boundaries."""
        sentences = self._SENTENCE_BOUNDARY.split(text.strip())
        result: list[str] = []
        for sent in sentences:
            stripped = sent.strip()
            if stripped:
                result.append(stripped)
        return result

    def _split_large(self, segments: list[str]) -> list[str]:
        """Split segments exceeding chunk_size on sentence boundaries."""
        result: list[str] = []
        for seg in segments:
            if len(seg) <= self.chunk_size:
                result.append(seg)
            else:
                result.extend(self._split_recursive(seg))
        return result

    def _split_recursive(self, text: str) -> list[str]:
        """Split text exceeding chunk_size by sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]
        sentence_parts = self._SENTENCE_BOUNDARY.split(text)
        chunks: list[str] = []
        buf = ""
        for part in sentence_parts:
            if len(buf) + len(part) + 1 <= self.chunk_size:
                buf = f"{buf}{part}".strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
        return chunks

    def _apply_overlap(self, segments: list[str]) -> list[str]:
        """Add overlap prefix from previous chunk."""
        if self.chunk_overlap <= 0 or len(segments) <= 1:
            return segments
        result: list[str] = []
        for i, seg in enumerate(segments):
            if i > 0:
                tail = segments[i - 1][-self.chunk_overlap:]
                seg = f"{tail}...{seg}"
            result.append(seg)
        return result


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._semantic = SemanticChunker(chunk_size, chunk_overlap)

    def chunk(self, text: str) -> list[dict[str, object]]:
        if not text or not text.strip():
            return []
        if SemanticChunker._MD_HEADER.search(text) or SemanticChunker._DOUBLE_NEWLINE.search(text):
            return self._semantic.chunk(text)
        chunks: list[dict[str, object]] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end]
            if content:
                chunks.append({"chunk_index": index, "content": content})
                index += 1
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return chunks
