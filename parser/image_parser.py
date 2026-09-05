"""Image parser: stdlib-first metadata extraction with optional EXIF/OCR.

Unlike the PDF/DOCX parsers, missing optional dependencies never fail the
scan — images always yield at least filename tokens, format and dimensions
(read with the standard library), so every indexed image is searchable.

- Dimensions: PNG IHDR, JPEG SOF markers, GIF logical screen, BMP header.
- Embedded text: PNG tEXt/iTXt/zTXt chunks (some tools store descriptions
  and even OCR results there).
- EXIF (requires Pillow, optional): camera model, timestamp, description.
- OCR (requires pytesseract + tesseract, optional): full text recognition.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from filemind.parser.base_parser import BaseParser, ParserError

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# EXIF tags worth surfacing as searchable text (Pillow tag ids).
_EXIF_TAGS = {270: "description", 271: "make", 272: "model", 306: "datetime"}


class ImageParser(BaseParser):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.extensions

    def parse(self, file_path: str) -> str:
        path = Path(file_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ParserError(f"Unable to read image {file_path}: {exc}") from exc

        parts: list[str] = [path.stem.replace("-", " ").replace("_", " ")]
        dimensions = self._dimensions(path.suffix.lower(), raw)
        if dimensions:
            width, height = dimensions
            parts.append(f"图片 {path.suffix.lstrip('.').lower()} 宽{width} 高{height}")
        parts.extend(self._png_text_chunks(raw))
        parts.extend(self._exif_text(path))
        parts.extend(self._ocr_text(path))
        return "\n".join(part for part in parts if part.strip())

    # ------------------------------------------------------------------
    # Dimensions (pure stdlib)
    # ------------------------------------------------------------------
    @staticmethod
    def _dimensions(suffix: str, raw: bytes) -> tuple[int, int] | None:
        try:
            if suffix == ".png" and raw.startswith(_PNG_SIGNATURE):
                return struct.unpack(">II", raw[16:24])
            if suffix in {".jpg", ".jpeg"} and raw[:2] == b"\xff\xd8":
                return ImageParser._jpeg_dimensions(raw)
            if suffix == ".gif" and raw[:4] in (b"GIF8",):
                width, height = struct.unpack("<HH", raw[6:10])
                return width, height
            if suffix == ".bmp" and raw[:2] == b"BM":
                width, height = struct.unpack("<ii", raw[18:26])
                return abs(width), abs(height)
            if suffix == ".webp" and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
                if raw[12:16] == b"VP8X":
                    width = int.from_bytes(raw[24:27], "little") + 1
                    height = int.from_bytes(raw[27:30], "little") + 1
                    return width, height
        except (struct.error, IndexError):
            return None
        return None

    @staticmethod
    def _jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
        """Walk JPEG markers to the first SOF frame header."""
        offset = 2
        while offset + 9 < len(raw):
            if raw[offset] != 0xFF:
                offset += 1
                continue
            marker = raw[offset + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", raw[offset + 5 : offset + 9])
                return width, height
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                offset += 2
                continue
            length = struct.unpack(">H", raw[offset + 2 : offset + 4])[0]
            offset += 2 + length
        return None

    # ------------------------------------------------------------------
    # Embedded PNG text chunks (pure stdlib)
    # ------------------------------------------------------------------
    @staticmethod
    def _png_text_chunks(raw: bytes) -> list[str]:
        if not raw.startswith(_PNG_SIGNATURE):
            return []
        texts: list[str] = []
        offset = len(_PNG_SIGNATURE)
        while offset + 8 <= len(raw):
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            chunk_type = raw[offset + 4 : offset + 8]
            data = raw[offset + 8 : offset + 8 + length]
            if chunk_type == b"IEND":
                break
            if chunk_type == b"tEXt" and b"\x00" in data:
                _keyword, _, value = data.partition(b"\x00")
                texts.append(ImageParser._decode_text(value))
            elif chunk_type == b"iTXt":
                texts.append(ImageParser._decode_itxt(data))
            elif chunk_type == b"zTXt" and b"\x00" in data:
                _keyword, _, rest = data.partition(b"\x00")
                try:
                    texts.append(ImageParser._decode_text(zlib.decompress(rest[1:])))
                except zlib.error:
                    pass
            offset += 12 + length  # length + type + data + crc
        return [text for text in texts if text]

    @staticmethod
    def _decode_text(value: bytes) -> str:
        for encoding in ("utf-8", "gb18030", "latin-1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return ""

    @staticmethod
    def _decode_itxt(data: bytes) -> str:
        # keyword\0 compression_flag compression_method language\0 translated\0 text
        parts = data.split(b"\x00", 1)
        if len(parts) != 2:
            return ""
        rest = parts[1]
        compressed = rest[:1] == b"\x01"
        remainder = rest[1:]
        # Skip compression method, language tag and translated keyword.
        for _ in range(3):
            _head, _sep, remainder = remainder.partition(b"\x00")
            if not _sep:
                return ""
        try:
            payload = zlib.decompress(remainder) if compressed else remainder
        except zlib.error:
            return ""
        return ImageParser._decode_text(payload)

    # ------------------------------------------------------------------
    # Optional EXIF via Pillow
    # ------------------------------------------------------------------
    @staticmethod
    def _exif_text(path: Path) -> list[str]:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return []
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                fields = [
                    f"EXIF {_EXIF_TAGS[tag]} {value}"
                    for tag, value in exif.items()
                    if tag in _EXIF_TAGS and str(value).strip()
                ]
                return fields
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Optional OCR via pytesseract
    # ------------------------------------------------------------------
    @staticmethod
    def _ocr_text(path: Path) -> list[str]:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return []
        try:
            with Image.open(path) as image:
                for language in ("chi_sim+eng", "eng"):
                    try:
                        text = pytesseract.image_to_string(image, lang=language)
                    except pytesseract.TesseractNotFoundError:
                        return []
                    if text and text.strip():
                        return [f"OCR {text.strip()}"]
                return []
        except Exception:
            return []
