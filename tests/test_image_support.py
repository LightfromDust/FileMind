"""Image storage and query support.

Images used to be dropped at the extension whitelist before ever reaching
the parser. With ImageParser registered and image extensions whitelisted,
a scanned directory now stores images, and PNG-embedded text (tEXt), EXIF
and OCR (when available) make image *content* searchable end-to-end:
scan -> chunk -> search -> Q&A -> classification.

All fixtures are crafted with the standard library (a valid PNG with tEXt
chunks and correct CRCs, a synthetic JPEG marker stream), so the suite has
zero optional dependencies.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from uuid import uuid4

from filemind.config import FileMindConfig
from filemind.indexer.embedder import Embedder
from filemind.parser import DEFAULT_PARSERS
from filemind.parser.image_parser import ImageParser
from filemind.scanner.file_scanner import FileScanner
from filemind.services.qa_service import QAService
from filemind.services.search_service import SearchService
from filemind.storage.file_repository import FileRepository

import asyncio


def _tmp() -> Path:
    path = Path("tests") / "_runtime" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _disable_embedding(monkeypatch) -> None:
    def fail_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(Embedder, "embed", fail_embed)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def make_png(width: int = 800, height: int = 600, texts: dict[str, str] | None = None) -> bytes:
    """A structurally valid PNG (signature, IHDR, tEXt chunks, IEND)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    for keyword, value in (texts or {}).items():
        payload += _png_chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + value.encode("utf-8"))
    return payload + _png_chunk(b"IEND", b"")


def make_jpeg(width: int = 1024, height: int = 768) -> bytes:
    """Synthetic JPEG marker stream: SOI + APP0 + SOF0 + EOI."""
    return (
        b"\xff\xd8"
        b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", height, width)
        + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x00"
        b"\xff\xd9"
    )


def test_image_parser_registered_and_matches_extensions():
    parser = next((p for p in DEFAULT_PARSERS if isinstance(p, ImageParser)), None)
    assert parser is not None, "ImageParser must be registered in DEFAULT_PARSERS"
    for name in ("a.jpg", "b.jpeg", "c.png", "d.bmp", "e.webp", "f.gif"):
        assert parser.can_parse(name), name
    assert not parser.can_parse("g.txt")


def test_png_parser_extracts_dimensions_and_embedded_text():
    png = make_png(
        width=800,
        height=600,
        texts={"Description": "增值税发票 金额 140,000 报销"},
    )
    path = Path(_tmp()) / "invoice_scan.png"
    path.write_bytes(png)

    text = ImageParser().parse(str(path))

    assert "宽800" in text and "高600" in text
    assert "增值税发票" in text and "140,000" in text and "报销" in text
    assert "invoice scan" in text, "filename tokens must be searchable"


def test_jpeg_parser_extracts_dimensions_from_sof_marker():
    path = Path(_tmp()) / "photo_HDR.jpg"
    path.write_bytes(make_jpeg(width=1024, height=768))

    text = ImageParser().parse(str(path))

    assert "宽1024" in text and "高768" in text


def test_gif_and_bmp_dimensions():
    tmp = _tmp()
    gif = tmp / "anim.gif"
    gif.write_bytes(b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00\x00")
    bmp = tmp / "icon.bmp"
    bmp.write_bytes(b"BM" + b"\x00" * 16 + struct.pack("<ii", 16, 16) + b"\x00" * 8)

    gif_text = ImageParser().parse(str(gif))
    bmp_text = ImageParser().parse(str(bmp))

    assert "宽320" in gif_text and "高240" in gif_text
    assert "宽16" in bmp_text and "高16" in bmp_text


def test_scan_indexes_images_and_categories_them(monkeypatch):
    _disable_embedding(monkeypatch)
    tmp = _tmp()
    config = FileMindConfig(
        db_path=tmp / "filemind.db",
        vector_index_path=tmp / "faiss.index",
        workspace_dir=tmp / "workspace",
    )
    workspace = tmp / "photos"
    workspace.mkdir()
    (workspace / "invoice_scan.png").write_bytes(
        make_png(texts={"Description": "增值税发票 金额 140,000 报销"})
    )
    (workspace / "photo_IMG_2031.png").write_bytes(make_png(width=4032, height=3024))
    (workspace / "meeting_whiteboard.jpg").write_bytes(make_jpeg(width=1920, height=1080))
    (workspace / "note.txt").write_text("普通文本", encoding="utf-8")

    stats = FileScanner(config, semantic_index=False).scan(workspace)

    assert stats["scanned"] == 4
    assert stats["indexed"] == 4, "images must no longer be skipped by the extension filter"
    assert stats["failed"] == 0

    repo = FileRepository(config.db_path)
    invoice = repo.get_file_by_path(workspace / "invoice_scan.png")
    photo = repo.get_file_by_path(workspace / "photo_IMG_2031.png")
    # Keyword evidence from the embedded text outweighs the .png extension vote.
    assert invoice["category"] == "发票/报销"
    # No content signal: the extension rule decides.
    assert photo["category"] == "图片资料"


def test_search_and_qa_over_image_content(monkeypatch):
    _disable_embedding(monkeypatch)
    tmp = _tmp()
    config = FileMindConfig(
        db_path=tmp / "filemind.db",
        vector_index_path=tmp / "faiss.index",
        workspace_dir=tmp / "workspace",
    )
    workspace = tmp / "photos"
    workspace.mkdir()
    (workspace / "invoice_scan.png").write_bytes(
        make_png(texts={"Description": "增值税发票 金额 140,000 报销"})
    )
    (workspace / "contract_scan.png").write_bytes(
        make_png(texts={"Description": "租赁合同 甲方 乙方 租期三年 违约责任"})
    )
    (workspace / "photo_IMG_2031.png").write_bytes(make_png(width=4032, height=3024))
    FileScanner(config, semantic_index=False).scan(workspace)

    # Search: embedded text is retrieved like any document chunk.
    results = SearchService(config).search_files("发票 金额 报销", top_k=3, semantic=True)
    assert results[0]["file_name"] == "invoice_scan.png"
    assert results[0]["search_mode"] == "keyword_fallback"

    hits = SearchService(config).search_chunks("租赁合同 违约", top_k=3, semantic=True)
    assert hits and hits[0]["file_name"] == "contract_scan.png"

    # Q&A: the answer cites the image as its source.
    qa = asyncio.run(QAService(config).answer_question("发票 金额", top_k=3))
    assert qa["source_files"], "image chunks must be reachable from QA"
    assert qa["source_files"][0]["file_name"] == "invoice_scan.png"
