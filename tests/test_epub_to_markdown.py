from __future__ import annotations

from pathlib import Path

from ebooklib import epub

from scripts.epub_to_markdown import AssetPolicy, convert_epub_to_markdown


def test_convert_epub_to_markdown_text_only_drops_nontext_assets(tmp_path: Path) -> None:
    epub_path = _build_sample_epub(tmp_path / "sample.epub")
    output_path = tmp_path / "sample.md"

    result = convert_epub_to_markdown(epub_path, output_path=output_path, asset_policy=AssetPolicy.TEXT_ONLY)

    content = output_path.read_text(encoding="utf-8")
    assert "# 测试 EPUB" in content
    assert "**正文**" in content
    assert "![插图]" not in content
    assert "<audio controls" not in content
    assert result.document_count == 1
    assert result.exported_asset_count == 0
    assert result.assets_dir is None


def test_convert_epub_to_markdown_keep_images_exports_image_assets(tmp_path: Path) -> None:
    epub_path = _build_sample_epub(tmp_path / "sample.epub")
    output_path = tmp_path / "sample.md"

    result = convert_epub_to_markdown(epub_path, output_path=output_path, asset_policy=AssetPolicy.KEEP_IMAGES)

    content = output_path.read_text(encoding="utf-8")
    assert "![插图](sample_assets/images/pic.png)" in content
    assert "<audio controls" not in content
    assert result.assets_dir is not None
    assert (result.assets_dir / "images" / "pic.png").exists()
    assert result.exported_asset_count == 1


def test_convert_epub_to_markdown_keep_all_exports_audio_assets(tmp_path: Path) -> None:
    epub_path = _build_sample_epub(tmp_path / "sample.epub")
    output_path = tmp_path / "sample.md"

    result = convert_epub_to_markdown(epub_path, output_path=output_path, asset_policy=AssetPolicy.KEEP_ALL)

    content = output_path.read_text(encoding="utf-8")
    assert "![插图](sample_assets/images/pic.png)" in content
    assert '<audio controls src="sample_assets/audio/theme.mp3"></audio>' in content
    assert result.assets_dir is not None
    assert (result.assets_dir / "audio" / "theme.mp3").exists()
    assert result.exported_asset_count == 2


def _build_sample_epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("sample-book")
    book.set_title("测试 EPUB")
    book.set_language("zh")
    book.add_author("测试作者")

    chapter = epub.EpubHtml(title="第一章", file_name="chapters/ch1.xhtml", lang="zh")
    chapter.content = """
    <html xmlns="http://www.w3.org/1999/xhtml">
      <head><title>第一章</title></head>
      <body>
        <h1>第一章 起风</h1>
        <p>这里是<strong>正文</strong>。</p>
        <img src="../images/pic.png" alt="插图" />
        <audio src="../audio/theme.mp3" controls="controls"></audio>
      </body>
    </html>
    """

    image_item = epub.EpubItem(
        uid="img-1",
        file_name="images/pic.png",
        media_type="image/png",
        content=_tiny_png_bytes(),
    )
    audio_item = epub.EpubItem(
        uid="audio-1",
        file_name="audio/theme.mp3",
        media_type="audio/mpeg",
        content=b"ID3fake-audio",
    )

    book.add_item(chapter)
    book.add_item(image_item)
    book.add_item(audio_item)
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    epub.write_epub(str(path), book)
    return path


def _tiny_png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0cIDATx\x9cc``\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
        b"\x18\xdd\x8d\xb1"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )