from __future__ import annotations

import argparse
import posixpath
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from markdownify import MarkdownConverter


class AssetPolicy(StrEnum):
    TEXT_ONLY = "text-only"
    KEEP_IMAGES = "keep-images"
    KEEP_ALL = "keep-all"


@dataclass(slots=True)
class ConversionResult:
    output_path: Path
    assets_dir: Path | None
    document_count: int
    exported_asset_count: int


class _AssetExporter:
    def __init__(
        self,
        *,
        book: epub.EpubBook,
        output_path: Path,
        assets_dir: Path | None,
        asset_policy: AssetPolicy,
    ) -> None:
        self.output_path = output_path
        self.assets_dir = assets_dir
        self.asset_policy = asset_policy
        self._items_by_path = {
            self._normalize_internal_path(item.file_name): item for item in book.get_items() if getattr(item, "file_name", None)
        }
        self._exported_paths: dict[str, str] = {}

    @property
    def exported_asset_count(self) -> int:
        return len(self._exported_paths)

    def allows_images(self) -> bool:
        return self.asset_policy in {AssetPolicy.KEEP_IMAGES, AssetPolicy.KEEP_ALL}

    def allows_nontext(self) -> bool:
        return self.asset_policy == AssetPolicy.KEEP_ALL

    def export(self, current_document: str, href: str | None) -> str | None:
        if not href or self.assets_dir is None:
            return None

        normalized_href = href.split("#", 1)[0].split("?", 1)[0].strip()
        if not normalized_href:
            return None

        internal_path = self._resolve_internal_path(current_document, normalized_href)
        if internal_path is None:
            return None

        if internal_path in self._exported_paths:
            return self._exported_paths[internal_path]

        item = self._items_by_path.get(internal_path)
        if item is None:
            return None

        target_path = self.assets_dir / Path(*internal_path.split("/"))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(item.get_content())

        relative_path = posixpath.relpath(target_path.as_posix(), start=self.output_path.parent.as_posix())
        self._exported_paths[internal_path] = relative_path
        return relative_path

    @staticmethod
    def _normalize_internal_path(path: str) -> str:
        normalized = posixpath.normpath(path.replace("\\", "/"))
        return normalized.lstrip("/")

    def _resolve_internal_path(self, current_document: str, href: str) -> str | None:
        base_dir = posixpath.dirname(self._normalize_internal_path(current_document))
        resolved = posixpath.normpath(posixpath.join(base_dir, href))
        resolved = resolved.lstrip("/")
        if resolved.startswith("../"):
            return None
        return resolved


class _EpubMarkdownConverter(MarkdownConverter):
    def __init__(self, *, asset_exporter: _AssetExporter, current_document: str) -> None:
        super().__init__(heading_style="ATX", bullets="-", strong_em_symbol="*")
        self._asset_exporter = asset_exporter
        self._current_document = current_document

    def convert_img(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (text, parent_tags)
        if not self._asset_exporter.allows_images():
            return ""

        src = el.attrs.get("src")
        asset_ref = self._asset_exporter.export(self._current_document, src)
        if asset_ref is None:
            return ""

        alt_text = (el.attrs.get("alt") or "image").strip()
        return f"![{alt_text}]({asset_ref})"

    def convert_audio(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (text, parent_tags)
        if not self._asset_exporter.allows_nontext():
            return ""

        asset_ref = self._asset_exporter.export(self._current_document, el.attrs.get("src"))
        if asset_ref is None:
            return ""

        return f'\n\n<audio controls src="{asset_ref}"></audio>\n\n'

    def convert_video(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (text, parent_tags)
        if not self._asset_exporter.allows_nontext():
            return ""

        asset_ref = self._asset_exporter.export(self._current_document, el.attrs.get("src"))
        if asset_ref is None:
            return ""

        return f'\n\n<video controls src="{asset_ref}"></video>\n\n'

    def convert_source(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (el, text, parent_tags)
        return ""

    def convert_svg(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (text, parent_tags)
        if not self._asset_exporter.allows_nontext():
            return ""
        return f"\n\n{str(el)}\n\n"

    def convert_object(self, el, text, parent_tags) -> str:  # noqa: ANN001, ANN201
        _ = (text, parent_tags)
        if not self._asset_exporter.allows_nontext():
            return ""

        asset_ref = self._asset_exporter.export(self._current_document, el.attrs.get("data"))
        if asset_ref is None:
            return ""

        mime_type = el.attrs.get("type") or "application/octet-stream"
        return f"\n\n[Embedded object: {mime_type}]({asset_ref})\n\n"


def convert_epub_to_markdown(
    input_path: Path,
    *,
    output_path: Path | None = None,
    asset_policy: AssetPolicy = AssetPolicy.TEXT_ONLY,
    assets_dir: Path | None = None,
    title: str | None = None,
) -> ConversionResult:
    input_path = input_path.expanduser().resolve()
    if output_path is None:
        output_path = input_path.with_suffix(".md")
    else:
        output_path = output_path.expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_assets_dir = _resolve_assets_dir(output_path=output_path, assets_dir=assets_dir, asset_policy=asset_policy)

    book = epub.read_epub(str(input_path))
    exporter = _AssetExporter(
        book=book,
        output_path=output_path,
        assets_dir=resolved_assets_dir,
        asset_policy=asset_policy,
    )

    fragments: list[str] = []
    top_level_title = title or _first_metadata(book, "DC", "title") or input_path.stem
    author = _first_metadata(book, "DC", "creator")
    fragments.append(f"# {top_level_title}")
    if author:
        fragments.append(f"> 作者：{author}")

    document_count = 0
    seen_ids: set[str] = set()
    for spine_entry in book.spine:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        if item_id == "nav":
            continue
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue

        properties = set(getattr(item, "properties", []) or [])
        if "nav" in properties:
            continue

        chapter_markdown = _convert_document_to_markdown(
            html=item.get_content().decode("utf-8", errors="replace"),
            current_document=item.file_name,
            exporter=exporter,
        )
        if not chapter_markdown:
            continue

        fragments.append(chapter_markdown)
        document_count += 1

    markdown_output = _normalize_markdown("\n\n".join(fragment for fragment in fragments if fragment.strip()))
    output_path.write_text(markdown_output + "\n", encoding="utf-8")

    return ConversionResult(
        output_path=output_path,
        assets_dir=resolved_assets_dir if exporter.exported_asset_count else None,
        document_count=document_count,
        exported_asset_count=exporter.exported_asset_count,
    )


def _convert_document_to_markdown(*, html: str, current_document: str, exporter: _AssetExporter) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for removable in soup.find_all(["script", "style", "meta", "link", "noscript"]):
        removable.decompose()

    root = soup.body or soup
    markdown = _EpubMarkdownConverter(asset_exporter=exporter, current_document=current_document).convert_soup(root)
    return _normalize_markdown(markdown)


def _first_metadata(book: epub.EpubBook, namespace: str, key: str) -> str | None:
    values = book.get_metadata(namespace, key)
    if not values:
        return None

    value = values[0][0]
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _normalize_markdown(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _resolve_assets_dir(*, output_path: Path, assets_dir: Path | None, asset_policy: AssetPolicy) -> Path | None:
    if asset_policy == AssetPolicy.TEXT_ONLY:
        return None

    if assets_dir is None:
        return output_path.parent / f"{output_path.stem}_assets"

    return assets_dir.expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an EPUB file to Markdown.")
    parser.add_argument("input", type=Path, help="Path to the source EPUB file.")
    parser.add_argument("-o", "--output", type=Path, help="Path to the output Markdown file.")
    parser.add_argument(
        "--asset-policy",
        choices=[policy.value for policy in AssetPolicy],
        default=AssetPolicy.TEXT_ONLY.value,
        help="How to handle images and other non-text assets.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Directory for extracted assets when the selected policy keeps them.",
    )
    parser.add_argument("--title", help="Override the top-level Markdown title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert_epub_to_markdown(
        args.input,
        output_path=args.output,
        asset_policy=AssetPolicy(args.asset_policy),
        assets_dir=args.assets_dir,
        title=args.title,
    )

    print(f"Converted {result.document_count} document(s) -> {result.output_path}")
    if result.assets_dir is not None:
        print(f"Exported {result.exported_asset_count} asset(s) -> {result.assets_dir}")
    else:
        print("No non-text assets were exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())