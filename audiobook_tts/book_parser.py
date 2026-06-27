from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .text_utils import clean_text, decode_text_bytes

HEADING_RE = re.compile(
    r"^\s*(глава|часть|книга|раздел)\b.{0,100}$|^\s*(пролог|эпилог)\b.{0,80}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    text: str


@dataclass(frozen=True)
class ParsedBook:
    title: str
    chapters: list[Chapter]


def parse_book(path: Path, txt_chapter_chars: int = 12000) -> ParsedBook:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path, txt_chapter_chars=txt_chapter_chars)
    if suffix == ".epub":
        return parse_epub(path)
    if suffix == ".fb2":
        return parse_fb2(path, txt_chapter_chars=txt_chapter_chars)
    raise ValueError(f"Unsupported file format: {suffix}")


def parse_txt(path: Path, txt_chapter_chars: int = 12000) -> ParsedBook:
    text = clean_text(decode_text_bytes(path.read_bytes()))
    if not text:
        raise ValueError("The TXT file does not contain readable text")

    lines = text.splitlines()
    heading_indexes = [
        index
        for index, line in enumerate(lines)
        if 1 <= len(line.strip()) <= 120 and HEADING_RE.match(line.strip())
    ]

    if len(heading_indexes) >= 2:
        chapters = _split_by_heading_lines(lines, heading_indexes)
    else:
        chapters = [Chapter(index=1, title=path.stem, text=text)]

    return ParsedBook(title=path.stem, chapters=_drop_empty_chapters(chapters))


def parse_epub(path: Path) -> ParsedBook:
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        from ebooklib import epub
    except ImportError as exc:
        raise RuntimeError(
            "EPUB support requires ebooklib and beautifulsoup4. "
            "Install dependencies from requirements.txt."
        ) from exc

    book = epub.read_epub(str(path))
    metadata_title = book.get_metadata("DC", "title")
    title = metadata_title[0][0] if metadata_title else path.stem
    chapters: list[Chapter] = []

    for spine_item in book.spine:
        if isinstance(spine_item, tuple):
            item_id, linear = spine_item[0], spine_item[1] if len(spine_item) > 1 else "yes"
        else:
            item_id, linear = spine_item, "yes"
        if linear == "no":
            continue
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()
        for link in soup.find_all("a"):
            epub_type = str(link.get("epub:type") or link.get("type") or "").lower()
            role = str(link.get("role") or "").lower()
            href = str(link.get("href") or "")
            label = link.get_text(" ", strip=True)
            if (
                "noteref" in epub_type.split()
                or role == "doc-noteref"
                or (href.startswith("#") and re.fullmatch(r"\[\d+\]", label))
            ):
                link.decompose()

        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(" ", strip=True) if heading else item.get_name()
        text = clean_text(soup.get_text("\n"))
        if len(text) >= 40:
            chapters.append(
                Chapter(index=len(chapters) + 1, title=chapter_title, text=text)
            )

    chapters = _drop_empty_chapters(chapters)
    if not chapters:
        raise ValueError("The EPUB file does not contain readable chapters")

    return ParsedBook(title=title or path.stem, chapters=chapters)


def parse_fb2(path: Path, txt_chapter_chars: int = 12000) -> ParsedBook:
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as exc:
        raise ValueError(f"The FB2 file is not valid XML: {exc}") from exc

    title = _fb2_book_title(root) or path.stem
    body = _fb2_main_body(root)
    if body is None:
        raise ValueError("The FB2 file does not contain a readable body")

    sections = [
        child
        for child in list(body)
        if _local_name(child.tag) == "section" and _fb2_element_text(child)
    ]
    chapters: list[Chapter] = []

    if sections:
        for section in sections:
            text = clean_text("\n\n".join(_fb2_text_blocks(section)))
            if text:
                chapters.append(
                    Chapter(
                        index=len(chapters) + 1,
                        title=_fb2_section_title(section) or f"Chapter {len(chapters) + 1}",
                        text=text,
                    )
                )
    else:
        text = clean_text("\n\n".join(_fb2_text_blocks(body)))
        chapters = [Chapter(index=1, title=title, text=text)]

    chapters = _drop_empty_chapters(chapters)
    if not chapters:
        raise ValueError("The FB2 file does not contain readable chapters")

    return ParsedBook(title=title, chapters=chapters)


def _split_by_heading_lines(lines: list[str], heading_indexes: list[int]) -> list[Chapter]:
    chapters: list[Chapter] = []
    first_heading = heading_indexes[0]
    if first_heading > 0:
        intro = clean_text("\n".join(lines[:first_heading]))
        if intro:
            chapters.append(Chapter(index=1, title="Intro", text=intro))

    boundaries = heading_indexes + [len(lines)]
    for offset, start in enumerate(heading_indexes):
        end = boundaries[offset + 1]
        title = lines[start].strip() or f"Chapter {len(chapters) + 1}"
        body = clean_text("\n".join(lines[start:end]))
        if body:
            chapters.append(Chapter(index=len(chapters) + 1, title=title, text=body))

    return chapters


def _drop_empty_chapters(chapters: list[Chapter]) -> list[Chapter]:
    result: list[Chapter] = []
    for chapter in chapters:
        text = clean_text(chapter.text)
        if text:
            result.append(Chapter(index=len(result) + 1, title=chapter.title, text=text))
    if not result:
        raise ValueError("No readable text was found")
    return result


def _fb2_book_title(root: ET.Element) -> str:
    description = _first_child(root, "description")
    title_info = _first_child(description, "title-info") if description is not None else None
    book_title = _first_child(title_info, "book-title") if title_info is not None else None
    return _fb2_element_text(book_title) if book_title is not None else ""


def _fb2_main_body(root: ET.Element) -> ET.Element | None:
    bodies = [child for child in list(root) if _local_name(child.tag) == "body"]
    for body in bodies:
        if body.attrib.get("name", "").lower() != "notes":
            return body
    return bodies[0] if bodies else None


def _fb2_section_title(section: ET.Element) -> str:
    title = _first_child(section, "title")
    return _fb2_element_text(title) if title is not None else ""


def _fb2_text_blocks(element: ET.Element) -> list[str]:
    ignored_tags = {"binary", "image"}
    block_tags = {"p", "v", "subtitle", "text-author"}
    tag = _local_name(element.tag)
    if tag in ignored_tags:
        return []
    if tag in block_tags:
        text = _fb2_element_text(element, skip_note_refs=True)
        return [text] if text else []

    blocks: list[str] = []
    if element.text and tag not in {"FictionBook", "description", "title-info", "document-info"}:
        text = clean_text(element.text)
        if text:
            blocks.append(text)
    for child in list(element):
        blocks.extend(_fb2_text_blocks(child))
        if child.tail:
            tail = clean_text(child.tail)
            if tail:
                blocks.append(tail)
    return blocks


def _fb2_element_text(
    element: ET.Element | None,
    *,
    skip_note_refs: bool = False,
) -> str:
    if element is None:
        return ""

    parts: list[str] = []

    def collect(current: ET.Element) -> None:
        if current.text and current.text.strip():
            parts.append(current.text.strip())
        for child in list(current):
            is_note_ref = (
                skip_note_refs
                and _local_name(child.tag) == "a"
                and child.attrib.get("type", "").lower() == "note"
            )
            if not is_note_ref:
                collect(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

    collect(element)
    return clean_text(" ".join(parts))


def _first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
