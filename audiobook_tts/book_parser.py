from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .text_utils import clean_text, decode_text_bytes, hard_split_text

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
        parts = hard_split_text(text, txt_chapter_chars)
        chapters = [
            Chapter(index=index + 1, title=f"Part {index + 1}", text=part)
            for index, part in enumerate(parts)
        ]

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
