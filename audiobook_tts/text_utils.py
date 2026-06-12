from __future__ import annotations

import re
from pathlib import Path

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines).strip()


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "koi8-r"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def safe_name(value: str, default: str = "book") -> str:
    name = SAFE_FILENAME_RE.sub("_", value).strip(" ._")
    name = re.sub(r"\s+", "_", name)
    return name[:80] or default


def default_title_from_path(path: Path) -> str:
    return safe_name(path.stem, default="book")


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    def add_piece(piece: str) -> None:
        nonlocal current
        piece = piece.strip()
        if not piece:
            return
        if len(piece) > max_chars:
            flush_current()
            chunks.extend(hard_split_text(piece, max_chars))
            return
        if not current:
            current = piece
            return
        if len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            flush_current()
            current = piece

    for paragraph in text.split("\n\n"):
        for sentence in SENTENCE_BOUNDARY_RE.split(paragraph):
            add_piece(sentence)
        flush_current()

    flush_current()
    return chunks


def hard_split_text(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = WHITESPACE_RE.sub(" ", text).strip()
    while len(remaining) > max_chars:
        split_at = max(
            remaining.rfind(" ", 0, max_chars),
            remaining.rfind(",", 0, max_chars),
            remaining.rfind(";", 0, max_chars),
            remaining.rfind(":", 0, max_chars),
        )
        if split_at < max_chars // 2:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts
