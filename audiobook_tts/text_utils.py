from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .number_normalizer import normalize_numbers_for_tts

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
UNICODE_SPACE_RE = re.compile(
    r"[\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+"
)
TTS_DECORATION_RE = re.compile(r"[*\"'`]+")
TTS_CHARS_PER_SECOND = 14.0
SENTENCE_END_CHARS = ".!?…"
SENTENCE_TRAILING_CHARS = "\"'»”)]}"
TEXT_DECODINGS = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1251",
    "windows-1251",
    "koi8-r",
)
TTS_TRANSLATION = str.maketrans(
    {
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00ab": "",
        "\u00bb": "",
        "\u2018": "",
        "\u2019": "",
        "\u201a": "",
        "\u201b": "",
        "\u201c": "",
        "\u201d": "",
        "\u201e": "",
        "\u201f": "",
        "\u2039": "",
        "\u203a": "",
    }
)


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines).strip()


def prepare_text_for_tts(text: str, *, normalize_numbers: bool = True) -> str:
    text = UNICODE_SPACE_RE.sub(" ", text.translate(TTS_TRANSLATION))
    if normalize_numbers:
        text = normalize_numbers_for_tts(text)
    text = "".join(_tts_safe_char(char) for char in text)
    text = TTS_DECORATION_RE.sub(" ", text)
    return clean_text(text)


def decode_text_bytes(data: bytes) -> str:
    candidates: list[tuple[int, int, str]] = []
    for index, encoding in enumerate(TEXT_DECODINGS):
        try:
            text = data.decode(encoding)
        except UnicodeError:
            continue
        candidates.append((_decoded_text_score(text), -index, text))

    if candidates:
        return max(candidates)[2]
    return data.decode("utf-8", errors="replace")


def _tts_safe_char(char: str) -> str:
    category = unicodedata.category(char)
    if category.startswith("M"):
        return ""
    if _is_emoji_or_symbol_pictograph(char):
        return " "
    if category.startswith("C"):
        return char if char in "\n\r\t" else " "
    return char


def _is_emoji_or_symbol_pictograph(char: str) -> bool:
    code = ord(char)
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0xFE00 <= code <= 0xFE0F
    )


def _decoded_text_score(text: str) -> int:
    score = 0
    for char in text:
        category = unicodedata.category(char)
        if char == "\ufffd" or char == "\x00":
            score -= 50
        elif category.startswith("C") and char not in "\n\r\t":
            score -= 20
        elif char.isalpha():
            score += 6
            if "а" <= char.lower() <= "я" or char in "ёЁ":
                score += 4
        elif char.isdigit() or char.isspace() or category.startswith("P"):
            score += 2
        elif char.isprintable():
            score += 1
    return score


def safe_name(value: str, default: str = "book") -> str:
    name = SAFE_FILENAME_RE.sub("_", value).strip(" ._")
    name = re.sub(r"\s+", "_", name)
    return name[:80] or default


def default_title_from_path(path: Path) -> str:
    return safe_name(path.stem, default="book")


def split_text_for_tts(
    text: str,
    max_chars: int,
    *,
    normalize_numbers: bool = True,
) -> list[str]:
    text = prepare_text_for_tts(text, normalize_numbers=normalize_numbers)
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
    return _merge_nonverbal_tts_chunks(chunks, max_chars)


def split_text_into_sentences(text: str, *, normalize_numbers: bool = True) -> list[str]:
    if normalize_numbers:
        text = normalize_numbers_for_tts(text)
    text = clean_text(text)
    if not text:
        return []

    raw_sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        for sentence in SENTENCE_BOUNDARY_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                raw_sentences.append(sentence)
    return _merge_heading_fragments(raw_sentences)


def _merge_heading_fragments(sentences: list[str]) -> list[str]:
    result: list[str] = []
    pending_prefix: list[str] = []

    for sentence in sentences:
        if _ends_sentence(sentence):
            if pending_prefix:
                sentence = f"{' '.join(pending_prefix)} {sentence}".strip()
                pending_prefix = []
            result.append(sentence)
        else:
            pending_prefix.append(sentence)

    if pending_prefix:
        suffix = " ".join(pending_prefix).strip()
        if result:
            result[-1] = f"{result[-1]} {suffix}".strip()
        else:
            result.append(suffix)

    return result


def _ends_sentence(value: str) -> bool:
    stripped = value.strip().rstrip(SENTENCE_TRAILING_CHARS)
    return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS


def estimate_tts_seconds(
    text: str,
    *,
    max_chunk_chars: int,
    pause_ms: int,
    speech_speed: float,
    normalize_numbers: bool = True,
) -> float:
    chunks = split_text_for_tts(
        text,
        max_chunk_chars,
        normalize_numbers=normalize_numbers,
    )
    if not chunks:
        return 0.0

    speech_seconds = sum(len(chunk) for chunk in chunks) / TTS_CHARS_PER_SECOND
    pause_seconds = max(0, len(chunks) - 1) * pause_ms / 1000
    return (speech_seconds + pause_seconds) / speech_speed


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


def _merge_nonverbal_tts_chunks(chunks: list[str], max_chars: int) -> list[str]:
    merged: list[str] = []
    pending_prefix: list[str] = []

    for raw_chunk in chunks:
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if not _has_letters(chunk):
            pending_prefix.append(chunk)
            continue

        if pending_prefix:
            prefix = " ".join(pending_prefix).strip()
            candidate = f"{prefix} {chunk}".strip()
            if len(candidate) <= max_chars or len(prefix) <= 40 or not merged:
                chunk = candidate
            else:
                merged[-1] = f"{merged[-1]} {prefix}".strip()
            pending_prefix = []

        merged.append(chunk)

    if pending_prefix:
        suffix = " ".join(pending_prefix).strip()
        if merged:
            merged[-1] = f"{merged[-1]} {suffix}".strip()
        else:
            merged.append(suffix)

    return merged


def _has_letters(value: str) -> bool:
    return any(char.isalpha() for char in value)
