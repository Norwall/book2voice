from __future__ import annotations

import re
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .audio import (
    audio_to_pcm16_bytes,
    concat_mp3_files,
    convert_wav_to_mp3,
    silence_pcm16_bytes,
)
from .book_parser import Chapter, parse_book
from .settings import GenerationSettings
from .text_utils import (
    default_title_from_path,
    estimate_tts_seconds,
    safe_name,
    split_text_for_tts,
    split_text_into_sentences,
)
from .tts_engine import SileroTtsEngine

ProgressCallback = Callable[["ProgressEvent"], None]
CancelCallback = Callable[[], bool]
WHITESPACE_PREVIEW_RE = re.compile(r"[ \t\r\n\f\v]+")
GENERATED_CHAPTER_FILE_RE = re.compile(r"^\d{3}\.mp3$")


class GenerationCancelled(RuntimeError):
    """Raised when audiobook generation is cancelled by the caller."""


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    chapter_index: int | None = None
    total_chapters: int | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    chapter_files: list[Path]
    merged_file: Path | None
    chapters_count: int


def generate_audiobook(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    merge: bool = False,
    settings: GenerationSettings | None = None,
    txt_chapter_chars: int = 12000,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> GenerationResult:
    settings = settings or GenerationSettings()
    settings.validate()
    input_path = input_path.resolve()
    parsed = parse_book(input_path, txt_chapter_chars=txt_chapter_chars)
    source_chapters = parsed.chapters
    chapters = _split_chapters_by_target_duration(source_chapters, settings)
    if len(chapters) > 999:
        raise ValueError("The book has more than 999 chapters; numeric filenames would overflow")

    target_dir = output_dir or make_default_output_dir(input_path, parsed.title)
    _ensure_output_dir_available(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    chapter_files: list[Path] = []
    merged_file = None
    try:
        _raise_if_cancelled(cancel_requested)
        _emit(
            progress,
            "parse",
            f"Prepared {len(chapters)} audio chapters from {len(source_chapters)} source chapters",
            total_chapters=len(chapters),
        )
        _raise_if_cancelled(cancel_requested)
        _emit(progress, "model", "Loading Silero model")
        engine = SileroTtsEngine(
            model_id=settings.model_id,
            torch_threads=settings.torch_threads,
        )
        _raise_if_cancelled(cancel_requested)

        with tempfile.TemporaryDirectory(prefix="_audiobook_", dir=str(target_dir)) as temp_dir:
            temp_path = Path(temp_dir)
            for index, chapter in enumerate(chapters, start=1):
                _raise_if_cancelled(cancel_requested)
                output_mp3 = target_dir / f"{index:03d}.mp3"
                output_wav = temp_path / f"{index:03d}.wav"
                _emit(
                    progress,
                    "chapter",
                    f"Generating {index:03d}: {chapter.title}",
                    chapter_index=index,
                    total_chapters=len(chapters),
                )
                _write_chapter_wav(
                    engine,
                    chapter,
                    output_wav,
                    settings,
                    cancel_requested=cancel_requested,
                )
                _raise_if_cancelled(cancel_requested)
                convert_wav_to_mp3(
                    output_wav,
                    output_mp3,
                    bitrate=settings.mp3_bitrate,
                    speed=settings.speech_speed,
                )
                chapter_files.append(output_mp3)
                _emit(
                    progress,
                    "chapter_done",
                    f"Saved {output_mp3.name}",
                    chapter_index=index,
                    total_chapters=len(chapters),
                    output_path=output_mp3,
                )

        if merge:
            merged_file = target_dir / "book.mp3"
            _raise_if_cancelled(cancel_requested)
            _emit(progress, "merge", "Merging chapters into one MP3")
            concat_mp3_files(chapter_files, merged_file)
            _emit(progress, "merge_done", "Saved book.mp3", output_path=merged_file)
    except GenerationCancelled:
        _emit(
            progress,
            "cancelled",
            "Audiobook generation cancelled",
            output_path=target_dir,
        )
        raise

    _emit(progress, "done", "Audiobook generation finished", output_path=target_dir)
    return GenerationResult(
        output_dir=target_dir,
        chapter_files=chapter_files,
        merged_file=merged_file,
        chapters_count=len(chapter_files),
    )


def _raise_if_cancelled(cancel_requested: CancelCallback | None) -> None:
    if cancel_requested and cancel_requested():
        raise GenerationCancelled("Audiobook generation was cancelled")


def make_default_output_dir(input_path: Path, book_title: str | None = None) -> Path:
    title = safe_name(book_title or default_title_from_path(input_path), default="book")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{title}_{stamp}"


def _ensure_output_dir_available(target_dir: Path) -> None:
    if target_dir.exists() and not target_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {target_dir}")
    if not target_dir.exists():
        return

    conflicts = sorted(
        path.name
        for path in target_dir.iterdir()
        if path.is_file()
        and (GENERATED_CHAPTER_FILE_RE.match(path.name) or path.name == "book.mp3")
    )
    if conflicts:
        preview = ", ".join(conflicts[:5])
        if len(conflicts) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Output directory already contains generated audiobook files "
            f"({preview}). Choose an empty folder or enable timestamped output."
        )


def _split_chapters_by_target_duration(
    source_chapters: list[Chapter],
    settings: GenerationSettings,
) -> list[Chapter]:
    target_seconds = settings.target_chapter_minutes * 60
    result: list[Chapter] = []
    current_sentences: list[str] = []
    current_text = ""
    current_title = ""

    def append_current() -> None:
        nonlocal current_sentences, current_text, current_title
        text = current_text.strip()
        if text:
            result.append(
                Chapter(
                    index=len(result) + 1,
                    title=current_title or f"Part {len(result) + 1}",
                    text=text,
                )
            )
        current_sentences = []
        current_text = ""
        current_title = ""

    for source_chapter in source_chapters:
        for sentence in split_text_into_sentences(source_chapter.text):
            candidate_text = f"{current_text} {sentence}".strip() if current_text else sentence
            candidate_seconds = estimate_tts_seconds(
                candidate_text,
                max_chunk_chars=settings.max_chunk_chars,
                pause_ms=settings.pause_ms,
                speech_speed=settings.speech_speed,
            )
            if current_sentences and candidate_seconds > target_seconds:
                append_current()
                candidate_text = sentence

            if not current_sentences:
                current_title = source_chapter.title
            current_sentences.append(sentence)
            current_text = candidate_text

    append_current()
    if not result:
        raise ValueError("No readable text was found")
    return result


def _write_chapter_wav(
    engine: SileroTtsEngine,
    chapter: Chapter,
    wav_path: Path,
    settings: GenerationSettings,
    *,
    cancel_requested: CancelCallback | None = None,
) -> None:
    chunks = split_text_for_tts(chapter.text, settings.max_chunk_chars)
    if not chunks:
        raise ValueError(f"Chapter {chapter.index} has no text after cleanup")

    silence = silence_pcm16_bytes(settings.sample_rate, settings.pause_ms)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(settings.sample_rate)
        for chunk_index, chunk in enumerate(chunks, start=1):
            _raise_if_cancelled(cancel_requested)
            try:
                audio = engine.synthesize(
                    chunk,
                    voice=settings.voice,
                    sample_rate=settings.sample_rate,
                )
            except Exception as exc:
                error = str(exc).strip() or type(exc).__name__
                preview = _visible_error_preview(chunk)
                raise RuntimeError(
                    f"TTS failed in chapter {chapter.index:03d} '{chapter.title}', "
                    f"chunk {chunk_index}/{len(chunks)}: {error}. "
                    f"Text: {preview!r}"
                ) from exc
            wav_file.writeframes(audio_to_pcm16_bytes(audio))
            if silence and chunk_index < len(chunks):
                wav_file.writeframes(silence)


def _visible_error_preview(text: str, max_chars: int = 160) -> str:
    preview = WHITESPACE_PREVIEW_RE.sub(" ", text).strip()[:max_chars]
    result: list[str] = []
    for char in preview:
        category = unicodedata.category(char)
        if (
            char == "\u00a0"
            or category.startswith(("C", "M"))
            or (char != " " and category.startswith("Z"))
        ):
            result.append(f"\\u{ord(char):04X}")
        else:
            result.append(char)
    return "".join(result)


def _emit(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    *,
    chapter_index: int | None = None,
    total_chapters: int | None = None,
    output_path: Path | None = None,
) -> None:
    if progress:
        progress(
            ProgressEvent(
                stage=stage,
                message=message,
                chapter_index=chapter_index,
                total_chapters=total_chapters,
                output_path=output_path,
            )
        )
