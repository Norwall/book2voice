from __future__ import annotations

import argparse
from pathlib import Path

from audiobook_tts.generator import ProgressEvent, generate_audiobook
from audiobook_tts.settings import SAMPLE_RATES, SILERO_VOICES, GenerationSettings


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Russian audiobook chapters with Silero TTS.")
    parser.add_argument("input", type=Path, help="Path to .epub or .txt book")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Also create book.mp3")
    parser.add_argument("--voice", choices=SILERO_VOICES, default="baya")
    parser.add_argument("--sample-rate", type=int, choices=SAMPLE_RATES, default=24000)
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed, 0.5..2.0")
    parser.add_argument("--pause-ms", type=int, default=350, help="Pause between TTS chunks")
    parser.add_argument("--chunk-chars", type=int, default=850, help="Max chars per TTS chunk")
    parser.add_argument(
        "--txt-chapter-chars",
        type=int,
        default=12000,
        help="Fallback chapter size for TXT files without chapter headings",
    )
    parser.add_argument("--threads", type=int, default=4, help="Torch CPU threads")
    args = parser.parse_args()

    settings = GenerationSettings(
        voice=args.voice,
        sample_rate=args.sample_rate,
        max_chunk_chars=args.chunk_chars,
        pause_ms=args.pause_ms,
        speech_speed=args.speed,
        torch_threads=args.threads,
    )

    def progress(event: ProgressEvent) -> None:
        prefix = ""
        if event.chapter_index and event.total_chapters:
            prefix = f"[{event.chapter_index:03d}/{event.total_chapters:03d}] "
        print(f"{prefix}{event.message}")

    result = generate_audiobook(
        args.input,
        output_dir=args.out,
        merge=args.merge,
        settings=settings,
        txt_chapter_chars=args.txt_chapter_chars,
        progress=progress,
    )
    print(f"Done: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
