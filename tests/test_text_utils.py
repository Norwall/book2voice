from __future__ import annotations

import unittest
from pathlib import Path

from audiobook_tts.book_parser import parse_book
from audiobook_tts.generator import _split_chapters_by_target_duration
from audiobook_tts.settings import GenerationSettings
from audiobook_tts.text_utils import prepare_text_for_tts, split_text_for_tts


class TextUtilsTests(unittest.TestCase):
    def test_prepare_text_for_tts_removes_combining_stress_mark(self) -> None:
        self.assertEqual(
            prepare_text_for_tts("я признаю\u0301 свою роль"),
            "я признаю свою роль",
        )

    def test_prepare_text_for_tts_normalizes_non_breaking_spaces(self) -> None:
        self.assertEqual(
            prepare_text_for_tts("Сказал:\u00a0-\u00a0Привет"),
            "Сказал: - Привет",
        )

    def test_split_text_for_tts_drops_decorative_marks_without_empty_chunks(self) -> None:
        chunks = split_text_for_tts("***\n\nя признаю\u0301 свою роль", 400)

        self.assertEqual(chunks, ["я признаю свою роль"])

    def test_domrabotnica_fb2_chunks_do_not_keep_known_silero_breakers(self) -> None:
        book_path = (
            Path(__file__).resolve().parents[1]
            / "MakFadden_Domrabotnica_1_Domrabotnica_RuLit_Me.fb2"
        )
        if not book_path.exists():
            self.skipTest("fixture FB2 book is not available")

        settings = GenerationSettings()
        chapters = _split_chapters_by_target_duration(parse_book(book_path).chapters, settings)
        bad_chunks = [
            (chapter.index, chunk_index)
            for chapter in chapters
            for chunk_index, chunk in enumerate(
                split_text_for_tts(chapter.text, settings.max_chunk_chars),
                start=1,
            )
            if "\u0301" in chunk or "\u00a0" in chunk
        ]

        self.assertEqual(bad_chunks, [])


if __name__ == "__main__":
    unittest.main()
