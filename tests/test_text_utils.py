from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook_tts.book_parser import parse_book
from audiobook_tts.generator import (
    _ensure_output_dir_available,
    _split_chapters_by_target_duration,
)
from audiobook_tts.settings import GenerationSettings
from audiobook_tts.text_utils import (
    decode_text_bytes,
    prepare_text_for_tts,
    split_text_for_tts,
)
from audiobook_tts.tts_engine import SileroTtsEngine


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

    def test_decode_text_bytes_supports_utf16_le_without_bom(self) -> None:
        text = "Глава 1\n\nПривет, мир!"

        self.assertEqual(decode_text_bytes(text.encode("utf-16-le")), text)

    def test_decode_text_bytes_keeps_cp1251_support(self) -> None:
        text = "Глава 1\n\nПривет, мир!"

        self.assertEqual(decode_text_bytes(text.encode("cp1251")), text)

    def test_prepare_text_for_tts_removes_emoji_and_private_control_chars(self) -> None:
        self.assertEqual(
            prepare_text_for_tts("Привет😀\ue000\x00мир\u200d!"),
            "Привет мир!",
        )

    def test_split_text_for_tts_drops_emoji_only_prefix_without_empty_chunks(self) -> None:
        chunks = split_text_for_tts("😀 ***\n\nТекст.", 400)

        self.assertEqual(chunks, ["Текст."])

    def test_existing_generated_mp3_files_block_output_directory_reuse(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "001.mp3").write_bytes(b"old")

            with self.assertRaisesRegex(ValueError, "already contains"):
                _ensure_output_dir_available(output_dir)

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


class SileroCacheRecoveryTests(unittest.TestCase):
    def test_corrupted_model_file_is_quarantined_and_downloaded_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            corrupt_path = Path(temp_dir) / "model.pt.corrupt"
            model_path.write_bytes(b"bad")
            engine = object.__new__(SileroTtsEngine)
            load_attempts = 0
            download_calls = 0

            def load_model(path: Path) -> object:
                nonlocal load_attempts
                load_attempts += 1
                if load_attempts == 1:
                    raise RuntimeError("bad package")
                return object()

            def download_model(torch: object, model_url: str, path: Path) -> None:
                nonlocal download_calls
                download_calls += 1
                path.write_bytes(b"good")

            engine._load_model = load_model
            engine._download_model = download_model

            model = engine._load_model_with_recovery(object(), "https://example/model.pt", model_path)

            self.assertIsNotNone(model)
            self.assertEqual(load_attempts, 2)
            self.assertEqual(download_calls, 1)
            self.assertEqual(model_path.read_bytes(), b"good")
            self.assertEqual(corrupt_path.read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
