from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from audiobook_tts.book_parser import Chapter, parse_book
from audiobook_tts.generator import _split_chapters_by_target_duration, _write_chapter_wav
from audiobook_tts.number_normalizer import normalize_numbers_for_tts
from audiobook_tts.preview import VOICE_PREVIEW_TEXT
from audiobook_tts.settings import GenerationSettings
from audiobook_tts.text_utils import prepare_text_for_tts, split_text_for_tts


class NumberNormalizerTests(unittest.TestCase):
    def test_common_numeric_formats_are_expanded(self) -> None:
        cases = {
            "Глава 1. Часть II.": "Глава первая. Часть вторая.",
            "В 2025 году.": "В две тысячи двадцать пятом году.",
            "Выпуск 2025 г.": "Выпуск две тысячи двадцать пятый год.",
            "В 2025 г. вышла книга.": "В две тысячи двадцать пятом году вышла книга.",
            "Дата 05.03.2025.": "Дата пятое марта две тысячи двадцать пятого года.",
            "Время 3:46 утра.": "Время три часа сорок шесть минут утра.",
            "Число 3,14.": "Число три целых четырнадцать сотых.",
            "Цена 25 ₽.": "Цена двадцать пять рублей.",
            "Цена $25.50.": "Цена двадцать пять долларов, пятьдесят центов.",
            "Готово 12%.": "Готово двенадцать процентов.",
            "Страницы 5–7.": "Страницы от пяти до семи.",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_numbers_for_tts(source), expected)

    def test_explicit_ordinals_use_suffix_gender_and_case(self) -> None:
        self.assertEqual(
            normalize_numbers_for_tts(
                "1-й, 2-я, 3-е, 4-го, 5-му, 6-м, 7-ю, 8-х, 9-ми"
            ),
            "первый, вторая, третье, четвёртого, пятому, шестом, "
            "седьмую, восьмых, девятыми",
        )
        self.assertEqual(
            normalize_numbers_for_tts("90-е годы и 90-х годов"),
            "девяностые годы и девяностых годов",
        )

    def test_phone_numbers_and_identifiers_are_read_digit_by_digit(self) -> None:
        self.assertEqual(
            normalize_numbers_for_tts("Позвонить 911."),
            "Позвонить девять один один.",
        )
        self.assertEqual(
            normalize_numbers_for_tts("Телефон +7 (912) 345-67-89."),
            "Телефон плюс семь девять один два три четыре пять шесть семь восемь девять.",
        )
        self.assertEqual(
            normalize_numbers_for_tts("Код 007."),
            "Код ноль ноль семь.",
        )

    def test_normalization_is_idempotent_and_leaves_no_digits(self) -> None:
        source = "Глава 21. 05.03.2025, 14:30, 25 ₽, код 007."
        normalized = normalize_numbers_for_tts(source)

        self.assertEqual(normalize_numbers_for_tts(normalized), normalized)
        self.assertIsNone(re.search(r"\d", normalized))

    def test_number_normalization_can_be_disabled(self) -> None:
        source = "Глава 1. В 2025 году."

        self.assertEqual(
            prepare_text_for_tts(source, normalize_numbers=False),
            source,
        )
        self.assertEqual(
            prepare_text_for_tts(source, normalize_numbers=True),
            "Глава первая. В две тысячи двадцать пятом году.",
        )

    def test_voice_preview_covers_supported_number_formats(self) -> None:
        normalized = " ".join(
            split_text_for_tts(
                VOICE_PREVIEW_TEXT,
                850,
                normalize_numbers=True,
            )
        )

        self.assertIsNone(re.search(r"\d", normalized))
        for expected in (
            "Глава первая",
            "Часть вторая",
            "пятое марта две тысячи двадцать пятого года",
            "три часа сорок шесть минут",
            "двадцать одна целая пять десятых",
            "двадцать пять рублей",
            "двенадцать процентов",
            "от пяти до семи",
            "ноль ноль семь",
            "девять один один",
            "обозначен как первый",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized)

        raw = " ".join(
            split_text_for_tts(
                VOICE_PREVIEW_TEXT,
                850,
                normalize_numbers=False,
            )
        )
        self.assertIn("Глава 1", raw)
        self.assertIn("Часть II", raw)
        self.assertIn("21,5", raw)
        self.assertIn("007", raw)


class NumberPipelineTests(unittest.TestCase):
    class _AudioEngine:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def synthesize(self, text: str, *, voice: str, sample_rate: int):
            self.calls.append(text)
            return np.zeros(100, dtype=np.float32)

    def test_chapter_splitting_and_tts_receive_the_same_normalized_text(self) -> None:
        settings = GenerationSettings(normalize_numbers=True)
        chapters = _split_chapters_by_target_duration(
            [Chapter(index=1, title="Test", text="Глава 1. Время 3:46 утра.")],
            settings,
        )
        self.assertIn("Глава первая", chapters[0].text)
        self.assertIn("три часа сорок шесть минут", chapters[0].text)

        engine = self._AudioEngine()
        with TemporaryDirectory() as temp_dir:
            _write_chapter_wav(
                engine,
                chapters[0],
                Path(temp_dir) / "chapter.wav",
                settings,
            )

        spoken = " ".join(engine.calls)
        self.assertIn("Глава первая", spoken)
        self.assertIn("три часа сорок шесть минут", spoken)
        self.assertIsNone(re.search(r"\d", spoken))

    def test_disabled_setting_preserves_digits_for_legacy_behavior(self) -> None:
        settings = GenerationSettings(normalize_numbers=False)
        chapters = _split_chapters_by_target_duration(
            [Chapter(index=1, title="Test", text="Глава 1. Текст.")],
            settings,
        )

        self.assertIn("Глава 1", chapters[0].text)


class FootnoteReferenceTests(unittest.TestCase):
    def test_fb2_semantic_note_reference_is_removed(self) -> None:
        content = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:xlink="http://www.w3.org/1999/xlink">
  <description><title-info><book-title>Тест</book-title></title-info></description>
  <body>
    <section>
      <title><p>Глава</p></title>
      <p>Фраза<a xlink:href="#n1" type="note">[1]</a> продолжается.</p>
      <p>Обычная запись [2] остаётся.</p>
    </section>
  </body>
</FictionBook>
"""
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "book.fb2"
            path.write_text(content, encoding="utf-8")
            parsed = parse_book(path)

        text = parsed.chapters[0].text
        self.assertIn("Фраза продолжается.", text)
        self.assertNotIn("[1]", text)
        self.assertIn("Обычная запись [2] остаётся.", text)

    def test_epub_semantic_note_reference_is_removed(self) -> None:
        from ebooklib import epub

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "book.epub"
            book = epub.EpubBook()
            book.set_identifier("test-book")
            book.set_title("Тест")
            book.set_language("ru")
            chapter = epub.EpubHtml(title="Глава", file_name="chapter.xhtml", lang="ru")
            chapter.content = (
                "<html xmlns:epub='http://www.idpf.org/2007/ops'><body>"
                "<h1>Глава</h1><p>Фраза<a epub:type='noteref' href='#n1'>[1]</a> "
                "продолжается после удалённой ссылки на примечание в тексте книги.</p>"
                "</body></html>"
            )
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.spine = [chapter]
            epub.write_epub(str(path), book)

            parsed = parse_book(path)

        text = parsed.chapters[0].text
        self.assertIn("Фраза", text)
        self.assertIn("продолжается после удалённой ссылки", text)
        self.assertNotIn("[1]", text)


if __name__ == "__main__":
    unittest.main()
