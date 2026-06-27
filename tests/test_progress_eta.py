from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from audiobook_tts.generator import ProgressEvent, generate_audiobook
from audiobook_tts.progress import GenerationTiming, format_eta_ru
from audiobook_tts.settings import GenerationSettings


class GenerationTimingTests(unittest.TestCase):
    def test_first_fragment_creates_initial_estimate(self) -> None:
        timing = GenerationTiming(total_chars=1000)

        self.assertIsNone(timing.estimated_remaining_seconds)
        timing.record_synthesis(100, 10.0)

        self.assertAlmostEqual(timing.progress_fraction, 0.1)
        self.assertAlmostEqual(timing.estimated_remaining_seconds or 0.0, 90.0)

    def test_encoding_measurement_refines_remaining_time(self) -> None:
        timing = GenerationTiming(total_chars=1000)
        timing.record_synthesis(100, 10.0)

        timing.record_encoding(100, 2.0)

        self.assertAlmostEqual(timing.estimated_remaining_seconds or 0.0, 108.0)

    def test_current_run_average_reacts_to_slower_fragments(self) -> None:
        timing = GenerationTiming(total_chars=1000)
        timing.record_synthesis(100, 10.0)
        first_estimate = timing.estimated_remaining_seconds

        timing.record_synthesis(100, 30.0)

        self.assertIsNotNone(first_estimate)
        self.assertGreater(timing.estimated_remaining_seconds or 0.0, first_estimate or 0.0)

    def test_progress_stays_below_complete_until_job_finishes(self) -> None:
        timing = GenerationTiming(total_chars=100)

        timing.record_synthesis(100, 1.0)

        self.assertEqual(timing.progress_fraction, 0.99)


class EtaFormattingTests(unittest.TestCase):
    def test_formats_duration_and_same_day_completion(self) -> None:
        now = datetime(2026, 6, 27, 17, 15)

        result = format_eta_ru(85 * 60, now=now)

        self.assertEqual(
            result,
            "Осталось примерно 1 ч 25 мин · Окончание около 18:40",
        )

    def test_includes_date_when_completion_crosses_midnight(self) -> None:
        now = datetime(2026, 6, 27, 23, 50)

        result = format_eta_ru(30 * 60, now=now)

        self.assertEqual(
            result,
            "Осталось примерно 30 мин · Окончание около 28.06 в 00:20",
        )


class ProgressEventIntegrationTests(unittest.TestCase):
    def test_generation_reports_fragment_progress_and_eta(self) -> None:
        class FakeEngine:
            def __init__(self, **_: object) -> None:
                pass

            def synthesize(self, text: str, *, voice: str, sample_rate: int) -> np.ndarray:
                return np.zeros(max(len(text), 1), dtype=np.float32)

        def fake_convert(wav_path: Path, mp3_path: Path, **_: object) -> None:
            self.assertTrue(wav_path.exists())
            mp3_path.write_bytes(b"mp3")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "book.txt"
            input_path.write_text(
                "Глава 1\n\nЭто короткий текст для проверки прогресса озвучки.",
                encoding="utf-8",
            )
            events: list[ProgressEvent] = []

            with (
                patch("audiobook_tts.generator.SileroTtsEngine", FakeEngine),
                patch("audiobook_tts.generator.convert_wav_to_mp3", fake_convert),
            ):
                generate_audiobook(
                    input_path,
                    output_dir=root / "output",
                    settings=GenerationSettings(max_chunk_chars=400),
                    progress=events.append,
                )

        fragment_events = [event for event in events if event.stage == "chunk_done"]
        self.assertGreaterEqual(len(fragment_events), 1)
        self.assertTrue(
            all(event.progress_fraction is not None for event in fragment_events)
        )
        self.assertTrue(
            all(event.estimated_remaining_seconds is not None for event in fragment_events)
        )
        self.assertEqual(fragment_events[-1].progress_fraction, 0.99)


if __name__ == "__main__":
    unittest.main()
