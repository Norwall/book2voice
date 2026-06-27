from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class GenerationTiming:
    total_chars: int
    synthesized_chars: int = 0
    encoded_chars: int = 0
    synthesis_seconds: float = 0.0
    encoding_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.total_chars <= 0:
            raise ValueError("total_chars must be positive")

    def record_synthesis(self, char_count: int, elapsed_seconds: float) -> None:
        self._validate_measurement(char_count, elapsed_seconds)
        self.synthesized_chars = min(
            self.total_chars,
            self.synthesized_chars + char_count,
        )
        self.synthesis_seconds += elapsed_seconds

    def record_encoding(self, char_count: int, elapsed_seconds: float) -> None:
        self._validate_measurement(char_count, elapsed_seconds)
        self.encoded_chars = min(self.total_chars, self.encoded_chars + char_count)
        self.encoding_seconds += elapsed_seconds

    @property
    def progress_fraction(self) -> float:
        return min(self.synthesized_chars / self.total_chars, 0.99)

    @property
    def estimated_remaining_seconds(self) -> float | None:
        if self.synthesized_chars <= 0:
            return None

        synthesis_rate = self.synthesis_seconds / self.synthesized_chars
        remaining = synthesis_rate * (self.total_chars - self.synthesized_chars)

        if self.encoded_chars > 0:
            encoding_rate = self.encoding_seconds / self.encoded_chars
            remaining += encoding_rate * (self.total_chars - self.encoded_chars)

        return max(0.0, remaining)

    @staticmethod
    def _validate_measurement(char_count: int, elapsed_seconds: float) -> None:
        if char_count <= 0:
            raise ValueError("char_count must be positive")
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            raise ValueError("elapsed_seconds must be a finite non-negative number")


def format_eta_ru(remaining_seconds: float, *, now: datetime | None = None) -> str:
    if remaining_seconds < 0 or not math.isfinite(remaining_seconds):
        raise ValueError("remaining_seconds must be a finite non-negative number")

    current_time = now or datetime.now().astimezone()
    completion_time = current_time + timedelta(seconds=remaining_seconds)
    rounded_minutes = max(1, int(remaining_seconds / 60 + 0.5))
    hours, minutes = divmod(rounded_minutes, 60)

    if hours and minutes:
        duration = f"{hours} ч {minutes} мин"
    elif hours:
        duration = f"{hours} ч"
    else:
        duration = f"{minutes} мин"

    if completion_time.date() == current_time.date():
        completion = completion_time.strftime("%H:%M")
    else:
        completion = completion_time.strftime("%d.%m в %H:%M")

    return f"Осталось примерно {duration} · Окончание около {completion}"
