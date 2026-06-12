from __future__ import annotations

from dataclasses import dataclass

SILERO_MODEL_ID = "v5_5_ru"
SILERO_VOICES = ("aidar", "baya", "kseniya", "xenia", "eugene")
SAMPLE_RATES = (8000, 24000, 48000)


@dataclass(frozen=True)
class GenerationSettings:
    voice: str = "baya"
    sample_rate: int = 24000
    max_chunk_chars: int = 850
    pause_ms: int = 350
    speech_speed: float = 1.0
    target_chapter_minutes: int = 20
    mp3_bitrate: str = "128k"
    torch_threads: int = 4
    model_id: str = SILERO_MODEL_ID

    def validate(self) -> None:
        if self.voice not in SILERO_VOICES:
            raise ValueError(f"Unknown Silero voice: {self.voice}")
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if not 250 <= self.max_chunk_chars <= 2000:
            raise ValueError("max_chunk_chars must be between 250 and 2000")
        if not 0 <= self.pause_ms <= 3000:
            raise ValueError("pause_ms must be between 0 and 3000")
        if not 0.5 <= self.speech_speed <= 2.0:
            raise ValueError("speech_speed must be between 0.5 and 2.0")
        if not 1 <= self.target_chapter_minutes <= 240:
            raise ValueError("target_chapter_minutes must be between 1 and 240")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
