from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .settings import SILERO_MODEL_ID
from .text_utils import hard_split_text

MODELS_YML_URL = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
MIN_TTS_RETRY_CHARS = 120
TTS_TOO_LONG_MARKERS = (
    "probably it's too long",
    "size of tensor a",
    "must match the size of tensor b",
    "5000",
)


class SileroTtsEngine:
    def __init__(
        self,
        *,
        model_id: str = SILERO_MODEL_ID,
        torch_threads: int = 4,
        cache_dir: Path | None = None,
    ):
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "Silero TTS requires torch and omegaconf. Install dependencies from requirements.txt."
            ) from exc

        self.torch = torch
        self.cache_dir = cache_dir or Path(".models") / "silero"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
        self.device = torch.device("cpu")

        models_yml = self._models_yml_path(torch)
        models = OmegaConf.load(models_yml)
        if model_id not in models.tts_models.ru:
            available = ", ".join(models.tts_models.ru.keys())
            raise ValueError(f"Unknown Silero RU model '{model_id}'. Available: {available}")

        model_url = models.tts_models.ru[model_id].latest.package
        model_path = self.cache_dir / _filename_from_url(model_url)
        if not model_path.exists():
            self._download_model(torch, model_url, model_path)

        self.model = self._load_model_with_recovery(torch, model_url, model_path)
        self.model.to(self.device)

    def synthesize(self, text: str, *, voice: str, sample_rate: int):
        with self.torch.inference_mode():
            return self.model.apply_tts(
                text=text,
                speaker=voice,
                sample_rate=sample_rate,
            )

    def _models_yml_path(self, torch) -> Path:
        target = self.cache_dir / "models.yml"
        if target.exists():
            return target

        root_copy = Path("latest_silero_models.yml")
        if root_copy.exists():
            shutil.copyfile(root_copy, target)
            return target

        torch.hub.download_url_to_file(MODELS_YML_URL, str(target), progress=False)
        return target

    def _download_model(self, torch, model_url: str, model_path: Path) -> None:
        for stale in self.cache_dir.glob(f"{model_path.name}*.partial"):
            stale.unlink(missing_ok=True)

        temp_path = model_path.with_name(f"{model_path.name}.download")
        temp_path.unlink(missing_ok=True)
        try:
            torch.hub.download_url_to_file(model_url, str(temp_path), progress=True)
            temp_path.replace(model_path)
        finally:
            temp_path.unlink(missing_ok=True)
            for stale in self.cache_dir.glob(f"{temp_path.name}*.partial"):
                stale.unlink(missing_ok=True)

    def _load_model_with_recovery(self, torch, model_url: str, model_path: Path):
        try:
            return self._load_model(model_path)
        except Exception:
            corrupt_path = (
                _quarantine_corrupted_model(model_path)
                if model_path.exists()
                else model_path
            )
            try:
                self._download_model(torch, model_url, model_path)
                return self._load_model(model_path)
            except Exception as retry_exc:
                raise RuntimeError(
                    "Silero model cache is corrupted and the model could not be "
                    f"downloaded again. Corrupted file was moved to {corrupt_path}. "
                    "Check internet connection or delete the .models/silero cache."
                ) from retry_exc

    def _load_model(self, model_path: Path):
        importer = self.torch.package.PackageImporter(str(model_path))
        return importer.load_pickle("tts_models", "model")


def synthesize_with_length_retry(
    engine: SileroTtsEngine,
    text: str,
    *,
    voice: str,
    sample_rate: int,
    min_chars: int = MIN_TTS_RETRY_CHARS,
) -> list[Any]:
    try:
        return [engine.synthesize(text, voice=voice, sample_rate=sample_rate)]
    except Exception as exc:
        if not _is_tts_too_long_error(exc) or len(text) <= min_chars:
            raise

        split_chars = max(min_chars, len(text) // 2)
        parts = hard_split_text(text, split_chars)
        if len(parts) <= 1:
            raise

    audio_parts: list[Any] = []
    for part in parts:
        audio_parts.extend(
            synthesize_with_length_retry(
                engine,
                part,
                voice=voice,
                sample_rate=sample_rate,
                min_chars=min_chars,
            )
        )
    return audio_parts


def _is_tts_too_long_error(exc: Exception) -> bool:
    message = _exception_chain_text(exc).lower()
    if TTS_TOO_LONG_MARKERS[0] in message:
        return True
    return all(marker in message for marker in TTS_TOO_LONG_MARKERS[1:])


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Cannot infer model filename from URL: {url}")
    return name


def _quarantine_corrupted_model(model_path: Path) -> Path:
    corrupt_path = model_path.with_name(f"{model_path.name}.corrupt")
    corrupt_path.unlink(missing_ok=True)
    model_path.replace(corrupt_path)
    return corrupt_path
