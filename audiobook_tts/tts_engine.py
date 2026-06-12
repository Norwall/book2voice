from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from .settings import SILERO_MODEL_ID

MODELS_YML_URL = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"


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

        importer = torch.package.PackageImporter(str(model_path))
        self.model = importer.load_pickle("tts_models", "model")
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


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Cannot infer model filename from URL: {url}")
    return name
