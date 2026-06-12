from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def audio_to_pcm16_bytes(audio: object) -> bytes:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio).reshape(-1)
    if array.dtype == np.int16:
        return array.tobytes()
    if array.dtype.kind in {"i", "u"}:
        array = array.astype(np.float32)
        max_value = max(float(np.max(np.abs(array))), 1.0)
        array = array / max_value
    array = np.clip(array.astype(np.float32), -1.0, 1.0)
    return (array * 32767).astype(np.int16).tobytes()


def silence_pcm16_bytes(sample_rate: int, pause_ms: int) -> bytes:
    if pause_ms <= 0:
        return b""
    sample_count = int(sample_rate * pause_ms / 1000)
    return b"\x00\x00" * sample_count


def convert_wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    *,
    bitrate: str,
    speed: float = 1.0,
) -> None:
    ffmpeg = _ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-vn",
    ]
    if abs(speed - 1.0) > 0.001:
        command.extend(["-filter:a", _atempo_filter(speed)])
    command.extend(["-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)])
    _run_ffmpeg(command)


def concat_mp3_files(mp3_paths: list[Path], output_path: Path) -> None:
    if not mp3_paths:
        raise ValueError("No MP3 files to concatenate")
    ffmpeg = _ffmpeg_exe()
    list_path = output_path.with_suffix(".ffconcat.txt")
    try:
        list_path.write_text(
            "\n".join(f"file '{_escape_ffconcat_path(path)}'" for path in mp3_paths),
            encoding="utf-8",
        )
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        _run_ffmpeg(command)
    finally:
        list_path.unlink(missing_ok=True)


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "MP3 encoding requires imageio-ffmpeg. Install dependencies from requirements.txt."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown FFmpeg error"
        raise RuntimeError(message)


def _atempo_filter(speed: float) -> str:
    factors: list[float] = []
    value = speed
    while value > 2.0:
        factors.append(2.0)
        value /= 2.0
    while value < 0.5:
        factors.append(0.5)
        value /= 0.5
    factors.append(value)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def _escape_ffconcat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", r"'\''")
