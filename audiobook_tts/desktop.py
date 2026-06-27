from __future__ import annotations

import os
from pathlib import Path


def open_folder(path: Path) -> None:
    folder = path.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Папка результата ещё не создана: {folder}")
    if os.name != "nt":
        raise RuntimeError("Открытие папки поддерживается только при локальном запуске на Windows")

    startfile = getattr(os, "startfile", None)
    if startfile is None:
        raise RuntimeError("Системная команда открытия папки недоступна")
    startfile(str(folder))
