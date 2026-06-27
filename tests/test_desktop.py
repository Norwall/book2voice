from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from audiobook_tts.desktop import open_folder


class OpenFolderTests(unittest.TestCase):
    def test_opens_resolved_folder_with_spaces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "audio book"
            folder.mkdir()

            with (
                patch("audiobook_tts.desktop.os.name", "nt"),
                patch("audiobook_tts.desktop.os.startfile", create=True) as startfile,
            ):
                open_folder(folder)

            startfile.assert_called_once_with(str(folder.resolve()))

    def test_rejects_missing_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_folder = Path(temp_dir) / "missing"

            with self.assertRaisesRegex(FileNotFoundError, "ещё не создана"):
                open_folder(missing_folder)


if __name__ == "__main__":
    unittest.main()
