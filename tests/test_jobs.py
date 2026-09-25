from pathlib import Path
from uuid import uuid4

from clairdoc_server.config import Settings
from clairdoc_server.jobs import OcrJobManager
from clairdoc_server.storage import LocalStorage


def test_ocr_arguments_are_compatible_with_legacy_ocrmypdf(tmp_path: Path) -> None:
    manager = OcrJobManager(
        LocalStorage(tmp_path / "data"),
        Settings(data_dir=tmp_path / "data", ocr_languages="fra+eng"),
    )

    arguments = manager._build_arguments("ocrmypdf", uuid4())

    assert arguments[0] == "ocrmypdf"
    assert "--skip-text" in arguments
    assert "--mode" not in arguments
    assert arguments[arguments.index("--rotate-pages-threshold") + 1] == "2"
    assert arguments[arguments.index("--oversample") + 1] == "300"
    assert arguments[arguments.index("--language") + 1] == "fra+eng"
