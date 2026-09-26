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
    assert arguments[arguments.index("--output-type") + 1] == "pdf"


def test_pdf_text_includes_existing_pages_and_form_values(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from clairdoc_server.models import ProjectCreate

    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Test"))
    job = storage.create_job("form.pdf", project.id, "form.pdf")
    reader = SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda: "Texte numérique existant")],
        get_fields=lambda: {"Montant": {"/V": "120"}},
    )
    monkeypatch.setattr("clairdoc_server.jobs.PdfReader", lambda path: reader)
    manager = OcrJobManager(storage, Settings(data_dir=tmp_path / "data"))
    storage.text_path(job.id).write_text("[OCR skipped on page 1]", encoding="utf-8")
    manager._extract_pdf_text(job.id)
    text = storage.text_path(job.id).read_text(encoding="utf-8")
    assert "Texte numérique existant" in text
    assert "Montant: 120" in text
    assert "OCR skipped" not in text
