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


def test_docx_embedded_images_are_ocred_and_blank_is_success(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import zipfile

    from clairdoc_server.models import JobStatus, ProjectCreate

    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Maison"))
    job = storage.create_job("scan.docx", project.id, "scan.docx")
    source = storage.source_path(job.id)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/media/image1.png", b"test-image")
    manager = OcrJobManager(storage, Settings(data_dir=tmp_path / "data"))
    monkeypatch.setattr("clairdoc_server.jobs.extract_document_text", lambda path: "")

    async def ocr(path):
        assert path.read_bytes() == b"test-image"
        return "Contrat extrait de l'image"

    monkeypatch.setattr(manager, "_ocr_image", ocr)
    asyncio.run(manager._process_non_pdf(job.id, source))
    assert "Contrat extrait" in storage.text_path(job.id).read_text(encoding="utf-8")
    assert storage.get_job(job.id).text_extraction_version == 2

    async def blank(path):
        return ""

    monkeypatch.setattr(manager, "_ocr_image", blank)
    asyncio.run(manager._process_non_pdf(job.id, source))
    result = storage.get_job(job.id)
    assert result.status == JobStatus.COMPLETED
    assert "nom uniquement" in result.text_warning
    assert storage.text_path(job.id).read_text(encoding="utf-8") == ""


def test_empty_pdf_retries_ocr_once_then_completes_by_name(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from clairdoc_server.models import JobStatus, ProjectCreate

    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Maison"))
    job = storage.create_job("plan.pdf", project.id, "plan.pdf")
    storage.input_path(job.id).write_bytes(b"original")
    manager = OcrJobManager(storage, Settings(data_dir=tmp_path / "data"))
    monkeypatch.setattr(manager, "_resolve_command", lambda: "ocrmypdf")
    calls = []

    async def run(arguments):
        calls.append(arguments)
        Path(arguments[-1]).write_bytes(b"output")
        return 0, b"", b"OCR completed"

    def extract(job_id):
        storage.text_path(job_id).write_text("", encoding="utf-8")
        return ""

    monkeypatch.setattr(manager, "_run_text_process", run)
    monkeypatch.setattr(manager, "_extract_pdf_text", extract)
    asyncio.run(manager._process(job.id))
    result = storage.get_job(job.id)
    assert result.status == JobStatus.COMPLETED
    assert result.text_warning
    assert len(calls) == 2
    assert "--redo-ocr" in calls[1]
    assert "--deskew" not in calls[1]
    assert storage.input_path(job.id).read_bytes() == b"original"
    asyncio.run(manager.recover_project_text(project.id))
    assert len(calls) == 2  # Blank documents do not trigger an infinite OCR loop.
