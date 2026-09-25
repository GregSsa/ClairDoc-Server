from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse

from . import __version__
from .models import HealthResponse, JobStatus, OcrJob, Project, ProjectCreate
from .security import require_api_key
from .storage import LocalStorage, RecordNotFoundError

router = APIRouter(prefix="/api/v1")
protected = APIRouter(dependencies=[Depends(require_api_key)])


def _storage(request: Request) -> LocalStorage:
    return request.app.state.storage


def _not_found(record: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{record} introuvable.")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        storage_ready=request.app.state.storage.root.is_dir(),
        ocr_available=request.app.state.jobs._resolve_command() is not None,
        authentication_configured=bool(settings.api_key),
    )


@protected.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(request: Request, payload: ProjectCreate) -> Project:
    return _storage(request).create_project(payload)


@protected.get("/projects/{project_id}", response_model=Project)
async def get_project(request: Request, project_id: UUID) -> Project:
    try:
        return _storage(request).get_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc


@protected.post("/ocr/jobs", response_model=OcrJob, status_code=status.HTTP_202_ACCEPTED)
async def create_ocr_job(
    request: Request,
    file: Annotated[UploadFile, File(description="Document PDF à OCRiser")],
    project_id: Annotated[UUID | None, Query()] = None,
) -> OcrJob:
    storage = _storage(request)
    if project_id is not None:
        try:
            storage.get_project(project_id)
        except RecordNotFoundError as exc:
            raise _not_found("Projet") from exc

    original_filename = Path(file.filename or "document.pdf").name
    if Path(original_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Seuls les fichiers PDF sont acceptés.")

    job = storage.create_job(original_filename, project_id)
    input_path = storage.input_path(job.id)
    total_bytes = 0
    signature = b""

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                if not signature:
                    signature = chunk[:5]
                total_bytes += len(chunk)
                if total_bytes > request.app.state.settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Le fichier dépasse la taille maximale autorisée.",
                    )
                destination.write(chunk)
    except HTTPException:
        input_path.unlink(missing_ok=True)
        job.status = JobStatus.FAILED
        job.error = "Téléversement refusé."
        storage.save_job(job)
        raise
    finally:
        await file.close()

    if signature != b"%PDF-":
        input_path.unlink(missing_ok=True)
        job.status = JobStatus.FAILED
        job.error = "Le contenu envoyé n'est pas un PDF valide."
        storage.save_job(job)
        raise HTTPException(status_code=415, detail="Le contenu envoyé n'est pas un PDF valide.")

    job.input_bytes = total_bytes
    storage.save_job(job)
    await request.app.state.jobs.enqueue(job.id)
    return job


@protected.get("/ocr/jobs/{job_id}", response_model=OcrJob)
async def get_ocr_job(request: Request, job_id: UUID) -> OcrJob:
    try:
        return _storage(request).get_job(job_id)
    except RecordNotFoundError as exc:
        raise _not_found("Travail OCR") from exc


def _completed_output(request: Request, job_id: UUID, kind: str) -> tuple[OcrJob, Path]:
    storage = _storage(request)
    try:
        job = storage.get_job(job_id)
    except RecordNotFoundError as exc:
        raise _not_found("Travail OCR") from exc
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Le travail OCR n'est pas terminé.")
    output_path = storage.output_path(job_id) if kind == "document" else storage.text_path(job_id)
    if not output_path.is_file():
        raise _not_found("Résultat OCR")
    return job, output_path


@protected.get("/ocr/jobs/{job_id}/document", response_class=FileResponse)
async def get_ocr_document(request: Request, job_id: UUID) -> FileResponse:
    job, output_path = _completed_output(request, job_id, "document")
    output_name = f"{Path(job.original_filename).stem}-ocr.pdf"
    return FileResponse(output_path, media_type="application/pdf", filename=output_name)


@protected.get("/ocr/jobs/{job_id}/text", response_class=PlainTextResponse)
async def get_ocr_text(request: Request, job_id: UUID) -> PlainTextResponse:
    _, output_path = _completed_output(request, job_id, "text")
    return PlainTextResponse(output_path.read_text(encoding="utf-8", errors="replace"))


router.include_router(protected)

