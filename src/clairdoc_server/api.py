import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse
from openai import OpenAIError

from . import __version__
from .extractors import SUPPORTED_EXTENSIONS
from .models import (
    AskRequest,
    AskResponse,
    BackupResponse,
    ConnectionResponse,
    DocumentLibrary,
    HealthResponse,
    IndexEstimate,
    IndexResponse,
    IndexTask,
    JobStatus,
    OcrJob,
    OrganizationPlan,
    Project,
    ProjectCreate,
    ProjectOcrState,
    ProjectUpdate,
    RuntimeInfo,
    RuntimeUpdate,
)
from .rag import NoDocumentsError, OpenAIConfigurationError, ProjectIndexNotFoundError
from .security import require_api_key
from .storage import LocalStorage, RecordNotFoundError

router = APIRouter(prefix="/api/v1")
protected = APIRouter(dependencies=[Depends(require_api_key)])
LLM_MODEL_OPTIONS = ["gpt-6-luna", "gpt-5.6-terra", "gpt-6-sol"]


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
        openai_configured=bool(settings.openai_api_key),
    )


@protected.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(request: Request, payload: ProjectCreate) -> Project:
    return _storage(request).create_project(payload)


@protected.get("/projects", response_model=list[Project])
async def list_projects(request: Request) -> list[Project]:
    return _storage(request).iter_projects()


@protected.get("/connection", response_model=ConnectionResponse)
async def verify_connection() -> ConnectionResponse:
    return ConnectionResponse(status="authenticated", version=__version__)


@protected.get("/projects/{project_id}", response_model=Project)
async def get_project(request: Request, project_id: UUID) -> Project:
    try:
        return _storage(request).get_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc


@protected.patch("/projects/{project_id}", response_model=Project)
async def update_project(request: Request, project_id: UUID, payload: ProjectUpdate) -> Project:
    try:
        project = _storage(request).get_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    if payload.name is not None:
        project.name = payload.name.strip()
    if "description" in payload.model_fields_set:
        project.description = payload.description
    if "source_root" in payload.model_fields_set:
        project.source_root = payload.source_root
    _storage(request).save_project(project)
    return project


@protected.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(request: Request, project_id: UUID) -> None:
    try:
        active_ocr = any(
            job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            for job in _storage(request).jobs_for_project(project_id)
        )
        active_index = any(
            task.project_id == project_id
            and task.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            for task in _storage(request).iter_index_tasks()
        )
        if active_ocr or active_index:
            raise HTTPException(
                status_code=409,
                detail="Attendez la fin des traitements avant de supprimer le projet.",
            )
        _storage(request).delete_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc


@protected.get("/runtime", response_model=RuntimeInfo)
async def runtime_info(request: Request) -> RuntimeInfo:
    settings = request.app.state.settings
    return RuntimeInfo(
        version=__version__,
        llm_model=_storage(request).get_llm_model(settings.llm_model),
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        ocr_languages=settings.ocr_languages,
        data_dir=str(settings.data_dir.resolve()),
        max_upload_mb=settings.max_upload_mb,
        max_index_tokens=settings.max_index_tokens,
        openai_configured=bool(settings.openai_api_key),
        tls_enabled=bool(settings.tls_certfile and settings.tls_keyfile),
        model_options=LLM_MODEL_OPTIONS,
    )


@protected.patch("/runtime", response_model=RuntimeInfo)
async def update_runtime(request: Request, payload: RuntimeUpdate) -> RuntimeInfo:
    if payload.llm_model not in LLM_MODEL_OPTIONS:
        raise HTTPException(status_code=422, detail="Ce modèle n'est pas proposé par ClairDoc.")
    _storage(request).set_llm_model(payload.llm_model)
    return await runtime_info(request)


@protected.get("/projects/{project_id}/ocr/jobs", response_model=list[OcrJob])
async def list_project_jobs(request: Request, project_id: UUID) -> list[OcrJob]:
    try:
        _storage(request).get_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    return _storage(request).jobs_for_project(project_id)


@protected.post("/projects/{project_id}/ocr/pause", response_model=ProjectOcrState)
async def pause_project_ocr(request: Request, project_id: UUID) -> ProjectOcrState:
    try:
        await request.app.state.jobs.pause_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    return ProjectOcrState(project_id=project_id, paused=True)


@protected.post("/projects/{project_id}/ocr/resume", response_model=ProjectOcrState)
async def resume_project_ocr(request: Request, project_id: UUID) -> ProjectOcrState:
    try:
        await request.app.state.jobs.resume_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    return ProjectOcrState(project_id=project_id, paused=False)


def _rag_error(exc: Exception) -> HTTPException:
    if isinstance(exc, OpenAIConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (NoDocumentsError, ProjectIndexNotFoundError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, OpenAIError):
        return HTTPException(
            status_code=502,
            detail="Le service OpenAI n'a pas pu traiter la demande.",
        )
    return HTTPException(status_code=500, detail="Erreur d'indexation inattendue.")


@protected.post("/projects/{project_id}/index", response_model=IndexResponse)
async def index_project(request: Request, project_id: UUID) -> IndexResponse:
    try:
        return await request.app.state.rag.index_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    except (OpenAIConfigurationError, NoDocumentsError, OpenAIError) as exc:
        raise _rag_error(exc) from exc


@protected.get("/projects/{project_id}/index/estimate", response_model=IndexEstimate)
async def estimate_project_index(request: Request, project_id: UUID) -> IndexEstimate:
    try:
        return await request.app.state.rag.estimate_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc


@protected.get("/projects/{project_id}/documents", response_model=DocumentLibrary)
async def list_project_documents(request: Request, project_id: UUID) -> DocumentLibrary:
    try:
        return request.app.state.rag.document_library(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc


@protected.post(
    "/projects/{project_id}/index/jobs",
    response_model=IndexTask,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_index_task(request: Request, project_id: UUID) -> IndexTask:
    try:
        _storage(request).get_project(project_id)
    except RecordNotFoundError as exc:
        raise _not_found("Projet") from exc
    return await request.app.state.index_jobs.create(project_id)


@protected.get("/index/jobs/{task_id}", response_model=IndexTask)
async def get_index_task(request: Request, task_id: UUID) -> IndexTask:
    try:
        return _storage(request).get_index_task(task_id)
    except RecordNotFoundError as exc:
        raise _not_found("Indexation") from exc


@protected.post("/index/jobs/{task_id}/retry", response_model=IndexTask)
async def retry_index_task(request: Request, task_id: UUID) -> IndexTask:
    try:
        task = _storage(request).get_index_task(task_id)
    except RecordNotFoundError as exc:
        raise _not_found("Indexation") from exc
    if task.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail="Seule une indexation en échec peut être relancée.",
        )
    return await request.app.state.index_jobs.retry(task_id)


@protected.post("/projects/{project_id}/ask", response_model=AskResponse)
async def ask_project(request: Request, project_id: UUID, payload: AskRequest) -> AskResponse:
    try:
        return await request.app.state.rag.ask(project_id, payload.question.strip(), payload.top_k)
    except (
        OpenAIConfigurationError,
        NoDocumentsError,
        ProjectIndexNotFoundError,
        OpenAIError,
    ) as exc:
        raise _rag_error(exc) from exc


@protected.post("/projects/{project_id}/organization/plan", response_model=OrganizationPlan)
async def create_organization_plan(request: Request, project_id: UUID) -> OrganizationPlan:
    try:
        return request.app.state.organization.build_plan(project_id)
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail="Le projet doit être indexé avant de préparer le classement.",
        ) from exc


@protected.post("/maintenance/backups", response_model=BackupResponse)
async def create_backup(request: Request) -> BackupResponse:
    path = await asyncio.to_thread(_storage(request).create_metadata_backup)
    return BackupResponse(
        path=str(path),
        size_bytes=path.stat().st_size,
        created_at=datetime.now(UTC),
    )


@protected.post("/ocr/jobs", response_model=OcrJob, status_code=status.HTTP_202_ACCEPTED)
async def create_ocr_job(
    request: Request,
    file: Annotated[UploadFile, File(description="Document PDF à OCRiser")],
    project_id: Annotated[UUID | None, Query()] = None,
) -> OcrJob:
    return await _create_document_job(request, file, project_id, {".pdf"}, None)


@protected.post("/document/jobs", response_model=OcrJob, status_code=status.HTTP_202_ACCEPTED)
async def create_document_job(
    request: Request,
    file: Annotated[UploadFile, File(description="Document à analyser")],
    project_id: Annotated[UUID | None, Query()] = None,
    source_relative_path: Annotated[str | None, Query(max_length=2000)] = None,
) -> OcrJob:
    return await _create_document_job(
        request,
        file,
        project_id,
        SUPPORTED_EXTENSIONS,
        source_relative_path,
    )


async def _create_document_job(
    request: Request,
    file: UploadFile,
    project_id: UUID | None,
    allowed_extensions: set[str],
    source_relative_path: str | None,
) -> OcrJob:
    storage = _storage(request)
    if project_id is not None:
        try:
            storage.get_project(project_id)
        except RecordNotFoundError as exc:
            raise _not_found("Projet") from exc

    original_filename = Path(file.filename or "document.pdf").name
    extension = Path(original_filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Ce format de document n'est pas pris en charge.",
        )

    if source_relative_path:
        relative = Path(source_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise HTTPException(status_code=400, detail="Chemin relatif source invalide.")
        source_relative_path = relative.as_posix()

    job = storage.create_job(original_filename, project_id, source_relative_path)
    input_path = storage.source_path(job.id)
    total_bytes = 0
    signature = b""
    digest = hashlib.sha256()

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                if not signature:
                    signature = chunk[:5]
                total_bytes += len(chunk)
                digest.update(chunk)
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

    if extension == ".pdf" and signature != b"%PDF-":
        input_path.unlink(missing_ok=True)
        job.status = JobStatus.FAILED
        job.error = "Le contenu envoyé n'est pas un PDF valide."
        storage.save_job(job)
        raise HTTPException(status_code=415, detail="Le contenu envoyé n'est pas un PDF valide.")

    job.input_bytes = total_bytes
    job.content_sha256 = digest.hexdigest()
    if project_id is not None:
        duplicate = storage.find_job_by_hash(project_id, job.content_sha256, job.id)
        if duplicate is not None:
            storage.delete_job(job.id)
            return duplicate
    storage.save_job(job)
    await request.app.state.jobs.enqueue(job.id)
    return job


@protected.get("/ocr/jobs/{job_id}", response_model=OcrJob)
async def get_ocr_job(request: Request, job_id: UUID) -> OcrJob:
    try:
        return _storage(request).get_job(job_id)
    except RecordNotFoundError as exc:
        raise _not_found("Travail OCR") from exc


@protected.post("/ocr/jobs/{job_id}/retry", response_model=OcrJob)
async def retry_ocr_job(request: Request, job_id: UUID) -> OcrJob:
    storage = _storage(request)
    try:
        job = storage.get_job(job_id)
    except RecordNotFoundError as exc:
        raise _not_found("Travail OCR") from exc
    if job.status != JobStatus.FAILED:
        raise HTTPException(status_code=409, detail="Seul un travail en échec peut être relancé.")
    if not storage.source_path(job_id).is_file():
        raise _not_found("Document source")
    await request.app.state.jobs.retry(job_id)
    return storage.get_job(job_id)


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
