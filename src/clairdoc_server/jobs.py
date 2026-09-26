import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from .config import Settings
from .extractors import IMAGE_EXTENSIONS, extract_document_text
from .models import JobStatus, utc_now
from .storage import LocalStorage, RecordNotFoundError

logger = logging.getLogger(__name__)


class OcrJobManager:
    def __init__(self, storage: LocalStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.queue: asyncio.Queue[UUID] = asyncio.Queue()
        self.workers: list[asyncio.Task[None]] = []
        self._resume_condition = asyncio.Condition()

    async def start(self) -> None:
        for job in self.storage.iter_jobs():
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job.status = JobStatus.QUEUED
                job.started_at = None
                job.error = None
                self.storage.save_job(job)
                await self.queue.put(job.id)

        self.workers = [
            asyncio.create_task(self._worker(index), name=f"ocr-worker-{index}")
            for index in range(self.settings.ocr_workers)
        ]

    async def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()
        for worker in self.workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self.workers.clear()

    async def enqueue(self, job_id: UUID) -> None:
        await self.queue.put(job_id)

    async def pause_project(self, project_id: UUID) -> None:
        project = self.storage.get_project(project_id)
        project.ocr_paused = True
        self.storage.save_project(project)

    async def resume_project(self, project_id: UUID) -> None:
        project = self.storage.get_project(project_id)
        project.ocr_paused = False
        self.storage.save_project(project)
        async with self._resume_condition:
            self._resume_condition.notify_all()

    async def retry(self, job_id: UUID) -> None:
        job = self.storage.get_job(job_id)
        job.status = JobStatus.QUEUED
        job.started_at = None
        job.completed_at = None
        job.error = None
        self.storage.save_job(job)
        await self.enqueue(job_id)

    async def _wait_if_paused(self, project_id: UUID | None) -> None:
        if project_id is None:
            return
        async with self._resume_condition:
            await self._resume_condition.wait_for(
                lambda: not self.storage.get_project(project_id).ocr_paused
            )

    async def _worker(self, index: int) -> None:
        logger.info("Démarrage du worker OCR %s", index)
        while True:
            job_id = await self.queue.get()
            try:
                await self._process(job_id)
            except Exception:
                logger.exception("Échec inattendu du travail OCR %s", job_id)
                with contextlib.suppress(RecordNotFoundError):
                    job = self.storage.get_job(job_id)
                    job.status = JobStatus.FAILED
                    job.completed_at = utc_now()
                    job.error = "Erreur interne pendant le traitement OCR."
                    self.storage.save_job(job)
            finally:
                self.queue.task_done()

    def _resolve_command(self) -> str | None:
        configured = Path(self.settings.ocr_command)
        if configured.is_file():
            return str(configured.resolve())
        return shutil.which(self.settings.ocr_command)

    def _resolve_tesseract(self) -> str | None:
        configured = Path(self.settings.tesseract_command)
        if configured.is_file():
            return str(configured.resolve())
        return shutil.which(self.settings.tesseract_command)

    def _build_arguments(self, command: str, job_id: UUID) -> list[str]:
        return [
            command,
            "--skip-text",
            "--rotate-pages",
            "--rotate-pages-threshold",
            "2",
            "--deskew",
            "--oversample",
            "300",
            "--output-type",
            "pdf",
            "--language",
            self.settings.ocr_languages,
            "--sidecar",
            str(self.storage.text_path(job_id)),
            str(self.storage.input_path(job_id)),
            str(self.storage.output_path(job_id)),
        ]

    async def _process(self, job_id: UUID) -> None:
        job = self.storage.get_job(job_id)
        await self._wait_if_paused(job.project_id)
        job = self.storage.get_job(job_id)
        source_path = self.storage.source_path(job_id)
        if source_path.suffix.lower() != ".pdf":
            await self._process_non_pdf(job_id, source_path)
            return
        command = self._resolve_command()
        if command is None:
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = "OCRmyPDF est introuvable sur le serveur. Vérifiez CLAIRDOC_OCR_COMMAND."
            self.storage.save_job(job)
            return

        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.error = None
        self.storage.save_job(job)

        self.storage.output_path(job_id).unlink(missing_ok=True)
        self.storage.text_path(job_id).unlink(missing_ok=True)

        arguments = self._build_arguments(command, job_id)

        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.ocr_timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = "Le délai maximal du traitement OCR a été dépassé."
            self.storage.save_job(job)
            return

        diagnostic = (stderr or stdout).decode("utf-8", errors="replace").strip()
        (self.storage.job_dir(job_id) / "ocr.log").write_text(diagnostic, encoding="utf-8")
        if process.returncode == 0:
            try:
                await asyncio.to_thread(self._extract_pdf_text, job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.completed_at = utc_now()
                job.error = f"PDF traité, mais extraction du texte impossible : {exc}"[-4000:]
                self.storage.save_job(job)
                return
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            job.error = None
            self.storage.save_job(job)
            logger.info("Travail OCR terminé : %s", job_id)
            return

        job.status = JobStatus.FAILED
        job.completed_at = utc_now()
        job.error = diagnostic[-4000:] or f"OCRmyPDF a retourné le code {process.returncode}."
        self.storage.save_job(job)

    def _extract_pdf_text(self, job_id: UUID) -> None:
        # The OCR sidecar omits pages with existing text. Read the whole resulting PDF.
        reader = PdfReader(self.storage.output_path(job_id))
        parts = [page.extract_text() or "" for page in reader.pages]
        for name, field in (reader.get_fields() or {}).items():
            value = field.get("/V")
            if value is not None:
                parts.append(f"{name}: {value}")
        text = "\n\f\n".join(parts)
        if not text.strip():
            raise ValueError("Aucun texte exploitable détecté dans le PDF.")
        self.storage.text_path(job_id).write_text(text, encoding="utf-8")

    async def _process_non_pdf(self, job_id: UUID, source_path: Path) -> None:
        job = self.storage.get_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.error = None
        self.storage.save_job(job)
        self.storage.text_path(job_id).unlink(missing_ok=True)

        try:
            if source_path.suffix.lower() in IMAGE_EXTENSIONS:
                command = self._resolve_tesseract()
                if command is None:
                    raise RuntimeError("Tesseract est introuvable sur le serveur.")
                process = await asyncio.create_subprocess_exec(
                    command,
                    str(source_path),
                    "stdout",
                    "-l",
                    self.settings.ocr_languages,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.settings.ocr_timeout_seconds
                )
                if process.returncode != 0:
                    diagnostic = stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(diagnostic[-4000:] or "Tesseract a échoué.")
                text = stdout.decode("utf-8", errors="replace")
            else:
                text = await asyncio.to_thread(extract_document_text, source_path)
            self.storage.text_path(job_id).write_text(text, encoding="utf-8")
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            self.storage.save_job(job)
            logger.info("Extraction terminée : %s", job_id)
        except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = str(exc)[-4000:] or "Extraction du document impossible."
            self.storage.save_job(job)
