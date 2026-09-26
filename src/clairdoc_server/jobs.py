import asyncio
import contextlib
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from .config import Settings
from .extractors import IMAGE_EXTENSIONS, extract_document_text
from .models import JobStatus, utc_now
from .storage import LocalStorage, RecordNotFoundError

logger = logging.getLogger(__name__)
EXTRACTION_VERSION = 2


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

        try:
            code, stdout, stderr = await self._run_text_process(arguments)
        except TimeoutError:
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = "Le délai maximal du traitement OCR a été dépassé."
            self.storage.save_job(job)
            return

        diagnostic = (stderr or stdout).decode("utf-8", errors="replace").strip()
        (self.storage.job_dir(job_id) / "ocr.log").write_text(diagnostic, encoding="utf-8")
        if code == 0:
            try:
                text = await asyncio.to_thread(self._extract_pdf_text, job_id)
                if not text.strip():
                    text = await self._redo_pdf_text(command, job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.completed_at = utc_now()
                job.error = f"PDF traité, mais extraction du texte impossible : {exc}"[-4000:]
                self.storage.save_job(job)
                return
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            job.error = None
            job.text_extraction_version = EXTRACTION_VERSION
            job.text_warning = (
                None if text.strip() else "Aucun texte récupéré : indexation par nom uniquement."
            )
            self.storage.save_job(job)
            logger.info("Travail OCR terminé : %s", job_id)
            return

        job.status = JobStatus.FAILED
        job.completed_at = utc_now()
        job.error = diagnostic[-4000:] or f"OCRmyPDF a retourné le code {code}."
        self.storage.save_job(job)

    def _extract_pdf_text(self, job_id: UUID) -> str:
        # The OCR sidecar omits pages with existing text. Read the whole resulting PDF.
        reader = PdfReader(self.storage.output_path(job_id))
        parts = [page.extract_text() or "" for page in reader.pages]
        for name, field in (reader.get_fields() or {}).items():
            value = field.get("/V")
            if value is not None:
                parts.append(f"{name}: {value}")
        text = "\n\f\n".join(parts)
        self.storage.text_path(job_id).write_text(text, encoding="utf-8")
        return text

    async def _run_text_process(self, arguments: list[str]) -> tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.ocr_timeout_seconds
            )
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise
        return process.returncode or 0, stdout, stderr

    async def _redo_pdf_text(self, command: str, job_id: UUID) -> str:
        # Redo cannot be combined with deskew/other image processing. Keep originals intact.
        with tempfile.TemporaryDirectory(
            prefix="redo-", dir=self.storage.job_dir(job_id)
        ) as folder:
            output = Path(folder) / "output.pdf"
            code, _, stderr = await self._run_text_process(
                [
                    command,
                    "--redo-ocr",
                    "--output-type",
                    "pdf",
                    "--optimize",
                    "0",
                    "-l",
                    self.settings.ocr_languages,
                    str(self.storage.input_path(job_id)),
                    str(output),
                ]
            )
            with (self.storage.job_dir(job_id) / "ocr.log").open("a", encoding="utf-8") as log:
                log.write("\nREPRISE OCR\n" + stderr.decode("utf-8", errors="replace"))
            if code != 0:
                raise RuntimeError(
                    "La reprise OCR a échoué : " + stderr.decode("utf-8", errors="replace")[-3000:]
                )
            output.replace(self.storage.output_path(job_id))
        return await asyncio.to_thread(self._extract_pdf_text, job_id)

    async def _ocr_image(self, path: Path) -> str:
        command = self._resolve_tesseract()
        if command is None:
            raise RuntimeError("Tesseract est introuvable sur le serveur.")
        code, stdout, stderr = await self._run_text_process(
            [command, str(path), "stdout", "-l", self.settings.ocr_languages]
        )
        if code != 0:
            raise RuntimeError(
                stderr.decode("utf-8", errors="replace")[-4000:] or "Tesseract a échoué."
            )
        return stdout.decode("utf-8", errors="replace")

    async def _docx_image_text(self, source: Path, job_id: UUID) -> str:
        parts = []
        with zipfile.ZipFile(source) as archive:
            for number, info in enumerate(archive.infolist()):
                suffix = Path(info.filename).suffix.lower()
                if not info.filename.startswith("word/media/") or suffix not in IMAGE_EXTENSIONS:
                    continue
                if info.file_size > self.settings.max_upload_bytes:
                    raise ValueError("Une image intégrée au Word dépasse la limite de taille.")
                with tempfile.TemporaryDirectory(
                    prefix="image-", dir=self.storage.job_dir(job_id)
                ) as folder:
                    image = Path(folder) / f"image-{number}{suffix}"
                    image.write_bytes(archive.read(info))
                    text = await self._ocr_image(image)
                if text.strip():
                    parts.append(f"Image intégrée {number + 1}\n{text}")
        return "\n\n".join(parts)

    async def recover_project_text(self, project_id: UUID) -> None:
        # Migrate previously completed imports once, without reimporting or touching originals.
        for job in self.storage.jobs_for_project(project_id):
            if (
                job.status != JobStatus.COMPLETED
                or job.text_extraction_version >= EXTRACTION_VERSION
            ):
                continue
            source = self.storage.source_path(job.id)
            if source.suffix.lower() == ".pdf" and self.storage.output_path(job.id).is_file():
                text = await asyncio.to_thread(self._extract_pdf_text, job.id)
                if text.strip():
                    job.text_extraction_version = EXTRACTION_VERSION
                    job.text_warning = None
                    self.storage.save_job(job)
                    continue
            await self._process(job.id)

    async def _process_non_pdf(self, job_id: UUID, source_path: Path) -> None:
        job = self.storage.get_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.error = None
        self.storage.save_job(job)
        self.storage.text_path(job_id).unlink(missing_ok=True)

        try:
            if source_path.suffix.lower() in IMAGE_EXTENSIONS:
                text = await self._ocr_image(source_path)
            else:
                text = await asyncio.to_thread(extract_document_text, source_path)
                if source_path.suffix.lower() == ".docx":
                    image_text = await self._docx_image_text(source_path, job_id)
                    text = "\n\n".join(part for part in (text, image_text) if part.strip())
            self.storage.text_path(job_id).write_text(text, encoding="utf-8")
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            job.text_extraction_version = EXTRACTION_VERSION
            job.text_warning = (
                None if text.strip() else "Aucun texte récupéré : indexation par nom uniquement."
            )
            self.storage.save_job(job)
            logger.info("Extraction terminée : %s", job_id)
        except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
            job.status = JobStatus.FAILED
            job.completed_at = utc_now()
            job.error = str(exc)[-4000:] or "Extraction du document impossible."
            self.storage.save_job(job)
