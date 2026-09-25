import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from uuid import UUID

from .config import Settings
from .models import JobStatus, utc_now
from .storage import LocalStorage, RecordNotFoundError

logger = logging.getLogger(__name__)


class OcrJobManager:
    def __init__(self, storage: LocalStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.queue: asyncio.Queue[UUID] = asyncio.Queue()
        self.workers: list[asyncio.Task[None]] = []

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

    async def _process(self, job_id: UUID) -> None:
        job = self.storage.get_job(job_id)
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

        arguments = [
            command,
            "--mode",
            "skip",
            "--rotate-pages",
            "--deskew",
            "--language",
            self.settings.ocr_languages,
            "--sidecar",
            str(self.storage.text_path(job_id)),
            str(self.storage.input_path(job_id)),
            str(self.storage.output_path(job_id)),
        ]

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

        if process.returncode == 0:
            job.status = JobStatus.COMPLETED
            job.completed_at = utc_now()
            job.error = None
            self.storage.save_job(job)
            logger.info("Travail OCR terminé : %s", job_id)
            return

        diagnostic = (stderr or stdout).decode("utf-8", errors="replace").strip()
        job.status = JobStatus.FAILED
        job.completed_at = utc_now()
        job.error = diagnostic[-4000:] or f"OCRmyPDF a retourné le code {process.returncode}."
        self.storage.save_job(job)
