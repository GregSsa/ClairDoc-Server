import asyncio
import contextlib
import logging
from uuid import UUID

from .config import Settings
from .models import IndexTask, JobStatus, utc_now
from .rag import RagService
from .storage import LocalStorage, RecordNotFoundError

logger = logging.getLogger(__name__)


class IndexJobManager:
    def __init__(self, storage: LocalStorage, rag: RagService, settings: Settings) -> None:
        self.storage = storage
        self.rag = rag
        self.settings = settings
        self.queue: asyncio.Queue[UUID] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        for task in self.storage.iter_index_tasks():
            if task.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                task.status = JobStatus.QUEUED
                task.started_at = None
                task.error = None
                self.storage.save_index_task(task)
                await self.queue.put(task.id)
        self.worker = asyncio.create_task(self._worker(), name="index-worker")

    async def stop(self) -> None:
        if self.worker is None:
            return
        self.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.worker
        self.worker = None

    async def create(self, project_id: UUID) -> IndexTask:
        for task in self.storage.iter_index_tasks():
            if task.project_id == project_id and task.status in {
                JobStatus.QUEUED,
                JobStatus.RUNNING,
            }:
                return task
        task = self.storage.create_index_task(project_id)
        await self.queue.put(task.id)
        return task

    async def retry(self, task_id: UUID) -> IndexTask:
        task = self.storage.get_index_task(task_id)
        task.status = JobStatus.QUEUED
        task.started_at = None
        task.completed_at = None
        task.error = None
        task.result = None
        self.storage.save_index_task(task)
        await self.queue.put(task.id)
        return task

    async def _worker(self) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                await self._process(task_id)
            except Exception:
                logger.exception("Échec inattendu de l'indexation %s", task_id)
                with contextlib.suppress(RecordNotFoundError):
                    task = self.storage.get_index_task(task_id)
                    task.status = JobStatus.FAILED
                    task.completed_at = utc_now()
                    task.error = "Erreur interne pendant l'indexation."
                    self.storage.save_index_task(task)
            finally:
                self.queue.task_done()

    async def _process(self, task_id: UUID) -> None:
        task = self.storage.get_index_task(task_id)
        task.status = JobStatus.RUNNING
        task.started_at = utc_now()
        task.error = None
        self.storage.save_index_task(task)
        try:
            estimate = await self.rag.estimate_project(task.project_id)
            task.estimate = estimate
            self.storage.save_index_task(task)
            if estimate.estimated_tokens > self.settings.max_index_tokens:
                raise RuntimeError(
                    "L'indexation dépasse la limite de sécurité configurée "
                    f"({self.settings.max_index_tokens:,} tokens)."
                )
            task.result = await self.rag.index_project(task.project_id)
            task.status = JobStatus.COMPLETED
            task.completed_at = utc_now()
            self.storage.save_index_task(task)
        except Exception as exc:
            task.status = JobStatus.FAILED
            task.completed_at = utc_now()
            task.error = str(exc)[-4000:] or "Indexation impossible."
            self.storage.save_index_task(task)
