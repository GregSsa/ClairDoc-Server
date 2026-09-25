import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .models import (
    Conversation,
    IndexTask,
    JobStatus,
    OcrJob,
    Project,
    ProjectCreate,
    ProjectMemory,
    utc_now,
)

SCHEMA_VERSION = 3


class RecordNotFoundError(FileNotFoundError):
    pass


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.projects_dir = self.root / "projects"
        self.jobs_dir = self.root / "jobs"
        self.logs_dir = self.root / "logs"
        self.indexes_dir = self.root / "indexes"
        self.prompts_dir = self.root / "prompts"
        self.plans_dir = self.root / "plans"
        self.index_tasks_dir = self.root / "index-tasks"
        self.backups_dir = self.root / "backups"
        self.conversations_dir = self.root / "conversations"
        self.memories_dir = self.root / "memories"
        self.action_logs_dir = self.root / "action-logs"
        self.runtime_path = self.root / "runtime.json"

    def initialize(self) -> None:
        for directory in (
            self.projects_dir,
            self.jobs_dir,
            self.logs_dir,
            self.indexes_dir,
            self.prompts_dir,
            self.plans_dir,
            self.index_tasks_dir,
            self.backups_dir,
            self.conversations_dir,
            self.memories_dir,
            self.action_logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.root / "schema.json",
            {"version": SCHEMA_VERSION, "updated_at": datetime.now(UTC).isoformat()},
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RecordNotFoundError(path.name) from exc

    def create_project(self, request: ProjectCreate) -> Project:
        project = Project(
            name=request.name.strip(),
            description=request.description,
            source_root=request.source_root,
        )
        self._write_json(
            self.projects_dir / f"{project.id}.json",
            project.model_dump(mode="json"),
        )
        self.write_project_memory(
            project.id,
            f"# {project.name}\n\n"
            "## Rôle du projet\n"
            f"{project.description or 'Projet documentaire ClairDoc.'}\n\n"
            "## Dossier source\n"
            f"{project.source_root or 'Non défini'}\n\n"
            "## Documents\nAucun document indexé pour le moment.\n",
        )
        return project

    def get_project(self, project_id: UUID) -> Project:
        payload = self._read_json(self.projects_dir / f"{project_id}.json")
        return Project.model_validate(payload)

    def save_project(self, project: Project) -> None:
        project.updated_at = utc_now()
        self._write_json(
            self.projects_dir / f"{project.id}.json",
            project.model_dump(mode="json"),
        )

    def delete_project(self, project_id: UUID) -> None:
        self.get_project(project_id)
        for job in self.jobs_for_project(project_id):
            self.delete_job(job.id)
        for path in (
            self.projects_dir / f"{project_id}.json",
            self.indexes_dir / f"{project_id}.json",
            self.plans_dir / f"{project_id}.json",
            self.memory_path(project_id),
        ):
            path.unlink(missing_ok=True)
        shutil.rmtree(self.conversations_dir / str(project_id), ignore_errors=True)
        shutil.rmtree(self.action_logs_dir / str(project_id), ignore_errors=True)
        for task in self.iter_index_tasks():
            if task.project_id == project_id:
                (self.index_tasks_dir / f"{task.id}.json").unlink(missing_ok=True)

    def iter_projects(self) -> list[Project]:
        projects: list[Project] = []
        if not self.projects_dir.exists():
            return projects
        for record_path in self.projects_dir.glob("*.json"):
            try:
                projects.append(Project.model_validate(self._read_json(record_path)))
            except (ValueError, OSError):
                continue
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def create_job(
        self,
        original_filename: str,
        project_id: UUID | None,
        source_relative_path: str | None = None,
    ) -> OcrJob:
        job = OcrJob(
            original_filename=original_filename,
            project_id=project_id,
            source_relative_path=source_relative_path,
        )
        self.job_dir(job.id).mkdir(parents=True, exist_ok=False)
        self.save_job(job)
        return job

    def save_job(self, job: OcrJob) -> None:
        job.updated_at = utc_now()
        self._write_json(self.job_record_path(job.id), job.model_dump(mode="json"))

    def get_job(self, job_id: UUID) -> OcrJob:
        return OcrJob.model_validate(self._read_json(self.job_record_path(job_id)))

    def iter_jobs(self) -> list[OcrJob]:
        jobs: list[OcrJob] = []
        if not self.jobs_dir.exists():
            return jobs
        for record_path in self.jobs_dir.glob("*/job.json"):
            try:
                jobs.append(OcrJob.model_validate(self._read_json(record_path)))
            except (ValueError, OSError):
                continue
        return jobs

    def jobs_for_project(self, project_id: UUID) -> list[OcrJob]:
        return sorted(
            (job for job in self.iter_jobs() if job.project_id == project_id),
            key=lambda job: job.created_at,
        )

    def find_job_by_hash(
        self, project_id: UUID, content_sha256: str, exclude_job_id: UUID | None = None
    ) -> OcrJob | None:
        for job in self.iter_jobs():
            if (
                job.id != exclude_job_id
                and job.project_id == project_id
                and job.content_sha256 == content_sha256
                and job.status != JobStatus.FAILED
            ):
                return job
        return None

    def delete_job(self, job_id: UUID) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def job_dir(self, job_id: UUID) -> Path:
        return self.jobs_dir / str(job_id)

    def job_record_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "job.json"

    def input_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "input.pdf"

    def source_path(self, job_id: UUID) -> Path:
        job = self.get_job(job_id)
        suffix = Path(job.original_filename).suffix.lower() or ".bin"
        candidate = self.job_dir(job_id) / f"input{suffix}"
        legacy = self.input_path(job_id)
        return candidate if candidate.exists() or not legacy.exists() else legacy

    def output_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "output.pdf"

    def text_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "output.txt"

    def index_path(self, project_id: UUID) -> Path:
        return self.indexes_dir / f"{project_id}.json"

    def read_index(self, project_id: UUID) -> dict[str, object]:
        return self._read_json(self.index_path(project_id))

    def write_index(self, project_id: UUID, payload: dict[str, object]) -> None:
        self._write_json(self.index_path(project_id), payload)

    def rag_prompt_path(self) -> Path:
        return self.prompts_dir / "rag-system.txt"

    def memory_path(self, project_id: UUID) -> Path:
        return self.memories_dir / f"{project_id}.md"

    def read_project_memory(self, project_id: UUID) -> ProjectMemory:
        self.get_project(project_id)
        path = self.memory_path(project_id)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        return ProjectMemory(
            project_id=project_id,
            content=content,
            updated_at=datetime.fromtimestamp(path.stat().st_mtime, UTC)
            if path.exists()
            else utc_now(),
        )

    def write_project_memory(self, project_id: UUID, content: str) -> ProjectMemory:
        path = self.memory_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(content.strip() + "\n", encoding="utf-8")
        temporary.replace(path)
        return self.read_project_memory(project_id)

    def conversation_path(self, project_id: UUID, conversation_id: UUID) -> Path:
        return self.conversations_dir / str(project_id) / f"{conversation_id}.json"

    def create_conversation(self, project_id: UUID, title: str | None = None) -> Conversation:
        self.get_project(project_id)
        conversation = Conversation(
            project_id=project_id,
            title=(title or "Nouvelle conversation").strip() or "Nouvelle conversation",
        )
        self.save_conversation(conversation)
        return conversation

    def save_conversation(self, conversation: Conversation) -> None:
        conversation.updated_at = utc_now()
        self._write_json(
            self.conversation_path(conversation.project_id, conversation.id),
            conversation.model_dump(mode="json"),
        )

    def get_conversation(self, project_id: UUID, conversation_id: UUID) -> Conversation:
        conversation = Conversation.model_validate(
            self._read_json(self.conversation_path(project_id, conversation_id))
        )
        if conversation.project_id != project_id:
            raise RecordNotFoundError(str(conversation_id))
        return conversation

    def conversations_for_project(self, project_id: UUID) -> list[Conversation]:
        self.get_project(project_id)
        directory = self.conversations_dir / str(project_id)
        conversations: list[Conversation] = []
        for path in directory.glob("*.json"):
            try:
                conversations.append(Conversation.model_validate(self._read_json(path)))
            except (ValueError, OSError):
                continue
        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)

    def delete_conversation(self, project_id: UUID, conversation_id: UUID) -> None:
        self.get_conversation(project_id, conversation_id)
        self.conversation_path(project_id, conversation_id).unlink()

    def log_action(self, project_id: UUID, payload: dict[str, object]) -> None:
        directory = self.action_logs_dir / str(project_id)
        stamp = datetime.now(UTC)
        payload = {"created_at": stamp.isoformat(), **payload}
        self._write_json(directory / f"{stamp.strftime('%Y%m%d-%H%M%S-%f')}.json", payload)

    def plan_path(self, project_id: UUID) -> Path:
        return self.plans_dir / f"{project_id}.json"

    def write_plan(self, project_id: UUID, payload: dict[str, object]) -> None:
        self._write_json(self.plan_path(project_id), payload)

    def create_index_task(self, project_id: UUID) -> IndexTask:
        task = IndexTask(project_id=project_id)
        self.save_index_task(task)
        return task

    def save_index_task(self, task: IndexTask) -> None:
        task.updated_at = utc_now()
        self._write_json(
            self.index_tasks_dir / f"{task.id}.json",
            task.model_dump(mode="json"),
        )

    def get_index_task(self, task_id: UUID) -> IndexTask:
        return IndexTask.model_validate(self._read_json(self.index_tasks_dir / f"{task_id}.json"))

    def iter_index_tasks(self) -> list[IndexTask]:
        tasks: list[IndexTask] = []
        for path in self.index_tasks_dir.glob("*.json"):
            try:
                tasks.append(IndexTask.model_validate(self._read_json(path)))
            except (ValueError, OSError):
                continue
        return tasks

    def create_metadata_backup(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        destination = self.backups_dir / f"clairdoc-metadata-{stamp}.zip"
        included_directories = (
            self.projects_dir,
            self.indexes_dir,
            self.prompts_dir,
            self.plans_dir,
            self.index_tasks_dir,
            self.conversations_dir,
            self.memories_dir,
            self.action_logs_dir,
        )
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            schema_path = self.root / "schema.json"
            if schema_path.is_file():
                archive.write(schema_path, schema_path.relative_to(self.root))
            if self.runtime_path.is_file():
                archive.write(self.runtime_path, self.runtime_path.relative_to(self.root))
            for directory in included_directories:
                for path in directory.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(self.root))
            for job in self.iter_jobs():
                record = self.job_record_path(job.id)
                text = self.text_path(job.id)
                if record.is_file():
                    archive.write(record, record.relative_to(self.root))
                if text.is_file():
                    archive.write(text, text.relative_to(self.root))
        return destination

    def get_llm_model(self, default: str) -> str:
        try:
            value = self._read_json(self.runtime_path).get("llm_model")
        except RecordNotFoundError:
            return default
        return str(value) if value else default

    def set_llm_model(self, model: str) -> None:
        self._write_json(
            self.runtime_path,
            {"llm_model": model, "updated_at": datetime.now(UTC).isoformat()},
        )
