import json
import shutil
from pathlib import Path
from uuid import UUID

from .models import JobStatus, OcrJob, Project, ProjectCreate, utc_now


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

    def initialize(self) -> None:
        for directory in (
            self.projects_dir,
            self.jobs_dir,
            self.logs_dir,
            self.indexes_dir,
            self.prompts_dir,
            self.plans_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

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
        project = Project(name=request.name.strip(), description=request.description)
        self._write_json(
            self.projects_dir / f"{project.id}.json",
            project.model_dump(mode="json"),
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

    def plan_path(self, project_id: UUID) -> Path:
        return self.plans_dir / f"{project_id}.json"

    def write_plan(self, project_id: UUID, payload: dict[str, object]) -> None:
        self._write_json(self.plan_path(project_id), payload)
