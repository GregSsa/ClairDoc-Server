import json
from pathlib import Path
from uuid import UUID

from .models import OcrJob, Project, ProjectCreate, utc_now


class RecordNotFoundError(FileNotFoundError):
    pass


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.projects_dir = self.root / "projects"
        self.jobs_dir = self.root / "jobs"
        self.logs_dir = self.root / "logs"

    def initialize(self) -> None:
        for directory in (self.projects_dir, self.jobs_dir, self.logs_dir):
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

    def create_job(self, original_filename: str, project_id: UUID | None) -> OcrJob:
        job = OcrJob(original_filename=original_filename, project_id=project_id)
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

    def job_dir(self, job_id: UUID) -> Path:
        return self.jobs_dir / str(job_id)

    def job_record_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "job.json"

    def input_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "input.pdf"

    def output_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "output.pdf"

    def text_path(self, job_id: UUID) -> Path:
        return self.job_dir(job_id) / "output.txt"

