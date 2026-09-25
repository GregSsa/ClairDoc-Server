from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    ocr_paused: bool = False


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OcrJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID | None = None
    original_filename: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_bytes: int = 0
    content_sha256: str | None = None
    error: str | None = None


class ProjectOcrState(BaseModel):
    project_id: UUID
    paused: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    storage_ready: bool
    ocr_available: bool
    authentication_configured: bool
    openai_configured: bool


class ConnectionResponse(BaseModel):
    status: str
    version: str


class IndexResponse(BaseModel):
    project_id: UUID
    documents_indexed: int
    documents_reused: int
    chunks_indexed: int
    embedding_model: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    document_name: str
    job_id: UUID
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
