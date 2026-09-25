from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    source_root: str | None = Field(default=None, max_length=4096)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    source_root: str | None = Field(default=None, max_length=4096)


class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    source_root: str | None = None
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
    source_relative_path: str | None = None
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


class RuntimeInfo(BaseModel):
    version: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    ocr_languages: str
    data_dir: str
    max_upload_mb: int
    max_index_tokens: int
    openai_configured: bool
    tls_enabled: bool


class IndexResponse(BaseModel):
    project_id: UUID
    documents_indexed: int
    documents_reused: int
    chunks_indexed: int
    embedding_model: str


class IndexEstimate(BaseModel):
    project_id: UUID
    documents_total: int
    documents_to_embed: int
    documents_reused: int
    estimated_tokens: int
    estimated_cost_usd: float
    price_per_million_tokens_usd: float
    embedding_model: str


class IndexTask(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimate: IndexEstimate | None = None
    result: IndexResponse | None = None
    error: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    document_name: str
    job_id: UUID
    chunk_index: int
    page_number: int | None = None
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str


class DocumentSummary(BaseModel):
    job_id: UUID
    name: str
    source_relative_path: str
    status: str
    category: str
    document_date: str | None = None
    organization: str | None = None
    people: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    chunks: int = 0


class DocumentRelationship(BaseModel):
    source_job_id: UUID
    target_job_id: UUID
    kind: str
    label: str


class DocumentLibrary(BaseModel):
    project_id: UUID
    documents: list[DocumentSummary]
    categories: list[str]
    relationships: list[DocumentRelationship]


class OrganizationEntry(BaseModel):
    job_id: UUID
    original_filename: str
    source_relative_path: str
    suggested_path: str
    category: str
    document_date: str | None = None
    organization: str | None = None
    reason: str


class OrganizationPlan(BaseModel):
    project_id: UUID
    created_at: datetime = Field(default_factory=utc_now)
    entries: list[OrganizationEntry]


class BackupResponse(BaseModel):
    path: str
    size_bytes: int
    created_at: datetime
