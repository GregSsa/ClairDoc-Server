from functools import cached_property
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLAIRDOC_",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: str | None = None
    data_dir: Path = Path("./data")
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    ocr_command: str = "ocrmypdf"
    ocr_languages: str = "fra+eng"
    ocr_workers: int = Field(default=1, ge=1, le=8)
    ocr_timeout_seconds: int = Field(default=7200, ge=60)
    max_upload_mb: int = Field(default=200, ge=1, le=4096)
    cors_origins: str = "tauri://localhost,http://tauri.localhost"
    log_level: str = "INFO"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "CLAIRDOC_OPENAI_API_KEY"),
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=512, ge=256, le=3072)
    llm_model: str = "gpt-6-luna"
    rag_top_k: int = Field(default=6, ge=1, le=20)
    rag_chunk_chars: int = Field(default=2400, ge=500, le=12000)
    rag_chunk_overlap: int = Field(default=300, ge=0, le=2000)

    @cached_property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @cached_property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
