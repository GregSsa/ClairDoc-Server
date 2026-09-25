from functools import cached_property
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLAIRDOC_",
        extra="ignore",
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

    @cached_property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @cached_property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

