from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import router
from .config import Settings
from .jobs import OcrJobManager
from .logging_config import configure_logging
from .organization import OrganizationService
from .rag import RagService
from .storage import LocalStorage


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    storage = LocalStorage(resolved_settings.data_dir)
    jobs = OcrJobManager(storage, resolved_settings)
    rag = RagService(storage, resolved_settings)
    organization = OrganizationService(storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage.initialize()
        configure_logging(storage.logs_dir, resolved_settings.log_level)
        app.state.settings = resolved_settings
        app.state.storage = storage
        app.state.jobs = jobs
        app.state.rag = rag
        app.state.organization = organization
        await jobs.start()
        try:
            yield
        finally:
            await jobs.stop()

    app = FastAPI(
        title="ClairDoc Server",
        summary="OCR local et services documentaires de ClairDoc",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-ClairDoc-Key"],
    )
    app.include_router(router)
    return app


app = create_app()
