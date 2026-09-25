import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "clairdoc_server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        ssl_certfile=str(settings.tls_certfile) if settings.tls_certfile else None,
        ssl_keyfile=str(settings.tls_keyfile) if settings.tls_keyfile else None,
    )


if __name__ == "__main__":
    main()
