import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "clairdoc_server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

