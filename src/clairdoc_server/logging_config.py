import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_directory: Path, level: str) -> None:
    log_directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = RotatingFileHandler(
        log_directory / "clairdoc-server.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    if not any(isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)

