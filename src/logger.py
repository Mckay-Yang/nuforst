from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOG_DIR = Path("log")

_root_logger: Optional[logging.Logger] = None
_listener: Optional[logging.handlers.QueueListener] = None
_handler_registered = False


def _ensure_log_dir() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def setup() -> logging.Logger:
    global _root_logger, _listener, _handler_registered

    if _root_logger is not None:
        return _root_logger

    log_dir = _ensure_log_dir()
    log_file = log_dir / f"run_{_timestamp_for_filename()}.log"

    logger = logging.getLogger("nufrost")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(q)
    logger.addHandler(queue_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(_build_formatter())
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(_build_formatter())
    console_handler.setLevel(logging.INFO)

    _listener = logging.handlers.QueueListener(
        q,
        file_handler,
        console_handler,
        respect_handler_level=True,
    )
    _listener.start()

    _root_logger = logger

    if not _handler_registered:
        atexit.register(shutdown)
        _handler_registered = True

    logger.info("Logger initialized, log file: %s", log_file)
    return logger


def get(name: str) -> logging.Logger:
    if _root_logger is None:
        setup()
    return _root_logger.getChild(name)


def shutdown() -> None:
    global _root_logger, _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
    _root_logger = None


def log(func_name: str, message: str, level: int = logging.INFO) -> None:
    if _root_logger is None:
        setup()
    record = _root_logger.makeRecord(
        name=_root_logger.name,
        level=level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.funcName = func_name
    _root_logger.handle(record)
