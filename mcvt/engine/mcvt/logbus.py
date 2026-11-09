"""Plain-English system log (NFR-U-03): file + database + live UI log panel."""

from __future__ import annotations

import logging
import logging.handlers
import threading
from pathlib import Path
from typing import Callable

_listeners: list[Callable[[str, str, str], None]] = []
_db = None
_lock = threading.Lock()


def setup(log_dir: Path, db=None) -> None:
    """Configure root logging to console + rotating file, and remember the DB."""
    global _db
    _db = db
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(log_dir / "engine.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    for noisy in ("ultralytics", "websocket", "urllib3", "deep_sort_realtime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_listener(fn: Callable[[str, str, str], None]) -> None:
    with _lock:
        _listeners.append(fn)


def event(level: str, source: str, message: str) -> None:
    """Record a user-facing message (shown in the UI log panel)."""
    logging.getLogger(source).log(getattr(logging, level.upper(), logging.INFO), message)
    if _db is not None:
        try:
            _db.log(level.lower(), source, message)
        except Exception:  # never let logging crash the pipeline
            logging.getLogger("logbus").exception("could not write log to database")
    with _lock:
        listeners = list(_listeners)
    for fn in listeners:
        try:
            fn(level.lower(), source, message)
        except Exception:
            pass


def info(source: str, message: str) -> None:
    event("info", source, message)


def warning(source: str, message: str) -> None:
    event("warning", source, message)


def error(source: str, message: str) -> None:
    event("error", source, message)
