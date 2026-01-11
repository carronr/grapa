"""Logging utilities for Grapa.

Provides best-effort logging setup with a package-local log file, falling back
to a user-writable cache directory or stdout when file logging is unavailable.
"""

import os
import sys
import logging

logger = logging.getLogger("grapa")


_LOGGING_KWARGS = {
    "level": logging.WARNING,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%m/%d/%Y %I:%M:%S %p",
}


def _resolve_log_path():
    """Return the preferred log file path within the package directory."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, "log_grapa.log")


def _resolve_log_path_fallback():
    """Return a user-writable log file path in the OS cache directory."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        folder = os.path.join(base, "grapa")
    else:
        base = os.getenv("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        folder = os.path.join(base, "grapa")
    return os.path.join(folder, "log_grapa.log")


def setup_logging():
    """Configure Grapa logging with a file handler and stdout fallback.

    Attempts to log to the package directory first, then a user cache path,
    and finally stdout. Adds a StreamHandler to the 'grapa' logger if missing.

    Returns:
        logging.Handler | None: The StreamHandler for the package logger, or None if
        a handler could not be added.
    """
    logfile = _resolve_log_path()
    try:
        logging.basicConfig(filename=logfile, **_LOGGING_KWARGS)
    except OSError:
        fallback = _resolve_log_path_fallback()
        try:
            os.makedirs(os.path.dirname(fallback), exist_ok=True)
            logging.basicConfig(filename=fallback, **_LOGGING_KWARGS)
        except OSError:
            logging.basicConfig(stream=sys.stdout, **_LOGGING_KWARGS)

    existing_handler = next(
        (h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None
    )
    if existing_handler is not None:
        return existing_handler

    logger_handler = logging.StreamHandler(sys.stdout)
    logger_handler.setLevel(_LOGGING_KWARGS["level"])
    logger_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(logger_handler)
    return logger_handler
