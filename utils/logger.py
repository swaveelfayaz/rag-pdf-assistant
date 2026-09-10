"""
utils/logger.py
---------------
Shared logger factory for the entire RAG pipeline.

WHY A FACTORY PATTERN?
  Python's `logging` module uses a *global registry* of loggers keyed
  by name. If every module called `logging.basicConfig()` independently,
  the first one to run would win — subsequent calls are silently ignored.
  The factory pattern solves this by:
    1. Configuring handlers exactly once (via a module-level flag).
    2. Returning a correctly named logger for each module:
         log = get_logger(__name__)  →  "ingestion.loader", etc.
  This gives you hierarchical filtering: setting level on "ingestion"
  silences both "ingestion.loader" and "ingestion.chunker" at once.

USAGE:
  from utils.logger import get_logger
  log = get_logger(__name__)
  log.info("Processing %d chunks", len(chunks))
"""

import logging
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

# The log file lives at the project root so it's easy to find.
# We compute the path relative to *this file* so the logger works
# regardless of what directory the user runs the app from.
_LOG_FILE = Path(__file__).resolve().parent.parent / "app.log"

# Single format string used by every handler so log lines are uniform.
# Fields: timestamp | level | module name | message
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guard flag so we only attach handlers once, even if get_logger is
# called many times (e.g., during pytest where modules are re-imported).
_configured: bool = False


def _configure_root_logger(level: int) -> None:
    """
    Attach console + file handlers to the root logger exactly once.

    This is intentionally private — external code should call
    get_logger(), not this function directly.

    Args:
        level: A logging level constant (e.g., logging.DEBUG).
    """
    global _configured
    if _configured:
        return  # idempotent — safe to call multiple times

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — writes INFO+ to stdout so the Streamlit terminal
    # shows progress without being flooded by DEBUG messages.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — writes ALL levels so you can diagnose DEBUG issues
    # after the fact without restarting with a different level.
    try:
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    except OSError:
        # If we can't write the log file (e.g., read-only filesystem on
        # Hugging Face Spaces), fall back to console-only gracefully.
        file_handler = None  # type: ignore[assignment]

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    if file_handler:
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger, configuring root handlers on first call.

    Convention: call this at module level with `__name__` so the logger
    name matches the Python module path.

        log = get_logger(__name__)   # e.g. "ingestion.loader"

    Args:
        name:  Logger name (use __name__ of the calling module).
        level: Minimum level for the root logger. Defaults to DEBUG so
               the file handler captures everything; the console handler
               independently filters to INFO.

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root_logger(level)
    return logging.getLogger(name)
