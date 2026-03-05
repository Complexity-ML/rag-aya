"""
RAG-Aya :: Logger

Structured logging with colors and timestamps, inspired by vLLM logging style.
"""

import logging
import os
import sys
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Colored log output with vLLM-style formatting."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        record.name = f"\033[35m{record.name}\033[0m"  # magenta
        return super().format(record)


def init_logger(name: str = "rag-aya", level: Optional[str] = None) -> logging.Logger:
    """
    Initialize a logger with colored output.

    Usage:
        from logger import init_logger
        logger = init_logger(__name__)
        logger.info("Indexing %d chunks", len(chunks))
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    env_level = os.environ.get("RAG_AYA_LOG_LEVEL")
    log_level = getattr(logging, (level or env_level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    fmt = ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
