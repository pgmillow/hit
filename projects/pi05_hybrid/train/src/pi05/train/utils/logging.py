"""Logging setup that plays well with Accelerate main-process output."""

from __future__ import annotations

import logging
from typing import Any


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)


def log_run_summary(logger: logging.Logger, summary: dict[str, Any]) -> None:
    logger.info("Run summary:")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)
