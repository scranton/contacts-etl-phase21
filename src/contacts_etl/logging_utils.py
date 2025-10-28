from __future__ import annotations

import logging
import os
from typing import Optional

from .config_loader import PipelineConfig


def _resolve_level(level_name: str) -> int:
    """
    Convert a case-insensitive logging level string to its numeric value.

    Falls back to logging.INFO when the provided string is not a valid level.
    """
    normalized = (level_name or "INFO").upper()
    if normalized.isdigit():
        return int(normalized)
    return getattr(logging, normalized, logging.INFO)


def configure_logging(config: PipelineConfig, level_override: Optional[str] = None) -> None:
    """
    Configure the root logger according to precedence:

    1. ``CONTACTS_ETL_LOG_LEVEL`` environment variable (if set)
    2. ``level_override`` provided by the caller (e.g., CLI flag)
    3. ``config.logging.level`` from ``config.yaml``
    4. Default ``WARNING`` level
    """
    env_level = os.getenv("CONTACTS_ETL_LOG_LEVEL")
    effective_level_name = env_level or level_override or config.logging.level or "WARNING"
    level_value = _resolve_level(effective_level_name)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level_value)
    else:
        logging.basicConfig(level=level_value)
