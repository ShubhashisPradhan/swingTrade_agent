"""Logging configuration.Centralized logger configuration using Loguru."""

from loguru import logger
import os

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    f"{LOG_DIR}/run.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

def get_logger():
    """Return configured logger."""
    return logger
