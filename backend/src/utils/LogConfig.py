import logging
import os
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _parse_level(level: Optional[str]) -> int:
    if not level:
        return logging.INFO
    value = str(level).strip().upper()
    return getattr(logging, value, logging.INFO)


def configure_logging(level: Optional[str] = None) -> None:
    """Configure backend logging once.

    - Logs go to stderr to avoid interfering with real-time stdout updates.
    - Safe to call multiple times.
    """

    root = logging.getLogger()
    if getattr(root, "_rve_configured", False):
        return

    env_level = os.environ.get("RVE_BACKEND_LOG_LEVEL")
    root.setLevel(_parse_level(level or env_level))

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)

    root._rve_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
