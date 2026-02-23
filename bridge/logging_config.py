"""Centralized logging configuration for the xAI bridge.

Configurable via environment variables:
  LOG_LEVEL     - Root log level (default: INFO). Options: DEBUG, INFO, WARNING, ERROR.
  DUMP_REQUESTS - Write full request/response JSON to files (default: false).
  DUMP_DIR      - Directory for request dumps (default: ./dumps).

Logger hierarchy:
  bridge.main       - Application lifecycle and HTTP layer
  bridge.forward    - Anthropic-to-OpenAI translation
  bridge.reverse    - OpenAI-to-Anthropic translation
  bridge.enrichment - Tool enrichment pipeline
  bridge.tokens     - Per-request token usage and enrichment overhead
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_configured = False


def configure_logging() -> None:
    """Set up structured logging from environment variables.

    Safe to call multiple times; only configures once.
    """
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format=_LOG_FORMAT, level=level, force=True)

    # Set bridge loggers to the configured level
    for name in ("bridge.main", "bridge.forward", "bridge.reverse", "bridge.enrichment", "bridge.tokens"):
        logging.getLogger(name).setLevel(level)

    # Quiet noisy third-party loggers at INFO
    for lib in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(lib).setLevel(max(level, logging.WARNING))

    _configured = True


def get_logger(module: str) -> logging.Logger:
    """Return a logger in the bridge hierarchy.

    Args:
        module: One of 'main', 'forward', 'reverse', 'enrichment'.
    """
    return logging.getLogger(f"bridge.{module}")


def is_dump_enabled() -> bool:
    """Check if request/response dumping is enabled."""
    return os.getenv("DUMP_REQUESTS", "false").lower() in ("true", "1", "yes")


def get_dump_dir() -> Path:
    """Return the dump directory path, creating it if needed."""
    dump_dir = Path(os.getenv("DUMP_DIR", "./dumps"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def dump_json(label: str, data: dict[str, Any]) -> Path | None:
    """Write a JSON payload to the dump directory.

    Args:
        label: Prefix for the filename (e.g., 'request', 'response').
        data: The payload to dump.

    Returns:
        Path to the written file, or None if dumping is disabled.
    """
    if not is_dump_enabled():
        return None

    dump_dir = get_dump_dir()
    ts = int(time.time() * 1000)
    path = dump_dir / f"{label}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def sanitize_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields from a request payload for logging.

    Strips Authorization headers and API keys. Returns a shallow copy
    with sensitive values replaced by '[REDACTED]'.
    """
    sanitized = dict(payload)

    # Strip top-level keys that might contain secrets
    for key in ("api_key", "authorization", "Authorization"):
        if key in sanitized:
            sanitized[key] = "[REDACTED]"

    # Strip from nested headers dict
    headers = sanitized.get("headers")
    if isinstance(headers, dict):
        sanitized["headers"] = {
            k: "[REDACTED]" if k.lower() in ("authorization", "x-api-key") else v
            for k, v in headers.items()
        }

    return sanitized
