"""Tests for bridge logging configuration.

Verifies log level control, logger hierarchy, dump functionality,
and request sanitization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bridge.logging_config import (
    configure_logging,
    dump_json,
    get_dump_dir,
    get_logger,
    is_dump_enabled,
    sanitize_request,
)


class TestConfigureLogging:
    """Verify logging setup from environment variables."""

    def setup_method(self) -> None:
        """Reset the configured flag before each test."""
        import bridge.logging_config
        bridge.logging_config._configured = False

    def test_default_level_is_info(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOG_LEVEL", None)
            configure_logging()
            logger = get_logger("main")
            assert logger.level == logging.INFO

    def test_debug_level_from_env(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            configure_logging()
            logger = get_logger("main")
            assert logger.level == logging.DEBUG

    def test_warning_level_from_env(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            configure_logging()
            logger = get_logger("main")
            assert logger.level == logging.WARNING

    def test_case_insensitive_level(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            configure_logging()
            logger = get_logger("main")
            assert logger.level == logging.DEBUG

    def test_all_bridge_loggers_configured(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            configure_logging()
            for name in ("main", "forward", "reverse", "enrichment"):
                assert get_logger(name).level == logging.DEBUG

    def test_idempotent_configuration(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            configure_logging()
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            configure_logging()  # Should not reconfigure
            logger = get_logger("main")
            assert logger.level == logging.DEBUG  # Still DEBUG from first call


class TestGetLogger:
    """Verify logger naming hierarchy."""

    def test_logger_name_prefix(self) -> None:
        logger = get_logger("main")
        assert logger.name == "bridge.main"

    def test_forward_logger(self) -> None:
        assert get_logger("forward").name == "bridge.forward"

    def test_reverse_logger(self) -> None:
        assert get_logger("reverse").name == "bridge.reverse"

    def test_enrichment_logger(self) -> None:
        assert get_logger("enrichment").name == "bridge.enrichment"


class TestDumpRequests:
    """Verify request/response dump functionality."""

    def test_dump_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DUMP_REQUESTS", None)
            assert is_dump_enabled() is False

    def test_dump_enabled_true(self) -> None:
        with patch.dict(os.environ, {"DUMP_REQUESTS": "true"}):
            assert is_dump_enabled() is True

    def test_dump_enabled_one(self) -> None:
        with patch.dict(os.environ, {"DUMP_REQUESTS": "1"}):
            assert is_dump_enabled() is True

    def test_dump_enabled_yes(self) -> None:
        with patch.dict(os.environ, {"DUMP_REQUESTS": "yes"}):
            assert is_dump_enabled() is True

    def test_dump_disabled_false(self) -> None:
        with patch.dict(os.environ, {"DUMP_REQUESTS": "false"}):
            assert is_dump_enabled() is False

    def test_dump_dir_default(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"DUMP_DIR": str(tmp_path / "dumps")}):
            dump_dir = get_dump_dir()
            assert dump_dir.exists()
            assert dump_dir.name == "dumps"

    def test_dump_json_writes_file(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {
            "DUMP_REQUESTS": "true",
            "DUMP_DIR": str(tmp_path),
        }):
            data = {"model": "grok-4", "messages": []}
            path = dump_json("request", data)
            assert path is not None
            assert path.exists()
            assert path.name.startswith("request_")
            assert path.suffix == ".json"
            written = json.loads(path.read_text())
            assert written["model"] == "grok-4"

    def test_dump_json_returns_none_when_disabled(self) -> None:
        with patch.dict(os.environ, {"DUMP_REQUESTS": "false"}):
            result = dump_json("request", {"data": "test"})
            assert result is None


class TestSanitizeRequest:
    """Verify sensitive field removal from request payloads."""

    def test_strips_authorization_header(self) -> None:
        payload = {
            "headers": {"Authorization": "Bearer sk-secret123", "Content-Type": "application/json"},
            "model": "grok-4",
        }
        sanitized = sanitize_request(payload)
        assert sanitized["headers"]["Authorization"] == "[REDACTED]"
        assert sanitized["headers"]["Content-Type"] == "application/json"

    def test_strips_api_key_header(self) -> None:
        payload = {"headers": {"X-Api-Key": "secret456"}}
        sanitized = sanitize_request(payload)
        assert sanitized["headers"]["X-Api-Key"] == "[REDACTED]"

    def test_strips_top_level_api_key(self) -> None:
        payload = {"api_key": "sk-12345", "model": "grok-4"}
        sanitized = sanitize_request(payload)
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["model"] == "grok-4"

    def test_preserves_non_sensitive_fields(self) -> None:
        payload = {"model": "grok-4", "messages": [{"role": "user", "content": "hi"}]}
        sanitized = sanitize_request(payload)
        assert sanitized == payload

    def test_does_not_mutate_original(self) -> None:
        payload = {"api_key": "secret", "model": "grok-4"}
        sanitize_request(payload)
        assert payload["api_key"] == "secret"
