"""Tests for bridge observability — the four observation points.

Verifies that INFO and DEBUG log messages are emitted at each
pipeline stage: request intake, enrichment, outgoing payload,
and response summary.

As of issue #51, the default API path is the Responses API.
Mock responses use Responses API format.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """TestClient for the bridge app."""
    return TestClient(app)


def _mock_xai_response(data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data

    async def mock_post(*args, **kwargs):
        return mock_resp

    return mock_post, mock_resp


_SIMPLE_XAI_RESPONSE = {
    "id": "resp_log1",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Hi there!"}],
        }
    ],
    "model": "grok-4-1-fast-reasoning",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class TestPoint2RequestLogging:
    """Point 2: Outgoing request summary at INFO level."""

    def test_info_logs_request_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.INFO, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                })
        assert any("POST /v1/messages" in r.message for r in caplog.records)
        assert any("model=" in r.message for r in caplog.records)
        assert any("messages=1" in r.message for r in caplog.records)

    def test_debug_logs_translated_payload(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.DEBUG, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                })
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Translated" in m for m in debug_msgs)

    def test_request_log_includes_tool_count(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.INFO, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tools": [
                        {"name": "Read", "description": "Read.", "input_schema": {"type": "object"}},
                        {"name": "Write", "description": "Write.", "input_schema": {"type": "object"}},
                    ],
                })
        assert any("tools=2" in r.message for r in caplog.records)


class TestPoint3ResponseLogging:
    """Point 3: Grok response summary at INFO level."""

    def test_info_logs_response_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.INFO, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                })
        # Responses handler logs "xAI Responses in ..."
        resp_logs = [r.message for r in caplog.records if "xAI Responses" in r.message]
        assert len(resp_logs) >= 1
        msg = resp_logs[0]
        assert "status=200" in msg

    def test_info_logs_output_types(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        tool_response = {
            "id": "resp_tools1",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": "{}",
                },
            ],
            "model": "grok-4-1-fast-reasoning",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        mock_post, _ = _mock_xai_response(tool_response)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.INFO, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Read file"}],
                })
        resp_logs = [r.message for r in caplog.records if "xAI Responses" in r.message]
        assert any("function_call" in m for m in resp_logs)

    def test_debug_logs_full_response_body(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.DEBUG, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                })
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("xAI Responses body" in m for m in debug_msgs)


class TestPoint1EnrichmentLogging:
    """Point 1: Enriched tool definitions logging."""

    def test_info_logs_tool_names_and_count(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="structural")
        tools = [
            {"name": "Read", "description": "Reads.", "input_schema": {"type": "object"}},
            {"name": "Write", "description": "Writes.", "input_schema": {"type": "object"}},
        ]
        with caplog.at_level(logging.INFO, logger="bridge.enrichment"):
            enricher.enrich(tools)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Enriching 2 tools" in m for m in info_msgs)
        assert any("Read" in m and "Write" in m for m in info_msgs)

    def test_debug_logs_applied_patterns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="structural")
        tools = [{"name": "Read", "description": "Reads.", "input_schema": {"type": "object"}}]
        with caplog.at_level(logging.DEBUG, logger="bridge.enrichment"):
            enricher.enrich(tools)

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Enrichment complete" in m for m in debug_msgs)
        assert any("structural_patterns=" in m for m in debug_msgs)

    def test_debug_logs_field_diff(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="full")
        tools = [{"name": "Read", "description": "Reads.", "input_schema": {"type": "object"}}]
        with caplog.at_level(logging.DEBUG, logger="bridge.enrichment"):
            enricher.enrich(tools)

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("fields added" in m for m in debug_msgs)

    def test_passthrough_skips_enrichment_logging(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="passthrough")
        tools = [{"name": "Read", "description": "Reads.", "input_schema": {"type": "object"}}]
        with caplog.at_level(logging.DEBUG, logger="bridge.enrichment"):
            enricher.enrich(tools)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any("Enriching" in m for m in info_msgs)


class TestNoApiKeyLeaks:
    """Ensure API keys are never logged."""

    def test_translated_request_has_no_bearer_token(
        self, client: TestClient, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_post, _ = _mock_xai_response(_SIMPLE_XAI_RESPONSE)
        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            with caplog.at_level(logging.DEBUG, logger="bridge.main"):
                client.post("/v1/messages", json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                })
        all_output = " ".join(r.message for r in caplog.records)
        assert "Bearer" not in all_output
        assert "XAI_API_KEY" not in all_output
