"""End-to-end tests for the bridge application.

Tests bridge startup, endpoint responses, enrichment mode toggling,
error handling, and request translation — all without requiring a live
xAI API key.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """TestClient for the bridge app."""
    return TestClient(app)


def _mock_xai_response(data: dict, status_code: int = 200):
    """Create a mock httpx response that works with async client."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data

    async def mock_post(*args, **kwargs):
        return mock_resp

    return mock_post, mock_resp


class TestBridgeStartup:
    """Verify the bridge starts and serves endpoints."""

    def test_manifest_returns_200(self, client: TestClient) -> None:
        resp = client.get("/manifest")
        assert resp.status_code == 200

    def test_manifest_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/manifest").json()
        assert data["name"] == "Claude Code xAI Bridge"
        assert "messages" in data["capabilities"]
        assert "tools" in data["capabilities"]
        assert "streaming" in data["capabilities"]
        assert "enrichment_modes" in data
        assert set(data["enrichment_modes"]) == {"passthrough", "structural", "full"}

    def test_manifest_links(self, client: TestClient) -> None:
        data = client.get("/manifest").json()
        links = data["_links"]
        assert links["self"]["href"] == "/manifest"
        assert links["messages"]["href"] == "/v1/messages"
        assert links["health"]["href"] == "/health"

    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "enrichment_mode" in data

    def test_health_enrichment_mode_is_valid(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["enrichment_mode"] in ("passthrough", "structural", "full")


class TestErrorHandling:
    """Verify error responses follow Anthropic format."""

    def test_thinking_feature_stripped_not_rejected(self, client: TestClient) -> None:
        """Thinking param is gracefully stripped, not rejected with 400."""
        mock_post, _ = _mock_xai_response({
            "id": "chatcmpl-think1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            resp = client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["content"][0]["text"] == "Hello!"
        assert "X-Bridge-Warning" in resp.headers
        assert "thinking" in resp.headers["X-Bridge-Warning"].lower()

    def test_unsupported_image_content(self, client: TestClient) -> None:
        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "abc"}}
            ]}],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert "image" in data["error"]["message"].lower()


class TestRequestTranslation:
    """Verify requests are correctly translated before reaching xAI."""

    def test_simple_text_round_trip(self, client: TestClient) -> None:
        """Mock xAI and verify full Anthropic → OpenAI → Anthropic translation."""
        mock_post, mock_resp = _mock_xai_response({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            resp = client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
            })

            assert resp.status_code == 200
            data = resp.json()

            # Verify Anthropic response format
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) > 0
            assert data["content"][0]["type"] == "text"
            assert data["content"][0]["text"] == "Hello!"
            assert data["stop_reason"] == "end_turn"
            assert "usage" in data

    def test_tool_definitions_reach_xai_enriched(self, client: TestClient) -> None:
        """Verify tools are enriched and translated to OpenAI format."""
        captured_request = {}

        async def capture_post(*args, **kwargs):
            captured_request.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "I'll read that file."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
            return mock_resp

        import main
        with patch.object(main.client, "post", side_effect=capture_post):
            resp = client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Read the file"}],
                "tools": [{
                    "name": "Read",
                    "description": "Reads a file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                }],
            })

            assert resp.status_code == 200

            # Verify OpenAI format tools were sent
            assert captured_request["tools"] is not None
            assert len(captured_request["tools"]) == 1
            tool = captured_request["tools"][0]
            assert tool["type"] == "function"
            assert tool["function"]["name"] == "Read"

    def test_tool_use_response_translated(self, client: TestClient) -> None:
        """Verify tool_calls from Grok become Anthropic tool_use blocks."""
        mock_post, _ = _mock_xai_response({
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps({"file_path": "/tmp/test.py"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        })

        import main
        with patch.object(main.client, "post", side_effect=mock_post):
            resp = client.post("/v1/messages", json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Read /tmp/test.py"}],
                "tools": [{
                    "name": "Read",
                    "description": "Reads a file.",
                    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }],
            })

            assert resp.status_code == 200
            data = resp.json()

            # Verify Anthropic tool_use format
            assert data["type"] == "message"
            assert data["stop_reason"] == "tool_use"
            tool_block = next(b for b in data["content"] if b["type"] == "tool_use")
            assert tool_block["name"] == "Read"
            assert tool_block["id"] == "call_abc123"
            assert tool_block["input"]["file_path"] == "/tmp/test.py"


class TestEnrichmentModeToggle:
    """Verify enrichment modes produce different results."""

    def test_passthrough_adds_no_fields(self) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="passthrough")
        tools = [{"name": "Read", "description": "Reads a file.", "input_schema": {"type": "object"}}]
        result = enricher.enrich(tools)
        assert result == tools

    def test_structural_adds_structural_fields(self) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="structural")
        tools = [{"name": "Read", "description": "Reads a file.", "input_schema": {"type": "object"}}]
        result = enricher.enrich(tools)
        assert len(result[0]) > len(tools[0])
        assert "behavioral_what" not in result[0]

    def test_full_adds_all_fields(self) -> None:
        from enrichment.factory import create_enricher
        enricher = create_enricher(mode="full")
        tools = [{"name": "Read", "description": "Reads a file.", "input_schema": {"type": "object"}}]
        result = enricher.enrich(tools)
        assert "behavioral_what" in result[0]
        assert "behavioral_why" in result[0]
        assert "behavioral_when" in result[0]

    def test_preamble_disabled(self) -> None:
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "false"}):
            import importlib
            import enrichment.system_preamble
            importlib.reload(enrichment.system_preamble)
            from enrichment.system_preamble import get_system_preamble
            assert get_system_preamble() == ""

    def test_preamble_enabled(self) -> None:
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "true"}):
            import importlib
            import enrichment.system_preamble
            importlib.reload(enrichment.system_preamble)
            from enrichment.system_preamble import get_system_preamble
            assert len(get_system_preamble()) > 0
            assert "Tool Preference Hierarchy" in get_system_preamble()
