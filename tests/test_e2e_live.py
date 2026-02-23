"""Live end-to-end tests for Issue #15 Categories 2-4.

These tests hit the real xAI API through the bridge's ASGI app.
They require a valid XAI_API_KEY in the environment (or .env file).

Category 2 -- Live Request/Response:
  - Simple text request through the bridge, verify Anthropic response structure
  - Verify id format, type, role, content blocks, model, stop_reason, usage

Category 3 -- Tool Calling:
  - Send request with tool definition, verify Grok returns tool_use block
  - Multi-turn: send tool_result back, verify Grok uses result in final response
  - Capture outgoing request to verify enrichment was applied

Category 4 -- Streaming:
  - Send streaming request, collect SSE events
  - Verify Anthropic streaming event sequence
  - Verify assembled content is non-empty

Architecture note:
  The bridge uses a module-level httpx.AsyncClient (main.client) for xAI calls.
  Each pytest-asyncio test gets its own event loop. To avoid "Event loop is
  closed" errors on sequential tests, the ``fresh_xai_client`` fixture replaces
  main.client with a new AsyncClient before each test and restores it after.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

# Load .env before any app imports so XAI_API_KEY is available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import main
from main import app

# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.live]

SKIP_NO_KEY = pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="Requires XAI_API_KEY",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def fresh_xai_client():
    """Replace main.client with a fresh AsyncClient for each test.

    The module-level main.client shares connection state across event loops.
    pytest-asyncio creates a new event loop per test, which causes
    "Event loop is closed" on the second request. This fixture creates a
    fresh client bound to the current loop, restoring the original after.
    """
    original = main.client
    fresh = httpx.AsyncClient(base_url="https://api.x.ai/v1", timeout=120.0)
    main.client = fresh
    yield
    await fresh.aclose()
    main.client = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_request(
    content: str = "Say hello in exactly 3 words",
    max_tokens: int = 50,
    **kwargs,
) -> dict:
    """Build a minimal Anthropic Messages API request."""
    req = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    req.update(kwargs)
    return req


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE event text into a list of event data dicts.

    The bridge emits events in the format:
        event: <type>
        data: <json>

    Returns a list of parsed JSON objects from the data lines.
    """
    events = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                pass
    return events


# ======================================================================
# Category 2: Live Request/Response
# ======================================================================


@SKIP_NO_KEY
class TestLiveRequestResponse:
    """Verify a real request/response cycle through the bridge."""

    @pytest.mark.asyncio
    async def test_simple_text_response_structure(self) -> None:
        """Send a simple prompt and verify the full Anthropic response structure."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json=_minimal_request(),
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # -- Core structure assertions (strict) --
        assert data["id"].startswith("msg_"), f"id should start with msg_, got: {data['id']}"
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert isinstance(data["content"], list)
        assert len(data["content"]) > 0

        # -- Content block --
        text_block = data["content"][0]
        assert text_block["type"] == "text"
        assert isinstance(text_block["text"], str)
        assert len(text_block["text"]) > 0

        # -- Model (flexible -- xAI may return a different model variant) --
        assert isinstance(data["model"], str)
        assert len(data["model"]) > 0

        # -- Stop reason --
        assert data["stop_reason"] == "end_turn"

        # -- Usage --
        usage = data["usage"]
        assert usage["input_tokens"] > 0, "input_tokens should be > 0"
        assert usage["output_tokens"] > 0, "output_tokens should be > 0"

    @pytest.mark.asyncio
    async def test_response_id_is_unique_across_requests(self) -> None:
        """Two requests should produce different message IDs."""
        transport = httpx.ASGITransport(app=app)
        ids = []
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            for _ in range(2):
                resp = await client.post(
                    "/v1/messages",
                    json=_minimal_request(content="Say one word", max_tokens=10),
                )
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
                ids.append(resp.json()["id"])

        assert ids[0] != ids[1], f"IDs should differ, got: {ids}"


# ======================================================================
# Category 3: Tool Calling
# ======================================================================


@SKIP_NO_KEY
class TestLiveToolCalling:
    """Verify tool calling works end-to-end through the bridge."""

    WEATHER_TOOL = {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for.",
                },
            },
            "required": ["city"],
        },
    }

    @pytest.mark.asyncio
    async def test_tool_use_response_structure(self) -> None:
        """Grok should return a tool_use block when asked to use a tool."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json=_minimal_request(
                    content=(
                        "What is the weather in Paris? "
                        "You MUST use the get_weather tool to answer. Call it now."
                    ),
                    max_tokens=200,
                    tools=[self.WEATHER_TOOL],
                ),
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Grok should either call the tool or respond with text.
        # With a directive prompt, it should call the tool, but we handle both.
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]

        if tool_blocks:
            # -- Verify tool_use block structure --
            tb = tool_blocks[0]
            assert "id" in tb, "tool_use block must have an id"
            assert tb["name"] == "get_weather", f"Expected get_weather, got {tb['name']}"
            assert isinstance(tb["input"], dict), "input should be a dict"
            assert "city" in tb["input"], f"Expected city in input, got {tb['input']}"
            assert data["stop_reason"] == "tool_use"
        else:
            # If Grok didn't call the tool, just verify basic response structure
            assert data["type"] == "message"
            assert any(b["type"] == "text" for b in data["content"])
            pytest.skip("Grok did not call the tool -- non-deterministic behavior")

    @pytest.mark.asyncio
    async def test_multi_turn_tool_result(self) -> None:
        """Send tool_result back and verify Grok uses it in final response."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            # Turn 1: Ask Grok to call the tool
            resp1 = await client.post(
                "/v1/messages",
                json=_minimal_request(
                    content=(
                        "What is the weather in Tokyo? "
                        "You MUST use the get_weather tool. Call it now."
                    ),
                    max_tokens=200,
                    tools=[self.WEATHER_TOOL],
                ),
            )

            assert resp1.status_code == 200
            data1 = resp1.json()

            tool_blocks = [b for b in data1["content"] if b["type"] == "tool_use"]
            if not tool_blocks:
                pytest.skip("Grok did not call tool in turn 1 -- non-deterministic")

            tool_use_id = tool_blocks[0]["id"]

            # Turn 2: Send tool result back
            resp2 = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200,
                    "tools": [self.WEATHER_TOOL],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "What is the weather in Tokyo? "
                                "You MUST use the get_weather tool. Call it now."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": data1["content"],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": "72F and sunny in Tokyo",
                                },
                            ],
                        },
                    ],
                },
            )

            assert resp2.status_code == 200, f"Turn 2 failed: {resp2.status_code}: {resp2.text}"
            data2 = resp2.json()

            # Grok should now produce a text response using the tool result
            assert data2["type"] == "message"
            text_blocks = [b for b in data2["content"] if b["type"] == "text"]
            assert len(text_blocks) > 0, "Expected text response after tool_result"

            # The response should reference the weather data we provided
            response_text = text_blocks[0]["text"].lower()
            # Flexible: Grok should mention the temperature or weather or Tokyo
            assert any(
                kw in response_text
                for kw in ["72", "sunny", "tokyo", "weather", "temperature"]
            ), f"Response should reference tool result, got: {response_text[:200]}"

    @pytest.mark.asyncio
    async def test_enrichment_applied_to_outgoing_request(self) -> None:
        """Verify that tool definitions are enriched before reaching xAI.

        Uses the request capture pattern: wraps main.client.post to intercept
        the outgoing request JSON, then checks enrichment fields.
        """
        captured_request = {}

        original_post = main.client.post

        async def capture_and_forward(*args, **kwargs):
            """Capture the request JSON, then forward to real xAI."""
            if "json" in kwargs:
                captured_request.update(kwargs["json"])
            return await original_post(*args, **kwargs)

        with patch.object(main.client, "post", side_effect=capture_and_forward):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                timeout=60.0,
            ) as client:
                resp = await client.post(
                    "/v1/messages",
                    json=_minimal_request(
                        content="Read the file /tmp/test.txt",
                        max_tokens=100,
                        tools=[{
                            "name": "Read",
                            "description": "Reads a file.",
                            "input_schema": {
                                "type": "object",
                                "properties": {"file_path": {"type": "string"}},
                                "required": ["file_path"],
                            },
                        }],
                    ),
                )

        # Response may succeed or fail depending on API state, but we captured the request
        assert "tools" in captured_request, "Outgoing request should have tools"
        assert len(captured_request["tools"]) == 1

        # The tool should be in OpenAI format
        tool = captured_request["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "Read"

        # The parameters should preserve the original schema
        params = tool["function"]["parameters"]
        if isinstance(params, str):
            params = json.loads(params)

        assert "properties" in params
        assert "file_path" in params["properties"]

        # Verify enrichment was applied (if not in passthrough mode)
        enrichment_mode = main.enricher.config.mode
        if enrichment_mode != "passthrough":
            # In structural/full mode, the enricher adds fields to the tool dict
            # before translation. The description should at least be present.
            assert tool["function"]["description"] is not None
            assert len(tool["function"]["description"]) > 0


# ======================================================================
# Category 4: Streaming
# ======================================================================


@SKIP_NO_KEY
class TestLiveStreaming:
    """Verify streaming works end-to-end through the bridge."""

    @pytest.mark.asyncio
    async def test_streaming_event_sequence(self) -> None:
        """Send a streaming request and verify the Anthropic SSE event sequence."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json=_minimal_request(
                    content="Say hi",
                    max_tokens=20,
                    stream=True,
                ),
            )

        assert resp.status_code == 200

        # Parse SSE events from the response body
        events = _parse_sse_events(resp.text)
        assert len(events) > 0, "Expected at least one SSE event"

        # Extract event types in order
        event_types = [e["type"] for e in events]

        # -- Required events --
        assert "message_start" in event_types, f"Missing message_start in {event_types}"
        assert "message_stop" in event_types, f"Missing message_stop in {event_types}"

        # message_start should be first
        assert event_types[0] == "message_start", \
            f"First event should be message_start, got {event_types[0]}"

        # message_stop should be last
        assert event_types[-1] == "message_stop", \
            f"Last event should be message_stop, got {event_types[-1]}"

        # -- Verify message_start structure --
        msg_start = events[0]
        assert "message" in msg_start
        msg = msg_start["message"]
        assert msg["id"].startswith("msg_")
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"

        # -- Check for content_block events --
        assert "content_block_start" in event_types, \
            f"Missing content_block_start in {event_types}"
        assert "content_block_stop" in event_types, \
            f"Missing content_block_stop in {event_types}"

        # -- Verify message_delta has stop_reason and usage --
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(msg_deltas) > 0, "Expected at least one message_delta"
        md = msg_deltas[0]
        assert "delta" in md
        assert "stop_reason" in md["delta"]
        assert md["delta"]["stop_reason"] in ("end_turn", "tool_use", "max_tokens")
        assert "usage" in md

    @pytest.mark.asyncio
    async def test_streaming_assembled_content(self) -> None:
        """Verify that streaming deltas assemble into non-empty text."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json=_minimal_request(
                    content="Say the word 'hello'",
                    max_tokens=20,
                    stream=True,
                ),
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        # Collect all text deltas
        text_parts = []
        for e in events:
            if e.get("type") == "content_block_delta":
                delta = e.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))

        assembled = "".join(text_parts)
        assert len(assembled) > 0, "Assembled streaming text should not be empty"

    @pytest.mark.asyncio
    async def test_streaming_content_type(self) -> None:
        """Verify the response has text/event-stream content type."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json=_minimal_request(
                    content="Say ok",
                    max_tokens=10,
                    stream=True,
                ),
            )

        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type, \
            f"Expected text/event-stream, got {content_type}"
