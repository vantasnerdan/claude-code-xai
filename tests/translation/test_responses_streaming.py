"""Tests for Responses API streaming adapter."""

import json
import pytest
from typing import Any, AsyncIterator

from translation.responses_streaming import ResponsesStreamAdapter


class MockAsyncIterator:
    """Helper to create async iterators from a list of strings."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._index = 0

    def __aiter__(self) -> "MockAsyncIterator":
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _data_line(event_type: str, data: dict[str, Any]) -> list[str]:
    """Create SSE event + data lines."""
    return [
        f"event: {event_type}",
        f"data: {json.dumps(data)}",
    ]


@pytest.mark.asyncio
async def test_text_streaming():
    """Test basic text output streaming."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.output_item.added", {
            "type": "response.output_item.added",
            "item": {"type": "message"},
        }),
        *_data_line("response.content_part.added", {
            "type": "response.content_part.added",
            "part": {"type": "output_text"},
        }),
        *_data_line("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": "Hello ",
        }),
        *_data_line("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": "world!",
        }),
        *_data_line("response.output_text.done", {
            "type": "response.output_text.done",
            "text": "Hello world!",
        }),
        *_data_line("response.output_item.done", {
            "type": "response.output_item.done",
        }),
        *_data_line("response.completed", {
            "type": "response.completed",
            "response": {
                "output": [{"type": "message"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types

    # Check text deltas.
    text_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "text_delta"]
    assert len(text_deltas) == 2
    assert text_deltas[0]["delta"]["text"] == "Hello "
    assert text_deltas[1]["delta"]["text"] == "world!"


@pytest.mark.asyncio
async def test_function_call_streaming():
    """Test function call output streaming."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.output_item.added", {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call_abc", "name": "Read"},
        }),
        *_data_line("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "delta": '{"file_',
        }),
        *_data_line("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "delta": 'path": "/tmp/x.py"}',
        }),
        *_data_line("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "arguments": '{"file_path": "/tmp/x.py"}',
        }),
        *_data_line("response.output_item.done", {
            "type": "response.output_item.done",
        }),
        *_data_line("response.completed", {
            "type": "response.completed",
            "response": {
                "output": [{"type": "function_call"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    types = [e["type"] for e in events]
    assert "content_block_start" in types

    # Find the tool_use block start.
    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Read"
    assert tool_starts[0]["content_block"]["id"] == "call_abc"

    # Check JSON deltas.
    json_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "input_json_delta"]
    assert len(json_deltas) == 2

    # Stop reason should be tool_use.
    msg_delta = [e for e in events if e.get("type") == "message_delta"]
    assert msg_delta[0]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_incomplete_response():
    """Test handling of response.incomplete event."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": "partial",
        }),
        *_data_line("response.incomplete", {
            "type": "response.incomplete",
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    msg_delta = [e for e in events if e.get("type") == "message_delta"]
    assert msg_delta[0]["delta"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_failed_response():
    """Test handling of response.failed event."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.failed", {
            "type": "response.failed",
            "response": {"error": {"message": "Model overloaded"}},
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 1
    assert "overloaded" in errors[0]["error"]["message"].lower()


@pytest.mark.asyncio
async def test_empty_stream():
    """Test adapter handles empty stream gracefully."""
    adapter = ResponsesStreamAdapter(MockAsyncIterator([]))
    events = [e async for e in adapter]

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "message_stop" in types


@pytest.mark.asyncio
async def test_usage_captured():
    """Test that usage stats are captured from response.completed."""
    lines = [
        *_data_line("response.completed", {
            "type": "response.completed",
            "response": {
                "output": [],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    _ = [e async for e in adapter]

    assert adapter.usage.get("input_tokens") == 100
    assert adapter.usage.get("output_tokens") == 50


@pytest.mark.asyncio
async def test_done_signal():
    """Test [DONE] signal terminates stream."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        "data: [DONE]",
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "message_stop" in types


@pytest.mark.asyncio
async def test_unescape_text_in_delta():
    """Test that literal \\n in text deltas is unescaped."""
    lines = [
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": "line1\\nline2",
        }),
        *_data_line("response.completed", {
            "type": "response.completed",
            "response": {"output": [], "usage": {}},
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    text_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "text_delta"]
    assert text_deltas[0]["delta"]["text"] == "line1\nline2"


@pytest.mark.asyncio
async def test_model_from_response_created():
    """Test model name is captured from response.created event."""
    lines = [
        *_data_line("response.created", {
            "type": "response.created",
            "response": {"model": "grok-4.20-multi-agent"},
        }),
        *_data_line("response.in_progress", {"type": "response.in_progress"}),
        *_data_line("response.completed", {
            "type": "response.completed",
            "response": {"output": [], "usage": {}},
        }),
    ]

    adapter = ResponsesStreamAdapter(MockAsyncIterator(lines))
    events = [e async for e in adapter]

    msg_start = [e for e in events if e.get("type") == "message_start"]
    assert msg_start[0]["message"]["model"] == "grok-4.20-multi-agent"
