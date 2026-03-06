"""Tests for prompt-based tool call detection in streaming.

Tests that <tool_call> blocks arriving across multiple SSE chunks
are properly buffered and emitted as tool_use content blocks.
"""

from __future__ import annotations

import json

import pytest

from translation.responses_streaming import ResponsesStreamAdapter


async def _lines_from(items: list[str]):
    """Create an async iterator from a list of SSE lines."""
    for item in items:
        yield item


def _text_delta(text: str) -> str:
    """Create an SSE data line for a text delta event."""
    data = {"type": "response.output_text.delta", "delta": text}
    return f"data: {json.dumps(data)}"


def _completed(
    output: list | None = None,
    usage: dict | None = None,
) -> str:
    """Create an SSE data line for a response.completed event."""
    data = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": output or [],
            "usage": usage or {"input_tokens": 100, "output_tokens": 50},
        },
    }
    return f"data: {json.dumps(data)}"


def _in_progress() -> str:
    data = {"type": "response.in_progress"}
    return f"data: {json.dumps(data)}"


def _content_part_added() -> str:
    data = {"type": "response.content_part.added", "part": {"type": "output_text"}}
    return f"data: {json.dumps(data)}"


@pytest.mark.asyncio
async def test_plain_text_streaming():
    """Plain text without tool calls streams normally."""
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta("Hello "),
        _text_delta("world!"),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    text_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "text_delta"]
    assert len(text_deltas) >= 1
    combined = "".join(d["delta"]["text"] for d in text_deltas)
    assert "Hello" in combined
    assert "world!" in combined

    # Stop reason should be end_turn.
    delta_events = [e for e in events if e.get("type") == "message_delta"]
    assert delta_events[-1]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_tool_call_in_single_chunk():
    """A complete <tool_call> in one chunk is parsed into tool_use."""
    tool_text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test.py"}}\n</tool_call>'
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta(tool_text),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Read"

    # Stop reason should be tool_use.
    delta_events = [e for e in events if e.get("type") == "message_delta"]
    assert delta_events[-1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_tool_call_across_chunks():
    """A <tool_call> split across multiple text deltas is buffered and parsed."""
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta("<tool_call>\n{"),
        _text_delta('"name": "Read", '),
        _text_delta('"parameters": {"file_path": "/test.py"}}'),
        _text_delta("\n</tool_call>"),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Read"


@pytest.mark.asyncio
async def test_text_before_tool_call():
    """Text before a tool call is emitted as text, tool call as tool_use."""
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta("Let me read that.\n\n"),
        _text_delta('<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test"}}\n</tool_call>'),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    text_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "text_delta"]
    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]

    assert len(text_deltas) >= 1
    assert len(tool_starts) == 1
    text_combined = "".join(d["delta"]["text"] for d in text_deltas)
    assert "read that" in text_combined


@pytest.mark.asyncio
async def test_multiple_tool_calls_streaming():
    """Multiple tool calls in stream are all parsed."""
    tool1 = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/a.py"}}\n</tool_call>\n'
    tool2 = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/b.py"}}\n</tool_call>'
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta(tool1),
        _text_delta(tool2),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 2


@pytest.mark.asyncio
async def test_tool_call_json_deltas():
    """Tool call parameters are emitted as input_json_delta."""
    tool_text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test.py"}}\n</tool_call>'
    lines = [
        _in_progress(),
        _content_part_added(),
        _text_delta(tool_text),
        _completed(),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    json_deltas = [e for e in events if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "input_json_delta"]
    assert len(json_deltas) >= 1
    combined = "".join(d["delta"]["partial_json"] for d in json_deltas)
    parsed = json.loads(combined)
    assert parsed["file_path"] == "/test.py"


@pytest.mark.asyncio
async def test_api_native_function_call_still_works():
    """API-native function_call events still work alongside prompt tools."""
    lines = [
        _in_progress(),
        f'data: {json.dumps({"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "call_1", "name": "Read"}})}',
        f'data: {json.dumps({"type": "response.function_call_arguments.delta", "delta": "{"})}',
        f'data: {json.dumps({"type": "response.function_call_arguments.delta", "delta": "\"file_path\": \"/test\"}"})}',
        f'data: {json.dumps({"type": "response.output_item.done", "item": {"type": "function_call"}})}',
        _completed(output=[{"type": "function_call"}]),
    ]
    adapter = ResponsesStreamAdapter(_lines_from(lines))
    events = [e async for e in adapter]

    tool_starts = [e for e in events if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Read"
