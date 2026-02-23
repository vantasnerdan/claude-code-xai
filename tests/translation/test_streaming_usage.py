"""Tests for streaming adapter usage tracking (Issue #26).

Verifies that OpenAIToAnthropicStreamAdapter captures token usage
from streaming chunks for post-stream token logging.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from translation.streaming import OpenAIToAnthropicStreamAdapter


class _AsyncLines:
    """Helper to create an async iterator from a list of SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __aiter__(self) -> _AsyncLines:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    chunk_id: str = "chatcmpl-stream1",
) -> str:
    """Build an SSE data line for an OpenAI streaming chunk."""
    choices: list[dict[str, Any]] = [{
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason,
    }]
    if content is not None:
        choices[0]["delta"]["content"] = content

    data: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": choices,
    }
    if usage is not None:
        data["usage"] = usage

    return f"data: {json.dumps(data)}"


@pytest.mark.asyncio
class TestStreamingUsageCapture:
    """Adapter captures usage from streaming chunks."""

    async def test_usage_empty_by_default(self) -> None:
        lines = _AsyncLines([
            _chunk(content="Hello"),
            _chunk(content=" world"),
            _chunk(finish_reason="stop"),
        ])
        adapter = OpenAIToAnthropicStreamAdapter(lines)
        events = [event async for event in adapter]
        assert len(events) > 0
        assert adapter.usage == {}

    async def test_usage_captured_from_final_chunk(self) -> None:
        lines = _AsyncLines([
            _chunk(content="Hello"),
            _chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
        ])
        adapter = OpenAIToAnthropicStreamAdapter(lines)
        _ = [event async for event in adapter]
        assert adapter.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    async def test_usage_captured_from_mid_stream_chunk(self) -> None:
        """Some providers send usage in a non-final chunk."""
        lines = _AsyncLines([
            _chunk(content="Hi"),
            _chunk(
                content=" there",
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            ),
            _chunk(finish_reason="stop"),
        ])
        adapter = OpenAIToAnthropicStreamAdapter(lines)
        _ = [event async for event in adapter]
        assert adapter.usage.get("prompt_tokens") == 50

    async def test_later_usage_overwrites_earlier(self) -> None:
        """If multiple chunks have usage, the last one wins."""
        lines = _AsyncLines([
            _chunk(
                content="Hi",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            _chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
            ),
        ])
        adapter = OpenAIToAnthropicStreamAdapter(lines)
        _ = [event async for event in adapter]
        assert adapter.usage["prompt_tokens"] == 50
        assert adapter.usage["total_tokens"] == 65
