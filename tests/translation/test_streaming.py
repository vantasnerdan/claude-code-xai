"""LEGACY: Tests for streaming translation: OpenAI SSE chunks <-> Anthropic SSE events.

These tests define the contract for the `translation.streaming` module,
which handles the LEGACY Chat Completions streaming format
(XAI_USE_CHAT_COMPLETIONS=true).

The PRIMARY streaming path is `translation.responses_streaming`,
tested in `test_responses_streaming.py`.

This is the hardest part of the translation layer because:
1. State must be maintained across chunks (partial tool call arguments)
2. Anthropic has explicit lifecycle events (block_start, block_stop) that OpenAI lacks
3. Error handling mid-stream requires graceful degradation
"""

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from translation.streaming import (
    translate_sse_event,
    OpenAIToAnthropicStreamAdapter,
)

from tests.translation.fixtures.streaming_events import (
    anthropic_message_start,
    anthropic_content_block_start,
    anthropic_content_block_delta,
    anthropic_tool_use_start,
    anthropic_tool_use_delta,
    anthropic_content_block_stop,
    anthropic_message_delta,
    anthropic_message_stop,
    anthropic_full_text_stream,
    openai_stream_chunks,
    openai_tool_call_stream_chunks,
)
from tests.translation.fixtures.openai_completions import (
    streaming_chunk,
    streaming_chunk_with_role,
    streaming_chunk_tool_call,
    streaming_chunk_finish,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _lines_to_async_iter(lines: list[str]) -> AsyncIterator[str]:
    """Convert a list of SSE data lines into an async iterator."""
    for line in lines:
        yield line


async def _collect_events(adapter: OpenAIToAnthropicStreamAdapter) -> list[dict[str, Any]]:
    """Collect all events from the adapter into a list."""
    events: list[dict[str, Any]] = []
    async for event in adapter:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Individual event translation
# ---------------------------------------------------------------------------


class TestSingleEventTranslation:
    """Translating individual OpenAI chunks to Anthropic events."""

    def test_first_chunk_produces_message_start(self) -> None:
        """The first chunk (with role in delta) produces a message_start event."""
        chunk = streaming_chunk_with_role()
        events = translate_sse_event(chunk, is_first=True)

        assert any(e["type"] == "message_start" for e in events)

    def test_text_delta_produces_content_block_delta(self) -> None:
        """A chunk with text content produces a content_block_delta event."""
        chunk = streaming_chunk()
        events = translate_sse_event(chunk, is_first=False)

        deltas = [e for e in events if e["type"] == "content_block_delta"]
        assert len(deltas) >= 1
        assert deltas[0]["delta"]["type"] == "text_delta"
        assert deltas[0]["delta"]["text"] == "Hello"

    def test_finish_chunk_produces_message_delta_and_stop(self) -> None:
        """The final chunk with finish_reason produces message_delta and message_stop."""
        chunk = streaming_chunk_finish()
        events = translate_sse_event(chunk, is_first=False, is_last=True)

        event_types = [e["type"] for e in events]
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_tool_call_chunk_produces_tool_use_events(self) -> None:
        """A chunk with tool_calls delta produces tool_use-related events."""
        chunk = streaming_chunk_tool_call()
        events = translate_sse_event(chunk, is_first=False)

        # Should produce at least one event related to tool use
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Full stream translation via adapter
# ---------------------------------------------------------------------------


class TestOpenAIToAnthropicStreamAdapter:
    """End-to-end streaming translation using the adapter."""

    @pytest.mark.asyncio
    async def test_text_stream_produces_full_event_sequence(self) -> None:
        """A complete text stream produces the full Anthropic event lifecycle."""
        lines = openai_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        event_types = [e["type"] for e in events]

        # Must have the lifecycle events
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_stop" in event_types

    @pytest.mark.asyncio
    async def test_text_content_preserved_across_deltas(self) -> None:
        """All text content from OpenAI deltas appears in Anthropic events."""
        lines = openai_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        full_text = "".join(d["delta"]["text"] for d in text_deltas)
        assert "Hello!" in full_text
        assert "How can I help?" in full_text

    @pytest.mark.asyncio
    async def test_message_start_is_first_event(self) -> None:
        """The first event emitted is always message_start."""
        lines = openai_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        assert events[0]["type"] == "message_start"

    @pytest.mark.asyncio
    async def test_message_stop_is_last_event(self) -> None:
        """The last event emitted is always message_stop."""
        lines = openai_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        assert events[-1]["type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_done_sentinel_terminates_stream(self) -> None:
        """The [DONE] sentinel line terminates the stream after emitting message_stop."""
        lines = openai_stream_chunks()
        assert lines[-1] == "data: [DONE]"

        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        # Stream should terminate cleanly
        assert events[-1]["type"] == "message_stop"


class TestToolCallStreaming:
    """Streaming tool call translation."""

    @pytest.mark.asyncio
    async def test_tool_call_stream_produces_tool_use_block(self) -> None:
        """A streamed tool call produces content_block_start with type=tool_use."""
        lines = openai_tool_call_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        block_starts = [e for e in events if e["type"] == "content_block_start"]
        tool_starts = [
            e for e in block_starts
            if e["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) >= 1

    @pytest.mark.asyncio
    async def test_partial_arguments_streamed_as_input_json_delta(self) -> None:
        """Incremental function arguments become input_json_delta events."""
        lines = openai_tool_call_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        json_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

        # Concatenated partial_json should form valid JSON
        full_json = "".join(d["delta"]["partial_json"] for d in json_deltas)
        parsed = json.loads(full_json)
        assert "file_path" in parsed

    @pytest.mark.asyncio
    async def test_tool_call_id_in_block_start(self) -> None:
        """The tool call ID appears in the content_block_start event."""
        lines = openai_tool_call_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        tool_starts = [
            e for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) >= 1
        assert "id" in tool_starts[0]["content_block"]
        assert tool_starts[0]["content_block"]["id"] == "call_stream_read"

    @pytest.mark.asyncio
    async def test_tool_name_in_block_start(self) -> None:
        """The function name appears in the content_block_start event."""
        lines = openai_tool_call_stream_chunks()
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        tool_starts = [
            e for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        assert tool_starts[0]["content_block"]["name"] == "Read"


class TestStreamErrorHandling:
    """Error conditions during streaming."""

    @pytest.mark.asyncio
    async def test_empty_stream_produces_minimal_events(self) -> None:
        """An empty stream (just [DONE]) still produces message_start and message_stop."""
        lines = ["data: [DONE]"]
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        event_types = [e["type"] for e in events]
        assert "message_start" in event_types
        assert "message_stop" in event_types

    @pytest.mark.asyncio
    async def test_malformed_json_chunk_skipped(self) -> None:
        """A malformed JSON chunk is skipped without crashing the stream."""
        lines = [
            'data: {"id":"test","object":"chat.completion.chunk","created":1709000300,'
            '"model":"grok-4-1-fast-reasoning","choices":[{"index":0,"delta":'
            '{"role":"assistant","content":""},"finish_reason":null}]}',
            "data: {INVALID JSON}",
            'data: {"id":"test","object":"chat.completion.chunk","created":1709000300,'
            '"model":"grok-4-1-fast-reasoning","choices":[{"index":0,"delta":'
            '{"content":"hello"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        # Should not crash; should produce events from valid chunks
        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert any("hello" in d["delta"]["text"] for d in text_deltas)

    @pytest.mark.asyncio
    async def test_connection_drop_produces_error_event(self) -> None:
        """If the source iterator raises an exception, an error event is produced."""
        async def failing_source() -> AsyncIterator[str]:
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1709000300,"model":"grok-4-1-fast-reasoning","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'
            raise ConnectionError("Connection reset by peer")

        adapter = OpenAIToAnthropicStreamAdapter(failing_source())
        events = await _collect_events(adapter)

        # Should produce an error event or handle gracefully
        event_types = [e["type"] for e in events]
        # At minimum, message_start should have been emitted before the error
        assert "message_start" in event_types
        # An error event should be present
        assert "error" in event_types or "message_stop" in event_types

    @pytest.mark.asyncio
    async def test_non_data_lines_ignored(self) -> None:
        """SSE lines that don't start with 'data: ' are ignored."""
        lines = [
            ": heartbeat",
            'data: {"id":"test","object":"chat.completion.chunk","created":1709000300,'
            '"model":"grok-4-1-fast-reasoning","choices":[{"index":0,"delta":'
            '{"role":"assistant","content":""},"finish_reason":null}]}',
            "event: ping",
            'data: {"id":"test","object":"chat.completion.chunk","created":1709000300,'
            '"model":"grok-4-1-fast-reasoning","choices":[{"index":0,"delta":'
            '{"content":"test"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        # Should produce valid events, ignoring non-data lines
        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert any("test" in d["delta"]["text"] for d in text_deltas)
