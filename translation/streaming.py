"""Streaming SSE translation: OpenAI delta stream -> Anthropic event stream.

OpenAI streams: data: {"choices": [{"delta": {...}}]} lines
Anthropic streams: typed events (message_start, content_block_start,
  content_block_delta, content_block_stop, message_delta, message_stop)

Must synthesize Anthropic event lifecycle from OpenAI's flat deltas.
Maintains state across chunks for tool call argument accumulation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator


def translate_sse_event(
    chunk: dict[str, Any],
    is_first: bool = False,
    is_last: bool = False,
) -> list[dict[str, Any]]:
    """Translate a single OpenAI chunk to Anthropic event(s).

    A single OpenAI chunk can produce multiple Anthropic events
    (e.g., first chunk produces message_start + content_block_start).

    Args:
        chunk: Parsed OpenAI streaming chunk.
        is_first: Whether this is the first chunk in the stream.
        is_last: Whether this is the last chunk (has finish_reason).

    Returns:
        List of Anthropic-format SSE events.
    """
    events: list[dict[str, Any]] = []
    choices = chunk.get("choices", [])
    if not choices:
        return events

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    if is_first:
        events.append(_make_message_start(chunk))

    # Text content delta
    text = delta.get("content")
    if text is not None and text != "":
        events.append({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })

    # Tool call deltas
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            if tc.get("id"):
                # New tool call: emit content_block_start
                events.append({
                    "type": "content_block_start",
                    "index": tc.get("index", 0),
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": func.get("name", ""),
                        "input": {},
                    },
                })
            args = func.get("arguments", "")
            if args:
                events.append({
                    "type": "content_block_delta",
                    "index": tc.get("index", 0),
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args,
                    },
                })

    if is_last or finish_reason is not None:
        from translation.config import STOP_REASON_MAP
        stop = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
        usage = chunk.get("usage", {})
        events.append({
            "type": "content_block_stop",
            "index": 0,
        })
        events.append({
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {
                "output_tokens": usage.get("completion_tokens", 0),
            },
        })
        events.append({"type": "message_stop"})

    return events


class OpenAIToAnthropicStreamAdapter:
    """Async adapter that converts an OpenAI SSE line stream to Anthropic events.

    Consumes an async iterator of SSE data lines (strings starting with
    "data: ") and yields Anthropic-format event dicts.

    Usage:
        source = async_sse_line_iterator()
        adapter = OpenAIToAnthropicStreamAdapter(source)
        async for event in adapter:
            # event is an Anthropic SSE event dict
            yield f"event: {event['type']}\\ndata: {json.dumps(event)}\\n\\n"
    """

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._source = source
        self._started = False
        self._finished = False
        self._block_open = False
        self._tool_block_open = False

    def __aiter__(self) -> OpenAIToAnthropicStreamAdapter:
        return self

    async def __anext__(self) -> dict[str, Any]:
        # Drain the event buffer first
        if hasattr(self, "_buffer") and self._buffer:
            return self._buffer.pop(0)

        if self._finished:
            raise StopAsyncIteration

        self._buffer: list[dict[str, Any]] = []
        await self._consume_next()

        if self._buffer:
            return self._buffer.pop(0)

        raise StopAsyncIteration

    async def _consume_next(self) -> None:
        """Read lines from source until we have events to emit."""
        try:
            while not self._buffer:
                try:
                    line = await self._source.__anext__()
                except StopAsyncIteration:
                    self._emit_end()
                    return

                events = self._process_line(line)
                self._buffer.extend(events)

        except ConnectionError as exc:
            # Emit error event on connection drop
            self._buffer.append({
                "type": "error",
                "error": {
                    "type": "connection_error",
                    "message": str(exc),
                },
            })
            self._finished = True

    def _process_line(self, line: str) -> list[dict[str, Any]]:
        """Process a single SSE data line."""
        # Skip non-data lines (comments, event type declarations, etc.)
        if not line.startswith("data: "):
            return []

        payload = line[6:]  # Strip "data: " prefix

        # Handle [DONE] sentinel
        if payload.strip() == "[DONE]":
            return self._emit_end()

        # Parse JSON
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # Skip malformed chunks silently
            return []

        return self._translate_chunk(chunk)

    def _translate_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate a parsed OpenAI chunk to Anthropic events."""
        events: list[dict[str, Any]] = []
        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # First chunk: emit message_start
        if not self._started:
            self._started = True
            events.append(_make_message_start(chunk))

        # Text content
        text = delta.get("content")
        if text is not None:
            if not self._block_open and not self._tool_block_open:
                events.append({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                self._block_open = True

            if text != "":
                events.append({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                })

        # Tool call deltas
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                idx = tc.get("index", 0)

                if tc.get("id"):
                    # Close text block if open
                    if self._block_open:
                        events.append({
                            "type": "content_block_stop",
                            "index": 0,
                        })
                        self._block_open = False

                    events.append({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": func.get("name", ""),
                            "input": {},
                        },
                    })
                    self._tool_block_open = True

                args = func.get("arguments", "")
                if args:
                    events.append({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args,
                        },
                    })

        # Finish
        if finish_reason is not None:
            events.extend(self._emit_finish(chunk, finish_reason))

        return events

    def _emit_finish(
        self, chunk: dict[str, Any], finish_reason: str
    ) -> list[dict[str, Any]]:
        """Emit the closing events for the stream."""
        from translation.config import STOP_REASON_MAP

        events: list[dict[str, Any]] = []

        # Close any open blocks
        if self._block_open:
            events.append({"type": "content_block_stop", "index": 0})
            self._block_open = False
        if self._tool_block_open:
            events.append({"type": "content_block_stop", "index": 0})
            self._tool_block_open = False

        stop = STOP_REASON_MAP.get(finish_reason, "end_turn")
        usage = chunk.get("usage", {})

        events.append({
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)},
        })
        events.append({"type": "message_stop"})
        self._finished = True
        return events

    def _emit_end(self) -> list[dict[str, Any]]:
        """Emit end-of-stream events if not already finished."""
        if self._finished:
            return []

        events: list[dict[str, Any]] = []

        if not self._started:
            events.append(_make_message_start_default())
            self._started = True

        if self._block_open:
            events.append({"type": "content_block_stop", "index": 0})
            self._block_open = False
        if self._tool_block_open:
            events.append({"type": "content_block_stop", "index": 0})
            self._tool_block_open = False

        events.append({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        events.append({"type": "message_stop"})
        self._finished = True
        return events


def _make_message_start(chunk: dict[str, Any]) -> dict[str, Any]:
    """Create a message_start event from the first OpenAI chunk."""
    chunk_id = chunk.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    model = chunk.get("model", "grok-4-1-fast-reasoning")

    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{chunk_id}" if not chunk_id.startswith("msg_") else chunk_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    }


def _make_message_start_default() -> dict[str, Any]:
    """Create a minimal message_start for edge cases (empty stream)."""
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "grok-4-1-fast-reasoning",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    }
