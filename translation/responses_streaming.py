"""Streaming SSE: xAI Responses API event stream -> Anthropic event stream.

The Responses API uses semantic events (response.output_text.delta,
response.function_call_arguments.delta, etc.) instead of the flat
chat.completion.chunk format used by Chat Completions.

This adapter converts those events to Anthropic's streaming protocol:
message_start, content_block_start/delta/stop, message_delta, message_stop.

For multi-agent models using prompt-based tools, text deltas are buffered
when a <tool_call> tag is detected. The buffer is flushed as a tool_use
block when </tool_call> is found.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from translation.config import STOP_REASON_MAP
from translation.reverse import unescape_text
from translation.tool_parser import parse_tool_calls_from_text, has_pending_tool_call


def _msg_start(model: str = "grok-4.20-multi-agent") -> dict[str, Any]:
    """Emit the Anthropic message_start event."""
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    }


def _close(
    reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Emit Anthropic message_delta + message_stop events."""
    u = usage or {}
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {
                "output_tokens": u.get("output_tokens", u.get("completion_tokens", 0)),
            },
        },
        {"type": "message_stop"},
    ]


class ResponsesStreamAdapter:
    """Async adapter: xAI Responses API SSE events -> Anthropic events.

    Handles both API-native function calls and prompt-based tool calls
    (detected via <tool_call> tags in text output).
    """

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._src = source
        self._started = False
        self._done = False
        self._text_block_open = False
        self._tool_block_open = False
        self._block_index = 0
        self._model = "grok-4.20-multi-agent"
        self._q: list[dict[str, Any]] = []
        self.usage: dict[str, int] = {}
        # Buffer for prompt-based tool call detection in streaming.
        self._text_buffer = ""
        self._buffering = False
        self._found_tool_calls = False

    def __aiter__(self) -> ResponsesStreamAdapter:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._q:
            return self._q.pop(0)
        if self._done:
            raise StopAsyncIteration
        await self._fill()
        if self._q:
            return self._q.pop(0)
        raise StopAsyncIteration

    async def _fill(self) -> None:
        """Read from source until we have queued events or source is done."""
        try:
            while not self._q:
                try:
                    line = await self._src.__anext__()
                except StopAsyncIteration:
                    self._q.extend(self._finalize())
                    return

                if not line.strip():
                    continue

                if line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                if payload == "[DONE]":
                    self._q.extend(self._finalize())
                    return
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                self._q.extend(self._handle_event(data))
        except ConnectionError as e:
            self._q.append({
                "type": "error",
                "error": {"type": "connection_error", "message": str(e)},
            })
            self._done = True

    def _handle_event(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Route a Responses API event to the appropriate handler."""
        event_type = data.get("type", "")
        events: list[dict[str, Any]] = []

        if event_type == "response.created":
            self._model = data.get("response", {}).get("model", self._model)

        elif event_type == "response.in_progress":
            if not self._started:
                self._started = True
                events.append(_msg_start(self._model))

        elif event_type == "response.output_item.added":
            if not self._started:
                self._started = True
                events.append(_msg_start(self._model))
            item = data.get("item", {})
            item_type = item.get("type", "")
            if item_type == "function_call":
                events.extend(self._start_tool_block(item))

        elif event_type == "response.output_text.delta":
            if not self._started:
                self._started = True
                events.append(_msg_start(self._model))
            text = data.get("delta", "")
            if text:
                events.extend(self._handle_text_delta(text))

        elif event_type == "response.output_text.done":
            # Text finalization -- flush any remaining buffer.
            events.extend(self._flush_text_buffer())

        elif event_type == "response.function_call_arguments.delta":
            if not self._started:
                self._started = True
                events.append(_msg_start(self._model))
            delta = data.get("delta", "")
            if delta:
                events.append({
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                })

        elif event_type == "response.function_call_arguments.done":
            pass

        elif event_type == "response.output_item.done":
            events.extend(self._close_current_block())

        elif event_type == "response.content_part.added":
            item = data.get("part", {})
            if item.get("type") == "output_text" and not self._text_block_open:
                events.extend(self._open_text_block())

        elif event_type == "response.content_part.done":
            pass

        elif event_type == "response.completed":
            response = data.get("response", {})
            usage = response.get("usage", {})
            self.usage = usage
            events.extend(self._flush_text_buffer())
            events.extend(self._close_current_block())
            stop = self._infer_stop(response)
            if self._found_tool_calls:
                stop = "tool_use"
            events.extend(_close(stop, usage))
            self._done = True

        elif event_type == "response.incomplete":
            events.extend(self._flush_text_buffer())
            events.extend(self._close_current_block())
            events.extend(_close("max_tokens"))
            self._done = True

        elif event_type == "response.failed":
            error = data.get("response", {}).get("error", {})
            events.extend(self._flush_text_buffer())
            events.extend(self._close_current_block())
            events.append({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error.get("message", "Responses API error"),
                },
            })
            self._done = True

        return events

    def _handle_text_delta(self, text: str) -> list[dict[str, Any]]:
        """Handle a text delta, buffering when tool calls are detected.

        Detects <tool_call> tags and buffers text until </tool_call> is
        found, then emits proper tool_use blocks.
        """
        self._text_buffer += text
        events: list[dict[str, Any]] = []

        # Check if we have any completed tool call blocks to parse.
        while "<tool_call>" in self._text_buffer and "</tool_call>" in self._text_buffer:
            events.extend(self._parse_buffered_tool_calls())

        # If we are in the middle of a tool call, keep buffering.
        if has_pending_tool_call(self._text_buffer):
            self._buffering = True
            return events

        # No pending tool call -- flush plain text.
        if self._text_buffer and not self._buffering:
            if not self._text_block_open:
                events.extend(self._open_text_block())
            events.append({
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "text_delta", "text": unescape_text(self._text_buffer)},
            })
            self._text_buffer = ""

        self._buffering = False
        return events

    def _parse_buffered_tool_calls(self) -> list[dict[str, Any]]:
        """Parse completed <tool_call> blocks from the buffer.

        Emits text before the tool call, then the tool_use block itself.
        Updates the buffer to contain only remaining unparsed text.
        """
        events: list[dict[str, Any]] = []
        content_blocks, tool_calls = parse_tool_calls_from_text(self._text_buffer)

        for block in content_blocks:
            if block["type"] == "text":
                text = block.get("text", "")
                if text:
                    if not self._text_block_open:
                        events.extend(self._open_text_block())
                    events.append({
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "text_delta", "text": unescape_text(text)},
                    })
                    # Close text block before tool block.
                    events.extend(self._close_current_block())
            elif block["type"] == "tool_use":
                self._found_tool_calls = True
                events.extend(self._close_current_block())
                self._tool_block_open = True
                events.append({
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                })
                # Send the full input as a single JSON delta.
                input_json = json.dumps(block.get("input", {}))
                events.append({
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "input_json_delta", "partial_json": input_json},
                })
                events.extend(self._close_current_block())

        self._text_buffer = ""
        return events

    def _flush_text_buffer(self) -> list[dict[str, Any]]:
        """Flush remaining text buffer as text content."""
        if not self._text_buffer:
            return []
        events: list[dict[str, Any]] = []
        # If there are tool calls in remaining buffer, parse them.
        if "<tool_call>" in self._text_buffer:
            events.extend(self._parse_buffered_tool_calls())
        elif self._text_buffer.strip():
            if not self._text_block_open:
                events.extend(self._open_text_block())
            events.append({
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "text_delta", "text": unescape_text(self._text_buffer)},
            })
        self._text_buffer = ""
        return events

    def _open_text_block(self) -> list[dict[str, Any]]:
        """Open a new text content block."""
        self._text_block_open = True
        return [{
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": {"type": "text", "text": ""},
        }]

    def _start_tool_block(
        self, item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Close any open block and start a tool_use content block."""
        events = self._close_current_block()
        self._tool_block_open = True
        events.append({
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": {
                "type": "tool_use",
                "id": item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": item.get("name", ""),
                "input": {},
            },
        })
        return events

    def _close_current_block(self) -> list[dict[str, Any]]:
        """Close any open content block and advance the index."""
        events: list[dict[str, Any]] = []
        if self._text_block_open or self._tool_block_open:
            events.append({
                "type": "content_block_stop",
                "index": self._block_index,
            })
            self._text_block_open = False
            self._tool_block_open = False
            self._block_index += 1
        return events

    def _finalize(self) -> list[dict[str, Any]]:
        """Emit closing events when the source stream ends."""
        if self._done:
            return []
        events: list[dict[str, Any]] = []
        if not self._started:
            events.append(_msg_start(self._model))
            self._started = True
        events.extend(self._flush_text_buffer())
        events.extend(self._close_current_block())
        stop = "tool_use" if self._found_tool_calls else "end_turn"
        events.extend(_close(stop))
        self._done = True
        return events

    @staticmethod
    def _infer_stop(response: dict[str, Any]) -> str:
        """Infer Anthropic stop_reason from a Responses API response."""
        status = response.get("status", "completed")
        if status == "incomplete":
            return "max_tokens"
        output = response.get("output", [])
        for item in output:
            if item.get("type") == "function_call":
                return "tool_use"
        return "end_turn"
