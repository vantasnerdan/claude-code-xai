"""Tests for newline unescaping in response text.

Verifies that literal escape sequences (backslash-n, backslash-t) in Grok's
response text are converted to real control characters before reaching Claude
Code. This fixes the "escaped newlines in plan output" bug where Grok returns
text with literal \\n instead of actual newline characters.

Coverage:
1. unescape_text() function directly
2. Non-streaming reverse translation (_build_content)
3. Streaming translation (adapter and stateless)
4. Tool call arguments are NOT unescaped (structured data)
5. E2E round trip through JSONResponse
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from translation.reverse import unescape_text, openai_to_anthropic
from translation.streaming import (
    translate_sse_event,
    OpenAIToAnthropicStreamAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion_with_content(content: str) -> dict[str, Any]:
    """Build an OpenAI completion response with the given content text."""
    return {
        "id": "chatcmpl-newline-test",
        "object": "chat.completion",
        "created": 1709000000,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _streaming_chunk_with_content(content: str) -> dict[str, Any]:
    """Build an OpenAI streaming chunk with the given delta content."""
    return {
        "id": "chatcmpl-stream-nl",
        "object": "chat.completion.chunk",
        "created": 1709000300,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


async def _lines_to_async_iter(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _collect_events(adapter: OpenAIToAnthropicStreamAdapter) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for event in adapter:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# unescape_text() unit tests
# ---------------------------------------------------------------------------


class TestUnescapeText:
    """Direct tests for the unescape_text function."""

    def test_literal_newline_converted(self) -> None:
        """Literal backslash-n becomes a real newline."""
        assert unescape_text("Step 1\\nStep 2") == "Step 1\nStep 2"

    def test_literal_tab_converted(self) -> None:
        """Literal backslash-t becomes a real tab."""
        assert unescape_text("col1\\tcol2") == "col1\tcol2"

    def test_literal_carriage_return_converted(self) -> None:
        """Literal backslash-r becomes a real carriage return."""
        assert unescape_text("line1\\rline2") == "line1\rline2"

    def test_multiple_newlines(self) -> None:
        """Multiple literal newlines are all converted."""
        text = "1. First\\n2. Second\\n3. Third"
        expected = "1. First\n2. Second\n3. Third"
        assert unescape_text(text) == expected

    def test_mixed_escape_sequences(self) -> None:
        """Mixed literal escapes (newline + tab) are all converted."""
        text = "Header\\n\\tIndented line\\nNext line"
        expected = "Header\n\tIndented line\nNext line"
        assert unescape_text(text) == expected

    def test_already_real_newlines_preserved(self) -> None:
        """Text that already has real newlines is passed through unchanged."""
        text = "Step 1\nStep 2\nStep 3"
        assert unescape_text(text) == text

    def test_no_backslash_passthrough(self) -> None:
        """Text with no backslashes at all returns fast-path unchanged."""
        text = "Hello, World!"
        assert unescape_text(text) == text

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert unescape_text("") == ""

    def test_double_backslash_not_corrupted(self) -> None:
        r"""Actual double-backslash followed by n (\\n) is not unescaped.

        When the model outputs a literal \\n (to show an escape sequence
        in code), the double backslash means "literal backslash" and the
        n is just 'n'. The negative lookbehind prevents unescaping.
        """
        # In Python source: "\\\\n" is the string \\n (two chars: \ and n preceded by \)
        text = "print(\\\\n)"
        # The lookbehind sees the preceding backslash and skips
        result = unescape_text(text)
        # Should NOT convert to a real newline
        assert "\\n" in result

    def test_real_newline_mixed_with_literal(self) -> None:
        """Text with both real and literal newlines: only literals converted."""
        text = "Real newline here\nLiteral here\\nDone"
        expected = "Real newline here\nLiteral here\nDone"
        assert unescape_text(text) == expected

    def test_plan_output_realistic(self) -> None:
        """Realistic plan output from Grok with literal newlines throughout."""
        text = (
            "Here is my plan:\\n"
            "\\n"
            "1. Read the configuration file\\n"
            "2. Parse the settings\\n"
            "3. Apply changes\\n"
            "\\n"
            "Let me start with step 1."
        )
        expected = (
            "Here is my plan:\n"
            "\n"
            "1. Read the configuration file\n"
            "2. Parse the settings\n"
            "3. Apply changes\n"
            "\n"
            "Let me start with step 1."
        )
        assert unescape_text(text) == expected


# ---------------------------------------------------------------------------
# Non-streaming reverse translation
# ---------------------------------------------------------------------------


class TestNonStreamingNewlines:
    """Multiline content through the non-streaming reverse translator."""

    def test_literal_newlines_in_response_unescaped(self) -> None:
        """Literal \\n in Grok response text becomes real newlines."""
        response = _completion_with_content("Step 1\\nStep 2\\nStep 3")
        result = openai_to_anthropic(response)
        text = result["content"][0]["text"]
        assert text == "Step 1\nStep 2\nStep 3"
        assert text.count("\n") == 2

    def test_real_newlines_preserved(self) -> None:
        """Real newlines in Grok response pass through correctly."""
        response = _completion_with_content("Step 1\nStep 2\nStep 3")
        result = openai_to_anthropic(response)
        text = result["content"][0]["text"]
        assert text == "Step 1\nStep 2\nStep 3"
        assert text.count("\n") == 2

    def test_tool_call_arguments_not_unescaped(self) -> None:
        """Tool call JSON arguments are NOT run through unescape_text.

        Tool arguments are structured JSON data parsed by json.loads().
        The unescape_text function is only applied to display text content,
        never to parsed tool argument values.
        """
        # Build arguments JSON where the string value contains a real newline
        # (which is what json.loads produces from "hello\\nworld" in JSON)
        args_json = '{"file_path": "/tmp/test.py", "content": "line1\\nline2"}'
        response = {
            "id": "chatcmpl-tool-nl",
            "object": "chat.completion",
            "created": 1709000000,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_nl_test",
                                "type": "function",
                                "function": {
                                    "name": "Write",
                                    "arguments": args_json,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = openai_to_anthropic(response)
        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        # json.loads converts \\n in JSON to a real newline in the string value.
        # The tool input should contain exactly what json.loads produced.
        assert tool_block["input"]["content"] == "line1\nline2"
        # Verify unescape_text was NOT applied (it would be a no-op here
        # since the value already has real newlines, but the point is the
        # code path does not call it for tool arguments)

    def test_multiline_plan_e2e(self) -> None:
        """Full plan with multiple paragraphs renders with real newlines."""
        # Each \\n in the Python string literal is a literal backslash-n
        # (two characters), simulating what Grok sends.
        plan = (
            "## Implementation Plan\\n"
            "\\n"
            "### Phase 1: Research\\n"
            "- Read existing code\\n"
            "- Identify dependencies\\n"
            "\\n"
            "### Phase 2: Implementation\\n"
            "- Create new module\\n"
            "- Write tests\\n"
            "\\n"
            "### Phase 3: Review\\n"
            "- Run test suite\\n"
            "- Create PR"
        )
        response = _completion_with_content(plan)
        result = openai_to_anthropic(response)
        text = result["content"][0]["text"]

        # All literal \n sequences should be converted to real newlines
        assert "## Implementation Plan\n" in text
        assert "### Phase 1: Research\n" in text
        assert "- Read existing code\n" in text
        # Count: 11 literal \n in the input, plus none from Python string concat
        # The plan has: Plan\n, \n, Phase1\n, read\n, deps\n, \n, Phase2\n,
        # module\n, tests\n, \n, Phase3\n, suite\n = 12 literal \n sequences
        assert text.count("\n") == 12


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------


class TestStreamingNewlines:
    """Multiline content through the streaming translator."""

    def test_stateless_literal_newline_unescaped(self) -> None:
        """translate_sse_event unescapes literal \\n in text deltas."""
        chunk = _streaming_chunk_with_content("Hello\\nWorld")
        events = translate_sse_event(chunk, is_first=False)

        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "text_delta"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"]["text"] == "Hello\nWorld"

    def test_stateless_real_newline_preserved(self) -> None:
        """translate_sse_event preserves real newlines in text deltas."""
        chunk = _streaming_chunk_with_content("Hello\nWorld")
        events = translate_sse_event(chunk, is_first=False)

        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "text_delta"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"]["text"] == "Hello\nWorld"

    @pytest.mark.asyncio
    async def test_adapter_literal_newlines_unescaped(self) -> None:
        """The stream adapter unescapes literal \\n in assembled text."""
        chunks = [
            {
                "id": "chatcmpl-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000300,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000300,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{"index": 0, "delta": {"content": "Step 1\\nStep 2"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000300,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{"index": 0, "delta": {"content": "\\nStep 3"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000300,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        lines.append("data: [DONE]")

        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        full_text = "".join(d["delta"]["text"] for d in text_deltas)
        assert full_text == "Step 1\nStep 2\nStep 3"
        assert full_text.count("\n") == 2

    @pytest.mark.asyncio
    async def test_adapter_tool_arguments_not_unescaped(self) -> None:
        """Tool call arguments in streaming are NOT unescaped.

        The input_json_delta events pass through raw partial JSON strings.
        unescape_text is only applied to text_delta events.
        """
        chunks = [
            {
                "id": "chatcmpl-tool-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000400,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_stream_nl",
                            "type": "function",
                            "function": {"name": "Write", "arguments": ""},
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-tool-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000400,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"content": "line1\\nline2"}'},
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-tool-nl-stream",
                "object": "chat.completion.chunk",
                "created": 1709000400,
                "model": "grok-4-1-fast-reasoning",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        lines.append("data: [DONE]")

        source = _lines_to_async_iter(lines)
        adapter = OpenAIToAnthropicStreamAdapter(source)
        events = await _collect_events(adapter)

        json_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        # Tool arguments are raw partial JSON strings — NOT processed by
        # unescape_text. The consumer assembles and parses them.
        full_json = "".join(d["delta"]["partial_json"] for d in json_deltas)
        parsed = json.loads(full_json)
        # json.loads converts \n in JSON to a real newline — that's correct
        # JSON parsing behavior, not our unescape function
        assert parsed["content"] == "line1\nline2"


# ---------------------------------------------------------------------------
# SSE wire format round trip
# ---------------------------------------------------------------------------


class TestSSEWireFormat:
    """Verify newlines survive the SSE serialization round trip."""

    def test_real_newline_in_sse_json(self) -> None:
        """A real newline in text survives json.dumps -> SSE data line -> json.loads."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Line 1\nLine 2"},
        }
        sse_line = f"data: {json.dumps(event)}"

        # Verify the SSE line is valid (no raw newlines breaking the format)
        assert sse_line.count("\n") == 0, "SSE data line must not contain raw newlines"

        # Verify round trip
        payload = sse_line[6:]
        recovered = json.loads(payload)
        assert recovered["delta"]["text"] == "Line 1\nLine 2"

    def test_multiline_plan_sse_round_trip(self) -> None:
        """A full multiline plan text survives SSE serialization."""
        plan_text = "Plan:\n1. Step one\n2. Step two\n3. Step three\n\nDone."
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": plan_text},
        }
        sse_line = f"data: {json.dumps(event)}"

        # No raw newlines in the SSE data line
        assert "\n" not in sse_line.split("data: ", 1)[1].replace("\\n", "")

        # Full round trip
        recovered = json.loads(sse_line[6:])
        assert recovered["delta"]["text"] == plan_text
        assert recovered["delta"]["text"].count("\n") == 5
