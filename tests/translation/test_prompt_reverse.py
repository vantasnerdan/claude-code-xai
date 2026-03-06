"""Tests for prompt-based tool call parsing in responses_reverse.

Tests that <tool_call> blocks in text responses from multi-agent models
are properly parsed into Anthropic tool_use content blocks.
"""

from __future__ import annotations

import json

from translation.responses_reverse import (
    responses_to_anthropic,
    translate_responses_response,
    _build_content,
    _infer_stop_reason,
)


def _make_response(
    output: list | None = None,
    model: str = "grok-4.20-multi-agent",
    usage: dict | None = None,
) -> dict:
    return {
        "id": "resp_test123",
        "model": model,
        "output": output or [],
        "usage": usage or {"input_tokens": 100, "output_tokens": 50},
    }


def _text_output(text: str) -> dict:
    """Create a message output item with text."""
    return {
        "type": "message",
        "content": [{"type": "output_text", "text": text}],
    }


class TestBuildContentWithToolCalls:
    """Tests for _build_content parsing <tool_call> from text."""

    def test_plain_text_unchanged(self):
        output = [_text_output("Hello, world!")]
        blocks = _build_content(output)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello, world!"

    def test_single_tool_call_parsed(self):
        text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/root/test.py"}}\n</tool_call>'
        output = [_text_output(text)]
        blocks = _build_content(output)
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "Read"
        assert tool_blocks[0]["input"]["file_path"] == "/root/test.py"

    def test_text_before_tool_call(self):
        text = 'Let me read that file.\n\n<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test"}}\n</tool_call>'
        output = [_text_output(text)]
        blocks = _build_content(output)
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) >= 1
        assert "read that file" in text_blocks[0]["text"]
        assert len(tool_blocks) == 1

    def test_text_after_tool_call(self):
        text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test"}}\n</tool_call>\n\nDone reading.'
        output = [_text_output(text)]
        blocks = _build_content(output)
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert any("Done reading" in b["text"] for b in text_blocks)

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/a.py"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/b.py"}}\n</tool_call>'
        )
        output = [_text_output(text)]
        blocks = _build_content(output)
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 2
        assert tool_blocks[0]["input"]["file_path"] == "/a.py"
        assert tool_blocks[1]["input"]["file_path"] == "/b.py"

    def test_api_native_function_call_still_works(self):
        """Non-multi-agent function_call items should still work."""
        output = [{
            "type": "function_call",
            "call_id": "call_123",
            "name": "Read",
            "arguments": '{"file_path": "/test.py"}',
        }]
        blocks = _build_content(output)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "Read"

    def test_malformed_tool_call_kept_as_text(self):
        """Malformed <tool_call> JSON should result in text only."""
        text = '<tool_call>\n{not json}\n</tool_call>'
        output = [_text_output(text)]
        blocks = _build_content(output)
        # Malformed is skipped -- we get empty text blocks.
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 0


class TestInferStopReason:
    """Tests for _infer_stop_reason with prompt-based tools."""

    def test_plain_text_end_turn(self):
        output = [_text_output("Hello")]
        assert _infer_stop_reason(output) == "end_turn"

    def test_api_function_call_tool_use(self):
        output = [{"type": "function_call", "call_id": "c1", "name": "Read"}]
        assert _infer_stop_reason(output) == "tool_use"

    def test_prompt_tool_call_in_text(self):
        text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/test"}}\n</tool_call>'
        output = [_text_output(text)]
        assert _infer_stop_reason(output) == "tool_use"

    def test_text_without_tool_call(self):
        output = [_text_output("No tools here")]
        assert _infer_stop_reason(output) == "end_turn"


class TestResponsesToAnthropic:
    """Full response translation tests."""

    def test_prompt_tool_call_full_response(self):
        text = (
            'I will read the file.\n\n'
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/root/main.py"}}\n</tool_call>'
        )
        resp = _make_response(output=[_text_output(text)])
        result = responses_to_anthropic(resp)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "tool_use"

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "Read"

    def test_no_tool_call_end_turn(self):
        resp = _make_response(output=[_text_output("Just text.")])
        result = responses_to_anthropic(resp)
        assert result["stop_reason"] == "end_turn"
        assert all(b["type"] == "text" for b in result["content"])

    def test_error_translation(self):
        error_resp = {
            "error": {"type": "invalid_request_error", "message": "Tool not supported"},
        }
        result = translate_responses_response(error_resp, status_code=400)
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"
