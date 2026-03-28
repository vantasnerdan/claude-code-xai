"""Tests for Responses API forward translation."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest

from translation.responses_forward import (
    anthropic_to_responses,
    _translate_messages,
    _extract_tool_result,
)
from translation.tools import translate_tools_responses as _translate_tools_responses


class TestAnthropicToResponses:
    """Tests for the main forward translation function."""

    def test_simple_user_message(self):
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false", "PREAMBLE_ENABLED": "false"}):
            from importlib import reload
            import translation.responses_forward
            reload(translation.responses_forward)
            result = translation.responses_forward.anthropic_to_responses(request)

        assert "input" in result
        assert "messages" not in result
        assert result["model"]  # Should be resolved
        assert result["store"] is False

    def test_system_prompt_in_input_array(self):
        request = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are a helper.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false", "PREAMBLE_ENABLED": "false"}):
            from importlib import reload
            import translation.responses_forward
            reload(translation.responses_forward)
            result = translation.responses_forward.anthropic_to_responses(request)

        system_msgs = [m for m in result["input"] if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a helper."

    def test_stream_flag_preserved(self):
        request = {
            "model": "claude-sonnet-4-20250514",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false", "PREAMBLE_ENABLED": "false"}):
            from importlib import reload
            import translation.responses_forward
            reload(translation.responses_forward)
            result = translation.responses_forward.anthropic_to_responses(request)

        assert result["stream"] is True

    def test_tools_translated_to_responses_format(self):
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Reads a file.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false", "PREAMBLE_ENABLED": "false"}):
            from importlib import reload
            import translation.responses_forward
            reload(translation.responses_forward)
            result = translation.responses_forward.anthropic_to_responses(request)

        assert "tools" in result
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "Read"
        # Responses API tools have name at top level, not nested in 'function'.
        assert "function" not in tool


class TestTranslateMessages:
    """Tests for message translation to Responses API input format."""

    def test_text_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _translate_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_content_block_message(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        result = _translate_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_tool_use_becomes_function_call(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test.py"},
                    }
                ],
            }
        ]
        result = _translate_messages(msgs)
        assert len(result) == 1
        fc = result[0]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "toolu_123"
        assert fc["name"] == "Read"
        args = json.loads(fc["arguments"])
        assert args["file_path"] == "/tmp/test.py"

    def test_tool_result_becomes_function_call_output(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "file contents here",
                    }
                ],
            }
        ]
        result = _translate_messages(msgs)
        assert len(result) == 1
        fco = result[0]
        assert fco["type"] == "function_call_output"
        assert fco["call_id"] == "toolu_123"
        assert fco["output"] == "file contents here"

    def test_mixed_text_and_tool_use(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_456",
                        "name": "Read",
                        "input": {"file_path": "/tmp/x.py"},
                    },
                ],
            }
        ]
        result = _translate_messages(msgs)
        # Should produce text message + function_call item.
        text_msgs = [m for m in result if m.get("role") == "assistant"]
        fc_msgs = [m for m in result if m.get("type") == "function_call"]
        assert len(text_msgs) == 1
        assert text_msgs[0]["content"] == "Let me read that file."
        assert len(fc_msgs) == 1

    def test_empty_content(self):
        msgs = [{"role": "user", "content": None}]
        result = _translate_messages(msgs)
        assert result == [{"role": "user", "content": ""}]

    def test_image_raises(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"data": "base64..."}}],
            }
        ]
        with pytest.raises(NotImplementedError, match="Image"):
            _translate_messages(msgs)


class TestExtractToolResult:
    """Tests for tool result extraction."""

    def test_string_content(self):
        block = {"tool_use_id": "t1", "content": "result text"}
        result = _extract_tool_result(block)
        assert result["type"] == "function_call_output"
        assert result["call_id"] == "t1"
        assert result["output"] == "result text"

    def test_list_content(self):
        block = {
            "tool_use_id": "t2",
            "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
        }
        result = _extract_tool_result(block)
        assert result["output"] == "line1\nline2"

    def test_empty_content(self):
        block = {"tool_use_id": "t3", "content": ""}
        result = _extract_tool_result(block)
        assert result["output"] == ""


class TestTranslateToolsResponses:
    """Tests for tool definition translation to Responses API format."""

    def test_flat_structure(self):
        anthropic_tools = [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        result = _translate_tools_responses(anthropic_tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["name"] == "Read"
        assert t["description"] == "Read a file"
        assert t["parameters"] == {"type": "object", "properties": {"path": {"type": "string"}}}
        # Must NOT have nested 'function' key.
        assert "function" not in t
