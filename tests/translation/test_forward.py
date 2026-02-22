"""Tests for forward translation: Anthropic Messages API -> OpenAI Chat Completions API.

These tests define the contract for the `translation.forward` module.
Every test imports from a module that does not exist yet — that is
intentional. The tests ARE the spec; Nexus implements against them.

The forward translator is responsible for:
1. Extracting the top-level system field into a system role message
2. Converting content block arrays into plain strings
3. Translating tool_use blocks into tool_calls arrays
4. Translating tool_result blocks into tool role messages
5. Converting Anthropic tool definitions (input_schema) to OpenAI format (parameters)
6. Preserving max_tokens and other request-level fields
"""

import json
from typing import Any

import pytest

from translation.forward import anthropic_to_openai, translate_messages, translate_tools

from tests.translation.fixtures.anthropic_messages import (
    simple_text_message,
    system_message_request,
    tool_use_response,
    tool_result_message,
    multi_turn_with_tools,
    parallel_tool_calls,
    full_request_with_tools,
)


class TestSystemMessageExtraction:
    """Top-level system field handling."""

    def test_system_field_becomes_system_role_message(self) -> None:
        """Top-level system field becomes a system role message at position 0.

        When preamble is enabled (default), the system content includes
        the preamble prepended to the original system text.
        """
        request = system_message_request()
        result = anthropic_to_openai(request)

        assert result["messages"][0]["role"] == "system"
        assert request["system"] in result["messages"][0]["content"]

    def test_system_field_absent_injects_preamble(self) -> None:
        """When no system field is present but preamble is enabled, a system message is injected."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        # Preamble is enabled by default, so a system message is always present
        assert result["messages"][0]["role"] == "system"
        assert "Tool Preference Hierarchy" in result["messages"][0]["content"]

    def test_empty_system_field_uses_preamble(self) -> None:
        """An empty system field still produces a system message from preamble."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "",
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        assert result["messages"][0]["role"] == "system"
        assert "Tool Preference Hierarchy" in result["messages"][0]["content"]

    def test_system_message_preserves_preceding_user_message_order(self) -> None:
        """System message is first; original user messages follow in order."""
        request = system_message_request()
        result = anthropic_to_openai(request)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"


class TestContentBlockTranslation:
    """Content block array <-> plain string conversion."""

    def test_single_text_block_to_string(self) -> None:
        """A single text content block becomes a plain string."""
        messages = [simple_text_message()]
        result = translate_messages(messages)

        assert result[0]["content"] == "Hello"
        assert isinstance(result[0]["content"], str)

    def test_multiple_text_blocks_concatenated(self) -> None:
        """Multiple text content blocks are joined with newlines."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ],
        }
        result = translate_messages([message])

        assert "First part." in result[0]["content"]
        assert "Second part." in result[0]["content"]

    def test_string_content_passthrough(self) -> None:
        """If content is already a string (non-standard but possible), pass through."""
        message = {"role": "user", "content": "Already a string"}
        result = translate_messages([message])

        assert result[0]["content"] == "Already a string"


class TestToolUseTranslation:
    """tool_use content blocks -> tool_calls array."""

    def test_tool_use_to_tool_calls(self) -> None:
        """A tool_use content block becomes a tool_calls entry with function object."""
        msg = tool_use_response()
        result = translate_messages([msg])

        assistant_msg = result[0]
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1

        tool_call = assistant_msg["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "Read"
        assert tool_call["id"] == "toolu_01A09q90qw90lq917835lhB"

        # Arguments must be a JSON string, not a dict
        args = json.loads(tool_call["function"]["arguments"])
        assert args["file_path"] == "/home/user/project/src/main.py"

    def test_tool_use_arguments_are_json_string(self) -> None:
        """OpenAI format requires arguments as a JSON string, not a dict."""
        msg = tool_use_response()
        result = translate_messages([msg])

        tool_call = result[0]["tool_calls"][0]
        assert isinstance(tool_call["function"]["arguments"], str)
        # Must be valid JSON
        parsed = json.loads(tool_call["function"]["arguments"])
        assert isinstance(parsed, dict)

    def test_tool_use_id_preserved(self) -> None:
        """The tool_use id maps to the tool_call id."""
        msg = tool_use_response()
        result = translate_messages([msg])

        tool_call = result[0]["tool_calls"][0]
        assert tool_call["id"] == msg["content"][0]["id"]


class TestToolResultTranslation:
    """tool_result content blocks -> tool role messages."""

    def test_tool_result_to_tool_message(self) -> None:
        """A tool_result block becomes a message with role=tool."""
        msg = tool_result_message()
        result = translate_messages([msg])

        tool_msg = result[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_01A09q90qw90lq917835lhB"

    def test_tool_result_content_extracted(self) -> None:
        """The content from the tool_result is flattened into the tool message content."""
        msg = tool_result_message()
        result = translate_messages([msg])

        tool_msg = result[0]
        assert isinstance(tool_msg["content"], str)
        assert "fastapi" in tool_msg["content"].lower()

    def test_tool_result_with_error(self) -> None:
        """A tool_result with is_error=true should still translate."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_error_123",
                    "is_error": True,
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: File not found",
                        }
                    ],
                }
            ],
        }
        result = translate_messages([msg])

        tool_msg = result[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_error_123"
        assert "Error: File not found" in tool_msg["content"]


class TestToolDefinitionTranslation:
    """Anthropic tool definitions -> OpenAI function definitions."""

    def test_input_schema_to_parameters(self, sample_tools: list[dict[str, Any]]) -> None:
        """input_schema in Anthropic tool defs becomes parameters in OpenAI format."""
        result = translate_tools(sample_tools)

        for tool in result:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "parameters" in func
            assert "input_schema" not in func

    def test_tool_name_preserved(self, sample_tools: list[dict[str, Any]]) -> None:
        """Tool names pass through unchanged."""
        result = translate_tools(sample_tools)

        names = [t["function"]["name"] for t in result]
        assert "Read" in names
        assert "Edit" in names
        assert "Write" in names
        assert "Bash" in names
        assert "Grep" in names
        assert "Glob" in names

    def test_tool_description_preserved(self, sample_tools: list[dict[str, Any]]) -> None:
        """Tool descriptions pass through unchanged."""
        result = translate_tools(sample_tools)

        read_tool = next(t for t in result if t["function"]["name"] == "Read")
        assert "Reads a file" in read_tool["function"]["description"]

    def test_schema_properties_preserved(self, sample_tools: list[dict[str, Any]]) -> None:
        """JSON Schema properties from input_schema land in parameters."""
        result = translate_tools(sample_tools)

        read_tool = next(t for t in result if t["function"]["name"] == "Read")
        params = read_tool["function"]["parameters"]
        assert "file_path" in params["properties"]
        assert params["required"] == ["file_path"]

    def test_empty_tools_list(self) -> None:
        """An empty tools list translates to an empty list."""
        result = translate_tools([])
        assert result == []


class TestRequestLevelFields:
    """Request-level field translation."""

    def test_max_tokens_preserved(self) -> None:
        """Required Anthropic max_tokens passes through to OpenAI max_tokens."""
        request = system_message_request()
        result = anthropic_to_openai(request)

        assert result["max_tokens"] == 4096

    def test_model_is_overridden(self) -> None:
        """The Anthropic model is replaced with the target Grok model."""
        request = system_message_request()
        result = anthropic_to_openai(request)

        # The forward translator should set the xAI model, not pass through Claude model
        assert "claude" not in result.get("model", "").lower()

    def test_temperature_default(self) -> None:
        """If no temperature is set, a sensible default is used."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        # Temperature should be present (either from request or default)
        assert "temperature" in result

    def test_stream_flag_preserved(self) -> None:
        """If stream=true in the request, it passes through."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "stream": True,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        assert result["stream"] is True


class TestMultiTurnConversation:
    """Full multi-turn conversation translation."""

    def test_multi_turn_message_count(self) -> None:
        """Multi-turn conversation produces the correct number of OpenAI messages."""
        messages = multi_turn_with_tools()
        result = translate_messages(messages)

        # user, assistant (with tool_calls), tool (result), assistant
        # The exact count depends on translation strategy, but must be >= 4
        assert len(result) >= 4

    def test_multi_turn_role_sequence(self) -> None:
        """Roles alternate correctly in the translated conversation."""
        messages = multi_turn_with_tools()
        result = translate_messages(messages)

        roles = [m["role"] for m in result]
        assert roles[0] == "user"
        assert roles[1] == "assistant"
        assert roles[2] == "tool"
        assert roles[3] == "assistant"

    def test_multi_turn_tool_id_consistency(self) -> None:
        """The tool_call id in the assistant message matches the tool_call_id in the tool message."""
        messages = multi_turn_with_tools()
        result = translate_messages(messages)

        # Find the assistant message with tool_calls
        assistant_msg = next(m for m in result if m["role"] == "assistant" and "tool_calls" in m)
        tool_msg = next(m for m in result if m["role"] == "tool")

        assert assistant_msg["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


class TestParallelToolCalls:
    """Multiple tool_use blocks in a single assistant message."""

    def test_multiple_tool_use_blocks(self) -> None:
        """Multiple tool_use blocks map to multiple entries in tool_calls."""
        msg = parallel_tool_calls()
        result = translate_messages([msg])

        assistant_msg = result[0]
        assert len(assistant_msg["tool_calls"]) == 2

    def test_parallel_tool_ids_unique(self) -> None:
        """Each parallel tool call has a distinct id."""
        msg = parallel_tool_calls()
        result = translate_messages([msg])

        ids = [tc["id"] for tc in result[0]["tool_calls"]]
        assert len(ids) == len(set(ids))

    def test_text_and_tool_calls_coexist(self) -> None:
        """When an assistant message has both text and tool_use, both are preserved."""
        msg = parallel_tool_calls()
        result = translate_messages([msg])

        assistant_msg = result[0]
        # Text content should be present
        assert assistant_msg.get("content") is not None
        assert "compare" in assistant_msg["content"].lower()
        # Tool calls should also be present
        assert len(assistant_msg["tool_calls"]) == 2
