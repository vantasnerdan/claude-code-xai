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

from translation.forward import anthropic_to_openai, translate_messages, translate_tools, _flatten_system

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


class TestFlattenSystem:
    """Tests for _flatten_system() helper."""

    def test_string_passthrough(self) -> None:
        """String system prompt passes through unchanged."""
        assert _flatten_system("Hello world.") == "Hello world."

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _flatten_system("") == ""

    def test_single_text_block(self) -> None:
        """Single text content block flattens to its text."""
        system = [{"type": "text", "text": "You are a coding assistant."}]
        assert _flatten_system(system) == "You are a coding assistant."

    def test_multiple_text_blocks_joined(self) -> None:
        """Multiple text blocks are joined with double newlines."""
        system = [
            {"type": "text", "text": "First block."},
            {"type": "text", "text": "Second block."},
        ]
        assert _flatten_system(system) == "First block.\n\nSecond block."

    def test_empty_text_blocks_skipped(self) -> None:
        """Empty text blocks are skipped."""
        system = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Content here."},
        ]
        assert _flatten_system(system) == "Content here."

    def test_empty_list_returns_empty_string(self) -> None:
        """Empty list returns empty string."""
        assert _flatten_system([]) == ""

    def test_non_text_blocks_skipped(self) -> None:
        """Non-text blocks are skipped during flattening."""
        system = [
            {"type": "text", "text": "Instructions."},
            {"type": "image", "source": {"data": "..."}},
        ]
        assert _flatten_system(system) == "Instructions."

    def test_invalid_type_raises(self) -> None:
        """Non-string non-list input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str or list"):
            _flatten_system(123)  # type: ignore[arg-type]


class TestListTypeSystemInForwardTranslation:
    """Tests for anthropic_to_openai() handling list-type system field.

    Claude Code sends system as a list of content blocks for
    streaming/opus requests. This crashed in PR #19 because
    strip_anthropic_identity() tried to run regex on a list.
    """

    def test_list_system_produces_system_message(self) -> None:
        """List-type system field is flattened and becomes system role message."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are a coding assistant."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert system_msg["role"] == "system"
        assert "You are a coding assistant." in system_msg["content"]

    def test_list_system_strips_identity(self) -> None:
        """Identity patterns are stripped from list-type system blocks."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": (
                        "You are powered by the model named Claude Opus 4.6. "
                        "The exact model ID is claude-opus-4-6."
                    ),
                },
                {"type": "text", "text": "You are a coding assistant."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert "powered by the model named" not in system_msg["content"]
        assert "claude-opus-4-6" not in system_msg["content"]
        assert "You are a coding assistant." in system_msg["content"]

    def test_list_system_includes_preamble(self) -> None:
        """Preamble is prepended to flattened list-type system content."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are a coding assistant."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert "You are Grok" in system_msg["content"]
        assert "Tool Preference Hierarchy" in system_msg["content"]
        assert "You are a coding assistant." in system_msg["content"]

    def test_list_system_realistic_claude_code_request(self) -> None:
        """Realistic Claude Code streaming request with list-type system."""
        request = {
            "model": "claude-opus-4-20250514",
            "max_tokens": 16384,
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": (
                        "You are powered by the model named Claude Opus 4.6. "
                        "The exact model ID is claude-opus-4-6.\n\n"
                        "Assistant knowledge cutoff is May 2025.\n\n"
                        "<claude_background_info>\n"
                        "The most recent frontier Claude model is Claude Opus 4.6 "
                        "(model ID: 'claude-opus-4-6').\n"
                        "</claude_background_info>\n\n"
                        "<fast_mode_info>\n"
                        "Fast mode for Claude Code uses the same Claude Opus 4.6 "
                        "model with faster output.\n"
                        "</fast_mode_info>"
                    ),
                },
                {
                    "type": "text",
                    "text": "You are an expert coding assistant.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {"role": "user", "content": "Fix the bug in main.py"},
            ],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert system_msg["role"] == "system"
        # Identity claims stripped
        assert "powered by" not in system_msg["content"]
        assert "claude-opus" not in system_msg["content"]
        assert "knowledge cutoff" not in system_msg["content"]
        assert "claude_background_info" not in system_msg["content"]
        # Preamble injected
        assert "You are Grok" in system_msg["content"]
        # Non-identity content preserved
        assert "expert coding assistant" in system_msg["content"]
        # Stream flag preserved
        assert result["stream"] is True

    def test_list_system_all_identity_blocks_produces_preamble_only(self) -> None:
        """When all system blocks are identity-only, result is preamble only."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are powered by Claude Opus 4.6."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert system_msg["role"] == "system"
        assert "You are Grok" in system_msg["content"]
        assert "powered by Claude" not in system_msg["content"]
