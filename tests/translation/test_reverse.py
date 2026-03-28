"""LEGACY: Tests for reverse translation: OpenAI Chat Completions API -> Anthropic Messages API.

These tests define the contract for the `translation.reverse` module,
which handles the LEGACY Chat Completions format (XAI_USE_CHAT_COMPLETIONS=true).

The PRIMARY reverse translation path is `translation.responses_reverse`,
tested in `test_responses_reverse.py`.

Key responsibilities of the legacy path:
1. String content -> content block arrays
2. tool_calls -> tool_use content blocks
3. finish_reason mapping (stop->end_turn, tool_calls->tool_use, length->max_tokens)
4. Usage field mapping (prompt_tokens->input_tokens, completion_tokens->output_tokens)
5. Error response translation with agentic standard compliance
"""

import json
from typing import Any

import pytest

from translation.reverse import openai_to_anthropic, translate_response

from tests.translation.fixtures.openai_completions import (
    simple_completion,
    tool_call_completion,
    multi_tool_call_completion,
    error_response_429,
    error_response_500,
    error_response_400,
)


class TestStringToContentBlocks:
    """String content -> Anthropic content block arrays."""

    def test_string_becomes_content_block_array(self) -> None:
        """Plain string content becomes a content block array with one text block."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello! How can I help you today?"

    def test_empty_string_becomes_empty_text_block(self) -> None:
        """Empty string content becomes a content block with empty text."""
        response = simple_completion()
        response["choices"][0]["message"]["content"] = ""
        result = openai_to_anthropic(response)

        content = result["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == ""

    def test_response_has_role_assistant(self) -> None:
        """The translated response always has role=assistant."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        assert result["role"] == "assistant"

    def test_response_has_message_id(self) -> None:
        """The translated response includes an id field."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        assert "id" in result
        assert isinstance(result["id"], str)
        assert len(result["id"]) > 0


class TestToolCallsToToolUse:
    """OpenAI tool_calls -> Anthropic tool_use content blocks."""

    def test_tool_call_becomes_tool_use_block(self) -> None:
        """A tool_calls entry becomes a tool_use content block."""
        response = tool_call_completion()
        result = openai_to_anthropic(response)

        content = result["content"]
        tool_blocks = [b for b in content if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1

        block = tool_blocks[0]
        assert block["name"] == "Read"
        assert block["id"] == "call_abc123"

    def test_tool_call_arguments_parsed(self) -> None:
        """The JSON string arguments are parsed into a dict for tool_use input."""
        response = tool_call_completion()
        result = openai_to_anthropic(response)

        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        assert isinstance(tool_block["input"], dict)
        assert tool_block["input"]["file_path"] == "/home/user/project/main.py"

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool_calls become multiple tool_use content blocks."""
        response = multi_tool_call_completion()
        result = openai_to_anthropic(response)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 2
        names = [b["name"] for b in tool_blocks]
        assert "Read" in names

    def test_tool_calls_with_text_content(self) -> None:
        """When content and tool_calls both exist, text block comes before tool_use blocks."""
        response = multi_tool_call_completion()
        result = openai_to_anthropic(response)

        content = result["content"]
        # First block should be text (from the content field)
        assert content[0]["type"] == "text"
        assert "compare" in content[0]["text"].lower()
        # Remaining blocks should be tool_use
        for block in content[1:]:
            assert block["type"] == "tool_use"

    def test_tool_call_ids_preserved(self) -> None:
        """Tool call IDs from OpenAI map to tool_use block IDs."""
        response = tool_call_completion()
        result = openai_to_anthropic(response)

        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        original_id = response["choices"][0]["message"]["tool_calls"][0]["id"]
        assert tool_block["id"] == original_id


class TestFinishReasonMapping:
    """OpenAI finish_reason -> Anthropic stop_reason."""

    def test_stop_maps_to_end_turn(self) -> None:
        """finish_reason 'stop' becomes stop_reason 'end_turn'."""
        response = simple_completion()
        assert response["choices"][0]["finish_reason"] == "stop"

        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"

    def test_tool_calls_maps_to_tool_use(self) -> None:
        """finish_reason 'tool_calls' becomes stop_reason 'tool_use'."""
        response = tool_call_completion()
        assert response["choices"][0]["finish_reason"] == "tool_calls"

        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "tool_use"

    def test_length_maps_to_max_tokens(self) -> None:
        """finish_reason 'length' becomes stop_reason 'max_tokens'."""
        response = simple_completion()
        response["choices"][0]["finish_reason"] = "length"

        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "max_tokens"

    def test_content_filter_maps_gracefully(self) -> None:
        """finish_reason 'content_filter' should map to a sensible stop_reason."""
        response = simple_completion()
        response["choices"][0]["finish_reason"] = "content_filter"

        result = openai_to_anthropic(response)
        # Should not crash; stop_reason should be present
        assert "stop_reason" in result
        assert result["stop_reason"] is not None


class TestUsageMapping:
    """OpenAI usage fields -> Anthropic usage fields."""

    def test_prompt_tokens_to_input_tokens(self) -> None:
        """prompt_tokens maps to input_tokens."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        assert result["usage"]["input_tokens"] == 12

    def test_completion_tokens_to_output_tokens(self) -> None:
        """completion_tokens maps to output_tokens."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        assert result["usage"]["output_tokens"] == 9

    def test_usage_required_fields_present(self) -> None:
        """Both input_tokens and output_tokens must be present."""
        response = simple_completion()
        result = openai_to_anthropic(response)

        assert "input_tokens" in result["usage"]
        assert "output_tokens" in result["usage"]

    def test_missing_usage_handled(self) -> None:
        """If the OpenAI response has no usage field, provide defaults."""
        response = simple_completion()
        del response["usage"]

        result = openai_to_anthropic(response)
        assert "usage" in result
        assert result["usage"]["input_tokens"] >= 0
        assert result["usage"]["output_tokens"] >= 0


class TestNullContentWithToolCalls:
    """Handle null content when tool_calls are present."""

    def test_null_content_no_text_block(self) -> None:
        """When content is null and tool_calls exist, no text block is produced."""
        response = tool_call_completion()
        assert response["choices"][0]["message"]["content"] is None

        result = openai_to_anthropic(response)
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        # Should either have no text blocks or an empty one — not a block with "None"
        for block in text_blocks:
            assert block["text"] != "None"
            assert block["text"] is not None

    def test_only_tool_use_blocks_when_content_null(self) -> None:
        """With null content and tool_calls, content is purely tool_use blocks."""
        response = tool_call_completion()
        response["choices"][0]["message"]["content"] = None

        result = openai_to_anthropic(response)
        # All blocks should be tool_use
        for block in result["content"]:
            assert block["type"] == "tool_use"


class TestErrorResponseTranslation:
    """OpenAI error responses -> Anthropic error format with agentic standard compliance."""

    def test_rate_limit_error_translation(self) -> None:
        """429 error translates to Anthropic error format with suggestion and _links."""
        error = error_response_429()
        result = translate_response(error, status_code=429)

        assert result["type"] == "error"
        assert "error" in result
        assert result["error"]["type"] == "rate_limit_error"
        assert "message" in result["error"]

    def test_server_error_translation(self) -> None:
        """500 error translates to Anthropic error format."""
        error = error_response_500()
        result = translate_response(error, status_code=500)

        assert result["type"] == "error"
        assert "error" in result

    def test_error_includes_suggestion(self) -> None:
        """Translated errors include a suggestion field (Agentic API Standard Pattern 3)."""
        error = error_response_429()
        result = translate_response(error, status_code=429)

        assert "suggestion" in result["error"] or "suggestion" in result
        # The suggestion should be actionable
        suggestion = result["error"].get("suggestion") or result.get("suggestion", "")
        assert len(suggestion) > 0

    def test_error_includes_links(self) -> None:
        """Translated errors include _links (Agentic API Standard Pattern 2)."""
        error = error_response_500()
        result = translate_response(error, status_code=500)

        # _links should be present at error level or top level
        links = result.get("_links") or result.get("error", {}).get("_links")
        assert links is not None
        assert "retry" in links or "manifest" in links

    def test_bad_request_error_translation(self) -> None:
        """400 error translates to Anthropic invalid_request_error."""
        error = error_response_400()
        result = translate_response(error, status_code=400)

        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"

    def test_error_with_parametrized_variants(
        self, openai_error_variant: tuple[int, dict[str, Any]]
    ) -> None:
        """All error types produce valid Anthropic error responses."""
        status_code, error = openai_error_variant
        result = translate_response(error, status_code=status_code)

        assert result["type"] == "error"
        assert "error" in result
        assert "type" in result["error"]
        assert "message" in result["error"]
