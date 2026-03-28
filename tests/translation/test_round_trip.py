"""LEGACY: Tests for round-trip translation fidelity via Chat Completions path.

These tests verify that translating forward (Anthropic -> OpenAI CC) and then
reverse (OpenAI CC -> Anthropic) preserves critical information. Perfect
round-trip fidelity is not expected (some fields are format-specific),
but tool IDs, content, and structural relationships must survive.

This tests the LEGACY Chat Completions round trip. For the PRIMARY
Responses API round trip, see test_responses_forward.py and
test_responses_reverse.py.
"""

import json
from typing import Any

import pytest

from translation.forward import anthropic_to_openai, translate_messages, translate_tools
from translation.reverse import openai_to_anthropic, translate_response

from tests.translation.fixtures.anthropic_messages import (
    simple_text_message,
    tool_use_response,
    tool_result_message,
    multi_turn_with_tools,
    parallel_tool_calls,
    full_request_with_tools,
)
from tests.translation.fixtures.openai_completions import (
    simple_completion,
    tool_call_completion,
    multi_tool_call_completion,
)


class TestToolCallingRoundTrip:
    """Full tool calling cycle through both translators."""

    def test_full_tool_calling_cycle(self, sample_tools: list[dict[str, Any]]) -> None:
        """Define tools -> send request -> get tool_use -> send result -> get response.

        This simulates the complete lifecycle:
        1. Anthropic request with tools -> translated to OpenAI format
        2. OpenAI response with tool_calls -> translated back to Anthropic
        3. Tool result sent back -> translated to OpenAI tool message
        4. Final response -> translated back to Anthropic
        """
        # Step 1: Forward-translate the request
        request = full_request_with_tools()
        openai_request = anthropic_to_openai(request)

        # Verify tools made it through
        assert openai_request.get("tools") is not None
        assert len(openai_request["tools"]) == 6

        # Step 2: Simulate Grok responding with a tool call
        grok_response = tool_call_completion()
        anthropic_response = openai_to_anthropic(grok_response)

        # Verify tool_use block exists
        tool_blocks = [b for b in anthropic_response["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1

        # Step 3: Forward-translate the tool result
        tool_result = tool_result_message()
        tool_result["content"][0]["tool_use_id"] = tool_blocks[0]["id"]
        openai_messages = translate_messages([tool_result])

        # Verify tool message was created
        tool_msgs = [m for m in openai_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == tool_blocks[0]["id"]

        # Step 4: Simulate final text response
        final_response = simple_completion()
        final_anthropic = openai_to_anthropic(final_response)

        assert final_anthropic["content"][0]["type"] == "text"
        assert len(final_anthropic["content"][0]["text"]) > 0


class TestIdPreservation:
    """Tool use IDs survive the forward -> reverse round trip."""

    def test_tool_use_id_survives_forward_translation(self) -> None:
        """tool_use ID from Anthropic format appears in the OpenAI tool_call."""
        msg = tool_use_response()
        original_id = msg["content"][0]["id"]

        result = translate_messages([msg])
        assert result[0]["tool_calls"][0]["id"] == original_id

    def test_tool_call_id_survives_reverse_translation(self) -> None:
        """tool_call ID from OpenAI format appears in the Anthropic tool_use block."""
        response = tool_call_completion()
        original_id = response["choices"][0]["message"]["tool_calls"][0]["id"]

        result = openai_to_anthropic(response)
        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_block["id"] == original_id

    def test_parallel_tool_ids_all_preserved(self) -> None:
        """All IDs from parallel tool calls survive forward translation."""
        msg = parallel_tool_calls()
        original_ids = {block["id"] for block in msg["content"] if block["type"] == "tool_use"}

        result = translate_messages([msg])
        translated_ids = {tc["id"] for tc in result[0]["tool_calls"]}

        assert original_ids == translated_ids

    def test_tool_result_id_matches_through_round_trip(self) -> None:
        """The tool_use_id in a tool_result message matches after forward translation."""
        msg = tool_result_message()
        original_id = msg["content"][0]["tool_use_id"]

        result = translate_messages([msg])
        assert result[0]["tool_call_id"] == original_id


class TestContentPreservation:
    """Content fidelity through translation."""

    def test_text_content_lossless(self) -> None:
        """Plain text survives forward -> reverse round trip."""
        # Forward: Anthropic message -> OpenAI message
        msg = simple_text_message()
        openai_msgs = translate_messages([msg])
        assert openai_msgs[0]["content"] == "Hello"

        # Simulate a response with the same content
        response = simple_completion()
        response["choices"][0]["message"]["content"] = "Hello"
        result = openai_to_anthropic(response)

        assert result["content"][0]["text"] == "Hello"

    def test_tool_arguments_lossless(self) -> None:
        """Tool arguments survive the round trip without data loss."""
        # Forward: Anthropic tool_use -> OpenAI tool_call
        msg = tool_use_response()
        original_input = msg["content"][0]["input"]
        openai_msgs = translate_messages([msg])

        args_json = openai_msgs[0]["tool_calls"][0]["function"]["arguments"]
        forward_args = json.loads(args_json)
        assert forward_args == original_input

        # Reverse: OpenAI tool_call -> Anthropic tool_use
        response = tool_call_completion()
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = args_json
        result = openai_to_anthropic(response)

        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_block["input"] == original_input

    def test_tool_definitions_schema_preserved(self, sample_tools: list[dict[str, Any]]) -> None:
        """Tool definition schemas survive forward translation."""
        result = translate_tools(sample_tools)

        for original, translated in zip(sample_tools, result):
            original_schema = original["input_schema"]
            translated_params = translated["function"]["parameters"]

            # Properties and required fields should match
            assert translated_params["properties"] == original_schema["properties"]
            if "required" in original_schema:
                assert translated_params["required"] == original_schema["required"]


class TestMultiTurnRoundTrip:
    """Multi-turn conversations through the translation layer."""

    def test_conversation_structure_preserved(self) -> None:
        """The structural relationships in a multi-turn conversation survive translation."""
        messages = multi_turn_with_tools()
        result = translate_messages(messages)

        # Find the tool call and its matching result
        assistant_with_tools = None
        tool_response = None
        for msg in result:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                assistant_with_tools = msg
            if msg["role"] == "tool":
                tool_response = msg

        assert assistant_with_tools is not None
        assert tool_response is not None

        # The tool response must reference the tool call
        call_id = assistant_with_tools["tool_calls"][0]["id"]
        assert tool_response["tool_call_id"] == call_id

    def test_message_ordering_preserved(self) -> None:
        """Message order is maintained through translation."""
        messages = multi_turn_with_tools()
        result = translate_messages(messages)

        roles = [m["role"] for m in result]
        # First message is user, conversation flows logically
        assert roles[0] == "user"
        # assistant with tool_calls must come before the tool response
        tool_call_idx = next(
            i for i, m in enumerate(result)
            if m["role"] == "assistant" and "tool_calls" in m
        )
        tool_result_idx = next(i for i, m in enumerate(result) if m["role"] == "tool")
        assert tool_call_idx < tool_result_idx
