"""Tests for prompt-based tool injection in responses_forward.

Tests that multi-agent models get tools serialized into the system prompt
and the API tools array stripped.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from translation.responses_forward import anthropic_to_responses


SAMPLE_TOOLS = [
    {
        "name": "Read",
        "description": "Reads a file.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Bash",
        "description": "Runs a command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def _make_request(
    model: str = "claude-sonnet-4-20250514",
    tools: list | None = None,
    messages: list | None = None,
    system: str = "You are helpful.",
    stream: bool = False,
) -> dict:
    req = {"model": model, "system": system, "stream": stream}
    if messages is not None:
        req["messages"] = messages
    else:
        req["messages"] = [{"role": "user", "content": "Hello"}]
    if tools is not None:
        req["tools"] = tools
    return req


class TestMultiAgentPromptTools:
    """Tests that multi-agent models use prompt-based tools."""

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_tools_not_in_api_body(self):
        """Multi-agent requests must NOT have 'tools' in the API body."""
        request = _make_request(tools=SAMPLE_TOOLS)
        # Need to reimport to pick up env var for model resolution.
        result = anthropic_to_responses(request)
        assert "tools" not in result

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_tools_injected_into_system_prompt(self):
        """Tool definitions should appear in the system prompt."""
        request = _make_request(tools=SAMPLE_TOOLS)
        result = anthropic_to_responses(request)
        system_msg = result["input"][0]
        assert system_msg["role"] == "system"
        assert "### Read" in system_msg["content"]
        assert "### Bash" in system_msg["content"]
        assert "<tool_call>" in system_msg["content"]

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_no_tools_no_injection(self):
        """Without tools, system prompt should not have tool instructions."""
        request = _make_request(tools=None)
        result = anthropic_to_responses(request)
        system_msg = result["input"][0]
        assert "### Read" not in system_msg["content"]

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_tool_use_in_history_converted_to_text(self):
        """tool_use blocks in message history should become text."""
        messages = [
            {"role": "user", "content": "Read /root/test.py"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Read",
                        "input": {"file_path": "/root/test.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "file contents here",
                    },
                ],
            },
            {"role": "user", "content": "What does it do?"},
        ]
        request = _make_request(tools=SAMPLE_TOOLS, messages=messages)
        result = anthropic_to_responses(request)

        # Find the assistant message that had tool_use -- should be text now.
        input_msgs = result["input"]
        # Skip system message.
        non_system = [m for m in input_msgs if m.get("role") != "system"]
        assert len(non_system) >= 3

        # The assistant message should contain <tool_call> text.
        assistant_msg = non_system[1]
        assert assistant_msg["role"] == "assistant"
        assert "<tool_call>" in assistant_msg["content"]
        assert '"Read"' in assistant_msg["content"]

        # The tool result should contain <tool_result> text.
        tool_result_msg = non_system[2]
        assert "<tool_result" in tool_result_msg["content"]
        assert "file contents here" in tool_result_msg["content"]

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_empty_tools_list_no_injection(self):
        request = _make_request(tools=[])
        result = anthropic_to_responses(request)
        assert "tools" not in result
        system_msg = result["input"][0]
        assert "Tool Calling" not in system_msg["content"]


class TestNonMultiAgentToolsPreserved:
    """Tests that non-multi-agent models still use API-level tools."""

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4-1-fast-reasoning"})
    def test_tools_in_api_body(self):
        """Non-multi-agent models should have tools in the API body."""
        request = _make_request(tools=SAMPLE_TOOLS)
        result = anthropic_to_responses(request)
        assert "tools" in result
        assert len(result["tools"]) == 2

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4-1-fast-reasoning"})
    def test_tools_not_in_system_prompt(self):
        """Non-multi-agent models should NOT have tools in system prompt."""
        request = _make_request(tools=SAMPLE_TOOLS)
        result = anthropic_to_responses(request)
        system_msg = result["input"][0]
        # Tool names may appear in preamble conventions, but not as ### headers.
        assert "### Read" not in system_msg["content"]


class TestPromptToolsRoundTrip:
    """End-to-end tests for prompt-based tool calling translation."""

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_system_prompt_preserved(self):
        """Original system prompt content should be preserved."""
        request = _make_request(
            system="You are a coding assistant.", tools=SAMPLE_TOOLS,
        )
        result = anthropic_to_responses(request)
        system_msg = result["input"][0]
        assert "coding assistant" in system_msg["content"]

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_model_resolved_correctly(self):
        request = _make_request(tools=SAMPLE_TOOLS)
        result = anthropic_to_responses(request)
        assert result["model"] == "grok-4.20-multi-agent"

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_stream_flag_preserved(self):
        request = _make_request(tools=SAMPLE_TOOLS, stream=True)
        result = anthropic_to_responses(request)
        assert result["stream"] is True

    @patch.dict(os.environ, {"GROK_MODEL": "grok-4.20-multi-agent"})
    def test_thinking_blocks_silently_skipped(self):
        """Thinking blocks in history should be skipped, not raise errors."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think..."},
                    {"type": "text", "text": "Hello!"},
                ],
            },
        ]
        request = _make_request(tools=SAMPLE_TOOLS, messages=messages)
        result = anthropic_to_responses(request)
        # Should not raise and should contain the text.
        non_system = [m for m in result["input"] if m.get("role") != "system"]
        assistant = non_system[1]
        assert "Hello!" in assistant["content"]
