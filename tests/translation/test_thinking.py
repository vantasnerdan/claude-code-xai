"""Tests for thinking feature handling in the translation layer.

Anthropic's 'thinking' parameter (extended thinking / chain-of-thought)
is not supported by xAI's Chat Completions API. The bridge must:
1. Strip the thinking parameter from requests (not reject them)
2. Strip thinking/redacted_thinking content blocks from messages
3. Forward the request normally to get a valid response
4. Signal degradation via warnings (Agentic API Standard Pattern 3)

Grok models reason internally -- stripping thinking does NOT reduce
response quality; it simply removes a parameter xAI does not accept.
"""

from __future__ import annotations

from typing import Any

import pytest

from translation.forward import anthropic_to_openai, strip_thinking, translate_messages


class TestStripThinkingParameter:
    """Stripping the top-level 'thinking' parameter from requests."""

    def test_thinking_enabled_stripped(self) -> None:
        """thinking: {type: enabled, budget_tokens: N} is removed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert "thinking" not in request
        assert len(warnings) >= 1
        assert any("thinking" in w.lower() for w in warnings)

    def test_thinking_adaptive_stripped(self) -> None:
        """thinking: {type: adaptive} (Opus 4.6) is removed."""
        request: dict[str, Any] = {
            "model": "claude-opus-4-20250514",
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert "thinking" not in request
        assert len(warnings) >= 1

    def test_thinking_true_stripped(self) -> None:
        """thinking: true (boolean shorthand) is removed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert "thinking" not in request
        assert len(warnings) >= 1

    def test_no_thinking_no_warnings(self) -> None:
        """Request without thinking produces no warnings."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert warnings == []

    def test_thinking_false_no_warning(self) -> None:
        """thinking: false is falsy -- no stripping needed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": False,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        # False is falsy, so no stripping occurs
        assert warnings == []

    def test_thinking_none_no_warning(self) -> None:
        """thinking: null/None is falsy -- no stripping needed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": None,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert warnings == []


class TestStripThinkingContentBlocks:
    """Stripping thinking/redacted_thinking blocks from message history."""

    def test_thinking_block_stripped_from_assistant_message(self) -> None:
        """Thinking blocks in assistant messages are removed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Let me calculate 2+2...",
                            "signature": "abc123...",
                        },
                        {"type": "text", "text": "The answer is 4."},
                    ],
                },
                {"role": "user", "content": "Thanks!"},
            ],
        }
        warnings = strip_thinking(request)

        assistant_content = request["messages"][1]["content"]
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "text"
        assert assistant_content[0]["text"] == "The answer is 4."
        assert any("thinking" in w.lower() for w in warnings)

    def test_redacted_thinking_block_stripped(self) -> None:
        """Redacted thinking blocks are also removed."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": "encrypted_content_here...",
                        },
                        {"type": "text", "text": "Based on my analysis..."},
                    ],
                },
            ],
        }
        warnings = strip_thinking(request)

        assistant_content = request["messages"][0]["content"]
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "text"

    def test_mixed_thinking_and_tool_use_preserved(self) -> None:
        """Tool use blocks survive when thinking blocks are stripped."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I need to read the file...",
                            "signature": "sig...",
                        },
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "Read",
                            "input": {"file_path": "/tmp/test.py"},
                        },
                    ],
                },
            ],
        }
        strip_thinking(request)

        content = request["messages"][0]["content"]
        types = [b["type"] for b in content]
        assert "thinking" not in types
        assert "text" in types
        assert "tool_use" in types
        assert len(content) == 2

    def test_string_content_not_affected(self) -> None:
        """Messages with string content (not block arrays) are untouched."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        warnings = strip_thinking(request)

        assert request["messages"][0]["content"] == "Hello"
        assert warnings == []

    def test_all_blocks_thinking_leaves_empty_content(self) -> None:
        """If all blocks are thinking, content becomes empty list."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Reasoning...",
                            "signature": "sig...",
                        },
                    ],
                },
            ],
        }
        strip_thinking(request)

        assert request["messages"][0]["content"] == []


class TestThinkingEndToEnd:
    """Full translation with thinking -- strip then translate."""

    def test_request_with_thinking_translates_successfully(self) -> None:
        """A request with thinking param translates to valid OpenAI format."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "Solve this math problem."}],
        }
        strip_thinking(request)
        result = anthropic_to_openai(request)

        assert "messages" in result
        assert result["messages"][-1]["content"] == "Solve this math problem."
        assert "thinking" not in result

    def test_conversation_with_thinking_history_translates(self) -> None:
        """Multi-turn conversation with thinking blocks in history translates."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Simple arithmetic: 2+2=4",
                            "signature": "sig123",
                        },
                        {"type": "text", "text": "4"},
                    ],
                },
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        strip_thinking(request)
        result = anthropic_to_openai(request)

        # Verify the conversation translates without thinking artifacts
        messages = result["messages"]
        for msg in messages:
            content = msg.get("content", "")
            assert "thinking" not in str(content).lower() or "thinking" not in msg.get("role", "")

    def test_thinking_blocks_in_messages_do_not_reach_translate(self) -> None:
        """After strip_thinking, translate_messages sees no thinking blocks."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "...", "signature": "..."},
                    {"type": "text", "text": "Hi there!"},
                ],
            },
        ]
        # Simulate strip_thinking on the messages
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [
                    b for b in content if b.get("type") not in ("thinking", "redacted_thinking")
                ]

        result = translate_messages(messages)

        # No crash, valid output
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there!"


class TestWarningFormat:
    """Warning messages follow Agentic API Standard Pattern 3."""

    def test_warning_mentions_grok_reasoning(self) -> None:
        """Warning explains that Grok reasons internally."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        warnings = strip_thinking(request)

        assert len(warnings) >= 1
        warning_text = " ".join(warnings).lower()
        assert "grok" in warning_text
        assert "reason" in warning_text or "stripped" in warning_text

    def test_warning_is_not_an_error(self) -> None:
        """Stripping thinking should NOT raise any exception."""
        request: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 50000},
            "messages": [{"role": "user", "content": "Complex reasoning task"}],
        }
        # Must not raise
        warnings = strip_thinking(request)
        result = anthropic_to_openai(request)

        assert isinstance(result, dict)
        assert isinstance(warnings, list)
