"""Sample streaming event payloads for SSE translation testing.

Anthropic and OpenAI use different SSE event formats. These fixtures
provide the raw event data for both sides so the streaming translator
can be validated.

Sections:
- Anthropic SSE events: what Claude Code expects to receive (used by all paths)
- OpenAI SSE events: LEGACY Chat Completions streaming format
  (used by translation.streaming when XAI_USE_CHAT_COMPLETIONS=true)
- For Responses API streaming fixtures, see test_responses_streaming.py
  which builds SSE lines inline using _data_line() helper.
"""

import json
from typing import Any


# ---------------------------------------------------------------------------
# Anthropic SSE events (what Claude Code expects to receive)
# ---------------------------------------------------------------------------


def anthropic_message_start() -> dict[str, Any]:
    """The message_start event that opens an Anthropic streaming response.

    This arrives first and contains the message metadata including id,
    model, role, and usage estimates.
    """
    return {
        "type": "message_start",
        "message": {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 25,
                "output_tokens": 1,
            },
        },
    }


def anthropic_content_block_start() -> dict[str, Any]:
    """Content block start event — signals a new text block is beginning."""
    return {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": "",
        },
    }


def anthropic_content_block_delta() -> dict[str, Any]:
    """Text delta event — an incremental text chunk within a content block."""
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "Hello! How can I",
        },
    }


def anthropic_tool_use_start() -> dict[str, Any]:
    """Content block start for a tool_use block."""
    return {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lhB",
            "name": "Read",
            "input": {},
        },
    }


def anthropic_tool_use_delta() -> dict[str, Any]:
    """Input JSON delta for a tool_use content block.

    Tool input is streamed as partial JSON strings that must be
    concatenated and parsed when the block ends.
    """
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": '{"file_path": "/home/user/proj',
        },
    }


def anthropic_content_block_stop() -> dict[str, Any]:
    """Content block stop event — signals the current block is complete."""
    return {
        "type": "content_block_stop",
        "index": 0,
    }


def anthropic_message_delta() -> dict[str, Any]:
    """Message delta with stop_reason — arrives near the end of the stream."""
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn",
            "stop_sequence": None,
        },
        "usage": {
            "output_tokens": 15,
        },
    }


def anthropic_message_stop() -> dict[str, Any]:
    """The final message_stop event that closes the stream."""
    return {
        "type": "message_stop",
    }


def anthropic_full_text_stream() -> list[dict[str, Any]]:
    """Complete sequence of Anthropic events for a simple text response."""
    return [
        anthropic_message_start(),
        anthropic_content_block_start(),
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello! "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "How can I help?"},
        },
        anthropic_content_block_stop(),
        anthropic_message_delta(),
        anthropic_message_stop(),
    ]


# ---------------------------------------------------------------------------
# LEGACY: OpenAI Chat Completions SSE data lines
# Used by translation.streaming (Chat Completions streaming adapter).
# The primary path now uses Responses API streaming (see test_responses_streaming.py).
# ---------------------------------------------------------------------------


def openai_stream_chunks() -> list[str]:
    """Complete list of OpenAI SSE data lines for a streamed text response.

    Each string is the content after 'data: ' in the SSE stream.
    The final '[DONE]' sentinel terminates the stream.
    """
    chunks = [
        {
            "id": "chatcmpl-stream001",
            "object": "chat.completion.chunk",
            "created": 1709000300,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream001",
            "object": "chat.completion.chunk",
            "created": 1709000300,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello! "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream001",
            "object": "chat.completion.chunk",
            "created": 1709000300,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "How can I help?"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream001",
            "object": "chat.completion.chunk",
            "created": 1709000300,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
            },
        },
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return lines


def openai_tool_call_stream_chunks() -> list[str]:
    """OpenAI SSE data lines for a streamed tool call response.

    Tool calls stream in parts: first chunk has the function name and
    the start of arguments; subsequent chunks append argument fragments.
    """
    chunks = [
        {
            "id": "chatcmpl-toolstream001",
            "object": "chat.completion.chunk",
            "created": 1709000400,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream_read",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": "",
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-toolstream001",
            "object": "chat.completion.chunk",
            "created": 1709000400,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": '{"file_path":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-toolstream001",
            "object": "chat.completion.chunk",
            "created": 1709000400,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": ' "/home/user/main.py"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-toolstream001",
            "object": "chat.completion.chunk",
            "created": 1709000400,
            "model": "grok-4-1-fast-reasoning",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return lines
