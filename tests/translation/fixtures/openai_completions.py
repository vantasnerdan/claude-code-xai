"""LEGACY: Sample OpenAI Chat Completions API payloads for translation testing.

These fixtures represent the LEGACY Chat Completions format that xAI/Grok
used to return before issue #51 migrated to the Responses API. They are
retained for testing the legacy translation.reverse and translation.streaming
modules which still handle this format when XAI_USE_CHAT_COMPLETIONS=true.

For PRIMARY path fixtures, see tests/translation/fixtures/responses_api.py.
"""

from typing import Any


def simple_completion() -> dict[str, Any]:
    """Basic assistant text response with stop finish reason."""
    return {
        "id": "chatcmpl-abc123def456",
        "object": "chat.completion",
        "created": 1709000000,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21,
        },
    }


def tool_call_completion() -> dict[str, Any]:
    """Response where the model invokes a tool via tool_calls.

    OpenAI format puts tool calls in the message object alongside
    (often null) content. Each tool_call has an id, type, and function
    object with name and arguments (JSON string).
    """
    return {
        "id": "chatcmpl-tool789xyz",
        "object": "chat.completion",
        "created": 1709000100,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path": "/home/user/project/main.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 25,
            "total_tokens": 175,
        },
    }


def multi_tool_call_completion() -> dict[str, Any]:
    """Response with multiple parallel tool calls."""
    return {
        "id": "chatcmpl-multi456",
        "object": "chat.completion",
        "created": 1709000200,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll read both files to compare them.",
                    "tool_calls": [
                        {
                            "id": "call_read_old",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path": "/home/user/project/old.py"}',
                            },
                        },
                        {
                            "id": "call_read_new",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path": "/home/user/project/new.py"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 45,
            "total_tokens": 245,
        },
    }


def streaming_chunk() -> dict[str, Any]:
    """Single SSE data chunk from a streaming response.

    In OpenAI streaming, each chunk has a delta instead of a full message.
    The delta contains incremental content.
    """
    return {
        "id": "chatcmpl-stream001",
        "object": "chat.completion.chunk",
        "created": 1709000300,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
    }


def streaming_chunk_with_role() -> dict[str, Any]:
    """First chunk in a stream — includes the role field in delta."""
    return {
        "id": "chatcmpl-stream001",
        "object": "chat.completion.chunk",
        "created": 1709000300,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                },
                "finish_reason": None,
            }
        ],
    }


def streaming_chunk_tool_call() -> dict[str, Any]:
    """Streaming chunk containing a tool call delta.

    Tool calls stream incrementally: first chunk has the function name,
    subsequent chunks append to arguments.
    """
    return {
        "id": "chatcmpl-stream002",
        "object": "chat.completion.chunk",
        "created": 1709000310,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_stream_abc",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_',
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def streaming_chunk_finish() -> dict[str, Any]:
    """Final chunk in a stream with finish_reason set."""
    return {
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
    }


def error_response_429() -> dict[str, Any]:
    """Rate limit error from the xAI API."""
    return {
        "error": {
            "message": "Rate limit exceeded. Please retry after 30 seconds.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }


def error_response_500() -> dict[str, Any]:
    """Internal server error from the xAI API."""
    return {
        "error": {
            "message": "Internal server error. The model encountered an unexpected condition.",
            "type": "server_error",
            "code": "internal_error",
        }
    }


def error_response_400() -> dict[str, Any]:
    """Bad request error — malformed input."""
    return {
        "error": {
            "message": "Invalid value for 'messages[0].content': expected string or array.",
            "type": "invalid_request_error",
            "code": "invalid_value",
        }
    }
