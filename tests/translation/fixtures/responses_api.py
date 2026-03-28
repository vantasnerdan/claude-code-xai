"""Sample xAI Responses API payloads for translation testing.

These fixtures represent what xAI/Grok returns via the Responses API,
which is the PRIMARY path since issue #51. The reverse translator
(translation.responses_reverse) converts these into Anthropic Messages
API format for Claude Code.
"""

from typing import Any


def simple_response() -> dict[str, Any]:
    """Basic text response from the Responses API."""
    return {
        "id": "resp_abc123def456",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello! How can I help you today?"},
                ],
            }
        ],
        "model": "grok-4-1-fast-reasoning",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 9,
        },
    }


def function_call_response() -> dict[str, Any]:
    """Response where the model invokes a tool via function_call.

    Responses API format puts function calls as top-level output items
    with type 'function_call', call_id, name, and arguments (JSON string).
    """
    return {
        "id": "resp_tool789xyz",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc123",
                "name": "Read",
                "arguments": '{"file_path": "/home/user/project/main.py"}',
            }
        ],
        "model": "grok-4-1-fast-reasoning",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 25,
        },
    }


def multi_function_call_response() -> dict[str, Any]:
    """Response with text and multiple parallel function calls."""
    return {
        "id": "resp_multi456",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "I'll read both files to compare them."},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_read_old",
                "name": "Read",
                "arguments": '{"file_path": "/home/user/project/old.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call_read_new",
                "name": "Read",
                "arguments": '{"file_path": "/home/user/project/new.py"}',
            },
        ],
        "model": "grok-4-1-fast-reasoning",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 45,
        },
    }


def reasoning_response() -> dict[str, Any]:
    """Response containing a reasoning block (should be stripped)."""
    return {
        "id": "resp_reasoning789",
        "output": [
            {"type": "reasoning", "encrypted_content": "abc123..."},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "The answer is 42."},
                ],
            },
        ],
        "model": "grok-4-1-fast-reasoning",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
        },
    }


def empty_output_response() -> dict[str, Any]:
    """Response with empty output array."""
    return {
        "id": "resp_empty",
        "output": [],
        "model": "grok-4-1-fast-reasoning",
        "usage": {},
    }


def error_response_429() -> dict[str, Any]:
    """Rate limit error from the xAI Responses API."""
    return {
        "error": {
            "message": "Rate limit exceeded. Please retry after 30 seconds.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }


def error_response_500() -> dict[str, Any]:
    """Internal server error from the xAI Responses API."""
    return {
        "error": {
            "message": "Internal server error. The model encountered an unexpected condition.",
            "type": "server_error",
            "code": "internal_error",
        }
    }


def error_response_400() -> dict[str, Any]:
    """Bad request error -- malformed input."""
    return {
        "error": {
            "message": "Invalid value for 'input[0].content': expected string or array.",
            "type": "invalid_request_error",
            "code": "invalid_value",
        }
    }
