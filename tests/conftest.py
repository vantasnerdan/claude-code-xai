"""Root conftest.py — shared pytest fixtures for all test modules.

Provides foundational fixtures used across the entire test suite:
tool definitions, sample requests, and sample responses.
"""

import pytest
from typing import Any


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Claude Code tool definitions with name, description, and input_schema.

    These mirror the actual tool schemas that Claude Code sends in the
    Anthropic Messages API. They are the contract that the translation
    layer must preserve.
    """
    return [
        {
            "name": "Read",
            "description": "Reads a file from the local filesystem.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "number",
                        "description": "The line number to start reading from.",
                    },
                    "limit": {
                        "type": "number",
                        "description": "The number of lines to read.",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences.",
                        "default": False,
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
        {
            "name": "Write",
            "description": "Writes a file to the local filesystem.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "Bash",
            "description": "Executes a given bash command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds.",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "Grep",
            "description": "Searches file contents using ripgrep.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files.",
                    },
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "Glob",
            "description": "Fast file pattern matching tool.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against.",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in.",
                    },
                },
                "required": ["pattern"],
            },
        },
    ]


@pytest.fixture
def anthropic_simple_request() -> dict[str, Any]:
    """A basic Anthropic Messages API request body.

    Minimal valid request: model, max_tokens, and a single user message.
    """
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"}
                ],
            }
        ],
    }


@pytest.fixture
def responses_api_simple_response() -> dict[str, Any]:
    """A basic xAI Responses API response.

    Standard non-streaming response with output array and usage stats.
    This is the PRIMARY format since issue #51.
    """
    return {
        "id": "resp_test123",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "I'm doing well, thank you! How can I help you?"},
                ],
            }
        ],
        "model": "grok-4-1-fast-reasoning",
        "usage": {
            "input_tokens": 15,
            "output_tokens": 12,
        },
    }


@pytest.fixture
def openai_simple_response() -> dict[str, Any]:
    """LEGACY: OpenAI Chat Completions response format.

    Retained for testing the legacy translation.reverse module.
    The primary path now uses Responses API format (see responses_api_simple_response).
    """
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1709000000,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 12,
            "total_tokens": 27,
        },
    }
