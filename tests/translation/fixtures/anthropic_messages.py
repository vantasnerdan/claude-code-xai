"""Sample Anthropic Messages API payloads for translation testing.

These fixtures represent real Claude Code traffic patterns. Every payload
matches the Anthropic Messages API specification so the translation layer
can be validated against ground truth.
"""

from typing import Any


def simple_text_message() -> dict[str, Any]:
    """A single user message with a text content block."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }


def system_message_request() -> dict[str, Any]:
    """Full request with top-level system field (Anthropic convention).

    In the Anthropic API, system is a top-level field, NOT a message role.
    The forward translator must extract it and convert to an OpenAI system
    role message.
    """
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are a helpful coding assistant. Follow instructions exactly.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain how Python decorators work."}
                ],
            }
        ],
    }


def tool_use_response() -> dict[str, Any]:
    """Assistant response containing a tool_use content block.

    This is what Claude returns when it wants to call a tool. The content
    array has a tool_use block with id, name, and input (the arguments).
    """
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lhB",
                "name": "Read",
                "input": {
                    "file_path": "/home/user/project/src/main.py",
                    "offset": 0,
                    "limit": 200,
                },
            }
        ],
    }


def tool_result_message() -> dict[str, Any]:
    """User message with a tool_result content block.

    After the tool executes, the result goes back as a user message with
    a tool_result block referencing the original tool_use_id.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A09q90qw90lq917835lhB",
                "content": [
                    {
                        "type": "text",
                        "text": '     1\tfrom fastapi import FastAPI\n     2\t\n     3\tapp = FastAPI(title="my-app")\n',
                    }
                ],
            }
        ],
    }


def multi_turn_with_tools() -> list[dict[str, Any]]:
    """Full multi-turn conversation: user -> assistant(tool_use) -> user(tool_result) -> assistant(text).

    This is the canonical Claude Code flow: user asks a question, Claude
    reads a file, receives the result, then responds with analysis.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What does the main function do?"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read the file to find out."},
                {
                    "type": "tool_use",
                    "id": "toolu_01XYZ123abc456def789gh",
                    "name": "Read",
                    "input": {"file_path": "/home/user/project/main.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01XYZ123abc456def789gh",
                    "content": [
                        {
                            "type": "text",
                            "text": 'def main():\n    print("hello world")\n',
                        }
                    ],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": 'The main function simply prints "hello world" to stdout.',
                }
            ],
        },
    ]


def parallel_tool_calls() -> dict[str, Any]:
    """Assistant message with multiple tool_use blocks (parallel tool calls).

    Claude Code sometimes issues multiple tool calls in a single response,
    e.g., reading multiple files or running a command while reading a file.
    """
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll read both files to compare them.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01AAA111bbb222ccc333ddd",
                "name": "Read",
                "input": {"file_path": "/home/user/project/old.py"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01EEE555fff666ggg777hhh",
                "name": "Read",
                "input": {"file_path": "/home/user/project/new.py"},
            },
        ],
    }


def full_request_with_tools() -> dict[str, Any]:
    """Complete Anthropic Messages API request body with tools defined.

    This is the full shape of what Claude Code sends to the API, including
    model, max_tokens, system, messages, and tool definitions with
    input_schema (JSON Schema for each tool's parameters).
    """
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "system": "You are an expert coding assistant with access to file system tools.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the README and summarize it."}
                ],
            }
        ],
        "tools": [
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
        ],
    }
