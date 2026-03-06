"""Tests for prompt-based tool serialization.

Tests the serialization of Anthropic tool definitions into system prompt
text for multi-agent models that don't support client-side tools.
"""

from __future__ import annotations

import json

from translation.prompt_tools import (
    serialize_tools_to_prompt,
    serialize_tool_results_to_text,
)


# --- Fixtures: sample Anthropic tool definitions ---

SAMPLE_READ_TOOL = {
    "name": "Read",
    "description": "Reads a file from the local filesystem.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "number", "description": "Line number to start from"},
            "limit": {"type": "number", "description": "Number of lines to read"},
        },
        "required": ["file_path"],
    },
}

SAMPLE_BASH_TOOL = {
    "name": "Bash",
    "description": "Executes a bash command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "number", "description": "Timeout in milliseconds"},
        },
        "required": ["command"],
    },
}

SAMPLE_EDIT_TOOL = {
    "name": "Edit",
    "description": "Performs exact string replacements in files.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}


class TestSerializeToolsToPrompt:
    """Tests for serialize_tools_to_prompt."""

    def test_empty_tools_returns_empty(self):
        result = serialize_tools_to_prompt([])
        assert result == ""

    def test_single_tool_contains_name(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert "### Read" in result

    def test_single_tool_contains_description(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert "Reads a file from the local filesystem." in result

    def test_single_tool_contains_schema(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert '"file_path"' in result
        assert '"required"' in result

    def test_multiple_tools_all_present(self):
        tools = [SAMPLE_READ_TOOL, SAMPLE_BASH_TOOL, SAMPLE_EDIT_TOOL]
        result = serialize_tools_to_prompt(tools)
        assert "### Read" in result
        assert "### Bash" in result
        assert "### Edit" in result

    def test_contains_tool_call_instructions(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        assert "Tool Calling" in result

    def test_contains_format_example(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert '"name":' in result
        assert '"parameters":' in result

    def test_schema_is_valid_json(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        # Extract the JSON code block for the schema.
        lines = result.split("\n")
        in_json = False
        json_lines = []
        for line in lines:
            if line.strip() == "```json":
                in_json = True
                continue
            elif line.strip() == "```" and in_json:
                break
            elif in_json:
                json_lines.append(line)
        if json_lines:
            schema = json.loads("\n".join(json_lines))
            assert schema["type"] == "object"
            assert "file_path" in schema["properties"]

    def test_tool_without_description(self):
        tool = {"name": "Minimal", "input_schema": {"type": "object"}}
        result = serialize_tools_to_prompt([tool])
        assert "### Minimal" in result
        assert "No description." in result

    def test_tool_without_schema(self):
        tool = {"name": "NoSchema", "description": "A tool."}
        result = serialize_tools_to_prompt([tool])
        assert "### NoSchema" in result
        assert "A tool." in result

    def test_multiple_tool_calls_instruction(self):
        result = serialize_tools_to_prompt([SAMPLE_READ_TOOL])
        assert "multiple tools" in result.lower() or "multiple <tool_call>" in result.lower()


class TestSerializeToolResultsToText:
    """Tests for serialize_tool_results_to_text."""

    def test_empty_results_returns_empty(self):
        result = serialize_tool_results_to_text([])
        assert result == ""

    def test_single_string_result(self):
        results = [{"tool_use_id": "toolu_abc", "name": "Read", "content": "file contents here"}]
        result = serialize_tool_results_to_text(results)
        assert '<tool_result name="Read" id="toolu_abc">' in result
        assert "file contents here" in result
        assert "</tool_result>" in result

    def test_result_without_name(self):
        results = [{"tool_use_id": "toolu_xyz", "content": "output"}]
        result = serialize_tool_results_to_text(results)
        assert '<tool_result id="toolu_xyz">' in result
        assert "output" in result

    def test_nested_content_blocks(self):
        results = [{
            "tool_use_id": "toolu_123",
            "name": "Bash",
            "content": [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ],
        }]
        result = serialize_tool_results_to_text(results)
        assert "line 1" in result
        assert "line 2" in result

    def test_multiple_results(self):
        results = [
            {"tool_use_id": "toolu_1", "name": "Read", "content": "content1"},
            {"tool_use_id": "toolu_2", "name": "Bash", "content": "content2"},
        ]
        result = serialize_tool_results_to_text(results)
        assert "toolu_1" in result
        assert "toolu_2" in result
        assert "content1" in result
        assert "content2" in result

    def test_result_with_string_content_in_list(self):
        results = [{
            "tool_use_id": "toolu_str",
            "content": ["plain string item"],
        }]
        result = serialize_tool_results_to_text(results)
        assert "plain string item" in result
