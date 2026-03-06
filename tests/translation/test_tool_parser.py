"""Tests for tool call parsing from model text output.

Tests the extraction of <tool_call> blocks from text and conversion
to Anthropic tool_use content blocks.
"""

from __future__ import annotations

import json

from translation.tool_parser import (
    parse_tool_calls_from_text,
    has_pending_tool_call,
)


class TestParseToolCallsFromText:
    """Tests for parse_tool_calls_from_text."""

    def test_no_tool_calls_returns_text(self):
        text = "Hello, this is a regular response."
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == text
        assert tools == []

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/root/test.py"}}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["type"] == "tool_use"
        assert tools[0]["name"] == "Read"
        assert tools[0]["input"]["file_path"] == "/root/test.py"
        assert tools[0]["id"].startswith("toolu_")

    def test_tool_call_with_surrounding_text(self):
        text = (
            "Let me read that file.\n\n"
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/root/test.py"}}\n</tool_call>\n\n'
            "I found the following content."
        )
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1

        # Should have text before, tool call, text after.
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) == 2
        assert len(tool_blocks) == 1
        assert "read that file" in text_blocks[0]["text"]
        assert "found the following" in text_blocks[1]["text"]

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/a.py"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/b.py"}}\n</tool_call>'
        )
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 2
        assert tools[0]["name"] == "Read"
        assert tools[0]["input"]["file_path"] == "/a.py"
        assert tools[1]["input"]["file_path"] == "/b.py"
        # Each should have a unique ID.
        assert tools[0]["id"] != tools[1]["id"]

    def test_tool_call_with_nested_json(self):
        params = {"command": 'echo "hello world"', "timeout": 30000}
        payload = json.dumps({"name": "Bash", "parameters": params})
        text = f"<tool_call>\n{payload}\n</tool_call>"
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["input"]["command"] == 'echo "hello world"'
        assert tools[0]["input"]["timeout"] == 30000

    def test_tool_call_with_complex_nested_params(self):
        params = {
            "file_path": "/root/test.py",
            "old_string": "def foo():\n    pass",
            "new_string": "def foo():\n    return 42",
        }
        payload = json.dumps({"name": "Edit", "parameters": params})
        text = f"<tool_call>\n{payload}\n</tool_call>"
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["name"] == "Edit"
        assert "def foo():" in tools[0]["input"]["old_string"]

    def test_empty_parameters(self):
        text = '<tool_call>\n{"name": "Read", "parameters": {}}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["input"] == {}

    def test_no_parameters_key(self):
        text = '<tool_call>\n{"name": "Read"}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["input"] == {}

    def test_malformed_json_skipped(self):
        text = '<tool_call>\n{not valid json}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert tools == []
        # The malformed block is silently skipped.

    def test_missing_name_skipped(self):
        text = '<tool_call>\n{"parameters": {"file_path": "/root/test.py"}}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert tools == []

    def test_non_dict_payload_skipped(self):
        text = '<tool_call>\n"just a string"\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert tools == []

    def test_mixed_valid_and_invalid(self):
        text = (
            '<tool_call>\n{"name": "Read", "parameters": {"file_path": "/a.py"}}\n</tool_call>\n'
            '<tool_call>\n{bad json}\n</tool_call>\n'
            '<tool_call>\n{"name": "Bash", "parameters": {"command": "ls"}}\n</tool_call>'
        )
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 2
        assert tools[0]["name"] == "Read"
        assert tools[1]["name"] == "Bash"

    def test_tool_call_single_line(self):
        text = '<tool_call>{"name": "Read", "parameters": {"file_path": "/test"}}</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["name"] == "Read"

    def test_empty_text_input(self):
        blocks, tools = parse_tool_calls_from_text("")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == ""
        assert tools == []

    def test_parameters_not_dict_treated_as_empty(self):
        text = '<tool_call>\n{"name": "Read", "parameters": "not a dict"}\n</tool_call>'
        blocks, tools = parse_tool_calls_from_text(text)
        assert len(tools) == 1
        assert tools[0]["input"] == {}

    def test_preserves_ordering(self):
        text = (
            "First text.\n"
            '<tool_call>\n{"name": "A", "parameters": {}}\n</tool_call>\n'
            "Middle text.\n"
            '<tool_call>\n{"name": "B", "parameters": {}}\n</tool_call>\n'
            "Last text."
        )
        blocks, tools = parse_tool_calls_from_text(text)
        types = [b["type"] for b in blocks]
        assert types == ["text", "tool_use", "text", "tool_use", "text"]


class TestHasPendingToolCall:
    """Tests for has_pending_tool_call."""

    def test_no_tags(self):
        assert not has_pending_tool_call("hello world")

    def test_complete_tag(self):
        assert not has_pending_tool_call('<tool_call>{"name": "x"}</tool_call>')

    def test_open_tag_only(self):
        assert has_pending_tool_call('<tool_call>{"name": "')

    def test_multiple_complete(self):
        text = '<tool_call>a</tool_call><tool_call>b</tool_call>'
        assert not has_pending_tool_call(text)

    def test_one_complete_one_open(self):
        text = '<tool_call>a</tool_call><tool_call>b'
        assert has_pending_tool_call(text)

    def test_empty_string(self):
        assert not has_pending_tool_call("")
