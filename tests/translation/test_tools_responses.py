"""Tests for translate_tools_responses() and enrich_tools() in tools.py.

Validates that the Responses API tool translation produces flat format
directly from Anthropic input, without the intermediate Chat Completions
hop that existed before issue #53 consolidation.
"""

from typing import Any

from translation.tools import (
    enrich_tools,
    translate_tools,
    translate_tools_responses,
)


class TestEnrichTools:
    """Tests for the shared enrichment core."""

    def test_empty_tools(self):
        assert enrich_tools([]) == []

    def test_preserves_name_description_schema(self):
        tools = [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        result = enrich_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "Read"
        assert result[0]["description"] == "Read a file"
        assert result[0]["input_schema"] == {"type": "object", "properties": {"path": {"type": "string"}}}


class TestTranslateToolsResponses:
    """Tests for Responses API tool formatting."""

    def test_flat_format(self):
        tools = [
            {
                "name": "Bash",
                "description": "Run a command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]
        result = translate_tools_responses(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert t["name"] == "Bash"
        assert t["description"] == "Run a command"
        assert t["parameters"] == {"type": "object", "properties": {"command": {"type": "string"}}}
        # Flat format -- no nested 'function' key.
        assert "function" not in t

    def test_multiple_tools(self):
        tools = [
            {"name": "Read", "description": "Read", "input_schema": {}},
            {"name": "Write", "description": "Write", "input_schema": {}},
        ]
        result = translate_tools_responses(tools)
        assert len(result) == 2
        assert result[0]["name"] == "Read"
        assert result[1]["name"] == "Write"

    def test_empty_tools(self):
        assert translate_tools_responses([]) == []


class TestTranslateToolsChatCompletions:
    """Tests for Chat Completions tool formatting (existing path)."""

    def test_nested_format(self):
        tools = [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        result = translate_tools(tools)
        assert len(result) == 1
        t = result[0]
        assert t["type"] == "function"
        assert "function" in t
        assert t["function"]["name"] == "Read"
        assert t["function"]["description"] == "Read a file"
        assert t["function"]["parameters"] == {"type": "object", "properties": {"path": {"type": "string"}}}


class TestFormatConsistency:
    """Tests that both format paths produce equivalent data from the same input."""

    def test_same_enrichment_different_format(self):
        tools = [
            {
                "name": "Grep",
                "description": "Search files",
                "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}},
            }
        ]
        chat_result = translate_tools(tools)
        responses_result = translate_tools_responses(tools)

        # Both should carry the same semantic content.
        chat_func = chat_result[0]["function"]
        resp_tool = responses_result[0]

        assert chat_func["name"] == resp_tool["name"]
        assert chat_func["description"] == resp_tool["description"]
        assert chat_func["parameters"] == resp_tool["parameters"]

        # But in different structures.
        assert "function" in chat_result[0]
        assert "function" not in responses_result[0]
