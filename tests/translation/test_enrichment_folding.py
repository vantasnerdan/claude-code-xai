"""Tests for enrichment metadata folding into tool descriptions.

Verifies that enrichment fields added by the engine (behavioral_what,
behavioral_why, behavioral_when, _links, _error_format, _near_miss,
_quality, _anti_patterns, outputSchema) are serialized into the
description field so the guest model receives them via the OpenAI format.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from translation.enrichment_folding import (
    fold_enrichment_into_description,
    _remove_remaining_enrichment_fields,
)


def _make_tool(
    name: str = "Read",
    description: str = "Reads a file.",
    **extra: Any,
) -> dict[str, Any]:
    """Create a minimal tool dict with optional enrichment fields."""
    tool: dict[str, Any] = {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": {}},
    }
    tool.update(extra)
    return tool


class TestBehavioralWhat:
    """behavioral_what is folded into description."""

    def test_what_appended_to_description(self) -> None:
        tool = _make_tool(behavioral_what="Enhanced: reads files from disk.")
        fold_enrichment_into_description([tool])
        assert "Enhanced: reads files from disk." in tool["description"]
        assert "[Enhanced Description]" in tool["description"]

    def test_original_description_preserved(self) -> None:
        tool = _make_tool(behavioral_what="Enhanced version.")
        fold_enrichment_into_description([tool])
        assert tool["description"].startswith("Reads a file.")

    def test_what_field_removed_after_folding(self) -> None:
        tool = _make_tool(behavioral_what="Enhanced version.")
        fold_enrichment_into_description([tool])
        assert "behavioral_what" not in tool


class TestBehavioralWhy:
    """behavioral_why is folded into description."""

    def test_why_dict_folded(self) -> None:
        why = {
            "problem_context": "Need to inspect file contents.",
            "failure_modes": ["File not found", "Permission denied"],
        }
        tool = _make_tool(behavioral_why=why)
        fold_enrichment_into_description([tool])
        assert "[Why Use This Tool]" in tool["description"]
        assert "Need to inspect file contents." in tool["description"]
        assert "File not found" in tool["description"]
        assert "Permission denied" in tool["description"]

    def test_why_string_folded(self) -> None:
        tool = _make_tool(behavioral_why="Use when you need file content.")
        fold_enrichment_into_description([tool])
        assert "Use when you need file content." in tool["description"]

    def test_why_field_removed(self) -> None:
        tool = _make_tool(behavioral_why={"problem_context": "test"})
        fold_enrichment_into_description([tool])
        assert "behavioral_why" not in tool


class TestBehavioralWhen:
    """behavioral_when is folded into description."""

    def test_when_prerequisites_folded(self) -> None:
        when = {"prerequisites": ["Read"], "use_before": ["Edit"]}
        tool = _make_tool(behavioral_when=when)
        fold_enrichment_into_description([tool])
        assert "[When To Use]" in tool["description"]
        assert "Prerequisites: Read" in tool["description"]
        assert "Use before: Edit" in tool["description"]

    def test_when_do_not_use_for(self) -> None:
        when = {"do_not_use_for": ["finding files", "searching content"]}
        tool = _make_tool(behavioral_when=when)
        fold_enrichment_into_description([tool])
        assert "Do NOT use for: finding files" in tool["description"]
        assert "Do NOT use for: searching content" in tool["description"]

    def test_when_sequencing(self) -> None:
        when = {"sequencing": "Always Read before Edit."}
        tool = _make_tool(behavioral_when=when)
        fold_enrichment_into_description([tool])
        assert "Sequencing: Always Read before Edit." in tool["description"]

    def test_when_field_removed(self) -> None:
        tool = _make_tool(behavioral_when={"prerequisites": ["Read"]})
        fold_enrichment_into_description([tool])
        assert "behavioral_when" not in tool


class TestLinks:
    """_links (HATEOAS) is folded into description."""

    def test_links_related_folded(self) -> None:
        links = {"related": ["Edit", "Write"]}
        tool = _make_tool(_links=links)
        fold_enrichment_into_description([tool])
        assert "[Navigation]" in tool["description"]
        assert "Related tools: Edit, Write" in tool["description"]

    def test_links_on_error_folded(self) -> None:
        links = {"on_error": {"file_not_found": "Use Glob first."}}
        tool = _make_tool(_links=links)
        fold_enrichment_into_description([tool])
        assert "On file_not_found: Use Glob first." in tool["description"]

    def test_links_field_removed(self) -> None:
        tool = _make_tool(_links={"related": ["Edit"]})
        fold_enrichment_into_description([tool])
        assert "_links" not in tool


class TestErrorFormat:
    """_error_format is folded into description."""

    def test_error_format_folded(self) -> None:
        error_fmt = {
            "errors": [
                {"error": "File not found", "suggestion": "Use Glob"},
            ],
            "format": {"error": "string", "suggestion": "string"},
        }
        tool = _make_tool(_error_format=error_fmt)
        fold_enrichment_into_description([tool])
        assert "[Error Handling]" in tool["description"]
        assert "File not found: Use Glob" in tool["description"]

    def test_error_format_field_removed(self) -> None:
        tool = _make_tool(_error_format={"errors": []})
        fold_enrichment_into_description([tool])
        assert "_error_format" not in tool


class TestNearMiss:
    """_near_miss is folded into description."""

    def test_near_miss_aliases_folded(self) -> None:
        near_miss = {"aliases": ["cat", "view"], "commonly_confused_with": ["Bash cat"]}
        tool = _make_tool(_near_miss=near_miss)
        fold_enrichment_into_description([tool])
        assert "[Aliases]" in tool["description"]
        assert "Also known as: cat, view" in tool["description"]
        assert "Commonly confused with: Bash cat" in tool["description"]

    def test_near_miss_field_removed(self) -> None:
        tool = _make_tool(_near_miss={"aliases": ["cat"]})
        fold_enrichment_into_description([tool])
        assert "_near_miss" not in tool


class TestAntiPatterns:
    """_anti_patterns is folded into description."""

    def test_anti_patterns_folded(self) -> None:
        anti = [
            {
                "anti_pattern": "Using Bash for file reads",
                "why_bad": "Loses tool tracking.",
                "do_instead": "Use Read tool.",
            }
        ]
        tool = _make_tool(_anti_patterns=anti)
        fold_enrichment_into_description([tool])
        assert "[Anti-Patterns]" in tool["description"]
        assert "AVOID: Using Bash for file reads" in tool["description"]
        assert "Why: Loses tool tracking." in tool["description"]
        assert "Instead: Use Read tool." in tool["description"]

    def test_anti_patterns_field_removed(self) -> None:
        tool = _make_tool(_anti_patterns=[{"anti_pattern": "test"}])
        fold_enrichment_into_description([tool])
        assert "_anti_patterns" not in tool


class TestOutputSchema:
    """outputSchema is folded into description."""

    def test_output_schema_folded(self) -> None:
        schema = {"type": "object", "properties": {"content": {"type": "string"}}}
        tool = _make_tool(outputSchema=schema)
        fold_enrichment_into_description([tool])
        assert "[Output Schema]" in tool["description"]
        assert '"content"' in tool["description"]

    def test_output_schema_field_removed(self) -> None:
        tool = _make_tool(outputSchema={"type": "string"})
        fold_enrichment_into_description([tool])
        assert "outputSchema" not in tool


class TestQuality:
    """_quality is folded into description."""

    def test_quality_folded(self) -> None:
        quality = {"gate": "required", "threshold": "95%"}
        tool = _make_tool(_quality=quality)
        fold_enrichment_into_description([tool])
        assert "[Quality]" in tool["description"]
        assert "gate: required" in tool["description"]

    def test_quality_field_removed(self) -> None:
        tool = _make_tool(_quality={"gate": "required"})
        fold_enrichment_into_description([tool])
        assert "_quality" not in tool


class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_no_enrichment_fields_description_unchanged(self) -> None:
        tool = _make_tool()
        original_desc = tool["description"]
        fold_enrichment_into_description([tool])
        assert tool["description"] == original_desc

    def test_empty_description_with_enrichment(self) -> None:
        tool = _make_tool(description="", behavioral_what="Enhanced desc.")
        fold_enrichment_into_description([tool])
        assert tool["description"] == "[Enhanced Description]\nEnhanced desc."

    def test_multiple_tools_all_folded(self) -> None:
        tools = [
            _make_tool(name="Read", behavioral_what="Read enhanced."),
            _make_tool(name="Edit", behavioral_what="Edit enhanced."),
        ]
        fold_enrichment_into_description(tools)
        assert "Read enhanced." in tools[0]["description"]
        assert "Edit enhanced." in tools[1]["description"]

    def test_multiple_enrichment_fields_all_folded(self) -> None:
        tool = _make_tool(
            behavioral_what="Enhanced.",
            behavioral_why={"problem_context": "Need file content."},
            _links={"related": ["Edit"]},
        )
        fold_enrichment_into_description([tool])
        assert "[Enhanced Description]" in tool["description"]
        assert "[Why Use This Tool]" in tool["description"]
        assert "[Navigation]" in tool["description"]

    def test_returns_same_list(self) -> None:
        tools = [_make_tool()]
        result = fold_enrichment_into_description(tools)
        assert result is tools

    def test_empty_list(self) -> None:
        result = fold_enrichment_into_description([])
        assert result == []


class TestRemoveRemainingEnrichment:
    """_remove_remaining_enrichment_fields cleans up."""

    def test_removes_manifest(self) -> None:
        tool = _make_tool(_manifest={"version": "1.0"})
        _remove_remaining_enrichment_fields(tool)
        assert "_manifest" not in tool

    def test_removes_registration(self) -> None:
        tool = _make_tool(_registration={"registered": True})
        _remove_remaining_enrichment_fields(tool)
        assert "_registration" not in tool

    def test_preserves_base_fields(self) -> None:
        tool = _make_tool()
        _remove_remaining_enrichment_fields(tool)
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
