"""Pattern 8: Warnings and quality flags.

Adds quality metadata to tool definitions — known limitations,
edge cases, and quality signals that help the model make better decisions.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class QualityGatesApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 8: Warnings + Quality.

    Adds warnings array and quality flags to tool definitions so the
    model is aware of limitations and edge cases before using a tool.

    Args:
        tool_data: Per-tool quality data from YAML. When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, dict[str, Any]] = {
        "Read": {
            "warnings": [
                "Lines longer than 2000 characters are truncated",
                "Large PDFs require pages parameter — reading without it will fail",
                "Default reads up to 2000 lines — use offset/limit for large files",
            ],
            "quality": {"reliability": "high", "coverage": "text, images, PDFs, notebooks"},
        },
        "Edit": {
            "warnings": [
                "old_string must be unique in the file or edit fails",
                "Indentation must match exactly (tabs vs spaces matter)",
                "Must Read the file first in the same conversation",
            ],
            "quality": {"reliability": "high", "safety": "exact-match prevents wrong-location edits"},
        },
        "Bash": {
            "warnings": [
                "Shell state does not persist between calls (only working directory does)",
                "Default timeout is 2 minutes — set timeout for long-running commands",
                "Interactive commands (-i flag) are not supported",
                "Output truncated at 30000 characters",
            ],
            "quality": {"reliability": "medium", "safety": "commands can have side effects"},
        },
        "Grep": {
            "warnings": [
                "Single-line matching by default — use multiline: true for cross-line patterns",
                "Literal braces need escaping in patterns",
            ],
            "quality": {"reliability": "high", "engine": "ripgrep"},
        },
    }

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        self._tool_quality = tool_data if tool_data is not None else self._DEFAULTS

    @property
    def pattern_number(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "Quality Gates"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add warnings and quality flags to tool definitions."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            quality_data = self._tool_quality.get(tool_name)
            if quality_data:
                tool["_quality"] = quality_data
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have quality metadata."""
        issues = []
        for tool in tools:
            tool_name = tool.get("name", "<unnamed>")
            if tool_name in self._tool_quality and "_quality" not in tool:
                issues.append({
                    "tool": tool_name,
                    "issue": "Missing _quality warnings and quality flags",
                    "severity": "warning",
                })
        return issues
