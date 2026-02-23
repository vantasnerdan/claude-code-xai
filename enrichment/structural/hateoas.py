"""Pattern 2: HATEOAS — Hypermedia links for navigation and recovery.

Adds _links to tool definitions so the model knows how to recover
from errors, find related tools, and navigate the tool ecosystem.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class HateoasApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 2: HATEOAS.

    Adds _links to each tool definition with navigation hints:
    related tools, error recovery paths, and documentation references.

    Args:
        tool_data: Per-tool link data from YAML. When None, uses built-in defaults.
    """

    # Built-in defaults (used when no YAML data is provided)
    _DEFAULTS: dict[str, dict[str, Any]] = {
        "Read": {
            "related": ["Edit", "Write", "Glob", "Grep"],
            "on_error": {"file_not_found": "Use Glob to locate the file first"},
        },
        "Edit": {
            "related": ["Read", "Write"],
            "on_error": {
                "old_string_not_found": "Use Read to see the actual file content",
                "old_string_not_unique": "Include more surrounding context in old_string",
            },
        },
        "Write": {
            "related": ["Read", "Edit"],
            "on_error": {"file_exists": "Use Read first, then prefer Edit over Write"},
        },
        "Bash": {
            "related": ["Read", "Grep", "Glob", "Edit", "Write"],
            "on_error": {"command_failed": "Check file paths are quoted if they contain spaces"},
        },
        "Grep": {
            "related": ["Read", "Glob"],
            "on_error": {"no_matches": "Try a broader pattern or different glob filter"},
        },
        "Glob": {
            "related": ["Read", "Grep"],
            "on_error": {"no_matches": "Try a broader pattern or check the directory path"},
        },
    }

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        self._tool_links = tool_data if tool_data is not None else self._DEFAULTS

    @property
    def pattern_number(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "HATEOAS"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add _links to tool definitions for navigation and recovery."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            links = self._tool_links.get(tool_name, {})
            if links:
                tool["_links"] = links
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have _links for navigation."""
        issues = []
        for tool in tools:
            if "_links" not in tool:
                issues.append({
                    "tool": tool.get("name", "<unnamed>"),
                    "issue": "Missing _links for navigation and error recovery",
                    "severity": "warning",
                })
        return issues
