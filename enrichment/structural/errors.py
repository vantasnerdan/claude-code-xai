"""Pattern 3: Standard error format with suggestion.

Enriches tool definitions with structured error format documentation
so the model knows what error responses look like and how to recover.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class ErrorFormatApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 3: Error Format.

    Adds error format documentation to tool definitions so the model
    knows the structure of error responses and how to handle them.

    Args:
        tool_data: Per-tool error data from YAML. When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, list[dict[str, str]]] = {
        "Read": [
            {
                "error": "File not found",
                "suggestion": "Use Glob to search for the file by name pattern",
            },
            {
                "error": "Empty file",
                "suggestion": "File exists but has no content — this is a valid state",
            },
        ],
        "Edit": [
            {
                "error": "old_string not found",
                "suggestion": "Read the file first to see exact content including whitespace",
            },
            {
                "error": "old_string not unique",
                "suggestion": "Include more surrounding lines to make the match unique",
            },
            {
                "error": "File not read yet",
                "suggestion": "You must Read the file at least once before editing it",
            },
        ],
        "Write": [
            {
                "error": "File not read yet",
                "suggestion": "Read existing files before overwriting them",
            },
        ],
        "Bash": [
            {
                "error": "Command timed out",
                "suggestion": "Increase timeout parameter (max 600000ms) or use run_in_background",
            },
            {
                "error": "Command failed",
                "suggestion": "Check that file paths with spaces are double-quoted",
            },
        ],
    }

    def __init__(self, tool_data: dict[str, list[dict[str, str]]] | None = None) -> None:
        self._tool_errors = tool_data if tool_data is not None else self._DEFAULTS

    @property
    def pattern_number(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "Error Format"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add standard error format documentation to tool definitions."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            errors = self._tool_errors.get(tool_name)
            if errors:
                tool["_error_format"] = {
                    "errors": errors,
                    "format": {
                        "error": "string — error description",
                        "suggestion": "string — how to fix it",
                    },
                }
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have error format documentation."""
        issues = []
        for tool in tools:
            tool_name = tool.get("name", "<unnamed>")
            if tool_name in self._tool_errors and "_error_format" not in tool:
                issues.append({
                    "tool": tool_name,
                    "issue": "Missing _error_format documentation",
                    "severity": "warning",
                })
        return issues
