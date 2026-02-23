"""Pattern 5: Near-miss suggestions (did_you_mean).

When a tool name is close but not exact, provide suggestions.
Also enriches tools with aliases and common misspellings.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class NearMissApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 5: Near-Miss.

    Enriches tool definitions with aliases and common misuses so the
    bridge can suggest corrections when a model tries to use a wrong tool.

    Args:
        tool_data: Per-tool alias data from YAML. When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, dict[str, Any]] = {
        "Read": {
            "aliases": ["cat", "view", "open", "show"],
            "commonly_confused_with": ["Bash cat", "Bash head"],
        },
        "Edit": {
            "aliases": ["replace", "modify", "patch"],
            "commonly_confused_with": ["Bash sed", "Write"],
        },
        "Write": {
            "aliases": ["create", "save", "output"],
            "commonly_confused_with": ["Edit", "Bash echo"],
        },
        "Grep": {
            "aliases": ["search", "find_in_files", "rg"],
            "commonly_confused_with": ["Bash grep", "Bash rg", "Glob"],
        },
        "Glob": {
            "aliases": ["find_files", "locate", "ls"],
            "commonly_confused_with": ["Bash find", "Bash ls", "Grep"],
        },
        "Bash": {
            "aliases": ["shell", "terminal", "exec", "run"],
            "commonly_confused_with": [],
        },
    }

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        self._tool_aliases = tool_data if tool_data is not None else self._DEFAULTS

    @property
    def pattern_number(self) -> int:
        return 5

    @property
    def name(self) -> str:
        return "Near-Miss"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add alias and confusion data to tool definitions."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            aliases = self._tool_aliases.get(tool_name)
            if aliases:
                tool["_near_miss"] = aliases
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have near-miss data."""
        issues = []
        for tool in tools:
            if "_near_miss" not in tool:
                issues.append({
                    "tool": tool.get("name", "<unnamed>"),
                    "issue": "Missing _near_miss alias data",
                    "severity": "warning",
                })
        return issues
