"""Pattern 15: WebMCP / tool registration compatibility.

Ensures tool definitions are compatible with WebMCP and other tool
registration systems by adding registration metadata.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class ToolRegistrationApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 15: Tool Registration.

    Adds WebMCP-compatible registration metadata to tool definitions
    so they can be consumed by any tool registration system.

    Args:
        registration_data: Static registration fields from YAML.
            When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, str] = {
        "protocol": "claude-code-bridge",
        "namespace": "claude_code",
        "version": "0.1.0",
    }

    def __init__(self, registration_data: dict[str, str] | None = None) -> None:
        self._registration_fields = (
            registration_data if registration_data is not None else self._DEFAULTS
        )

    @property
    def pattern_number(self) -> int:
        return 15

    @property
    def name(self) -> str:
        return "Tool Registration"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add tool registration metadata for WebMCP compatibility."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            tool["_registration"] = {
                "tool_name": tool_name,
                **self._registration_fields,
            }
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have registration metadata."""
        issues = []
        for tool in tools:
            if "_registration" not in tool:
                issues.append({
                    "tool": tool.get("name", "<unnamed>"),
                    "issue": "Missing _registration metadata for tool registration systems",
                    "severity": "warning",
                })
        return issues
