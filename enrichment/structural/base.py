"""Base class for structural enrichment pattern applicators.

Each pattern from the Agentic API Standard gets its own applicator.
Applicators are independent and composable — the engine pipelines them.
"""
import copy
from abc import ABC, abstractmethod
from typing import Any


class PatternApplicator(ABC):
    """Base class for structural enrichment patterns.

    Each pattern from the Agentic API Standard gets its own applicator.
    Applicators are independent and composable.
    """

    @property
    @abstractmethod
    def pattern_number(self) -> int:
        """The Agentic API Standard pattern number (1-20)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pattern name."""
        ...

    @abstractmethod
    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enrich tool definitions with this pattern.

        Must not modify the original tools list — return a new list.
        Must not change the semantic meaning of tool descriptions.
        """
        ...

    @abstractmethod
    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check tool definitions for compliance with this pattern.

        Returns a list of issue dicts:
        [{"tool": name, "issue": description, "severity": "error"|"warning"}]
        """
        ...


class DataDrivenPatternApplicator(PatternApplicator):
    """Data-driven structural pattern applicator using YAML as single source of truth.

    Extracts the common deepcopy + per-tool lookup loop from the 6 per-tool
    applicators. This resolves the bulk of the 258 clone groups.

    Subclasses define:
      - pattern_number, name properties
      - _field_name class attr (e.g. "_links", "_quality")
      - optional _apply_to_tool hook for complex cases (default: set field = data)
    """

    def __init__(self, tool_data: dict[str, Any] | None = None) -> None:
        self._tool_data = tool_data or {}

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Common apply logic for data-driven patterns.

        Deep copies the list (immutability), looks up per-tool data by name,
        and applies it via _apply_to_tool hook.
        """
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            data = self._tool_data.get(tool_name)
            if data:
                self._apply_to_tool(tool, tool_name, data)
        return enriched

    def _apply_to_tool(self, tool: dict[str, Any], tool_name: str, data: Any) -> None:
        """Hook for subclasses to customize how data is applied to a tool.

        Default: set the _field_name to the data.
        Override for complex cases (e.g. wrapping in _error_format dict).
        """
        if hasattr(self, "_field_name"):
            tool[self._field_name] = data

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Default validation: warn if expected field is missing for known tools.

        Subclasses should override with pattern-specific checks.
        """
        issues: list[dict[str, Any]] = []
        for tool in tools:
            tool_name = tool.get("name", "<unnamed>")
            if tool_name in self._tool_data and hasattr(self, "_field_name"):
                field = getattr(self, "_field_name")
                if field not in tool:
                    issues.append({
                        "tool": tool_name,
                        "issue": f"Missing {field} for this pattern",
                        "severity": "warning",
                    })
        return issues
