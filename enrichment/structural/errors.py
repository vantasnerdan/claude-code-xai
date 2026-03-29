"""Pattern 3: Standard error format with suggestion.

Enriches tool definitions with structured error format documentation
so the model knows what error responses look like and how to recover.
"""
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class ErrorFormatApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 3: Error Format.

    Uses DataDrivenPatternApplicator base with custom _apply_to_tool for
    wrapping the errors list. YAML is single source of truth.
    """

    _field_name = "_error_format"

    @property
    def pattern_number(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "Error Format"

    def _apply_to_tool(self, tool: dict[str, Any], tool_name: str, data: Any) -> None:
        """Wrap errors list in standard _error_format structure."""
        tool["_error_format"] = {
            "errors": data,
            "format": {
                "error": "string — error description",
                "suggestion": "string — how to fix it",
            },
        }
