"""Pattern 6: Self-describing schemas (inputSchema/outputSchema).

Ensures every tool definition has explicit input and output schemas
so the model knows exactly what to send and what to expect back.
This is the most important structural pattern for tool-use accuracy.
"""
import copy
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class SelfDescribingApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 6: Self-Describing Schemas.

    Ensures every tool has inputSchema and outputSchema. Uses DataDriven
    base for init (YAML single source of truth). Overrides apply/validate
    due to inputSchema stub logic.
    """

    _field_name = "outputSchema"

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        super().__init__(tool_data)
        self._output_schemas = self._tool_data

    @property
    def pattern_number(self) -> int:
        return 6

    @property
    def name(self) -> str:
        return "Self-Describing"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add inputSchema and outputSchema to tool definitions.

        Preserves existing inputSchema. Adds outputSchema from known schemas.
        Adds a stub inputSchema only if completely missing.
        """
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")

            # Preserve existing inputSchema, add stub if missing
            if "inputSchema" not in tool and "input_schema" not in tool:
                tool["inputSchema"] = {
                    "type": "object",
                    "properties": {},
                    "description": f"Input parameters for {tool_name} (schema not available)",
                }

            # Add outputSchema from known schemas
            output_schema = self._output_schemas.get(tool_name)
            if output_schema:
                tool["outputSchema"] = output_schema

        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have input and output schemas."""
        issues = []
        for tool in tools:
            tool_name = tool.get("name", "<unnamed>")
            has_input = "inputSchema" in tool or "input_schema" in tool
            has_output = "outputSchema" in tool

            if not has_input:
                issues.append({
                    "tool": tool_name,
                    "issue": "Missing inputSchema — model cannot know what parameters to send",
                    "severity": "error",
                })
            if not has_output:
                issues.append({
                    "tool": tool_name,
                    "issue": "Missing outputSchema — model cannot know what response to expect",
                    "severity": "warning",
                })

        return issues
