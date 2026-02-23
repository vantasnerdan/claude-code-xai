"""Pattern 6: Self-describing schemas (inputSchema/outputSchema).

Ensures every tool definition has explicit input and output schemas
so the model knows exactly what to send and what to expect back.
This is the most important structural pattern for tool-use accuracy.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class SelfDescribingApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 6: Self-Describing Schemas.

    Ensures every tool has inputSchema and outputSchema. If inputSchema
    exists (from the original tool definition), it is preserved. If missing,
    a stub is added. outputSchema is always added since tool definitions
    typically lack it.

    Args:
        tool_data: Per-tool output schema data from YAML. When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, dict[str, Any]] = {
        "Read": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "File content with line numbers (cat -n format)",
                },
            },
        },
        "Edit": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "error": {"type": "string", "description": "Error message if edit failed"},
            },
        },
        "Write": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "file_path": {"type": "string", "description": "Path of written file"},
            },
        },
        "Bash": {
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
            },
        },
        "Grep": {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "description": "Matching lines, file paths, or counts depending on output_mode",
                },
            },
        },
        "Glob": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Matching file paths sorted by modification time",
                },
            },
        },
    }

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        self._output_schemas = tool_data if tool_data is not None else self._DEFAULTS

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
