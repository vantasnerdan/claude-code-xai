"""WHEN dimension behavioral enricher.

Adds prerequisite, alternative, and sequencing knowledge to tool definitions —
the reasoning about WHEN to use a tool relative to other tools.
"""
import copy
from typing import Any

from enrichment.behavioral.base import BehavioralEnricher
from enrichment.behavioral.tool_knowledge import TOOL_KNOWLEDGE


class WhenEnricher(BehavioralEnricher):
    """Enriches tool definitions with WHEN context.

    The WHEN dimension captures:
      - prerequisites: Tools that should be used before this one
      - use_before: Tools this one should precede
      - use_instead_of: Tools this one replaces
      - do_not_use_for: Operations where another tool is preferred
      - sequencing: Natural language description of tool ordering

    Args:
        tool_data: Per-tool WHEN dicts from YAML. When None, uses built-in TOOL_KNOWLEDGE.
    """

    def __init__(self, tool_data: dict[str, dict[str, Any]] | None = None) -> None:
        self._tool_data = tool_data

    @property
    def dimension(self) -> str:
        return "when"

    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Add sequencing and prerequisite data to tool definition.

        Adds a 'behavioral_when' key with prerequisites, alternatives,
        and sequencing guidance.
        """
        enriched = copy.deepcopy(tool)
        tool_name = tool.get("name", "")

        if self._tool_data is not None:
            when_data = self._tool_data.get(tool_name)
            if when_data:
                enriched["behavioral_when"] = when_data
        else:
            knowledge = TOOL_KNOWLEDGE.get(tool_name)
            if knowledge and "when" in knowledge:
                enriched["behavioral_when"] = knowledge["when"]

        return enriched
