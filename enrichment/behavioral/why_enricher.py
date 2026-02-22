"""WHY dimension behavioral enricher.

Adds problem context and failure modes to tool definitions — the reasoning
about WHY to use a tool and what goes wrong when you misuse it.
"""
import copy
from typing import Any

from enrichment.behavioral.base import BehavioralEnricher
from enrichment.behavioral.tool_knowledge import TOOL_KNOWLEDGE


class WhyEnricher(BehavioralEnricher):
    """Enriches tool definitions with WHY context.

    The WHY dimension captures:
      - problem_context: What problem this tool solves
      - failure_modes: What goes wrong when misused
    """

    @property
    def dimension(self) -> str:
        return "why"

    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Add problem context and failure modes to tool definition.

        Adds a 'behavioral_why' key with problem_context and failure_modes.
        """
        enriched = copy.deepcopy(tool)
        tool_name = tool.get("name", "")
        knowledge = TOOL_KNOWLEDGE.get(tool_name)

        if knowledge and "why" in knowledge:
            enriched["behavioral_why"] = knowledge["why"]

        return enriched
