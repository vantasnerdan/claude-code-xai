"""WHAT dimension behavioral enricher.

Enhances tool descriptions with richer detail about what the tool does,
drawn from the knowledge base of Claude Code's trained behavior.
"""
import copy
from typing import Any

from enrichment.behavioral.base import BehavioralEnricher
from enrichment.behavioral.tool_knowledge import TOOL_KNOWLEDGE


class WhatEnricher(BehavioralEnricher):
    """Enriches tool definitions with enhanced WHAT descriptions.

    The WHAT dimension replaces or augments the raw tool description with
    a more detailed version that captures nuances learned through RL training.

    Args:
        tool_data: Per-tool WHAT strings from YAML. When None, uses built-in TOOL_KNOWLEDGE.
    """

    def __init__(self, tool_data: dict[str, str] | None = None) -> None:
        self._tool_data = tool_data

    @property
    def dimension(self) -> str:
        return "what"

    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Add enhanced description to tool definition.

        Adds a 'behavioral_what' key with the enriched description.
        Preserves the original 'description' field unchanged.
        """
        enriched = copy.deepcopy(tool)
        tool_name = tool.get("name", "")

        if self._tool_data is not None:
            what_text = self._tool_data.get(tool_name)
            if what_text:
                enriched["behavioral_what"] = what_text
        else:
            knowledge = TOOL_KNOWLEDGE.get(tool_name)
            if knowledge and "what" in knowledge:
                enriched["behavioral_what"] = knowledge["what"]

        return enriched
