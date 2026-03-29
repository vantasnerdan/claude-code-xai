"""Base class for behavioral enrichment dimensions.

Behavioral enrichment transfers Claude Code's RL-trained tool-use
instincts into explicit structured data that any model can use.
The three dimensions are:
  WHAT — Enhanced descriptions of what the tool does
  WHY  — Problem context, failure modes, reasoning about when NOT to use
  WHEN — Prerequisites, alternatives, sequencing relative to other tools
"""
import copy
from abc import ABC, abstractmethod
from typing import Any


class BehavioralEnricher(ABC):
    """Base class for behavioral enrichment dimensions.

    Behavioral enrichment transfers Claude Code's RL-trained tool-use
    instincts into explicit structured data that any model can use.
    """

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Which dimension: 'what', 'why', or 'when'."""
        ...

    @abstractmethod
    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Enrich a single tool definition with behavioral knowledge.

        Must not modify the original tool dict — return a new dict.
        """
        ...


class DataDrivenBehavioralEnricher(BehavioralEnricher):
    """Data-driven behavioral enricher using YAML as single source of truth.

    Extracts common logic (deepcopy + tool name lookup) from the three
    dimension-specific enrichers to eliminate duplication (part of fixing
    258 clone groups).

    YAML is now the *only* source — no TOOL_KNOWLEDGE fallback (structural fix).
    Subclasses must define:
      - dimension property
      - _enrichment_key class attribute (e.g. "behavioral_what")
    """

    _enrichment_key: str

    def __init__(self, tool_data: dict[str, Any] | None = None) -> None:
        self._tool_data = tool_data or {}

    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Common implementation for all behavioral dimensions.

        Performs deep copy (immutability guarantee) and looks up data
        by tool name in the YAML-provided structure.
        """
        enriched = copy.deepcopy(tool)
        tool_name = tool.get("name", "")

        data = self._tool_data.get(tool_name)
        if data:
            enriched[self._enrichment_key] = data

        return enriched
