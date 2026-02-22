"""Base class for behavioral enrichment dimensions.

Behavioral enrichment transfers Claude Code's RL-trained tool-use
instincts into explicit structured data that any model can use.
The three dimensions are:
  WHAT — Enhanced descriptions of what the tool does
  WHY  — Problem context, failure modes, reasoning about when NOT to use
  WHEN — Prerequisites, alternatives, sequencing relative to other tools
"""
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
