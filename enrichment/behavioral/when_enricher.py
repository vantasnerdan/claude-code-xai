"""WHEN dimension behavioral enricher.

Adds prerequisite, alternative, and sequencing knowledge to tool definitions —
the reasoning about WHEN to use a tool relative to other tools.
"""
from typing import Any

from enrichment.behavioral.base import DataDrivenBehavioralEnricher


class WhenEnricher(DataDrivenBehavioralEnricher):
    """Enriches tool definitions with WHEN context.

    The WHEN dimension captures:
      - prerequisites: Tools that should be used before this one
      - use_before: Tools this one should precede
      - use_instead_of: Tools this one replaces
      - do_not_use_for: Operations where another tool is preferred
      - sequencing: Natural language description of tool ordering

    Uses DataDrivenBehavioralEnricher base. YAML is single source of truth.
    Note: when schema is non-uniform (do_not_use_for for some tools).
    """

    _enrichment_key = "behavioral_when"

    @property
    def dimension(self) -> str:
        return "when"
