"""WHY dimension behavioral enricher.

Adds problem context and failure modes to tool definitions — the reasoning
about WHY to use a tool and what goes wrong when you misuse it.
"""
from typing import Any

from enrichment.behavioral.base import DataDrivenBehavioralEnricher


class WhyEnricher(DataDrivenBehavioralEnricher):
    """Enriches tool definitions with WHY context.

    The WHY dimension captures:
      - problem_context: What problem this tool solves
      - failure_modes: What goes wrong when misused

    Uses DataDrivenBehavioralEnricher base. YAML is single source of truth.
    """

    _enrichment_key = "behavioral_why"

    @property
    def dimension(self) -> str:
        return "why"
