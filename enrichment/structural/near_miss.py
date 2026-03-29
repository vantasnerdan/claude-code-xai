"""Pattern 5: Near-miss suggestions (did_you_mean).

When a tool name is close but not exact, provide suggestions.
Also enriches tools with aliases and common misspellings.
"""
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class NearMissApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 5: Near-Miss.

    Uses DataDrivenPatternApplicator base. YAML is single source of truth.
    """

    _field_name = "_near_miss"

    @property
    def pattern_number(self) -> int:
        return 5

    @property
    def name(self) -> str:
        return "Near-Miss"
